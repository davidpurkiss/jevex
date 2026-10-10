import asyncio
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from jevex import (
    Candidate,
    Context,
    Document,
    DomLocation,
    Field,
    NormaliserStep,
    Questions,
    SchemaSpec,
    Statement,
)
from jevex.entities import EntityScope
from jevex.errors import PartError
from jevex.generators import GeneratorRegistry, GeneratorSpec, RegexGenerator
from jevex.interfaces import CandidateSelector, ParsedDocument, Scope
from jevex.jev import Choice, ChoiceAnswer, JevResponse, JSONContent, Noul, Question
from jevex.layout import MAX_SECTION_CHARS, Component
from jevex.learn import GeneratorSnapshot
from jevex.normalise import NormaliseStage
from jevex.pipeline import Pipeline, VisionValue
from jevex.results import Conflict, FieldMeta
from jevex.select import (
    CandidateStage,
    JevCandidateSelector,
    SelectStage,
    field_statements,
    statement_state,
)
from jevex.testing import FakeJev

LOC = DomLocation(dom_path="/p")


class Car(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    fuel_type: Literal["petrol", "diesel", "ev"] = Field(description="Fuel type")
    automatic: bool = Field(description="has an automatic gearbox")
    colours: list[Literal["red", "blue", "grey"]] = Field(
        default_factory=list, description="Colours"
    )
    trims: list[str] = Field(default_factory=list, description="Trim names")
    model: str = Field(description="Model name")


class Rival(BaseModel):
    zero_to_62_s: float = Field(description="Acceleration", unit="s")


SPEC = SchemaSpec.from_model(Car)


def st(sid: str, text: str, component: str = "c1", trail: list[str] | None = None) -> Statement:
    return Statement(
        id=sid,
        text=text,
        kind="sentence",
        component_id=component,
        location=LOC,
        heading_trail=trail or [],
    )


def context(
    fake: FakeJev,
    statements: list[Statement],
    categories: dict[str, str],
    models: tuple[type[BaseModel], ...] = (Car,),
    document: Document | None = None,
) -> Context:
    ctx = Context.create(
        document or Document.from_bytes(b"<p/>", url="https://example.com"),
        [SchemaSpec.from_model(m) for m in models],
        fake.client(),
    )
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={s.id: s for s in statements},
    )
    components = sorted({s.component_id for s in statements})
    for run in ctx.schemas.values():
        run.scopes = [EntityScope(label="doc", component_ids=components)]
        for sid, name in categories.items():
            run.categories[sid] = ChoiceAnswer(
                choice=name, confidence=0.9, probabilities={name: 0.9}
            )
    return ctx


async def run_both(ctx: Context) -> None:
    await CandidateStage().run(ctx)
    await SelectStage().run(ctx)


def only_call_questions(fake: FakeJev) -> dict[str, Question]:
    [call] = fake.calls
    return call.questions


# --- which statements ------------------------------------------------------------------


async def test_only_statements_categorised_as_a_field_are_used() -> None:
    a, b, c = st("s1", "0-62 mph in 9.1 s"), st("s2", "Great value"), st("s3", "Other", "c9")
    ctx = context(FakeJev(), [a, b, c], {"s1": "zero_to_62_s", "s2": "none", "s3": "model"})
    run = ctx.schemas["Car"]
    run.scopes = [EntityScope(label="doc", component_ids=["c1"])]  # s3 is out of scope
    pairs = field_statements(ctx, run, run.scopes[0])
    assert [(s.id, f.name) for s, f in pairs] == [("s1", "zero_to_62_s")]


def test_state_includes_the_heading_trail() -> None:
    assert statement_state(st("s", "9.1 s", trail=["Specs", "Performance"])) == {
        "statement": "9.1 s",
        "section": "Specs › Performance",
    }
    assert statement_state(st("s", "9.1 s")) == {"statement": "9.1 s"}


# --- candidates ------------------------------------------------------------------------


def test_a_statement_also_goes_to_fields_with_enough_category_probability() -> None:
    a = st("s1", "Automatic, 0-62 mph in 9.1 s")
    ctx = context(FakeJev(), [a], {})
    run = ctx.schemas["Car"]
    run.categories["s1"] = ChoiceAnswer(
        choice="zero_to_62_s",
        confidence=0.5,
        probabilities={"zero_to_62_s": 0.5, "automatic": 0.35, "model": 0.05, "none": 0.1},
    )
    pairs = field_statements(ctx, run, run.scopes[0])
    assert [f.name for _, f in pairs] == ["zero_to_62_s", "automatic"]
    run.categories["s1"] = ChoiceAnswer(
        choice="model",
        confidence=0.95,
        probabilities={"model": 0.95, "automatic": 0.05},
    )
    assert [f.name for _, f in field_statements(ctx, run, run.scopes[0])] == ["model"]
    # A "none" answer routes nowhere, however close a field came.
    run.categories["s1"] = ChoiceAnswer(
        choice="none", confidence=0.6, probabilities={"none": 0.6, "automatic": 0.4}
    )
    assert field_statements(ctx, run, run.scopes[0]) == []


def test_found_fields_are_left_out_unless_asked_for() -> None:
    ctx = context(FakeJev(), [st("s1", "0-62 mph in 9.1 s")], {"s1": "zero_to_62_s"})
    run = ctx.schemas["Car"]
    run.set_field("doc", "zero_to_62_s", FieldMeta(value=9.1, confidence=0.2))
    assert field_statements(ctx, run, run.scopes[0]) == []
    pairs = field_statements(ctx, run, run.scopes[0], include_found=True)
    assert [f.name for _, f in pairs] == ["zero_to_62_s"]


async def test_a_second_field_route_never_makes_a_bool_false() -> None:
    about = st("s1", "Automatic gearbox as standard")
    other = st("s2", "Diesel engine, manual option")
    fake = (
        FakeJev()
        .noul("automatic gearbox", p=0.85, state="as standard")
        .noul("automatic gearbox", p=0.05, state="manual option")
    )
    ctx = context(fake, [about, other], {"s1": "automatic"})
    run = ctx.schemas["Car"]
    run.categories["s2"] = ChoiceAnswer(
        choice="fuel_type", confidence=0.6, probabilities={"fuel_type": 0.6, "automatic": 0.3}
    )
    await run_both(ctx)
    meta = run.fields["doc"]["automatic"]
    assert meta.value is True
    assert meta.confidence == 0.85


async def test_candidates_are_generated_for_fields_that_need_them() -> None:
    a, b = st("s1", "0-62 mph in 9.1 s"), st("s2", "Runs on diesel")
    ctx = context(FakeJev(), [a, b], {"s1": "zero_to_62_s", "s2": "fuel_type"})
    await CandidateStage().run(ctx)
    run = ctx.schemas["Car"]
    assert [c.raw for c in run.candidates[("s1", "zero_to_62_s")]] == [
        "0",
        "0-62 mph",
        "62 mph",
        "9.1 s",
    ]
    assert ("s2", "fuel_type") not in run.candidates  # enums need no candidates


async def test_the_candidate_stage_records_the_generators_it_ran() -> None:
    a = st("s1", "0-62 mph in 9.1 s")
    ctx = context(FakeJev(), [a], {"s1": "zero_to_62_s"})
    timed = RegexGenerator(
        id="timed", pattern=r"(\d+) s", scope=Scope(fields=frozenset({"zero_to_62_s"}))
    )
    named = RegexGenerator(id="named", pattern=r"\w+", scope=Scope(fields=frozenset({"model"})))
    await CandidateStage(registry=GeneratorRegistry([timed, named])).run(ctx)
    assert ctx.generators_ran == {"timed"}


class Crashes:
    """A generator with a bug."""

    id = "crashes"
    scope = Scope(fields=frozenset({"zero_to_62_s"}))

    def generate(self, statement: Statement) -> list[Candidate]:
        raise IndexError("group 2 out of range")


async def test_a_generator_that_raises_is_skipped_and_reported() -> None:
    a, b = st("s1", "0-62 mph in 9.1 s"), st("s2", "0-62 in 8 s")
    ctx = context(FakeJev(), [a, b], {"s1": "zero_to_62_s", "s2": "zero_to_62_s"})
    timed = RegexGenerator(
        id="timed", pattern=r"([\d.]+) s", scope=Scope(fields=frozenset({"zero_to_62_s"}))
    )
    await CandidateStage(registry=GeneratorRegistry([Crashes(), timed])).run(ctx)
    run = ctx.schemas["Car"]
    # The other generators still give their candidates.
    assert [c.raw for c in run.candidates[("s1", "zero_to_62_s")]] == ["9.1 s"]
    assert [c.raw for c in run.candidates[("s2", "zero_to_62_s")]] == ["8 s"]
    assert ctx.errors.errors == [
        PartError(
            stage="candidates",
            kind="generator",
            part="crashes",
            type="IndexError",
            message="group 2 out of range",
            count=2,
        )
    ]
    assert ctx.generators_ran == {"crashes", "timed"}


def test_the_registry_raises_without_an_error_handler() -> None:
    registry = GeneratorRegistry([Crashes()])
    field = SPEC.field("zero_to_62_s")
    with pytest.raises(IndexError):
        registry.generate(st("s1", "9 s"), field, schema="Car")
    failed: list[tuple[str, Exception]] = []
    found = registry.generate(
        st("s1", "9 s"), field, schema="Car", on_error=lambda g, e: failed.append((g.id, e))
    )
    assert found == []
    assert [(gid, type(e)) for gid, e in failed] == [("crashes", IndexError)]


LEARNED = GeneratorSpec.from_yaml(
    """
id: gen-acme
field: Car.zero_to_62_s
scope: {sources: [www.acme-cars.com]}
match: {regex: 'in (\\d+(?:\\.\\d+)?) s', group: 1}
"""
)


@pytest.mark.parametrize(
    ("document", "runs"),
    [
        (Document.from_bytes(b"<p/>", url="https://acme-cars.com/cars/1"), True),
        (Document.from_bytes(b"<p/>", url="https://WWW.Acme-Cars.com/cars/1"), True),
        (Document.from_bytes(b"<p/>", site="acme-cars.com"), True),
        (Document.from_bytes(b"<p/>", url="https://other.com/cars/1"), False),
        (Document.from_bytes(b"<p/>", url="https://shop.acme-cars.com/a"), False),
        (Document.from_bytes(b"<p/>"), False),
    ],
)
async def test_source_scoped_learned_generators_run_only_on_their_sources(
    document: Document, runs: bool
) -> None:
    ctx = context(
        FakeJev(), [st("s1", "0-62 mph in 9.1 s")], {"s1": "zero_to_62_s"}, document=document
    )
    ctx.generators = GeneratorSnapshot(1, GeneratorRegistry([LEARNED.to_generator()]))
    await CandidateStage(registry=GeneratorRegistry()).run(ctx)
    found = ctx.schemas["Car"].candidates[("s1", "zero_to_62_s")]
    assert [(c.raw, c.generator_id) for c in found] == ([("9.1", "gen-acme")] if runs else [])
    assert ctx.generators_ran == ({"gen-acme"} if runs else set())


async def test_the_stage_locale_reads_decimal_commas_through_to_the_value() -> None:
    fake = FakeJev().choice("Which of these is the 0-62 mph time", "9,1 s", confidence=0.8)
    ctx = context(fake, [st("s1", "0-100 km/h in 9,1 s")], {"s1": "zero_to_62_s"})
    await CandidateStage(locale="de-DE").run(ctx)
    run = ctx.schemas["Car"]
    assert "9,1 s" in [c.raw for c in run.candidates[("s1", "zero_to_62_s")]]
    await SelectStage().run(ctx)
    await NormaliseStage().run(ctx)
    assert run.fields["doc"]["zero_to_62_s"].value == 9.1


async def test_without_a_locale_a_decimal_comma_is_not_one_number() -> None:
    ctx = context(FakeJev(), [st("s1", "0-100 km/h in 9,1 s")], {"s1": "zero_to_62_s"})
    await CandidateStage().run(ctx)
    raws = [c.raw for c in ctx.schemas["Car"].candidates[("s1", "zero_to_62_s")]]
    assert "9,1 s" not in raws


def page(
    lang: str | None = None, *, content_language: str | None = None, locale: str | None = None
) -> Document:
    """An HTML page, with ``<html lang>`` when ``lang`` is given."""
    attr = f' lang="{lang}"' if lang else ""
    markup = f"<!doctype html><html{attr}><body><p>0-100 km/h in 9,1 s</p></body></html>"
    return Document.from_bytes(
        markup.encode(),
        url="https://example.com/car",
        content_language=content_language,
        locale=locale,
    )


async def test_a_de_de_page_is_read_by_its_own_locale_without_a_stage_locale() -> None:
    fake = FakeJev().choice("Which of these is the 0-62 mph time", "9,1 s", confidence=0.8)
    ctx = context(
        fake, [st("s1", "0-100 km/h in 9,1 s")], {"s1": "zero_to_62_s"}, document=page("de-DE")
    )
    await CandidateStage().run(ctx)
    run = ctx.schemas["Car"]
    assert ctx.locale == "de-DE"
    assert "9,1 s" in [c.raw for c in run.candidates[("s1", "zero_to_62_s")]]
    await SelectStage().run(ctx)
    await NormaliseStage().run(ctx)
    assert run.fields["doc"]["zero_to_62_s"].value == 9.1


@pytest.mark.parametrize(
    ("document", "stage_locale", "comma"),
    [
        (page(), None, False),  # nothing says: en-GB
        (page(), "de-DE", True),  # the stage's locale is the fallback
        (page("en-GB"), "de-DE", False),  # the page's own locale wins over the stage's
        (page(content_language="de-AT"), None, True),
        (page("en-GB", locale="de-DE"), None, True),  # the caller's wins over the page's
    ],
)
async def test_the_stage_locale_is_only_the_fallback(
    document: Document, stage_locale: str | None, comma: bool
) -> None:
    ctx = context(
        FakeJev(), [st("s1", "0-100 km/h in 9,1 s")], {"s1": "zero_to_62_s"}, document=document
    )
    await CandidateStage(locale=stage_locale).run(ctx)
    raws = [c.raw for c in ctx.schemas["Car"].candidates[("s1", "zero_to_62_s")]]
    assert ("9,1 s" in raws) is comma


GERMAN = GeneratorSpec.from_yaml(
    """
id: gen-de
field: Car.zero_to_62_s
scope: {locale: de}
match: {regex: 'in (\\d+(?:,\\d+)?) s', group: 1}
"""
)


@pytest.mark.parametrize(
    ("document", "runs"),
    [
        (page("de-DE"), True),
        (page("de_at"), True),
        (page(content_language="de-CH"), True),
        (page("en-GB"), False),
        (page(), False),  # no language info: a locale-scoped generator mustn't guess
    ],
)
async def test_locale_scoped_learned_generators_run_by_the_pages_locale(
    document: Document, runs: bool
) -> None:
    ctx = context(
        FakeJev(), [st("s1", "0-100 km/h in 9,1 s")], {"s1": "zero_to_62_s"}, document=document
    )
    ctx.generators = GeneratorSnapshot(1, GeneratorRegistry([GERMAN.to_generator()]))
    await CandidateStage(registry=GeneratorRegistry()).run(ctx)
    found = ctx.schemas["Car"].candidates[("s1", "zero_to_62_s")]
    assert [(c.raw, c.generator_id) for c in found] == ([("9,1", "gen-de")] if runs else [])
    assert ctx.generators_ran == ({"gen-de"} if runs else set())


async def test_no_generator_runs_without_a_statement_for_its_field() -> None:
    ctx = context(FakeJev(), [st("s1", "Runs on diesel")], {"s1": "fuel_type"})
    await CandidateStage().run(ctx)
    assert ctx.generators_ran == set()


def test_statement_state_caps_the_heading_trail() -> None:
    state = statement_state(st("s1", "9.1 s", trail=["Kestrova", "word " * 1000, "Performance"]))
    assert isinstance(state, dict)
    assert state["statement"] == "9.1 s"
    section = str(state["section"])
    assert section.startswith("Kestrova › word word")
    assert section.endswith("… › Performance")
    assert len(section) <= MAX_SECTION_CHARS
    assert statement_state(st("s2", "9.1 s")) == {"statement": "9.1 s"}


# --- selection over candidates ---------------------------------------------------------


async def test_choice_over_candidates_plus_none() -> None:
    fake = FakeJev().choice("Which of these is the 0-62 mph time", "9.1 s", confidence=0.8)
    ctx = context(
        fake, [st("s1", "0-62 mph in 9.1 s", trail=["Performance"])], {"s1": "zero_to_62_s"}
    )
    await run_both(ctx)

    [call] = fake.calls
    assert call.state == {"statement": "0-62 mph in 9.1 s", "section": "Performance"}
    assert call.questions == {
        "Car.zero_to_62_s/choice0": Choice(
            instructions="Which of these is the 0-62 mph time (s)?",
            options={
                "0": None,
                "0-62 mph": None,
                "62 mph": None,
                "9.1 s": None,
                "none": "None of these is the 0-62 mph time",
            },
        )
    }
    sel = ctx.schemas["Car"].selections[("doc", "zero_to_62_s", "s1")]
    assert sel.candidate is not None
    assert sel.candidate.raw == "9.1 s"
    assert sel.confidence == 0.8
    assert set(sel.alternatives) == {"0", "0-62 mph", "62 mph"}
    assert [c.raw for c in sel.accepted] == ["9.1 s"]


async def test_none_is_a_selection_without_a_candidate() -> None:
    fake = FakeJev().choice("Which of these", "none", confidence=0.7)
    ctx = context(fake, [st("s1", "0-62 mph in 9.1 s")], {"s1": "zero_to_62_s"})
    await run_both(ctx)
    sel = ctx.schemas["Car"].selections[("doc", "zero_to_62_s", "s1")]
    assert sel.candidate is None
    assert sel.confidence == 0.7


async def test_literal_none_spans_and_duplicate_spans_become_one_option_each() -> None:
    fake = FakeJev()
    ctx = context(fake, [st("s1", "Model: none")], {"s1": "model"})
    await run_both(ctx)
    question = only_call_questions(fake)["Car.model/choice0"]
    assert isinstance(question, Choice)
    # "none" is the reserved option; the whole short statement is a candidate too.
    assert list(question.options) == ["Model", "Model: none", "none"]


async def test_a_name_run_offers_each_of_its_parts() -> None:
    fake = FakeJev().choice("Which of these is the model name?", "Kestrova")
    ctx = context(fake, [st("s1", "Delmaro Kestrova SE")], {"s1": "model"})
    await run_both(ctx)
    question = only_call_questions(fake)["Car.model/choice0"]
    assert question == Choice(
        instructions="Which of these is the model name?",
        options={
            "Delmaro": None,
            "Delmaro Kestrova": None,
            "Delmaro Kestrova SE": None,
            "Kestrova": None,
            "Kestrova SE": None,
            "SE": None,
            "none": "None of these is the model name",
        },
    )
    sel = ctx.schemas["Car"].selections[("doc", "model", "s1")]
    assert sel.candidate is not None
    assert (sel.candidate.raw, sel.candidate.generator_id) == ("Kestrova", "noun_phrase")


async def test_more_than_254_candidates_are_split_across_choices() -> None:
    text = " ".join(str(i) for i in range(300))
    fake = FakeJev().choice("Which of these", lambda q: "299" if "299" in q.options else "none")
    ctx = context(fake, [st("s1", text)], {"s1": "zero_to_62_s"})
    await run_both(ctx)
    questions = only_call_questions(fake)
    assert sorted(questions) == ["Car.zero_to_62_s/choice0", "Car.zero_to_62_s/choice1"]
    assert all(isinstance(q, Choice) and len(q.options) <= 255 for q in questions.values())
    sel = ctx.schemas["Car"].selections[("doc", "zero_to_62_s", "s1")]
    assert sel.candidate is not None
    assert sel.candidate.raw == "299"


async def test_list_candidate_fields_accept_several_spans_per_statement() -> None:
    fake = FakeJev(default_p=0.1)
    fake.noul('"SE"', p=0.9)  # the member question quotes the span exactly
    fake.noul('"GT"', p=0.8)
    ctx = context(fake, [st("s1", "Trims: SE, GT")], {"s1": "trims"})
    await run_both(ctx)
    questions = only_call_questions(fake)
    assert all(isinstance(q, Noul) for q in questions.values())
    assert Noul(instructions='Does the statement give "SE" as one of the trim names?') in (
        questions.values()
    )
    sel = ctx.schemas["Car"].selections[("doc", "trims", "s1")]
    assert [c.raw for c in sel.accepted] == ["SE", "GT"]
    assert sel.candidate is not None
    assert sel.candidate.raw == "SE"


async def test_no_candidates_means_no_question() -> None:
    fake = FakeJev(strict=True)
    ctx = context(fake, [st("s1", "Quick off the line")], {"s1": "zero_to_62_s"})
    await run_both(ctx)
    assert fake.calls == []
    assert ctx.schemas["Car"].selections == {}


# --- direct answers --------------------------------------------------------------------


async def test_enum_is_answered_directly() -> None:
    fake = FakeJev().choice("What is the fuel type", "diesel", confidence=0.93)
    ctx = context(fake, [st("s1", "Runs on diesel")], {"s1": "fuel_type"})
    await run_both(ctx)
    assert only_call_questions(fake) == {
        "Car.fuel_type/enum": Choice(
            instructions="What is the fuel type?",
            options={
                "petrol": None,
                "diesel": None,
                "ev": None,
                "not stated": "The statement does not state the fuel type",
            },
        )
    }
    meta = ctx.schemas["Car"].fields["doc"]["fuel_type"]
    assert (meta.value, meta.confidence, meta.method) == ("diesel", 0.93, "jev")
    assert meta.source is not None
    assert meta.source.statement == "Runs on diesel"
    assert {a.value for a in meta.alternatives} == {"petrol", "ev"}  # never "not stated"


async def test_enum_not_stated_records_nothing() -> None:
    fake = FakeJev().choice("What is the fuel type", "not stated")
    ctx = context(fake, [st("s1", "Fuel: see brochure")], {"s1": "fuel_type"})
    await run_both(ctx)
    assert "fuel_type" not in ctx.schemas["Car"].fields.get("doc", {})


async def test_most_confident_enum_answer_wins_ties_to_the_earliest() -> None:
    fake = FakeJev()
    fake.choice("What is the fuel type", "petrol", confidence=0.9, state="petrol engine")
    fake.choice("What is the fuel type", "diesel", confidence=0.9, state="diesel only")
    a, b = st("s1", "A petrol engine"), st("s2", "Available as diesel only")
    ctx = context(fake, [a, b], {"s1": "fuel_type", "s2": "fuel_type"})
    await run_both(ctx)
    assert ctx.schemas["Car"].fields["doc"]["fuel_type"].value == "petrol"


async def test_bool_via_noul_both_ways() -> None:
    fake = FakeJev()
    fake.noul("automatic gearbox", p=0.2, state="manual")
    fake.noul("automatic gearbox", p=0.9, state="auto box")
    ctx = context(fake, [st("s1", "Six-speed manual gearbox")], {"s1": "automatic"})
    await run_both(ctx)
    assert only_call_questions(fake) == {
        "Car.automatic/bool": Noul(
            instructions="Does the statement say it has an automatic gearbox?"
        )
    }
    meta = ctx.schemas["Car"].fields["doc"]["automatic"]
    assert (meta.value, meta.confidence) == (False, 0.8)

    ctx = context(fake, [st("s1", "An auto box")], {"s1": "automatic"})
    await run_both(ctx)
    meta = ctx.schemas["Car"].fields["doc"]["automatic"]
    assert (meta.value, meta.confidence) == (True, 0.9)


async def test_direct_answers_prefer_the_entitys_own_statements_over_shared_ones() -> None:
    fake = FakeJev()
    fake.noul("automatic gearbox", p=0.95, state="Every")
    fake.noul("automatic gearbox", p=0.3, state="manual")
    own = st("s1", "The SE has a manual gearbox", component="c1")
    everyone = st("s2", "Every trim has an automatic gearbox", component="c2")
    ctx = context(fake, [own, everyone], {"s1": "automatic", "s2": "automatic"})
    run = ctx.schemas["Car"]
    run.scopes = [
        EntityScope(label="SE", statement_ids=["s1"], shared_statement_ids=["s2"]),
        EntityScope(label="SE L", shared_statement_ids=["s2"]),
    ]
    await run_both(ctx)
    se, se_l = run.fields["SE"]["automatic"], run.fields["SE L"]["automatic"]
    # SE's own statement wins even though the shared one is more confident.
    assert (se.value, se.shared) == (False, False)
    assert (se_l.value, se_l.shared) == (True, True)
    assert len(fake.calls) == 2  # the shared statement is still asked about once


async def test_list_enum_collects_options_within_and_across_statements() -> None:
    fake = FakeJev(default_p=0.05)
    fake.noul('"red"', p=0.9, state="red and blue")
    fake.noul('"blue"', p=0.8, state="red and blue")
    fake.noul('"grey"', p=0.95, state="grey")
    a, b = st("s1", "Paint: red and blue"), st("s2", "Also grey")
    ctx = context(fake, [a, b], {"s1": "colours", "s2": "colours"})
    await run_both(ctx)
    assert (
        Noul(instructions='Does the statement give "red" as one of the colours?') in fake.questions
    )
    meta = ctx.schemas["Car"].fields["doc"]["colours"]
    assert meta.value == ["red", "blue", "grey"]
    assert all(alt.value not in meta.value for alt in meta.alternatives)


async def test_results_dont_depend_on_reply_order() -> None:
    class Slow(FakeJev):
        async def system_one(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
            if "first" in str(state):
                await asyncio.sleep(0.02)  # the earlier statement's reply arrives last
            return await super().system_one(state, questions)

    fake = Slow(default_p=0.05)
    fake.noul('"red"', p=0.9, state="first")
    fake.noul('"grey"', p=0.9, state="second")
    a, b = st("s1", "The first: red"), st("s2", "The second: grey")
    ctx = context(fake, [a, b], {"s1": "colours", "s2": "colours"})
    await run_both(ctx)
    assert ctx.schemas["Car"].fields["doc"]["colours"].value == ["red", "grey"]


async def test_values_from_other_routes_are_not_overwritten() -> None:
    fake = FakeJev().choice("What is the fuel type", "diesel")
    ctx = context(fake, [st("s1", "Runs on diesel")], {"s1": "fuel_type"})
    ctx.schemas["Car"].set_field("doc", "fuel_type", FieldMeta(value="petrol", method="structured"))
    await run_both(ctx)
    meta = ctx.schemas["Car"].fields["doc"]["fuel_type"]
    assert (meta.value, meta.method) == ("petrol", "structured")
    assert fake.calls == []  # nothing is asked about a field another route found


async def test_merge_asks_about_found_fields_and_records_the_losing_value() -> None:
    fake = FakeJev().choice("What is the fuel type", "diesel", confidence=0.9)
    ctx = context(fake, [st("s1", "Runs on diesel")], {"s1": "fuel_type"})
    run = ctx.schemas["Car"]
    run.merge = True
    run.set_field(
        "doc", "fuel_type", FieldMeta(value="petrol", confidence=0.6, method="structured")
    )
    await run_both(ctx)
    meta = run.fields["doc"]["fuel_type"]
    assert (meta.value, meta.method, meta.confidence) == ("diesel", "jev", 0.9)
    assert meta.conflicts == [Conflict(value="petrol", method="structured", confidence=0.6)]


# --- the unit of a bare number ---------------------------------------------------------


class Engine(BaseModel):
    power_ps: float = Field(description="Power", unit="PS")
    economy: float = Field(description="Fuel economy", unit="l/100km")
    price: Decimal = Field(description="Price", unit="GBP")
    doors: int = Field(description="Number of doors")


async def extract_engine(
    fake: FakeJev, text: str, field: str, document: Document | None = None
) -> FieldMeta:
    ctx = context(fake, [st("s1", text)], {"s1": field}, models=(Engine,), document=document)
    await run_both(ctx)
    await NormaliseStage().run(ctx)
    return ctx.schemas["Engine"].fields["doc"][field]


@pytest.mark.parametrize(("unit", "value"), [("kW", 149.558), ("PS", 110)])
async def test_a_bare_number_is_read_in_the_unit_jev_picks(unit: str, value: float) -> None:
    fake = (
        FakeJev()
        .choice("Which of these is the power", "110", confidence=0.9)
        .choice('Which unit is "110" in?', unit, confidence=0.95)
    )
    meta = await extract_engine(fake, "Power (kW) · SE: 110", "power_ps")
    assert only_call_questions(fake) == {
        "Engine.power_ps/choice0": Choice(
            instructions="Which of these is the power (PS)?",
            options={"110": None, "none": "None of these is the power"},
        ),
        "Engine.power_ps#unit0": Choice(
            instructions='Which unit is "110" in?', options={"PS": None, "kW": None}
        ),
    }
    assert meta.value == pytest.approx(value, abs=0.001)
    assert meta.confidence == 0.9


async def test_a_value_is_only_as_confident_as_its_unit() -> None:
    fake = (
        FakeJev()
        .choice("Which of these is the power", "110", confidence=0.9)
        .choice('Which unit is "110" in?', "kW", confidence=0.55)
    )
    meta = await extract_engine(fake, "Power (kW) · SE: 110", "power_ps")
    assert meta.value == pytest.approx(149.558, abs=0.001)
    assert meta.confidence == 0.55


async def test_a_list_value_is_only_as_confident_as_its_least_sure_unit() -> None:
    class Powers(BaseModel):
        outputs_ps: list[float] = Field(
            default_factory=list, description="Power outputs", unit="PS"
        )

    fake = (
        FakeJev()
        .noul("as one of the power outputs", p=0.9)
        .choice('Which unit is "110" in?', "kW", confidence=0.6)
        .choice('Which unit is "150" in?', "PS", confidence=0.95)
    )
    ctx = context(fake, [st("s1", "Power (kW): 110 / 150")], {"s1": "outputs_ps"}, models=(Powers,))
    await run_both(ctx)
    await NormaliseStage().run(ctx)
    meta = ctx.schemas["Powers"].fields["doc"]["outputs_ps"]
    assert meta.value == [pytest.approx(149.558, abs=0.001), 150]
    assert meta.confidence == 0.6


async def test_each_bare_span_gets_its_own_unit_question() -> None:
    fake = (
        FakeJev()
        .choice("Which of these is the power", "150", confidence=0.9)
        .choice('Which unit is "110" in?', "kW")
        .choice('Which unit is "150" in?', "PS")
    )
    ctx = context(
        fake,
        [st("s1", "Power (kW): 110 / 150 / 184 hp, 250 Nm")],
        {"s1": "power_ps"},
        models=(Engine,),
    )
    await run_both(ctx)
    units = {k: q for k, q in only_call_questions(fake).items() if "#unit" in k}
    # "184 hp" says its unit already; Nm is torque, so it isn't offered.
    options: dict[str, JSONContent | None] = {"PS": None, "kW": None, "hp": None}
    assert units == {
        "Engine.power_ps#unit0": Choice(instructions='Which unit is "110" in?', options=options),
        "Engine.power_ps#unit1": Choice(instructions='Which unit is "150" in?', options=options),
    }
    chains = {c.raw: c.normalise for c in ctx.schemas["Engine"].candidates[("s1", "power_ps")]}
    assert chains["110"] == [
        NormaliserStep(name="parse_number"),
        NormaliserStep(name="unit", args={"from": "kW"}),
    ]
    assert chains["150"] == [
        NormaliserStep(name="parse_number"),
        NormaliserStep(name="unit", args={"from": "PS"}),
    ]
    assert chains["184 hp"] == [
        NormaliserStep(name="parse_number"),
        NormaliserStep(name="unit", args={"from": "hp"}),
    ]


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("Power (PS) · SE: 110", "power_ps"),  # only the field's own unit
        ("Power: 110 kW", "power_ps"),  # the number carries its unit
        ("Power · SE: 110, 0-62 mph in 9 s", "power_ps"),  # units of other dimensions
        ("Price (USD) · SE: 18,495", "price"),  # a currency: nothing to convert
        ("Doors (5 m long) · 5", "doors"),  # a field without a unit
    ],
)
async def test_statements_naming_no_other_unit_ask_nothing_new(text: str, field: str) -> None:
    fake = FakeJev(strict=True).choice("Which of these is the", "none")
    await run_both(context(fake, [st("s1", text)], {"s1": field}, models=(Engine,)))
    assert list(only_call_questions(fake)) == [f"Engine.{field}/choice0"]


async def test_the_unit_question_can_be_overridden() -> None:
    class Overridden(BaseModel):
        power_ps: float = Field(
            description="Power",
            unit="PS",
            questions=Questions(unit='Is "{value}" in PS or kW? ({description})'),
        )

    fake = FakeJev()
    ctx = context(
        fake, [st("s1", "Power (kW) · SE: 110")], {"s1": "power_ps"}, models=(Overridden,)
    )
    await run_both(ctx)
    question = only_call_questions(fake)["Overridden.power_ps#unit0"]
    assert question.instructions == 'Is "110" in PS or kW? (Power (PS))'


async def test_mpg_is_read_in_the_pages_gallons() -> None:
    fake = (
        FakeJev()
        .choice("Which of these is the fuel economy", "40", confidence=0.9)
        .choice('Which unit is "40" in?', "mpg")
    )
    us = Document.from_bytes(b"<p/>", url="https://example.com", locale="en-US")
    meta = await extract_engine(fake, "Economy (mpg) · Combined: 40", "economy", document=us)
    assert meta.value == pytest.approx(235.214583 / 40)


async def test_mpg_is_read_in_the_candidate_stages_gallons_without_a_page_locale() -> None:
    fake = (
        FakeJev()
        .choice("Which of these is the fuel economy", "40", confidence=0.9)
        .choice('Which unit is "40" in?', "mpg")
    )
    ctx = context(
        fake, [st("s1", "Economy (mpg) · Combined: 40")], {"s1": "economy"}, models=(Engine,)
    )
    candidates = CandidateStage(locale="en-US")
    ctx.pipeline = Pipeline([candidates, SelectStage(), NormaliseStage()])
    for stage in ctx.pipeline:
        await stage.run(ctx)
    meta = ctx.schemas["Engine"].fields["doc"]["economy"]
    assert meta.value == pytest.approx(235.214583 / 40)


async def test_a_unit_that_cannot_convert_gives_a_normalise_error_not_a_value() -> None:
    fake = (
        FakeJev()
        .choice("Which of these is the fuel economy", "0", confidence=0.9)
        .choice('Which unit is "0" in?', "mpg")
    )
    meta = await extract_engine(fake, "Economy (mpg) · Electric: 0", "economy")
    assert meta.value is None
    assert meta.error == "can't convert zero fuel economy"
    assert meta.confidence == 0.9


# --- batching --------------------------------------------------------------------------


async def test_one_request_per_statement_across_schemas_and_scopes() -> None:
    fake = FakeJev()
    ctx = context(fake, [st("s1", "0-62 in 9.1 s")], {"s1": "zero_to_62_s"}, models=(Car, Rival))
    ctx.schemas["Car"].scopes.append(EntityScope(label="also", component_ids=["c1"]))
    await run_both(ctx)
    questions = only_call_questions(fake)
    assert sorted(questions) == ["Car.zero_to_62_s/choice0", "Rival.zero_to_62_s/choice0"]
    car = ctx.schemas["Car"].selections
    assert ("doc", "zero_to_62_s", "s1") in car
    assert ("also", "zero_to_62_s", "s1") in car


async def test_one_request_per_statement() -> None:
    fake = FakeJev()
    statements = [st(f"s{i}", f"0-62 in {i}.5 s") for i in range(5)]
    ctx = context(fake, statements, {s.id: "zero_to_62_s" for s in statements})
    await run_both(ctx)
    assert len(fake.calls) == 5


# --- pluggable selector ----------------------------------------------------------------


def test_default_selector_satisfies_the_protocol() -> None:
    assert isinstance(JevCandidateSelector(), CandidateSelector)


def test_stages_are_in_the_default_pipeline_in_order() -> None:
    from jevex.extractor import default_pipeline

    names = default_pipeline().names
    assert names.index("candidates") < names.index("select") < names.index("normalise")


async def test_list_enum_options_follow_the_text_order_within_a_statement() -> None:
    fake = FakeJev(default_p=0.05)
    fake.noul('"red"', p=0.9)
    fake.noul('"grey"', p=0.9)
    ctx = context(fake, [st("s1", "Grey or red")], {"s1": "colours"})
    await run_both(ctx)
    assert ctx.schemas["Car"].fields["doc"]["colours"].value == ["grey", "red"]


async def test_answers_from_a_vision_statement_are_tagged_vision() -> None:
    fake = FakeJev().choice("What is the fuel type", "diesel", confidence=0.93)
    said = st("s1", "The badge on the boot reads TDI, a diesel").model_copy(
        update={"kind": "vision"}
    )
    ctx = context(fake, [said], {"s1": "fuel_type"})
    await run_both(ctx)
    meta = ctx.schemas["Car"].fields["doc"]["fuel_type"]
    assert (meta.value, meta.method) == ("diesel", "vision")
    assert ctx.schemas["Car"].vision_values == {("doc", "fuel_type"): [VisionValue("s1", "diesel")]}


async def test_list_options_a_text_statement_also_gives_need_no_vision_check() -> None:
    fake = FakeJev(default_p=0.05).noul('"red"', p=0.9).noul('"grey"', p=0.9, state="Grey")
    said = st("s1", "Grey and red paint").model_copy(update={"kind": "vision"})
    ctx = context(fake, [said, st("s2", "In red")], {"s1": "colours", "s2": "colours"})
    await run_both(ctx)
    run = ctx.schemas["Car"]
    assert run.fields["doc"]["colours"].value == ["grey", "red"]
    assert run.vision_values == {("doc", "colours"): [VisionValue("s1", "grey")]}
    picks = run.value_picks[("doc", "colours")]
    assert [(p.items, p.method, p.source.statement_id) for p in picks] == [
        (("grey", "red"), "vision", "s1"),
        (("red",), "jev", "s2"),
    ]
    assert picks[0].source == run.fields["doc"]["colours"].source

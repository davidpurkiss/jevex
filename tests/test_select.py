import asyncio
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel

from jevex import Context, Document, DomLocation, Field, SchemaSpec, Statement
from jevex.entities import EntityScope
from jevex.interfaces import CandidateSelector, ParsedDocument
from jevex.jev import Choice, ChoiceAnswer, JevResponse, Noul, Question
from jevex.layout import MAX_SECTION_CHARS, Component
from jevex.results import FieldMeta
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
) -> Context:
    ctx = Context.create(
        Document.from_bytes(b"<p/>", url="https://example.com"),
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
        probabilities={"zero_to_62_s": 0.5, "automatic": 0.35, "model": 0.1, "none": 0.05},
    )
    pairs = field_statements(ctx, run, run.scopes[0])
    assert [f.name for _, f in pairs] == ["zero_to_62_s", "automatic"]
    run.categories["s1"] = ChoiceAnswer(
        choice="model",
        confidence=0.75,
        probabilities={"model": 0.75, "automatic": 0.25},
    )
    assert [f.name for _, f in field_statements(ctx, run, run.scopes[0])] == ["model"]
    # A "none" answer routes nowhere, however close a field came.
    run.categories["s1"] = ChoiceAnswer(
        choice="none", confidence=0.6, probabilities={"none": 0.6, "automatic": 0.4}
    )
    assert field_statements(ctx, run, run.scopes[0]) == []


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

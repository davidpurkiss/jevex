from typing import Literal

from pydantic import BaseModel

from jevex import Context, Document, DomLocation, Field, SchemaSpec, Statement
from jevex.entities import EntityScope
from jevex.interfaces import ParsedDocument
from jevex.jev import Choice, ChoiceAnswer, Noul
from jevex.layout import Component
from jevex.select import CandidateStage, SelectStage, field_statements, statement_state
from jevex.testing import FakeJev

LOC = DomLocation(dom_path="/p")


class Car(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    fuel_type: Literal["petrol", "diesel", "ev"] = Field(description="Fuel type")
    automatic: bool = Field(description="has an automatic gearbox")
    colours: list[Literal["red", "blue", "grey"]] = Field(
        default_factory=list, description="Colours"
    )
    model: str = Field(description="Model name")


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


def context(fake: FakeJev, statements: list[Statement], categories: dict[str, str]) -> Context:
    ctx = Context.create(
        Document.from_bytes(b"<p/>", url="https://example.com"), [SPEC], fake.client()
    )
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={s.id: s for s in statements},
    )
    run = ctx.schemas["Car"]
    run.scopes = [
        EntityScope(label="doc", component_ids=sorted({s.component_id for s in statements}))
    ]
    for sid, name in categories.items():
        run.categories[sid] = ChoiceAnswer(choice=name, confidence=0.9, probabilities={name: 0.9})
    return ctx


async def run_both(ctx: Context) -> None:
    await CandidateStage().run(ctx)
    await SelectStage().run(ctx)


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


# --- selection over candidates ---------------------------------------------------------


async def test_choice_over_candidates_plus_none() -> None:
    fake = FakeJev().choice("Which of these is the 0-62 mph time", "9.1 s", confidence=0.8)
    a = st("s1", "0-62 mph in 9.1 s", trail=["Performance"])
    ctx = context(fake, [a], {"s1": "zero_to_62_s"})
    await run_both(ctx)

    [call] = fake.calls
    assert call.state == {"statement": "0-62 mph in 9.1 s", "section": "Performance"}
    question = call.questions["zero_to_62_s"]
    assert question == Choice(
        instructions="Which of these is the 0-62 mph time (s)?",
        options={
            "0": None,
            "0-62 mph": None,
            "62 mph": None,
            "9.1 s": None,
            "none": "None of these is the 0-62 mph time",
        },
    )
    sel = ctx.schemas["Car"].selections[("doc", "zero_to_62_s", "s1")]
    assert sel.candidate is not None
    assert sel.candidate.raw == "9.1 s"
    assert sel.confidence == 0.8
    assert set(sel.alternatives) == {"0", "0-62 mph", "62 mph"}


async def test_none_is_a_selection_without_a_candidate() -> None:
    fake = FakeJev().choice("Which of these", "none", confidence=0.7)
    ctx = context(fake, [st("s1", "0-62 mph in 9.1 s")], {"s1": "zero_to_62_s"})
    await run_both(ctx)
    sel = ctx.schemas["Car"].selections[("doc", "zero_to_62_s", "s1")]
    assert sel.candidate is None
    assert sel.confidence == 0.7


async def test_no_candidates_means_no_question() -> None:
    fake = FakeJev(strict=True)
    ctx = context(fake, [st("s1", "Quick off the line")], {"s1": "zero_to_62_s"})
    await run_both(ctx)
    assert fake.calls == []
    assert ctx.schemas["Car"].selections == {}


# --- direct answers --------------------------------------------------------------------


async def test_enum_is_answered_directly() -> None:
    fake = FakeJev().choice("What is the Fuel type", "diesel", confidence=0.93)
    ctx = context(fake, [st("s1", "Runs on diesel")], {"s1": "fuel_type"})
    await run_both(ctx)
    [call] = fake.calls
    assert call.questions["fuel_type"] == SPEC.field("fuel_type").enum_question()
    meta = ctx.schemas["Car"].fields["doc"]["fuel_type"]
    assert (meta.value, meta.confidence, meta.method) == ("diesel", 0.93, "jev")
    assert meta.source is not None
    assert meta.source.statement == "Runs on diesel"
    assert {a.value for a in meta.alternatives} == {"petrol", "ev", "not stated"}


async def test_enum_not_stated_records_nothing() -> None:
    fake = FakeJev().choice("What is the Fuel type", "not stated")
    ctx = context(fake, [st("s1", "Fuel: see brochure")], {"s1": "fuel_type"})
    await run_both(ctx)
    assert "fuel_type" not in ctx.schemas["Car"].fields.get("doc", {})


async def test_most_confident_enum_answer_wins_across_statements() -> None:
    fake = FakeJev()
    fake.choice("What is the Fuel type", "petrol", confidence=0.6, state="petrol engine")
    fake.choice("What is the Fuel type", "diesel", confidence=0.95, state="diesel only")
    a, b = st("s1", "A petrol engine"), st("s2", "Available as diesel only")
    ctx = context(fake, [a, b], {"s1": "fuel_type", "s2": "fuel_type"})
    await run_both(ctx)
    assert ctx.schemas["Car"].fields["doc"]["fuel_type"].value == "diesel"


async def test_bool_via_noul() -> None:
    fake = FakeJev().noul("Does the statement say has an automatic gearbox", p=0.2)
    ctx = context(fake, [st("s1", "Six-speed manual gearbox")], {"s1": "automatic"})
    await run_both(ctx)
    [call] = fake.calls
    assert call.questions["automatic"] == Noul(
        instructions="Does the statement say has an automatic gearbox?"
    )
    meta = ctx.schemas["Car"].fields["doc"]["automatic"]
    assert meta.value is False
    assert meta.confidence == 0.8
    assert meta.found


async def test_list_enum_collects_every_stated_option() -> None:
    fake = FakeJev()
    fake.choice("What is the Colours", "red", state="red")
    fake.choice("What is the Colours", "grey", state="grey")
    a, b, c = st("s1", "Paint: red"), st("s2", "Also grey"), st("s3", "Or red again")
    ctx = context(fake, [a, b, c], {"s1": "colours", "s2": "colours", "s3": "colours"})
    await run_both(ctx)
    assert ctx.schemas["Car"].fields["doc"]["colours"].value == ["red", "grey"]


async def test_one_request_per_statement() -> None:
    fake = FakeJev()
    statements = [st(f"s{i}", f"0-62 in {i}.5 s") for i in range(5)]
    ctx = context(fake, statements, {s.id: "zero_to_62_s" for s in statements})
    await run_both(ctx)
    assert len(fake.calls) == 5
    assert all(len(call.questions) == 1 for call in fake.calls)


def test_stages_are_in_the_default_pipeline_in_order() -> None:
    from jevex.extractor import default_pipeline

    names = default_pipeline().names
    assert names.index("candidates") < names.index("select")

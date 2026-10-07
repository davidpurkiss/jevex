from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel

from jevex import (
    Budgets,
    Context,
    DocBudget,
    Document,
    DomLocation,
    Extractor,
    Field,
    FieldSpec,
    LLMAnswer,
    Pipeline,
    SchemaSpec,
    Span,
    Statement,
)
from jevex.budgets import DocumentBudget
from jevex.entities import EntityScope
from jevex.extractor import default_pipeline
from jevex.fallback import PROMPT, FallbackStage, LLMFieldExtractor, LLMOutput, output_model
from jevex.interfaces import LLMExtractor, ParsedDocument, Selection
from jevex.jev import ChoiceAnswer, Noul
from jevex.layout import Component
from jevex.llm import LLMError
from jevex.pipeline import VisionValue
from jevex.results import Alternative, FieldMeta, Source
from jevex.statements import Candidate
from jevex.testing import FakeJev, FakeLLM

LOC = DomLocation(dom_path="/p")


class Car(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    seats: int = Field(description="Number of seats", ge=1)
    launched: date = Field(description="Launch date")
    trims: list[str] = Field(default_factory=list, description="Trim names")
    automatic: bool = Field(description="has an automatic gearbox")


SPEC = SchemaSpec.from_model(Car)
TEXT = "It reaches 62 mph from rest in 9.1 seconds."


def st(sid: str, text: str = TEXT, trail: list[str] | None = None) -> Statement:
    return Statement(
        id=sid,
        text=text,
        kind="sentence",
        component_id="c1",
        location=LOC,
        heading_trail=trail or [],
    )


def context(
    fake: FakeJev,
    statements: list[Statement],
    categories: dict[str, str],
    *,
    category_p: float = 0.9,
    scopes: tuple[str, ...] = ("doc",),
) -> Context:
    ctx = Context.create(
        Document.from_bytes(b"<p/>", url="https://example.com"), [SPEC], fake.client()
    )
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={s.id: s for s in statements},
    )
    run = ctx.schemas["Car"]
    run.scopes = [EntityScope(label=label, component_ids=["c1"]) for label in scopes]
    for sid, name in categories.items():
        run.categories[sid] = ChoiceAnswer(
            choice=name, confidence=category_p, probabilities={name: category_p}
        )
    return ctx


def candidate(raw: str, start: int = 0) -> Candidate:
    return Candidate(span=Span(start=start, end=start + len(raw)), raw=raw, generator_id="g")


def picked(ctx: Context, sid: str, name: str, raw: str | None, confidence: float) -> None:
    """Leave what candidates, select and normalise would have: a pick (or "none")."""
    run = ctx.schemas["Car"]
    statement = ctx.parsed.statements[sid] if ctx.parsed else None
    assert statement is not None
    cand = candidate(raw or "9", max(statement.text.find(raw or "9"), 0))
    run.candidates[(sid, name)] = [cand]
    for scope in run.scopes:
        run.selections[(scope.label, name, sid)] = Selection(
            candidate=cand if raw else None,
            confidence=confidence,
            accepted=[cand] if raw else [],
        )
        if raw:
            run.set_field(
                scope.label,
                name,
                FieldMeta(
                    value=float(raw),
                    confidence=confidence,
                    method="generator",
                    source=Source(statement_id=sid, statement=statement.text, span=cand.span),
                ),
            )


def llm(**output: Any) -> FakeLLM:
    return FakeLLM(lambda _p, _s: {"stated": True, **output})


VERIFY = "The statement states that the 0-62 mph time (s) is 9.1."


def meta(ctx: Context, name: str = "zero_to_62_s", scope: str = "doc") -> FieldMeta:
    return ctx.schemas["Car"].fields[scope][name]


# --- off by default ----------------------------------------------------------------------


async def test_without_an_llm_the_stage_does_nothing() -> None:
    fake = FakeJev(strict=True)
    ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"})
    await FallbackStage().run(ctx)
    assert fake.calls == []
    assert ctx.schemas["Car"].fields == {}


def test_the_stage_is_in_the_default_pipeline_after_normalise() -> None:
    names = default_pipeline().names
    assert names.index("fallback") == names.index("normalise") + 1


@pytest.mark.parametrize("name", ["category_threshold", "fallback_threshold", "verify_threshold"])
def test_thresholds_must_be_probabilities(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        FallbackStage(**{name: 1.5})  # pyright: ignore[reportArgumentType]


# --- triggers ----------------------------------------------------------------------------


async def test_no_candidates_falls_back_and_a_verified_answer_is_used() -> None:
    fake = FakeJev(strict=True).noul(VERIFY, p=0.95)
    ctx = context(fake, [st("s1", trail=["Performance"])], {"s1": "zero_to_62_s"})
    ctx.extraction_llm = model = llm(value=9.1, evidence="9.1 seconds")
    await FallbackStage().run(ctx)

    [call] = model.calls
    assert call.prompt == (
        "Read one statement from a document and extract a single field from it.\n\n"
        "Section: Performance\n"
        f"Statement: {TEXT}\n\n"
        "Field: zero_to_62_s\n"
        "Description: 0-62 mph time (s)\n"
        "Type: a number\n\n"
        'If the statement states the 0-62 mph time (s), set "stated" to true, give the value '
        'in s, and copy\nthe words of the statement that state it into "evidence", exactly as '
        'they appear. If it\ndoesn\'t state it, set "stated" to false and leave the rest '
        "empty. Never guess."
    )
    [jev_call] = fake.calls
    assert jev_call.state == {"statement": TEXT, "section": "Performance"}
    assert jev_call.questions == {"Car.zero_to_62_s/verify": Noul(instructions=VERIFY)}

    m = meta(ctx)
    assert (m.value, m.confidence, m.method, m.verified) == (9.1, 0.95, "llm", True)
    assert m.source is not None
    assert m.source.span is not None
    assert m.source.span.of(TEXT) == "9.1 seconds"
    assert m.source.statement_id == "s1"
    [example] = ctx.verified
    assert example.field == "Car.zero_to_62_s"
    assert (example.value, example.probability, example.source) == (9.1, 0.95, "llm")
    assert example.evidence == (TEXT.index("9.1"), TEXT.index("9.1") + len("9.1 seconds"))
    assert example.context["heading_trail"] == ["Performance"]
    assert example.document_source == "example.com"  # for the learner's generator scoping


@pytest.mark.parametrize(
    ("content", "locale"), [(b'<html lang="de-DE"><p/></html>', "de-DE"), (b"<p/>", None)]
)
async def test_the_example_carries_the_documents_own_locale(
    content: bytes, locale: str | None
) -> None:
    fake = FakeJev(strict=True).noul(VERIFY, p=0.95)
    ctx = context(fake, [st("s1", trail=["Performance"])], {"s1": "zero_to_62_s"})
    ctx.document = Document.from_bytes(content, url="https://example.com")
    ctx.extraction_llm = llm(value=9.1, evidence="9.1 seconds")
    await FallbackStage().run(ctx)
    [example] = ctx.verified
    assert example.locale == locale  # what the learner scopes its generator to
    assert ("locale" in example.context) == (locale is not None)
    assert example.context["heading_trail"] == ["Performance"]


async def test_none_falls_back_only_when_the_category_was_confident() -> None:
    for category_p, calls in [(0.6, 1), (0.4, 0)]:
        fake = FakeJev().noul(VERIFY, p=0.95)
        ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"}, category_p=category_p)
        picked(ctx, "s1", "zero_to_62_s", None, 0.9)
        ctx.extraction_llm = model = llm(value=9.1, evidence="9.1")
        await FallbackStage(category_threshold=0.5).run(ctx)
        assert len(model.calls) == calls


async def test_low_confidence_falls_back_and_the_jev_answer_becomes_an_alternative() -> None:
    fake = FakeJev().noul(VERIFY, p=0.9)
    ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"})
    picked(ctx, "s1", "zero_to_62_s", "62", 0.3)
    ctx.extraction_llm = llm(value=9.1, evidence="9.1 seconds")
    await FallbackStage(fallback_threshold=0.5).run(ctx)
    m = meta(ctx)
    assert (m.value, m.method) == (9.1, "llm")
    assert m.alternatives == [Alternative(value=62.0, raw="62", p=0.3)]


async def test_a_confident_selection_or_another_routes_value_is_left_alone() -> None:
    fake = FakeJev(strict=True)
    ctx = context(fake, [st("s1"), st("s2", "Seats five.")], {"s1": "zero_to_62_s", "s2": "seats"})
    picked(ctx, "s1", "zero_to_62_s", "9.1", 0.8)
    ctx.schemas["Car"].set_field("doc", "seats", FieldMeta(value=5, method="structured"))
    ctx.extraction_llm = model = llm(value=1, evidence="x")
    await FallbackStage().run(ctx)
    assert model.calls == []
    assert fake.calls == []


async def test_enum_and_bool_fields_never_fall_back() -> None:
    ctx = context(FakeJev(strict=True), [st("s1", "Automatic.")], {"s1": "automatic"})
    ctx.extraction_llm = model = llm(value=True, evidence="Automatic")
    await FallbackStage().run(ctx)
    assert model.calls == []


# --- checks before verification ----------------------------------------------------------


async def test_rejected_answer_keeps_the_jev_answer_and_lists_the_llm_value() -> None:
    fake = FakeJev().noul(VERIFY, p=0.4)
    ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"})
    picked(ctx, "s1", "zero_to_62_s", "62", 0.3)
    ctx.extraction_llm = llm(value=9.1, evidence="9.1 seconds")
    await FallbackStage().run(ctx)
    m = meta(ctx)
    assert (m.value, m.confidence, m.method, m.verified) == (62.0, 0.3, "generator", None)
    assert m.alternatives == [Alternative(value=9.1, raw="9.1 seconds", p=0.4)]
    assert ctx.verified == []
    [event] = ctx.events
    assert (event.stage, event.kind) == ("fallback", "llm_rejected")
    assert event.data["trigger"] == "low_confidence"


async def test_a_rejected_answer_equal_to_the_jev_value_isnt_listed_as_its_alternative() -> None:
    ctx = context(FakeJev().noul(VERIFY, p=0.4), [st("s1")], {"s1": "zero_to_62_s"})
    picked(ctx, "s1", "zero_to_62_s", "9.1", 0.3)
    ctx.extraction_llm = llm(value=9.1, evidence="9.1")
    await FallbackStage().run(ctx)
    m = meta(ctx)
    assert (m.value, m.confidence, m.alternatives) == (9.1, 0.3, [])
    assert [e.kind for e in ctx.events] == ["llm_rejected"]


async def test_rejected_answer_with_no_jev_answer_leaves_the_field_empty() -> None:
    ctx = context(FakeJev().noul(VERIFY, p=0.1), [st("s1")], {"s1": "zero_to_62_s"})
    ctx.extraction_llm = llm(value=9.1, evidence="9.1")
    await FallbackStage().run(ctx)
    m = meta(ctx)
    assert not m.found
    assert m.alternatives == [Alternative(value=9.1, raw="9.1", p=0.1)]


@pytest.mark.parametrize("evidence", ["9.1 secs", "", "  "])
async def test_evidence_not_in_the_statement_is_dropped_unverified(evidence: str) -> None:
    fake = FakeJev(strict=True)
    ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"})
    ctx.extraction_llm = llm(value=9.1, evidence=evidence)
    await FallbackStage().run(ctx)
    assert fake.calls == []
    assert ctx.schemas["Car"].fields == {}
    assert [e.kind for e in ctx.events] == ["llm_no_evidence"]


@pytest.mark.parametrize("evidence", ["9.1 s", "9.1\u00a0s", "9.1  s"])
async def test_evidence_matches_whatever_whitespace_separates_its_words(evidence: str) -> None:
    fake = FakeJev().noul(VERIFY, p=0.95)
    ctx = context(fake, [st("s1", "0-62 mph in 9.1\u00a0s")], {"s1": "zero_to_62_s"})
    ctx.extraction_llm = llm(value=9.1, evidence=evidence)
    await FallbackStage().run(ctx)
    m = meta(ctx)
    assert m.value == 9.1
    assert m.source is not None
    assert m.source.span == Span(start=12, end=17)


async def test_a_value_that_doesnt_fit_the_field_is_dropped_unverified() -> None:
    fake = FakeJev(strict=True)
    ctx = context(fake, [st("s1", "Seats: none")], {"s1": "seats"})
    ctx.extraction_llm = llm(value=0, evidence="none")
    await FallbackStage().run(ctx)
    assert fake.calls == []
    [event] = ctx.events
    assert event.kind == "llm_invalid"
    assert "seats" in event.message


async def test_not_stated_asks_jev_nothing() -> None:
    fake = FakeJev(strict=True)
    ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"})
    ctx.extraction_llm = FakeLLM([{"stated": False}])
    await FallbackStage().run(ctx)
    assert fake.calls == []
    assert ctx.schemas["Car"].fields == {}


async def test_an_llm_error_is_an_event_and_other_fields_carry_on() -> None:
    text = "Launched 3 March 2024, 0-62 in 9.1 s."

    def answer(prompt: str, _schema: type[BaseModel]) -> object:
        if "zero_to_62_s" in prompt:
            raise LLMError("upstream 500")
        return {"stated": True, "value": "2024-03-03", "evidence": "3 March 2024"}

    fake = FakeJev().noul("launch date is 2024-03-03", p=0.9)
    ctx = context(fake, [st("s1", text)], {"s1": "zero_to_62_s"})
    run = ctx.schemas["Car"]
    run.categories["s1"] = ChoiceAnswer(
        choice="zero_to_62_s", confidence=0.6, probabilities={"zero_to_62_s": 0.6, "launched": 0.4}
    )
    ctx.extraction_llm = FakeLLM(answer)
    await FallbackStage().run(ctx)
    assert [e.kind for e in ctx.events] == ["llm_error"]
    assert "upstream 500" in ctx.events[0].message
    assert meta(ctx, "launched").value == date(2024, 3, 3)


# --- batching, budgets, scopes ------------------------------------------------------------


async def test_every_check_about_one_statement_is_one_request() -> None:
    text = "Five seats, and 0-62 in 9.1 s."

    def answer(prompt: str, _schema: type[BaseModel]) -> object:
        if "zero_to_62_s" in prompt:
            return {"stated": True, "value": 9.1, "evidence": "9.1 s"}
        return {"stated": True, "value": 5, "evidence": "Five seats"}

    fake = FakeJev(default_p=0.9)
    ctx = context(fake, [st("s1", text)], {})
    ctx.schemas["Car"].categories["s1"] = ChoiceAnswer(
        choice="seats", confidence=0.6, probabilities={"seats": 0.6, "zero_to_62_s": 0.4}
    )
    ctx.extraction_llm = FakeLLM(answer)
    await FallbackStage().run(ctx)
    [call] = fake.calls
    assert call.questions == {
        "Car.seats/verify": Noul(
            instructions="The statement states that the number of seats is 5."
        ),
        "Car.zero_to_62_s/verify": Noul(instructions=VERIFY),
    }
    assert (meta(ctx, "seats").value, meta(ctx).value) == (5, 9.1)


async def test_a_document_budget_stops_further_llm_calls() -> None:
    statements = [st("s1"), st("s2", "Zero to 62 takes 9.1 s.")]
    ctx = context(FakeJev(default_p=0.9), statements, {"s1": "zero_to_62_s", "s2": "zero_to_62_s"})
    ctx.budget = DocumentBudget(Budgets(per_document=DocBudget(max_llm_calls=1)))
    ctx.extraction_llm = model = llm(value=9.1, evidence="9.1")
    await FallbackStage().run(ctx)
    assert len(model.calls) == 1
    assert [e.limit for e in ctx.budget.events] == ["max_llm_calls"]
    assert meta(ctx).value == 9.1


async def test_a_statement_in_several_scopes_is_asked_once_and_recorded_in_each() -> None:
    fake = FakeJev().noul(VERIFY, p=0.9)
    ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"}, scopes=("SE", "SEL"))
    ctx.extraction_llm = model = llm(value=9.1, evidence="9.1")
    await FallbackStage().run(ctx)
    assert len(model.calls) == 1
    assert len(fake.calls) == 1
    assert meta(ctx, scope="SE").value == meta(ctx, scope="SEL").value == 9.1
    assert len(ctx.verified) == 1


async def test_the_most_certain_verified_statement_wins() -> None:
    statements = [st("s1", "Quick: 9.1 s."), st("s2", "0-62 takes 9.4 s.")]
    fake = FakeJev().noul("is 9.1", p=0.85).noul("is 9.4", p=0.97)
    ctx = context(fake, statements, {"s1": "zero_to_62_s", "s2": "zero_to_62_s"})

    def answer(prompt: str, _schema: type[BaseModel]) -> object:
        value = "9.1" if "Quick" in prompt else "9.4"
        return {"stated": True, "value": float(value), "evidence": f"{value} s"}

    ctx.extraction_llm = FakeLLM(answer)
    await FallbackStage().run(ctx)
    m = meta(ctx)
    assert (m.value, m.confidence) == (9.4, 0.97)
    assert m.alternatives == [Alternative(value=9.1, raw="9.1 s", p=0.85)]
    assert len(ctx.verified) == 2


async def test_list_fields_verify_each_item() -> None:
    text = "Choose from SE, SEL or the Turbo."
    fake = FakeJev().noul('"SE"', p=0.9).noul('"SEL"', p=0.85).noul('"GTI"', p=0.1)
    ctx = context(fake, [st("s1", text)], {"s1": "trims"})
    ctx.extraction_llm = llm(value=["SE", "SEL", "GTI", "SE"], evidence="SE, SEL")
    await FallbackStage().run(ctx)
    [call] = fake.calls
    assert call.questions["Car.trims/member0"] == Noul(
        instructions='Does the statement give "SE" as one of the trim names?'
    )
    assert len(call.questions) == 3  # duplicates are asked once
    m = meta(ctx, "trims")
    assert (m.value, m.confidence, m.verified) == (["SE", "SEL"], 0.85, True)
    assert m.alternatives == [Alternative(value="GTI", raw="SE, SEL", p=0.1)]
    assert [e.value for e in ctx.verified] == ["SE", "SEL"]


async def test_list_items_from_every_verified_statement_are_merged_in_order() -> None:
    statements = [
        st("s1", "Trims: SE and SEL."),
        st("s2", "Also available as the GTI."),
        st("s3", "Every trim, even the R."),
    ]
    fake = FakeJev(default_p=0.9).noul('"SEL"', p=0.8)
    ctx = context(fake, statements, {"s1": "trims", "s2": "trims", "s3": "trims"})
    ctx.schemas["Car"].scopes = [
        EntityScope(label="doc", component_ids=["c1"], shared_statement_ids=["s3"])
    ]

    def answer(prompt: str, _schema: type[BaseModel]) -> object:
        if "GTI" in prompt:
            return {"stated": True, "value": ["GTI"], "evidence": "GTI"}
        if "the R" in prompt:
            return {"stated": True, "value": ["R"], "evidence": "R"}
        return {"stated": True, "value": ["SE", "SEL"], "evidence": "SE and SEL"}

    ctx.extraction_llm = FakeLLM(answer)
    await FallbackStage().run(ctx)
    m = meta(ctx, "trims")
    # Own statements' items only (the shared R is left out), in document order.
    assert (m.value, m.confidence, m.shared) == (["SE", "SEL", "GTI"], 0.8, False)
    assert m.source is not None
    assert m.source.statement_id == "s2"  # the most certain contributing statement


async def test_an_llm_answer_keeps_the_alternatives_jev_weighed() -> None:
    fake = FakeJev().noul(VERIFY, p=0.9)
    ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"})
    picked(ctx, "s1", "zero_to_62_s", "62", 0.3)
    run = ctx.schemas["Car"]
    weighed = [Alternative(value="9.1", raw="9.1", p=0.25), Alternative(value=8.0, p=0.1)]
    run.set_field("doc", "zero_to_62_s", meta(ctx).model_copy(update={"alternatives": weighed}))
    ctx.extraction_llm = llm(value=9.1, evidence="9.1")
    await FallbackStage().run(ctx)
    assert meta(ctx).alternatives == [
        Alternative(value=62.0, raw="62", p=0.3),
        Alternative(value="9.1", raw="9.1", p=0.25),
        Alternative(value=8.0, p=0.1),
    ]


async def test_an_llm_answer_confirming_the_jev_value_doesnt_list_it_again() -> None:
    fake = FakeJev().noul(VERIFY, p=0.9)
    ctx = context(fake, [st("s1")], {"s1": "zero_to_62_s"})
    picked(ctx, "s1", "zero_to_62_s", "9.1", 0.3)
    ctx.extraction_llm = llm(value=9.1, evidence="9.1")
    await FallbackStage().run(ctx)
    m = meta(ctx)
    assert (m.value, m.confidence, m.alternatives) == (9.1, 0.9, [])


async def test_a_custom_extractor_is_used() -> None:
    @dataclass
    class Rules:
        async def extract(
            self, statement: Statement, field: FieldSpec, budget: DocumentBudget
        ) -> LLMAnswer | None:
            return LLMAnswer(value=9.1, evidence="9.1")

    assert isinstance(Rules(), LLMExtractor)
    ctx = context(FakeJev().noul(VERIFY, p=0.9), [st("s1")], {"s1": "zero_to_62_s"})
    await FallbackStage(extractor=Rules()).run(ctx)
    assert meta(ctx).method == "llm"


def test_output_models_type_the_value_for_structured_output() -> None:
    number = output_model(SPEC.field("zero_to_62_s"))
    assert issubclass(number, LLMOutput)
    assert number.model_json_schema()["properties"]["value"]["anyOf"][0] == {"type": "number"}
    trims = output_model(SPEC.field("trims"))
    schema = trims.model_json_schema()["properties"]["value"]["anyOf"][0]
    assert schema == {"items": {"type": "string"}, "type": "array"}
    assert output_model(SPEC.field("zero_to_62_s")) is number


async def test_the_prompt_can_be_overridden() -> None:
    model = FakeLLM([{"stated": True, "value": 9.1, "evidence": "9.1"}])
    extractor = LLMFieldExtractor(model, prompt="{name} ({type}{unit}): {section}{statement}")
    answer = await extractor.extract(st("s1"), SPEC.field("zero_to_62_s"), DocumentBudget())
    assert answer == LLMAnswer(value=9.1, evidence="9.1")
    assert model.calls[0].prompt == f"zero_to_62_s (a number in s): {TEXT}"
    assert extractor.prompt != PROMPT


# --- through the extractor ---------------------------------------------------------------


async def test_extraction_llm_turns_the_fallback_on_and_is_metered() -> None:
    @dataclass
    class Setup:
        name: str = "normalise"

        async def run(self, ctx: Context) -> None:
            ctx.parsed = ParsedDocument(
                document=ctx.document,
                root=Component(id="root", type="section", location=LOC),
                statements={"s1": st("s1")},
            )
            run = ctx.schemas["Car"]
            run.scopes = [EntityScope(label="doc", component_ids=["c1"])]
            run.categories["s1"] = ChoiceAnswer(
                choice="zero_to_62_s", confidence=0.9, probabilities={"zero_to_62_s": 0.9}
            )

    model = FakeLLM(
        lambda _p, _s: {"stated": True, "value": 9.1, "evidence": "9.1 seconds"},
        price=(1.0, 1.0),
    )
    extractor = Extractor(
        [Car],
        jev=FakeJev().noul(VERIFY, p=0.9).client(),
        pipeline=Pipeline([Setup(), FallbackStage()]),
        extraction_llm=model,
    )
    result = await extractor.extract(Document.from_bytes(b"<p/>"))
    item = result.one(Car)
    assert item.record.zero_to_62_s == 9.1
    assert item.meta.zero_to_62_s.method == "llm"
    assert result.meta.llm.calls == 1
    assert result.meta.llm.cost > 0


# --- vision values -------------------------------------------------------------------------


def seen(sid: str, text: str = "The 0-62 mph time is 9.1 s.") -> Statement:
    return st(sid, text).model_copy(update={"kind": "vision"})


def from_vision(
    ctx: Context, name: str, value: Any, items: list[VisionValue], *, confidence: float = 0.95
) -> None:
    """Leave what the normalise (or select) stage would have for a vision value."""
    run = ctx.schemas["Car"]
    statement = ctx.parsed.statements[items[0].statement_id] if ctx.parsed else None
    assert statement is not None
    for scope in run.scopes:
        run.set_field(
            scope.label,
            name,
            FieldMeta(
                value=value,
                confidence=confidence,
                method="vision",
                source=Source(statement_id=statement.id, statement=statement.text),
                alternatives=[Alternative(value=8.0, raw="8", p=0.02)],
            ),
        )
        run.vision_values[(scope.label, name)] = items


VISION_SPAN = Span(start=21, end=24)


async def test_a_verified_vision_value_stays_marked_verified_without_an_llm() -> None:
    fake = FakeJev(strict=True).noul(VERIFY, p=0.9, state="9.1 s")
    ctx = context(fake, [seen("v1")], {"v1": "zero_to_62_s"})
    from_vision(ctx, "zero_to_62_s", 9.1, [VisionValue("v1", 9.1, VISION_SPAN)])
    await FallbackStage().run(ctx)
    [call] = fake.calls
    assert call.questions == {"Car.zero_to_62_s/vision0": Noul(instructions=VERIFY)}
    m = meta(ctx)
    assert (m.value, m.method, m.verified, m.confidence) == (9.1, "vision", True, 0.9)
    [example] = ctx.verified
    assert (example.field, example.value, example.source) == ("Car.zero_to_62_s", 9.1, "vision")
    assert (example.statement, example.evidence, example.probability) == (
        "The 0-62 mph time is 9.1 s.",
        (21, 24),
        0.9,
    )
    assert example.context["kind"] == "vision"
    assert ctx.events == []


async def test_verification_never_raises_a_vision_values_confidence() -> None:
    fake = FakeJev().noul(VERIFY, p=0.99)
    ctx = context(fake, [seen("v1")], {"v1": "zero_to_62_s"})
    from_vision(ctx, "zero_to_62_s", 9.1, [VisionValue("v1", 9.1)], confidence=0.7)
    await FallbackStage().run(ctx)
    assert (meta(ctx).confidence, meta(ctx).verified) == (0.7, True)


async def test_a_rejected_vision_value_becomes_an_alternative_and_an_event() -> None:
    fake = FakeJev().noul(VERIFY, p=0.3)
    ctx = context(fake, [seen("v1")], {"v1": "zero_to_62_s"})
    from_vision(ctx, "zero_to_62_s", 9.1, [VisionValue("v1", 9.1, VISION_SPAN)])
    await FallbackStage().run(ctx)
    m = meta(ctx)
    assert (m.found, m.method, m.verified) == (False, None, None)
    assert m.alternatives == [
        Alternative(value=9.1, raw="9.1", p=0.3),
        Alternative(value=8.0, raw="8", p=0.02),
    ]
    assert ctx.verified == []
    [event] = ctx.events
    assert (event.stage, event.kind) == ("fallback", "vision_rejected")
    assert event.data == {
        "schema": "Car",
        "field": "zero_to_62_s",
        "statement_id": "v1",
        "value": 9.1,
        "p": 0.3,
    }


async def test_a_rejected_vision_value_leaves_the_field_to_the_llm_fallback() -> None:
    fake = FakeJev().noul(VERIFY, p=0.3, state="9.1 s.").noul(VERIFY, p=0.9, state="seconds")
    ctx = context(fake, [seen("v1"), st("s1")], {"v1": "zero_to_62_s", "s1": "zero_to_62_s"})
    from_vision(ctx, "zero_to_62_s", 9.1, [VisionValue("v1", 9.1)])
    ctx.extraction_llm = llm(value=9.1, evidence="9.1 seconds")
    await FallbackStage().run(ctx)
    m = meta(ctx)
    assert (m.value, m.method, m.verified) == (9.1, "llm", True)
    assert m.source is not None
    assert m.source.statement_id == "s1"


async def test_list_fields_verify_only_the_items_vision_alone_gave() -> None:
    fake = FakeJev().noul('"GTI"', p=0.9).noul('"R"', p=0.2)
    statements = [seen("v1", "Trims: SE, GTI and R."), st("s1", "Choose the SE.")]
    ctx = context(fake, statements, {"v1": "trims", "s1": "trims"})
    from_vision(
        ctx,
        "trims",
        ["SE", "GTI", "R"],
        [VisionValue("v1", "GTI", Span(start=11, end=14)), VisionValue("v1", "R")],
    )
    await FallbackStage().run(ctx)
    [call] = fake.calls
    assert call.questions == {
        "Car.trims/vision0": Noul(
            instructions='Does the statement give "GTI" as one of the trim names?'
        ),
        "Car.trims/vision1": Noul(
            instructions='Does the statement give "R" as one of the trim names?'
        ),
    }
    m = meta(ctx, "trims")
    assert (m.value, m.verified, m.confidence) == (["SE", "GTI"], True, 0.9)
    assert Alternative(value="R", raw=None, p=0.2) in m.alternatives
    assert [(e.value, e.evidence) for e in ctx.verified] == [("GTI", (11, 14))]


async def test_a_list_left_with_no_items_is_emptied() -> None:
    fake = FakeJev().noul('"GTI"', p=0.1)
    ctx = context(fake, [seen("v1", "Trims: GTI.")], {"v1": "trims"})
    from_vision(ctx, "trims", ["GTI"], [VisionValue("v1", "GTI")])
    await FallbackStage().run(ctx)
    assert not meta(ctx, "trims").found


async def test_a_bool_vision_value_is_verified_as_a_claim() -> None:
    claim = "The statement says it has an automatic gearbox."
    fake = FakeJev(strict=True).noul(claim, p=0.95)
    ctx = context(fake, [seen("v1", "Automatic gearbox.")], {"v1": "automatic"})
    from_vision(ctx, "automatic", True, [VisionValue("v1", True)])
    await FallbackStage().run(ctx)
    assert [q.instructions for q in fake.questions] == [claim]
    assert meta(ctx, "automatic").verified is True


async def test_a_vision_value_is_asked_about_once_for_every_scope_sharing_it() -> None:
    fake = FakeJev().noul(VERIFY, p=0.9)
    ctx = context(fake, [seen("v1")], {"v1": "zero_to_62_s"}, scopes=("a", "b"))
    from_vision(ctx, "zero_to_62_s", 9.1, [VisionValue("v1", 9.1)])
    await FallbackStage().run(ctx)
    [call] = fake.calls
    assert len(call.questions) == 1
    assert meta(ctx, scope="a").verified is True
    assert meta(ctx, scope="b").verified is True
    assert len(ctx.verified) == 1


async def test_vision_values_that_no_longer_stand_are_left_alone() -> None:
    fake = FakeJev(strict=True)
    ctx = context(fake, [seen("v1")], {"v1": "zero_to_62_s"})
    run = ctx.schemas["Car"]
    # Embedded data won a merge: its value stands, not the vision one.
    run.set_field("doc", "zero_to_62_s", FieldMeta(value=9.1, method="structured"))
    run.vision_values[("doc", "zero_to_62_s")] = [VisionValue("v1", 9.1)]
    # A list the vision item was taken out of.
    run.set_field("doc", "trims", FieldMeta(value=["SE"], method="generator"))
    run.vision_values[("doc", "trims")] = [VisionValue("v1", "GTI")]
    await FallbackStage().run(ctx)
    assert fake.calls == []
    assert meta(ctx).verified is None


async def test_a_custom_verify_threshold_applies_to_vision_values() -> None:
    fake = FakeJev().noul(VERIFY, p=0.6)
    ctx = context(fake, [seen("v1")], {"v1": "zero_to_62_s"})
    from_vision(ctx, "zero_to_62_s", 9.1, [VisionValue("v1", 9.1)])
    await FallbackStage(verify_threshold=0.5).run(ctx)
    assert meta(ctx).verified is True


async def test_an_item_one_vision_statement_verifies_stays_though_another_didnt() -> None:
    fake = FakeJev().noul('"GTI"', p=0.9, state="Trims: GTI.").noul('"GTI"', p=0.2, state="badge")
    statements = [seen("v1", "Trims: GTI."), seen("v2", "The badge reads GTI.")]
    ctx = context(fake, statements, {"v1": "trims", "v2": "trims"})
    from_vision(ctx, "trims", ["GTI"], [VisionValue("v1", "GTI"), VisionValue("v2", "GTI")])
    await FallbackStage().run(ctx)
    m = meta(ctx, "trims")
    assert (m.value, m.verified, m.confidence) == (["GTI"], True, 0.9)
    assert all(a.value != "GTI" for a in m.alternatives)
    assert [(e.value, e.statement) for e in ctx.verified] == [("GTI", "Trims: GTI.")]
    assert [e.data["statement_id"] for e in ctx.events] == ["v2"]

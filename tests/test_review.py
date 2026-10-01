from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel

from jevex import (
    Budgets,
    Context,
    Document,
    DomLocation,
    Extractor,
    Field,
    FieldMeta,
    Pipeline,
    ReviewItem,
    ReviewQueue,
    ReviewSink,
    RunBudget,
    SchemaSpec,
    Source,
    Span,
    Statement,
    VerifiedExample,
    example_id,
    review_items,
)
from jevex.entities import EntityScope
from jevex.interfaces import ParsedDocument
from jevex.jev import Choice
from jevex.layout import Component
from jevex.learn import LearnStage
from jevex.results import build_extracted
from jevex.review import REVIEW_THRESHOLD
from jevex.store import open_store
from jevex.testing import FakeJev, FakeLLM

LOC = DomLocation(dom_path="/p")
URL = "https://example.com/car"
TEXT = "62 mph takes 9.1 seconds"
SPAN = Span(start=13, end=16)


class Car(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    launched: date = Field(description="Launch date")
    doors: int = Field(description="Number of doors")
    colours: list[str] = Field(default_factory=list, description="Paint colours")


class Trim(BaseModel):
    power_ps: int = Field(description="Power", unit="PS")


class Range(BaseModel):
    name: str = Field(description="Range name")
    trims: list[Trim] = Field(default_factory=list, description="Trims")


SPEC = SchemaSpec.from_model(Car)
STATEMENT = Statement(
    id="s1",
    text=TEXT,
    kind="list_item",
    component_id="c1",
    heading_trail=["Performance"],
    location=LOC,
)


def meta(value: Any, confidence: float | None, *, span: Span | None = SPAN) -> FieldMeta:
    return FieldMeta(
        value=value,
        confidence=confidence,
        method="jev",
        source=Source(url=URL, component_id="c1", statement_id="s1", statement=TEXT, span=span),
    )


def item(value: Any = 9.1, confidence: float = 0.6, **kw: Any) -> ReviewItem:
    [found] = review_items(
        [build_extracted(SPEC, "doc", {"zero_to_62_s": meta(value, confidence, **kw)})],
        statements={"s1": STATEMENT},
        url=URL,
    )
    return found


# --- review_items --------------------------------------------------------------------------


def test_only_found_values_below_the_threshold_are_sent() -> None:
    record = build_extracted(
        SPEC,
        "doc",
        {
            "zero_to_62_s": meta(9.1, 0.6),
            "launched": meta(date(2024, 1, 1), REVIEW_THRESHOLD),  # at the threshold: kept
            "doors": FieldMeta(value=5, method="structured"),  # no confidence: not a question
        },
    )
    unfound = build_extracted(SPEC, "other", {"doors": FieldMeta(confidence=0.1)})
    [found] = review_items([record, unfound], statements={"s1": STATEMENT}, url=URL)
    assert found.field == "Car.zero_to_62_s"
    assert found.entity == "doc"
    assert found.url == URL
    assert found.meta == meta(9.1, 0.6)
    assert found.threshold == REVIEW_THRESHOLD
    assert found.context == {"heading_trail": ["Performance"], "kind": "list_item"}
    assert found.id.startswith("rv-")


def test_per_field_thresholds_beat_the_default() -> None:
    record = build_extracted(
        SPEC, "doc", {"zero_to_62_s": meta(9.1, 0.6), "launched": meta(date(2024, 1, 1), 0.6)}
    )
    got = review_items(
        [record], threshold=0.5, thresholds={"launched": 0.7, "Car.zero_to_62_s": 0.4}
    )
    assert [(i.field, i.threshold) for i in got] == [("Car.launched", 0.7)]


def test_a_value_filtered_out_of_its_record_is_still_sent() -> None:
    record = build_extracted(SPEC, "doc", {"zero_to_62_s": meta(9.1, 0.6)}, threshold=0.7)
    [found] = review_items([record])
    assert found.meta.filtered
    assert found.meta.value == 9.1


def test_child_records_values_are_sent_with_their_nested_field() -> None:
    parent = SchemaSpec.from_model(Range)
    trims = parent.child("trims")
    kids = [
        build_extracted(trims, "GT", {"power_ps": meta(150, 0.3)}),
        build_extracted(trims, "SE", {"power_ps": meta(110, 0.9)}),
    ]
    record = build_extracted(parent, "doc", {"name": meta("Golf", 0.95)}, children={"trims": kids})
    got = review_items(
        [record], thresholds={"Range.trims.power_ps": 0.5}, document_source="example.com"
    )
    assert [(i.field, i.entity, i.threshold) for i in got] == [("Range.trims.power_ps", "GT", 0.5)]
    assert got[0].document_source == "example.com"


def test_an_item_without_a_known_statement_has_no_context() -> None:
    record = build_extracted(SPEC, "doc", {"zero_to_62_s": meta(9.1, 0.6)})
    [found] = review_items([record])
    assert found.context == {}
    assert found.url is None
    assert found.document_source is None
    assert found.example(9.1).document_source is None


def test_item_ids_are_stable_and_tell_entities_apart() -> None:
    def ids(entity: str) -> list[str]:
        record = build_extracted(SPEC, entity, {"zero_to_62_s": meta(9.1, 0.6)})
        return [i.id for i in review_items([record], url=URL)]

    assert ids("A") == ids("A")
    assert ids("A") != ids("B")


def test_item_ids_tell_statements_apart_without_a_url() -> None:
    def ids(text: str) -> list[str]:
        m = meta(9.1, 0.6).model_copy(
            update={"source": Source(statement_id="s1", statement=text, span=SPAN)}
        )
        return [i.id for i in review_items([build_extracted(SPEC, "doc", {"zero_to_62_s": m})])]

    assert ids(TEXT) != ids("0-62 mph: 9.1 s")


def test_review_queue_is_a_review_sink() -> None:
    assert isinstance(ReviewQueue(), ReviewSink)


# --- ReviewItem.example --------------------------------------------------------------------


def test_confirming_the_value_reuses_its_span_and_the_fallbacks_example_id() -> None:
    ex = item().example(9.1)
    assert ex == VerifiedExample(
        id=example_id("Car.zero_to_62_s", TEXT, 9.1),
        field="Car.zero_to_62_s",
        statement=TEXT,
        value=9.1,
        evidence=(13, 16),
        context={"heading_trail": ["Performance"], "kind": "list_item"},
        source="human",
        created_at=ex.created_at,
    )


def test_items_and_their_examples_keep_the_documents_locale() -> None:
    record = build_extracted(SPEC, "doc", {"zero_to_62_s": meta(9.1, 0.6)})
    [found] = review_items([record], statements={"s1": STATEMENT}, locale="de-DE")
    context = {"heading_trail": ["Performance"], "kind": "list_item", "locale": "de-DE"}
    assert found.context == context
    assert found.example(9.1).locale == "de-DE"
    [unknown] = review_items([record], statements={"s1": STATEMENT})
    assert "locale" not in unknown.context
    assert unknown.example(9.1).locale is None


def test_the_example_keeps_the_documents_source() -> None:
    found = item().model_copy(update={"document_source": "cars.example.com"})
    assert found.example(9.1).document_source == "cars.example.com"


def test_a_correction_has_only_the_evidence_given() -> None:
    wrong = item(62.0, span=Span(start=0, end=2))
    assert wrong.example(9.1).evidence is None
    assert wrong.example(9.1, evidence=(13, 16)).evidence == (13, 16)


@pytest.mark.parametrize(
    ("value", "evidence", "message"),
    [
        (None, None, "needs a value"),
        (9.1, (13, 13), "isn't a span"),
        (9.1, (-1, 3), "isn't a span"),
        (9.1, (13, len(TEXT) + 1), "isn't a span"),
    ],
)
def test_bad_answers_are_refused(
    value: Any, evidence: tuple[int, int] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        item().example(value, evidence=evidence)


def test_a_value_without_a_statement_cant_become_an_example() -> None:
    record = build_extracted(SPEC, "doc", {"zero_to_62_s": FieldMeta(value=9.1, confidence=0.2)})
    [found] = review_items([record])
    with pytest.raises(ValueError, match="no source statement"):
        found.example(9.1)


# --- the extractor ---------------------------------------------------------------------------


@dataclass
class Found:
    """Records what earlier stages would have found, as given."""

    metas: dict[str, FieldMeta]
    name: str = "normalise"

    async def run(self, ctx: Context) -> None:
        ctx.parsed = ParsedDocument(
            document=ctx.document,
            root=Component(id="root", type="section", location=LOC),
            statements={"s1": STATEMENT},
        )
        run = ctx.schemas["Car"]
        run.scopes = [EntityScope(label="doc", component_ids=["c1"])]
        for name, m in self.metas.items():
            run.set_field("doc", name, m)


class Failing:
    async def send(self, items: Any) -> None:
        raise RuntimeError("review queue is down")


def extractor(metas: dict[str, FieldMeta], **kw: Any) -> Extractor:
    return Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Found(metas)]), **kw)


DOC = Document.from_bytes(b"<p/>", url=URL)


async def test_extract_sends_each_documents_uncertain_values_to_the_sink() -> None:
    queue = ReviewQueue()
    metas = {"zero_to_62_s": meta(9.1, 0.6), "doors": meta(5, 0.95)}
    async with extractor(metas, review_sink=queue, threshold=0.7) as ex:
        result = await ex.extract(DOC)
        assert result.one(Car).record.zero_to_62_s is None  # the record threshold applies
        await ex.extract(DOC)
    first, second = queue.items
    assert first == second
    assert first.field == "Car.zero_to_62_s"
    assert first.context == {"heading_trail": ["Performance"], "kind": "list_item"}
    assert first.url == URL
    assert first.document_source == "example.com"


async def test_a_documents_site_is_the_items_and_the_feedbacks_source() -> None:
    queue = ReviewQueue()
    store = open_store(":memory:")
    doc = Document.from_bytes(b"<p/>", url=URL, site="Example-Cars.co.uk")
    async with extractor({"zero_to_62_s": meta(9.1, 0.6)}, review_sink=queue, store=store) as ex:
        await ex.extract(doc)
        [review] = queue.items
        example = await ex.feedback(review, 9.1)
    assert review.document_source == "example-cars.co.uk"
    assert example.document_source == "example-cars.co.uk"
    assert await store.examples("Car.zero_to_62_s") == [example]


async def test_the_items_have_the_documents_own_locale() -> None:
    queue = ReviewQueue()
    store = open_store(":memory:")
    doc = Document.from_bytes(b'<html lang="de-DE"><p/></html>', url=URL)
    async with extractor({"zero_to_62_s": meta(9.1, 0.6)}, review_sink=queue, store=store) as ex:
        await ex.extract(doc)
        await ex.extract(DOC)
        german, unknown = queue.items
        example = await ex.feedback(german, 9.1)
    assert german.context["locale"] == "de-DE"
    assert "locale" not in unknown.context
    assert example.locale == "de-DE"
    assert [e.locale for e in await store.examples("Car.zero_to_62_s")] == ["de-DE"]


async def test_a_document_with_nothing_uncertain_isnt_sent() -> None:
    calls: list[int] = []

    class Counting:
        async def send(self, items: Any) -> None:
            calls.append(len(items))

    metas = {"zero_to_62_s": meta(9.1, 0.6)}
    async with extractor(metas, review_sink=Counting(), review_threshold=0.5) as ex:
        await ex.extract(DOC)
    assert calls == []


async def test_review_thresholds_are_per_field() -> None:
    queue = ReviewQueue()
    metas = {"zero_to_62_s": meta(9.1, 0.6), "doors": meta(5, 0.95)}
    async with extractor(
        metas, review_sink=queue, review_thresholds={"Car.doors": 0.99, "zero_to_62_s": 0.5}
    ) as ex:
        await ex.extract(DOC)
    assert [i.field for i in queue.items] == ["Car.doors"]


async def test_a_failing_sink_fails_the_extraction() -> None:
    async with extractor({"zero_to_62_s": meta(9.1, 0.6)}, review_sink=Failing()) as ex:
        with pytest.raises(RuntimeError, match="review queue is down"):
            await ex.extract(DOC)


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"review_threshold": 1.5}, "review_threshold must be between 0 and 1"),
        ({"review_threshold": -0.1}, "review_threshold must be between 0 and 1"),
        ({"review_thresholds": {"Car.colour": 0.5}}, r"unknown fields: \['Car.colour'\]"),
    ],
)
def test_review_settings_are_checked(kw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Extractor([Car], **kw)


def test_review_thresholds_accept_nested_fields() -> None:
    Extractor([Range], review_thresholds={"Range.trims.power_ps": 0.5})


async def test_feedback_stores_a_human_example_in_place_of_the_llm_one() -> None:
    store = open_store(":memory:")
    llm_example = VerifiedExample(
        id=example_id("Car.zero_to_62_s", TEXT, 9.1),
        field="Car.zero_to_62_s",
        statement=TEXT,
        value=9.1,
        probability=0.85,
    )
    await store.add_example(llm_example)
    async with extractor({}, store=store) as ex:
        got = await ex.feedback(item(), 9.1)
    assert got.source == "human"
    assert await store.examples("Car.zero_to_62_s") == [got]


async def test_feedback_normalises_the_value_like_the_fallback() -> None:
    store = open_store(":memory:")
    async with extractor({}, store=store) as ex:
        got = await ex.feedback(item(), "9.1")
    assert got.value == 9.1
    assert got.id == example_id("Car.zero_to_62_s", TEXT, 9.1)
    assert got.evidence == (13, 16)


async def test_feedback_that_doesnt_fit_the_field_is_refused() -> None:
    store = open_store(":memory:")
    async with extractor({}, store=store) as ex:
        with pytest.raises(ValueError, match="doesn't fit zero_to_62_s"):
            await ex.feedback(item(), "fast")
    assert await store.examples() == []


async def test_list_fields_take_one_item_per_feedback() -> None:
    text = "Comes in red and blue."
    m = FieldMeta(
        value=["red", "green"],
        confidence=0.5,
        source=Source(statement_id="s1", statement=text, span=Span(start=9, end=12)),
    )
    [review] = review_items([build_extracted(SPEC, "doc", {"colours": m})])
    store = open_store(":memory:")
    async with extractor({}, store=store) as ex:
        with pytest.raises(ValueError, match="give one item per feedback call"):
            await ex.feedback(review, ["red", "blue"])
        red = await ex.feedback(review, "red")
        blue = await ex.feedback(review, "blue", evidence=(17, 21))
    assert (red.value, red.evidence) == ("red", None)  # the span is one item's, unknown which
    assert (blue.value, blue.evidence) == ("blue", (17, 21))
    assert red.id == example_id("Car.colours", text, "red")
    assert {e.id for e in await store.examples("Car.colours")} == {red.id, blue.id}


async def test_feedback_in_compile_mode_is_stored_for_jevex_learn() -> None:
    store = open_store(":memory:")
    async with extractor({}, store=store, learn_mode="compile", generator_llm=FakeLLM([])) as ex:
        got = await ex.feedback(item(), 9.1)
    assert await store.examples("Car.zero_to_62_s") == [got]


async def test_feedback_without_a_store_is_refused() -> None:
    async with extractor({}) as ex:
        with pytest.raises(ValueError, match="pass store="):
            await ex.feedback(item(), 9.1)


async def test_feedback_isnt_kept_in_a_store_the_extractor_opened_for_itself() -> None:
    budgets = Budgets(run=RunBudget(max_spend=1.0))
    async with extractor({}, budgets=budgets) as ex:
        with pytest.raises(ValueError, match="pass store="):
            await ex.feedback(item(), 9.1)


async def test_feedback_for_another_extractors_field_is_refused() -> None:
    other = item().model_copy(update={"field": "Bike.zero_to_62_s"})
    async with extractor({}, store=open_store(":memory:")) as ex:
        with pytest.raises(ValueError, match=r"Bike\.zero_to_62_s isn.t a field"):
            await ex.feedback(other, 9.1)


DRAFT = {"regex": r"(\d+(?:\.\d+)?) seconds", "group": 1, "normalise": ["parse_number"]}


def pick(value: str) -> Any:
    def answer(q: Choice) -> str:
        return value if value in q.options else "none"

    return answer


async def test_the_learner_learns_from_feedback_and_the_review_round_trip() -> None:
    queue = ReviewQueue()
    fake = FakeJev().choice(None, pick("9.1"))
    generator_llm = FakeLLM([DRAFT])
    async with Extractor(
        [Car],
        jev=fake.client(),
        pipeline=Pipeline([Found({"zero_to_62_s": meta(9.1, 0.4)}), LearnStage()]),
        generator_llm=generator_llm,
        review_sink=queue,
    ) as ex:
        await ex.extract(DOC)
        [review] = queue.items
        example = await ex.feedback(review, 9.1)
        await ex.wait_for_learning()
        learner = await ex.learner()
        assert learner is not None
        [outcome] = learner.outcomes
        store = await ex.store()
        assert store is not None
        stored = await store.examples("Car.zero_to_62_s")
    assert outcome.example_id == example.id
    assert outcome.status == "accepted"
    assert stored == [example]
    assert len(generator_llm.calls) == 1

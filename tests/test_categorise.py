from collections.abc import Mapping
from decimal import Decimal

import pytest
from pydantic import BaseModel

from jevex import (
    CategoriseStage,
    Component,
    Context,
    Document,
    DomLocation,
    Extractor,
    Field,
    JevStatementClassifier,
    Questions,
    SchemaSpec,
    Statement,
)
from jevex.entities import EntityScope
from jevex.extractor import default_pipeline
from jevex.interfaces import ParsedDocument, StatementClassifier
from jevex.jev import (
    Choice,
    ChoiceAnswer,
    JevClient,
    JevResponse,
    JSONContent,
    NoulAnswer,
    Question,
    UnexpectedAnswerError,
)
from jevex.layout import LayoutStage
from jevex.split import StatementStage
from jevex.testing import FakeJev

LOC = DomLocation(dom_path="/p")


class Car(BaseModel):
    """A car's specification."""

    price: Decimal = Field(description="Price", unit="GBP")
    power_kw: float = Field(description="Engine power", unit="kW", group="performance")
    zero_to_62_s: float = Field(
        description="0-62 mph time",
        unit="s",
        group="performance",
        questions=Questions(categorise="How long it takes to reach 62 mph"),
    )


def st(sid: str, text: str, component: str = "c1", kind: str = "sentence") -> Statement:
    return Statement.model_validate(
        {"id": sid, "text": text, "kind": kind, "component_id": component, "location": LOC}
    )


def context(fake: FakeJev, statements: list[Statement], components: list[str]) -> Context:
    ctx = Context.create(Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(Car)], fake.client())
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={s.id: s for s in statements},
    )
    ctx.schemas["Car"].scopes = [EntityScope(label="doc", component_ids=components)]
    return ctx


def options(fake: FakeJev) -> list[list[str]]:
    out: list[list[str]] = []
    for call in fake.calls:
        [q] = call.questions.values()
        assert isinstance(q, Choice)
        out.append(list(q.options))
    return out


def test_jev_statement_classifier_is_a_statement_classifier() -> None:
    assert isinstance(JevStatementClassifier(), StatementClassifier)


async def test_each_statement_is_one_choice_with_its_distribution_kept() -> None:
    fake = FakeJev().choice("Which detail", "price", confidence=0.8, state="24,995")
    ctx = context(fake, [st("s1", "From £24,995."), st("s2", "A lovely car.")], ["c1"])
    await CategoriseStage().run(ctx)
    cats = ctx.schemas["Car"].categories
    assert cats["s1"].choice == "price"
    assert cats["s1"].confidence == 0.8
    assert cats["s1"].probabilities["price"] == 0.8
    assert cats["s2"].choice == "none"  # "none" answers are kept too
    assert len(fake.calls) == 2
    assert fake.calls[0].state == {"statement": "From £24,995."}


class Engine(BaseModel):
    size_cc: int = Field(description="Engine size", unit="cc")


class CarWithEngine(BaseModel):
    """A car."""

    price: Decimal = Field(description="Price", unit="GBP")
    engine: Engine = Field(description="The engine")


class Book(BaseModel):
    """A book."""

    title: str = Field(description="Title")


async def test_every_schemas_choice_about_a_statement_goes_in_one_request() -> None:
    fake = FakeJev().choice(
        "Which detail", lambda q: "price" if "price" in q.options else "none", state="24,995"
    )
    ctx = Context.create(
        Document.from_bytes(b"<p/>"),
        [SchemaSpec.from_model(Car), SchemaSpec.from_model(Book)],
        fake.client(),
    )
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={"s1": st("s1", "From £24,995")},
    )
    for run in ctx.schemas.values():
        run.scopes = [EntityScope(label="doc", component_ids=["c1"])]
    await CategoriseStage().run(ctx)
    [call] = fake.calls
    assert set(call.questions) == {"Car", "Book"}
    assert ctx.schemas["Car"].categories["s1"].choice == "price"
    assert ctx.schemas["Book"].categories["s1"].choice == "none"


async def test_statements_with_only_nested_model_fields_arent_asked() -> None:
    fake = FakeJev()
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(CarWithEngine)], fake.client()
    )
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={"s1": st("s1", "1,498 cc", "c2"), "s2": st("s2", "£24,995", "c3")},
    )
    run = ctx.schemas["CarWithEngine"]
    run.scopes = [EntityScope(label="doc", component_ids=["c2", "c3"])]
    run.component_ids = {"engine": ["c2"], "price": ["c3"]}
    await CategoriseStage().run(ctx)
    assert [c.state for c in fake.calls] == [{"statement": "£24,995"}]


async def test_a_non_choice_answer_is_an_error() -> None:
    class NoulForEverything:
        async def system_one(
            self, state: JSONContent, questions: Mapping[str, Question]
        ) -> JevResponse:
            return JevResponse(
                answers={k: NoulAnswer(p=0.9) for k in questions}, input_tokens=1, model="fake"
            )

    ctx = context(FakeJev(), [st("s1", "From £24,995")], ["c1"])
    ctx.jev = JevClient(NoulForEverything())
    with pytest.raises(UnexpectedAnswerError, match="Choice"):
        await CategoriseStage().run(ctx)


async def test_gated_components_limit_the_options_and_skip_irrelevant_statements() -> None:
    fake = FakeJev()
    ctx = context(
        fake,
        [
            st("s1", "0-62 mph in 9.1 s", "perf"),
            st("s2", "From £24,995", "price"),
            st("s3", "x", "c9"),
        ],
        ["perf", "price", "c9"],
    )
    ctx.schemas["Car"].component_ids = {"price": ["price"], "performance": ["perf"]}
    await CategoriseStage().run(ctx)
    assert options(fake) == [["power_kw", "zero_to_62_s", "none"], ["price", "none"]]
    assert set(ctx.schemas["Car"].categories) == {"s1", "s2"}  # s3 passed for no group


async def test_only_in_scope_unstructured_uncategorised_statements_are_asked_once() -> None:
    fake = FakeJev()
    statements = [
        st("s1", "in both scopes", "c1"),
        st("s2", "out of scope", "c2"),
        st("s3", "price: 24995", "c1", kind="structured"),
        st("s4", "already done", "c1"),
    ]
    ctx = context(fake, statements, ["c1"])
    run = ctx.schemas["Car"]
    run.scopes.append(EntityScope(label="other", component_ids=["c1"]))
    run.categories["s4"] = ChoiceAnswer(
        choice="price", confidence=1.0, probabilities={"price": 1.0}
    )
    await CategoriseStage().run(ctx)
    assert [c.state for c in fake.calls] == [{"statement": "in both scopes"}]
    assert run.categories["s4"].choice == "price"


async def test_stage_does_nothing_without_a_parsed_document() -> None:
    fake = FakeJev()
    ctx = context(fake, [st("s1", "x")], ["c1"])
    ctx.parsed = None
    await CategoriseStage().run(ctx)
    assert fake.calls == []


def test_categorise_is_a_default_stage_between_statements_and_candidates() -> None:
    names = [s.name for s in default_pipeline().stages]
    assert names.index("statements") < names.index("categorise") < names.index("candidates")


# --- end to end: HTML in, typed values out -------------------------------------------

PAGE = b"""<!doctype html><html><head><title>Delmaro Kestrova</title></head><body>
<nav><a href="/">Home</a> <a href="/cars">Cars</a></nav>
<main>
  <h1>Delmaro Kestrova 1.5 SE</h1>
  <p>A roomy family hatchback. It is on sale now.</p>
  <section><h2>Performance</h2>
    <dl><dt>Power</dt><dd>150 PS</dd><dt>0-62 mph</dt><dd>9.1 s</dd></dl>
  </section>
  <section><h2>Price</h2><p>On the road from &pound;24,995.</p></section>
  <aside><h2>Newsletter</h2><p>Sign up for our weekly deals.</p></aside>
</main>
<footer>&copy; 2026 Example Motors</footer>
</body></html>"""


class Trim(BaseModel):
    """A car trim."""

    trim: str = Field(description="Trim name")


async def test_a_section_s_name_is_categorised_beside_its_siblings_names() -> None:
    fake = FakeJev(strict=True).choice("Which detail", "none")
    page = (
        b"<section><h2>SE</h2><p>150PS.</p></section><section><h2>Sport</h2><p>180PS.</p></section>"
    )
    ctx = Context.create(
        Document.from_bytes(page, content_type="text/html"),
        [SchemaSpec.from_model(Trim)],
        fake.client(),
    )
    await LayoutStage().run(ctx)
    await StatementStage().run(ctx)
    assert ctx.parsed is not None
    every = [c.id for c in ctx.parsed.root.walk()]
    ctx.schemas["Trim"].scopes = [EntityScope(label="doc", component_ids=every)]
    await CategoriseStage().run(ctx)
    assert [c.state for c in fake.calls] == [
        {"statement": "SE", "sibling_labels": "SE, Sport"},
        {"statement": "150PS.", "section": "SE"},
        {"statement": "Sport", "sibling_labels": "SE, Sport"},
        {"statement": "180PS.", "section": "Sport"},
    ]


def pick_first_candidate(q: Choice) -> str:
    return next(o for o in q.options if o != "none")


async def test_html_to_values_through_the_default_pipeline() -> None:
    fake = (
        FakeJev(strict=True)
        .noul("Does this document include", p=0.95)
        .noul("Does this section contain", p=0.1)
        .noul("contain the price", p=0.9, state="24,995")
        .noul("engine power", p=0.9, state="Power: 150 PS")
        .choice("Which detail", "none")
        .choice("Which detail", "power_kw", confidence=0.9, state="Power: 150 PS")
        .choice("Which detail", "zero_to_62_s", confidence=0.9, state="0-62 mph: 9.1 s")
        .choice("Which detail", "price", confidence=0.9, state="24,995")
        .choice("Which of these", pick_first_candidate, confidence=0.9)
        # Options here are "0", "0-62 mph", "62 mph", "9.1 s", "none".
        .choice("Which of these", "9.1 s", confidence=0.9, state="0-62 mph: 9.1 s")
    )
    async with Extractor([Car], jev=fake.client()) as ex:
        result = await ex.extract(Document.from_bytes(PAGE, url="https://cars.test/kestrova"))

    car = result.one(Car)
    assert car.strict() == Car(price=Decimal("24995"), power_kw=110.324812, zero_to_62_s=9.1)
    assert result.values["Car"]["document"] == {
        "price": Decimal("24995"),
        "power_kw": 110.324812,  # 150 PS
        "zero_to_62_s": 9.1,
    }
    categorised = [
        (c.state["statement"], list(q.options))
        for c in fake.calls
        if isinstance(c.state, dict)
        for q in c.questions.values()
        if isinstance(q, Choice) and "detail" in q.instructions
    ]
    perf = ["power_kw", "zero_to_62_s", "none"]
    # Only statements in sections that passed the component gate are categorised, over
    # only the fields that section passed for. The intro and newsletter never are.
    assert categorised == [
        ("Performance", perf),
        ("Power: 150 PS", perf),
        ("0-62 mph: 9.1 s", perf),
        ("Price", ["price", "none"]),
        ("On the road from £24,995.", ["price", "none"]),
    ]
    assert result.meta.jev.requests == len(fake.calls) == 13

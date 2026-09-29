from decimal import Decimal

import pytest
from pydantic import BaseModel

from jevex import (
    BoilerplateCleaner,
    Component,
    ComponentGateStage,
    Context,
    Document,
    DomLocation,
    EntityStage,
    Field,
    NoulComponentGate,
    Questions,
    SchemaSpec,
    gate_units,
)
from jevex.extractor import default_pipeline
from jevex.interfaces import ComponentGate, ParsedDocument
from jevex.layout_html import HtmlLayoutParser
from jevex.testing import FakeJev


class Car(BaseModel):
    """A car's specification."""

    price: Decimal = Field(description="Price", unit="GBP")
    power_kw: float = Field(description="Engine power", unit="kW", group="performance")
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s", group="performance")


class Book(BaseModel):
    """A book."""

    title: str = Field(
        description="Title", questions=Questions(component_gate="Is this the book's title?")
    )


def comp(
    type_: str,
    text: str = "",
    cid: str = "c",
    *children: Component,
    trail: list[str] | None = None,
) -> Component:
    return Component.model_validate(
        {
            "id": cid,
            "type": type_,
            "text": text,
            "children": list(children),
            "heading_trail": trail or [],
            "location": DomLocation(dom_path=f"/{cid}"),
        }
    )


def page() -> Component:
    return comp(
        "section",
        "",
        "root",
        comp("heading", "Delmaro Kestrova", "h1"),
        comp("paragraph", "A roomy family hatchback.", "p1"),
        comp(
            "section",
            "",
            "s-price",
            comp("heading", "Pricing", "h2", trail=["Delmaro Kestrova"]),
            comp("paragraph", "From £24,995 on the road.", "p2", trail=["Pricing"]),
        ),
        comp(
            "section",
            "",
            "s-perf",
            comp("heading", "Performance", "h3", trail=["Delmaro Kestrova"]),
            comp(
                "list",
                "",
                "l1",
                comp("list_item", "Power: 110 kW", "li1", trail=["Performance"]),
                comp("list_item", "0-62 mph: 9.1 s", "li2", trail=["Performance"]),
                trail=["Performance"],
            ),
        ),
        comp("table", "Trim | Price\nSE | £24,995", "t1"),
        comp("paragraph", "Book a test drive today.", "p3"),
    )


# --- units ---------------------------------------------------------------------------


def test_units_are_each_containers_blocks_plus_tables() -> None:
    units = gate_units(page())
    assert [(u.id, u.component_ids) for u in units] == [
        ("root#0", ("h1", "p1")),
        ("s-price#0", ("h2", "p2")),
        ("s-perf#0", ("h3", "l1", "li1", "li2")),
        ("t1", ("t1",)),
        ("root#1", ("p3",)),
    ]
    perf = units[2]
    assert perf.text == "Performance\nPower: 110 kW\n0-62 mph: 9.1 s"
    assert perf.state() == {"content": perf.text, "section": "Delmaro Kestrova"}
    assert units[0].state() == {"content": "Delmaro Kestrova\nA roomy family hatchback."}


def test_long_runs_are_chunked_and_texts_capped() -> None:
    paragraphs = [comp("paragraph", "x" * 40, f"p{i}") for i in range(5)]
    units = gate_units(comp("section", "", "root", *paragraphs), max_chars=100)
    assert [u.component_ids for u in units] == [("p0", "p1"), ("p2", "p3"), ("p4",)]
    big = gate_units(comp("section", "", "r", comp("paragraph", "y" * 500, "p")), max_chars=100)
    assert len(big[0].text) == 100


def test_empty_blocks_make_no_unit_and_a_bare_root_is_one_unit() -> None:
    assert gate_units(comp("section", "", "r", comp("paragraph", "  ", "p"))) == []
    [unit] = gate_units(comp("paragraph", "Just text.", "p"))
    assert unit.component_ids == ("p",)


# --- the gate ------------------------------------------------------------------------


def parsed(root: Component) -> ParsedDocument:
    return ParsedDocument(document=Document.from_bytes(b"<p/>"), root=root)


def test_noul_component_gate_is_a_component_gate() -> None:
    assert isinstance(NoulComponentGate(), ComponentGate)


async def test_groups_pass_the_units_jev_says_contain_them() -> None:
    fake = (
        FakeJev()
        .noul("contain the price (GBP)?", p=0.9, state="24,995")
        .noul("engine power (kW) or 0-62", p=0.8, state="Power: 110 kW")
    )
    result = await NoulComponentGate().gate(
        parsed(page()), [SchemaSpec.from_model(Car)], fake.client()
    )
    groups = result["Car"]
    # Passing units bring their ancestors, in reading order.
    assert groups["price"] == ["root", "s-price", "h2", "p2", "t1"]
    assert groups["performance"] == ["root", "s-perf", "h3", "l1", "li1", "li2"]


async def test_all_schemas_questions_about_a_unit_go_in_one_request() -> None:
    fake = FakeJev()
    specs = [SchemaSpec.from_model(Car), SchemaSpec.from_model(Book)]
    await NoulComponentGate().gate(parsed(page()), specs, fake.client())
    assert len(fake.calls) == len(gate_units(page()))
    assert set(fake.calls[0].questions) == {"Car.price", "Car.performance", "Book.title"}
    assert fake.calls[0].questions["Book.title"].instructions == "Is this the book's title?"


async def test_threshold_is_inclusive_and_validated() -> None:
    fake = FakeJev(default_p=0.3)
    result = await NoulComponentGate(threshold=0.3).gate(
        parsed(page()), [SchemaSpec.from_model(Book)], fake.client()
    )
    assert "p3" in result["Book"]["title"]
    strict = await NoulComponentGate(threshold=0.31).gate(
        parsed(page()), [SchemaSpec.from_model(Book)], fake.client()
    )
    assert strict["Book"]["title"] == []
    with pytest.raises(ValueError, match="threshold"):
        NoulComponentGate(threshold=1.5)
    with pytest.raises(ValueError, match="max_chars"):
        NoulComponentGate(max_chars=0)


# --- the stage and what it does downstream -------------------------------------------


def context(fake: FakeJev, root: Component | None) -> Context:
    ctx = Context.create(
        Document.from_bytes(b"<p/>"),
        [SchemaSpec.from_model(Car), SchemaSpec.from_model(Book)],
        fake.client(),
    )
    if root is not None:
        ctx.parsed = parsed(root)
    return ctx


async def test_stage_sets_component_ids_and_scopes_keep_only_passing_components() -> None:
    fake = FakeJev().noul("price", p=0.9, state="24,995")
    ctx = context(fake, page())
    await ComponentGateStage().run(ctx)
    await EntityStage().run(ctx)

    car = ctx.schemas["Car"]
    assert car.component_ids is not None
    assert car.component_ids["performance"] == []
    [scope] = car.scopes
    assert scope.component_ids == ["root", "s-price", "h2", "p2", "t1"]
    assert [f.name for f in car.relevant_fields("p2")] == ["price"]
    assert car.relevant_fields("li1") == []

    book = ctx.schemas["Book"]
    assert book.scopes[0].component_ids == []
    assert [e.kind for e in ctx.events] == ["no_relevant_components"]


async def test_without_a_gate_everything_is_relevant() -> None:
    ctx = context(FakeJev(), page())
    await EntityStage().run(ctx)
    car = ctx.schemas["Car"]
    assert car.component_ids is None
    assert car.relevant_components() is None
    assert [f.name for f in car.relevant_fields("li1")] == ["price", "power_kw", "zero_to_62_s"]
    assert len(car.scopes[0].component_ids) == len(list(page().walk()))


async def test_stage_skips_without_a_parsed_document_or_active_schemas() -> None:
    fake = FakeJev()
    ctx = context(fake, None)
    await ComponentGateStage().run(ctx)
    assert ctx.schemas["Car"].component_ids is None
    ctx = context(fake, page())
    for run in ctx.schemas.values():
        run.deactivate()
    await ComponentGateStage().run(ctx)
    assert fake.calls == []


def test_component_gate_is_a_default_stage_between_layout_and_entities() -> None:
    names = [s.name for s in default_pipeline().stages]
    assert names.index("layout") < names.index("component_gate") < names.index("entities")


# --- a real page ---------------------------------------------------------------------

SPEC_PAGE = b"""<!doctype html><html><head><title>Delmaro Kestrova</title></head><body>
<nav><a href="/">Home</a> <a href="/cars">Cars</a></nav>
<main>
  <h1>Delmaro Kestrova 1.5 SE</h1>
  <p>A roomy family hatchback with a frugal petrol engine.</p>
  <section><h2>Performance</h2>
    <dl><dt>Power</dt><dd>110 kW</dd><dt>0-62 mph</dt><dd>9.1 s</dd></dl>
  </section>
  <section><h2>Price</h2><p>On the road from &pound;24,995.</p></section>
  <aside><h2>Newsletter</h2><p>Sign up for our weekly deals.</p></aside>
</main>
<footer>&copy; 2026 Example Motors</footer>
</body></html>"""


async def test_a_real_page_gates_by_section() -> None:
    doc = BoilerplateCleaner().clean(Document.from_bytes(SPEC_PAGE, url="https://cars.test/k"))
    root = await HtmlLayoutParser().parse(doc)
    fake = (
        FakeJev()
        .noul("engine power", p=0.9, state="0-62 mph: 9.1 s")
        .noul("price", p=0.9, state="24,995")
    )
    result = await NoulComponentGate().gate(
        ParsedDocument(document=doc, root=root), [SchemaSpec.from_model(Car)], fake.client()
    )
    by_id = {c.id: c for c in root.walk()}

    def texts(ids: list[str]) -> set[str]:
        return {by_id[i].text for i in ids if by_id[i].text}

    assert "0-62 mph: 9.1 s" in texts(result["Car"]["performance"])
    assert "On the road from £24,995." in texts(result["Car"]["price"])
    newsletter = texts(result["Car"]["price"]) | texts(result["Car"]["performance"])
    assert "Sign up for our weekly deals." not in newsletter

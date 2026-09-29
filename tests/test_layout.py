import pytest
from pydantic import BaseModel, ValidationError

from jevex import (
    BBox,
    Component,
    Context,
    Document,
    DomLocation,
    Extractor,
    Field,
    HtmlLayoutParser,
    ImageLocation,
    LayoutStage,
    PageLocation,
    Pipeline,
    SchemaSpec,
    TableCell,
)
from jevex.extractor import default_pipeline
from jevex.testing import FakeJev


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")


def tree() -> Component:
    return Component(
        id="c0",
        type="section",
        heading_trail=["Specifications"],
        location=DomLocation(dom_path="/html/body/main"),
        children=[
            Component(
                id="c1",
                type="heading",
                text="Performance",
                location=DomLocation(dom_path="/html/body/main/h2"),
            ),
            Component(
                id="c2",
                type="table",
                heading_trail=["Specifications", "Performance"],
                location=PageLocation(page=3, bbox=BBox(x0=10, y0=20, x1=300, y1=400)),
                cells=[TableCell(row=0, col=1, text="SE", header=True, col_span=2)],
                children=[
                    Component(
                        id="c3",
                        type="image",
                        location=ImageLocation(src="https://example.com/chart.png"),
                    )
                ],
            ),
        ],
    )


def test_walk_is_depth_first_in_reading_order() -> None:
    assert [c.id for c in tree().walk()] == ["c0", "c1", "c2", "c3"]


def test_find() -> None:
    root = tree()
    found = root.find("c3")
    assert found is not None
    assert found.type == "image"
    assert root.find("missing") is None


def test_round_trip_keeps_location_kinds() -> None:
    root = tree()
    restored = Component.model_validate_json(root.model_dump_json())
    assert restored == root
    assert isinstance(restored.children[1].location, PageLocation)
    assert isinstance(restored.children[1].children[0].location, ImageLocation)


def test_bbox_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=10, y0=0, x1=5, y1=5)


def test_page_is_one_based() -> None:
    with pytest.raises(ValidationError):
        PageLocation(page=0)


def test_unknown_component_type_rejected() -> None:
    with pytest.raises(ValidationError):
        Component.model_validate(
            {"id": "x", "type": "sidebar", "location": {"kind": "dom", "dom_path": "/"}}
        )


def test_table_cell_spans_and_positions_are_validated() -> None:
    with pytest.raises(ValidationError):
        TableCell(row=0, col=0, text="x", row_span=0)
    with pytest.raises(ValidationError):
        TableCell(row=-1, col=0, text="x")


# --- Stage -------------------------------------------------------------------------------


def html(body: str) -> Document:
    markup = f"<html><body>{body}</body></html>"
    return Document.from_bytes(markup.encode(), content_type="text/html")


def context(document: Document) -> Context:
    return Context.create(document, [SchemaSpec.from_model(Car)], FakeJev().client())


class Fixed:
    """A parser for one content type that returns a fixed tree."""

    def __init__(self, content_type: str, root: Component) -> None:
        self.content_type = content_type
        self.root = root

    def supports(self, document: Document) -> bool:
        return document.content_type == self.content_type

    async def parse(self, document: Document) -> Component:
        return self.root


async def test_stage_parses_html_by_default() -> None:
    doc = html("<h1>Golf</h1><p>1.5 TSI</p>")
    ctx = context(doc)
    await LayoutStage().run(ctx)
    assert ctx.parsed is not None
    assert ctx.parsed.document is doc
    assert [(c.type, c.text) for c in ctx.parsed.root.walk()] == [
        ("section", ""),
        ("heading", "Golf"),
        ("paragraph", "1.5 TSI"),
    ]
    assert ctx.parsed.statements == {}
    assert ctx.events == []


async def test_stage_uses_the_first_parser_that_supports_the_document() -> None:
    pdf = Document.from_bytes(b"%PDF-1.7\n", content_type="application/pdf")
    ctx = context(pdf)
    await LayoutStage([HtmlLayoutParser(), Fixed("application/pdf", tree())]).run(ctx)
    assert ctx.parsed is not None
    assert ctx.parsed.root == tree()


async def test_stage_records_an_event_when_no_parser_supports_the_document() -> None:
    ctx = context(Document.from_bytes(b"\x89PNG\r\n\x1a\n", content_type="image/png"))
    await LayoutStage().run(ctx)
    assert ctx.parsed is None
    assert not ctx.stopped
    (event,) = ctx.events
    assert (event.stage, event.kind, event.data) == (
        "layout",
        "layout_skipped",
        {"content_type": "image/png"},
    )
    assert event.message == "no layout parser supports image/png"


async def test_stage_does_not_swallow_parser_errors() -> None:
    class Broken:
        def supports(self, document: Document) -> bool:
            return True

        async def parse(self, document: Document) -> Component:
            raise ValueError("bad tree")

    with pytest.raises(ValueError, match="bad tree"):
        await LayoutStage([Broken()]).run(context(html("<p>x</p>")))


def test_default_pipeline_lays_out_after_gating() -> None:
    assert default_pipeline().names[:3] == ["clean", "document_gate", "layout"]


async def test_extractor_lays_out_the_cleaned_document() -> None:
    seen: list[Component] = []

    class Capture:
        name = "capture"

        async def run(self, ctx: Context) -> None:
            assert ctx.parsed is not None
            seen.append(ctx.parsed.root)

    fake = FakeJev().noul("Does this document describe a car?", p=0.9)
    pipeline: Pipeline = default_pipeline().append(Capture())
    async with Extractor([Car], jev=fake.client(), pipeline=pipeline) as ex:
        await ex.extract(html("<nav>Menu</nav><p>Golf</p>"))
    (root,) = seen
    assert [c.text for c in root.walk() if c.text] == ["Golf"]

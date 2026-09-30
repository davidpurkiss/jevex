from decimal import Decimal

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
    PdfLayoutParser,
    Pipeline,
    SchemaSpec,
    TableCell,
    layout,
    section_text,
)
from jevex.extractor import default_pipeline
from jevex.jev import Choice
from jevex.layout import MAX_HEADING_CHARS, MAX_SECTION_CHARS
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


def test_default_parsers_read_pdfs_when_the_pdf_extra_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(layout, "_docling_installed", lambda: True)
    assert [type(p) for p in LayoutStage().parsers] == [HtmlLayoutParser, PdfLayoutParser]
    monkeypatch.setattr(layout, "_docling_installed", lambda: False)
    assert [type(p) for p in LayoutStage().parsers] == [HtmlLayoutParser]


async def test_stage_suggests_the_pdf_extra_when_a_pdf_cannot_be_laid_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(layout, "_docling_installed", lambda: False)
    ctx = context(Document.from_bytes(b"%PDF-1.7\n"))
    await LayoutStage().run(ctx)
    assert ctx.parsed is None
    (event,) = ctx.events
    assert event.kind == "layout_skipped"
    assert event.message == (
        "no layout parser supports application/pdf; install jevex[pdf] for the default PDF parser"
    )


async def test_stage_does_not_swallow_parser_errors() -> None:
    class Broken:
        def supports(self, document: Document) -> bool:
            return True

        async def parse(self, document: Document) -> Component:
            raise ValueError("bad tree")

    with pytest.raises(ValueError, match="bad tree"):
        await LayoutStage([Broken()]).run(context(html("<p>x</p>")))


def test_default_pipeline_lays_out_after_gating() -> None:
    assert default_pipeline().names[:4] == ["clean", "document_gate", "structured", "layout"]


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


# --- section text ----------------------------------------------------------------------


def test_section_text_joins_a_short_trail_unchanged() -> None:
    assert section_text(["Specs", "Performance"]) == "Specs › Performance"
    assert section_text([]) == ""
    assert section_text(["Specs", "  ", "Performance"]) == "Specs › Performance"


def test_a_long_heading_is_shortened_at_a_word_boundary() -> None:
    text = section_text(["Kestrova", "word " * 100], max_heading_chars=30)
    assert text == "Kestrova › word word word word word word…"
    assert len(text.split(" › ")[1]) <= 30


def test_a_long_heading_without_spaces_is_cut_mid_word() -> None:
    assert section_text(["x" * 50], max_heading_chars=10) == "x" * 9 + "…"


def test_a_long_trail_keeps_the_outermost_and_innermost_headings() -> None:
    trail = ["Kestrova", "Specs", "Engine", "Performance", "Acceleration"]
    # "Performance" would make it 41 characters
    assert section_text(trail, max_chars=40) == "Kestrova › … › Acceleration"
    assert section_text(trail, max_chars=41) == "Kestrova › … › Performance › Acceleration"


def test_when_outermost_and_innermost_do_not_fit_only_the_innermost_is_kept() -> None:
    assert section_text(["Kestrova", "Acceleration"], max_chars=15) == "Acceleration"
    assert section_text(["Kestrova", "Acceleration times"], max_chars=12) == "Acceleratio…"


@pytest.mark.parametrize("trail", [["word " * 40_000], ["word " * 400] * 50, ["h"] * 1000])
def test_section_text_is_never_longer_than_the_cap(trail: list[str]) -> None:
    text = section_text(trail)
    assert text
    assert len(text) <= MAX_SECTION_CHARS
    assert all(len(h) <= MAX_HEADING_CHARS for h in text.split(" › "))


@pytest.mark.parametrize("kwargs", [{"max_chars": 0}, {"max_heading_chars": 0}])
def test_section_text_rejects_a_non_positive_cap(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        section_text(["Specs"], **kwargs)


class PricedCar(BaseModel):
    """A car."""

    price: Decimal = Field(description="Price", unit="GBP")


def pick_first_candidate(q: Choice) -> str:
    return next(o for o in q.options if o != "none")


async def test_a_200k_character_heading_is_extracted_without_raising() -> None:
    heading = "word " * 40_000
    body = f"<main><h1>{heading}</h1><h2>{heading}</h2><p>Price: £24,995</p></main>"
    fake = (
        FakeJev(default_p=1.0)
        .choice("Which detail", "price", state="24,995")
        .choice("Which of these", pick_first_candidate, state="24,995")
    )
    async with Extractor([PricedCar], jev=fake.client()) as ex:
        result = await ex.extract(html(body))

    assert result.one(PricedCar).strict() == PricedCar(price=Decimal("24995"))
    sections = [
        str(c.state["section"])
        for c in fake.calls
        if isinstance(c.state, dict) and "section" in c.state
    ]
    assert any("content" in c.state for c in fake.calls if isinstance(c.state, dict))
    assert any("statement" in c.state for c in fake.calls if isinstance(c.state, dict))
    assert sections
    assert max(len(s) for s in sections) <= MAX_SECTION_CHARS

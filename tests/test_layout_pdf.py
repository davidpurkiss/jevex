"""PDF layout through Docling.

``fixtures/pdf/spec.pdf`` is ``spec.html`` printed by headless Chrome, and
``spec.docling.json`` is what :class:`DoclingConverter` made of it (Docling 2.131), so
the mapping is tested on real Docling output without loading its models. The ``live``
test converts the PDF for real (it downloads Docling's models on first run).

Skipped when the ``pdf`` extra isn't installed (``uv sync --all-extras``); CI installs it.
"""

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

pytest.importorskip("docling")

import docling.document_converter
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table_data import TableCell as DlCell
from docling_core.types.doc.items.table.table_data import TableData
from docling_core.types.doc.labels import DocItemLabel

from jevex import (
    BBox,
    Component,
    DoclingConverter,
    Document,
    DocumentGateStage,
    Extractor,
    Field,
    LayoutStage,
    PageLocation,
    PdfLayoutError,
    PdfLayoutParser,
    Pipeline,
    SchemaConfig,
    TableCell,
    UnsupportedDocumentError,
    _pdfium,
    table_statements,
)
from jevex.interfaces import LayoutParser, PagedLayoutParser
from jevex.layout_pdf import from_docling
from jevex.testing import FakeJev

FIXTURES = Path(__file__).parent / "fixtures" / "pdf"


def outline(root: Component) -> list[tuple[int, str, str]]:
    """(depth, type, text) for every component below the root, in reading order."""
    out: list[tuple[int, str, str]] = []

    def visit(c: Component, depth: int) -> None:
        for child in c.children:
            out.append((depth, child.type, child.text))
            visit(child, depth + 1)

    visit(root, 0)
    return out


def where(c: Component) -> tuple[int, tuple[int, int, int, int] | None]:
    """The page and the bbox rounded to whole points."""
    assert isinstance(c.location, PageLocation)
    b = c.location.bbox
    return c.location.page, (round(b.x0), round(b.y0), round(b.x1), round(b.y1)) if b else None


def spec_document() -> DoclingDocument:
    return DoclingDocument.model_validate_json((FIXTURES / "spec.docling.json").read_text())


def pdf_document() -> Document:
    return Document.from_path(FIXTURES / "spec.pdf")


def new_doc(*pages: int) -> DoclingDocument:
    doc = DoclingDocument(name="test")
    for n in pages or (1,):
        doc.add_page(page_no=n, size=Size(width=600, height=800))
    return doc


def prov(
    page: int,
    box: tuple[float, float, float, float] = (10, 10, 100, 20),
    origin: CoordOrigin = CoordOrigin.TOPLEFT,
) -> ProvenanceItem:
    left, top, right, bottom = box
    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=origin),
        charspan=(0, 0),
    )


def cell(
    text: str,
    row: int,
    col: int,
    *,
    rows: int = 1,
    cols: int = 1,
    column_header: bool = False,
    row_header: bool = False,
    row_section: bool = False,
) -> DlCell:
    return DlCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + rows,
        start_col_offset_idx=col,
        end_col_offset_idx=col + cols,
        row_span=rows,
        col_span=cols,
        column_header=column_header,
        row_header=row_header,
        row_section=row_section,
    )


# --- Real Docling output -----------------------------------------------------------------


def test_spec_sheet_maps_reading_order_headings_lists_and_tables() -> None:
    root = from_docling(spec_document())
    assert root.type == "section"
    assert outline(root) == [
        (0, "heading", "Skoda Octavia Estate"),
        (
            0,
            "paragraph",
            "The Octavia Estate combines a large boot with efficient engines. Prices start "
            "at £27,500 on the road.",
        ),
        (0, "section", ""),
        (1, "heading", "Engines"),
        (1, "paragraph", "Two petrol engines and one diesel are available."),
        (1, "list", ""),
        (2, "list_item", "1.5 TSI 150PS manual"),
        (2, "list_item", "1.5 TSI 150PS DSG"),
        (2, "list_item", "2.0 TDI 115PS manual"),
        # Docling ranks "Engines" above "Performance" and "Dimensions": its font-size
        # reading counts the descender in "g".
        (1, "section", ""),
        (2, "heading", "Performance"),
        (
            2,
            "table",
            "1.5 TSI SE | 2.0 TDI SE L\n0-62 mph (s) | 8.5 | 10.4\n"
            "Top speed (mph) | 139 | 128\nCO2 (g/km) | 131 | 118",
        ),
        (2, "paragraph", "Figures are for the manufacturer's test cycle."),
        (1, "section", ""),
        (2, "heading", "Dimensions"),
        (2, "section", ""),
        (3, "heading", "Exterior"),
        (3, "paragraph", "Length is 4,698 mm and width is 1,829 mm."),
        (2, "section", ""),
        (3, "heading", "Boot"),
        (3, "paragraph", "The boot holds 640 litres with the seats up."),
    ]
    assert [c.id for c in root.walk()] == [f"c{n}" for n in range(len(outline(root)) + 1)]


def test_spec_sheet_components_carry_heading_trails() -> None:
    trails = {c.text: c.heading_trail for c in from_docling(spec_document()).walk() if c.text}
    assert trails["Skoda Octavia Estate"] == []
    assert trails["1.5 TSI 150PS manual"] == ["Skoda Octavia Estate", "Engines"]
    assert trails["The boot holds 640 litres with the seats up."] == [
        "Skoda Octavia Estate",
        "Engines",
        "Dimensions",
        "Boot",
    ]


def test_spec_sheet_locations_are_pages_and_top_left_boxes() -> None:
    root = from_docling(spec_document())
    by_text = {c.text: c for c in root.walk() if c.text}
    # Docling's box is bottom-left (t=702.6, b=684.3 on a 792 pt page); ours is top-left.
    assert where(by_text["Skoda Octavia Estate"]) == (1, (84, 89, 329, 108))
    assert where(by_text["Length is 4,698 mm and width is 1,829 mm."]) == (2, (84, 108, 297, 119))
    (table,) = [c for c in root.walk() if c.type == "table"]
    assert where(table) == (1, (84, 311, 328, 388))
    (items,) = [c for c in root.walk() if c.type == "list"]
    assert where(items) == (1, (114, 227, 225, 260))  # the union of its items
    # "Engines" runs from page 1 to page 2: a page but no single box.
    engines = next(c for c in root.walk() if c.children and c.children[0].text == "Engines")
    assert where(engines) == (1, None)
    assert where(root) == (1, None)


def test_spec_sheet_table_cells_render_with_their_headers() -> None:
    (table,) = [c for c in from_docling(spec_document()).walk() if c.type == "table"]
    assert table.cells[:4] == [
        TableCell(row=0, col=1, text="1.5 TSI SE", header=True),
        TableCell(row=0, col=2, text="2.0 TDI SE L", header=True),
        TableCell(row=1, col=0, text="0-62 mph (s)", header=True),
        TableCell(row=1, col=1, text="8.5"),
    ]
    assert [s.text for s in table_statements(table)] == [
        "0-62 mph (s) · 1.5 TSI SE: 8.5",
        "0-62 mph (s) · 2.0 TDI SE L: 10.4",
        "Top speed (mph) · 1.5 TSI SE: 139",
        "Top speed (mph) · 2.0 TDI SE L: 128",
        "CO2 (g/km) · 1.5 TSI SE: 131",
        "CO2 (g/km) · 2.0 TDI SE L: 118",
    ]


@pytest.mark.live
async def test_docling_converts_the_spec_sheet_as_recorded() -> None:
    root = await PdfLayoutParser().parse(pdf_document())
    assert root == from_docling(spec_document())


# --- Mapping -----------------------------------------------------------------------------


def test_a_title_outranks_section_headers() -> None:
    doc = new_doc()
    doc.add_title("Octavia", prov=prov(1))
    doc.add_heading("Engines", level=1, prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "1.5 TSI", prov=prov(1))
    doc.add_heading("Prices", level=1, prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "£27,500", prov=prov(1))
    root = from_docling(doc)
    assert outline(root) == [
        (0, "heading", "Octavia"),
        (0, "section", ""),
        (1, "heading", "Engines"),
        (1, "paragraph", "1.5 TSI"),
        (0, "section", ""),
        (1, "heading", "Prices"),
        (1, "paragraph", "£27,500"),
    ]
    assert root.children[2].children[1].heading_trail == ["Octavia", "Prices"]


def test_deeper_section_headers_nest() -> None:
    doc = new_doc()
    doc.add_heading("Specifications", level=1, prov=prov(1))
    doc.add_heading("Performance", level=2, prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "0-62 mph in 8.5 s", prov=prov(1))
    doc.add_heading("Dimensions", level=2, prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "Length 4,698 mm", prov=prov(1))
    trails = {c.text: c.heading_trail for c in from_docling(doc).walk() if c.type == "paragraph"}
    assert trails == {
        "0-62 mph in 8.5 s": ["Specifications", "Performance"],
        "Length 4,698 mm": ["Specifications", "Dimensions"],
    }


def test_captions_and_footnotes_attach_to_their_table_or_picture() -> None:
    doc = new_doc()
    table_caption = doc.add_text(DocItemLabel.CAPTION, "Table 1: Performance", prov=prov(1))
    table = doc.add_table(
        data=TableData(num_rows=1, num_cols=1, table_cells=[cell("8.5", 0, 0)]),
        caption=table_caption,
        prov=prov(1, (10, 30, 200, 90)),
    )
    note = doc.add_text(DocItemLabel.FOOTNOTE, "Manufacturer's figures.", prov=prov(1))
    table.footnotes.append(note.get_ref())
    figure_caption = doc.add_text(DocItemLabel.CAPTION, "Figure 1: Boot", prov=prov(2))
    doc.add_picture(caption=figure_caption, prov=prov(2, (50, 60, 300, 400)))
    root = from_docling(doc)
    assert outline(root) == [
        (0, "table", "8.5"),
        (1, "caption", "Table 1: Performance"),
        (1, "paragraph", "Manufacturer's figures."),
        (0, "image", ""),
        (1, "caption", "Figure 1: Boot"),
    ]
    assert where(root.children[1]) == (2, (50, 60, 300, 400))


def test_a_caption_that_is_also_a_child_appears_once() -> None:
    doc = new_doc()
    picture = doc.add_picture(prov=prov(1))
    caption = doc.add_text(DocItemLabel.CAPTION, "Figure 1", prov=prov(1), parent=picture)
    picture.captions.append(caption.get_ref())
    assert outline(from_docling(doc)) == [(0, "image", ""), (1, "caption", "Figure 1")]


def test_text_inside_a_picture_becomes_its_children() -> None:
    doc = new_doc()
    picture = doc.add_picture(prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "0-62 mph 8.5 s", prov=prov(1), parent=picture)
    assert outline(from_docling(doc)) == [(0, "image", ""), (1, "paragraph", "0-62 mph 8.5 s")]


def test_page_furniture_is_left_out() -> None:
    doc = new_doc()
    doc.add_text(DocItemLabel.PAGE_HEADER, "Octavia brochure", prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "Prices from £27,500.", prov=prov(1))
    doc.add_text(
        DocItemLabel.TEXT, "Page 1 of 12", prov=prov(1), content_layer=ContentLayer.FURNITURE
    )
    doc.add_text(DocItemLabel.PAGE_FOOTER, "© Skoda", prov=prov(1))
    assert outline(from_docling(doc)) == [(0, "paragraph", "Prices from £27,500.")]


def test_inline_runs_join_into_one_paragraph() -> None:
    doc = new_doc()
    group = doc.add_inline_group()
    doc.add_text(DocItemLabel.TEXT, "Boot space is", prov=prov(1, (10, 10, 80, 20)), parent=group)
    doc.add_text(DocItemLabel.TEXT, " 640 litres", prov=prov(1, (80, 10, 140, 22)), parent=group)
    doc.add_text(DocItemLabel.TEXT, ".", prov=prov(1, (140, 10, 145, 20)), parent=group)
    root = from_docling(doc)
    assert outline(root) == [(0, "paragraph", "Boot space is 640 litres.")]
    assert where(root.children[0]) == (1, (10, 10, 145, 22))


def test_inline_runs_keep_brackets_tight() -> None:
    doc = new_doc()
    group = doc.add_inline_group()
    for run in ["CO2", "(", "WLTP", ")", ": 131 g/km"]:
        doc.add_text(DocItemLabel.TEXT, run, prov=prov(1), parent=group)
    assert outline(from_docling(doc)) == [(0, "paragraph", "CO2 (WLTP): 131 g/km")]


def test_empty_inline_groups_and_text_are_dropped() -> None:
    doc = new_doc()
    doc.add_inline_group()
    doc.add_text(DocItemLabel.TEXT, "   ", prov=prov(1))
    doc.add_text(DocItemLabel.FORMULA, "", prov=prov(1))
    assert outline(from_docling(doc)) == []


@pytest.mark.parametrize(
    ("marker", "text", "expected"),
    [
        ("•", "1.5 TSI", "1.5 TSI"),
        ("-", "Heated seats", "Heated seats"),
        ("1.", "Open the boot", "Open the boot"),
        ("b)", "Fold the seats", "Fold the seats"),
        ("(iv)", "Lock", "Lock"),
        ("1.5", "TSI 150PS manual", "1.5 TSI 150PS manual"),  # a number, not a marker
        ("2.0", "TDI", "2.0 TDI"),
        ("", "Sunroof", "Sunroof"),
    ],
)
def test_list_item_markers(marker: str, text: str, expected: str) -> None:
    doc = new_doc()
    group = doc.add_list_group()
    doc.add_list_item(text, marker=marker, enumerated=bool(marker), prov=prov(1), parent=group)
    assert outline(from_docling(doc)) == [(0, "list", ""), (1, "list_item", expected)]


def test_nested_lists_stay_under_their_item() -> None:
    doc = new_doc()
    outer = doc.add_list_group()
    item = doc.add_list_item("Engines", prov=prov(1), parent=outer)
    inner = doc.add_list_group(parent=item)
    doc.add_list_item("1.5 TSI", prov=prov(1), parent=inner)
    assert outline(from_docling(doc)) == [
        (0, "list", ""),
        (1, "list_item", "Engines"),
        (2, "list", ""),
        (3, "list_item", "1.5 TSI"),
    ]


def test_list_items_outside_a_list_are_wrapped_in_one() -> None:
    # docling-core now puts every list item in a list group, but older documents didn't.
    doc = new_doc()
    group = doc.add_list_group()
    doc.add_list_item("Heated seats", prov=prov(1, (10, 10, 90, 20)), parent=group)
    doc.add_list_item("Sunroof", prov=prov(1, (10, 30, 70, 40)), parent=group)
    doc.add_text(DocItemLabel.TEXT, "Options cost extra.", prov=prov(1))
    other = doc.add_list_group()
    doc.add_list_item("Tow bar", prov=prov(1), parent=other)
    doc.body.children = [*group.children, doc.body.children[1], *other.children]
    root = from_docling(doc)
    assert outline(root) == [
        (0, "list", ""),
        (1, "list_item", "Heated seats"),
        (1, "list_item", "Sunroof"),
        (0, "paragraph", "Options cost extra."),
        (0, "list", ""),
        (1, "list_item", "Tow bar"),
    ]
    assert where(root.children[0]) == (1, (10, 10, 90, 40))


def test_table_cells_keep_spans_and_mark_every_kind_of_header() -> None:
    doc = new_doc()
    doc.add_table(
        data=TableData(
            num_rows=4,
            num_cols=3,
            table_cells=[
                cell("Trim", 0, 0, column_header=True),
                cell("SE", 0, 1, cols=2, column_header=True),
                cell("SE", 0, 1, cols=2, column_header=True),  # a spanning cell listed twice
                cell("Performance", 1, 0, cols=3, row_section=True),
                cell("0-62 mph (s)", 2, 0, row_header=True),
                cell("8.5", 2, 1, rows=2),
                cell("", 2, 2),
                cell("Top speed", 3, 0, row_header=True),
            ],
        ),
        prov=prov(1),
    )
    (table,) = from_docling(doc).children
    assert table.cells == [
        TableCell(row=0, col=0, text="Trim", header=True),
        TableCell(row=0, col=1, text="SE", header=True, col_span=2),
        TableCell(row=1, col=0, text="Performance", header=True, col_span=3),
        TableCell(row=2, col=0, text="0-62 mph (s)", header=True),
        TableCell(row=2, col=1, text="8.5", row_span=2),
        TableCell(row=3, col=0, text="Top speed", header=True),
    ]
    assert table.text == "Trim | SE\nPerformance\n0-62 mph (s) | 8.5\nTop speed"


def test_a_table_without_text_leaves_only_its_caption() -> None:
    doc = new_doc()
    caption = doc.add_text(DocItemLabel.CAPTION, "Table 2", prov=prov(1))
    doc.add_table(
        data=TableData(num_rows=1, num_cols=1, table_cells=[cell(" ", 0, 0)]),
        caption=caption,
        prov=prov(1),
    )
    assert outline(from_docling(doc)) == [(0, "caption", "Table 2")]


def test_boxes_are_flipped_to_top_left_and_unioned_on_the_first_page() -> None:
    doc = new_doc(1, 2)
    text = doc.add_text(
        DocItemLabel.TEXT, "A paragraph", prov=prov(1, (10, 790, 300, 700), CoordOrigin.BOTTOMLEFT)
    )
    text.prov.append(prov(1, (10, 690, 200, 680), CoordOrigin.BOTTOMLEFT))
    text.prov.append(prov(2, (10, 10, 300, 40)))  # continues on the next page
    (paragraph,) = from_docling(doc).children
    assert paragraph.location == PageLocation(page=1, bbox=BBox(x0=10, y0=10, x1=300, y1=120))


def test_a_bottom_left_box_on_an_unknown_page_has_no_bbox() -> None:
    doc = new_doc(1)
    doc.add_text(
        DocItemLabel.TEXT, "Orphan", prov=prov(3, (10, 50, 90, 40), CoordOrigin.BOTTOMLEFT)
    )
    (paragraph,) = from_docling(doc).children
    assert paragraph.location == PageLocation(page=3)


def test_items_without_provenance_sit_on_page_one() -> None:
    doc = new_doc()
    doc.add_text(DocItemLabel.TEXT, "Unplaced")
    (paragraph,) = from_docling(doc).children
    assert paragraph.location == PageLocation(page=1)


def test_an_empty_document_is_an_empty_section() -> None:
    root = from_docling(new_doc())
    assert root == Component(id="c0", type="section", location=PageLocation(page=1))


def test_groups_other_than_lists_add_no_level() -> None:
    doc = new_doc()
    group = doc.add_group(name="key-values")
    doc.add_heading("Engine", level=1, prov=prov(1), parent=group)
    doc.add_text(DocItemLabel.TEXT, "1.5 TSI", prov=prov(1), parent=group)
    assert outline(from_docling(doc)) == [(0, "heading", "Engine"), (0, "paragraph", "1.5 TSI")]


def test_a_heading_in_a_group_ranks_against_the_headings_around_it() -> None:
    doc = new_doc()
    doc.add_heading("Specs", level=1, prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "Intro", prov=prov(1))
    group = doc.add_group(name="key-values")
    doc.add_heading("Engine", level=1, prov=prov(1), parent=group)
    doc.add_text(DocItemLabel.TEXT, "1.5 TSI", prov=prov(1), parent=group)
    doc.add_text(DocItemLabel.TEXT, "After the group", prov=prov(1))
    doc.add_heading("Prices", level=2, prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "£27,500", prov=prov(1))
    root = from_docling(doc)
    trails = {c.text: c.heading_trail for c in root.walk() if c.text}
    assert trails == {
        "Specs": [],
        "Intro": ["Specs"],
        "Engine": [],
        "1.5 TSI": ["Engine"],
        "After the group": ["Engine"],
        "Prices": ["Engine"],
        "£27,500": ["Engine", "Prices"],
    }
    assert [(c.type, c.text) for c in root.children] == [
        ("section", ""),
        ("section", ""),
    ]


def test_headings_inside_a_container_do_not_leak_out() -> None:
    doc = new_doc()
    doc.add_heading("Options", level=1, prov=prov(1))
    group = doc.add_list_group()
    item = doc.add_list_item("Packs", prov=prov(1), parent=group)
    doc.add_heading("Winter pack", level=2, prov=prov(1), parent=item)
    doc.add_text(DocItemLabel.TEXT, "Heated seats", prov=prov(1), parent=item)
    doc.add_text(DocItemLabel.TEXT, "Prices include VAT.", prov=prov(1))
    trails = {c.text: c.heading_trail for c in from_docling(doc).walk() if c.type == "paragraph"}
    assert trails == {
        "Heated seats": ["Options", "Winter pack"],
        "Prices include VAT.": ["Options"],
    }


def test_same_rank_headings_inside_a_container_close_each_other() -> None:
    doc = new_doc()
    doc.add_heading("Options", level=1, prov=prov(1))
    group = doc.add_list_group()
    item = doc.add_list_item("Packs", prov=prov(1), parent=group)
    doc.add_heading("Winter pack", level=2, prov=prov(1), parent=item)
    doc.add_text(DocItemLabel.TEXT, "Heated seats", prov=prov(1), parent=item)
    doc.add_heading("Tech pack", level=2, prov=prov(1), parent=item)
    doc.add_text(DocItemLabel.TEXT, "Head-up display", prov=prov(1), parent=item)
    picture = doc.add_picture(prov=prov(1))
    doc.add_heading("Front", level=1, prov=prov(1), parent=picture)
    doc.add_text(DocItemLabel.TEXT, "LED lights", prov=prov(1), parent=picture)
    doc.add_heading("Rear", level=1, prov=prov(1), parent=picture)
    doc.add_text(DocItemLabel.TEXT, "Tow bar", prov=prov(1), parent=picture)
    trails = {c.text: c.heading_trail for c in from_docling(doc).walk() if c.type == "paragraph"}
    assert trails == {
        "Heated seats": ["Options", "Winter pack"],
        "Head-up display": ["Options", "Tech pack"],
        "LED lights": ["Options", "Front"],
        "Tow bar": ["Options", "Rear"],
    }


# --- Parser ------------------------------------------------------------------------------


def test_parser_satisfies_the_protocol_and_reads_only_pdfs() -> None:
    parser = PdfLayoutParser(convert=lambda _: new_doc())
    assert isinstance(parser, LayoutParser)
    assert parser.supports(pdf_document())
    assert not parser.supports(Document.from_bytes(b"<html></html>", content_type="text/html"))


async def test_parser_maps_what_its_converter_returns_off_the_event_loop() -> None:
    threads: list[threading.Thread] = []
    seen: list[Document] = []

    def convert(document: Document) -> DoclingDocument:
        threads.append(threading.current_thread())
        seen.append(document)
        return spec_document()

    document = pdf_document()
    root = await PdfLayoutParser(convert=convert).parse(document)
    assert root == from_docling(spec_document())
    assert seen == [document]
    assert threads != [threading.current_thread()]


async def test_parser_rejects_other_documents() -> None:
    parser = PdfLayoutParser(convert=lambda _: new_doc())
    with pytest.raises(UnsupportedDocumentError, match="reads PDFs, not text/html"):
        await parser.parse(Document.from_bytes(b"<p>x</p>", content_type="text/html"))


async def test_parser_does_not_swallow_converter_errors() -> None:
    def convert(document: Document) -> DoclingDocument:
        raise PdfLayoutError("broken xref")

    with pytest.raises(PdfLayoutError, match="broken xref"):
        await PdfLayoutParser(convert=convert).parse(pdf_document())


# --- Leaving pages out -------------------------------------------------------------------


def test_parser_can_leave_pages_out() -> None:
    assert isinstance(PdfLayoutParser(convert=lambda _: new_doc()), PagedLayoutParser)


async def test_parser_converts_only_the_pages_kept_and_numbers_them_as_the_original() -> None:
    seen: list[Document] = []

    def convert(document: Document) -> DoclingDocument:
        seen.append(document)
        doc = new_doc(1)
        doc.add_text(DocItemLabel.TEXT, "Boot", prov=prov(1))
        doc.add_text(DocItemLabel.TEXT, "Unplaced")
        return doc

    document = pdf_document().model_copy(update={"url": "https://example.com/spec.pdf"})
    root = await PdfLayoutParser(convert=convert).parse_pages(document, frozenset({1}))

    (part,) = seen
    assert part.url == document.url
    assert part.content_type == "application/pdf"
    assert _pdfium.page_texts(part.content) == _pdfium.page_texts(document.content)[1:]
    assert [(c.text, where(c)) for c in root.children] == [
        ("Boot", (2, (10, 10, 100, 20))),
        ("Unplaced", (2, None)),
    ]
    assert where(root) == (2, (10, 10, 100, 20))


async def test_parser_converts_the_whole_pdf_when_no_page_is_left_out() -> None:
    seen: list[Document] = []

    def convert(document: Document) -> DoclingDocument:
        seen.append(document)
        return spec_document()

    document = pdf_document()
    parser = PdfLayoutParser(convert=convert)
    assert await parser.parse_pages(document, frozenset()) == from_docling(spec_document())
    # Numbers that aren't pages of the PDF are ignored.
    assert await parser.parse_pages(document, frozenset({0, 3})) == from_docling(spec_document())
    assert seen == [document, document]


async def test_parser_leaving_every_page_out_gives_an_empty_section() -> None:
    def convert(document: Document) -> DoclingDocument:
        raise AssertionError("nothing to convert")

    root = await PdfLayoutParser(convert=convert).parse_pages(pdf_document(), frozenset({1, 2}))
    assert root == Component(id="c0", type="section", location=PageLocation(page=1))


async def test_parser_rejects_other_documents_when_leaving_pages_out() -> None:
    parser = PdfLayoutParser(convert=lambda _: new_doc())
    with pytest.raises(UnsupportedDocumentError, match="reads PDFs, not text/html"):
        await parser.parse_pages(
            Document.from_bytes(b"<p>x</p>", content_type="text/html"), frozenset({1})
        )


def test_from_docling_numbers_pages_as_the_original() -> None:
    doc = new_doc(1, 2)
    doc.add_text(DocItemLabel.TEXT, "First kept", prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "Second kept", prov=prov(2))
    root = from_docling(doc, pages=[3, 7])
    assert [where(c)[0] for c in root.children] == [3, 7]
    assert where(root) == (3, None)


def test_from_docling_rejects_a_page_outside_the_pages_converted() -> None:
    doc = new_doc(1)
    doc.add_text(DocItemLabel.TEXT, "Stray", prov=prov(2))
    with pytest.raises(PdfLayoutError, match="Docling placed an item on page 2 of a 1-page PDF"):
        from_docling(doc, pages=[4])


async def test_extractor_lays_out_only_the_pdf_pages_the_gate_passed() -> None:
    class Brochure(BaseModel):
        """A car brochure page."""

        __jevex__ = SchemaConfig(gate_unit="page")

        model: str = Field(description="Model name")

    converted: list[list[str]] = []

    def convert(document: Document) -> DoclingDocument:
        converted.append([t.split("\r\n")[0] for t in _pdfium.page_texts(document.content)])
        doc = new_doc(1)
        doc.add_text(DocItemLabel.TEXT, "Performance", prov=prov(1))
        return doc

    question = "Does this document describe a car brochure page?"
    fake = (
        FakeJev(strict=True)
        .noul(question, p=0.9, state="Skoda Octavia Estate")
        .noul(question, p=0.1, state="Dimensions")
    )
    pipeline = Pipeline([DocumentGateStage(), LayoutStage([PdfLayoutParser(convert=convert)])])
    async with Extractor([Brochure], jev=fake.client(), pipeline=pipeline) as ex:
        result = await ex.extract(pdf_document())

    assert converted == [["Skoda Octavia Estate"]]
    assert result.meta.gates["Brochure"].passed_pages == [1]
    assert [(e.kind, e.data) for e in result.meta.events] == [("pages_skipped", {"pages": [2]})]


# --- Default converter -------------------------------------------------------------------


class FakeDocling:
    """Stands in for Docling's ``DocumentConverter``, which would load models."""

    instances: list["FakeDocling"] = []  # noqa: RUF012
    result: Any = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.sources: list[Any] = []
        FakeDocling.instances.append(self)

    def convert(self, source: Any, *, raises_on_error: bool) -> Any:
        assert raises_on_error is False
        self.sources.append(source)
        return FakeDocling.result


@pytest.fixture
def fake_docling(monkeypatch: pytest.MonkeyPatch) -> type[FakeDocling]:
    FakeDocling.instances = []
    FakeDocling.result = SimpleNamespace(
        status=ConversionStatus.SUCCESS, errors=[], document=spec_document()
    )
    monkeypatch.setattr(docling.document_converter, "DocumentConverter", FakeDocling)
    return FakeDocling


def test_converter_runs_docling_without_ocr_and_with_heading_levels(
    fake_docling: type[FakeDocling],
) -> None:
    converter = DoclingConverter()
    assert fake_docling.instances == []  # models load on first use
    document = pdf_document()
    assert converter(document) == spec_document()
    (docling,) = fake_docling.instances
    assert docling.kwargs["allowed_formats"] == [InputFormat.PDF]
    options = docling.kwargs["format_options"][InputFormat.PDF].pipeline_options
    assert isinstance(options, PdfPipelineOptions)
    assert options.do_ocr is False
    assert options.do_table_structure is True
    assert options.heading_hierarchy_options.enabled is True
    assert options.generate_parsed_pages is True
    (source,) = docling.sources
    assert source.name == "document.pdf"
    assert source.stream.read() == document.content


def test_converter_reuses_docling_across_documents(fake_docling: type[FakeDocling]) -> None:
    converter = DoclingConverter()
    converter(pdf_document())
    converter(pdf_document())
    (docling,) = fake_docling.instances
    assert len(docling.sources) == 2


def test_converter_accepts_partial_conversions(fake_docling: type[FakeDocling]) -> None:
    fake_docling.result.status = ConversionStatus.PARTIAL_SUCCESS
    assert DoclingConverter()(pdf_document()) == spec_document()


def test_converter_raises_when_docling_fails(fake_docling: type[FakeDocling]) -> None:
    fake_docling.result = SimpleNamespace(
        status=ConversionStatus.FAILURE,
        errors=[SimpleNamespace(error_message="Invalid PDF: no xref table")],
        document=new_doc(),
    )
    with pytest.raises(PdfLayoutError, match="Invalid PDF: no xref table"):
        DoclingConverter()(pdf_document())


def test_converter_failure_without_details(fake_docling: type[FakeDocling]) -> None:
    fake_docling.result = SimpleNamespace(
        status=ConversionStatus.FAILURE, errors=[], document=new_doc()
    )
    with pytest.raises(PdfLayoutError, match="no details"):
        DoclingConverter()(pdf_document())

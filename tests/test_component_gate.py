from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

import pytest
from pydantic import BaseModel

from jevex import (
    BoilerplateCleaner,
    CategoriseStage,
    Component,
    ComponentGateResult,
    ComponentGateStage,
    Context,
    Document,
    DomLocation,
    EntityStage,
    Extractor,
    Field,
    FieldMeta,
    LayoutStage,
    MultiEntity,
    NoulComponentGate,
    ParentChild,
    Pipeline,
    Questions,
    SchemaConfig,
    SchemaSpec,
    SingleEntity,
    Statement,
    StatementStage,
    StructuredStage,
    gate_units,
)
from jevex.component_gate import _question_key  # pyright: ignore[reportPrivateUsage]
from jevex.entities import EntityScope
from jevex.extractor import default_pipeline
from jevex.interfaces import ComponentGate, ParsedDocument
from jevex.jev import (
    Choice,
    ChoiceAnswer,
    JevClient,
    JevResponse,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    ScoreAnswer,
    UnexpectedAnswerError,
)
from jevex.layout import MAX_SECTION_CHARS, TableCell
from jevex.layout_html import HtmlLayoutParser
from jevex.resolve import SINGLE_ENTITY_LABEL
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


def test_a_unit_states_a_capped_heading_trail() -> None:
    trail = ["Kestrova", "word " * 1000, "Performance"]
    [unit] = gate_units(comp("section", "", "r", comp("paragraph", "9.1 s", "p", trail=trail)))
    section = unit.state()["section"]
    assert section.startswith("Kestrova › word word")
    assert section.endswith("… › Performance")
    assert len(section) <= MAX_SECTION_CHARS


def test_long_runs_are_chunked() -> None:
    paragraphs = [comp("paragraph", "x" * 40, f"p{i}") for i in range(5)]
    units = gate_units(comp("section", "", "root", *paragraphs), max_chars=100)
    assert [u.component_ids for u in units] == [("p0", "p1"), ("p2", "p3"), ("p4",)]


def test_an_oversized_paragraph_is_split_not_cut() -> None:
    sentences = [f"Sentence {i} is here." for i in range(30)] + ["It costs £24,995."]
    text = " ".join(sentences)
    units = gate_units(comp("section", "", "r", comp("paragraph", text, "p")), max_chars=100)
    assert len(units) > 1
    assert all(len(u.text) <= 100 for u in units)
    assert all(u.component_ids == ("p",) for u in units)
    assert "It costs £24,995." in units[-1].text
    # Nothing is lost: every sentence is whole in some unit.
    assert all(any(sent in u.text for u in units) for sent in sentences)


def test_a_sentence_longer_than_the_cap_becomes_overlapping_windows() -> None:
    text = "x" * 150 + " price 24995 " + "y" * 150
    units = gate_units(comp("section", "", "r", comp("paragraph", text, "p")), max_chars=100)
    assert all(len(u.text) <= 100 for u in units)
    assert any("24995" in u.text for u in units)


def table(cid: str, rows: int) -> Component:
    grid = [("Trim", "Price")] + [(f"Trim {r}", f"£{20000 + r}") for r in range(1, rows + 1)]
    cells = [
        {"row": r, "col": c, "text": text, "header": r == 0}
        for r, row in enumerate(grid)
        for c, text in enumerate(row)
    ]
    return Component.model_validate(
        {
            "id": cid,
            "type": "table",
            "text": "\n".join(" | ".join(row) for row in grid),
            "cells": cells,
            "location": DomLocation(dom_path=f"/{cid}"),
        }
    )


def test_an_oversized_table_is_split_by_rows_with_its_header_repeated() -> None:
    units = gate_units(comp("section", "", "r", table("t", 40)), max_chars=120)
    assert len(units) > 1
    assert [u.id for u in units][:2] == ["t:0", "t:1"]
    for u in units:
        assert len(u.text) <= 120
        assert u.text.startswith("Trim | Price\n")
        assert u.component_ids == ("t",)
    assert "Trim 40 | £20040" in units[-1].text


def test_an_oversized_tables_blank_row_is_a_body_row_without_gaps() -> None:
    t = table("t", 40)
    blank = [
        TableCell(row=41, col=0, text="Towing", header=True),
        TableCell(row=41, col=1, text=""),
    ]
    t = t.model_copy(update={"cells": [*t.cells, *blank]})
    units = gate_units(comp("section", "", "r", t), max_chars=120)
    lines = [line for u in units for line in u.text.split("\n")]
    assert all(u.text.startswith("Trim | Price\nTrim ") for u in units)
    assert lines.count("Towing") == 1
    assert units[-1].text.endswith("Trim 40 | £20040\nTowing")
    assert not any(line.startswith(" |") or line.endswith("| ") for line in lines)


def plain_table(*grid: tuple[str, ...], cid: str = "t") -> Component:
    """A table of ``td`` cells only: no header cells."""
    return Component.model_validate(
        {
            "id": cid,
            "type": "table",
            "text": "\n".join(" | ".join(row) for row in grid),
            "cells": [
                {"row": r, "col": c, "text": text}
                for r, row in enumerate(grid)
                for c, text in enumerate(row)
            ],
            "location": DomLocation(dom_path=f"/{cid}"),
        }
    )


def spec_grid(rows: int) -> list[tuple[str, ...]]:
    return [("Spec", "SE", "GT")] + [(f"Spec {r}", f"{r}0 PS", f"{r}5 PS") for r in range(1, rows)]


def test_an_oversized_table_without_headers_splits_as_rows() -> None:
    # Whether its first row is a header isn't known yet: the gate asks about the first piece.
    units = gate_units(comp("section", "", "r", plain_table(*spec_grid(30))), max_chars=120)
    assert len(units) > 1
    assert units[0].text.startswith("Spec | SE | GT\nSpec 1 | 10 PS")
    assert not any("Spec | SE | GT" in u.text for u in units[1:])


def banded_table(*rows: tuple[str, ...]) -> Component:
    """A table whose rows of one cell are bands spanning it and whose rows starting with
    an empty cell are header rows; other rows are a header cell then data cells."""
    cells: list[TableCell] = []
    for r, row in enumerate(rows):
        if len(row) == 1:
            cells.append(TableCell(row=r, col=0, text=row[0], header=True, col_span=3))
            continue
        header_row = row[0] == ""
        for c, text in enumerate(row):
            cells.append(TableCell(row=r, col=c, text=text, header=header_row or c == 0))
    return Component.model_validate(
        {
            "id": "t",
            "type": "table",
            "text": "\n".join(" | ".join(row) for row in rows),
            "cells": cells,
            "children": [comp("caption", "Kestrova specification", "cap")],
            "location": DomLocation(dom_path="/t"),
        }
    )


def test_an_oversized_tables_mid_table_bands_stay_with_the_rows_they_group() -> None:
    perf = [(f"Power {r}", f"{r}0 PS", f"{r}5 PS") for r in range(1, 9)]
    econ = [(f"MPG {r}", f"{r}1", f"{r}2") for r in range(1, 9)]
    t = banded_table(("", "SE", "GT"), ("Performance",), *perf, ("Economy",), *econ)
    units = gate_units(comp("section", "", "r", t), max_chars=120)
    texts = [u.text for u in units]
    assert texts == [
        "Kestrova specification\nSE | GT\nPerformance\nPower 1 | 10 PS | 15 PS\n"
        "Power 2 | 20 PS | 25 PS\nPower 3 | 30 PS | 35 PS",
        "Kestrova specification\nSE | GT\nPerformance\nPower 4 | 40 PS | 45 PS\n"
        "Power 5 | 50 PS | 55 PS\nPower 6 | 60 PS | 65 PS",
        "Kestrova specification\nSE | GT\nPerformance\nPower 7 | 70 PS | 75 PS\n"
        "Power 8 | 80 PS | 85 PS\nEconomy\nMPG 1 | 11 | 12",
        "Kestrova specification\nSE | GT\nEconomy\nMPG 2 | 21 | 22\nMPG 3 | 31 | 32\n"
        "MPG 4 | 41 | 42\nMPG 5 | 51 | 52\nMPG 6 | 61 | 62",
        "Kestrova specification\nSE | GT\nEconomy\nMPG 7 | 71 | 72\nMPG 8 | 81 | 82",
    ]
    assert all(len(text) <= 120 for text in texts)


def test_an_oversized_tables_leading_bands_and_repeated_header_row_read_as_its_cells() -> None:
    rows = [(f"Power {r}", f"{r}0 PS", f"{r}5 PS") for r in range(1, 5)]
    t = banded_table(
        ("Technical data",), ("", "SE", "GT"), ("Performance",), *rows, ("", "SE L", "R"), *rows
    )
    units = gate_units(comp("section", "", "r", t), max_chars=110)
    texts = [u.text for u in units]
    assert texts[0].startswith(
        "Kestrova specification\nTechnical data\nSE | GT\nPerformance\nPower 1 |"
    )
    # Later pieces repeat the band in force, not the title band above the header row, and
    # the header row repeated mid-table replaces the leading one from there on.
    assert texts[1].startswith("Kestrova specification\nSE | GT\nPerformance\nPower ")
    assert texts[-1].startswith("Kestrova specification\nSE L | R\nPerformance\nPower ")
    assert "SE L | R" not in texts[0]
    assert all(len(text) <= 110 for text in texts)


def test_an_oversized_tables_context_over_half_a_piece_is_not_repeated() -> None:
    rows = [(f"Power {r}", f"{r}0 PS", f"{r}5 PS") for r in range(1, 9)]
    t = banded_table(("", "SE " + "x" * 60, "GT"), *rows)
    units = gate_units(comp("section", "", "r", t), max_chars=120)
    lines = [line for u in units for line in u.text.split("\n")]
    assert len(units) > 1
    assert lines == [
        "Kestrova specification",
        f"SE {'x' * 60} | GT",
        *(" | ".join(row) for row in rows),
    ]
    assert all(len(u.text) <= 120 for u in units)


def test_an_oversized_list_is_split_by_items() -> None:
    items = [comp("list_item", f"Feature number {i}", f"li{i}") for i in range(20)]
    units = gate_units(comp("section", "", "r", comp("list", "", "l", *items)), max_chars=100)
    assert len(units) > 1
    assert all(len(u.text) <= 100 for u in units)
    assert all(u.component_ids == ("l", *(f"li{i}" for i in range(20))) for u in units)
    lines = [line for u in units for line in u.text.split("\n")]
    assert lines == [f"Feature number {i}" for i in range(20)]


def test_a_heading_before_a_long_paragraph_leads_its_first_piece() -> None:
    root = comp(
        "section", "", "r", comp("heading", "H" * 50, "h"), comp("paragraph", "p " * 990, "p")
    )
    units = gate_units(root)
    assert all(u.component_ids == ("h", "p") for u in units)
    assert units[0].text.startswith("H" * 50 + "\n")
    assert all(len(u.text) <= 2000 for u in units)


def test_many_leading_headings_get_their_own_unit_and_no_unit_is_oversized() -> None:
    headings = [comp("heading", f"Heading number {i} " * 2, f"h{i}") for i in range(40)]
    root = comp("section", "", "r", *headings, table("t", 300))
    units = gate_units(root)
    assert all(len(u.text) <= 2000 for u in units)
    assert units[0].component_ids == tuple(f"h{i}" for i in range(40))
    assert all(u.component_ids == ("t",) for u in units[1:])


def test_headings_join_the_table_or_section_they_introduce() -> None:
    root = comp(
        "section",
        "",
        "root",
        comp("heading", "Delmaro Kestrova", "h1"),
        comp("section", "", "s", comp("paragraph", "A roomy hatchback.", "p")),
        comp("heading", "Specifications", "h2"),
        table("t", 2),
        comp("heading", "Trailing", "h3"),
    )
    units = gate_units(root)
    assert [(u.id, u.component_ids) for u in units] == [
        ("s#0", ("h1", "p")),
        ("t", ("h2", "t")),
        ("root#end", ("h3",)),
    ]
    assert units[0].text == "Delmaro Kestrova\nA roomy hatchback."
    assert units[1].text.startswith("Specifications\nTrim | Price")


def test_empty_blocks_make_no_unit_and_a_bare_root_is_one_unit() -> None:
    assert gate_units(comp("section", "", "r", comp("paragraph", "  ", "p"))) == []
    [unit] = gate_units(comp("paragraph", "Just text.", "p"))
    assert unit.component_ids == ("p",)


def scanned_page(alt: str = "") -> Component:
    """An image as the image stage leaves it: OCR paragraphs and headed sections below it."""
    return comp(
        "image",
        alt,
        "img",
        comp("paragraph", "Delmaro Kestrova brochure", "img-t0", trail=["Brochure"]),
        comp(
            "section",
            "",
            "img-t1",
            comp("heading", "Performance", "img-t2", trail=["Brochure"]),
            comp("paragraph", "0-62 mph: 9.1 s", "img-t3", trail=["Brochure", "Performance"]),
            comp("paragraph", "Power: 110 kW", "img-t4", trail=["Brochure", "Performance"]),
            trail=["Brochure"],
        ),
        comp(
            "section",
            "",
            "img-t5",
            comp("heading", "Price", "img-t6", trail=["Brochure"]),
            comp("paragraph", "From £24,995 on the road.", "img-t7", trail=["Brochure", "Price"]),
            trail=["Brochure"],
        ),
        trail=["Brochure"],
    )


def test_an_image_with_text_read_from_it_is_cut_like_a_section() -> None:
    root = comp(
        "section",
        "",
        "root",
        comp("paragraph", "Download the brochure.", "p1"),
        scanned_page(alt="Brochure page 1"),
        comp("paragraph", "Book a test drive today.", "p2"),
    )
    units = gate_units(root)
    assert [(u.id, u.component_ids) for u in units] == [
        ("root#0", ("p1",)),
        ("img#0", ("img", "img-t0")),
        ("img-t1#0", ("img-t2", "img-t3", "img-t4")),
        ("img-t5#0", ("img-t6", "img-t7")),
        ("root#1", ("p2",)),
    ]
    assert units[1].text == "Brochure page 1\nDelmaro Kestrova brochure"
    assert units[2].state() == {
        "content": "Performance\n0-62 mph: 9.1 s\nPower: 110 kW",
        "section": "Brochure",
    }


def test_an_images_ocr_headings_reach_the_section_of_units_below_them() -> None:
    units = gate_units(comp("section", "", "root", scanned_page()), max_chars=30)
    perf = [u for u in units if "img-t3" in u.component_ids or "img-t4" in u.component_ids]
    assert [(u.component_ids, u.state()) for u in perf] == [
        (("img-t2", "img-t3"), {"content": "Performance\n0-62 mph: 9.1 s", "section": "Brochure"}),
        (("img-t4",), {"content": "Power: 110 kW", "section": "Brochure › Performance"}),
    ]
    assert all(len(u.text) <= 30 for u in units)


def test_a_heading_before_a_read_image_opens_its_first_unit() -> None:
    root = comp("section", "", "root", comp("heading", "Specifications", "h"), scanned_page())
    first = gate_units(root)[0]
    assert (first.id, first.component_ids) == ("img#0", ("h", "img-t0"))
    assert first.text == "Specifications\nDelmaro Kestrova brochure"


def test_an_image_alone_at_the_root_is_cut_like_a_section() -> None:
    units = gate_units(scanned_page())
    assert [u.component_ids for u in units] == [
        ("img-t0",),
        ("img-t2", "img-t3", "img-t4"),
        ("img-t6", "img-t7"),
    ]


def test_an_image_without_text_read_from_it_is_one_block_as_before() -> None:
    root = comp(
        "section",
        "",
        "root",
        comp("paragraph", "The Kestrova.", "p1"),
        comp("image", "Side view", "img1"),
        comp("image", "Front view", "img2", comp("caption", "Figure 2: the grille", "cap")),
        comp("image", "", "img3"),
        comp("paragraph", "Book a test drive.", "p2"),
    )
    [unit] = gate_units(root)
    assert unit.component_ids == ("p1", "img1", "img2", "cap", "img3", "p2")
    assert unit.text == (
        "The Kestrova.\nSide view\nFront view\nFigure 2: the grille\nBook a test drive."
    )


def test_a_pdf_figure_with_text_found_in_it_is_cut_like_a_section() -> None:
    figure = comp(
        "image",
        "",
        "f",
        comp("caption", "Figure 3: prices", "cap"),
        comp("paragraph", "Prices exclude VAT.", "fn"),
    )
    root = comp(
        "section",
        "",
        "root",
        comp("paragraph", "The Kestrova.", "p1"),
        figure,
        comp("paragraph", "Book a test drive.", "p2"),
    )
    assert [(u.id, u.component_ids) for u in gate_units(root)] == [
        ("root#0", ("p1",)),
        ("f#0", ("cap", "fn")),
        ("root#1", ("p2",)),
    ]


def read_inline_image(cid: str = "img") -> Component:
    """An inline image the image stage read: a paragraph and a headed section found in it."""
    return comp(
        "image",
        "Brochure",
        cid,
        comp("paragraph", "Delmaro Kestrova brochure", f"{cid}-t0"),
        comp(
            "section",
            "",
            f"{cid}-t1",
            comp("heading", "Price", f"{cid}-t2"),
            comp("paragraph", "From £24,995 on the road.", f"{cid}-t3", trail=["Price"]),
        ),
    )


def test_a_read_image_in_a_paragraph_is_lifted_out_after_the_paragraphs_own_text() -> None:
    root = comp(
        "section",
        "",
        "root",
        comp("paragraph", "The Kestrova.", "p1"),
        comp("paragraph", "See the brochure below.", "p", read_inline_image()),
        comp("paragraph", "Book a test drive.", "p2"),
    )
    units = gate_units(root)
    assert [(u.id, u.component_ids) for u in units] == [
        ("root#0", ("p1", "p")),
        ("img#0", ("img", "img-t0")),
        ("img-t1#0", ("img-t2", "img-t3")),
        ("root#1", ("p2",)),
    ]
    assert units[0].text == "The Kestrova.\nSee the brochure below."
    assert units[1].text == "Brochure\nDelmaro Kestrova brochure"


def test_a_read_image_in_a_list_item_is_lifted_out_after_the_list() -> None:
    items = comp(
        "list",
        "",
        "l",
        comp("list_item", "Power: 110 kW", "li1"),
        comp("list_item", "Brochure:", "li2", read_inline_image()),
    )
    root = comp("section", "", "root", comp("paragraph", "The Kestrova.", "p1"), items)
    units = gate_units(root)
    assert [(u.id, u.component_ids) for u in units] == [
        ("root#0", ("p1", "l", "li1", "li2")),
        ("img#0", ("img", "img-t0")),
        ("img-t1#0", ("img-t2", "img-t3")),
    ]
    assert units[0].text == "The Kestrova.\nPower: 110 kW\nBrochure:"


def test_a_block_holding_only_a_read_image_gives_no_unit_of_its_own() -> None:
    root = comp("section", "", "root", comp("paragraph", "", "p", read_inline_image()))
    assert [u.component_ids for u in gate_units(root)] == [
        ("img", "img-t0"),
        ("img-t2", "img-t3"),
    ]


def test_a_root_block_holding_a_read_image_is_cut_into_its_text_and_the_image() -> None:
    root = comp("paragraph", "See the brochure below.", "p", read_inline_image())
    assert [(u.id, u.component_ids) for u in gate_units(root)] == [
        ("p#0", ("p",)),
        ("img#0", ("img", "img-t0")),
        ("img-t1#0", ("img-t2", "img-t3")),
    ]


def test_only_images_are_lifted_out_of_a_block_not_containers_inside_it() -> None:
    card = comp(
        "section",
        "",
        "card",
        comp("paragraph", "Sharp Objects £47.82", "cp"),
        comp("paragraph", "", "cq", read_inline_image()),
    )
    root = comp("section", "", "root", comp("list", "", "l", comp("list_item", "", "li", card)))
    assert [u.component_ids for u in gate_units(root)] == [
        ("l", "li", "card", "cp", "cq"),
        ("img", "img-t0"),
        ("img-t2", "img-t3"),
    ]


def test_an_inline_image_without_text_read_from_it_is_gated_as_before() -> None:
    figure = comp("image", "Front view", "img2", comp("caption", "Figure 2: the grille", "cap"))
    root = comp(
        "section",
        "",
        "root",
        comp("paragraph", "The Kestrova.", "p1", comp("image", "Side view", "img1")),
        comp("list", "", "l", comp("list_item", "Front:", "li", figure)),
    )
    [unit] = gate_units(root)
    assert unit.component_ids == ("p1", "img1", "l", "li", "img2", "cap")
    assert unit.text == "The Kestrova.\nSide view\nFront:\nFront view\nFigure 2: the grille"


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
    groups = result.components["Car"]
    # Passing units bring their ancestors, in reading order.
    assert groups["price"] == ["root", "s-price", "h2", "p2", "t1"]
    assert groups["performance"] == ["root", "s-perf", "h3", "l1", "li1", "li2"]


async def test_a_value_past_max_chars_still_passes_its_component() -> None:
    text = " ".join(f"Sentence {i} is filler." for i in range(200)) + " It costs £24,995."
    root = comp("section", "", "root", comp("paragraph", text, "p"))
    assert len(text) > 2000
    fake = FakeJev().noul("contain the price (GBP)?", p=0.9, state="24,995")
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    assert result.components["Car"]["price"] == ["root", "p"]


async def test_a_scanned_page_passes_only_the_ocr_text_jev_says_contains_the_field() -> None:
    fake = FakeJev().noul("contain the price (GBP)?", p=0.9, state="24,995")
    root = comp("section", "", "root", scanned_page(alt="Brochure page 1"))
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    # The image passes as the price section's ancestor; the rest of the page doesn't.
    assert result.components["Car"]["price"] == ["root", "img", "img-t5", "img-t6", "img-t7"]
    assert result.components["Car"]["performance"] == []


def with_children(root: Component, cid: str, children: list[Component]) -> Component:
    if root.id == cid:
        return root.model_copy(update={"children": children})
    kids = [with_children(c, cid, children) for c in root.children]
    return root.model_copy(update={"children": kids})


async def test_an_inline_image_passes_only_the_text_read_from_it_that_holds_the_field() -> None:
    html = (
        b"<html><body><p>The Kestrova.</p>"
        b"<p>See the brochure: <img src='b.png' alt='Brochure'></p></body></html>"
    )
    root = await HtmlLayoutParser().parse(Document.from_bytes(html))
    [image] = [c for c in root.walk() if c.type == "image"]
    root = with_children(root, image.id, read_inline_image(image.id).children)
    fake = FakeJev().noul("contain the price (GBP)?", p=0.9, state="24,995")
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    by_id = {c.id: c for c in root.walk()}
    passed = [(by_id[c].type, by_id[c].text) for c in result.components["Car"]["price"]]
    # The paragraph and image pass as the price section's ancestors; the intro doesn't.
    assert passed == [
        ("section", ""),
        ("paragraph", "See the brochure:"),
        ("image", "Brochure"),
        ("section", ""),
        ("heading", "Price"),
        ("paragraph", "From £24,995 on the road."),
    ]


async def test_non_noul_answers_are_an_error() -> None:
    class ChoiceBackend:
        async def system_one(
            self, state: JSONContent, questions: Mapping[str, Question]
        ) -> JevResponse:
            answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {
                k: ChoiceAnswer(choice="x", confidence=1.0, probabilities={"x": 1.0})
                for k in questions
            }
            return JevResponse(answers=answers, input_tokens=1, model="fake")

    with pytest.raises(UnexpectedAnswerError, match="expected a Noul answer"):
        await NoulComponentGate().gate(
            parsed(page()), [SchemaSpec.from_model(Car)], JevClient(ChoiceBackend())
        )


async def test_all_schemas_questions_about_a_unit_go_in_one_request() -> None:
    fake = FakeJev()
    specs = [SchemaSpec.from_model(Car), SchemaSpec.from_model(Book)]
    await NoulComponentGate().gate(parsed(page()), specs, fake.client())
    assert len(fake.calls) == len(gate_units(page()))
    assert set(fake.calls[0].questions) == {"Car.price", "Car.performance", "Book.title"}
    questions = fake.calls[0].questions
    assert questions["Car.price"].instructions == "Does this section contain the price (GBP)?"
    assert questions["Car.performance"].instructions == (
        "Does this section contain the engine power (kW) or 0-62 mph time (s)?"
    )
    assert questions["Book.title"].instructions == "Is this the book's title?"


async def test_threshold_is_inclusive_and_validated() -> None:
    fake = FakeJev(default_p=0.3)
    result = await NoulComponentGate(threshold=0.3).gate(
        parsed(page()), [SchemaSpec.from_model(Book)], fake.client()
    )
    assert "p3" in result.components["Book"]["title"]
    strict = await NoulComponentGate(threshold=0.31).gate(
        parsed(page()), [SchemaSpec.from_model(Book)], fake.client()
    )
    assert strict.components["Book"]["title"] == []
    with pytest.raises(ValueError, match="threshold"):
        NoulComponentGate(threshold=1.5)
    with pytest.raises(ValueError, match="max_chars"):
        NoulComponentGate(max_chars=0)


TABLE_HEADERS = "Does the table's first row name its columns, and its first column name its rows?"
TABLE_LABELS = "Does the table's first column label the value beside it in each row?"


def headers_asked(fake: FakeJev) -> list[tuple[str, JSONContent]]:
    """The table questions asked: (key, instructions)."""
    return [
        (key, q.instructions)
        for call in fake.calls
        for key, q in call.questions.items()
        if key.startswith("table ")
    ]


async def test_a_header_less_comparison_table_is_asked_about_its_headers() -> None:
    grid = [("Spec", "SE", "GT"), ("Power", "150 PS", "200 PS"), ("0-62 mph", "9.1", "7.4")]
    root = comp("section", "", "root", comp("paragraph", "From £24,995.", "p"), plain_table(*grid))
    fake = FakeJev().noul(TABLE_HEADERS, p=0.8)
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    assert headers_asked(fake) == [("table t", TABLE_HEADERS)]
    # In the table's own request, with its groups.
    [call] = [c for c in fake.calls if "table t" in c.questions]
    assert set(call.questions) == {"Car.price", "Car.performance", "table t"}
    assert call.state == {
        "content": "Spec | SE | GT\nPower | 150 PS | 200 PS\n0-62 mph | 9.1 | 7.4"
    }
    assert result.headed_tables == {"t"}
    # Asking doesn't pass the table for any group.
    assert result.components["Car"] == {"price": [], "performance": []}


async def test_a_no_or_unsure_answer_leaves_the_table_without_headers() -> None:
    grid = [("Gearbox", "Manual", "Auto"), ("Power", "150 PS", "200 PS")]
    root = comp("section", "", "root", plain_table(*grid))
    for p in (0.1, 0.49):
        fake = FakeJev().noul(TABLE_HEADERS, p=p)
        result = await NoulComponentGate().gate(
            parsed(root), [SchemaSpec.from_model(Car)], fake.client()
        )
        assert headers_asked(fake) == [("table t", TABLE_HEADERS)]
        assert result.headed_tables == frozenset()
    fake = FakeJev().noul(TABLE_HEADERS, p=0.5)
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    assert result.headed_tables == {"t"}


async def test_a_header_less_two_column_table_is_asked_whether_it_holds_labels() -> None:
    root = comp(
        "section",
        "",
        "root",
        plain_table(("Engine", "1.5 TSI"), ("Power", "150 PS"), cid="kv"),
        plain_table(("Smith", "London"), ("Jones", "Leeds"), cid="people"),
    )
    fake = (
        FakeJev().noul(TABLE_LABELS, p=0.9, state="Engine").noul(TABLE_LABELS, p=0.1, state="Smith")
    )
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    assert headers_asked(fake) == [("table kv", TABLE_LABELS), ("table people", TABLE_LABELS)]
    assert result.headed_tables == {"kv"}


async def test_tables_with_headers_or_without_a_header_shape_are_not_asked_about() -> None:
    headed = Component.model_validate(
        {
            "id": "th",
            "type": "table",
            "text": "Spec | SE\nPower | 150 PS",
            "cells": [
                {"row": 0, "col": 0, "text": "Spec", "header": True},
                {"row": 0, "col": 1, "text": "SE", "header": True},
                {"row": 1, "col": 0, "text": "Power"},
                {"row": 1, "col": 1, "text": "150 PS"},
            ],
            "location": DomLocation(dom_path="/th"),
        }
    )
    records = plain_table(("Name", "City", "Role"), ("Alice", "London", "Engineer"), cid="rec")
    years = plain_table(("2019", "150 PS"), ("2021", "163 PS"), cid="years")
    root = comp("section", "", "root", headed, records, years)
    fake = FakeJev(default_p=1.0)
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    assert len(fake.calls) == 3
    assert headers_asked(fake) == []
    assert result.headed_tables == frozenset()


async def test_an_oversized_header_less_table_is_asked_about_with_its_first_piece() -> None:
    root = comp("section", "", "root", plain_table(*spec_grid(30)))
    fake = FakeJev().noul(TABLE_HEADERS, p=0.9)
    gate = NoulComponentGate(max_chars=120)
    result = await gate.gate(parsed(root), [SchemaSpec.from_model(Car)], fake.client())
    units = gate_units(root, max_chars=120)
    assert len(fake.calls) == len(units) > 1
    [first] = [c for c in fake.calls if "table t" in c.questions]
    assert first.state == units[0].state()
    assert result.headed_tables == {"t"}


async def test_the_table_question_is_the_first_schemas_and_can_be_overridden() -> None:
    class Spec(BaseModel):
        """A spec sheet."""

        __jevex__ = SchemaConfig(
            table_headers_question="Are the trims across the top and the specs down the side?",
            table_labels_question="Is each row a spec and its value?",
        )

        power_kw: float = Field(description="Engine power", unit="kW")

    root = comp(
        "section",
        "",
        "root",
        plain_table(("Spec", "SE", "GT"), ("Power", "150 PS", "200 PS"), cid="cmp"),
        plain_table(("Engine", "1.5 TSI"), cid="kv"),
    )
    fake = FakeJev()
    specs = [SchemaSpec.from_model(Spec), SchemaSpec.from_model(Car)]
    await NoulComponentGate().gate(parsed(root), specs, fake.client())
    assert headers_asked(fake) == [
        ("table cmp", "Are the trims across the top and the specs down the side?"),
        ("table kv", "Is each row a spec and its value?"),
    ]


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


async def test_the_statements_read_a_header_less_table_as_jev_answered() -> None:
    html = (
        b"<html><body><h1>Kestrova</h1><table>"
        b"<tr><td>Spec</td><td>SE</td><td>GT</td></tr>"
        b"<tr><td>Power</td><td>150 PS</td><td>200 PS</td></tr>"
        b"</table></body></html>"
    )
    root = await HtmlLayoutParser().parse(Document.from_bytes(html))
    for p, expected in [
        (0.9, ["SE", "GT", "Power", "Power · SE: 150 PS", "Power · GT: 200 PS"]),
        (0.2, ["Spec | SE | GT", "Power | 150 PS | 200 PS"]),
    ]:
        fake = FakeJev().noul(TABLE_HEADERS, p=p)
        ctx = context(fake, root)
        await ComponentGateStage().run(ctx)
        await StatementStage().run(ctx)
        assert ctx.parsed is not None
        tables = [s.text for s in ctx.parsed.statements.values() if s.table is not None]
        assert tables == expected


async def test_the_resolver_sees_only_gated_components() -> None:
    seen: list[list[str]] = []

    class Recording:
        async def resolve(
            self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
        ) -> list[EntityScope]:
            ids = [c.id for c in parsed.root.walk()]
            seen.append(ids)
            return [EntityScope(label="doc", component_ids=ids)]

    fake = FakeJev().noul("price", p=0.9, state="24,995")
    ctx = context(fake, page())
    await ComponentGateStage().run(ctx)
    await EntityStage(resolver=Recording()).run(ctx)
    # Car passed the price section; Book passed nothing, so it sees only the root.
    assert sorted(seen) == [["root"], ["root", "s-price", "h2", "p2", "t1"]]


def test_restricted_to_prunes_the_tree_and_keeps_structured_statements() -> None:
    loc = DomLocation(dom_path="/")
    statements = {
        "p1.0": Statement(id="p1.0", text="a", kind="sentence", component_id="p1", location=loc),
        "p2.0": Statement(id="p2.0", text="b", kind="sentence", component_id="p2", location=loc),
        "ld.0": Statement(id="ld.0", text="c", kind="structured", component_id="ld", location=loc),
    }
    doc = ParsedDocument(document=Document.from_bytes(b"<p/>"), root=page(), statements=statements)
    view = doc.restricted_to(["s-price", "h2", "p2"])
    assert [c.id for c in view.root.walk()] == ["root", "s-price", "h2", "p2"]
    assert set(view.statements) == {"p2.0", "ld.0"}
    assert [c.id for c in doc.root.walk()][:3] == ["root", "h1", "p1"]  # the original is untouched


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


# --- nested models ---------------------------------------------------------------------


class Trim(BaseModel):
    """One trim of a car."""

    power_kw: float = Field(description="Engine power", unit="kW")
    price: Decimal = Field(description="Trim price", unit="GBP")


class CarModel(BaseModel):
    """A car model page."""

    name: str = Field(description="Model name")
    trims: list[Trim] = Field(description="Trims")


@dataclass
class TrimsIn:
    """A resolver giving the whole view to one parent and one child."""

    async def resolve(
        self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
    ) -> list[EntityScope]:
        ids = [c.id for c in parsed.root.walk()]
        return [
            EntityScope(label="doc", component_ids=ids),
            EntityScope(label="SE", component_ids=ids, parent="doc", field="trims"),
        ]


async def test_nested_models_are_gated_per_field_in_the_same_requests() -> None:
    fake = (
        FakeJev()
        .noul("engine power (kW)?", p=0.9, state="Power: 110 kW")
        .noul("trim price (GBP)?", p=0.9, state="SE | £24,995")
    )
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(CarModel)], fake.client()
    )
    ctx.parsed = parsed(page())
    await ComponentGateStage().run(ctx)

    assert len(fake.calls) == len(gate_units(page()))
    questions = {k: q.instructions for k, q in fake.calls[0].questions.items()}
    assert questions == {
        "CarModel.name": "Does this section contain the model name?",
        "CarModel.trims": (
            "Does this section contain the engine power (kW) or trim price (GBP) of the trims?"
        ),
        "CarModel.trims.power_kw": "Does this section contain the engine power (kW)?",
        "CarModel.trims.price": "Does this section contain the trim price (GBP)?",
    }
    run = ctx.schemas["CarModel"]
    assert run.child_component_ids == {
        "trims": {
            "power_kw": ["root", "s-perf", "h3", "l1", "li1", "li2"],
            "price": ["root", "t1"],
        }
    }
    # Jev said no to "the trims" everywhere, but what passed a nested field reaches the
    # trims group, so the resolver sees it.
    assert run.component_ids == {
        "name": [],
        "trims": ["root", "s-perf", "h3", "l1", "li1", "li2", "t1"],
    }

    await EntityStage(resolver=TrimsIn()).run(ctx)
    child = ctx.schemas["CarModel.trims"]
    assert child.component_ids == run.child_component_ids["trims"]
    assert [f.name for f in child.relevant_fields("li1")] == ["power_kw"]
    assert [f.name for f in child.relevant_fields("t1")] == ["price"]
    assert child.relevant_fields("p3") == []


async def test_a_nested_field_keeps_what_passed_its_own_question() -> None:
    fake = FakeJev().noul("of the trims?", p=0.9, state="Book a test drive")
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(CarModel)], fake.client()
    )
    ctx.parsed = parsed(page())
    await ComponentGateStage().run(ctx)
    run = ctx.schemas["CarModel"]
    assert run.component_ids is not None
    assert "p3" in run.component_ids["trims"]
    assert run.child_component_ids == {"trims": {"power_kw": [], "price": []}}


async def test_a_gate_that_ignores_nested_models_leaves_the_child_run_ungated() -> None:
    asked: list[list[str]] = []

    class ParentsOnly:
        async def gate(
            self, parsed: ParsedDocument, schemas: list[SchemaSpec], jev: JevClient
        ) -> ComponentGateResult:
            asked.append([s.name for s in schemas])
            ids = [c.id for c in parsed.root.walk()]
            return ComponentGateResult(components={"CarModel": {"name": ids, "trims": ids}})

    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(CarModel)], FakeJev().client()
    )
    ctx.parsed = parsed(page())
    await ComponentGateStage(gate=ParentsOnly()).run(ctx)
    assert asked == [["CarModel", "CarModel.trims"]]
    assert ctx.schemas["CarModel"].child_component_ids == {}
    await EntityStage(resolver=TrimsIn()).run(ctx)
    child = ctx.schemas["CarModel.trims"]
    assert child.component_ids is None
    assert [f.name for f in child.relevant_fields("li1")] == ["power_kw", "price"]


async def test_a_dotted_group_does_not_collide_with_a_nested_models_group() -> None:
    class Listing(BaseModel):
        """A car model listing."""

        list_price: Decimal = Field(description="List price", unit="GBP", group="trims.price")
        trims: list[Trim] = Field(description="Trims")

    fake = (
        FakeJev()
        .noul("list price (GBP)?", p=0.9, state="Book a test drive")
        .noul("trim price (GBP)?", p=0.9, state="SE | £24,995")
    )
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(Listing)], fake.client()
    )
    ctx.parsed = parsed(page())
    await ComponentGateStage().run(ctx)

    questions = {k: q.instructions for k, q in fake.calls[0].questions.items()}
    assert questions == {
        "Listing.trims.price": "Does this section contain the list price (GBP)?",
        "Listing.trims": (
            "Does this section contain the engine power (kW) or trim price (GBP) of the trims?"
        ),
        "Listing.trims.power_kw": "Does this section contain the engine power (kW)?",
        "Listing.trims.price#2": "Does this section contain the trim price (GBP)?",
    }
    run = ctx.schemas["Listing"]
    assert run.component_ids is not None
    assert run.component_ids["trims.price"] == ["root", "p3"]
    assert run.child_component_ids["trims"]["price"] == ["root", "t1"]


def test_a_taken_question_key_gets_a_free_suffix() -> None:
    taken = {"A.b": 1, "A.b#2": 2}
    assert _question_key({}, "A.b") == "A.b"
    assert _question_key(taken, "A.b") == "A.b#3"
    assert _question_key(taken, "A.b#2") == "A.b#2#2"


async def test_a_nested_model_jevex_cannot_extract_is_not_gated() -> None:
    class Odd(BaseModel):
        tags: dict[str, str] = Field(description="Tags")

    class Reserved(BaseModel):
        none: str = Field(description="Nothing")

    class Page(BaseModel):
        name: str = Field(description="Model name")
        odd: Odd = Field(description="Odd bits")
        reserved: Reserved = Field(description="Reserved bits")

    fake = FakeJev()
    ctx = Context.create(Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(Page)], fake.client())
    ctx.parsed = parsed(page())
    await ComponentGateStage().run(ctx)
    assert set(fake.calls[0].questions) == {"Page.name", "Page.odd", "Page.reserved"}
    assert fake.calls[0].questions["Page.odd"].instructions == (
        "Does this section contain the odd bits?"
    )
    assert ctx.schemas["Page"].child_component_ids == {}


# --- groups embedded data already filled ---------------------------------------------


def found_price(fake: FakeJev, *models: type[BaseModel]) -> Context:
    """A context where an earlier route (embedded data) found the car's price."""
    specs = [SchemaSpec.from_model(m) for m in models or (Car,)]
    ctx = Context.create(Document.from_bytes(b"<p/>"), specs, fake.client())
    ctx.parsed = parsed(page())
    ctx.schemas["Car"].set_field(
        SINGLE_ENTITY_LABEL, "price", FieldMeta(value=Decimal(24995), method="structured")
    )
    return ctx


def gate_questions(fake: FakeJev) -> dict[str, JSONContent]:
    return {k: q.instructions for k, q in fake.calls[0].questions.items()}


async def test_a_single_entity_pipeline_does_not_gate_groups_already_found() -> None:
    fake = FakeJev().noul("engine power", p=0.9, state="Power: 110 kW")
    ctx = found_price(fake)
    await Pipeline([ComponentGateStage(), EntityStage()]).run(ctx)

    assert len(fake.calls) == len(gate_units(page()))
    assert gate_questions(fake) == {
        "Car.performance": "Does this section contain the engine power (kW) or 0-62 mph time (s)?"
    }
    run = ctx.schemas["Car"]
    assert run.ungated_groups == {"price"}
    assert run.component_ids == {"performance": ["root", "s-perf", "h3", "l1", "li1", "li2"]}
    [event] = [e for e in ctx.events if e.kind == "groups_not_gated"]
    assert event.message == "Car: price already found; not gated"
    assert event.data == {"schema": "Car", "groups": ["price"]}
    # The found field stays a categorise option wherever another group passed, so "From
    # £24,995" in a performance section isn't put down to power.
    assert [f.name for f in run.relevant_fields("li1")] == ["price", "power_kw", "zero_to_62_s"]
    assert run.relevant_fields("p3") == []


async def test_other_resolvers_still_gate_groups_already_found() -> None:
    resolvers = [
        MultiEntity(),
        ParentChild(field="trims", children="table_columns"),
        SingleEntity(label="car"),  # values found on "document" wouldn't reach its record
    ]
    for resolver in resolvers:
        fake = FakeJev()
        ctx = found_price(fake)
        await Pipeline([ComponentGateStage(), EntityStage(resolver=resolver)]).run(ctx)
        assert set(fake.calls[0].questions) == {"Car.price", "Car.performance"}, resolver
        assert ctx.schemas["Car"].ungated_groups == set()
        assert "groups_not_gated" not in [e.kind for e in ctx.events]


async def test_found_groups_are_gated_without_a_pipeline_or_entity_stage() -> None:
    fake = FakeJev()
    await ComponentGateStage().run(found_price(fake))  # run by hand: no pipeline to check
    fake2 = FakeJev()
    await Pipeline([ComponentGateStage()]).run(found_price(fake2))
    for f in (fake, fake2):
        assert set(f.calls[0].questions) == {"Car.price", "Car.performance"}


async def test_skip_found_overrides_the_resolver_check() -> None:
    fake = FakeJev()
    await ComponentGateStage(skip_found=True).run(found_price(fake))
    assert set(fake.calls[0].questions) == {"Car.performance"}
    fake = FakeJev()
    ctx = found_price(fake)
    await Pipeline([ComponentGateStage(skip_found=False), EntityStage()]).run(ctx)
    assert set(fake.calls[0].questions) == {"Car.price", "Car.performance"}


async def test_a_group_is_gated_while_any_of_its_fields_is_still_needed() -> None:
    fake = FakeJev()
    ctx = found_price(fake)
    run = ctx.schemas["Car"]
    run.set_field(SINGLE_ENTITY_LABEL, "power_kw", FieldMeta(value=110.0, method="structured"))
    run.set_field(SINGLE_ENTITY_LABEL, "zero_to_62_s", FieldMeta(value=None, method="structured"))
    await ComponentGateStage(skip_found=True).run(ctx)
    # 0-62 wasn't found (an empty value isn't a find), so its group is still asked about.
    assert set(fake.calls[0].questions) == {"Car.performance"}


async def test_merge_mode_gates_every_group() -> None:
    fake = FakeJev()
    ctx = found_price(fake)
    ctx.schemas["Car"].merge = True
    await ComponentGateStage(skip_found=True).run(ctx)
    assert set(fake.calls[0].questions) == {"Car.price", "Car.performance"}


async def test_other_schemas_are_still_gated_in_the_same_requests() -> None:
    fake = FakeJev()
    await ComponentGateStage(skip_found=True).run(found_price(fake, Car, Book))
    assert len(fake.calls) == len(gate_units(page()))
    assert gate_questions(fake) == {
        "Car.performance": "Does this section contain the engine power (kW) or 0-62 mph time (s)?",
        "Book.title": "Is this the book's title?",
    }


async def test_a_found_nested_field_drops_its_models_questions_too() -> None:
    fake = FakeJev()
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(CarModel)], fake.client()
    )
    ctx.parsed = parsed(page())
    run = ctx.schemas["CarModel"]
    trims = [{"power_kw": 110.0, "price": Decimal(24995)}]
    run.set_field(SINGLE_ENTITY_LABEL, "trims", FieldMeta(value=trims, method="structured"))
    await ComponentGateStage(skip_found=True).run(ctx)
    assert gate_questions(fake) == {"CarModel.name": "Does this section contain the model name?"}
    assert run.child_component_ids == {}
    assert run.component_ids == {"name": []}


async def test_with_every_group_found_nothing_is_asked_and_nothing_is_reported_missing() -> None:
    fake = FakeJev(strict=True)
    ctx = found_price(fake)
    run = ctx.schemas["Car"]
    for name in ("power_kw", "zero_to_62_s"):
        run.set_field(SINGLE_ENTITY_LABEL, name, FieldMeta(value=1.0, method="structured"))
    await ComponentGateStage(skip_found=True).run(ctx)
    assert fake.calls == []
    assert run.component_ids == {}
    assert run.relevant_fields("li1") == []
    assert [e.kind for e in ctx.events] == ["groups_not_gated"]


FILLED_PAGE = b"""<!doctype html><html><head>
<script type="application/ld+json">{"@type": "Car", "offers": {"price": 24995}}</script>
</head><body><main>
  <h1>Delmaro Kestrova 1.5 SE</h1>
  <section><h2>Performance</h2>
    <ul><li>Power: 110 kW</li><li>0-62 mph: 9.1 s</li></ul>
  </section>
  <section><h2>Price</h2><p>On the road from &pound;24,995.</p></section>
</main></body></html>"""


async def test_fill_gaps_with_the_default_resolver_skips_the_gate_for_embedded_values() -> None:
    fake = (
        FakeJev()
        .choice('key path "offers.price"', lambda q: "price" if "price" in q.options else "none")
        .choice('key path "@type"', "none")
        .noul("engine power", p=0.9, state="Power: 110 kW")
    )
    pipeline = Pipeline(
        [
            StructuredStage(mode="fill_gaps"),
            LayoutStage(),
            ComponentGateStage(),
            StatementStage(),
            EntityStage(),
            CategoriseStage(),
        ]
    )
    ex = Extractor([Car], jev=fake.client(), pipeline=pipeline)
    result = await ex.extract(Document.from_bytes(FILLED_PAGE, url="https://cars.test/k"))

    nouls = [c for c in fake.calls if any(isinstance(q, Noul) for q in c.questions.values())]
    assert nouls
    assert {k for c in nouls for k in c.questions} == {"Car.performance"}
    categorised = [c for c in fake.calls if "Car" in c.questions]
    assert categorised
    for call in categorised:
        question = call.questions["Car"]
        assert isinstance(question, Choice)
        assert list(question.options) == ["price", "power_kw", "zero_to_62_s", "none"]
    assert result.one(Car).record.price == Decimal(24995)


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

    assert "0-62 mph: 9.1 s" in texts(result.components["Car"]["performance"])
    assert "On the road from £24,995." in texts(result.components["Car"]["price"])
    newsletter = texts(result.components["Car"]["price"]) | texts(
        result.components["Car"]["performance"]
    )
    assert "Sign up for our weekly deals." not in newsletter

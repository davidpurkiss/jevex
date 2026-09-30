from pathlib import Path

import pytest

from jevex import (
    BoilerplateCleaner,
    Component,
    Document,
    HtmlLayoutParser,
    TableCell,
    UnsupportedDocumentError,
)
from jevex.interfaces import LayoutParser
from jevex.layout import DomLocation
from jevex.layout_html import MAX_COL_SPAN, MAX_COMPONENT_DEPTH, parse_html

FIXTURES = Path(__file__).parent / "fixtures" / "clean"


def path(c: Component) -> str:
    assert isinstance(c.location, DomLocation)
    return c.location.dom_path


def outline(root: Component) -> list[tuple[int, str, str]]:
    """(depth, type, text) for every component below the root, in reading order."""
    out: list[tuple[int, str, str]] = []

    def visit(c: Component, depth: int) -> None:
        for child in c.children:
            out.append((depth, child.type, child.text))
            visit(child, depth + 1)

    visit(root, 0)
    return out


def only(root: Component, kind: str) -> list[Component]:
    return [c for c in root.walk() if c.type == kind]


def html(body: str, *, content_type: str = "text/html") -> Document:
    return Document.from_bytes(
        f"<!DOCTYPE html><html><head><title>t</title></head><body>{body}</body></html>".encode(),
        content_type=content_type,
    )


# --- Block elements ----------------------------------------------------------------------


def test_block_elements_become_paragraphs_and_wrappers_add_no_level() -> None:
    root = parse_html(
        "<div class=wrap><div><p>First  <b>bold</b>\n text.</p>"
        "<div>Engine: <span>1.5</span> TSI</div></div></div>"
    )
    assert root.type == "section"
    assert path(root) == "/html/body"
    assert outline(root) == [
        (0, "paragraph", "First bold text."),
        (0, "paragraph", "Engine: 1.5 TSI"),
    ]
    assert [path(c) for c in root.children] == [
        "/html/body/div/div/p",
        "/html/body/div/div/div",
    ]


def test_loose_text_between_blocks_becomes_its_own_paragraph() -> None:
    root = parse_html("<div>Intro <p>Middle</p> tail <a href=#>link</a></div>")
    assert outline(root) == [
        (0, "paragraph", "Intro"),
        (0, "paragraph", "Middle"),
        (0, "paragraph", "tail link"),
    ]
    paths = [path(c) for c in root.children]
    assert paths == ["/html/body/div", "/html/body/div/p", "/html/body/div"]


def test_line_breaks_are_kept_and_whitespace_collapsed() -> None:
    root = parse_html("<p>  Power:\t150&nbsp;PS<br>Torque: 250 Nm<br><br> </p>")
    assert root.children[0].text == "Power: 150\xa0PS\nTorque: 250 Nm"


def test_pre_keeps_its_whitespace() -> None:
    root = parse_html("<pre>\n  0-62   9.1\n  Top    130\n</pre>")
    assert root.children[0].text == "  0-62   9.1\n  Top    130"


def test_inline_element_holding_a_block_is_read_as_a_wrapper() -> None:
    root = parse_html('<a href="/golf"><div>Golf</div><div>£27,000</div></a>')
    assert outline(root) == [(0, "paragraph", "Golf"), (0, "paragraph", "£27,000")]


def test_unrendered_and_hidden_content_is_skipped() -> None:
    root = parse_html(
        "<p>Shown</p><script>var x = '<p>no</p>';</script><style>p{}</style>"
        "<template><p>tpl</p></template><p hidden>h</p><p aria-hidden='true'>aria</p>"
        "<div style='display: none'><p>css</p></div><select><option>opt</option></select>"
        "<svg><title>icon</title><path d='M0'/><path d='M1'/></svg><p>End</p>"
    )
    # aria-hidden only hides from screen readers; the text is still on screen.
    assert [c.text for c in root.walk() if c.text] == ["Shown", "aria", "End"]


async def test_declarative_shadow_roots_survive_cleaning_and_are_read() -> None:
    doc = html(
        "<div><template shadowrootmode='open'><h2>Shadow title</h2><p>shadow text</p>"
        "</template></div><x-a><template shadowroot='open'><p>legacy</p></template></x-a>"
        "<p>after</p>"
    )
    root = await HtmlLayoutParser().parse(BoilerplateCleaner().clean(doc))
    assert outline(root) == [
        (0, "heading", "Shadow title"),
        (0, "paragraph", "shadow text"),
        (0, "paragraph", "legacy"),
        (0, "paragraph", "after"),
    ]
    assert path(root.children[1]) == "/html/body/div/template/p"


def test_end_tag_br_and_stray_end_tag_p_break_the_text() -> None:
    # Browsers read "</br>" as "<br>", and a "</p>" with no open <p> as an empty <p>.
    assert [c.text for c in parse_html("<p>Line one</br>Line two</p>").walk() if c.text] == [
        "Line one\nLine two"
    ]
    root = parse_html("<div>Engine</p>1.5 TSI</div>")
    assert [c.text for c in root.walk() if c.text] == ["Engine", "1.5 TSI"]


def test_empty_markup_gives_an_empty_root() -> None:
    root = parse_html("")
    assert root.id == "c0"
    assert root.children == []
    assert path(root) == "/html/body"


# --- Headings ----------------------------------------------------------------------------


def test_headings_open_nested_implicit_sections() -> None:
    root = parse_html(
        "<h1>Golf</h1><p>Intro</p>"
        "<div><h2>Specifications</h2><h3>Performance</h3><p>9.1 s</p>"
        "<h3>Economy</h3><p>47.9 mpg</p></div>"
        "<h2>Prices</h2><p>£27k</p>"
    )
    assert outline(root) == [
        (0, "heading", "Golf"),
        (0, "paragraph", "Intro"),
        (0, "section", ""),
        (1, "heading", "Specifications"),
        (1, "section", ""),
        (2, "heading", "Performance"),
        (2, "paragraph", "9.1 s"),
        (1, "section", ""),
        (2, "heading", "Economy"),
        (2, "paragraph", "47.9 mpg"),
        (0, "section", ""),
        (1, "heading", "Prices"),
        (1, "paragraph", "£27k"),
    ]
    # An implicit section sits at its heading.
    assert path(root.children[2]) == "/html/body/div/h2"


def test_heading_trail_on_every_component() -> None:
    root = parse_html(
        "<h1>Golf</h1><h2>Specifications</h2><h3>Performance</h3><p>9.1 s</p>"
        "<h3>Economy</h3><p>47.9 mpg</p><h2>Prices</h2><p>£27k</p>"
    )
    trails = {c.text or c.type: c.heading_trail for c in root.walk() if c.type != "section"}
    assert trails == {
        "Golf": [],
        "Specifications": ["Golf"],
        "Performance": ["Golf", "Specifications"],
        "9.1 s": ["Golf", "Specifications", "Performance"],
        "Economy": ["Golf", "Specifications"],
        "47.9 mpg": ["Golf", "Specifications", "Economy"],
        "Prices": ["Golf"],
        "£27k": ["Golf", "Prices"],
    }
    sections = [c for c in root.walk() if c.type == "section" and c is not root]
    # A section's trail is its heading's: the headings above it.
    assert [s.heading_trail for s in sections] == [
        ["Golf"],
        ["Golf", "Specifications"],
        ["Golf", "Specifications"],
        ["Golf"],
    ]


def test_skipped_heading_levels_still_nest() -> None:
    root = parse_html("<h2>Specs</h2><h4>Boot</h4><p>380 l</p><h3>Engine</h3><p>1.5</p>")
    trails = {c.text: c.heading_trail for c in only(root, "paragraph")}
    assert trails == {"380 l": ["Specs", "Boot"], "1.5": ["Specs", "Engine"]}


def test_headings_inside_sectioning_elements_stay_inside() -> None:
    root = parse_html(
        "<h1>Golf</h1><article><h2>Review</h2><p>Good</p></article>"
        "<aside><h2>Related</h2><p>Polo</p></aside><p>Back in the page</p>"
    )
    assert outline(root) == [
        (0, "heading", "Golf"),
        (0, "section", ""),
        (1, "heading", "Review"),
        (1, "paragraph", "Good"),
        (0, "breakout", ""),
        (1, "heading", "Related"),
        (1, "paragraph", "Polo"),
        (0, "paragraph", "Back in the page"),
    ]
    trails = {c.text: c.heading_trail for c in only(root, "paragraph")}
    assert trails == {
        "Good": ["Golf", "Review"],
        "Polo": ["Golf", "Related"],
        "Back in the page": ["Golf"],
    }


def test_outer_headings_survive_same_rank_headings_in_sectioning_elements() -> None:
    root = parse_html(
        "<h1>Golf</h1><section><h1>Performance</h1><p>9.1 s</p></section>"
        "<h2>Specs</h2><aside><h2>Related</h2><p>Polo</p></aside><p>Boot: 380 l</p>"
    )
    trails = {c.text: c.heading_trail for c in only(root, "paragraph")}
    assert trails == {
        "9.1 s": ["Golf", "Performance"],
        "Polo": ["Golf", "Specs", "Related"],
        "Boot: 380 l": ["Golf", "Specs"],
    }


def test_a_mismatched_heading_end_tag_still_ends_the_heading() -> None:
    root = parse_html("<h2>Specs</h3><p>Power 150 PS</p><p>Torque</p>")
    assert outline(root) == [
        (0, "heading", "Specs"),
        (0, "paragraph", "Power 150 PS"),
        (0, "paragraph", "Torque"),
    ]


def test_aria_headings_and_empty_headings() -> None:
    root = parse_html(
        "<div role=heading aria-level=3>Performance</div><p>9.1 s</p>"
        "<h2><img src=logo.png></h2><p>after</p>"
    )
    assert only(root, "heading")[0].text == "Performance"
    # An empty heading opens nothing, so "after" stays under Performance.
    assert only(root, "paragraph")[1].heading_trail == ["Performance"]
    assert len(only(root, "heading")) == 1


def test_heading_text_flattens_markup() -> None:
    root = parse_html("<h2>Golf <small>SE&nbsp;L</small><br>2026</h2>")
    assert root.children[0].text == "Golf SE\xa0L 2026"


# --- Lists -------------------------------------------------------------------------------


def test_lists_hold_items_and_nested_lists_are_children() -> None:
    root = parse_html(
        "<ul><li>Petrol<li><p>Diesel</p><p>2.0 TDI</p><ul><li>150 PS</li></ul>"
        "<li><h3>Hybrid</h3>eHybrid</ul><ol><li>First</ol>"
    )
    assert outline(root) == [
        (0, "list", ""),
        (1, "list_item", "Petrol"),
        (1, "list_item", "Diesel\n2.0 TDI"),
        (2, "list", ""),
        (3, "list_item", "150 PS"),
        (1, "list_item", "Hybrid\neHybrid"),
        (0, "list", ""),
        (1, "list_item", "First"),
    ]
    assert [path(c) for c in only(root, "list_item")] == [
        "/html/body/ul/li[1]",
        "/html/body/ul/li[2]",
        "/html/body/ul/li[2]/ul/li",
        "/html/body/ul/li[3]",
        "/html/body/ol/li",
    ]


def test_invalid_children_of_a_list_are_read_as_items() -> None:
    root = parse_html("<ul><li>A</li><ul><li>B</li></ul><div>C</div><li> </li></ul>")
    assert outline(root) == [
        (0, "list", ""),
        (1, "list_item", "A"),
        (1, "list_item", ""),
        (2, "list", ""),
        (3, "list_item", "B"),
        (1, "list_item", "C"),
    ]


def test_definition_lists_give_one_item_per_pair() -> None:
    root = parse_html(
        "<dl><dt>Engine:</dt><dd>1.5 TSI</dd><dd>2.0 TDI</dd>"
        "<div><dt>CO2</dt><dt>Emissions</dt><dd>130 g/km</dd></div>"
        "<dd>orphan value</dd><dt>Warranty</dt><dt>Colour</dt></dl>"
    )
    assert [c.text for c in only(root, "list_item")] == [
        "Engine: 1.5 TSI",
        "Engine: 2.0 TDI",
        "CO2, Emissions: 130 g/km",
        "CO2, Emissions: orphan value",
        "Warranty, Colour",
    ]
    assert [path(c) for c in only(root, "list_item")][2:] == [
        "/html/body/dl/div/dd",
        "/html/body/dl/dd[3]",
        "/html/body/dl/dt[3]",
    ]


def test_empty_lists_are_dropped() -> None:
    assert parse_html("<ul><li> </li></ul><dl><dt></dt></dl><ol></ol>").children == []


# --- Tables ------------------------------------------------------------------------------


def test_tables_give_cells_on_a_grid_with_headers_marked() -> None:
    root = parse_html(
        "<table><caption>Performance</caption>"
        "<thead><tr><td></td><td>1.5 TSI SE</td><th>2.0 TDI SE</th></tr></thead>"
        "<tr><th>0-62 mph (s)<td>9.1<td>8.5"
        "<tr><th>Top speed (mph)</th><td>130</td><td>137</td></tr></table>"
    )
    (table,) = root.children
    assert table.type == "table"
    assert path(table) == "/html/body/table"
    assert table.text == (
        "1.5 TSI SE | 2.0 TDI SE\n0-62 mph (s) | 9.1 | 8.5\nTop speed (mph) | 130 | 137"
    )
    assert table.cells == [
        TableCell(row=0, col=1, text="1.5 TSI SE", header=True),
        TableCell(row=0, col=2, text="2.0 TDI SE", header=True),
        TableCell(row=1, col=0, text="0-62 mph (s)", header=True),
        TableCell(row=1, col=1, text="9.1"),
        TableCell(row=1, col=2, text="8.5"),
        TableCell(row=2, col=0, text="Top speed (mph)", header=True),
        TableCell(row=2, col=1, text="130"),
        TableCell(row=2, col=2, text="137"),
    ]
    assert outline(table) == [(0, "caption", "Performance")]


def test_row_and_column_spans_shift_later_cells() -> None:
    root = parse_html(
        "<table><tr><th rowspan=2>Engine</th><th colspan='2'>Trim</th></tr>"
        "<tr><th>SE</th><th>SE L</th></tr>"
        "<tr><td>1.5</td><td rowspan=0>yes</td><td>no</td></tr>"
        "<tr><td>2.0</td><td>maybe</td></tr></table>"
    )
    cells = {(c.row, c.col): (c.text, c.row_span, c.col_span) for c in root.children[0].cells}
    assert cells == {
        (0, 0): ("Engine", 2, 1),
        (0, 1): ("Trim", 1, 2),
        (1, 1): ("SE", 1, 1),
        (1, 2): ("SE L", 1, 1),
        (2, 0): ("1.5", 1, 1),
        (2, 1): ("yes", 2, 1),  # rowspan=0 runs to the last row
        (2, 2): ("no", 1, 1),
        (3, 0): ("2.0", 1, 1),
        (3, 2): ("maybe", 1, 1),
    }


def headers(markup: str) -> dict[tuple[int, int], bool]:
    (table,) = parse_html(f"<table>{markup}</table>").children
    return {(c.row, c.col): c.header for c in table.cells}


def test_bold_td_labels_in_the_header_row_and_first_column_are_headers() -> None:
    assert headers(
        "<tr><td></td><td><b>SE</b></td><td><strong>GT</strong></td></tr>"
        "<tr><td><b>Power</b></td><td>150</td><td><b>200</b></td></tr>"
        "<tr><td colspan=3><b><span>Economy</span></b></td></tr>"
        "<tr><td><strong>MPG:</strong> </td><td>50</td><td>45</td></tr>"
    ) == {
        (0, 1): True,
        (0, 2): True,
        (1, 0): True,
        (1, 1): False,
        (1, 2): False,  # a bold value outside the header row and first column
        (2, 0): True,
        (3, 0): True,  # a colon after the bold label is still a label
        (3, 1): False,
        (3, 2): False,
    }


def test_bold_labels_need_a_consistent_first_column() -> None:
    # One label in bold isn't a label column: a bold total stays data.
    assert headers(
        "<tr><td>Engine</td><td>1.5 TSI</td></tr><tr><td><b>Total</b></td><td>£25,000</td></tr>"
    ) == {(0, 0): False, (0, 1): False, (1, 0): False, (1, 1): False}
    # A bold header row counts without a label column, and th cells count as labels.
    assert headers(
        "<tr><td><b>Spec</b></td><td><b>SE</b></td></tr>"
        "<tr><td>Power</td><td>150</td></tr>"
        "<tr><th>Torque</th><td><b>250</b></td></tr>"
    ) == {(0, 0): True, (0, 1): True, (1, 0): False, (1, 1): False, (2, 0): True, (2, 1): False}


def test_a_bold_first_value_in_a_label_value_table_is_data() -> None:
    assert headers(
        "<tr><td><b>Engine</b></td><td><b>1.5 TSI</b></td></tr>"
        "<tr><td><b>Power</b></td><td>150</td></tr>"
    ) == {(0, 0): True, (0, 1): False, (1, 0): True, (1, 1): False}
    # With an empty corner, the first row is a column header row.
    assert headers(
        "<tr><td></td><td><b>SE</b></td></tr>"
        "<tr><td><b>Power</b></td><td>150</td></tr>"
        "<tr><td><b>Torque</b></td><td>250</td></tr>"
    ) == {(0, 1): True, (1, 0): True, (1, 1): False, (2, 0): True, (2, 1): False}


def test_a_table_set_all_in_bold_keeps_its_values_as_data() -> None:
    assert headers(
        "<tr><td><b>Engine</b></td><td><b>1.5 TSI</b></td></tr>"
        "<tr><td><b>Power</b></td><td><b>150 PS</b></td></tr>"
    ) == {(0, 0): True, (0, 1): False, (1, 0): True, (1, 1): False}


@pytest.mark.parametrize(
    "content",
    [
        "<b>Engine</b> size",
        "Engine <strong>size</strong>",
        "<b>Engine</b><img src='i.png' alt='size'>",
        "<em>Engine</em>",
    ],
)
def test_cells_only_partly_bold_are_data(content: str) -> None:
    assert headers(
        f"<tr><td>{content}</td><td>1.5 TSI</td></tr><tr><td><b>Power</b></td><td>150</td></tr>"
    ) == {
        (0, 0): False,
        (0, 1): False,
        (1, 0): False,
        (1, 1): False,
    }


def test_spans_are_clamped() -> None:
    root = parse_html(
        "<table><tr><td rowspan=99 colspan=100000>a</td><td>b</td></tr><tr><td>c</td></tr></table>"
    )
    cells = root.children[0].cells
    assert cells[0] == TableCell(row=0, col=0, text="a", row_span=2, col_span=MAX_COL_SPAN)
    assert cells[1].col == MAX_COL_SPAN
    assert cells[2] == TableCell(row=1, col=MAX_COL_SPAN, text="c")


def test_rows_without_tbody_get_the_browser_path() -> None:
    root = parse_html("<table><tr><td>a</td></tr></table><table><td>b</td></table>")
    assert [path(c) for c in root.children] == ["/html/body/table[1]", "/html/body/table[2]"]
    assert [c.cells[0].text for c in root.children] == ["a", "b"]


def test_layout_tables_are_read_as_containers() -> None:
    root = parse_html(
        "<table><tr><td><h2>Specs</h2><table><tr><td>Power</td><td>150 PS</td></tr></table>"
        "</td><td>Sidebar text</td></tr></table>"
        "<table role=presentation><tr><td>Left</td><td>Right</td></tr></table>"
    )
    assert outline(root) == [
        (0, "heading", "Specs"),
        (0, "table", "Power | 150 PS"),
        (0, "paragraph", "Sidebar text"),
        (0, "paragraph", "Left"),
        (0, "paragraph", "Right"),
    ]


@pytest.mark.parametrize(
    "markup",
    [
        "<table><caption>Performance<tr><td>Power</td><td>150 PS</td></tr></table>",
        "<table><colgroup><col span=2><tr><td>Power</td><td>150 PS</td></tr></table>",
        "<table><tr><td>Power</td><td>150 PS</td></tr><caption>Performance</caption></table>",
        "<table><form action=x><tr><td>Power</td><td>150 PS</td></tr></form></table>",
    ],
)
def test_unclosed_captions_colgroups_and_forms_keep_the_rows(markup: str) -> None:
    root = parse_html(markup)
    tables = only(root, "table")
    assert [t.text for t in tables] == ["Power | 150 PS"]
    assert [path(c) for c in tables] == ["/html/body/table"]
    assert [c.text for c in only(root, "caption")] == (
        ["Performance"] if "caption" in markup else []
    )


def test_content_between_rows_moves_in_front_of_the_table() -> None:
    root = parse_html(
        "<p>Before</p><table>Loose text<div>In a <b>div</b><tr><td>Power</td><td>150 PS</td>"
        "</tr>Late</table><p>After</p>"
    )
    assert outline(root) == [
        (0, "paragraph", "Before"),
        (0, "paragraph", "Loose text"),
        (0, "paragraph", "In a div"),
        (0, "paragraph", "Late"),
        (0, "table", "Power | 150 PS"),
        (0, "paragraph", "After"),
    ]
    assert [path(c) for c in root.children] == [
        "/html/body/p[1]",
        "/html/body",
        "/html/body/div",
        "/html/body",
        "/html/body/table",
        "/html/body/p[2]",
    ]


@pytest.mark.parametrize(
    "markup",
    [
        "<table><tbody><template><tr><td>{{ name }}</td></tr></template>"
        "<tr><td>Golf</td></tr></tbody></table>",
        "<table><tr><td>Golf<template><td>{{ name }}</td></template></td></tr></table>",
    ],
)
def test_table_recovery_does_not_reach_into_templates(markup: str) -> None:
    root = parse_html(markup)
    assert outline(root) == [(0, "table", "Golf")]


def test_a_template_end_tag_closes_cells_left_open_in_it() -> None:
    root = parse_html(
        "<table><tbody><template><tr><td>{{ a }}<td>{{ b }}</template>"
        "<tr><td>Golf</td></tr></tbody></table><p>after</p>"
    )
    assert outline(root) == [(0, "table", "Golf"), (0, "paragraph", "after")]


def test_a_table_opened_between_rows_ends_the_open_table() -> None:
    root = parse_html(
        "<table><tr><td>a</td></tr><table><tr><td>b</td></tr></table></table><p>after</p>"
    )
    assert [(c.text, path(c)) for c in root.children] == [
        ("a", "/html/body/table[1]"),
        ("b", "/html/body/table[2]"),
        ("after", "/html/body/p"),
    ]


def test_headings_in_header_cells_keep_a_data_table() -> None:
    root = parse_html("<table><tr><th><h3>Power</h3></th><td>150 PS</td></tr></table>")
    assert outline(root) == [(0, "table", "Power | 150 PS")]


def test_tfoot_rows_come_last_wherever_they_are_written() -> None:
    root = parse_html(
        "<table><thead><tr><th>h</th></tr></thead><tfoot><tr><td>foot</td></tr></tfoot>"
        "<tbody><tr><td>b</td></tr></tbody></table>"
    )
    (table,) = only(root, "table")
    assert table.text == "h\nb\nfoot"
    assert [(c.row, c.text) for c in table.cells] == [(0, "h"), (1, "b"), (2, "foot")]


def test_image_alt_text_is_the_text_of_cells_and_headings() -> None:
    root = parse_html(
        "<h1><img alt='Brand' src=logo.png></h1>"
        "<table><tr><th>ABS</th><td><img src=tick.png alt='Yes'></td>"
        "<td><img src=cross.png alt='No'> (extra)</td></tr></table>"
    )
    assert only(root, "heading")[0].text == "Brand"
    assert [c.text for c in only(root, "table")[0].cells] == ["ABS", "Yes", "No (extra)"]
    assert only(root, "image") == []  # the alt text is already in the text


def test_a_table_without_text_is_dropped_but_keeps_its_caption() -> None:
    root = parse_html("<table><caption>Empty</caption><tr><td> </td></tr></table><table></table>")
    assert outline(root) == [(0, "caption", "Empty")]


def test_stray_end_tags_do_not_escape_a_cell() -> None:
    root = parse_html("<div><table><tr><td>a</div>b</td><td>c</td></tr></table><p>after</p></div>")
    (table, after) = root.children
    assert [c.text for c in table.cells] == ["ab", "c"]
    assert path(after) == "/html/body/div/p"


# --- Figures and images ------------------------------------------------------------------


def test_figure_with_one_image_attaches_its_caption() -> None:
    root = parse_html(
        "<figure><picture><source srcset=a.webp><img src=a.jpg alt=' Golf  in blue '></picture>"
        "<figcaption>Our <em>test</em> car</figcaption></figure>"
    )
    assert outline(root) == [(0, "image", "Golf in blue"), (1, "caption", "Our test car")]
    assert path(root.children[0]) == "/html/body/figure/picture/img"


def test_figure_with_a_table_attaches_its_caption() -> None:
    root = parse_html(
        "<figure><table><tr><td>a</td></tr></table><figcaption>Prices</figcaption></figure>"
    )
    assert outline(root) == [(0, "table", "a"), (1, "caption", "Prices")]


def test_figure_with_mixed_content_is_a_section() -> None:
    root = parse_html(
        "<figure><blockquote>Best in class</blockquote><figcaption>Autocar</figcaption></figure>"
    )
    assert outline(root) == [
        (0, "section", ""),
        (1, "paragraph", "Best in class"),
        (1, "caption", "Autocar"),
    ]


def test_images_carry_alt_text_and_tracking_pixels_are_dropped() -> None:
    root = parse_html(
        "<p>Price <img src=badge.png alt='Best buy'> £27k</p>"
        "<p><img data-src=lazy.jpg></p>"
        "<img src=pixel.gif width=1 height=1><img alt=''><img src=x.png width=0>"
    )
    assert outline(root) == [
        (0, "paragraph", "Price £27k"),
        (1, "image", "Best buy"),
        (0, "image", ""),
    ]
    assert path(root.children[1]) == "/html/body/p[2]/img"


def test_images_carry_their_url_resolved_against_the_document() -> None:
    root = parse_html(
        "<img src=a.png alt=A>"
        "<img src='data:image/gif;base64,R0lGOD' data-src='/img/lazy.jpg' alt=B>"
        "<img srcset='wide.jpg 2x, narrow.jpg 1x' alt=C>"
        "<img src='data:image/png;base64,iVBOR' alt=D>"
        "<img src='https://cdn.example.net/e.png' alt=E>"
        "<img alt=F>",
        base_url="https://example.com/cars/golf.html",
    )
    assert [(c.text, c.src) for c in only(root, "image")] == [
        ("A", "https://example.com/cars/a.png"),
        ("B", "https://example.com/img/lazy.jpg"),
        ("C", "https://example.com/cars/wide.jpg"),
        ("D", "data:image/png;base64,iVBOR"),
        ("E", "https://cdn.example.net/e.png"),
        ("F", None),
    ]
    assert only(parse_html("<img src=a.png>"), "image")[0].src == "a.png"


async def test_parser_resolves_image_urls_against_the_documents_url() -> None:
    document = html("<img src=a.png>").model_copy(update={"url": "https://example.com/x/"})
    root = await HtmlLayoutParser().parse(document)
    assert [c.src for c in only(root, "image")] == ["https://example.com/x/a.png"]


# --- DOM paths and recovery --------------------------------------------------------------


def test_paths_number_only_repeated_tags_and_implied_end_tags() -> None:
    root = parse_html(
        "<main><p>one<p>two<div>three</div><section><h2>A</h2></section><section>x</section>"
    )
    main = root.children[0]
    assert [(c.type, path(c)) for c in main.walk()] == [
        ("section", "/html/body/main"),
        ("paragraph", "/html/body/main/p[1]"),
        ("paragraph", "/html/body/main/p[2]"),
        ("paragraph", "/html/body/main/div"),
        ("section", "/html/body/main/section[1]"),
        ("heading", "/html/body/main/section[1]/h2"),
        ("section", "/html/body/main/section[2]"),
        ("paragraph", "/html/body/main/section[2]"),
    ]


def test_fragments_get_implied_html_and_body() -> None:
    root = parse_html("<title>x</title>Loose text<p>para</p>")
    assert [(c.text, path(c)) for c in root.children] == [
        ("Loose text", "/html/body"),
        ("para", "/html/body/p"),
    ]


def test_ids_follow_reading_order() -> None:
    root = parse_html("<h1>A</h1><ul><li>x</li><li>y</li></ul><p>z</p>")
    assert [c.id for c in root.walk()] == [f"c{i}" for i in range(6)]


def test_unclosed_nesting_is_capped_without_losing_text() -> None:
    # Past MAX_DEPTH new elements become siblings, so later blocks still break the text.
    root = parse_html(
        "<p>before</p>" + "<font>" * 5000 + "deep" + "<div>block</div>text<p>after</p>"
    )
    assert [c.text for c in root.walk() if c.text] == ["before", "deep", "block", "text", "after"]
    assert Component.model_validate_json(root.model_dump_json()) == root


@pytest.mark.parametrize("opener", ["<font>", "<div>"])
@pytest.mark.parametrize("depth", [250, 254, 400])
def test_tables_and_templates_survive_the_depth_cap(opener: str, depth: int) -> None:
    root = parse_html(
        opener * depth + "<table><tr><td>a</td></tr><p>x</p><tr><td>b</td></tr></table>"
        "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"
        "<template><p>secret</p></template>"
        "<ul><li>a<li>b</ul><dl><dt>Colour<dd>Red</dl>"
    )
    first, grid = only(root, "table")
    assert first.text == "a\nb"
    assert grid.text == "1 | 2\n3 | 4"
    assert [(c.row, c.col) for c in grid.cells] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    texts = [c.text for c in root.walk() if c.text]
    assert texts.index("x") < texts.index("a\nb")  # still moved in front of its table
    assert "secret" not in "".join(texts)
    assert [c.text for c in only(root, "list_item")] == ["a", "b", "Colour: Red"]


def test_endlessly_nested_tables_stop_growing_but_keep_their_text() -> None:
    root = parse_html(
        "<table><tr><td>cell" * 500 + "<script>var x</script><template><p>tpl</p></template>"
    )
    assert "var x" not in root.model_dump_json()
    assert "tpl" not in root.model_dump_json()
    # Nested tables are layout tables, so cells' text becomes paragraphs. Past
    # MAX_DEPTH_HARD neighbouring cells can run together, but no text is dropped.
    assert sum(c.text.count("cell") for c in root.walk()) == 500
    assert Component.model_validate_json(root.model_dump_json()) == root


def test_deep_component_trees_are_flattened_at_the_depth_limit() -> None:
    root = parse_html("<ul><li>x" * 100 + "<table><tr><td>t</td></tr></table>")
    depth, node = 0, root
    while node.children:
        node, depth = node.children[0], depth + 1
    assert depth == MAX_COMPONENT_DEPTH
    assert [c.text for c in root.walk() if c.text][-2:] == ["x", "t"]
    assert only(root, "table")[0].cells == [TableCell(row=0, col=0, text="t")]
    assert len([c for c in root.walk() if c.text == "x"]) == 100
    assert Component.model_validate_json(root.model_dump_json()) == root


# --- Parser ------------------------------------------------------------------------------


async def test_parser_reads_html_and_xhtml_documents() -> None:
    parser = HtmlLayoutParser()
    assert isinstance(parser, LayoutParser)
    root = await parser.parse(html("<p>Golf</p>"))
    assert root.children[0].text == "Golf"
    assert parser.supports(html("", content_type="application/xhtml+xml"))


async def test_parser_rejects_other_documents() -> None:
    pdf = Document.from_bytes(b"%PDF-1.7\n", content_type="application/pdf")
    parser = HtmlLayoutParser()
    assert not parser.supports(pdf)
    with pytest.raises(UnsupportedDocumentError, match="application/pdf"):
        await parser.parse(pdf)


async def test_undecodable_bytes_are_read_as_windows_1252() -> None:
    # No charset and not UTF-8: decode_html keeps the bytes as lone surrogates, which
    # JSON can't hold.
    doc = Document.from_bytes(b"<p>caf\xe9 \xa320 \x81</p>", content_type="text/html")
    root = await HtmlLayoutParser().parse(doc)
    assert root.children[0].text == "café £20 \ufffd"
    assert Component.model_validate_json(root.model_dump_json()) == root


async def test_parser_decodes_the_page_charset() -> None:
    doc = Document.from_bytes(
        '<meta charset="windows-1252"><p>Caf\xe9 \xa34</p>'.encode("cp1252"),
        content_type="text/html",
    )
    root = await HtmlLayoutParser().parse(doc)
    assert root.children[0].text == "Café £4"


@pytest.mark.parametrize(
    ("page", "heading", "facts"),
    [
        ("books_catalogue.html", "A Light in the Attic", ["£51.77", "In stock (22 available)"]),
        ("news_article.html", "Council approves new cycle lanes", ["8.4 km", "café"]),
        ("shop_product.html", "Anker PowerCore 10000", ["£21.99", "10,000 mAh", "5 V / 2.4 A"]),
        ("wordpress_post.html", "Our week with the Golf SE L", ["1,498\xa0cc", "47.9 mpg"]),
    ],
)
async def test_cleaned_pages_segment_under_their_title(
    page: str, heading: str, facts: list[str]
) -> None:
    doc = BoilerplateCleaner().clean(Document.from_path(FIXTURES / page))
    root = await HtmlLayoutParser().parse(doc)
    assert heading in [c.text for c in only(root, "heading")]
    for fact in facts:
        holders = [c for c in root.walk() if fact in c.text]
        assert holders, fact
        assert all(c.heading_trail[:1] == [heading] for c in holders), fact

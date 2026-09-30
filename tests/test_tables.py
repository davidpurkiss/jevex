import asyncio

from jevex import (
    BoilerplateCleaner,
    Component,
    DefaultSplitter,
    Document,
    DomLocation,
    SchemaSpec,
)
from jevex.generators import default_registry
from jevex.layout import TableCell
from jevex.layout_html import HtmlLayoutParser
from jevex.tables import table_statements
from jevex.testsite import VehicleSpec, generate, render


def cell(row: int, col: int, text: str, *, header: bool = False, cols: int = 1) -> TableCell:
    return TableCell(row=row, col=col, text=text, header=header, col_span=cols)


def table(*cells: TableCell, trail: list[str] | None = None) -> Component:
    return Component(
        id="t1",
        type="table",
        text="(rows)",
        cells=list(cells),
        heading_trail=trail or [],
        location=DomLocation(dom_path="/html/body/table"),
    )


def texts(component: Component) -> list[str]:
    return [s.text for s in table_statements(component)]


def test_cells_carry_row_and_column_headers() -> None:
    t = table(
        cell(0, 0, "Specification", header=True),
        cell(0, 1, "SE", header=True),
        cell(0, 2, "GT", header=True),
        cell(1, 0, "0-62 mph (s)", header=True),
        cell(1, 1, "9.1"),
        cell(1, 2, "7.4"),
        cell(2, 0, "Price", header=True),
        cell(2, 1, "£24,995"),
        cell(2, 2, " "),  # empty cells are skipped
    )
    assert texts(t) == [
        "0-62 mph (s) · SE: 9.1",
        "0-62 mph (s) · GT: 7.4",
        "Price · SE: £24,995",
    ]


def test_stacked_and_spanning_headers_join_and_bands_group_rows() -> None:
    t = table(
        cell(0, 0, "", header=True),
        cell(0, 1, "1.5 TSI", header=True, cols=2),
        cell(1, 1, "SE", header=True),
        cell(1, 2, "SE L", header=True),
        cell(2, 0, "Performance", header=True, cols=3),
        cell(3, 0, "0-62 mph (s)", header=True),
        cell(3, 1, "9.1"),
        cell(3, 2, "8.9"),
        cell(4, 0, "Economy", header=True, cols=3),
        cell(5, 0, "Combined (mpg)", header=True),
        cell(5, 1, "52.3", cols=2),  # one value for both trims
    )
    assert texts(t) == [
        "Performance › 0-62 mph (s) · 1.5 TSI SE: 9.1",
        "Performance › 0-62 mph (s) · 1.5 TSI SE L: 8.9",
        "Economy › Combined (mpg) · 1.5 TSI SE / 1.5 TSI SE L: 52.3",
    ]
    spanning = table_statements(t)[-1].table
    assert spanning is not None
    assert spanning.col_headers == ["1.5 TSI SE", "1.5 TSI SE L"]
    assert spanning.group == "Economy"


def test_label_value_tables_and_header_only_tables() -> None:
    pairs = table(
        cell(0, 0, "Engine", header=True),
        cell(0, 1, "1.5 TSI"),
        cell(1, 0, "Power", header=True),
        cell(1, 1, "150 PS"),
    )
    assert texts(pairs) == ["Engine: 1.5 TSI", "Power: 150 PS"]
    columns = table(
        cell(0, 0, "Price", header=True),
        cell(0, 1, "Mileage", header=True),
        cell(1, 0, "£9,995"),
        cell(1, 1, "42,000"),
    )
    assert texts(columns) == ["Price: £9,995", "Mileage: 42,000"]


def test_a_table_without_headers_gives_one_statement_per_row() -> None:
    t = table(cell(0, 0, "a"), cell(0, 1, "b"), cell(1, 0, "c"), cell(1, 1, "d"))
    assert texts(t) == ["a | b", "c | d"]
    assert texts(table()) == []


def test_statements_carry_the_tables_context() -> None:
    t = table(
        cell(0, 0, "", header=True),
        cell(0, 1, "SE", header=True),
        cell(1, 0, "Price", header=True),
        cell(1, 1, "£24,995"),
        trail=["Delmaro Kestrova", "Specifications"],
    )
    [s] = table_statements(t)
    assert s.id == "t1.r1c1"
    assert s.kind == "table_cell"
    assert s.component_id == "t1"
    assert s.heading_trail == ["Delmaro Kestrova", "Specifications"]
    assert s.table is not None
    assert (s.table.row, s.table.col, s.table.row_headers) == (1, 1, ["Price"])
    assert DefaultSplitter().split(t) == [s]


def test_rendered_cells_give_the_value_as_a_candidate() -> None:
    [s] = table_statements(
        table(
            cell(0, 1, "SE", header=True),
            cell(1, 0, "0-62 mph (s)", header=True),
            cell(1, 1, "9.1"),
        )
    )
    field = SchemaSpec.from_model(VehicleSpec).field("zero_to_62_s")
    raws = [c.raw for c in default_registry().generate(s, field, schema="VehicleSpec")]
    assert "9.1" in raws


HTML = b"""<html><body><main><h2>Specifications</h2>
<table>
  <thead><tr><th></th><th>SE</th><th>GT</th></tr></thead>
  <tbody>
    <tr><th colspan="3">Performance</th></tr>
    <tr><th>0-62 mph (s)</th><td>9.1</td><td>7.4</td></tr>
    <tr><th colspan="3">Price</th></tr>
    <tr><th>On the road</th><td>&pound;24,995</td><td>&pound;31,250</td></tr>
  </tbody>
</table></main></body></html>"""


def test_html_tables_end_to_end() -> None:
    doc = BoilerplateCleaner().clean(Document.from_bytes(HTML, url="https://cars.test/"))
    root = asyncio.run(HtmlLayoutParser().parse(doc))
    [t] = [c for c in root.walk() if c.type == "table"]
    assert texts(t) == [
        "Performance › 0-62 mph (s) · SE: 9.1",
        "Performance › 0-62 mph (s) · GT: 7.4",
        "Price › On the road · SE: £24,995",
        "Price › On the road · GT: £31,250",
    ]


def test_the_test_sites_spec_tables_give_a_statement_per_trim_and_spec() -> None:
    pages = [p for p in render(generate(42)) if p.family == "table"]
    assert pages
    for p in pages:
        doc = BoilerplateCleaner().clean(Document.from_bytes(p.html.encode(), url="https://x/"))
        root = asyncio.run(HtmlLayoutParser().parse(doc))
        [t] = [c for c in root.walk() if c.type == "table"]
        statements = table_statements(t)
        trims = [r["entity"] for r in p.records]
        by_trim = {
            trim: [s for s in statements if s.table and s.table.col_headers == [trim]]
            for trim in trims
        }
        rows = {s.table.row for s in statements if s.table}
        for trim, mine in by_trim.items():
            assert len(mine) == len(rows), (p.path, trim)  # every spec row, once per trim
        assert all(" · " in s.text and ": " in s.text for s in statements)


def html_table(markup: str) -> Component:
    page = f"<html><body><main><table>{markup}</table></main></body></html>".encode()
    doc = BoilerplateCleaner().clean(Document.from_bytes(page, url="https://cars.test/"))
    root = asyncio.run(HtmlLayoutParser().parse(doc))
    [t] = [c for c in root.walk() if c.type == "table"]
    return t


def test_rowspan_row_headers_apply_to_every_row_they_cover() -> None:
    t = html_table(
        "<thead><tr><th colspan=2></th><th>SE</th><th>GT</th></tr></thead>"
        "<tr><th rowspan=2>Performance</th><th>0-62 mph (s)</th><td>9.1</td><td>7.4</td></tr>"
        "<tr><th>Top speed (mph)</th><td>130</td><td>155</td></tr>"
    )
    assert texts(t) == [
        "Performance · 0-62 mph (s) · SE: 9.1",
        "Performance · 0-62 mph (s) · GT: 7.4",
        "Performance · Top speed (mph) · SE: 130",
        "Performance · Top speed (mph) · GT: 155",
    ]


def test_a_band_above_the_header_row_is_a_group_not_the_headers() -> None:
    t = html_table(
        "<tr><th colspan=3>Technical data</th></tr>"
        "<tr><th></th><th>SE</th><th>GT</th></tr>"
        "<tr><th>Power</th><td>150 PS</td><td>200 PS</td></tr>"
    )
    assert texts(t) == [
        "Technical data › Power · SE: 150 PS",
        "Technical data › Power · GT: 200 PS",
    ]
    assert [s.table.col_headers for s in table_statements(t) if s.table] == [["SE"], ["GT"]]


def test_a_header_row_repeated_mid_table_replaces_the_column_headers() -> None:
    t = html_table(
        "<tr><th></th><th>SE</th><th>GT</th></tr>"
        "<tr><th>Power</th><td>150 PS</td><td>200 PS</td></tr>"
        "<tr><th></th><th>SE L</th><th>R</th></tr>"
        "<tr><th>Torque</th><td>250 Nm</td><td>320 Nm</td></tr>"
    )
    assert texts(t)[2:] == ["Torque · SE L: 250 Nm", "Torque · R: 320 Nm"]


def test_a_data_cell_spanning_rows_takes_each_rows_header() -> None:
    t = html_table(
        "<tr><th></th><th>SE</th></tr>"
        "<tr><th>Engine</th><td rowspan=2>1.5 TSI</td></tr>"
        "<tr><th>Gearbox</th></tr>"
    )
    assert texts(t) == ["Engine · Gearbox · SE: 1.5 TSI"]


def test_a_table_of_only_headers_keeps_its_content_as_rows() -> None:
    t = html_table("<tr><th>Engine</th><th>1.5 TSI</th></tr><tr><th>Power</th><th>150 PS</th></tr>")
    assert texts(t) == ["Engine | 1.5 TSI", "Power | 150 PS"]

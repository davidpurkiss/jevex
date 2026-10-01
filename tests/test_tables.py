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
from jevex.tables import blank_rows, header_prefix, infer_headers, table_statements
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
    t = table(
        *(cell(0, c, text) for c, text in enumerate(["Kestrova", "SE", "£24,995"])),
        *(cell(1, c, text) for c, text in enumerate(["Kestrova", "GT", "£31,250"])),
    )
    assert texts(t) == ["Kestrova | SE | £24,995", "Kestrova | GT | £31,250"]
    assert texts(table()) == []


def test_a_two_column_table_without_headers_reads_as_labels_and_values() -> None:
    t = table(
        cell(0, 0, "Engine:"),
        cell(0, 1, "1.5 TSI"),
        cell(1, 0, "0-62 mph (s)"),
        cell(1, 1, "9.1"),
        cell(2, 1, "a value without its label"),
    )
    statements = table_statements(t)
    assert [s.text for s in statements] == [
        "Engine: 1.5 TSI",
        "0-62 mph (s): 9.1",
        "a value without its label",
    ]
    assert [s.id for s in statements] == ["t1.r0c1", "t1.r1c1", "t1.r2"]
    ref = statements[0].table
    assert ref is not None
    assert (ref.row, ref.col, ref.row_headers, ref.col_headers) == (0, 1, ["Engine"], [])


def test_two_columns_without_labels_or_with_spans_stay_rows() -> None:
    numbers = table(cell(0, 0, "2019"), cell(0, 1, "150 PS"), cell(1, 0, "2021"), cell(1, 1, "163"))
    assert texts(numbers) == ["2019 | 150 PS", "2021 | 163"]
    spanning = table(cell(0, 0, "Engine"), cell(0, 1, "1.5 TSI"), cell(1, 0, "Note", cols=2))
    assert texts(spanning) == ["Engine | 1.5 TSI", "Note"]


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
    [s] = table_statements(t)
    assert s.table is not None
    assert (s.table.row_headers, s.table.row_labels) == (
        ["Engine", "Gearbox"],
        ["Engine", "Gearbox"],
    )


def test_a_rows_label_joins_its_headers_one_label_per_covered_row() -> None:
    t = html_table(
        "<tr><th></th><th></th><th>Warranty</th></tr>"
        "<tr><th rowspan=2>Kestrova</th><th>SE</th><td rowspan=2>3 years</td></tr>"
        "<tr><th>SE L</th></tr>"
        "<tr><th>Ardent</th><th>GT</th><td>5 years</td></tr>"
    )
    refs = [s.table for s in table_statements(t) if s.table]
    assert [(r.row_headers, r.row_labels) for r in refs] == [
        (["Kestrova", "SE", "SE L"], ["Kestrova SE", "Kestrova SE L"]),
        (["Ardent", "GT"], ["Ardent GT"]),
    ]
    assert texts(t)[0] == "Kestrova · SE · SE L · Warranty: 3 years"


def test_a_covered_row_without_headers_gives_no_row_label() -> None:
    t = html_table(
        "<tr><th></th><th>SE</th></tr><tr><th>Engine</th><td rowspan=2>1.5 TSI</td></tr><tr></tr>"
    )
    [s] = table_statements(t)
    assert s.table is not None
    assert (s.table.row_headers, s.table.row_labels) == (["Engine"], ["Engine"])


def test_a_table_of_only_headers_keeps_its_content_as_rows() -> None:
    t = html_table("<tr><th>Engine</th><th>1.5 TSI</th></tr><tr><th>Power</th><th>150 PS</th></tr>")
    assert texts(t) == ["Engine | 1.5 TSI", "Power | 150 PS"]


def test_bands_without_colspan_next_to_the_header_row_are_groups() -> None:
    t = html_table(
        "<thead><tr><th></th><th>SE</th><th>GT</th></tr></thead>"
        "<tr><th>Performance</th></tr>"
        "<tr><th>Power</th><td>150</td><td>200</td></tr>"
        "<tr><th>Economy</th></tr>"
        "<tr><th>MPG</th><td>50</td><td>45</td></tr>"
    )
    assert texts(t) == [
        "Performance › Power · SE: 150",
        "Performance › Power · GT: 200",
        "Economy › MPG · SE: 50",
        "Economy › MPG · GT: 45",
    ]
    above = html_table(
        "<tr><th>Technical data</th></tr><tr><th></th><th>SE</th><th>GT</th></tr>"
        "<tr><th>Power</th><td>150</td><td>200</td></tr>"
    )
    assert texts(above) == ["Technical data › Power · SE: 150", "Technical data › Power · GT: 200"]


def test_a_one_trim_tables_repeated_header_row_stays_a_header() -> None:
    t = html_table(
        "<tr><th></th><th>SE</th></tr><tr><th>Power</th><td>150</td></tr>"
        "<tr><th></th><th>SE</th></tr><tr><th>Torque</th><td>250</td></tr>"
    )
    assert texts(t) == ["Power · SE: 150", "Torque · SE: 250"]


def test_a_label_with_blank_values_is_a_row_without_data_not_a_band() -> None:
    t = html_table(
        "<tr><th></th><th>SE</th><th>GT</th></tr>"
        "<tr><th>Towing</th><td></td><td>&nbsp;</td></tr>"
        "<tr><th>Torque</th><td>250</td><td>320</td></tr>"
    )
    assert texts(t) == ["Torque · SE: 250", "Torque · GT: 320"]
    assert blank_rows(t) == {1}
    # Under a row header spanning down, the blank row isn't a header row either.
    spanned = html_table(
        "<tr><th colspan=2></th><th>SE</th><th>GT</th></tr>"
        "<tr><th rowspan=2>Performance</th><th>Power</th><td>150</td><td>200</td></tr>"
        "<tr><th>Towing</th><td></td><td></td></tr>"
        "<tr><th>Economy</th></tr>"
        "<tr><th></th><th>MPG</th><td>50</td><td>45</td></tr>"
    )
    assert texts(spanned) == [
        "Performance · Power · SE: 150",
        "Performance · Power · GT: 200",
        "Economy › MPG · SE: 50",
        "Economy › MPG · GT: 45",
    ]


def test_a_blank_row_keeps_a_row_header_spanning_into_the_rows_below() -> None:
    t = html_table(
        "<thead><tr><th></th><th>SE</th><th>GT</th></tr></thead>"
        "<tr><th rowspan=2>Towing (kg)</th><td></td><td></td></tr>"
        "<tr><td>750</td><td>1000</td></tr>"
        "<tr><th>Power</th><td>150</td><td>200</td></tr>"
    )
    assert texts(t) == [
        "Towing (kg) · SE: 750",
        "Towing (kg) · GT: 1000",
        "Power · SE: 150",
        "Power · GT: 200",
    ]


def test_empty_corners_and_spacer_columns_do_not_make_blank_rows() -> None:
    corner = html_table(
        "<tr><td></td><th>SE</th><th>GT</th><td class=gap></td></tr>"
        "<tr><th>Power</th><td>150</td><td>200</td><td class=gap></td></tr>"
    )
    assert texts(corner) == ["Power · SE: 150", "Power · GT: 200"]
    assert blank_rows(corner) == set()
    # An empty cell in a column that never holds data leaves a band a band.
    band = table(
        cell(0, 1, "SE", header=True),
        cell(1, 0, "Performance", header=True),
        cell(1, 2, ""),
        cell(2, 0, "Power", header=True),
        cell(2, 1, "150"),
    )
    assert texts(band) == ["Performance › Power · SE: 150"]
    assert blank_rows(band) == set()


def test_a_table_of_empty_cells_gives_no_statements() -> None:
    assert table_statements(table(cell(0, 0, "", header=True), cell(1, 0, " "))) == []


def test_header_prefix_is_what_a_cells_text_starts_with() -> None:
    t = html_table(
        "<thead><tr><th></th><th>SE</th><th>GT</th></tr></thead>"
        "<tr><th>Performance</th></tr>"
        "<tr><th>Power</th><td>150</td><td>200</td></tr>"
    )
    plain = table(cell(0, 0, "Engine"), cell(0, 1, "1.5 TSI"), cell(0, 2, "2.0 TDI"))
    labelled = table(cell(0, 0, "Colour", header=True), cell(0, 1, "Red"))
    got = [
        (header_prefix(s.table), s.text)
        for s in [*table_statements(t), *table_statements(plain), *table_statements(labelled)]
        if s.table is not None
    ]
    assert got == [
        ("Performance › Power · SE: ", "Performance › Power · SE: 150"),
        ("Performance › Power · GT: ", "Performance › Power · GT: 200"),
        ("", "Engine | 1.5 TSI | 2.0 TDI"),
        ("Colour: ", "Colour: Red"),
    ]


def test_bold_td_labels_are_headers_through_the_cleaner_and_parser() -> None:
    pairs = html_table(
        "<tr><td><strong>Engine:</strong></td><td>1.5 TSI</td></tr>"
        "<tr><td><b>Power</b></td><td>150 PS</td></tr>"
    )
    assert texts(pairs) == ["Engine: 1.5 TSI", "Power: 150 PS"]
    comparison = html_table(
        "<tr><td><b>Spec</b></td><td><b>SE</b></td><td><b>GT</b></td></tr>"
        "<tr><td><b>Power</b></td><td>150 PS</td><td>200 PS</td></tr>"
        "<tr><td colspan=3><strong>Economy</strong></td></tr>"
        "<tr><td><b>Combined (mpg)</b></td><td>52.3</td><td><b>45.6</b></td></tr>"
    )
    statements = table_statements(comparison)
    assert [s.text for s in statements] == [
        "Power · SE: 150 PS",
        "Power · GT: 200 PS",
        "Economy › Combined (mpg) · SE: 52.3",
        "Economy › Combined (mpg) · GT: 45.6",  # a bold value stays a value
    ]
    assert [s.table.col_headers for s in statements if s.table] == [["SE"], ["GT"]] * 2


def test_header_less_two_column_html_table_reads_as_labels_and_values() -> None:
    t = html_table("<tr><td>Engine</td><td>1.5 TSI</td></tr><tr><td>Power</td><td>150 PS</td></tr>")
    assert texts(t) == ["Engine: 1.5 TSI", "Power: 150 PS"]


def test_a_label_with_an_empty_value_in_a_label_value_table_gives_nothing() -> None:
    plain = html_table(
        "<tr><td>Engine</td><td>1.5 TSI</td></tr>"
        "<tr><td>Towing</td><td></td></tr>"
        "<tr><td>Kerb weight</td><td>&nbsp;</td></tr>"
        "<tr><td>Power</td><td>150 PS</td></tr>"
    )
    assert texts(plain) == ["Engine: 1.5 TSI", "Power: 150 PS"]
    assert [s.id.split(".")[-1] for s in table_statements(plain)] == ["r0c1", "r3c1"]
    # The bold-label version of the same table gave nothing already, as a blank row.
    bold = html_table(
        "<tr><td><b>Engine</b></td><td>1.5 TSI</td></tr>"
        "<tr><td><b>Towing</b></td><td></td></tr>"
        "<tr><td><b>Power</b></td><td>150 PS</td></tr>"
    )
    assert texts(bold) == texts(plain) == ["Engine: 1.5 TSI", "Power: 150 PS"]


def test_a_header_less_table_that_isnt_label_value_keeps_rows_with_empty_cells() -> None:
    # Numbers in the first column: rows of values, so an empty value keeps its row.
    numbers = html_table("<tr><td>2019</td><td>150 PS</td></tr><tr><td>2021</td><td></td></tr>")
    assert texts(numbers) == ["2019 | 150 PS", "2021"]
    # Wider than two columns: one statement per row, empty cells left out.
    wide = html_table(
        "<tr><td>Kestrova</td><td>SE</td><td>£24,995</td></tr>"
        "<tr><td>Kestrova</td><td></td><td></td></tr>"
    )
    assert texts(wide) == ["Kestrova | SE | £24,995", "Kestrova"]
    # A label alone in its row, with no value cell at all, isn't a label without a value.
    no_cell = table(cell(0, 0, "Engine"), cell(0, 1, "1.5 TSI"), cell(1, 0, "Notes"))
    assert texts(no_cell) == ["Engine: 1.5 TSI", "Notes"]
    # A value whose label cell is empty stays a statement.
    no_label = table(cell(0, 0, "Engine"), cell(0, 1, "1.5 TSI"), cell(1, 0, ""), cell(1, 1, "Red"))
    assert texts(no_label) == ["Engine: 1.5 TSI", "Red"]


def test_a_header_less_comparison_table_infers_its_first_row_and_column_as_headers() -> None:
    t = html_table(
        "<tr><td>Spec</td><td>1.5 TSI SE</td><td>GT</td></tr>"
        "<tr><td>Power</td><td>150 PS</td><td>200 PS</td></tr>"
        "<tr><td>0-62 mph</td><td>9.1</td><td>7.4</td></tr>"
        "<tr><td>Gearbox</td><td>Manual</td><td>Automatic</td></tr>"
        "<tr><td>Towing (kg)</td><td></td><td></td></tr>"
        "<tr><td>Price</td><td>&pound;24,995</td><td>&pound;31,250</td></tr>"
    )
    statements = table_statements(t)
    assert [s.text for s in statements] == [
        "Power · 1.5 TSI SE: 150 PS",
        "Power · GT: 200 PS",
        "0-62 mph · 1.5 TSI SE: 9.1",
        "0-62 mph · GT: 7.4",
        "Gearbox · 1.5 TSI SE: Manual",
        "Gearbox · GT: Automatic",
        "Price · 1.5 TSI SE: £24,995",
        "Price · GT: £31,250",
    ]
    ref = statements[1].table
    assert ref is not None
    assert (ref.row, ref.col, ref.row_headers, ref.col_headers) == (1, 2, ["Power"], ["GT"])
    assert [s.table.col_headers for s in statements if s.table] == [["1.5 TSI SE"], ["GT"]] * 4
    assert blank_rows(infer_headers(t)) == {4}


def test_an_inferred_comparison_table_keeps_an_empty_corner_and_bands() -> None:
    t = html_table(
        "<tr><td></td><td>SE</td><td>GT</td></tr>"
        "<tr><td colspan=3>Performance</td></tr>"
        "<tr><td>Power</td><td>150 PS</td><td>200 PS</td></tr>"
        "<tr><td>Economy</td></tr>"
        "<tr><td>Combined (mpg)</td><td colspan=2>52.3</td></tr>"
    )
    assert texts(t) == [
        "Performance › Power · SE: 150 PS",
        "Performance › Power · GT: 200 PS",
        "Economy › Combined (mpg) · SE / GT: 52.3",
    ]


def test_a_header_less_table_of_plain_records_stays_rows() -> None:
    records = html_table(
        "<tr><td>Name</td><td>City</td><td>Role</td></tr>"
        "<tr><td>Alice</td><td>London</td><td>Engineer</td></tr>"
        "<tr><td>Bob</td><td>Leeds</td><td>Designer</td></tr>"
    )
    assert texts(records) == [
        "Name | City | Role",
        "Alice | London | Engineer",
        "Bob | Leeds | Designer",
    ]
    assert infer_headers(records) is records


def test_header_inference_needs_a_comparison_tables_shape() -> None:
    def rows(*grid: tuple[str, ...]) -> Component:
        cells = [cell(r, c, text) for r, row in enumerate(grid) for c, text in enumerate(row)]
        return table(*cells)

    unchanged = [
        # The first row is already data: its values aren't labels.
        rows(("Power", "150 PS", "200 PS"), ("Torque", "250 Nm", "320 Nm")),
        # A first column of years, not labels.
        rows(("Year", "Power", "Torque"), ("2019", "150 PS", "250 Nm")),
        # A column the first row doesn't name.
        rows(("Spec", "SE", ""), ("Power", "150 PS", "200 PS")),
        # A row without a label.
        rows(("Spec", "SE", "GT"), ("Power", "150 PS", "200 PS"), ("", "250 Nm", "320 Nm")),
        # Values mostly text: records, not specs.
        rows(("Spec", "SE", "GT"), ("Power", "150 PS", "Petrol"), ("Gearbox", "Manual", "Auto")),
        # Only a first row, or labels without values.
        rows(("Spec", "SE", "GT")),
        rows(("Spec", "SE", "GT"), ("Performance",)),
    ]
    for t in unchanged:
        assert infer_headers(t) is t
        assert all(s.table and not s.table.col_headers for s in table_statements(t))
    # Tables that already have headers are left as they are.
    headed = table(cell(0, 1, "SE", header=True), cell(1, 0, "Power"), cell(1, 1, "150 PS"))
    assert infer_headers(headed) is headed

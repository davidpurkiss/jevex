"""Table cells as statements, rendered with their headers (spec: *Statement splitting by
component type*: "Table: one per cell, rendered with its headers").

A spec table's cell "9.1" means nothing alone; rendered as
``0-62 mph (s) · 1.5 TSI SE: 9.1`` it is a complete statement. For each data cell:

- **Column headers** are the header cells above it in the table's header rows (rows made
  only of header cells, before the first data row). Stacked header rows join with a
  space ("1.5 TSI" over "SE" → "1.5 TSI SE"); a spanning header applies to every column
  it covers, and a cell spanning columns names each ("SE / SE L").
- **Row headers** are the header cells in its row (including ones spanning down from
  rows above, ``<th rowspan=2>Performance</th>``); a cell spanning rows takes each row's.
  A row's headers joined are its label ("Kestrova" over "SE" → "Kestrova SE"), kept per
  row a cell covers.
- **Group headers** are body rows holding only header cells (a band like "Performance"
  across the table; a full-width header row counts as one even at the top). They
  prefix the rows below them until the next band. A header row repeated mid-table
  replaces the column headers from there on:
  ``Performance › 0-62 mph (s) · 1.5 TSI SE: 9.1``.
- **Blank rows** hold only header cells followed by empty data cells, in columns that hold
  data elsewhere ("Towing" with no value for any trim). They give no statement and are
  neither a band nor a header row, so the rows below keep their own headers.

In a table with headers on both axes (a comparison table), each header cell is also a
``table_header`` statement, because a header can be a value no cell states: the trim "SE"
heading its column. Its text is the header's own label (``SE``, stacked headers joined).
The corner, the header-row text over the row headers ("Trim", "Specification"), goes on
:attr:`TableCellRef.corner <jevex.statements.TableCellRef.corner>`, and Jev sees it as
context: whether it names the row headers ("Trim" over SE, SE L), the column headers, or
neither is Jev's to judge. So are the other labels on the header's axis
(:attr:`~jevex.statements.TableCellRef.axis_labels`): "Sport" alone needn't read as a trim,
but among "SE, Sport, GT" it does. Each label gives one statement however many columns or rows
repeat it. Headers
that are only field labels ("Fuel") are categorised like any statement, and Jev finds
no value in them. Key/value tables and tables with column headers alone give none: their
headers label the values, and every cell's text already holds them.

The text is ``[group › ][row headers · ][column headers: ]value``, with a header's
trailing colon dropped. A table without any header (or made only of headers) gives one
statement per row, its cells joined with ``" | "``; a two-column one without headers whose
first column holds labels (not numbers) reads as ``label: value`` instead (a label whose
value cell is empty giving nothing, like a blank row), and a wider one shaped like a
comparison table has its headers inferred (:func:`infer_headers`). The headers also travel
structured on :attr:`Statement.table <jevex.statements.Statement.table>`, so an
entity resolver can split a comparison table by column (one trim per column).

This works on :class:`~jevex.layout.TableCell` grids from any layout parser (HTML and
PDF).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from jevex.statements import Statement, TableCellRef

if TYPE_CHECKING:
    from jevex.layout import Component, TableCell

_WHITESPACE = re.compile(r"[ \t\n\r\f\v]+")  # a no-break space stays: it can group thousands

RowRole = Literal["header", "band", "body"]
"""A table row's part in reading the table (:func:`row_roles`)."""


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _label(text: str) -> str:
    """A header's text without the colon labels often end with ("Engine:")."""
    return _clean(text).rstrip(":").rstrip()


def table_statements(table: Component) -> list[Statement]:
    """One statement per non-empty data cell of ``table`` (or per row, without headers).

    A table with headers on both axes also gives a ``table_header`` statement per header
    label: a column's before the first body row, a row's before that row's cells."""
    table = infer_headers(table)
    cells = [c for c in table.cells if _clean(c.text)]
    if not cells:
        return []
    width = max(c.col + c.col_span for c in cells)
    rows: dict[int, list[TableCell]] = {}
    for c in cells:
        rows.setdefault(c.row, []).append(c)
    ordered = sorted(rows)

    if all(c.header for c in cells) or not any(c.header for c in cells):
        # No headers, or nothing but headers: nothing to attach, so one statement per row,
        # except that a two-column table without headers is read as labels and values.
        pairs = not any(c.header for c in cells) and _label_value(cells, width)
        empty_values = {c.row for c in table.cells if c.col == 1 and not _clean(c.text)}
        out: list[Statement] = []
        for r in ordered:
            row = sorted(rows[r], key=lambda c: c.col)
            if pairs and len(row) == 1 and row[0].col == 0 and r in empty_values:
                continue  # a label without its value ("Towing | "), as a blank row gives nothing
            if pairs and len(row) == 2:
                label, value = _label(row[0].text), _clean(row[1].text)
                ref = TableCellRef(row=r, col=1, row_headers=[label], row_labels=[label])
                out.append(_statement(table, f"r{r}c1", _render(None, [label], [], value), ref))
            else:
                text = " | ".join(_clean(c.text) for c in row)
                out.append(_statement(table, f"r{r}", text, TableCellRef(row=r, col=0)))
        return out

    def headers_of(header_rows: list[int]) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for r in header_rows:
            for c in rows[r]:
                for col in range(c.col, c.col + c.col_span):
                    out.setdefault(col, []).append(_label(c.text))
        return out

    roles = row_roles(table)

    # Header rows: the leading rows made only of header cells. A band among them
    # ("Technical data" above the header row, "Performance" just below it) is a group for
    # the body, not a column header.
    group: str | None = None
    leading: list[int] = []
    header_rows: list[int] = []
    for r in ordered:
        if roles[r] == "body":
            break
        leading.append(r)
        if roles[r] == "band":
            group = _label(rows[r][0].text)
        else:
            header_rows.append(r)
    col_headers = headers_of(header_rows)
    body = [r for r in ordered if r not in leading]

    # Row headers can span rows (<th rowspan=2>Performance</th>): index them by every row
    # they cover.
    row_header_cells: dict[int, list[TableCell]] = {}
    for r in body:
        if roles[r] != "body":
            continue
        for c in rows[r]:
            if c.header:
                for covered in range(c.row, c.row + c.row_span):
                    row_header_cells.setdefault(covered, []).append(c)

    # Header statements, only with headers on both axes (see the module docstring).
    row_header_cols = {c.col for cells in row_header_cells.values() for c in cells}
    both_axes = bool(header_rows) and bool(row_header_cols)
    seen: set[str] = set()
    # A header over no data (a blank row's "Towing", an empty column) names nothing.
    data = [c for c in cells if not c.header]
    data_rows = {row for c in data for row in range(c.row, c.row + c.row_span)}
    data_cols = {col for c in data for col in range(c.col, c.col + c.col_span)}

    def row_labels_of(c: TableCell) -> list[str]:
        """One label per row ``c`` covers: that row's headers, joined ("Kestrova SE")."""
        per_row = [
            [_label(h.text) for h in sorted(row_header_cells.get(covered, []), key=lambda h: h.col)]
            for covered in range(c.row, c.row + c.row_span)
        ]
        return list(dict.fromkeys(" ".join(headers) for headers in per_row if headers))

    row_axis = list(
        dict.fromkeys(
            _label(c.text)
            for r in body
            if roles[r] == "body"
            for c in rows[r]
            if c.header
            and _label(c.text)
            and any(r2 in data_rows for r2 in range(c.row, c.row + c.row_span))
        )
    )
    out: list[Statement] = []
    if both_axes:
        out += _column_header_statements(table, header_rows[-1], col_headers, data_cols, seen)
    for r in body:
        row = sorted(rows[r], key=lambda c: c.col)
        if roles[r] != "body":
            if roles[r] == "band":
                group = _label(row[0].text)  # a band ("Performance")
            else:
                col_headers = headers_of([r])  # a header row repeated mid-table
                if both_axes:
                    out += _column_header_statements(table, r, col_headers, data_cols, seen)
            continue
        for c in row:
            if c.header:
                label = _label(c.text)
                covers_data = any(r2 in data_rows for r2 in range(c.row, c.row + c.row_span))
                if both_axes and label and covers_data and label not in seen:
                    seen.add(label)
                    ref = TableCellRef(
                        row=r,
                        col=c.col,
                        row_headers=[label],
                        row_labels=row_labels_of(c),
                        corner=" ".join(col_headers.get(c.col, [])) or None,
                        axis_labels=row_axis,
                    )
                    out.append(_statement(table, f"h{r}c{c.col}", label, ref, "table_header"))
                continue
            # A data cell spanning rows takes every covered row's headers, and one label
            # per covered row: its headers, joined ("Kestrova SE").
            per_row = [
                [
                    _label(h.text)
                    for h in sorted(row_header_cells.get(covered, []), key=lambda h: h.col)
                ]
                for covered in range(c.row, c.row + c.row_span)
            ]
            row_headers = list(dict.fromkeys(h for headers in per_row for h in headers))
            row_labels = list(dict.fromkeys(" ".join(headers) for headers in per_row if headers))
            # One label per column covered: its stacked headers, joined.
            labels = [
                " ".join(col_headers[col])
                for col in range(c.col, c.col + c.col_span)
                if col_headers.get(col)
            ]
            columns = list(dict.fromkeys(labels))
            out.append(
                _statement(
                    table,
                    f"r{r}c{c.col}",
                    _render(group, row_headers, columns, _clean(c.text)),
                    TableCellRef(
                        row=r,
                        col=c.col,
                        row_headers=row_headers,
                        row_labels=row_labels,
                        col_headers=columns,
                        group=group,
                    ),
                )
            )
    return out


def _label_value(cells: list[TableCell], width: int) -> bool:
    """Whether a table without headers reads as ``label | value`` rows: two columns, no
    spans, and a label (some letter, not just a number) in every first-column cell."""
    return (
        width == 2
        and all(c.row_span == 1 and c.col_span == 1 for c in cells)
        and all(any(ch.isalpha() for ch in c.text) for c in cells if c.col == 0)
    )


def infer_headers(table: Component) -> Component:
    """``table`` with its first row and first column marked as headers, when it has no
    header cells but reads as a comparison table (``Spec | SE | GT`` over
    ``Power | 150 PS | 200 PS`` in plain ``td``); otherwise ``table`` unchanged.

    It must be at least three columns wide (two columns read as ``label: value``). The
    first row must name every other column with a name (more letters than digits:
    ``1.5 TSI`` is one, ``150 PS`` and ``2019`` aren't), and every row below must start with
    a label (some letter: ``0-62 mph``, not ``2019``). The cells below the first row and
    right of the first column must mostly be number-like. Those checks are the guard: a
    table of plain records (``Name | City | Role``) has text in its body just like its
    first row, so it keeps one statement per row, and so does a table whose first row is
    already number-like data (``Power | 150 PS | 200 PS``). A first row of text data
    (``Gearbox | Manual | Automatic``) can't be told from column names by shape, so it is
    read as one.
    """
    cells = [c for c in table.cells if _clean(c.text)]
    if not cells or any(c.header for c in table.cells):
        return table
    width = max(c.col + c.col_span for c in cells)
    top = min(c.row for c in cells)
    body_rows = {c.row for c in cells} - {top}
    first_row = [c for c in cells if c.row == top and c.col > 0]
    labels = [c for c in cells if c.row > top and c.col == 0]
    values = [c for c in cells if c.row > top and c.col > 0]
    named = {col for c in first_row for col in range(c.col, c.col + c.col_span)}
    labelled = {covered for c in labels for covered in range(c.row, c.row + c.row_span)}
    if (
        width < 3
        or not values
        or named != set(range(1, width))
        or not body_rows <= labelled
        or not all(_is_name(c.text) for c in first_row)
        or not all(any(ch.isalpha() for ch in c.text) for c in labels)
        or sum(_number_like(c.text) for c in values) * 2 <= len(values)
    ):
        return table
    inferred = [
        c.model_copy(update={"header": True}) if c.row == top or c.col == 0 else c
        for c in table.cells
    ]
    return table.model_copy(update={"cells": inferred})


def _digits_and_letters(text: str) -> tuple[int, int]:
    return sum(ch.isdigit() for ch in text), sum(ch.isalpha() for ch in text)


def _is_name(text: str) -> bool:
    digits, letters = _digits_and_letters(text)
    return letters > digits


def _number_like(text: str) -> bool:
    digits, letters = _digits_and_letters(text)
    return digits > 0 and digits >= letters


def row_roles(table: Component) -> dict[int, RowRole]:
    """How :func:`table_statements` reads each row of ``table`` that has text.

    ``"band"`` is a group header ("Performance") that prefixes the rows below it until the
    next band. ``"header"`` is a row made only of header cells giving column headers: the
    leading ones stack, and one repeated after the first body row replaces them.
    Everything else is ``"body"``, including blank rows (:func:`blank_rows`) and rows a
    data cell spans into. A table made only of headers, or without any, is all body rows.
    Pass ``infer_headers(table)`` to read a header-less comparison table as its statements
    do.
    """
    cells = [c for c in table.cells if _clean(c.text)]
    roles: dict[int, RowRole] = {c.row: "body" for c in cells}
    if all(c.header for c in cells) or not any(c.header for c in cells):
        return roles
    width = max(c.col + c.col_span for c in cells)
    has_data = {
        covered for c in cells if not c.header for covered in range(c.row, c.row + c.row_span)
    }
    has_data |= blank_rows(table)
    rows: dict[int, list[TableCell]] = {}
    for c in cells:
        rows.setdefault(c.row, []).append(c)
    for r, row in rows.items():
        if r not in has_data and all(c.header for c in row):
            roles[r] = "band" if _is_band(row, width) else "header"
    return roles


def blank_rows(table: Component) -> set[int]:
    """The rows of ``table`` that hold data cells, all empty: their filled cells are all
    headers, followed by an empty data cell in a column that holds data in other rows
    (``Towing | | ``). Such a row is neither a band nor a header row. An empty cell before
    the headers is a corner (`` | SE | GT``), and one in a column that never holds data a
    spacer."""
    filled = [c for c in table.cells if _clean(c.text)]
    data_columns = {col for c in filled if not c.header for col in range(c.col, c.col + c.col_span)}
    end: dict[int, int] = {}  # row -> the column after its last filled cell
    labels_only: set[int] = set()
    for c in filled:
        end[c.row] = max(end.get(c.row, 0), c.col + c.col_span)
        labels_only.add(c.row)
    labels_only -= {c.row for c in filled if not c.header}
    return {
        c.row
        for c in table.cells
        if c.row in labels_only
        and not c.header
        and not _clean(c.text)
        and c.col >= end[c.row]
        and c.col in data_columns
    }


def _is_band(row: list[TableCell], width: int) -> bool:
    """A group header: one header cell alone at the start of its row ("Performance"),
    spanning the table or not. A lone cell further right is a column header instead (a
    one-trim table's repeated header row, after its empty corner cell)."""
    return len(row) == 1 and row[0].header and row[0].col == 0 and width > 1


def header_prefix(ref: TableCellRef) -> str:
    """What a cell's text starts with before its value: ``"Performance › 0-62 mph (s) ·
    1.5 TSI SE: "``, or ``""`` for a cell without headers."""
    return _render(ref.group, ref.row_headers, ref.col_headers, "")


def _render(group: str | None, rows: list[str], cols: list[str], value: str) -> str:
    text = f"{' / '.join(cols)}: {value}" if cols else value
    if rows:
        text = f"{' · '.join(rows)} · {text}" if cols else f"{' · '.join(rows)}: {value}"
    return f"{group} › {text}" if group else text


def _column_header_statements(
    table: Component,
    row: int,
    col_headers: dict[int, list[str]],
    data_cols: set[int],
    seen: set[str],
) -> list[Statement]:
    """A ``table_header`` statement per label of a column holding data, unless already
    ``seen``: its stacked headers joined (``1.5 TSI SE``), with the corner (the header-row
    text left of the first data column) and the row's column labels as context."""
    first = min(data_cols, default=0)
    corner = " ".join(
        " ".join(labels) for col, labels in sorted(col_headers.items()) if col < first
    )
    labels = {
        col: label
        for col in sorted(col_headers)
        if col in data_cols and (label := " ".join(col_headers[col]))
    }
    axis = list(dict.fromkeys(labels.values()))
    out: list[Statement] = []
    for col, label in labels.items():
        if label in seen:
            continue
        seen.add(label)
        ref = TableCellRef(
            row=row, col=col, col_headers=[label], corner=corner.strip() or None, axis_labels=axis
        )
        out.append(_statement(table, f"h{row}c{col}", label, ref, "table_header"))
    return out


def _statement(
    table: Component,
    suffix: str,
    text: str,
    ref: TableCellRef,
    kind: Literal["table_cell", "table_header"] = "table_cell",
) -> Statement:
    return Statement(
        id=f"{table.id}.{suffix}",
        text=text,
        kind=kind,
        component_id=table.id,
        heading_trail=list(table.heading_trail),
        location=table.location,
        table=ref,
    )

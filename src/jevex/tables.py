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
- **Group headers** are body rows holding only header cells (a band like "Performance"
  across the table; a full-width header row counts as one even at the top). They
  prefix the rows below them until the next band. A header row repeated mid-table
  replaces the column headers from there on:
  ``Performance › 0-62 mph (s) · 1.5 TSI SE: 9.1``.

The text is ``[group › ][row headers · ][column headers: ]value``. A table without any
header (or made only of headers) gives one statement per row, its cells joined with
``" | "``. The headers also
travel structured on :attr:`Statement.table <jevex.statements.Statement.table>`, so an
entity resolver can split a comparison table by column (one trim per column).

This works on :class:`~jevex.layout.TableCell` grids from any layout parser (HTML now,
PDF with #24).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from jevex.statements import Statement, TableCellRef

if TYPE_CHECKING:
    from jevex.layout import Component, TableCell

_WHITESPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def table_statements(table: Component) -> list[Statement]:
    """One statement per non-empty data cell of ``table`` (or per row, without headers)."""
    cells = [c for c in table.cells if _clean(c.text)]
    if not cells:
        return []
    width = max(c.col + c.col_span for c in cells)
    rows: dict[int, list[TableCell]] = {}
    for c in cells:
        rows.setdefault(c.row, []).append(c)
    ordered = sorted(rows)

    if all(c.header for c in cells) or not any(c.header for c in cells):
        # No headers, or nothing but headers: nothing to attach, so one statement per row.
        return [
            _statement(
                table,
                f"r{r}",
                " | ".join(_clean(c.text) for c in sorted(rows[r], key=lambda c: c.col)),
                TableCellRef(row=r, col=0),
            )
            for r in ordered
        ]

    def headers_of(header_rows: list[int]) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for r in header_rows:
            for c in rows[r]:
                for col in range(c.col, c.col + c.col_span):
                    out.setdefault(col, []).append(_clean(c.text))
        return out

    # Header rows: the leading rows made only of header cells. A full-width band among
    # them ("Technical data") is a group for the body, not a column header.
    group: str | None = None
    leading: list[int] = []
    header_rows: list[int] = []
    for r in ordered:
        if not all(c.header for c in rows[r]):
            break
        leading.append(r)
        if _is_band(rows[r], width):
            group = _clean(rows[r][0].text)
        else:
            header_rows.append(r)
    col_headers = headers_of(header_rows)
    body = [r for r in ordered if r not in leading]
    # Rows a data cell spans into aren't header-only, even if their own cells all are.
    has_data = {
        covered for c in cells if not c.header for covered in range(c.row, c.row + c.row_span)
    }

    def header_only(r: int) -> bool:
        return r not in has_data and all(c.header for c in rows[r])

    # Row headers can span rows (<th rowspan=2>Performance</th>): index them by every row
    # they cover.
    row_header_cells: dict[int, list[TableCell]] = {}
    for r in body:
        if header_only(r):
            continue
        for c in rows[r]:
            if c.header:
                for covered in range(c.row, c.row + c.row_span):
                    row_header_cells.setdefault(covered, []).append(c)

    out: list[Statement] = []
    for r in body:
        row = sorted(rows[r], key=lambda c: c.col)
        if header_only(r):
            if len(row) == 1:
                group = _clean(row[0].text)  # a band ("Performance")
            else:
                col_headers = headers_of([r])  # a header row repeated mid-table
            continue
        for c in row:
            if c.header:
                continue
            # A data cell spanning rows takes every covered row's headers.
            row_headers = list(
                dict.fromkeys(
                    _clean(h.text)
                    for covered in range(c.row, c.row + c.row_span)
                    for h in sorted(row_header_cells.get(covered, []), key=lambda h: h.col)
                )
            )
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
                        col_headers=columns,
                        group=group,
                    ),
                )
            )
    return out


def _is_band(row: list[TableCell], width: int) -> bool:
    """A group header: one header cell spanning the whole table ("Performance")."""
    return len(row) == 1 and row[0].header and row[0].col == 0 and row[0].col_span >= width > 1


def _render(group: str | None, rows: list[str], cols: list[str], value: str) -> str:
    text = f"{' / '.join(cols)}: {value}" if cols else value
    if rows:
        text = f"{' · '.join(rows)} · {text}" if cols else f"{' · '.join(rows)}: {value}"
    return f"{group} › {text}" if group else text


def _statement(table: Component, suffix: str, text: str, ref: TableCellRef) -> Statement:
    return Statement(
        id=f"{table.id}.{suffix}",
        text=text,
        kind="table_cell",
        component_id=table.id,
        heading_trail=list(table.heading_trail),
        location=table.location,
        table=ref,
    )

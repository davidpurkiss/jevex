"""Table cells as statements, rendered with their headers (spec: *Statement splitting by
component type*: "Table: one per cell, rendered with its headers").

A spec table's cell "9.1" means nothing alone; rendered as
``0-62 mph (s) · 1.5 TSI SE: 9.1`` it is a complete statement. For each data cell:

- **Column headers** are the header cells above it in the table's header rows (rows made
  only of header cells, before the first data row). Stacked header rows join with a
  space ("1.5 TSI" over "SE" → "1.5 TSI SE"); a spanning header applies to every column
  it covers, and a cell spanning columns names each ("SE / SE L").
- **Row headers** are the header cells to its left in its own row.
- **Group headers** are body rows holding only header cells (a band like "Performance"
  across the table; a full-width header row counts as one even at the top). They
  prefix the rows below them until the next band:
  ``Performance › 0-62 mph (s) · 1.5 TSI SE: 9.1``.

The text is ``[group › ][row headers · ][column headers: ]value``. A table without any
header gives one statement per row, its cells joined with ``" | "``. The headers also
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

    if not any(c.header for c in cells):
        return [
            _statement(
                table,
                f"r{r}",
                " | ".join(_clean(c.text) for c in sorted(rows[r], key=lambda c: c.col)),
                TableCellRef(row=r, col=0),
            )
            for r in ordered
        ]

    # Header rows: leading rows made only of header cells.
    header_rows: list[int] = []
    for r in ordered:
        if all(c.header for c in rows[r]) and not _is_band(rows[r], width):
            header_rows.append(r)
        else:
            break
    # The row-header columns: columns holding header cells in body rows (left of data).
    body = [r for r in ordered if r not in header_rows]
    col_headers: dict[int, list[str]] = {}
    for r in header_rows:
        for c in rows[r]:
            for col in range(c.col, c.col + c.col_span):
                col_headers.setdefault(col, []).append(_clean(c.text))

    out: list[Statement] = []
    group: str | None = None
    for r in body:
        row = sorted(rows[r], key=lambda c: c.col)
        if all(c.header for c in row):
            # A band ("Performance"), whether it spans the table or sits in one cell.
            group = " ".join(_clean(c.text) for c in row)
            continue
        row_headers = [_clean(c.text) for c in row if c.header]
        for c in row:
            if c.header:
                continue
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

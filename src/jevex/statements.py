"""Statements and the candidate values generated from them."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from jevex.layout import Location

StatementKind = Literal[
    "sentence",
    "list_item",
    "key_value",
    "table_cell",
    "table_header",
    "alt_text",
    "caption",
    "ocr",
    "structured",
    "vision",
]


class Span(BaseModel):
    """Half-open character offsets ``[start, end)`` into a statement's text."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end < self.start:
            raise ValueError("span end must not be before start")
        return self

    def of(self, text: str) -> str:
        """The substring of ``text`` this span covers."""
        if self.end > len(text):
            raise ValueError(f"span {self.start}:{self.end} is outside text of length {len(text)}")
        return text[self.start : self.end]


class TableCellRef(BaseModel):
    """Where a ``table_cell`` or ``table_header`` statement sits in its table, with the
    headers that give it meaning. ``col_headers`` has one label per column the cell covers
    (stacked header rows joined: "1.5 TSI SE"); entity resolvers use it to split a
    comparison table by column.
    ``row_headers`` holds every header of the rows the cell covers, as its text shows them,
    and ``row_labels`` one label per covered row (that row's headers joined: "Kestrova
    SE"), so a cell spanning two rows can go to each row's entity. A column header's
    statement has only ``col_headers`` (its own label), a row header's only
    ``row_headers`` (its own label) and ``row_labels``."""

    model_config = ConfigDict(frozen=True)

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_headers: list[str] = Field(default_factory=list[str])
    row_labels: list[str] = Field(default_factory=list[str])
    col_headers: list[str] = Field(default_factory=list[str])
    group: str | None = None
    corner: str | None = None
    """A header statement's corner cell: the header-row text over the row headers
    ("Trim", "Specification"). Jev sees it as context, since only Jev can tell which axis
    it names."""
    axis_labels: list[str] = Field(default_factory=list[str])
    """A header statement's fellow labels on its axis, itself included ("SE", "Sport",
    "GT"): Jev sees them as context, since "Sport" alone needn't read as a trim."""


class Statement(BaseModel):
    """One atomic piece of text: a sentence, list item, key/value pair or table cell.

    Table cells carry their headers in ``text``, e.g.
    ``Performance › 0-62 mph (s) · 1.5 TSI SE: 9.1``.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    kind: StatementKind
    component_id: str
    heading_trail: list[str] = Field(default_factory=list[str])
    location: Location
    table: TableCellRef | None = None
    """Set on ``table_cell`` statements."""


class NormaliserStep(BaseModel):
    """One step in a normaliser chain.

    Accepts and serialises to the compact YAML form used in generator specs:
    ``"parse_number"`` or ``{"unit": {"from": "s", "to": "s"}}``.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict[str, Any])

    @model_validator(mode="before")
    @classmethod
    def _from_compact(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        if not isinstance(data, dict):
            return data
        compact = cast("dict[str, Any]", data)
        if len(compact) != 1 or "name" in compact:
            return compact
        ((name, args),) = compact.items()
        if args is None:
            args = {}
        elif not isinstance(args, dict):
            raise ValueError(f"arguments for normaliser {name!r} must be a mapping")
        return {"name": name, "args": cast("dict[str, Any]", args)}

    @model_serializer
    def _to_compact(self) -> str | dict[str, dict[str, Any]]:
        return {self.name: self.args} if self.args else self.name


class Candidate(BaseModel):
    """A verbatim span a generator proposes as a field value, with how to normalise it."""

    model_config = ConfigDict(frozen=True)

    span: Span
    raw: str
    normalise: list[NormaliserStep] = Field(default_factory=list[NormaliserStep])
    generator_id: str

    @classmethod
    def from_statement(
        cls,
        statement: Statement,
        span: Span,
        *,
        generator_id: str,
        normalise: list[NormaliserStep] | None = None,
    ) -> Candidate:
        """Build a candidate whose ``raw`` is guaranteed to match the statement text."""
        return cls(
            span=span,
            raw=span.of(statement.text),
            normalise=normalise or [],
            generator_id=generator_id,
        )

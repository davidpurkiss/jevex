"""The generic component tree every layout parser produces, and the layout stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from jevex.interfaces import LayoutParser
    from jevex.pipeline import Context

ComponentType = Literal[
    "section",
    "column",
    "heading",
    "paragraph",
    "list",
    "list_item",
    "table",
    "image",
    "breakout",
    "caption",
]


class UnsupportedDocumentError(ValueError):
    """A layout parser was given a document type it doesn't read."""


class BBox(BaseModel):
    """A rectangle in page coordinates: (x0, y0) top-left, (x1, y1) bottom-right."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox corners must be ordered (x0 <= x1, y0 <= y1)")
        return self


class DomLocation(BaseModel):
    """Where a component sits in an HTML document."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["dom"] = "dom"
    dom_path: str


class PageLocation(BaseModel):
    """Where a component sits on a PDF page (1-based page number)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["page"] = "page"
    page: int = Field(ge=1)
    bbox: BBox | None = None


class ImageLocation(BaseModel):
    """Text found inside an image: the image's URL or page, plus the OCR line's bbox."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["image"] = "image"
    src: str | None = None
    page: int | None = Field(default=None, ge=1)
    bbox: BBox | None = None


Location = Annotated[DomLocation | PageLocation | ImageLocation, Field(discriminator="kind")]


class TableCell(BaseModel):
    """One cell of a table component, placed on the table's grid (0-based).

    A cell spanning several rows or columns appears once, at its top-left position.
    ``header`` marks header cells (``<th>``, or any cell in ``<thead>``), whose text gives
    the other cells their meaning.
    """

    model_config = ConfigDict(frozen=True)

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    text: str
    header: bool = False
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)


class Component(BaseModel):
    """A node in the layout tree. Stages after layout only ever see components.

    ``text`` is the component's own text: containers (sections, breakouts, lists) leave
    it empty and hold their content in ``children``. A table's text is its rows, one per
    line with cells joined by ``" | "``, and its cells are also given structured in
    ``cells``.
    """

    id: str
    type: ComponentType
    text: str = ""
    children: list[Component] = Field(default_factory=list["Component"])
    heading_trail: list[str] = Field(default_factory=list[str])
    location: Location
    cells: list[TableCell] = Field(default_factory=list[TableCell])

    def walk(self) -> Iterator[Component]:
        """Yield this component and all descendants, depth first, in reading order."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, component_id: str) -> Component | None:
        return next((c for c in self.walk() if c.id == component_id), None)


MAX_HEADING_CHARS = 200
"""Each heading in a ``section`` is shortened to this. A heading is a title, and one this
long is really a paragraph marked up as a heading."""

MAX_SECTION_CHARS = 500
"""The most a ``section`` can take up in a Jev state. It rides along with every gate unit
and statement below its headings, so it has to stay small next to their content."""

SECTION_SEPARATOR = " › "
_ELLIPSIS = "…"


def _shorten(text: str, max_chars: int) -> str:
    """``text`` cut to at most ``max_chars`` with a trailing "…", at the last whitespace in
    the second half when there is one."""
    if len(text) <= max_chars:
        return text
    keep = max_chars - len(_ELLIPSIS)
    space = text.rfind(" ", keep // 2, keep + 1)
    return text[: space if space > 0 else keep].rstrip() + _ELLIPSIS


def section_text(
    trail: Sequence[str],
    *,
    max_heading_chars: int = MAX_HEADING_CHARS,
    max_chars: int = MAX_SECTION_CHARS,
) -> str:
    """A heading trail as Jev sees it: the headings joined with " › ", at most
    ``max_chars`` long.

    Each heading is shortened to ``max_heading_chars`` (ending "…"). If the trail is still
    too long, headings are dropped from the middle, keeping the outermost (often the page's
    title) and as many of the innermost as fit, with "…" where the others were. When
    even the outermost and innermost together don't fit, only the innermost is kept, cut
    to ``max_chars``. Empty headings are skipped.
    """
    if max_heading_chars < 1 or max_chars < 1:
        raise ValueError(
            f"max_heading_chars and max_chars must be positive, got {max_heading_chars} "
            f"and {max_chars}"
        )
    headings = [_shorten(h, max_heading_chars) for t in trail if (h := t.strip())]
    text = SECTION_SEPARATOR.join(headings)
    if len(text) <= max_chars:
        return text
    first, inner = headings[0], headings[1:]
    kept: list[str] = []
    while inner:
        candidate = SECTION_SEPARATOR.join([first, _ELLIPSIS, inner[-1], *kept])
        if len(candidate) > max_chars:
            break
        kept.insert(0, inner.pop())
    if kept:
        return SECTION_SEPARATOR.join([first, _ELLIPSIS, *kept])
    return _shorten(headings[-1], max_chars)


def _default_parsers() -> list[LayoutParser]:
    # Imported here because the HTML parser builds on this module's models.
    from jevex.layout_html import HtmlLayoutParser

    return [HtmlLayoutParser()]


@dataclass
class LayoutStage:
    """Runs the first :class:`~jevex.interfaces.LayoutParser` that supports the document.

    The tree lands on ``ctx.parsed``. When no parser supports the content type, the stage
    records a ``layout_skipped`` event and leaves ``ctx.parsed`` unset, so later stages
    can still use what the structured-data stage found.
    """

    parsers: list[LayoutParser] = field(default_factory=_default_parsers)
    name: str = "layout"

    async def run(self, ctx: Context) -> None:
        from jevex.interfaces import ParsedDocument  # interfaces imports this module

        parser = next((p for p in self.parsers if p.supports(ctx.document)), None)
        if parser is None:
            ctx.event(
                self.name,
                "layout_skipped",
                f"no layout parser supports {ctx.document.content_type}",
                content_type=ctx.document.content_type,
            )
            return
        root = await parser.parse(ctx.document)
        ctx.parsed = ParsedDocument(document=ctx.document, root=root)

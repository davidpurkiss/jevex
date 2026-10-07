"""The generic component tree every layout parser produces, and the layout stage."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevex.errors import DocumentError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from jevex.interfaces import LayoutParser
    from jevex.pipeline import Context, SchemaRun

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


class UnsupportedDocumentError(DocumentError, ValueError):
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
    ``header`` marks header cells (``<th>``, any cell in ``<thead>``, or a ``<td>`` label
    set only in bold), whose text gives the other cells their meaning. Empty cells are
    kept, with ``text=""``: a label followed by blank data cells ("Towing | | ") is a row
    whose values are missing, not a band over the rows below.
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
    ``cells``. An image's text is its alt text, and ``src`` its URL when it has one (for
    HTML, resolved against the document's URL); the image stage adds the text found in it
    as children.
    """

    id: str
    type: ComponentType
    text: str = ""
    children: list[Component] = Field(default_factory=list["Component"])
    heading_trail: list[str] = Field(default_factory=list[str])
    location: Location
    cells: list[TableCell] = Field(default_factory=list[TableCell])
    src: str | None = None

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


def _docling_installed() -> bool:
    """Whether Docling (the ``pdf`` extra) is importable."""
    return importlib.util.find_spec("docling") is not None


def _default_parsers() -> list[LayoutParser]:
    # Imported here because the parsers build on this module's models.
    from jevex.layout_html import HtmlLayoutParser
    from jevex.layout_pdf import PdfLayoutParser

    parsers: list[LayoutParser] = [HtmlLayoutParser()]
    if _docling_installed():
        parsers.append(PdfLayoutParser())
    return parsers


@dataclass
class LayoutStage:
    """Runs the first :class:`~jevex.interfaces.LayoutParser` that supports the document.

    The tree lands on ``ctx.parsed``. When no parser supports the content type, the stage
    records a ``layout_skipped`` event and leaves ``ctx.parsed`` unset, so later stages
    can still use what the structured-data stage found. The default parsers read HTML,
    and PDFs when the ``pdf`` extra is installed.

    Pages the document gate ruled out for every active schema (:func:`gated_out_pages`)
    are left out when the parser is a :class:`~jevex.interfaces.PagedLayoutParser`, with
    a ``pages_skipped`` event. Another parser lays out every page, with a
    ``pages_not_skipped`` event.
    """

    parsers: list[LayoutParser] = field(default_factory=_default_parsers)
    name: str = "layout"

    async def run(self, ctx: Context) -> None:
        # interfaces imports this module
        from jevex.interfaces import PagedLayoutParser, ParsedDocument

        parser = next((p for p in self.parsers if p.supports(ctx.document)), None)
        if parser is None:
            detail = f"no layout parser supports {ctx.document.content_type}"
            if ctx.document.is_pdf and not _docling_installed():
                detail += "; install jevex[pdf] for the default PDF parser"
            ctx.event(
                self.name,
                "layout_skipped",
                detail,
                content_type=ctx.document.content_type,
            )
            return
        skip = gated_out_pages(ctx.active)
        if not skip:
            root = await parser.parse(ctx.document)
        elif isinstance(parser, PagedLayoutParser):
            root = await parser.parse_pages(ctx.document, skip)
            ctx.event(
                self.name,
                "pages_skipped",
                f"left out {len(skip)} page(s) the document gate ruled out",
                pages=sorted(skip),
            )
        else:
            root = await parser.parse(ctx.document)
            ctx.event(
                self.name,
                "pages_not_skipped",
                f"{type(parser).__name__} can't leave pages out, so it laid out the "
                f"{len(skip)} page(s) the document gate ruled out",
                pages=sorted(skip),
            )
        ctx.parsed = ParsedDocument(document=ctx.document, root=root)


def gated_out_pages(runs: Sequence[SchemaRun]) -> frozenset[int]:
    """The pages no schema in ``runs`` needs: those the document gate asked about and every
    schema failed. A schema gated per document, or not gated, needs every page, and a page
    the gate didn't ask about (no text layer) is needed by every schema."""
    skip: set[int] | None = None
    for run in runs:
        gate = run.gate
        if gate is None or not gate.pages:
            return frozenset()
        failed = set(gate.pages) - set(gate.passed_pages)
        skip = failed if skip is None else skip & failed
    return frozenset(skip or ())

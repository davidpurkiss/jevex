"""Stage 5 for PDFs: lay out a PDF with Docling and map it to the component tree
(``pip install jevex[pdf]``).

:class:`PdfLayoutParser` runs `Docling <https://github.com/docling-project/docling>`_'s
standard PDF pipeline (layout model, reading order, TableFormer table structure and heading
levels) and turns the resulting ``DoclingDocument`` into the same generic tree the HTML
parser builds, so every later stage is format-agnostic:

- Titles and section headers become headings. As in HTML, a heading opens an implicit
  section holding everything after it up to the next heading of the same or higher rank.
  Docling's layout model finds headings without a level, so its heading-hierarchy step
  ranks them from the PDF's bookmarks, their numbering or their font style. A title
  outranks every section header.
- List groups become lists of list items. Docling reads a leading number as the item's
  marker ("1.5 TSI 150PS" loses its "1.5"), so a marker that isn't a bullet or an
  enumerator like ``1.``, ``a)`` or ``(iv)`` is put back in front of the text.
- Tables become tables with their cells on a grid (spans kept; column headers, row
  headers and section rows marked as headers), ready for :func:`jevex.tables.table_statements`.
- Pictures become image components (no alt text in a PDF; the image stage reads them).
  Captions and footnotes a table or picture points to become its children.
- Other text (paragraphs, footnotes, code, formulas...) becomes paragraphs. Text Docling
  splits into inline runs is joined back into one paragraph.
- Page headers and footers, and anything outside Docling's body layer, are left out, as
  the HTML cleaner strips navigation and footers.

Every component's location is a :class:`~jevex.layout.PageLocation`: the page it starts on
and its bbox there in top-left page coordinates (PDF points). A container gets the union of
its children's boxes when they all sit on one page, and no bbox when it spans pages.
Anything Docling gives no position sits on page 1 without a bbox.

OCR is off: scanned pages and images are the image stage's job. Docling downloads its
layout and table models (about 500 MB) from Hugging Face the first time it runs.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING

from jevex.layout import BBox, Component, PageLocation, TableCell, UnsupportedDocumentError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.common.reference import ProvenanceItem, RefItem
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.items.node import DocItem, NodeItem
    from docling_core.types.doc.items.table.table import TableItem

    from jevex.document import Document
    from jevex.layout import ComponentType

_WHITESPACE = re.compile(r"\s+")
_NO_SPACE_BEFORE = frozenset(".,;:!?)]}%")
_LIST_MARKER = re.compile(
    r"[•◦▪▫●○■□‣∙·*+\-–—]"  # bullets
    r"|\(?(?:\d{1,3}|[a-zA-Z]|[ivxlcdmIVXLCDM]{1,6})[.)]"  # 1.  a)  (iv)  IV.
)


class PdfLayoutError(Exception):
    """Docling couldn't convert the PDF."""


class DoclingConverter:
    """Converts PDF documents with Docling's standard pipeline, OCR off.

    The Docling converter loads its models on first use and is reused after that. Calls
    are serialised, since the converter isn't safe to share between threads.
    """

    def __init__(self) -> None:
        self._converter: DocumentConverter | None = None
        self._lock = threading.Lock()

    def __call__(self, document: Document) -> DoclingDocument:
        from docling.datamodel.base_models import ConversionStatus, DocumentStream

        source = DocumentStream(name="document.pdf", stream=BytesIO(document.content))
        with self._lock:
            if self._converter is None:
                self._converter = _docling_converter()
            result = self._converter.convert(source, raises_on_error=False)
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            errors = "; ".join(e.error_message for e in result.errors) or "no details"
            raise PdfLayoutError(f"Docling couldn't convert the PDF ({result.status}): {errors}")
        return result.document


def _docling_converter() -> DocumentConverter:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        # Heading levels from font style need the parsed pages.
        generate_parsed_pages=True,
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


@dataclass
class PdfLayoutParser:
    """The default :class:`~jevex.interfaces.LayoutParser` for PDFs, built on Docling.

    ``convert`` turns a PDF into a ``DoclingDocument``. The default runs Docling locally
    (:class:`DoclingConverter`); pass your own to change Docling's options or to call a
    Docling service. It runs in a worker thread, since conversion is CPU-bound and takes
    seconds per page.
    """

    convert: Callable[[Document], DoclingDocument] = field(default_factory=DoclingConverter)

    def supports(self, document: Document) -> bool:
        return document.is_pdf

    async def parse(self, document: Document) -> Component:
        if not document.is_pdf:
            raise UnsupportedDocumentError(
                f"PdfLayoutParser reads PDFs, not {document.content_type}"
            )
        return from_docling(await asyncio.to_thread(self.convert, document))


def from_docling(doc: DoclingDocument) -> Component:
    """Map a ``DoclingDocument`` to a component tree rooted at a ``section``.

    Component ids are ``c0``, ``c1``... in reading order, so they are stable for the same
    document.
    """
    blocks = _group(_Mapper(doc).children(doc.body))
    if len(blocks) == 1 and blocks[0].type == "section":
        # A document that starts with its own heading needs no extra level.
        blocks = blocks[0].children
    return _Converter().convert(_container("section", blocks))


# --- Docling items -----------------------------------------------------------------------


@dataclass
class _Block:
    """A component before ids and heading trails are assigned."""

    type: ComponentType
    text: str = ""
    children: list[_Block] = field(default_factory=list["_Block"])
    level: int = 0
    """A heading's rank (1 is highest). On a section, the rank of the heading opening it."""
    cells: list[TableCell] = field(default_factory=list[TableCell])
    page: int | None = None
    bbox: BBox | None = None


class _Mapper:
    def __init__(self, doc: DoclingDocument) -> None:
        self.doc = doc
        self.attached: set[str] = set()
        """Captions and footnotes placed under the table or picture that points to them."""
        for item in (*doc.tables, *doc.pictures):
            self.attached.update(r.cref for r in (*item.captions, *item.footnotes))

    def children(
        self, node: NodeItem, *, attach: Sequence[RefItem] = (), in_list: bool = False
    ) -> list[_Block]:
        """Blocks for the ``attach`` items and then the node's children, in order. List
        items outside a list (``in_list`` false) are wrapped in one."""
        own = {r.cref for r in attach}
        seen: set[str] = set()  # a caption can be both attached and a child
        blocks: list[_Block] = []
        for ref in [*attach, *node.children]:
            if ref.cref in seen or (ref.cref in self.attached and ref.cref not in own):
                continue
            seen.add(ref.cref)
            blocks.extend(self.item(ref.resolve(self.doc)))
        return blocks if in_list else _wrap_loose_items(blocks)

    def item(self, item: NodeItem) -> list[_Block]:
        from docling_core.types.doc.common.content_layer import ContentLayer
        from docling_core.types.doc.items.group import GroupItem
        from docling_core.types.doc.items.picture.picture import PictureItem
        from docling_core.types.doc.items.table.table import TableItem
        from docling_core.types.doc.items.text import (
            ListItem,
            SectionHeaderItem,
            TextItem,
            TitleItem,
        )
        from docling_core.types.doc.labels import DocItemLabel, GroupLabel

        if item.content_layer != ContentLayer.BODY:
            return []
        if isinstance(item, GroupItem):
            if item.label in (GroupLabel.LIST, GroupLabel.ORDERED_LIST):
                items = self.children(item, in_list=True)
                return [_container("list", items)] if items else []
            if item.label == GroupLabel.INLINE:
                return self._inline(item)
            return self.children(item)  # sections, forms, key/value areas...: no level
        if isinstance(item, TableItem):
            return self._table(item)
        if isinstance(item, PictureItem):
            children = self.children(item, attach=[*item.captions, *item.footnotes])
            return [self._located(_Block("image", children=children), item)]
        if not isinstance(item, TextItem):
            return self.children(item)  # key/value and form items: their text, if any
        if item.label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
            return []
        text = _clean(item.text)
        if isinstance(item, ListItem):
            if item.marker and not _LIST_MARKER.fullmatch(item.marker.strip()):
                text = _clean(f"{item.marker} {item.text}")
            block = _Block("list_item", text, children=self.children(item))
            return [self._located(block, item)] if text or block.children else []
        if isinstance(item, TitleItem):
            block = _Block("heading", text, level=1)
        elif isinstance(item, SectionHeaderItem):
            block = _Block("heading", text, level=item.level + 1)
        elif item.label == DocItemLabel.CAPTION:
            block = _Block("caption", text)
        else:
            block = _Block("paragraph", text)
        return ([self._located(block, item)] if text else []) + self.children(item)

    def _inline(self, group: NodeItem) -> list[_Block]:
        """One paragraph for text Docling split into runs (a link or bold word mid-line)."""
        from docling_core.types.doc.items.node import DocItem
        from docling_core.types.doc.items.text import TextItem

        texts: list[str] = []
        located: list[DocItem] = []
        for item in _descendants(self.doc, group):
            if isinstance(item, TextItem) and (text := _clean(item.text)):
                texts.append(text)
            if isinstance(item, DocItem):
                located.append(item)
        if not texts:
            return []
        block = _Block("paragraph", _join_runs(texts))
        block.page, block.bbox = _where(self.doc, [p for item in located for p in item.prov])
        return [block]

    def _located(self, block: _Block, item: DocItem) -> _Block:
        block.page, block.bbox = _where(self.doc, item.prov)
        return block

    def _table(self, item: TableItem) -> list[_Block]:
        cells: dict[tuple[int, int], TableCell] = {}
        for c in item.data.table_cells:
            position = (c.start_row_offset_idx, c.start_col_offset_idx)
            text = _clean(c.text)
            if not text or position in cells:  # a spanning cell can be listed per slot
                continue
            cells[position] = TableCell(
                row=c.start_row_offset_idx,
                col=c.start_col_offset_idx,
                text=text,
                header=c.column_header or c.row_header or c.row_section,
                row_span=max(1, c.end_row_offset_idx - c.start_row_offset_idx),
                col_span=max(1, c.end_col_offset_idx - c.start_col_offset_idx),
            )
        captions = self.children(item, attach=[*item.captions, *item.footnotes])
        if not cells:
            return captions
        grid = [cells[p] for p in sorted(cells)]
        by_row: dict[int, list[str]] = {}
        for cell in grid:
            by_row.setdefault(cell.row, []).append(cell.text)
        text = "\n".join(" | ".join(texts) for texts in by_row.values())
        return [self._located(_Block("table", text, children=captions, cells=grid), item)]


def _descendants(doc: DoclingDocument, node: NodeItem) -> list[NodeItem]:
    out: list[NodeItem] = []
    for ref in node.children:
        child = ref.resolve(doc)
        out.append(child)
        out.extend(_descendants(doc, child))
    return out


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _join_runs(texts: list[str]) -> str:
    """Inline runs joined with spaces, but none before closing punctuation ("640 litres" +
    "." is "640 litres.") or after an opening bracket."""
    out = texts[0]
    for text in texts[1:]:
        out += text if text[0] in _NO_SPACE_BEFORE or out[-1] in "([{" else f" {text}"
    return out


def _wrap_loose_items(blocks: list[_Block]) -> list[_Block]:
    """Wrap each run of list items found outside a list group in a list."""
    out: list[_Block] = []
    run: list[_Block] = []
    for block in [*blocks, None]:
        if block is not None and block.type == "list_item":
            run.append(block)
            continue
        if run:
            out.append(_container("list", run))
            run = []
        if block is not None:
            out.append(block)
    return out


def _group(blocks: list[_Block]) -> list[_Block]:
    """Wrap each heading and the blocks after it (up to a heading of the same or higher
    rank) in an implicit section, recursively."""
    out: list[_Block] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.type != "heading":
            out.append(block)
            i += 1
            continue
        end = i + 1
        while end < len(blocks) and not (
            blocks[end].type == "heading" and blocks[end].level <= block.level
        ):
            end += 1
        section = _container("section", [block, *_group(blocks[i + 1 : end])])
        section.level = block.level
        out.append(section)
        i = end
    return out


# --- Locations ---------------------------------------------------------------------------


def _container(kind: ComponentType, children: list[_Block]) -> _Block:
    """A block holding ``children``, located where they are."""
    block = _Block(kind, children=children)
    block.page, block.bbox = _span(children)
    return block


def _span(blocks: list[_Block]) -> tuple[int | None, BBox | None]:
    """The first page among ``blocks``, and their union bbox if they're all on it."""
    placed = [b for b in blocks if b.page is not None]
    if not placed:
        return None, None
    page = placed[0].page
    boxes = [b.bbox for b in placed if b.page == page]
    if len(boxes) < len(placed) or any(b is None for b in boxes):
        return page, None
    return page, _union([b for b in boxes if b is not None])


def _union(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def _where(doc: DoclingDocument, provs: list[ProvenanceItem]) -> tuple[int | None, BBox | None]:
    """The page an item starts on, and the union of its boxes on that page."""
    if not provs:
        return None, None
    page = provs[0].page_no
    boxes = [_box(doc, p) for p in provs if p.page_no == page]
    if any(b is None for b in boxes):
        return page, None
    return page, _union([b for b in boxes if b is not None])


def _box(doc: DoclingDocument, prov: ProvenanceItem) -> BBox | None:
    """A provenance box in top-left page coordinates, or ``None`` when the page's height
    (needed to flip a bottom-left box) is unknown."""
    from docling_core.types.doc.base import CoordOrigin

    bbox = prov.bbox
    if bbox.coord_origin != CoordOrigin.TOPLEFT:
        page = doc.pages.get(prov.page_no)
        if page is None:
            return None
        bbox = bbox.to_top_left_origin(page.size.height)
    return BBox(
        x0=min(bbox.l, bbox.r),
        y0=min(bbox.t, bbox.b),
        x1=max(bbox.l, bbox.r),
        y1=max(bbox.t, bbox.b),
    )


# --- Components --------------------------------------------------------------------------


class _Converter:
    """Turns blocks into components, numbering ids and tracking the heading trail."""

    def __init__(self) -> None:
        self._next = 0
        self._headings: list[tuple[int, str]] = []
        self._floor = 0
        """Headings below this index belong to enclosing containers (lists, tables,
        images), which a heading inside one doesn't close."""

    def convert(self, block: _Block) -> Component:
        component_id = f"c{self._next}"
        self._next += 1
        implicit = block.type == "section" and block.level > 0
        if implicit:
            while len(self._headings) > self._floor and self._headings[-1][0] >= block.level:
                self._headings.pop()
        trail = [text for _, text in self._headings]
        outer, outer_floor = list(self._headings), self._floor
        if not implicit:
            self._floor = len(self._headings)
        children = [self.convert(child) for child in block.children]
        # Headings seen inside a container stay inside it.
        self._headings, self._floor = outer, outer_floor
        if block.type == "heading":
            self._headings.append((block.level, block.text))
        return Component(
            id=component_id,
            type=block.type,
            text=block.text,
            children=children,
            heading_trail=trail,
            location=PageLocation(page=block.page or 1, bbox=block.bbox),
            cells=block.cells,
        )

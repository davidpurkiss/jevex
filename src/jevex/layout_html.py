"""Stage 5 for HTML: segment a page into the generic component tree.

:class:`HtmlLayoutParser` builds a light DOM with the standard library's
:class:`html.parser.HTMLParser`, recovering from sloppy markup the way browsers do
(implied ``</p>``, ``</li>``, ``</td>``..., ``<tbody>`` inserted, stray end tags ignored), so
each component's ``dom_path`` matches the one a browser's devtools would show for the page
it was given. It then walks the DOM and maps it to components:

- ``section``, ``article``, ``main`` and ``nav`` become sections, ``aside`` a breakout.
- ``h1``–``h6`` (and ``role="heading"``) become headings. A heading also opens an implicit
  section holding everything after it up to the next heading of the same or higher rank,
  so a heading per trim or engine comes with its subtree.
- ``ul``/``ol`` become lists of list items, and a ``dl`` a list with one ``term: value``
  item per pair.
- A ``table`` becomes a table with its cells on a grid (spans resolved, header cells
  marked, including ``td`` labels set only in bold). Tables used for page layout (nested
  tables or headings in cells, or ``role="presentation"``) are read as plain containers
  instead.
- A ``figure`` holding one image or table attaches its ``figcaption`` to it.
- Any other text becomes paragraphs: one per block element (``p``, ``div``, ``li``...),
  and one per run of loose text between blocks. Wrapper ``div`` elements add no level.
- ``img`` becomes an image component whose text is its alt text and whose ``src`` is its
  URL (a lazy loader's ``data-src`` first, then ``src``, then the first ``srcset``
  candidate), resolved against the document's URL.

Hidden content (``hidden``, inline ``display:none``) and
non-rendered elements (scripts, styles, forms' option lists, SVG...) are skipped. A
``template`` is skipped too, except a declarative shadow root (``shadowrootmode``), which
browsers render in place. The
parser never produces ``column`` components: HTML columns come from CSS, which it doesn't
read.

Every component carries its ``heading_trail``: the text of the headings it sits under.
Headings inside a sectioning element (``section``, ``aside``...) don't leak out of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from jevex.clean import html_text_of, readable
from jevex.layout import Component, DomLocation, TableCell, UnsupportedDocumentError

if TYPE_CHECKING:
    from jevex.document import Document
    from jevex.layout import ComponentType

MAX_DEPTH = 256
"""Deepest element nesting kept. Old pages with thousands of unclosed ``<font>`` tags would
otherwise nest that deep. Past this depth a new element closes the innermost one and takes
its place as a sibling (as Blink does), so blocks still break the text. Table structure,
templates and skipped elements are never closed this way (see ``MAX_DEPTH_HARD``)."""

MAX_DEPTH_HARD = MAX_DEPTH + 64
"""Where the stack stops growing at all, for tables nested in cells without end: new elements
are added but not opened, so their content joins the innermost element."""

MAX_COMPONENT_DEPTH = 64
"""Deepest component nesting below the root. Deeper components are flattened into their
ancestor at this depth, so the tree stays within what JSON parsers accept (Pydantic's stops at about
100 levels of components)."""

MAX_COL_SPAN = 1000
"""The HTML limit on ``colspan``; ``rowspan`` is clamped to the rows left in the table."""

SECTION_TAGS = frozenset({"section", "article", "main", "nav"})

SKIP_TAGS = frozenset(
    {
        "audio",
        "canvas",
        "datalist",
        "embed",
        "head",
        "iframe",
        "math",
        "noscript",
        "object",
        "script",
        "select",
        "style",
        "svg",
        "template",
        "textarea",
        "title",
        "video",
    }
)
"""Elements whose content is never rendered as page text."""

BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "center",
        "dd",
        "details",
        "dialog",
        "dir",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "legend",
        "li",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "search",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
"""Elements that start a new block. Anything else is inline, unless it contains a block,
in which case it is read as a wrapper (as browsers lay out ``<a><div>...</div></a>``)."""

_HEADINGS = {f"h{n}": n for n in range(1, 7)}
_LIST_TAGS = frozenset({"ul", "ol", "menu", "dir"})
_HEAD_TAGS = frozenset({"base", "link", "meta", "noscript", "script", "style", "template", "title"})
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "keygen",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_TABLE_SECTIONS = frozenset({"thead", "tbody", "tfoot"})
_CELLS = frozenset({"td", "th"})
_BOLD = frozenset({"b", "strong"})
_TABLE_PARTS = _TABLE_SECTIONS | _CELLS | {"tr", "caption", "colgroup", "col"}
_TABLE_CONTEXT = _TABLE_SECTIONS | {"table", "tr"}
"""Where only table parts belong: other content found here is moved in front of the table."""
_CAP_KEEP = _TABLE_PARTS | _LIST_TAGS | {"table", "template", "dl", "li", "dt", "dd"}
"""Elements the depth cap never closes: closing them would change what is shown, or undo a
table's or list's structure."""
_IN_TABLE = _TABLE_PARTS | {"script", "style", "template", "input"}

# Start tags that close an open <p> (the HTML parsing algorithm's list).
_CLOSES_P = (BLOCK_TAGS - _TABLE_PARTS - {"body", "li", "dd", "dt", "caption"}) | {
    "li",
    "dd",
    "dt",
    "xmp",
    "plaintext",
    "listing",
}
# Elements an end tag (or an implied end) can't reach past. A stray </div> inside a cell
# doesn't close a <div> the table sits in.
_SCOPE = frozenset({"html", "body", "table", "td", "th", "caption", "template", "button"})

_WHITESPACE = re.compile(r"[ \t\n\r\f]+")  # HTML whitespace; a no-break space stays
_SPACES = re.compile(r" {2,}")
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)
_LEADING_INT = re.compile(r"\s*(\d+)")


class HtmlLayoutParser:
    """The default :class:`~jevex.interfaces.LayoutParser` for HTML and XHTML documents."""

    def supports(self, document: Document) -> bool:
        return document.is_html

    async def parse(self, document: Document) -> Component:
        if not document.is_html:
            raise UnsupportedDocumentError(
                f"HtmlLayoutParser reads HTML, not {document.content_type}"
            )
        return parse_html(html_text_of(document.content), base_url=document.url)


def parse_html(markup: str, *, base_url: str | None = None) -> Component:
    """Segment HTML markup into a component tree rooted at a ``section`` for ``<body>``.

    Component ids are ``c0``, ``c1``... in reading order, so they are stable for the same
    markup. Image URLs are resolved against ``base_url`` when it's given.
    """
    builder = _TreeBuilder()
    builder.feed(readable(markup))  # markup passed in may still hold escaped bytes
    builder.close()
    segmenter = _Segmenter(base_url)
    body = builder.ensure_body()
    root = _Block("section", segmenter.path(body), children=segmenter.contents(body))
    return _Converter().convert(root)


# --- DOM ---------------------------------------------------------------------------------


@dataclass(eq=False)
class _Node:
    tag: str
    attrs: dict[str, str]
    parent: _Node | None = None
    children: list[_Node | str] = field(default_factory=list["_Node | str"])
    index: int = 1
    """1-based position among the parent's children with the same tag."""
    counts: dict[str, int] = field(default_factory=dict[str, int])
    has_block: bool = False
    """Whether a block element sits anywhere inside."""
    skip: bool = False

    def elements(self) -> list[_Node]:
        return [c for c in self.children if isinstance(c, _Node) and not c.skip]

    def append(self, tag: str, attrs: dict[str, str], *, before: _Node | None = None) -> _Node:
        """Add a child element, last or in front of ``before``.

        Only the element being built is ever inserted in front of, so a new element is
        still the last of its tag and its ``index`` is the running count.
        """
        skip = self.skip or (tag in SKIP_TAGS and not _shadow_root(tag, attrs)) or _hidden(attrs)
        node = _Node(tag, attrs, parent=self, skip=skip)
        node.index = self.counts[tag] = self.counts.get(tag, 0) + 1
        if before is None:
            self.children.append(node)
        else:
            self.children.insert(self.children.index(before), node)
        if not skip and (tag in BLOCK_TAGS or tag in _HEADINGS):
            ancestor: _Node | None = self
            while ancestor is not None and not ancestor.has_block:
                ancestor.has_block = True
                ancestor = ancestor.parent
        return node


def _shadow_root(tag: str, attrs: dict[str, str]) -> bool:
    """A declarative shadow DOM ``template``: browsers render its content in place."""
    return tag == "template" and ("shadowrootmode" in attrs or "shadowroot" in attrs)


def _hidden(attrs: dict[str, str]) -> bool:
    # aria-hidden content is still on screen, so it stays.
    return "hidden" in attrs or _HIDDEN_STYLE.search(attrs.get("style", "")) is not None


class _TreeBuilder(HTMLParser):
    """Builds a :class:`_Node` tree, fixing up markup the way browsers do.

    Only the recoveries that change which block a piece of text lands in, or the element
    path to it, are implemented: implied ``html``/``head``/``body``, implied end tags for
    ``p``, ``li``, ``dt``/``dd``, table parts and headings, ``<tbody>`` around bare rows,
    content misplaced between table rows moved in front of the table, and end tags that
    have no matching open element in scope being ignored.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html = _Node("html", {})
        self.head: _Node | None = None
        self.body: _Node | None = None
        self.stack: list[_Node] = [self.html]

    def ensure_body(self) -> _Node:
        if self.body is None:
            del self.stack[1:]  # closes <head> and anything left open in it
            self.body = self.html.append("body", {})
            self.stack.append(self.body)
        return self.body

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Browsers keep the first of duplicated attributes.
        values: dict[str, str] = {}
        for name, value in attrs:
            values.setdefault(name, value or "")
        self._start(tag, values)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # In HTML, "<div/>" opens a <div>; only void elements are self-closing. Inside
        # skipped content (SVG, MathML) "/>" does close, which keeps icons shallow.
        if self.stack[-1].skip and tag not in _VOID_TAGS:
            self._start(tag, dict.fromkeys((name for name, _ in attrs), ""), closed=True)
        else:
            self.handle_starttag(tag, attrs)

    def _start(self, tag: str, attrs: dict[str, str], *, closed: bool = False) -> None:
        if tag == "html":
            return
        if tag == "head":
            if self.head is None and self.body is None:
                self.head = self.html.append("head", attrs)
                self.stack.append(self.head)
            return
        if tag == "body":
            self.ensure_body()
            return
        if self.body is None and tag not in _HEAD_TAGS:
            self.ensure_body()
        self._imply_end_tags(tag)
        # Void elements are never opened, so only other tags need room on the stack.
        if tag not in _VOID_TAGS and len(self.stack) >= MAX_DEPTH and not self._make_room():
            # Nothing left to close: add it without opening it, so its content joins the
            # innermost element. A skipped element still opens (once), so its content
            # stays hidden.
            top = self.stack[-1]
            node = top.append(tag, attrs)
            if node.skip and not top.skip:
                self.stack.append(node)
            return
        parent = self.stack[-1]
        table = self._open_table()
        if parent.tag in _TABLE_CONTEXT and tag not in _IN_TABLE and table is not None:
            if tag == "form":
                parent.append(tag, attrs)  # browsers keep it, empty, and read on
                return
            # Anything else between rows is shown in front of the table ("foster parenting").
            assert table.parent is not None
            parent, before = table.parent, table
        else:
            before = None
        if tag == "tr" and parent.tag == "table":
            parent = self._push(parent.append("tbody", {}))
        elif tag in _CELLS and parent.tag in _TABLE_SECTIONS | {"table"}:
            if parent.tag == "table":
                parent = self._push(parent.append("tbody", {}))
            parent = self._push(parent.append("tr", {}))
        node = parent.append(tag, attrs, before=before)
        if not closed and tag not in _VOID_TAGS:
            self.stack.append(node)

    def _make_room(self) -> bool:
        """At the depth cap, close elements so the next one opens as a sibling.

        Table and list structure, templates and skipped subtrees stay open, so the stack can pass
        the cap by those (tables nested in cells, for instance). Past ``MAX_DEPTH_HARD``
        this gives up and returns False.
        """
        while len(self.stack) >= MAX_DEPTH:
            node = self.stack[-1]
            parent = node.parent
            keep = node.tag in _CAP_KEEP or (node.skip and parent is not None and not parent.skip)
            if keep:
                break
            self.stack.pop()
        return len(self.stack) < MAX_DEPTH_HARD

    def _push(self, node: _Node) -> _Node:
        self.stack.append(node)
        return node

    def _open_table(self) -> _Node | None:
        """The innermost open table, unless a ``template`` (its own context) is nearer."""
        for node in reversed(self.stack):
            if node.tag == "template":
                return None
            if node.tag == "table":
                return node
        return None

    def _imply_end_tags(self, tag: str) -> None:
        if tag in _TABLE_PARTS and self._open_table() is not None:
            # Back to the table's own structure: close what was opened in a cell or in
            # front of the table, then an unclosed caption or colgroup.
            while self.stack[-1].tag not in _TABLE_PARTS | {"table"}:
                self.stack.pop()
            if tag in ("caption", "colgroup"):
                while self.stack[-1].tag != "table":
                    self.stack.pop()
            elif tag != "col":
                self._close("caption", "colgroup", stop=frozenset({"table"}))
        if tag == "table" and self.stack[-1].tag in _TABLE_CONTEXT:
            self._close("table", stop=frozenset())  # a table between rows ends the open one
        if tag in _CLOSES_P or tag in _HEADINGS or tag == "table":
            self._close("p", stop=_SCOPE)
        if tag in _HEADINGS and self.stack[-1].tag in _HEADINGS:
            self.stack.pop()
        elif tag == "li":
            self._close("li", stop=_SCOPE | _LIST_TAGS)
        elif tag in ("dt", "dd"):
            self._close("dt", "dd", stop=_SCOPE | {"dl"})
        elif tag in _TABLE_SECTIONS:
            self._close(*_TABLE_SECTIONS, stop=frozenset({"table"}))
        elif tag == "tr":
            self._close("tr", stop=frozenset({"table"} | _TABLE_SECTIONS))
        elif tag in _CELLS:
            self._close(*_CELLS, stop=frozenset({"table", "tr"}))

    def _close(self, *tags: str, stop: frozenset[str]) -> bool:
        """Close the nearest open element named in ``tags``, unless ``stop`` comes first.

        A ``template`` always stops the search: its content is a context of its own.
        """
        for i in range(len(self.stack) - 1, 0, -1):
            node = self.stack[i]
            if node.tag in tags:
                del self.stack[i:]
                return True
            if node.tag in stop or node.tag == "template":
                return False
        return False

    def handle_endtag(self, tag: str) -> None:
        if tag in ("html", "body"):
            return  # content after </body> still belongs to the body, as in browsers
        if tag in _HEADINGS:
            self._close(*_HEADINGS, stop=_SCOPE)  # "<h2>...</h3>" still ends the heading
            return
        if tag == "br":
            self._start("br", {})  # browsers read "</br>" as "<br>"
            return
        if tag == "p":
            # A "</p>" with no open <p> makes an empty one, which still breaks the text.
            if not self._close("p", stop=_SCOPE) and self.body is not None:
                self._start("p", {})
                self._close("p", stop=_SCOPE)
            return
        stop = _SCOPE
        if tag == "template":
            stop = frozenset[str]()  # closes back to the template, whatever is left open
        elif tag == "table":
            stop = frozenset[str]()
        elif tag in _TABLE_PARTS:
            stop = frozenset({"table"})
        self._close(tag, stop=stop)

    def handle_data(self, data: str) -> None:
        if self.body is None:
            if self.stack[-1].tag in _HEAD_TAGS:
                self.stack[-1].children.append(data)
                return
            if not data.strip(" \t\n\r\f"):
                return
            self.ensure_body()
        parent = self.stack[-1]
        table = self._open_table()
        if parent.tag in _TABLE_CONTEXT and table is not None and data.strip(" \t\n\r\f"):
            assert table.parent is not None
            siblings = table.parent.children
            siblings.insert(siblings.index(table), data)
            return
        parent.children.append(data)


# --- Segmenting --------------------------------------------------------------------------


@dataclass(eq=False)
class _Block:
    """A component before ids and heading trails are assigned."""

    type: ComponentType
    path: str
    text: str = ""
    children: list[_Block] = field(default_factory=list["_Block"])
    level: int = 0
    """A heading's rank (1-6). On a section, non-zero marks it as implicit: opened by a
    heading of that rank rather than a sectioning element."""
    cells: list[TableCell] = field(default_factory=list[TableCell])
    src: str | None = None


@dataclass
class _Inline:
    """Inline content gathered until the next block: text pieces and images."""

    pieces: list[str] = field(default_factory=list[str])
    images: list[_Block] = field(default_factory=list[_Block])
    alt_text: bool = False
    """Put images' alt text in the text instead of collecting them: headings, cells and
    definitions are one line of text, and a tick icon in a cell is its answer."""


class _Segmenter:
    def __init__(self, base_url: str | None = None) -> None:
        self._paths: dict[int, str] = {}
        self._base_url = base_url

    def path(self, node: _Node) -> str:
        """The element's absolute path, XPath style: ``/html/body/div[2]/p``.

        A position is given only when the parent has more than one child with the tag.
        """
        chain: list[_Node] = []
        current: _Node | None = node
        while current is not None and id(current) not in self._paths:
            chain.append(current)
            current = current.parent
        prefix = self._paths[id(current)] if current is not None else ""
        for n in reversed(chain):
            step = n.tag
            if n.parent is not None and n.parent.counts[n.tag] > 1:
                step = f"{n.tag}[{n.index}]"
            prefix = self._paths[id(n)] = f"{prefix}/{step}"
        return prefix

    def contents(self, node: _Node) -> list[_Block]:
        """The node's content, grouped into implicit sections by its headings."""
        blocks = _group(self.segment(node))
        if len(blocks) == 1 and blocks[0].type == "section" and blocks[0].level:
            # A container that starts with its own heading needs no extra level.
            return blocks[0].children
        return blocks

    def segment(self, node: _Node, *, pre: bool = False) -> list[_Block]:
        """Blocks for the node's children, with loose inline runs as paragraphs."""
        blocks: list[_Block] = []
        inline = _Inline()
        for child in node.children:
            if isinstance(child, str):
                inline.pieces.append(child if pre else _WHITESPACE.sub(" ", child))
            elif child.skip:
                continue
            elif _is_block(child):
                self._flush(inline, node, blocks, pre=pre)
                inline = _Inline()
                blocks.extend(self.block(child, pre=pre))
            else:
                self._inline(child, inline, pre=pre)
        self._flush(inline, node, blocks, pre=pre)
        return blocks

    def block(self, node: _Node, *, pre: bool = False) -> list[_Block]:
        tag = node.tag
        level = _heading_level(node)
        path = self.path(node)
        if level:
            text = self.flat_text(node)
            return [_Block("heading", path, text, level=level)] if text else []
        if tag in _LIST_TAGS:
            return self._list(node)
        if tag == "dl":
            return self._definitions(node)
        if tag == "table" and not _is_layout_table(node):
            return self._table(node)
        if tag == "figure":
            return self._figure(node)
        if tag in ("figcaption", "caption"):
            text = self.flat_text(node)
            return [_Block("caption", path, text)] if text else []
        if tag in SECTION_TAGS or tag == "aside":
            children = self.contents(node)
            kind: ComponentType = "breakout" if tag == "aside" else "section"
            return [_Block(kind, path, children=children)] if children else []
        return self.segment(node, pre=pre or tag == "pre")

    def flat_text(self, node: _Node) -> str:
        """All of the node's text on one line: for headings, cells and definitions."""
        inline = _Inline(alt_text=True)
        self._inline(node, inline, pre=node.tag == "pre")
        return " ".join(_finish(inline.pieces, pre=False).split("\n"))

    def _inline(self, node: _Node, inline: _Inline, *, pre: bool) -> None:
        if node.tag == "br":
            inline.pieces.append("\n")
            return
        if node.tag == "img":
            image = self._image(node)
            if image is None:
                pass
            elif inline.alt_text:
                inline.pieces.append(f" {image.text} ")
            else:
                inline.images.append(image)
            return
        pre = pre or node.tag == "pre"
        breaks = _is_block(node)
        if breaks:
            inline.pieces.append("\n")
        for child in node.children:
            if isinstance(child, str):
                inline.pieces.append(child if pre else _WHITESPACE.sub(" ", child))
            elif not child.skip:
                self._inline(child, inline, pre=pre)
        if breaks:
            inline.pieces.append("\n")

    def _flush(self, inline: _Inline, node: _Node, blocks: list[_Block], *, pre: bool) -> None:
        text = _finish(inline.pieces, pre=pre)
        if text:
            blocks.append(_Block("paragraph", self.path(node), text, children=inline.images))
        else:
            blocks.extend(inline.images)

    def _image(self, node: _Node) -> _Block | None:
        if any(_int_attr(node, name, 2) <= 1 for name in ("width", "height")):
            return None  # tracking pixel or spacer
        alt = _WHITESPACE.sub(" ", node.attrs.get("alt", "")).strip()
        src = _image_src(node)
        if not alt and src is None:
            return None
        if src is not None and self._base_url and not src.startswith("data:"):
            src = urljoin(self._base_url, src)
        return _Block("image", self.path(node), alt, src=src)

    def _list(self, node: _Node) -> list[_Block]:
        items = [item for child in node.elements() if (item := self._item(child)) is not None]
        return [_Block("list", self.path(node), children=items)] if items else []

    def _item(self, node: _Node) -> _Block | None:
        """A list item: its text blocks merged into its text, anything else as children.

        Only ``li`` is expected here, but invalid markup puts other elements (often a
        nested list) straight into a list, and browsers render them as items too.
        """
        blocks = self.segment(node) if node.tag == "li" else self.block(node)
        lines: list[str] = []
        children: list[_Block] = []
        for b in blocks:
            if b.type in ("paragraph", "heading", "caption"):
                lines.append(b.text)
                children.extend(b.children)
            else:
                children.append(b)
        text = "\n".join(lines)
        if not text and not children:
            return None
        return _Block("list_item", self.path(node), text, children=children)

    def _definitions(self, node: _Node) -> list[_Block]:
        """One ``term: value`` item per ``dd``; terms without a value become items alone."""
        items: list[_Block] = []
        terms: list[str] = []
        last_term: _Node | None = None
        answered = False
        for child in _definition_parts(node):
            text = self.flat_text(child)
            if child.tag == "dt":
                if answered:
                    terms, answered = [], False
                if text:
                    terms.append(text.rstrip(":").rstrip())
                    last_term = child
            elif text:
                label = ", ".join(terms)
                items.append(
                    _Block("list_item", self.path(child), f"{label}: {text}" if label else text)
                )
                answered = True
        if terms and not answered and last_term is not None:
            items.append(_Block("list_item", self.path(last_term), ", ".join(terms)))
        return [_Block("list", self.path(node), children=items)] if items else []

    def _table(self, node: _Node) -> list[_Block]:
        captions: list[_Block] = []
        rows: list[tuple[_Node, bool]] = []
        foot: list[tuple[_Node, bool]] = []  # rendered last, wherever it is in the markup
        for child in node.elements():
            if child.tag == "caption":
                captions.extend(self.block(child))
            elif child.tag in _TABLE_SECTIONS:
                group = foot if child.tag == "tfoot" else rows
                group.extend(
                    (tr, child.tag == "thead") for tr in child.elements() if tr.tag == "tr"
                )
            elif child.tag == "tr":
                rows.append((child, False))
        rows += foot
        cells: list[TableCell] = []
        bold: set[tuple[int, int]] = set()  # (row, col) of td cells whose text is all bold
        busy: dict[int, int] = {}  # column -> first row where it is free again
        for r, (tr, in_head) in enumerate(rows):
            c = 0
            for cell in tr.elements():
                if cell.tag not in _CELLS:
                    continue
                while busy.get(c, 0) > r:
                    c += 1
                row_span = _int_attr(cell, "rowspan", 1) or len(rows) - r  # 0: to the end
                row_span = max(1, min(row_span, len(rows) - r))
                col_span = max(1, min(_int_attr(cell, "colspan", 1), MAX_COL_SPAN))
                for k in range(c, c + col_span):
                    busy[k] = r + row_span
                text = self.flat_text(cell)
                if text:
                    if cell.tag == "td" and not in_head and _bold_only(cell):
                        bold.add((r, c))
                    cells.append(
                        TableCell(
                            row=r,
                            col=c,
                            text=text,
                            header=in_head or cell.tag == "th",
                            row_span=row_span,
                            col_span=col_span,
                        )
                    )
                c += col_span
        if not cells:
            return captions
        if bold:
            cells = _bold_headers(cells, bold)
        by_row: dict[int, list[str]] = {}
        for cell in cells:
            by_row.setdefault(cell.row, []).append(cell.text)
        text = "\n".join(" | ".join(texts) for texts in by_row.values())
        return [_Block("table", self.path(node), text, children=captions, cells=cells)]

    def _figure(self, node: _Node) -> list[_Block]:
        blocks = self.segment(node)
        captions = [b for b in blocks if b.type == "caption"]
        media = [b for b in blocks if b.type != "caption"]
        if len(media) == 1 and media[0].type in ("image", "table"):
            media[0].children.extend(captions)
            return media
        return [_Block("section", self.path(node), children=blocks)] if blocks else []


def _is_block(node: _Node) -> bool:
    return node.tag in BLOCK_TAGS or node.has_block or _heading_level(node) > 0


def _heading_level(node: _Node) -> int:
    if node.tag in _HEADINGS:
        return _HEADINGS[node.tag]
    if node.attrs.get("role", "").strip().lower() == "heading":
        return max(1, min(_int_attr(node, "aria-level", 2), 6))
    return 0


def _is_layout_table(table: _Node) -> bool:
    """Whether a table arranges the page rather than holding data: it has a nested table,
    or headings in its data cells (a heading in a ``th`` is just a header)."""
    if table.attrs.get("role", "").strip().lower() in ("presentation", "none"):
        return True
    pending = [(child, False) for child in table.elements()]
    while pending:
        node, in_td = pending.pop()
        if node.tag == "table" or (in_td and _heading_level(node)):
            return True
        in_td = in_td or node.tag == "td"
        pending.extend((child, in_td) for child in node.elements())
    return False


def _bold_only(cell: _Node) -> bool:
    """Whether all of a cell's text is in ``b``/``strong`` (a colon after it aside)."""
    found = False
    pending: list[tuple[_Node | str, bool]] = [(child, False) for child in cell.children]
    while pending:
        node, inside = pending.pop()
        if isinstance(node, str):
            if node.replace(":", "").strip():
                if not inside:
                    return False
                found = True
        elif not node.skip:
            if node.tag == "img" and not inside and node.attrs.get("alt", "").strip():
                return False  # its alt text is part of the cell's text
            inside = inside or node.tag in _BOLD
            pending.extend((child, inside) for child in node.children)
    return found


def _bold_headers(cells: list[TableCell], bold: set[tuple[int, int]]) -> list[TableCell]:
    """Mark bold-only ``td`` cells as headers where headers go.

    Many sites build tables from ``td`` alone and set their labels in bold. A bold cell
    counts as a header in the leading rows made only of headers and bold cells (unless
    every row is, as in a table set all in bold, or the table is two columns of labels
    and values), or in the first column when that column's cells below the header rows
    all are. A bold value elsewhere (a total, a
    highlighted price) stays data.
    """

    def labelled(c: TableCell) -> bool:
        return c.header or (c.row, c.col) in bold

    rows: dict[int, list[TableCell]] = {}
    for c in cells:
        rows.setdefault(c.row, []).append(c)
    header_rows: set[int] = set()
    for r in sorted(rows):
        if not all(labelled(c) for c in rows[r]):
            break
        header_rows.add(r)
    else:
        header_rows = set()
    width = max(c.col + c.col_span for c in cells)
    if width == 2 and all(labelled(c) for c in cells if c.col == 0):
        # Labels and values: a bold first value ("Engine | 1.5 TSI") isn't a column header.
        header_rows = set()
    first_column = [c for c in cells if c.col == 0 and c.row not in header_rows]
    column = bool(first_column) and all(labelled(c) for c in first_column)
    return [
        c.model_copy(update={"header": True})
        if (c.row, c.col) in bold and (c.row in header_rows or (column and c.col == 0))
        else c
        for c in cells
    ]


def _definition_parts(dl: _Node) -> list[_Node]:
    """The ``dt`` and ``dd`` elements of a list, including those grouped in ``div``s."""
    parts: list[_Node] = []
    for child in dl.elements():
        if child.tag in ("dt", "dd"):
            parts.append(child)
        elif child.tag == "div":
            parts.extend(g for g in child.elements() if g.tag in ("dt", "dd"))
    return parts


def _int_attr(node: _Node, name: str, default: int) -> int:
    match = _LEADING_INT.match(node.attrs.get(name, ""))
    return int(match.group(1)) if match else default


def _image_src(node: _Node) -> str | None:
    """An ``img``'s URL: ``data-src`` first (lazy loaders keep the real image there and a
    placeholder in ``src``), then ``src``, then the first ``srcset`` candidate."""
    for name in ("data-src", "src"):
        if value := node.attrs.get(name, "").strip():
            return value
    # "a.jpg 1x, b.jpg 2x" or "a.jpg, b.jpg": the URL runs to the first space or comma.
    words = node.attrs.get("srcset", "").split()
    return (words[0].rstrip(",") or None) if words else None


def _finish(pieces: list[str], *, pre: bool) -> str:
    """Join inline pieces into text: one line per ``<br>``, blank lines dropped.

    Outside ``pre``, spaces are collapsed and lines trimmed. Inside, lines keep their
    indentation and only blank lines at the ends go.
    """
    lines = "".join(pieces).split("\n")
    if pre:
        lines = [line.rstrip() for line in lines]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)
    lines = [_SPACES.sub(" ", line).strip(" ") for line in lines]
    return "\n".join(line for line in lines if line)


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
        out.append(
            _Block(
                "section",
                block.path,
                children=[block, *_group(blocks[i + 1 : end])],
                level=block.level,
            )
        )
        i = end
    return out


# --- Components --------------------------------------------------------------------------


class _Converter:
    """Turns blocks into components, numbering ids and tracking the heading trail."""

    def __init__(self) -> None:
        self._next = 0
        self._headings: list[tuple[int, str]] = []
        self._floor = 0
        """Headings below this index belong to enclosing sectioning elements, which a
        heading inside one doesn't close, whatever its rank."""

    def convert(self, block: _Block, depth: int = 0) -> Component:
        component_id = f"c{self._next}"
        self._next += 1
        if block.level:
            while len(self._headings) > self._floor and self._headings[-1][0] >= block.level:
                self._headings.pop()
        trail = [text for _, text in self._headings]
        outer, outer_floor = list(self._headings), self._floor
        if not block.level:
            self._floor = len(self._headings)
        kids = block.children
        if depth + 1 >= MAX_COMPONENT_DEPTH:
            kids = _flattened(block)
        children = [self.convert(child, depth + 1) for child in kids]
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
            location=DomLocation(dom_path=block.path),
            cells=block.cells,
            src=block.src,
        )


def _flattened(block: _Block) -> list[_Block]:
    """The block's descendants in reading order, without children and minus containers."""
    out: list[_Block] = []
    pending = list(reversed(block.children))
    while pending:
        b = pending.pop()
        if b.text or b.cells or b.type == "image":
            out.append(_Block(b.type, b.path, b.text, level=b.level, cells=b.cells, src=b.src))
        pending.extend(reversed(b.children))
    return out

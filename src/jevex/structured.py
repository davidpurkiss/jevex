"""Stage 4 sources: the machine-readable data a page embeds.

:class:`EmbeddedDataReader` finds every blob of embedded data in an HTML document and
returns it as plain JSON values, one :class:`StructuredBlob` each:

- **JSON-LD** ``<script type="application/ld+json">``: one blob per top-level node, with
  ``@graph`` expanded and bare ``{"@id": ...}`` references to other nodes on the page
  inlined one level deep, so a ``Product`` carries its ``Offer`` even when the page lists
  them apart. Referenced nodes are still blobs of their own.
- **Microdata** (``itemscope``/``itemprop``) and **RDFa** (``typeof``/``property``): one
  blob per top-level item, shaped like JSON-LD (``@type``, then one key per property), so
  later stages treat every schema.org source alike. RDFa properties outside any item
  (Open Graph ``<meta property="og:...">``) form one untyped blob.
- **App state:** JSON scripts such as ``__NEXT_DATA__`` and Nuxt 3's ``__NUXT_DATA__``
  (decoded from its ``devalue`` form), and script assignments such as
  ``window.__INITIAL_STATE__ = {...}`` or Nuxt 2's ``window.__NUXT__=(function(a){...}(1))``.
  Those are JavaScript, not JSON, so a small literal parser reads them without running
  any code.
- **``data-*`` attributes:** one blob per element, minus attributes that are markup
  plumbing (test ids, framework and analytics hooks; see :data:`DATA_ATTRIBUTE_NOISE`).

Reading is CPU-only and never fails on bad data: a blob that can't be parsed is reported
in :attr:`EmbeddedData.skipped` with the reason. Turning blobs into ``structured``
statements, fingerprinting their shape and mapping key paths to fields is the next step
of the structured-data stage.

Run it on the cleaned document: :class:`~jevex.clean.BoilerplateCleaner` keeps data
scripts wherever they appear, but microdata and ``data-*`` attributes inside removed
boilerplate go with it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Literal, cast, get_args
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from jevex.clean import STATE_PATTERN, html_text_of
from jevex.layout import DomLocation

if TYPE_CHECKING:
    from jevex.document import Document

StructuredSource = Literal["json_ld", "microdata", "rdfa", "app_state", "data_attributes"]

ALL_SOURCES: frozenset[StructuredSource] = frozenset(get_args(StructuredSource))

DATA_ATTRIBUTE_NOISE = re.compile(
    r"(?:test|testid|test-id|qa|cy|e2e|automation-id"
    r"|reactid|reactroot|react-.*|v-[0-9a-f]{6,}|v-app|server-rendered|n-head|hid|emotion"
    r"|styled.*|turbo.*|controller|action|bs-.*|toggle|target|dismiss|parent|ride|slide.*"
    r"|ga|ga-.*|gtm.*|track.*|tracking.*|analytics.*|event.*|component.*|module.*"
    r"|src|srcset|sizes|lazy.*|ll-status|aos.*|slick.*|swiper.*|tooltip.*|placement"
    r"|original-title|toggle-.*|ajax.*|nosnippet"
    r"|loading-text|complete-text|reset-text)",  # Bootstrap button states
    re.IGNORECASE,
)
"""``data-*`` attribute names (without ``data-``) that carry no record data: test ids,
framework bookkeeping, Bootstrap/Stimulus hooks, analytics tags and lazy-loading. Matched
against the whole name."""

_JSON_TYPE = re.compile(r"application/(?:[-\w.]+\+)?json", re.IGNORECASE)
_JS_TYPES = frozenset({"", "module", "text/javascript", "application/javascript"})
_SCHEMA_ORG = re.compile(r"^(?:https?://(?:www\.)?schema\.org/|schema:)", re.IGNORECASE)
_STATE_ASSIGNMENT = re.compile(
    r"""(?:\b(?:window|self|globalThis)\s*(?:\.\s*|\[\s*["']))?"""
    r"""(__[A-Z][A-Z0-9_]*__)(?:["']\s*\])?\s*=(?!=)\s*"""
)
_WRAPPER_START = re.compile(r"^\s*(?:<!--|(?://|/\*)?\s*<!\[CDATA\[(?:\s*\*/)?)")
_WRAPPER_END = re.compile(r"(?:(?://|/\*)?\s*\]\]>(?:\s*\*/)?|(?://\s*)?-->)\s*$")
_SPACE = re.compile(r"\s+")

MAX_BLOB_NODES = 1_000_000
"""The most values a blob may hold once shared references are written out in full. Nuxt
payloads and function-style state share values by reference, so a few hundred bytes can
stand for a tree too large to serialise or flatten; such blobs are skipped."""

# Elements with no end tag; they never go on the open-element stack.
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
# An opening tag that implicitly closes an open sibling, as browsers do: the tags it
# closes, and the containers that stop the search.
_IMPLIED_END: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "li": (frozenset({"li"}), frozenset({"ul", "ol", "menu"})),
    "dt": (frozenset({"dt", "dd"}), frozenset({"dl"})),
    "dd": (frozenset({"dt", "dd"}), frozenset({"dl"})),
    "tr": (frozenset({"tr", "td", "th"}), frozenset({"table", "thead", "tbody", "tfoot"})),
    "td": (frozenset({"td", "th"}), frozenset({"tr", "table"})),
    "th": (frozenset({"td", "th"}), frozenset({"tr", "table"})),
    "option": (frozenset({"option"}), frozenset({"select", "datalist", "optgroup"})),
}
_CLOSES_P = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "div",
        "dl",
        "fieldset",
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
        "hr",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "ul",
    }
)
_NO_TEXT = frozenset({"script", "style", "template", "noscript"})
# Where a microdata or RDFa property takes its value from, by tag (after ``content``).
_SRC_TAGS = frozenset({"audio", "embed", "iframe", "img", "source", "track", "video"})
_HREF_TAGS = frozenset({"a", "area", "link"})


class StructuredBlob(BaseModel):
    """One piece of embedded data, as JSON values (objects, lists, strings, numbers...).

    ``types`` are the schema.org types it declares, shortened to their names
    (``https://schema.org/Car`` → ``Car``). ``name`` is the script ``id`` or variable an
    app-state blob came from (``__NEXT_DATA__``, ``window.__INITIAL_STATE__``), and
    ``None`` for the other sources. ``location`` is the element it was read from.
    """

    model_config = ConfigDict(frozen=True)

    source: StructuredSource
    data: Any
    types: list[str] = Field(default_factory=list[str])
    name: str | None = None
    location: DomLocation


class SkippedBlob(BaseModel):
    """Embedded data that was found but couldn't be read, and why."""

    model_config = ConfigDict(frozen=True)

    source: StructuredSource
    reason: str
    name: str | None = None
    location: DomLocation


class EmbeddedData(BaseModel):
    """Everything :class:`EmbeddedDataReader` found in one document.

    Blobs are grouped by source (JSON-LD, microdata, RDFa, app state, ``data-*``) and in
    document order within each.
    """

    model_config = ConfigDict(frozen=True)

    blobs: list[StructuredBlob] = Field(default_factory=list[StructuredBlob])
    skipped: list[SkippedBlob] = Field(default_factory=list[SkippedBlob])

    def of_type(self, *types: str) -> list[StructuredBlob]:
        """Blobs declaring any of the given schema.org types (short names, e.g. ``"Car"``)."""
        wanted = set(types)
        return [b for b in self.blobs if wanted.intersection(b.types)]

    def from_source(self, source: StructuredSource) -> list[StructuredBlob]:
        return [b for b in self.blobs if b.source == source]


class EmbeddedDataReader:
    """Reads JSON-LD, microdata, RDFa, app state and ``data-*`` attributes from HTML.

    ``sources`` narrows what is read. ``data_attribute_noise`` is matched against each
    ``data-*`` name (without the prefix) to leave out plumbing; pass ``None`` to keep
    every attribute. Non-HTML documents have no embedded data.
    """

    def __init__(
        self,
        *,
        sources: frozenset[StructuredSource] = ALL_SOURCES,
        data_attribute_noise: re.Pattern[str] | None = DATA_ATTRIBUTE_NOISE,
    ) -> None:
        unknown = sorted(set(sources) - ALL_SOURCES)
        if unknown:
            raise ValueError(
                f"unknown structured sources {unknown}; expected {sorted(ALL_SOURCES)}"
            )
        self.sources = sources
        self.data_attribute_noise = data_attribute_noise

    def read(self, document: Document) -> EmbeddedData:
        if not document.is_html or not self.sources:
            return EmbeddedData()
        builder = _TreeBuilder()
        builder.feed(html_text_of(document.content))
        builder.close()
        out = _Collector(builder.root, document.url)
        if "json_ld" in self.sources:
            out.json_ld()
        if "microdata" in self.sources:
            out.microdata()
        if "rdfa" in self.sources:
            out.rdfa()
        if "app_state" in self.sources:
            out.app_state()
        if "data_attributes" in self.sources:
            out.data_attributes(self.data_attribute_noise)
        return EmbeddedData(blobs=out.blobs, skipped=out.skipped)


def schema_type(name: str) -> str:
    """A schema.org type or property name without its vocabulary: ``schema:Car`` → ``Car``."""
    return _SCHEMA_ORG.sub("", name.strip())


# --- HTML tree ---------------------------------------------------------------------------


@dataclass(eq=False)
class _Node:
    tag: str
    attrs: dict[str, str]
    parent: _Node | None = None
    children: list[_Node | str] = field(default_factory=list["_Node | str"])
    index: int = 1
    """1-based position among the parent's children with the same tag."""
    counts: dict[str, int] = field(default_factory=dict[str, int])
    cached_path: str | None = field(default=None, init=False, repr=False)

    def append(self, child: _Node) -> None:
        self.counts[child.tag] = child.index = self.counts.get(child.tag, 0) + 1
        child.parent = self
        self.children.append(child)

    @property
    def path(self) -> str:
        """XPath style, as in the layout tree: ``/html/body/div[2]/p``.

        A position is given only when the parent has more than one child with the tag.
        """
        chain: list[_Node] = []
        node: _Node | None = self
        while node is not None and node.cached_path is None:
            chain.append(node)
            node = node.parent
        prefix = (node.cached_path or "") if node is not None else ""
        for n in reversed(chain):
            if n.parent is None:
                n.cached_path = prefix = ""
                continue
            step = n.tag if n.parent.counts[n.tag] == 1 else f"{n.tag}[{n.index}]"
            n.cached_path = prefix = f"{prefix}/{step}"
        return prefix

    @property
    def location(self) -> DomLocation:
        return DomLocation(dom_path=self.path or "/")

    def walk(self) -> list[_Node]:
        """This node and its descendant elements, in document order."""
        out: list[_Node] = []
        stack: list[_Node] = [self]
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(c for c in reversed(node.children) if isinstance(c, _Node))
        return out

    def raw_text(self) -> str:
        return "".join(c for c in self.children if isinstance(c, str))

    def text(self) -> str:
        """Rendered-ish text: descendant text without scripts, whitespace collapsed."""
        parts: list[str] = []
        stack: list[_Node | str] = [self]
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                parts.append(item)
            elif item.tag not in _NO_TEXT:
                stack.extend(reversed(item.children))
        return _SPACE.sub(" ", "".join(parts)).strip()


class _TreeBuilder(HTMLParser):
    """A small element tree with browser-style recovery for unclosed elements.

    An end tag closes the nearest open element with its name and everything opened
    inside it; a stray end tag is ignored. List items, table cells, definitions and
    paragraphs close their open siblings the way browsers do, so items stay separate.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("", {})
        self.open: list[_Node] = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._close_implied(tag)
        values: dict[str, str] = {}
        for name, value in attrs:
            values.setdefault(name, value or "")
        node = _Node(tag, values)
        self.open[-1].append(node)
        if tag not in _VOID_TAGS:
            self.open.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for depth in range(len(self.open) - 1, 0, -1):
            if self.open[depth].tag == tag:
                del self.open[depth:]
                return

    def handle_data(self, data: str) -> None:
        self.open[-1].children.append(data)

    def _close_implied(self, tag: str) -> None:
        if tag in _IMPLIED_END:
            closes, stops = _IMPLIED_END[tag]
            for depth in range(len(self.open) - 1, 0, -1):
                open_tag = self.open[depth].tag
                if open_tag in stops:
                    return
                if open_tag in closes:
                    del self.open[depth:]
                    return
        elif tag in _CLOSES_P and self.open[-1].tag == "p":
            self.open.pop()


# --- Collecting blobs --------------------------------------------------------------------


class _Collector:
    def __init__(self, root: _Node, url: str | None) -> None:
        self.root = root
        self.nodes = root.walk()
        base = next((n.attrs.get("href") for n in self.nodes if n.tag == "base"), None)
        self.base_url = (urljoin(url, base) if url else base) if base else url
        self.blobs: list[StructuredBlob] = []
        self.skipped: list[SkippedBlob] = []

    def add(
        self,
        source: StructuredSource,
        node: _Node,
        data: Any,
        *,
        types: list[str] | None = None,
        name: str | None = None,
    ) -> None:
        if not _fits(data, MAX_BLOB_NODES):
            self.skip(source, node, f"expands to more than {MAX_BLOB_NODES:,} values", name=name)
            return
        self.blobs.append(
            StructuredBlob(
                source=source, data=data, types=types or [], name=name, location=node.location
            )
        )

    def skip(
        self, source: StructuredSource, node: _Node, reason: str, *, name: str | None = None
    ) -> None:
        self.skipped.append(
            SkippedBlob(source=source, reason=reason, name=name, location=node.location)
        )

    def url(self, value: str) -> str:
        return urljoin(self.base_url, value) if self.base_url else value

    # JSON-LD

    def json_ld(self) -> None:
        nodes: list[tuple[_Node, dict[str, Any]]] = []
        for script in self.nodes:
            if script.tag != "script" or _script_type(script) != "application/ld+json":
                continue
            try:
                value = _parse_json_or_js(_unwrap(script.raw_text()))
            except _ParseError as e:
                self.skip("json_ld", script, f"invalid JSON-LD: {e}")
                continue
            nodes.extend((script, n) for n in _json_ld_nodes(value))
        by_id: dict[str, dict[str, Any]] = {}
        for _, node in nodes:
            ref = node.get("@id")
            if isinstance(ref, str) and len(node) > 1:
                by_id.setdefault(ref, node)
        for script, node in nodes:
            try:
                data = _inline_refs(node, by_id, node.get("@id"))
            except RecursionError:
                self.skip("json_ld", script, "invalid JSON-LD: nested too deeply")
                continue
            self.add("json_ld", script, data, types=_types(node.get("@type")))

    # Microdata

    def microdata(self) -> None:
        by_id = {n.attrs["id"]: n for n in self.nodes if "id" in n.attrs}
        for node in self.nodes:
            if "itemscope" in node.attrs and "itemprop" not in node.attrs:
                try:
                    item = self._item(node, by_id, frozenset())
                except RecursionError:
                    self.skip("microdata", node, "items nested too deeply")
                    continue
                self.add("microdata", node, item, types=_types(item.get("@type")))

    def _item(self, scope: _Node, by_id: dict[str, _Node], seen: frozenset[int]) -> dict[str, Any]:
        """The microdata item ``scope`` starts (HTML spec: *associating names with items*)."""
        seen = seen | {id(scope)}
        item: dict[str, Any] = {}
        types = [schema_type(t) for t in scope.attrs.get("itemtype", "").split()]
        if types:
            item["@type"] = types[0] if len(types) == 1 else types
        if scope.attrs.get("itemid"):
            item["@id"] = self.url(scope.attrs["itemid"])
        refs = [by_id[i] for i in scope.attrs.get("itemref", "").split() if i in by_id]
        # Children in document order, then the elements ``itemref`` names.
        stack: list[_Node | str] = [*reversed(refs), *reversed(scope.children)]
        properties: list[_Node] = []
        while stack:
            child = stack.pop()
            if not isinstance(child, _Node) or id(child) in seen:
                continue
            if "itemprop" in child.attrs:
                properties.append(child)
            if "itemscope" not in child.attrs:
                stack.extend(reversed(child.children))
        for prop in properties:
            if "itemscope" in prop.attrs:
                value: Any = self._item(prop, by_id, seen)
            else:
                value = self._value(prop, "itemprop")
            for name in prop.attrs["itemprop"].split():
                _add_property(item, schema_type(name), value)
        return item

    def _value(self, node: _Node, attr: Literal["itemprop", "property"]) -> str:
        """A property's value: ``content``, then the tag's URL or value attribute, then text.

        ``content`` counts on any element, not just ``meta``, because sites (and Google's
        examples) put machine values there: ``<span itemprop="price" content="9.99">``.
        """
        attrs = node.attrs
        if "content" in attrs:
            return attrs["content"]
        if attr == "property" and "resource" in attrs:
            return self.url(attrs["resource"])
        tag = node.tag
        if tag in _SRC_TAGS and "src" in attrs:
            return self.url(attrs["src"])
        if tag in _HREF_TAGS and "href" in attrs:
            return self.url(attrs["href"])
        if tag == "object" and "data" in attrs:
            return self.url(attrs["data"])
        if tag in ("data", "meter") and "value" in attrs:
            return attrs["value"]
        if tag == "time" and "datetime" in attrs:
            return attrs["datetime"]
        return node.text()

    # RDFa

    def rdfa(self) -> None:
        page: dict[str, Any] = {}
        items: list[tuple[_Node, dict[str, Any]]] = []

        stack: list[tuple[_Node, dict[str, Any] | None]] = [(self.root, None)]
        while stack:
            node, subject = stack.pop()
            attrs = node.attrs
            properties = [schema_type(p) for p in attrs.get("property", "").split()]
            if "typeof" in attrs:
                item: dict[str, Any] = {}
                types = [schema_type(t) for t in attrs["typeof"].split()]
                if types:
                    item["@type"] = types[0] if len(types) == 1 else types
                if "resource" in attrs:
                    item["@id"] = self.url(attrs["resource"])
                if properties:
                    for name in properties:
                        _add_property(subject if subject is not None else page, name, item)
                else:
                    items.append((node, item))
                subject = item
            elif properties:
                value = self._value(node, "property")
                for name in properties:
                    _add_property(subject if subject is not None else page, name, value)
            stack.extend((c, subject) for c in reversed(node.children) if isinstance(c, _Node))
        for node, item in items:
            self.add("rdfa", node, item, types=_types(item.get("@type")))
        if page:
            self.add("rdfa", self.root, page)

    # App state

    def app_state(self) -> None:
        for script in self.nodes:
            if script.tag != "script":
                continue
            kind = _script_type(script)
            text = script.raw_text()
            if kind != "application/ld+json" and _JSON_TYPE.fullmatch(kind):
                self._json_script(script, text)
            elif kind in _JS_TYPES and STATE_PATTERN.search(text):
                self._state_script(script, text)

    def _json_script(self, script: _Node, text: str) -> None:
        name = script.attrs.get("id") or None
        if not text.strip():
            return
        try:
            value = _parse_json_or_js(_unwrap(text))
            if name == "__NUXT_DATA__" or "data-nuxt-data" in script.attrs:
                value = unflatten_devalue(value)
        except _ParseError as e:
            self.skip("app_state", script, f"invalid JSON: {e}", name=name)
            return
        self.add("app_state", script, value, name=name)

    def _state_script(self, script: _Node, text: str) -> None:
        pos = 0
        while match := _STATE_ASSIGNMENT.search(text, pos):
            name = f"window.{match.group(1)}"
            pos = match.end()
            try:
                value, pos = _JsParser(text, match.end()).value_at_start()
            except _ParseError as e:
                self.skip("app_state", script, f"unreadable {name}: {e}", name=name)
                continue
            self.add("app_state", script, value, name=name)

    # data-* attributes

    def data_attributes(self, noise: re.Pattern[str] | None) -> None:
        for node in self.nodes:
            if node.tag in ("script", "style", "template"):
                continue
            data: dict[str, Any] = {}
            for attr, value in node.attrs.items():
                if not attr.startswith("data-") or len(attr) == 5 or not value.strip():
                    continue
                key = attr[5:]
                if noise is not None and noise.fullmatch(key):
                    continue
                data[key] = _attribute_value(value)
            if data:
                self.add("data_attributes", node, data)


def _fits(data: Any, limit: int) -> bool:
    """Whether ``data``, walked as a tree, holds at most ``limit`` values. Stops early."""
    count = 0
    stack: list[Any] = [data]
    while stack:
        value = stack.pop()
        count += 1
        if count > limit:
            return False
        if isinstance(value, dict):
            stack.extend(cast("dict[str, Any]", value).values())
        elif isinstance(value, list):
            stack.extend(cast("list[Any]", value))
    return True


def _script_type(script: _Node) -> str:
    return script.attrs.get("type", "").split(";", 1)[0].strip().lower()


def _unwrap(text: str) -> str:
    """Strip the HTML comment or CDATA wrapper old pages put around script data."""
    return _WRAPPER_END.sub("", _WRAPPER_START.sub("", text)).strip()


def _attribute_value(value: str) -> Any:
    """A ``data-*`` value; JSON objects and arrays (``data-product='{...}'``) are parsed."""
    stripped = value.strip()
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(stripped, parse_constant=_no_constant)
        except (ValueError, RecursionError):
            pass
    return value


def _types(value: Any) -> list[str]:
    raw = cast("list[Any]", value) if isinstance(value, list) else [value]
    return [schema_type(t) for t in raw if isinstance(t, str) and t.strip()]


def _add_property(item: dict[str, Any], name: str, value: Any) -> None:
    """Add a value, turning a repeated property into a list, as JSON-LD would write it."""
    if name not in item:
        item[name] = value
    elif isinstance(item[name], list):
        cast("list[Any]", item[name]).append(value)
    else:
        item[name] = [item[name], value]


def _json_ld_nodes(value: Any) -> list[dict[str, Any]]:
    """Top-level nodes: a list's items, a ``@graph``'s nodes, or the object itself."""
    nodes: list[dict[str, Any]] = []
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(cast("list[Any]", item)))
        elif isinstance(item, dict):
            node = cast("dict[str, Any]", item)
            if "@graph" in node:
                stack.append(node["@graph"])
            else:
                nodes.append({k: v for k, v in node.items() if k != "@context"})
    return nodes


def _inline_refs(value: Any, by_id: dict[str, dict[str, Any]], own_id: Any) -> Any:
    """Replace ``{"@id": ...}`` references with the node they name, one level deep.

    The inlined node's own references stay as they are: going further could loop, and a
    graph whose nodes share references would grow exponentially.
    """
    if isinstance(value, list):
        return [_inline_refs(v, by_id, own_id) for v in cast("list[Any]", value)]
    if not isinstance(value, dict):
        return value
    node = cast("dict[str, Any]", value)
    ref = node.get("@id")
    if len(node) == 1 and isinstance(ref, str) and ref in by_id and ref != own_id:
        return by_id[ref]
    return {k: _inline_refs(v, by_id, own_id) for k, v in node.items()}


# --- Parsing ------------------------------------------------------------------------------


class _ParseError(ValueError):
    pass


def _no_constant(_: str) -> None:
    """``NaN`` and ``Infinity`` in JSON read as ``None``, like ``null``: they aren't values
    a record can hold, and ``nan`` would never compare equal."""


def _parse_json_or_js(text: str) -> Any:
    """JSON, or failing that a JavaScript literal (trailing commas, single quotes...)."""
    try:
        return json.loads(text, strict=False, parse_constant=_no_constant)
    except ValueError:
        pass
    except RecursionError:
        raise _ParseError("nested too deeply") from None
    parser = _JsParser(text, 0)
    value, _ = parser.value_at_start()
    parser.accept(";")
    parser.skip_space()
    if parser.pos < len(text):
        raise parser.error("unexpected text after the value")
    return value


_UNDEFINED = -1
_HOLE = -2
_NAN = -3
_POSITIVE_INFINITY = -4
_NEGATIVE_INFINITY = -5
_NEGATIVE_ZERO = -6


def unflatten_devalue(flat: Any) -> Any:
    """Decode the ``devalue`` format Nuxt 3 uses for ``__NUXT_DATA__``.

    The payload is an array of values in which objects and arrays hold indexes into the
    array, and tagged arrays (``["Date", "2026-..."]``, ``["Reactive", 3]``) mark special
    values. Dates come back as ISO strings, sets as lists, maps as objects when their keys
    are strings, and ``undefined``, ``NaN`` and infinities as ``None``. A value that
    refers back to itself is cut to ``None`` where it loops. Values referred to more than
    once are decoded once and shared, so treat the result as read-only.
    """
    if isinstance(flat, int) and not isinstance(flat, bool):
        return _special(flat)
    if not isinstance(flat, list) or not flat:
        raise _ParseError("devalue payload must be a non-empty array")
    values = cast("list[Any]", flat)

    hydrated: dict[int, Any] = {}
    active: set[int] = set()

    def hydrate(index: Any) -> Any:
        if not isinstance(index, int) or isinstance(index, bool):
            raise _ParseError(f"devalue reference {index!r} is not an index")
        if index < 0:
            return _special(index)
        if index >= len(values):
            raise _ParseError(f"devalue reference {index} is out of range")
        # Shared references reuse one decoded value, so the work stays linear however
        # often a payload repeats them.
        if index in hydrated:
            return hydrated[index]
        if index in active:
            return None
        active.add(index)
        try:
            hydrated[index] = decode(values[index])
        finally:
            active.discard(index)
        return hydrated[index]

    def decode(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: hydrate(v) for k, v in cast("dict[str, Any]", value).items()}
        if not isinstance(value, list):
            return value
        items = cast("list[Any]", value)
        if not items or not isinstance(items[0], str):
            return [None if i == _HOLE else hydrate(i) for i in items]
        tag, args = items[0], items[1:]
        if tag == "BigInt":
            try:
                return int(args[0])
            except (IndexError, TypeError, ValueError):
                raise _ParseError(f"bad devalue BigInt {args!r}") from None
        if tag in ("Date", "RegExp", "URL"):
            return args[0] if args else None
        if tag == "Set":
            return [hydrate(i) for i in args]
        if tag in ("Map", "null"):
            pairs = [
                (hydrate(args[i]) if tag == "Map" else args[i], hydrate(args[i + 1]))
                for i in range(0, len(args) - 1, 2)
            ]
            if all(isinstance(k, str) for k, _ in pairs):
                return dict(pairs)
            return [list(p) for p in pairs]
        # Reactive, ShallowReactive, Ref, Object and other wrappers hold one value.
        return hydrate(args[0]) if args else None

    try:
        return hydrate(0)
    except RecursionError:
        raise _ParseError("devalue payload is nested too deeply") from None


def _special(index: int) -> Any:
    if index == _NEGATIVE_ZERO:
        return -0.0
    if index in (_UNDEFINED, _HOLE, _NAN, _POSITIVE_INFINITY, _NEGATIVE_INFINITY):
        return None
    raise _ParseError(f"devalue reference {index} is not a special value")


_JS_NUMBER = re.compile(
    r"[-+]?(?:0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)
_JS_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_JS_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
_JS_CONSTANTS: dict[str, Any] = {
    "true": True,
    "false": False,
    "null": None,
    "undefined": None,
    "NaN": None,
    "Infinity": None,
}


class _JsParser:
    """Reads one JavaScript literal expression, without running anything.

    Handles what pages put in state assignments: object and array literals (unquoted or
    quoted keys, trailing commas, holes), strings in any quotes, numbers, ``!0``/``!1``,
    ``void 0``, ``JSON.parse("...")``, ``new Date(...)`` and an immediately invoked
    ``function(a, b){ ...; return {...} }(x, y)`` whose parameters the literal refers to
    (Nuxt 2). Other identifiers read as ``None``.
    """

    def __init__(self, text: str, pos: int, env: dict[str, Any] | None = None) -> None:
        self.text = text
        self.pos = pos
        self.env = env or {}

    def value_at_start(self) -> tuple[Any, int]:
        try:
            value = self.value()
        except RecursionError:
            raise _ParseError("nested too deeply") from None
        return value, self.pos

    def error(self, message: str) -> _ParseError:
        return _ParseError(f"{message} at offset {self.pos}")

    def skip_space(self) -> None:
        text, n = self.text, len(self.text)
        while self.pos < n:
            if text[self.pos].isspace():
                self.pos += 1
            elif text.startswith("//", self.pos):
                end = text.find("\n", self.pos)
                self.pos = n if end < 0 else end
            elif text.startswith("/*", self.pos):
                end = text.find("*/", self.pos + 2)
                if end < 0:
                    raise self.error("unterminated comment")
                self.pos = end + 2
            else:
                return

    def peek(self) -> str:
        self.skip_space()
        return self.text[self.pos : self.pos + 1]

    def expect(self, token: str) -> None:
        self.skip_space()
        if not self.text.startswith(token, self.pos):
            raise self.error(f"expected {token!r}")
        self.pos += len(token)

    def accept(self, token: str) -> bool:
        self.skip_space()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def ident(self) -> str | None:
        self.skip_space()
        match = _JS_IDENT.match(self.text, self.pos)
        if not match:
            return None
        self.pos = match.end()
        return match.group()

    def value(self) -> Any:
        char = self.peek()
        if not char:
            raise self.error("expected a value")
        if char == "{":
            return self.object()
        if char == "[":
            return self.array()
        if char in "\"'`":
            return self.string()
        if char == "(":
            return self.parenthesised()
        if char == "!":
            self.pos += 1
            return not self.value()
        match = _JS_NUMBER.match(self.text, self.pos)
        if match:
            self.pos = match.end()
            return _js_number(match.group())
        if char in "-+":
            self.pos += 1
            sign = -1 if char == "-" else 1
            operand = self.value()
            return sign * operand if isinstance(operand, int | float) else None
        return self.word()

    def word(self) -> Any:
        start = self.pos
        name = self.ident()
        if name is None:
            raise self.error("unexpected character")
        if name == "void":
            self.value()
            return None
        if name == "function":
            self.pos = start
            params, returns = self.function()
            if self.peek() != "(":
                raise self.error("expected a function call")
            return self.call(params, returns)
        if name == "new":
            constructor = self.ident()
            args = self.arguments() if self.peek() == "(" else []
            if constructor == "Date" and args:
                return args[0]
            return args[0] if len(args) == 1 else None
        if name == "JSON" and self.accept(".") and self.ident() == "parse":
            args = self.arguments()
            if not args or not isinstance(args[0], str):
                raise self.error("JSON.parse needs a string")
            try:
                return json.loads(args[0], strict=False, parse_constant=_no_constant)
            except ValueError as e:
                raise self.error(f"JSON.parse argument is not JSON ({e})") from None
        while self.accept("."):
            if self.ident() is None:
                raise self.error("expected a property name")
        if self.peek() == "(":
            # Some other call: its result can't be known without running it.
            self.arguments()
            return None
        if name in _JS_CONSTANTS:
            return _JS_CONSTANTS[name]
        return self.env.get(name)

    def parenthesised(self) -> Any:
        self.expect("(")
        match = _JS_IDENT.match(self.text, self.pos)
        if not (match and match.group() == "function"):
            value = self.value()
            self.expect(")")
            return value
        params, returns = self.function()
        if self.peek() == "(":
            # (function(a){...}(1))
            value = self.call(params, returns)
            self.expect(")")
            return value
        # (function(a){...})(1)
        self.expect(")")
        if self.peek() != "(":
            raise self.error("expected a function call")
        return self.call(params, returns)

    def function(self) -> tuple[list[str], int | None]:
        """A function expression's parameters, and where its body's ``return`` value starts.

        The body's statements before its top-level ``return`` are skipped; they only
        patch up shared references, which a data reader can do without.
        """
        self.expect("function")
        self.ident()
        self.expect("(")
        params: list[str] = []
        while not self.accept(")"):
            param = self.ident()
            if param is None:
                raise self.error("expected a parameter name")
            params.append(param)
            self.accept(",")
        return params, self.skip_block()

    def call(self, params: list[str], returns: int | None) -> Any:
        """Call a function read by :meth:`function`: its return value, given the arguments."""
        args = self.arguments()
        if returns is None:
            return None
        env = {**self.env, **dict(zip(params, args, strict=False))}
        value, _ = _JsParser(self.text, returns, env).value_at_start()
        return value

    def skip_block(self) -> int | None:
        """Skip a ``{...}`` body; return where its top-level ``return`` value starts."""
        self.expect("{")
        depth = 1
        returns: int | None = None
        text, n = self.text, len(self.text)
        while depth:
            self.skip_space()
            if self.pos >= n:
                raise self.error("unterminated function body")
            char = text[self.pos]
            if char in "\"'`":
                self.string()
                continue
            match = _JS_IDENT.match(text, self.pos)
            if match:
                self.pos = match.end()
                if depth == 1 and returns is None and match.group() == "return":
                    self.skip_space()
                    returns = self.pos
                continue
            if char in "{[(":
                depth += 1
            elif char in "}])":
                depth -= 1
            self.pos += 1
        return returns

    def arguments(self) -> list[Any]:
        self.expect("(")
        args: list[Any] = []
        while not self.accept(")"):
            args.append(self.value())
            if not self.accept(","):
                self.expect(")")
                break
        return args

    def object(self) -> dict[str, Any]:
        self.expect("{")
        out: dict[str, Any] = {}
        while not self.accept("}"):
            char = self.peek()
            if not char:
                raise self.error("expected a property name")
            if char in "\"'`":
                key = self.string()
            elif char == "[":
                raise self.error("computed keys are not supported")
            else:
                match = _JS_NUMBER.match(self.text, self.pos)
                if match and not _JS_IDENT.match(self.text, self.pos):
                    self.pos = match.end()
                    number = _js_number(match.group())
                    key = match.group() if number is None else str(number)
                else:
                    name = self.ident()
                    if name is None:
                        raise self.error("expected a property name")
                    key = name
            if self.accept(":"):
                out[key] = self.value()
            else:
                out[key] = self.env.get(key)
            if not self.accept(","):
                self.expect("}")
                break
        return out

    def array(self) -> list[Any]:
        self.expect("[")
        out: list[Any] = []
        while not self.accept("]"):
            if self.peek() == ",":
                self.pos += 1
                out.append(None)
                continue
            out.append(self.value())
            if not self.accept(","):
                self.expect("]")
                break
        return out

    def string(self) -> str:
        self.skip_space()
        text, n = self.text, len(self.text)
        quote = text[self.pos]
        self.pos += 1
        parts: list[str] = []
        while True:
            if self.pos >= n:
                raise self.error("unterminated string")
            char = text[self.pos]
            if char == quote:
                self.pos += 1
                return "".join(parts)
            if quote == "`" and text.startswith("${", self.pos):
                raise self.error("template substitutions are not supported")
            if char != "\\":
                parts.append(char)
                self.pos += 1
                continue
            escape = text[self.pos + 1 : self.pos + 2]
            self.pos += 2
            if escape in _JS_ESCAPES:
                parts.append(_JS_ESCAPES[escape])
            elif escape == "x":
                parts.append(self.hex_char(2))
            elif escape == "u":
                if text.startswith("{", self.pos):
                    end = text.find("}", self.pos)
                    if end < 0:
                        raise self.error("bad \\u{...} escape")
                    parts.append(self.code_point(text[self.pos + 1 : end]))
                    self.pos = end + 1
                else:
                    parts.append(self.hex_char(4))
            elif escape in ("\n", "\u2028", "\u2029"):
                pass
            elif escape == "\r":
                if text.startswith("\n", self.pos):
                    self.pos += 1
            else:
                parts.append(escape)

    def hex_char(self, length: int) -> str:
        digits = self.text[self.pos : self.pos + length]
        self.pos += length
        return self.code_point(digits if len(digits) == length else "")

    def code_point(self, digits: str) -> str:
        try:
            return chr(int(digits, 16))
        except ValueError:
            raise self.error(f"bad escape {digits!r}") from None


def _js_number(literal: str) -> int | float | None:
    sign = -1 if literal.startswith("-") else 1
    body = literal.lstrip("+-")
    prefix = body[:2].lower()
    if prefix in ("0x", "0o", "0b"):
        return sign * int(body[2:], {"0x": 16, "0o": 8, "0b": 2}[prefix])
    if any(c in body for c in ".eE"):
        number = sign * float(body)
        return number if math.isfinite(number) else None
    return sign * int(body)

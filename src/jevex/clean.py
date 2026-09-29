"""Stage 2, clean: strip boilerplate from HTML before layout.

The default :class:`BoilerplateCleaner` removes whole subtrees that never hold a record's
details: scripts and styles, navigation, page-level headers and footers, and cookie or
consent banners. Taking them out early keeps them out of the component tree, so later
stages ask Jev fewer questions.

It is a single streaming pass over the standard library's :class:`html.parser.HTMLParser`,
so the core install needs no HTML dependency. Markup it keeps is copied through verbatim
(entities stay escaped) and re-encoded in the document's own charset, so the layout parser
sees the page as it was, minus the removed elements.

Scripts that carry data rather than code (JSON-LD, ``__NEXT_DATA__``, Nuxt payloads,
``window.__INITIAL_STATE__``) are kept wherever they appear, even inside removed
boilerplate, because the structured-data stage runs after this one and reads them.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jevex.document import Document
    from jevex.interfaces import Cleaner
    from jevex.pipeline import Context

DROP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "iframe", "object", "embed", "nav"}
)
"""Elements removed wherever they appear. A ``template`` holding declarative shadow DOM
(``shadowrootmode``) is rendered by browsers, so it stays."""

PAGE_TAGS = frozenset({"header", "footer"})
"""Elements removed only at page level. Inside an article or section they often carry the
title, byline or date, so they stay (the HTML ``banner``/``contentinfo`` rule)."""

SECTIONING_TAGS = frozenset({"article", "aside", "main", "nav", "section"})

DROP_ROLES = frozenset({"navigation", "banner", "contentinfo"})
"""ARIA landmark roles removed like their equivalent tags."""

CONSENT_PATTERN = re.compile(
    r"cookie[-_ ]?(?:banner|bar|notice|notification|popup|modal|law|message|warning"
    r"|disclaimer|dialog|overlay|wall|settings|preferences)"
    r"|(?:cookie|gdpr)[-_ ]?consent(?![-_]?(?:active|given|set|accepted|granted|ed\b))"
    r"|consent[-_](?:banner|bar|modal|dialog|popup|overlay|manager|notice|box)"
    r"|gdpr[-_](?:banner|bar|popup|notice|modal|dialog|overlay)"
    r"|onetrust|cookiebot|didomi|usercentrics|trustarc|truste[-_]|qc-cmp|cmpbox|osano-cm"
    r"|iubenda|sp_message|cc-(?:window|banner)",
    re.IGNORECASE,
)
"""Matched against ``id``, ``class`` and ``aria-label`` to find cookie and consent banners.
Only banner-shaped names and consent-platform names count: a bare "cookie", "consent" or
"gdpr" doesn't, so recipe pages, consent forms and GDPR guides survive, and neither do
wrapper flags such as ``cookie-consent-active``."""

STATE_PATTERN = re.compile(
    r"__(?:NEXT_DATA|NUXT|NUXT_DATA|INITIAL_STATE|PRELOADED_STATE|APOLLO_STATE)__"
)
"""Marks a plain ``<script>`` that embeds app state, which the structured-data stage reads."""

PROTECTED_TAGS = frozenset({"html", "head", "body", "main", "article"})
"""Never removed by role or pattern: sites put consent flags on ``<body class=...>``."""

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

_JSON_TYPE = re.compile(r"application/(?:[-\w.]+\+)?json", re.IGNORECASE)
_JS_TYPES = frozenset({"", "module", "text/javascript", "application/javascript"})

_META_CHARSET = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([-\w.:]+)", re.IGNORECASE)
# The UTF-16 BOM is decoded as U+FEFF and written back, so the byte order is kept.
_BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
_ASCII = bytes(range(0x20, 0x7F))


class BoilerplateCleaner:
    """The default :class:`~jevex.interfaces.Cleaner`: drops boilerplate subtrees from HTML.

    Non-HTML documents, and HTML with nothing to remove, are returned unchanged. Pass
    other tag sets or another pattern to widen or narrow what counts as boilerplate.
    ``keep_data_scripts=False`` drops JSON and app-state scripts too, for pipelines
    without a structured-data stage.
    """

    def __init__(
        self,
        *,
        drop_tags: frozenset[str] = DROP_TAGS,
        page_tags: frozenset[str] = PAGE_TAGS,
        drop_roles: frozenset[str] = DROP_ROLES,
        pattern: re.Pattern[str] | None = CONSENT_PATTERN,
        keep_data_scripts: bool = True,
    ) -> None:
        self.drop_tags = drop_tags
        self.page_tags = page_tags
        self.drop_roles = drop_roles
        self.pattern = pattern
        self.keep_data_scripts = keep_data_scripts

    def clean(self, document: Document) -> Document:
        if not document.is_html:
            return document
        text, encoding = _decode(document.content)
        stripper = _Stripper(self)
        stripper.feed(text)
        stripper.close()
        if not stripper.removed:
            return document
        content = "".join(stripper.out).encode(encoding, "surrogateescape")
        return document.model_copy(update={"content": content})

    def drops(self, tag: str, attrs: list[tuple[str, str | None]], open_tags: list[str]) -> bool:
        """Whether an element starting here is boilerplate, given its open ancestors."""
        if tag == "template" and any(n in ("shadowrootmode", "shadowroot") for n, _ in attrs):
            return False
        if tag in self.drop_tags:
            return True
        if tag in self.page_tags and SECTIONING_TAGS.isdisjoint(open_tags):
            return True
        if tag in PROTECTED_TAGS:
            return False
        values = {name: value or "" for name, value in attrs}
        if values.get("role", "").strip().lower() in self.drop_roles:
            return True
        if self.pattern is None:
            return False
        return any(
            self.pattern.search(values.get(name, "")) for name in ("id", "class", "aria-label")
        )

    def keeps_script(self, attrs: list[tuple[str, str | None]], text: str) -> bool:
        """Whether a script that would be removed carries data the structured stage reads."""
        if not self.keep_data_scripts:
            return False
        kind = (dict(attrs).get("type") or "").split(";", 1)[0].strip().lower()
        if _JSON_TYPE.fullmatch(kind):
            return True
        return kind in _JS_TYPES and STATE_PATTERN.search(text) is not None


class _Stripper(HTMLParser):
    """Copies markup through, skipping boilerplate subtrees and comments.

    Unclosed elements are handled the way browsers recover: an end tag closes the
    nearest open element with that name and everything opened inside it. A skipped
    subtree therefore ends when its own end tag arrives, or when an ancestor closes.

    A script that would be removed is buffered instead, because whether it holds data is
    only known once its text has been read. The parser reads script text raw, so nothing
    else can start before its end tag.
    """

    def __init__(self, cleaner: BoilerplateCleaner) -> None:
        super().__init__(convert_charrefs=False)
        self.cleaner = cleaner
        self.out: list[str] = []
        self.open: list[str] = []
        self.skip_from: int | None = None
        """Stack depth of the element being skipped, or None when copying."""
        self.removed = 0
        self.script: tuple[list[tuple[str, str | None]], list[str]] | None = None
        """Attributes and text so far of a script held back for :meth:`keeps_script`."""

    def _emit(self, text: str) -> None:
        if self.skip_from is None:
            self.out.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs, self.get_starttag_text() or "", closed=tag in _VOID_TAGS)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs, self.get_starttag_text() or "", closed=True)

    def _start(
        self, tag: str, attrs: list[tuple[str, str | None]], text: str, *, closed: bool
    ) -> None:
        if (
            tag == "script"
            and not closed
            and (self.skip_from is not None or self.cleaner.drops(tag, attrs, self.open))
        ):
            self.script = (attrs, [text])
            return
        if self.skip_from is None and self.cleaner.drops(tag, attrs, self.open):
            self.removed += 1
            if not closed:
                self.skip_from = len(self.open)
                self.open.append(tag)
            return
        self._emit(text)
        if not closed:
            self.open.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.script is not None and tag == "script":
            attrs, parts = self.script
            self.script = None
            if self.cleaner.keeps_script(attrs, "".join(parts[1:])):
                self.out.append("".join(parts) + "</script>")
            else:
                self.removed += 1
            return
        if tag not in self.open:
            self._emit(f"</{tag}>")
            return
        depth = len(self.open) - 1 - self.open[::-1].index(tag)
        del self.open[depth:]
        if self.skip_from is not None and depth <= self.skip_from:
            ends_skipped = depth == self.skip_from
            self.skip_from = None
            if ends_skipped:
                return
        self._emit(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self.script is not None:
            self.script[1].append(data)
        else:
            self._emit(data)

    def handle_entityref(self, name: str) -> None:
        self._emit(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._emit(f"&#{name};")

    def handle_decl(self, decl: str) -> None:
        self._emit(f"<!{decl}>")

    def unknown_decl(self, data: str) -> None:
        # CDATA sections end with "]]>", conditional sections with "]>".
        self._emit(f"<![{data}]]>" if data.startswith("CDATA[") else f"<![{data}]>")

    def handle_pi(self, data: str) -> None:
        self._emit(f"<?{data}>")

    def handle_comment(self, data: str) -> None:
        self.removed += 1


@dataclass
class CleanStage:
    """Runs a :class:`~jevex.interfaces.Cleaner` over the document before anything reads it."""

    cleaner: Cleaner = field(default_factory=BoilerplateCleaner)
    name: str = "clean"

    async def run(self, ctx: Context) -> None:
        ctx.document = self.cleaner.clean(ctx.document)


def _decode(content: bytes) -> tuple[str, str]:
    """Decode with the page's charset if that round-trips exactly, else as UTF-8.

    ``surrogateescape`` keeps undecodable bytes, so UTF-8 always round-trips. Falling back
    means a page with a broken or exotic charset is still cleaned (or passed through)
    rather than failing the extraction.
    """
    encoding = _encoding(content)
    try:
        text = content.decode(encoding, "surrogateescape")
        if text.encode(encoding, "surrogateescape") == content:
            return text, encoding
    except UnicodeError:
        pass
    return content.decode("utf-8", "surrogateescape"), "utf-8"


def _encoding(content: bytes) -> str:
    """The charset to use: BOM, then an ASCII-compatible ``<meta charset>``, then UTF-8."""
    for bom, encoding in _BOMS:
        if content.startswith(bom):
            return encoding
    match = _META_CHARSET.search(content[:4096])
    if not match:
        return "utf-8"
    try:
        name = codecs.lookup(match.group(1).decode("ascii")).name
        # A meta tag readable as ASCII means the bytes are in an ASCII-compatible charset
        # (the WHATWG rule), which rules out UTF-16/32, UTF-7 and EBCDIC.
        compatible = _ASCII.decode("ascii").encode(name) == _ASCII
    except (LookupError, UnicodeError):
        return "utf-8"
    return name if compatible else "utf-8"

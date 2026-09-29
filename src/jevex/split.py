"""Statement splitting: components → atomic statements (spec: *Statement splitting by
component type*).

=====================  ===========================================  ===============
Component              Statements                                    Kind
=====================  ===========================================  ===============
paragraph              one per sentence (pysbd); a ``Label: value``  ``sentence`` /
                       line (after a ``<br>``) is one pair           ``key_value``
list item              one each; a ``Label: value`` item, a ``dd``   ``list_item`` /
                       with its term, or each pair line is a pair    ``key_value``
heading                one (a product page's title is its ``h1``)    ``sentence``
caption                one                                           ``caption``
image                  its alt text                                  ``alt_text``
table                  none yet: cells rendered with headers are     (#25)
                       #25
containers             none; their children are split instead
=====================  ===========================================  ===============

Each statement carries its component's ``heading_trail`` and ``location``. Ids are
``<component id>.<n>``, so they're unique and stable for a given tree.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Protocol, cast

from jevex.layout import DomLocation
from jevex.statements import Statement

if TYPE_CHECKING:
    from collections.abc import Callable

    from jevex.interfaces import StatementSplitter
    from jevex.layout import Component
    from jevex.pipeline import Context
    from jevex.statements import StatementKind

# ``Label: value``: "Engine: 1.5 TSI", "Price : £24,995", "ISBN-13: 978-0-14-032872-1".
_KEY_VALUE = re.compile(
    r"^(?P<label>[^:\uff1a\n]{1,60}?)\s*(?P<sep>[:\uff1a])\s*(?P<value>\S.*)$", re.S
)
_MAX_LABEL_WORDS = 8
_WHITESPACE = re.compile(r"\s+")


def is_key_value(text: str) -> bool:
    """Whether ``text`` reads as one ``Label: value`` pair."""
    text = text.strip()
    m = _KEY_VALUE.match(text)
    if m is None:
        return False
    label, value = m["label"].strip(), m["value"]
    if not any(ch.isalpha() for ch in label) or len(label.split()) > _MAX_LABEL_WORDS:
        return False
    if value.startswith("//"):  # a URL: "https://..."
        return False
    # A clock time or ratio has digits right against the colon: "at 10:30", "ratio 16:9".
    # "Series 5: 2019" and "ISBN-13: 978-..." have a space, so they're pairs.
    sep = m.start("sep")
    return not (text[sep - 1 : sep].isdigit() and text[sep + 1 : sep + 2].isdigit())


def _dl_part(component: Component) -> str | None:
    """``"dt"`` or ``"dd"`` when the layout parser built this item from a ``dl``."""
    location = component.location
    if not isinstance(location, DomLocation):
        return None
    last = location.dom_path.rsplit("/", 1)[-1].split("[", 1)[0]
    return last if last in ("dt", "dd") else None


class _Segmenter(Protocol):
    def segment(self, text: str) -> list[_TextSpan]: ...


class _TextSpan(Protocol):
    @property
    def sent(self) -> str: ...


@cache
def _pysbd_language(language: str) -> tuple[Callable[..., _Segmenter], str]:
    """The pysbd ``Segmenter`` class and the language code to use (``en`` if unknown)."""
    with warnings.catch_warnings():
        # pysbd 0.3.4 has invalid escape sequences, which 3.12+ warns about on first compile.
        warnings.simplefilter("ignore", SyntaxWarning)
        import pysbd  # pyright: ignore[reportMissingTypeStubs]
        from pysbd.languages import LANGUAGE_CODES  # pyright: ignore[reportMissingTypeStubs]

    codes = cast("dict[str, object]", LANGUAGE_CODES)
    segmenter = cast("Callable[..., _Segmenter]", pysbd.Segmenter)
    return segmenter, language if language in codes else "en"


# Abbreviations pysbd's English rules end a sentence on, but which pages use mid-sentence:
# "approx. 300 miles", "£24,995 excl. VAT", "Max. speed", "4 cyl. and 6 gears", "Vol. 2".
ABBREVIATIONS = frozenset(
    {
        "approx", "ca", "circa", "cyl", "est", "excl", "incl", "max", "min", "no", "nos",
        "nr", "vol", "vols", "pp", "p", "ed", "eds", "ref", "tel", "dept", "avg", "std",
        "opt", "rrp", "orig", "wt", "ht", "dia", "qty", "misc",
    }
)  # fmt: skip
_LAST_WORD = re.compile(r"(\w+)\.$")
# What may follow a mid-sentence abbreviation: a lowercase word, a number, a symbol or an
# all-caps acronym ("excl. VAT"). A capitalised word starts a real sentence ("5 min. Then").
_CONTINUES = re.compile(r"^(?:[a-z0-9£$€(\[%&+\-–]|[A-Z]{2,5}\b)")
# A fragment ending in a dotted model name ("The VW ID.3") has no sentence-ending mark.
_DOTTED_NAME = re.compile(r"\b\w+\.\d\w*$")
MAX_SEGMENT_CHARS = 3000
"""pysbd slows quadratically on long lines, so longer lines are cut at clear sentence
boundaries into pieces of about this size first."""
_CLEAR_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


def _chunks(text: str) -> list[str]:
    if len(text) <= MAX_SEGMENT_CHARS:
        return [text]
    out: list[str] = []
    start = 0
    for m in _CLEAR_BOUNDARY.finditer(text):
        if m.start() - start >= MAX_SEGMENT_CHARS:
            out.append(text[start : m.start()])
            start = m.end()
    out.append(text[start:])
    return out


def _mis_split(fragment: str, following: str) -> bool:
    if _DOTTED_NAME.search(fragment):
        return True
    word = _LAST_WORD.search(fragment)
    return (
        word is not None
        and word.group(1).lower() in ABBREVIATIONS
        and _CONTINUES.match(following) is not None
    )


def sentences(text: str, *, language: str = "en") -> list[str]:
    """``text`` split into sentences with pysbd, stripped and without empties.

    Fragments pysbd splits after a mid-sentence abbreviation (:data:`ABBREVIATIONS`) or a
    dotted model name ("ID.3") are joined back up. A new pysbd segmenter is built per call:
    they keep per-call state, so sharing one across threads loses text.
    """
    segmenter_class, lang = _pysbd_language(language)
    pieces: list[str] = []
    for chunk in _chunks(text):
        segmenter = segmenter_class(language=lang, clean=False, char_span=True)
        pieces.extend(s for span in segmenter.segment(chunk) if (s := span.sent.strip()))
    out: list[str] = []
    for piece in pieces:
        if out and _mis_split(out[-1], piece):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


@dataclass(frozen=True)
class DefaultSplitter:
    """The default :class:`~jevex.interfaces.StatementSplitter`.

    ``language`` is the pysbd language code (``en``, ``de``, ``fr``...); unsupported codes
    fall back to English. Only the component itself is split, never its children: the
    stage calls the splitter on every component in the tree.
    """

    language: str = "en"

    def split(self, component: Component) -> list[Statement]:
        kind = component.type
        text = component.text.strip()
        if not text:
            return []
        pieces: list[tuple[str, StatementKind]]
        if kind == "paragraph":
            pieces = self._paragraph(text)
        elif kind == "list_item":
            pieces = self._list_item(text, _dl_part(component))
        elif kind == "heading":
            pieces = [(_WHITESPACE.sub(" ", text), "sentence")]
        elif kind == "caption":
            pieces = [(_WHITESPACE.sub(" ", text), "caption")]
        elif kind == "image":
            pieces = [(_WHITESPACE.sub(" ", text), "alt_text")]
        else:  # containers; tables are #25
            return []
        return [
            Statement(
                id=f"{component.id}.{i}",
                text=piece,
                kind=piece_kind,
                component_id=component.id,
                heading_trail=list(component.heading_trail),
                location=component.location,
            )
            for i, (piece, piece_kind) in enumerate(pieces)
        ]

    def _paragraph(self, text: str) -> list[tuple[str, StatementKind]]:
        out: list[tuple[str, StatementKind]] = []
        for line in _lines(text):
            said = sentences(line, language=self.language)
            if len(said) == 1 and is_key_value(line):
                out.append((line, "key_value"))
            else:
                out.extend((s, "sentence") for s in said)
        return out

    def _list_item(self, text: str, dl_part: str | None) -> list[tuple[str, StatementKind]]:
        """One statement per item, unless its lines are ``Label: value`` pairs.

        ``<li>Engine: 1.5 TSI<br>Power: 150 PS</li>`` gives two pairs. A ``dl`` item is a
        pair only when it is a ``dd`` rendered with its term; a lone ``dt`` is an item.
        """
        lines = _lines(text)
        if dl_part is None and len(lines) > 1 and any(is_key_value(line) for line in lines):
            return [(line, "key_value" if is_key_value(line) else "list_item") for line in lines]
        one = " ".join(lines)
        pair = is_key_value(one) and dl_part != "dt"
        return [(one, "key_value" if pair else "list_item")]


def _lines(text: str) -> list[str]:
    return [line for raw in text.split("\n") if (line := _WHITESPACE.sub(" ", raw).strip())]


class DuplicateStatementError(ValueError):
    """Two statements on one document share an id (a splitter or earlier stage bug)."""


@dataclass
class StatementStage:
    """Splits every component of ``ctx.parsed`` into statements (stage 9).

    Statements already on the document (from the structured-data stage) are kept; new
    ones are added in reading order, and an id clash raises
    :class:`DuplicateStatementError` rather than replacing one. Without a parsed document
    the stage does nothing.
    """

    splitter: StatementSplitter = field(default_factory=DefaultSplitter)
    name: str = "statements"

    async def run(self, ctx: Context) -> None:
        parsed = ctx.parsed
        if parsed is None:
            return
        for component in parsed.root.walk():
            for statement in self.splitter.split(component):
                if statement.id in parsed.statements:
                    raise DuplicateStatementError(
                        f"statement id {statement.id!r} (component {component.id!r}) is "
                        "already on the document"
                    )
                parsed.statements[statement.id] = statement

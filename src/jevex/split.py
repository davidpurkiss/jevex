"""Statement splitting: components → atomic statements (spec: *Statement splitting by
component type*).

=====================  ===========================================  ===============
Component              Statements                                    Kind
=====================  ===========================================  ===============
paragraph              one per sentence (pysbd); a ``Label: value``  ``sentence`` /
                       line (after a ``<br>``) is one pair           ``key_value``
list item              one each; a ``dl`` item or ``Label: value``   ``list_item`` /
                       item is a pair                                ``key_value``
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
    from jevex.interfaces import StatementSplitter
    from jevex.layout import Component
    from jevex.pipeline import Context
    from jevex.statements import StatementKind

# ``Label: value``: "Engine: 1.5 TSI", "Price : £24,995", "0-62 mph: 9.1 s".
_KEY_VALUE = re.compile(r"^(?P<label>[^:\uff1a\n]{1,60}?)\s*[:\uff1a]\s*(?P<value>\S.*)$", re.S)
_MAX_LABEL_WORDS = 8
_WHITESPACE = re.compile(r"\s+")


def is_key_value(text: str) -> bool:
    """Whether ``text`` reads as one ``Label: value`` pair."""
    m = _KEY_VALUE.match(text.strip())
    if m is None:
        return False
    label, value = m["label"].strip(), m["value"]
    if not any(ch.isalpha() for ch in label) or len(label.split()) > _MAX_LABEL_WORDS:
        return False
    if value.startswith("//"):  # a URL: "https://..."
        return False
    # A clock time or ratio: "starts at 10:30", "ratio 16:9".
    return not (label[-1].isdigit() and value[0].isdigit())


def _in_definition_list(component: Component) -> bool:
    location = component.location
    if not isinstance(location, DomLocation):
        return False
    last = location.dom_path.rsplit("/", 1)[-1]
    return last.split("[", 1)[0] in ("dt", "dd")


class _Segmenter(Protocol):
    def segment(self, text: str) -> list[_TextSpan]: ...


class _TextSpan(Protocol):
    @property
    def sent(self) -> str: ...


@cache
def _segmenter(language: str) -> _Segmenter:
    with warnings.catch_warnings():
        # pysbd 0.3.4 has invalid escape sequences, which 3.12+ warns about on first compile.
        warnings.simplefilter("ignore", SyntaxWarning)
        import pysbd  # pyright: ignore[reportMissingTypeStubs]
        from pysbd.languages import LANGUAGE_CODES  # pyright: ignore[reportMissingTypeStubs]

    codes = cast("dict[str, object]", LANGUAGE_CODES)
    lang = language if language in codes else "en"
    return cast("_Segmenter", pysbd.Segmenter(language=lang, clean=False, char_span=True))


def sentences(text: str, *, language: str = "en") -> list[str]:
    """``text`` split into sentences with pysbd, stripped and without empties."""
    spans = _segmenter(language).segment(text)
    return [s for span in spans if (s := span.sent.strip())]


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
            one = _WHITESPACE.sub(" ", text)
            pair = _in_definition_list(component) or is_key_value(one)
            pieces = [(one, "key_value" if pair else "list_item")]
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
        for line in text.split("\n"):
            line = _WHITESPACE.sub(" ", line).strip()
            if not line:
                continue
            if is_key_value(line) and len(sentences(line, language=self.language)) == 1:
                out.append((line, "key_value"))
            else:
                out.extend((s, "sentence") for s in sentences(line, language=self.language))
        return out


@dataclass
class StatementStage:
    """Splits every component of ``ctx.parsed`` into statements (stage 9).

    Statements already on the document (from the structured-data stage) are kept; new
    ones are added in reading order. Without a parsed document the stage does nothing.
    """

    splitter: StatementSplitter = field(default_factory=DefaultSplitter)
    name: str = "statements"

    async def run(self, ctx: Context) -> None:
        parsed = ctx.parsed
        if parsed is None:
            return
        for component in parsed.root.walk():
            for statement in self.splitter.split(component):
                parsed.statements[statement.id] = statement

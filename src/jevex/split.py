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
table                  one per cell, with its row and column         ``table_cell`` /
                       headers, and one per header label of a        ``table_header``
                       table with headers on both axes
                       (:mod:`jevex.tables`)
containers             none; their children are split instead
=====================  ===========================================  ===============

Text the image stage read from an image (a paragraph or heading with an
:class:`~jevex.layout.ImageLocation`) is split the same way, but its sentences are ``ocr``
statements. Components that already have statements (the image stage's ``vision``
statements) aren't split again.

Each statement carries its component's ``heading_trail`` and ``location``. Ids are
``<component id>.<n>`` (table cells: ``<table id>.r<row>c<col>``, headers
``<table id>.h<row>c<col>``), so they're unique
and stable for a given tree.

A statement longer than :data:`MAX_STATEMENT_CHARS` (a whole product description in one
``<li>`` or table cell) is too big for one Jev state, so :class:`StatementStage` cuts it
into pieces (:func:`cut_statement`), whichever splitter made it. Pieces get ids
``<statement id>:<n>``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast

from jevex.interfaces import LocaleAwareSplitter
from jevex.layout import DomLocation, ImageLocation
from jevex.locales import locale_conventions
from jevex.resolve import sibling_labels
from jevex.select import candidate_locale
from jevex.statements import Statement
from jevex.tables import header_prefix, infer_headers, table_statements

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jevex.interfaces import StatementSplitter
    from jevex.layout import Component
    from jevex.pipeline import Context
    from jevex.statements import StatementKind

# ``Label: value``: "Engine: 1.5 TSI", "Price : £24,995", "ISBN-13: 978-0-14-032872-1".
_KEY_VALUE = re.compile(
    r"^(?P<label>[^:\uff1a\n]{1,60}?)\s*(?P<sep>[:\uff1a])\s*(?P<value>\S.*)$", re.S
)
_MAX_LABEL_WORDS = 8
_WHITESPACE = re.compile(r"[ \t\n\r\f\v]+")  # a no-break space stays: it can group thousands


def is_key_value(text: str) -> bool:
    """Whether ``text`` reads as one ``Label: value`` pair.

    That the text before the colon labels the value is a guess, made in code because it
    shapes the statements (and OCR lines) Jev is later asked about; Jev still judges
    which field, if any, the pair states."""
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
LANGUAGE_ABBREVIATIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        # "ca. 25.000 € inkl. MwSt.", "zzgl. Überführung", "max. Leistung", "zul. Gesamtgewicht"
        "de": frozenset(
            {
                "inkl", "exkl", "zzgl", "abzgl", "ca", "bzw", "ggf", "evtl", "bspw", "max",
                "nr", "mwst", "ust", "vgl", "lt", "gem", "mtl", "eff", "zul",
            }
        ),
    }
)  # fmt: skip
"""Mid-sentence abbreviations by pysbd language code, used on top of :data:`ABBREVIATIONS`
for a page in that language."""
_LAST_WORD = re.compile(r"(\w+)\.$")
# What may follow a mid-sentence abbreviation: a lowercase word, a number, a symbol or an
# all-caps acronym ("excl. VAT"). A capitalised word starts a real sentence ("5 min. Then").
_CONTINUES = re.compile(r"^(?:[a-z0-9£$€(\[%&+\-–]|[A-Z]{2,5}\b)")
# German capitalises nouns, so a capitalised word after an abbreviation that qualifies what
# follows it continues the sentence ("inkl. Versand"), unless it's a word that usually opens
# one. Noun abbreviations ("MwSt.", "Nr.") often end a sentence, so they aren't listed.
_PREPOSITIVE = {
    "de": frozenset(
        {
            "inkl", "exkl", "zzgl", "abzgl", "ca", "bzw", "ggf", "evtl", "bspw", "max", "vgl",
            "lt", "gem", "mtl", "eff", "zul",
        }
    ),
}  # fmt: skip
_SENTENCE_OPENERS = {
    "de": frozenset(
        {
            "Der", "Die", "Das", "Den", "Dem", "Des", "Ein", "Eine", "Einen", "Einem", "Einer",
            "Eines", "Er", "Sie", "Es", "Wir", "Ich", "Ihr", "Man", "Dieser", "Diese", "Dieses",
        }
    ),
}  # fmt: skip
_FIRST_WORD = re.compile(r"^[^\W\d_]+")
# A fragment ending in a dotted model name ("The VW ID.3") has no sentence-ending mark.
_DOTTED_NAME = re.compile(r"\b\w+\.\d\w*$")
MAX_SEGMENT_CHARS = 3000
"""pysbd slows quadratically on long lines, so longer lines are cut at clear sentence
boundaries into pieces of about this size first."""
_CLEAR_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


def _chunks(text: str) -> list[str]:
    """Pieces of about :data:`MAX_SEGMENT_CHARS`, cut at clear sentence boundaries, or at
    whitespace when a stretch twice that long has none."""
    out: list[str] = []
    while len(text) > MAX_SEGMENT_CHARS:
        limit = 2 * MAX_SEGMENT_CHARS
        cut = None
        for m in _CLEAR_BOUNDARY.finditer(text, MAX_SEGMENT_CHARS, limit):
            cut = (m.start(), m.end())
            break
        if cut is None:
            space = text.rfind(" ", MAX_SEGMENT_CHARS, limit)
            if space == -1:
                space = text.find(" ", limit)
            if space == -1:
                break
            cut = (space, space + 1)
        out.append(text[: cut[0]])
        text = text[cut[1] :]
    out.append(text)
    return out


def _mis_split(fragment: str, following: str, language: str) -> bool:
    if _DOTTED_NAME.search(fragment):
        return True
    word = _LAST_WORD.search(fragment)
    if word is None:
        return False
    abbreviation = word.group(1).lower()
    if (
        abbreviation in ABBREVIATIONS
        or abbreviation in LANGUAGE_ABBREVIATIONS.get(language, frozenset())
    ) and _CONTINUES.match(following):
        return True
    first = _FIRST_WORD.match(following)
    return (
        abbreviation in _PREPOSITIVE.get(language, frozenset())
        and first is not None
        and first.group() not in _SENTENCE_OPENERS.get(language, frozenset())
    )


def sentences(text: str, *, language: str = "en") -> list[str]:
    """``text`` split into sentences with pysbd, stripped and without empties.

    Fragments pysbd splits after a mid-sentence abbreviation (:data:`ABBREVIATIONS`, plus
    ``language``'s own in :data:`LANGUAGE_ABBREVIATIONS`) or a dotted model name ("ID.3")
    are joined back up. Those word lists guess where a sentence ends; they are code
    because sentences are what Jev is asked about, and asking about each full stop would
    cost a question per sentence. A new pysbd segmenter is built per call:
    they keep per-call state, so sharing one across threads loses text.
    """
    segmenter_class, lang = _pysbd_language(language)
    pieces: list[str] = []
    for chunk in _chunks(text):
        segmenter = segmenter_class(language=lang, clean=False, char_span=True)
        pieces.extend(s for span in segmenter.segment(chunk) if (s := span.sent.strip()))
    out: list[str] = []
    for piece in pieces:
        if out and _mis_split(out[-1], piece, lang):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


@dataclass(frozen=True)
class DefaultSplitter:
    """The default :class:`~jevex.interfaces.StatementSplitter`.

    ``language`` is the pysbd language code (``en``, ``de``, ``fr``...); unsupported codes
    fall back to English. :class:`StatementStage` splits each document by its own
    language instead (:meth:`split_in`). Only the component itself is split, never its
    children: the stage calls the splitter on every component in the tree.
    """

    language: str = "en"

    def split(self, component: Component) -> list[Statement]:
        return self.split_in(component, None)

    def split_in(self, component: Component, locale: str | None) -> list[Statement]:
        """Split with the sentence rules of ``locale``'s language (``de-AT`` → ``de``), or
        of ``language`` when ``locale`` is ``None`` or empty."""
        language = locale_conventions(locale).language if locale else self.language
        kind = component.type
        text = component.text.strip()
        if not text:
            return []
        pieces: list[tuple[str, StatementKind]]
        if kind == "paragraph":
            pieces = self._paragraph(text, language)
        elif kind == "list_item":
            pieces = self._list_item(text, _dl_part(component))
        elif kind == "heading":
            pieces = [(_WHITESPACE.sub(" ", text), "sentence")]
        elif kind == "caption":
            pieces = [(_WHITESPACE.sub(" ", text), "caption")]
        elif kind == "image":
            pieces = [(_WHITESPACE.sub(" ", text), "alt_text")]
        elif kind == "table":
            return table_statements(component)
        else:  # containers: their children are split instead
            return []
        if isinstance(component.location, ImageLocation):
            pieces = [(p, "ocr" if k == "sentence" else k) for p, k in pieces]
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

    def _paragraph(self, text: str, language: str) -> list[tuple[str, StatementKind]]:
        out: list[tuple[str, StatementKind]] = []
        for line in _lines(text):
            said = sentences(line, language=language)
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
        # The layout parser renders a dd as "term: value", so trust it even for labels
        # is_key_value would doubt ("0-62: 9.1 s", "2019: Launched").
        pair = ": " in one if dl_part == "dd" else is_key_value(one) and dl_part != "dt"
        return [(one, "key_value" if pair else "list_item")]


def _lines(text: str) -> list[str]:
    return [line for raw in text.split("\n") if (line := _WHITESPACE.sub(" ", raw).strip())]


MAX_STATEMENT_CHARS = 4000
"""Statements longer than this are cut into pieces. Jev takes about 115k characters of
state plus question, but a statement is meant to state one thing, and a shorter state
leaves room for the heading trail and a Choice over many fields or candidates."""
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _cut_text(text: str, max_chars: int) -> list[str]:
    """``text`` in pieces of at most ``max_chars``, cut at the last sentence end in the
    second half of each piece, else the last whitespace there, else mid-word."""
    out: list[str] = []
    while len(text) > max_chars:
        floor = max_chars // 2
        window = text[: max_chars + 1]  # a cut at max_chars keeps the piece within it
        cut = None
        for pattern in (_SENTENCE_END, _WHITESPACE):
            ends = [m.start() for m in pattern.finditer(window, floor) if m.start() > 0]
            if ends:
                cut = ends[-1]
                break
        if cut is None:
            out.append(text[:max_chars])
            text = text[max_chars:].lstrip()
        else:
            out.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
    if text:
        out.append(text)
    return out


def _lead(statement: Statement) -> str:
    """The part every piece of a cut statement repeats: a table cell's headers or a
    pair's label, so each piece still says what it's about."""
    text = statement.text
    if statement.table is not None:
        prefix = header_prefix(statement.table)
        return prefix if prefix and text.startswith(prefix) else ""
    if statement.kind == "key_value" and (m := _KEY_VALUE.match(text)) is not None:
        return text[: m.start("value")]
    return ""


def cut_statement(statement: Statement, max_chars: int = MAX_STATEMENT_CHARS) -> list[Statement]:
    """``statement`` in pieces of at most ``max_chars`` characters, or ``[statement]`` if it
    fits.

    Cuts fall at a sentence end where the second half of a piece has one, else at
    whitespace, else mid-word; pieces don't overlap, so a value straddling a mid-word cut
    is lost. A table cell's headers or a pair's label (when no longer than half of
    ``max_chars``) lead every piece: ``Notes: …`` stays a pair. Pieces keep everything
    else about the statement; their ids are ``<id>:<n>``.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if len(statement.text) <= max_chars:
        return [statement]
    lead = _lead(statement)
    if len(lead) > max_chars // 2:
        lead = ""
    pieces = _cut_text(statement.text[len(lead) :].strip(), max_chars - len(lead))
    return [
        statement.model_copy(update={"id": f"{statement.id}:{i}", "text": f"{lead}{piece}"})
        for i, piece in enumerate(pieces)
    ]


class DuplicateStatementError(ValueError):
    """Two statements on one document share an id (a splitter or earlier stage bug)."""


@dataclass
class StatementStage:
    """Splits every component of ``ctx.parsed`` into statements (stage 8).

    Statements already on the document are kept: those of components outside the tree
    (the structured-data stage's) first, then the rest in reading order, with a
    component that already has statements (the image stage's ``vision`` statements) not
    split again. An id clash raises :class:`DuplicateStatementError` rather than
    replacing one. A statement longer than
    ``max_chars`` is cut (:func:`cut_statement`) and a ``statements_cut`` event lists
    which. Without a parsed document the stage does nothing. A table without header cells
    that the component gate found headers in (:attr:`Context.headed_tables
    <jevex.pipeline.Context.headed_tables>`) is split with them marked
    (:func:`~jevex.tables.infer_headers`). A statement naming one of a run of sibling
    sections or cards (a trim's heading) gets the run's names as context
    (:func:`~jevex.resolve.sibling_labels`), whichever splitter made it.

    A :class:`~jevex.interfaces.LocaleAwareSplitter` (the default) splits by the
    document's locale (:attr:`Context.locale <jevex.pipeline.Context.locale>`: the
    caller's, ``<html lang>`` or ``Content-Language``), else the stage's ``locale``, else
    the pipeline's :class:`~jevex.CandidateStage` ``locale`` (so one configured locale
    serves both), else the splitter's own language (English unless set). A language pysbd
    doesn't know is split as English.
    """

    splitter: StatementSplitter = field(default_factory=DefaultSplitter)
    locale: str | None = None
    max_chars: int = MAX_STATEMENT_CHARS
    name: str = "statements"

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError(f"max_chars must be positive, got {self.max_chars}")

    async def run(self, ctx: Context) -> None:
        parsed = ctx.parsed
        if parsed is None:
            return
        in_tree = {c.id for c in parsed.root.walk()}
        existing: dict[str, list[Statement]] = {}
        statements: dict[str, Statement] = {}
        for statement in parsed.statements.values():
            if statement.component_id in in_tree:
                existing.setdefault(statement.component_id, []).append(statement)
            else:
                statements[statement.id] = statement
        cut: dict[str, int] = {}
        split = self._split_for(ctx.locale or self.locale or candidate_locale(ctx))
        for component in parsed.root.walk():
            if component.id in existing:
                statements.update((s.id, s) for s in existing[component.id])
                continue
            if component.id in ctx.headed_tables:
                component = infer_headers(component)
            for whole in split(component):
                pieces = cut_statement(whole, self.max_chars)
                if len(pieces) > 1:
                    cut[whole.id] = len(pieces)
                for statement in pieces:
                    if statement.id in statements or statement.id in parsed.statements:
                        raise DuplicateStatementError(
                            f"statement id {statement.id!r} (component {component.id!r}) is "
                            "already on the document"
                        )
                    statements[statement.id] = statement
        parsed.statements = statements
        for sid, labels in sibling_labels(parsed).items():
            statements[sid] = statements[sid].model_copy(update={"sibling_labels": labels})
        if cut:
            ctx.event(
                self.name,
                "statements_cut",
                f"{len(cut)} statement(s) over {self.max_chars} characters were cut into "
                f"{sum(cut.values())} pieces",
                pieces=cut,
            )

    def _split_for(self, locale: str | None) -> Callable[[Component], list[Statement]]:
        splitter = self.splitter
        if isinstance(splitter, LocaleAwareSplitter):
            return lambda component: splitter.split_in(component, locale)
        return splitter.split

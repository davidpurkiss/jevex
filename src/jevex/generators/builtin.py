"""Built-in candidate generators.

Generators are built for recall: they propose every plausible verbatim span, and Jev picks
the right one. Each candidate carries the normaliser chain that turns its raw text into a
typed value. The normalisers themselves are the normaliser stage's job (#19).

Normaliser steps these generators emit:

- ``parse_number``: "18,495" → 18495, "9.1" → 9.1
- ``{unit: {from: <canonical>}}``: the unit found; the normaliser converts to the field's unit
- ``{parse_money: {currency: <code>}}``: "£18,495", "25k GBP", "£1.5m", "€2bn",
  "£1.5 million" → amount in that currency, with any multiplier (k, m, bn, thousand,
  million, billion) applied
- ``{parse_date: {order?, precision?}}``: dates, month-years and years
- ``parse_range``: "5–7" → [5, 7]
- ``strip``: trim whitespace and trailing punctuation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jevex.generators.units import spellings
from jevex.interfaces import Scope
from jevex.statements import Candidate, NormaliserStep, Span, Statement

if TYPE_CHECKING:
    from jevex.schema import FieldSpec

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_NUMBER = re.compile(rf"(?<![\w.,])(?:{_NUM})(?![\w]|[.,]\d)")


def _unit_alternation() -> str:
    parts: list[str] = []
    for spelling, _canonical, case_sensitive in spellings():
        escaped = re.escape(spelling).replace(r"\ ", r"\s?")
        parts.append(escaped if case_sensitive else f"(?i:{escaped})")
    return "|".join(parts)


_UNIT_TO_CANONICAL = {
    (spelling if case_sensitive else spelling.lower()): canonical
    for spelling, canonical, case_sensitive in spellings()
}
_NUMBER_WITH_UNIT = re.compile(
    rf"(?<![\w.,])(?P<num>{_NUM})\s?(?P<unit>{_unit_alternation()})(?![A-Za-z0-9])"
)


def _canonical_unit(found: str) -> str:
    compact = re.sub(r"\s+", " ", found)
    return _UNIT_TO_CANONICAL.get(compact) or _UNIT_TO_CANONICAL.get(compact.lower(), compact)


def _candidate(
    statement: Statement, start: int, end: int, generator_id: str, *steps: NormaliserStep
) -> Candidate:
    return Candidate.from_statement(
        statement, Span(start=start, end=end), generator_id=generator_id, normalise=list(steps)
    )


def _step(name: str, **args: object) -> NormaliserStep:
    return NormaliserStep(name=name, args=dict(args))


@dataclass(frozen=True)
class NumberWithUnit:
    """Numbers with a unit from the lexicon ("150PS", "9.1 s"), plus bare numbers."""

    id: str = "number_with_unit"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"number"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        text = statement.text
        out: list[Candidate] = []
        covered: set[int] = set()
        for m in _NUMBER_WITH_UNIT.finditer(text):
            unit = _canonical_unit(m.group("unit"))
            out.append(
                _candidate(
                    statement,
                    m.start(),
                    m.end(),
                    self.id,
                    _step("parse_number"),
                    _step("unit", **{"from": unit}),
                )
            )
            covered.add(m.start("num"))
        for m in _NUMBER.finditer(text):
            if m.start() not in covered:
                out.append(
                    _candidate(statement, m.start(), m.end(), self.id, _step("parse_number"))
                )
        return sorted(out, key=lambda c: c.span.start)


_CURRENCY_SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY"}
_CODES = "GBP|USD|EUR|JPY|CHF|AUD|CAD"
# "£25k", "£1.5m", "€2bn", and spelled or spaced: "£1.5 million", "EUR 3 bn", "£2 m".
_MULTIPLIER = r"(?:bn|[kKmM](?![a-zA-Z])|\s?(?i:million|billion|thousand|mn|bn|m)\b)"
# Every amount must end cleanly: "£18,4950" or "£1.5x" yield nothing rather than a
# truncated (and silently wrong) "£18,495" / "£1.5". A period suffix may follow
# directly: "£299pm", "£1,200pcm", "£45pw", "£30,000pa".
_END = r"(?:(?![\w]|[.,]\d)|(?=p(?:cm|m|a|w)\b))"
_MONEY = re.compile(
    rf"(?P<sym>[£$€¥])\s?(?:{_NUM}){_MULTIPLIER}?{_END}"
    rf"|(?<![\w.,])(?:{_NUM}){_MULTIPLIER}?\s?(?P<c2>{_CODES})\b"
    rf"|\b(?P<c3>{_CODES})\s?(?:{_NUM}){_MULTIPLIER}?{_END}"
)


@dataclass(frozen=True)
class Money:
    """Amounts with a currency symbol or code: "£18,495", "25k GBP", "EUR 30,000"."""

    id: str = "money"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"number"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        out: list[Candidate] = []
        for m in _MONEY.finditer(statement.text):
            if m.group("sym"):
                currency = _CURRENCY_SYMBOLS[m.group("sym")]
            else:
                currency = m.group("c2") or m.group("c3")
            out.append(
                _candidate(
                    statement,
                    m.start(),
                    m.end(),
                    self.id,
                    _step("parse_money", currency=currency),
                )
            )
        return out


_MONTH = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)
_DATE_PATTERNS: tuple[tuple[re.Pattern[str], dict[str, object]], ...] = (
    (re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)"), {"order": "ymd"}),
    (re.compile(r"(?<!\d)\d{1,2}[/.]\d{1,2}[/.]\d{4}(?!\d)"), {"order": "dmy"}),
    (
        re.compile(rf"(?<!\d)\d{{1,2}}(?:st|nd|rd|th)?\s+(?i:{_MONTH})\.?,?\s+\d{{4}}(?!\d)"),
        {"order": "dmy"},
    ),
    (
        re.compile(rf"\b(?i:{_MONTH})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,\s+\d{{4}}(?!\d)"),
        {"order": "mdy"},
    ),
    (re.compile(rf"\b(?i:{_MONTH})\.?\s+\d{{4}}(?!\d)"), {"precision": "month"}),
)


@dataclass(frozen=True)
class DateGenerator:
    """Absolute dates and month-years: "2024-03-12", "12/03/2024", "12 March 2024", "March 2024"."""

    id: str = "date"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"date"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        out: list[Candidate] = []
        seen: set[tuple[int, int]] = set()
        for pattern, args in _DATE_PATTERNS:
            for m in pattern.finditer(statement.text):
                if (m.start(), m.end()) in seen:
                    continue
                seen.add((m.start(), m.end()))
                out.append(
                    _candidate(statement, m.start(), m.end(), self.id, _step("parse_date", **args))
                )
        return sorted(out, key=lambda c: (c.span.start, -c.span.end))


# Standalone years, plus the model-year form "MY2024"; never inside VINs or part numbers.
_YEAR = re.compile(r"(?:(?<=MY)|(?<![\w.,]))(?:19|20)\d{2}(?![\w]|[.,]\d)")


@dataclass(frozen=True)
class Year:
    """Four-digit years 1900–2099, including model years ("2024 model year")."""

    id: str = "year"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"date", "number"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        return [
            _candidate(
                statement, m.start(), m.end(), self.id, _step("parse_date", precision="year")
            )
            for m in _YEAR.finditer(statement.text)
        ]


_RANGE = re.compile(
    rf"(?<![\w.,/-])(?P<lo>{_NUM})\s*(?:[-–—]|to)\s*(?P<hi>{_NUM})(?!\w|[.,/–-]\d)"
    rf"(?:\s?(?P<unit>{_unit_alternation()})(?![A-Za-z0-9]))?"
    rf"|\b(?i:between)\s+(?P<lo2>{_NUM})\s+(?i:and)\s+(?P<hi2>{_NUM})"
)


@dataclass(frozen=True)
class Range:
    """Numeric ranges: "5–7", "5 to 7", "between 5 and 7", optionally with a unit."""

    id: str = "range"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"number"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        out: list[Candidate] = []
        for m in _RANGE.finditer(statement.text):
            steps = [_step("parse_range")]
            if m.group("unit"):
                steps.append(_step("unit", **{"from": _canonical_unit(m.group("unit"))}))
            out.append(_candidate(statement, m.start(), m.end(), self.id, *steps))
        return out


_KEY_VALUE = re.compile(r":\s+(?=\S)")
_TRAILING = " \t\r\n.;,"
_DIGIT = re.compile(r"\d")


def _number_chain(value: str) -> list[NormaliserStep] | None:
    """The chain for the first quantity in ``value``: a range, money, a number with a unit or
    a bare number, in that order when two start at the same place."""
    found: list[tuple[int, int, list[NormaliserStep] | None]] = []
    if m := _RANGE.search(value):
        steps = [_step("parse_range")]
        if m.group("unit"):
            steps.append(_step("unit", **{"from": _canonical_unit(m.group("unit"))}))
        found.append((m.start(), 0, steps))
    if m := _MONEY.search(value):
        currency = (
            _CURRENCY_SYMBOLS[m.group("sym")] if m.group("sym") else m.group("c2") or m.group("c3")
        )
        found.append((m.start(), 1, [_step("parse_money", currency=currency)]))
    if m := _NUMBER_WITH_UNIT.search(value):
        steps = [_step("parse_number"), _step("unit", **{"from": _canonical_unit(m.group("unit"))})]
        found.append((m.start(), 2, steps))
    if m := _NUMBER.search(value):
        # A bare number with more after it ("1.5 TSI 150PS") could be any of them: skip it.
        bare = None if _DIGIT.search(value, m.end()) else [_step("parse_number")]
        found.append((m.start(), 3, bare))
    return _first(value, found)


def _date_chain(value: str) -> list[NormaliserStep] | None:
    """The chain for the first date in ``value``, longest form first; a bare year last."""
    found: list[tuple[int, int, list[NormaliserStep] | None]] = []
    for rank, (pattern, args) in enumerate(_DATE_PATTERNS):
        if m := pattern.search(value):
            found.append((m.start(), rank, [_step("parse_date", **args)]))
    if m := _YEAR.search(value):
        found.append((m.start(), len(_DATE_PATTERNS), [_step("parse_date", precision="year")]))
    return _first(value, found)


def _first(
    value: str, found: list[tuple[int, int, list[NormaliserStep] | None]]
) -> list[NormaliserStep] | None:
    """The earliest match's chain. None if nothing matched, or if a digit comes before it:
    the parsers read the first number in the raw text, which would then be the wrong one."""
    if not found:
        return None
    start, _, steps = min(found, key=lambda f: (f[0], f[1]))
    return None if _DIGIT.search(value, 0, start) else steps


@dataclass(frozen=True)
class KeyValue:
    """The value side of ``label: value`` statements, including rendered table cells.

    The chain fits the field (via ``generate_for``): numbers get ``parse_number`` plus the
    unit found, ``parse_money`` or ``parse_range``; dates get ``parse_date``; strings get
    ``strip``. A number or date field gets no candidate when the value holds nothing that
    could parse. Without a field, ``generate`` gives the ``strip`` chain.
    """

    id: str = "key_value"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"number", "date", "str"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        span = _value_span(statement.text)
        if span is None:
            return []
        return [_candidate(statement, *span, self.id, _step("strip"))]

    def generate_for(self, statement: Statement, field: FieldSpec) -> list[Candidate]:
        span = _value_span(statement.text)
        if span is None:
            return []
        value = statement.text[span[0] : span[1]]
        if field.kind == "number":
            steps = _number_chain(value)
        elif field.kind == "date":
            steps = _date_chain(value)
        else:
            steps = [_step("strip")]
        if steps is None:
            return []
        return [_candidate(statement, *span, self.id, *steps)]


def _value_span(text: str) -> tuple[int, int] | None:
    """Offsets of the text after the last ``: `` separator, minus trailing punctuation."""
    separators = list(_KEY_VALUE.finditer(text))
    if not separators:
        return None
    start = separators[-1].end()
    end = len(text.rstrip(_TRAILING))
    return (start, end) if end > start else None


_WORD = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|[\w][\w'’&/+-]*(?:\.\d+)?")
# A comma between digits ("18,495") is a thousands separator, not a phrase break.
_BREAK = re.compile(r"(?<!\d),|,(?!\d)|[;:()\[\]!?\"“”|·›]|\.(?=\s|$)|\s[-–—]\s")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "nor",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "without",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "not",
        "no",
        "yes",
        "also",
        "very",
        "more",
        "most",
        "less",
        "least",
        "than",
        "then",
        "so",
        "such",
        "only",
        "just",
        "up",
        "down",
        "out",
        "over",
        "under",
        "into",
        "onto",
        "via",
        "per",
        "each",
        "every",
        "all",
        "any",
        "some",
        "both",
        "either",
        "neither",
        "our",
        "your",
        "their",
        "his",
        "her",
        "we",
        "you",
        "they",
        "i",
    ]
)
MAX_PHRASE_WORDS = 8


@dataclass(frozen=True)
class NounPhrase:
    """Word runs between punctuation and stopwords: candidate names, colours, trims.

    A light heuristic chunker with no model dependency. "Available in Moonstone Grey
    metallic" gives "Available" and "Moonstone Grey metallic".
    """

    id: str = "noun_phrase"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"str"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        text = statement.text
        breaks = {m.start() for m in _BREAK.finditer(text)}
        out: list[Candidate] = []
        run: list[re.Match[str]] = []

        def flush() -> None:
            # Long runs become consecutive chunks, so no words are dropped.
            for i in range(0, len(run), MAX_PHRASE_WORDS):
                words = run[i : i + MAX_PHRASE_WORDS]
                if not all(w.group().replace(".", "").isdigit() for w in words):
                    out.append(
                        _candidate(
                            statement, words[0].start(), words[-1].end(), self.id, _step("strip")
                        )
                    )
            run.clear()

        last_end = 0
        for word in _WORD.finditer(text):
            gap = range(last_end, word.start())
            if run and any(i in breaks for i in gap):
                flush()
            if word.group().lower() in STOPWORDS:
                flush()
            else:
                run.append(word)
            last_end = word.end()
        flush()
        return out


MAX_WHOLE_WORDS = 16
_WHOLE_TRAILING = " \t\r\n.,;:"
"""Stripped from the end of a whole statement. ``?`` and ``!`` stay: they can belong to a
title ("Who Moved My Cheese?")."""


@dataclass(frozen=True)
class WholeStatement:
    """A short statement's whole text, for names and titles that stopwords would split.

    "A Light in the Attic" as a heading gives the candidate "A Light in the Attic" (the
    noun-phrase chunker gives only "Light" and "Attic"). Statements of more than
    ``max_words`` words aren't proposed: whole sentences are rarely a value. Nor are
    ``key_value`` and ``table_cell`` statements, whose value :class:`KeyValue` finds, or
    statements for ``list[...]`` fields, where the whole text is several values at once.
    """

    id: str = "whole_statement"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"str"})))
    max_words: int = MAX_WHOLE_WORDS

    def generate(self, statement: Statement) -> list[Candidate]:
        if statement.kind in ("key_value", "table_cell"):
            return []
        text = statement.text
        # Count words first, so a long statement costs one split and no scanning.
        if len(text.split(maxsplit=self.max_words)) > self.max_words:
            return []
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip(_WHOLE_TRAILING))
        if end <= start:
            return []
        return [_candidate(statement, start, end, self.id, _step("strip"))]

    def generate_for(self, statement: Statement, field: FieldSpec) -> list[Candidate]:
        return [] if field.many else self.generate(statement)


BUILTIN_GENERATORS = (
    NumberWithUnit(),
    Money(),
    DateGenerator(),
    Year(),
    Range(),
    KeyValue(),
    NounPhrase(),
    WholeStatement(),
)

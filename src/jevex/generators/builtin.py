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

The number, money, range, date and key-value generators match the way the document's
locale writes numbers and dates (:mod:`jevex.locales`): on a ``de-DE`` page "1.234,5 kg"
is one number, "18.495 €", "18.495,- €" and "1,5 Mio. €" amounts, "1,4 bis 2,0 l" a range
and "12. März 2024" a date, and their chains carry ``decimal: ","``; on an ``en-US`` page
"03/12/2024" is read month first and mpg in US gallons; on a ``de-CH`` page "1’250.50" is
one number. Month names, multipliers and range words are English plus the page language's
(:mod:`jevex.locales` lists them and what isn't read). An unknown locale is read as en-GB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

from jevex.generators.units import spellings
from jevex.interfaces import Scope
from jevex.locales import (
    EN_GB,
    LONG_SCALE_BILLION,
    MONTH_NAMES,
    MULTIPLIERS,
    RANGE_WORDS,
    locale_conventions,
    localise_steps,
)
from jevex.statements import Candidate, NormaliserStep, Span, Statement

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jevex.locales import LocaleConventions
    from jevex.schema import FieldSpec


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


_CURRENCY_SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY"}
_CODES = "GBP|USD|EUR|JPY|CHF|AUD|CAD"
# "£25k", "£1.5m", "€2bn", and spelled or spaced: "£1.5 million", "EUR 3 bn", "£2 m".
_ENGLISH = "million|billion|thousand|mn|bn|m"
# Every amount must end cleanly: "£18,4950" or "£1.5x" yield nothing rather than a
# truncated (and silently wrong) "£18,495" / "£1.5". A period suffix may follow
# directly: "£299pm", "£1,200pcm", "£45pw", "£30,000pa".
_END = r"(?:(?![\w]|[.,]\d)|(?=p(?:cm|m|a|w)\b))"
_MONTH = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)

DatePatterns = tuple[tuple[re.Pattern[str], dict[str, object]], ...]


@dataclass(frozen=True)
class _Patterns:
    """One locale's compiled patterns."""

    conventions: LocaleConventions
    number: re.Pattern[str]
    number_with_unit: re.Pattern[str]
    money: re.Pattern[str]
    dates: DatePatterns
    range: re.Pattern[str]

    def steps(self, *steps: NormaliserStep) -> list[NormaliserStep]:
        """An en-GB chain with the arguments this locale needs."""
        return localise_steps(steps, self.conventions)


def _num(conventions: LocaleConventions) -> str:
    group = f"[{conventions.thousands}]" if len(conventions.thousands) > 1 else ","
    if conventions.decimal == ".":
        return rf"\d{{1,3}}(?:{group}\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?"
    return rf"\d{{1,3}}(?:{group}\d{{3}})+(?:,\d+)?|\d+(?:,\d+)?"


def _words(words: Iterable[str]) -> str:
    """An alternation of words, longest first; a space in one matches any whitespace."""
    ordered = sorted(set(words), key=lambda w: (-len(w), w))
    return "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in ordered)


def _months(language: str) -> str:
    """English month names, then the language's own (longest first)."""
    extra = set(MONTH_NAMES.get(language, {})) - set(MONTH_NAMES["en"])
    if not extra:
        return _MONTH
    return f"{_MONTH}|{_words(extra)}"


def _multiplier(language: str) -> str:
    """English multipliers, plus the language's own with an optional dot ("1,5 Mio. €").
    Where "billion" is 10^12 the English word isn't one."""
    english = "million|thousand|mn|bn|m" if language in LONG_SCALE_BILLION else _ENGLISH
    own = MULTIPLIERS.get(language)
    local = rf"|\s?(?i:{_words(own)})\.?(?![^\W\d_])" if own else ""
    return rf"(?:bn|[kKmM](?![a-zA-Z]){local}|\s?(?i:{english})\b)"


def _round(conventions: LocaleConventions) -> str:
    """A round amount's dash after the decimal mark: "18.495,- €", "Fr. 1’250.–"."""
    if conventions.decimal == ",":
        return "(?:,[-–—]{1,2})?"
    return r"(?:\.[-–—]{1,2})?" if conventions.apostrophe_groups else ""


def _range(num: str, units: str, language: str) -> str:
    """Ranges with a dash or "to", or "between … and", plus the language's own words.

    A dash between numbers can also be a name or a score ("0-62 mph", "3-1"), and
    Spanish "a" or Italian "e" join much else: the range is only proposed, and Jev's
    Choice tells the readings apart."""
    to, between, and_ = "", "between", "and"
    if words := RANGE_WORDS.get(language):
        to = rf"|\s+(?i:{_words(words.to)})\s+"
        between = f"between|{_words(words.between)}"
        and_ = f"and|{_words(words.and_)}"
    return (
        rf"(?<![\w.,/-])(?P<lo>{num})(?:\s*(?:[-–—]|to)\s*{to})(?P<hi>{num})(?!\w|[.,/–-]\d)"
        rf"(?:\s?(?P<unit>{units})(?![A-Za-z0-9]))?"
        rf"|\b(?i:{between})\s+(?P<lo2>{num})\s+(?i:{and_})\s+(?P<hi2>{num})"
    )


@cache
def _patterns(conventions: LocaleConventions) -> _Patterns:
    language = conventions.language
    num = _num(conventions)
    units = _unit_alternation()
    mult = _multiplier(language)
    amount = f"(?:{num}){_round(conventions)}{mult}?"
    # Without a multiplier, "€ 1 Billion" or "€ 1,2 Bio." would give a truncated "€ 1".
    end = rf"(?!\s?(?i:billion|bio\b)){_END}" if language in LONG_SCALE_BILLION else _END
    money = (
        rf"(?P<sym>[£$€¥])\s?{amount}{end}"
        rf"|(?<![\w.,]){amount}\s?(?P<c2>{_CODES})\b"
        rf"|\b(?P<c3>{_CODES})\s?{amount}{end}"
    )
    if conventions.currency_after:  # "18.495 €", "18.495,50 €", "1,5 Mio. €"
        money += rf"|(?<![\w.,]){amount}\s?(?P<sym2>[£$€¥])"
    month = _months(language)
    # German writes the day as an ordinal with a dot ("12. März 2024"), French the 1st as
    # "1er"; Spanish puts "de" around the month ("12 de marzo de 2024").
    ordinal = "(?:st|nd|rd|th)?" if language == "en" else r"(?:st|nd|rd|th|er|\.)?"
    of = r"(?:\s+de)?" if language == "es" else ""
    return _Patterns(
        conventions=conventions,
        number=re.compile(rf"(?<![\w.,])(?:{num})(?![\w]|[.,]\d)"),
        number_with_unit=re.compile(
            rf"(?<![\w.,])(?P<num>{num})\s?(?P<unit>{units})(?![A-Za-z0-9])"
        ),
        money=re.compile(money),
        dates=(
            (re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)"), {"order": "ymd"}),
            (
                re.compile(r"(?<!\d)\d{1,2}[/.]\d{1,2}[/.]\d{4}(?!\d)"),
                {"order": conventions.date_order},
            ),
            (
                re.compile(
                    rf"(?<!\d)\d{{1,2}}{ordinal}{of}\s+(?i:{month})\.?,?{of}\s+\d{{4}}(?!\d)"
                ),
                {"order": "dmy"},
            ),
            (
                re.compile(rf"\b(?i:{month})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,\s+\d{{4}}(?!\d)"),
                {"order": "mdy"},
            ),
            (re.compile(rf"\b(?i:{month})\.?{of}\s+\d{{4}}(?!\d)"), {"precision": "month"}),
        ),
        range=re.compile(_range(num, units, language)),
    )


def _for_locale(locale: str | None) -> _Patterns:
    return _patterns(locale_conventions(locale))


_EN_GB = _patterns(EN_GB)


def _currency(m: re.Match[str]) -> str:
    if m.group("sym"):
        return _CURRENCY_SYMBOLS[m.group("sym")]
    if "sym2" in m.re.groupindex and m.group("sym2"):
        return _CURRENCY_SYMBOLS[m.group("sym2")]
    return m.group("c2") or m.group("c3")


@dataclass(frozen=True)
class NumberWithUnit:
    """Numbers with a unit from the lexicon ("150PS", "9.1 s"), plus bare numbers."""

    id: str = "number_with_unit"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"number"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        return self._generate(statement, _EN_GB)

    def generate_in(
        self, statement: Statement, field: FieldSpec, locale: str | None
    ) -> list[Candidate]:
        return self._generate(statement, _for_locale(locale))

    def _generate(self, statement: Statement, patterns: _Patterns) -> list[Candidate]:
        text = statement.text
        out: list[Candidate] = []
        covered: set[int] = set()
        for m in patterns.number_with_unit.finditer(text):
            unit = _canonical_unit(m.group("unit"))
            steps = patterns.steps(_step("parse_number"), _step("unit", **{"from": unit}))
            out.append(_candidate(statement, m.start(), m.end(), self.id, *steps))
            covered.add(m.start("num"))
        for m in patterns.number.finditer(text):
            if m.start() not in covered:
                steps = patterns.steps(_step("parse_number"))
                out.append(_candidate(statement, m.start(), m.end(), self.id, *steps))
        return sorted(out, key=lambda c: c.span.start)


@dataclass(frozen=True)
class Money:
    """Amounts with a currency symbol or code: "£18,495", "25k GBP", "EUR 30,000"."""

    id: str = "money"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"number"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        return self._generate(statement, _EN_GB)

    def generate_in(
        self, statement: Statement, field: FieldSpec, locale: str | None
    ) -> list[Candidate]:
        return self._generate(statement, _for_locale(locale))

    def _generate(self, statement: Statement, patterns: _Patterns) -> list[Candidate]:
        return [
            _candidate(
                statement,
                m.start(),
                m.end(),
                self.id,
                *patterns.steps(_step("parse_money", currency=_currency(m))),
            )
            for m in patterns.money.finditer(statement.text)
        ]


@dataclass(frozen=True)
class DateGenerator:
    """Absolute dates and month-years: "2024-03-12", "12/03/2024", "12 March 2024", "March 2024"."""

    id: str = "date"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"date"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        return self._generate(statement, _EN_GB)

    def generate_in(
        self, statement: Statement, field: FieldSpec, locale: str | None
    ) -> list[Candidate]:
        return self._generate(statement, _for_locale(locale))

    def _generate(self, statement: Statement, patterns: _Patterns) -> list[Candidate]:
        out: list[Candidate] = []
        seen: set[tuple[int, int]] = set()
        for pattern, args in patterns.dates:
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
    """Four-digit years 1900–2099, including model years ("2024 model year").

    Earlier four-digit numbers aren't proposed: in a statement they are far more often
    quantities ("1498 cc", "1600 kg") than years, and each would be one more option for
    Jev to rule out.
    """

    id: str = "year"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"date", "number"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        return [
            _candidate(
                statement, m.start(), m.end(), self.id, _step("parse_date", precision="year")
            )
            for m in _YEAR.finditer(statement.text)
        ]


def _range_steps(m: re.Match[str], patterns: _Patterns) -> list[NormaliserStep]:
    steps = [_step("parse_range")]
    if m.group("unit"):
        steps.append(_step("unit", **{"from": _canonical_unit(m.group("unit"))}))
    return patterns.steps(*steps)


@dataclass(frozen=True)
class Range:
    """Numeric ranges: "5–7", "5 to 7", "between 5 and 7", optionally with a unit."""

    id: str = "range"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"number"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        return self._generate(statement, _EN_GB)

    def generate_in(
        self, statement: Statement, field: FieldSpec, locale: str | None
    ) -> list[Candidate]:
        return self._generate(statement, _for_locale(locale))

    def _generate(self, statement: Statement, patterns: _Patterns) -> list[Candidate]:
        return [
            _candidate(statement, m.start(), m.end(), self.id, *_range_steps(m, patterns))
            for m in patterns.range.finditer(statement.text)
        ]


_KEY_VALUE = re.compile(r":\s+(?=\S)")
_TRAILING = " \t\r\n.;,"
_DIGIT = re.compile(r"\d")


def _number_chain(value: str, patterns: _Patterns) -> list[NormaliserStep] | None:
    """The chain for the first quantity in ``value``: a range, money, a number with a unit or
    a bare number, in that order when two start at the same place."""
    found: list[tuple[int, int, list[NormaliserStep] | None]] = []
    if m := patterns.range.search(value):
        found.append((m.start(), 0, _range_steps(m, patterns)))
    if m := patterns.money.search(value):
        found.append((m.start(), 1, patterns.steps(_step("parse_money", currency=_currency(m)))))
    if m := patterns.number_with_unit.search(value):
        unit = _canonical_unit(m.group("unit"))
        steps = patterns.steps(_step("parse_number"), _step("unit", **{"from": unit}))
        found.append((m.start(), 2, steps))
    if m := patterns.number.search(value):
        # A bare number with more after it ("1.5 TSI 150PS") could be any of them: skip it.
        bare = None if _DIGIT.search(value, m.end()) else patterns.steps(_step("parse_number"))
        found.append((m.start(), 3, bare))
    return _first(value, found)


def _date_chain(value: str, patterns: _Patterns) -> list[NormaliserStep] | None:
    """The chain for the first date in ``value``, longest form first; a bare year last."""
    found: list[tuple[int, int, list[NormaliserStep] | None]] = []
    for rank, (pattern, args) in enumerate(patterns.dates):
        if m := pattern.search(value):
            found.append((m.start(), rank, [_step("parse_date", **args)]))
    if m := _YEAR.search(value):
        found.append((m.start(), len(patterns.dates), [_step("parse_date", precision="year")]))
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
        return self._generate(statement, field, _EN_GB)

    def generate_in(
        self, statement: Statement, field: FieldSpec, locale: str | None
    ) -> list[Candidate]:
        return self._generate(statement, field, _for_locale(locale))

    def _generate(
        self, statement: Statement, field: FieldSpec, patterns: _Patterns
    ) -> list[Candidate]:
        span = _value_span(statement.text)
        if span is None:
            return []
        value = statement.text[span[0] : span[1]]
        if field.kind == "number":
            steps = _number_chain(value, patterns)
        elif field.kind == "date":
            steps = _date_chain(value, patterns)
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
    metallic" gives "Available" and "Moonstone Grey metallic", then that run's sub-runs:
    nothing marks where one value ends and the next begins in a name such as "Delmaro
    Kestrova SE" (a make, a model and a trim), so a run of up to ``MAX_PHRASE_WORDS``
    words also gives every contiguous part of it, and Jev picks. A longer run is cut
    into consecutive chunks of that many words, with no sub-runs.

    Where runs break is a guess about where values end, made in code because Jev can
    only pick spans, not propose them. Each sub-run is an option in select's Choice, and
    their number grows with the square of a run's length, so runs break wherever a value
    rarely goes on: at punctuation, at a full stop followed by a space ("St. Ives" breaks
    too), and at :data:`STOPWORDS` (English function words, on every page). A name that
    holds one ("Lord of the Rings") comes whole only from :class:`WholeStatement` or the
    LLM fallback.
    """

    id: str = "noun_phrase"
    scope: Scope = field(default_factory=lambda: Scope(kinds=frozenset({"str"})))

    def generate(self, statement: Statement) -> list[Candidate]:
        text = statement.text
        breaks = {m.start() for m in _BREAK.finditer(text)}
        out: list[Candidate] = []
        run: list[re.Match[str]] = []

        def flush() -> None:
            n = len(run)
            if n <= MAX_PHRASE_WORDS:
                phrases = [run[i:j] for i in range(n) for j in range(i + 1, n + 1)]
            else:
                phrases = [run[i : i + MAX_PHRASE_WORDS] for i in range(0, n, MAX_PHRASE_WORDS)]
            for words in phrases:
                if not all(w.group().replace(",", "").replace(".", "").isdigit() for w in words):
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

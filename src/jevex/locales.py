"""How a locale writes numbers, dates and units (spec: *Open questions*, locale handling).

Generators match a page's numbers and dates the way its locale writes them, and the
normaliser chains they emit read them back the same way. :func:`locale_conventions`
maps a BCP 47 tag (``de-DE``, ``en_US``, ``fr``) to :class:`LocaleConventions`; unknown
or missing tags get en-GB's, which is what jevex assumed before locales were handled.
These conventions aren't asked of Jev: "03/04/2024" or "45 mpg" is written the same
under either convention, so when the statement doesn't say, only the document's locale
tells which date or which gallon it means.

:func:`localise_steps` turns an en-GB normaliser chain into one for a locale by adding
the arguments that differ (``decimal``, ``order``, ``gallon``), never overriding ones the
chain already sets. Built-in generators use it with the document's locale; a declarative
generator uses it with its own scope's locale, so a generator scoped ``de-DE`` reads
"1.234,5" as 1234.5 without spelling that out in every step.

:func:`document_locale` finds a document's own locale (the caller's, the page's
``<html lang>``, its ``Content-Language``), so one pipeline reads each page by its own
conventions and runs only the locale-scoped generators meant for it. A document that
doesn't say takes the extractor's ``locale``, if it has one (``Extractor(locale=)``).
Tags are kept canonical (:func:`canonical_locale`: ``de_de`` is ``de-DE``), so generator
scopes learned from differently written tags are the same scope.

Beyond the decimal mark, a page's language adds its own words: month names
(:data:`MONTH_NAMES`: German, French, Spanish, Italian, Dutch), amount multipliers
(:data:`MULTIPLIERS`: "1,5 Mio. €", "2 Mrd. EUR") and range words (:data:`RANGE_WORDS`:
"1,4 bis 2,0 l", "zwischen 4 und 5"). Decimal-comma pages also write round amounts as
"18.495,- €", and Swiss ones group thousands with an apostrophe ("1’250.50").

Out of scope, so not read:

- long-scale "Billion"/"Bio." (German) and "billion" (French), 10^12: on those pages no
  amount is proposed with them, rather than a wrong one;
- Spanish "mil" on its own ("15 mil €"): in English it's a million or a thousandth of an
  inch, and the normalisers read multiplier words whatever the page's language;
  "mil millones" is read;
- month names in languages other than those above, and Spanish ordinals ("1º de mayo");
- plain spaces as thousands separators ("18 495 €", see
  :data:`THOUSANDS_AFTER_DECIMAL_COMMA`) and Indian lakh grouping ("1,00,000").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Literal

from jevex.clean import html_text_of
from jevex.document import LOCALE_TAG
from jevex.statements import NormaliserStep

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jevex.document import Document

DecimalMark = Literal[".", ","]
DateOrder = Literal["dmy", "mdy"]
Gallon = Literal["uk", "us"]

THOUSANDS_AFTER_DECIMAL_COMMA = ".\u00a0\u202f\u2009"
"""Thousands separators when the decimal mark is a comma: a dot, a no-break space, a
narrow no-break space or a thin space. A plain space isn't one: "5 300" is as often two
numbers."""


@dataclass(frozen=True)
class LocaleConventions:
    """How one locale writes the things generators look for.

    ``decimal`` is the decimal mark (the thousands separator is the other one, or a
    no-break space after a decimal comma). ``date_order`` reads all-numeric dates such as
    "03/12/2024". ``gallon`` is the gallon mpg is measured in. ``currency_after``: amounts
    may put the symbol after the number ("18.495 €"). ``language`` picks the month names,
    multipliers and range words matched on top of English ones. ``apostrophe_groups``:
    thousands may also be grouped with an apostrophe, as in Switzerland ("1’250.50").
    ``dollar`` and ``yen`` are the currencies "$" and "¥" stand for ("AUD" in Australia,
    "CNY" in China).
    """

    decimal: DecimalMark = "."
    date_order: DateOrder = "dmy"
    gallon: Gallon = "uk"
    currency_after: bool = False
    language: str = "en"
    apostrophe_groups: bool = False
    dollar: str = "USD"
    yen: str = "JPY"

    def currency(self, symbol: str) -> str:
        """The currency code a symbol (£, $, € or ¥) stands for in this locale."""
        return {"£": "GBP", "$": self.dollar, "€": "EUR", "¥": self.yen}[symbol]

    @property
    def thousands(self) -> str:
        """The characters that group thousands."""
        if self.decimal == ",":
            return THOUSANDS_AFTER_DECIMAL_COMMA
        return ",'’" if self.apostrophe_groups else ","


EN_GB = LocaleConventions()
"""The conventions used when the locale is unknown."""

_DECIMAL_COMMA_LANGUAGES = frozenset(
    {
        "bg",
        "ca",
        "cs",
        "da",
        "de",
        "el",
        "es",
        "et",
        "eu",
        "fi",
        "fr",
        "gl",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "lt",
        "lv",
        "nb",
        "nl",
        "nn",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sr",
        "sv",
        "tr",
        "uk",
        "vi",
    }
)
_US_REGIONS = frozenset({"US"})
_POINT_DECIMAL_REGIONS = {
    "de": frozenset({"CH", "LI"}),
    "it": frozenset({"CH"}),
    "es": frozenset({"MX", "GT", "HN", "NI", "SV", "PA", "DO", "PR", "PE"}),
}
"""Regions where a decimal-comma language writes a decimal point (de-CH: "1.25 kg")."""
_APOSTROPHE_REGIONS = frozenset({"CH", "LI"})
"""Regions that group a decimal-point number's thousands with an apostrophe ("1’250.50")."""
_DOLLARS = {"AU": "AUD", "CA": "CAD", "NZ": "NZD", "HK": "HKD", "SG": "SGD", "MX": "MXN"}
"""Regions whose own currency "$" stands for; elsewhere it's USD."""
_YENS = {"CN": "CNY"}
"""Regions whose own currency "¥" stands for; elsewhere it's JPY."""

MONTH_NAMES: dict[str, dict[str, int]] = {
    "en": {
        name: i
        for i, names in enumerate(
            [
                ("january", "jan"),
                ("february", "feb"),
                ("march", "mar"),
                ("april", "apr"),
                ("may",),
                ("june", "jun"),
                ("july", "jul"),
                ("august", "aug"),
                ("september", "sept", "sep"),
                ("october", "oct"),
                ("november", "nov"),
                ("december", "dec"),
            ],
            start=1,
        )
        for name in names
    },
    "de": {
        name: i
        for i, names in enumerate(
            [
                ("januar", "jänner", "jan", "jän"),
                ("februar", "feb"),
                ("märz", "mär", "mrz"),
                ("april", "apr"),
                ("mai",),
                ("juni", "jun"),
                ("juli", "jul"),
                ("august", "aug"),
                ("september", "sept", "sep"),
                ("oktober", "okt"),
                ("november", "nov"),
                ("dezember", "dez"),
            ],
            start=1,
        )
        for name in names
    },
    "fr": {
        name: i
        for i, names in enumerate(
            [
                ("janvier", "janv"),
                ("février", "fevrier", "févr", "fevr", "fév"),
                ("mars",),
                ("avril", "avr"),
                ("mai",),
                ("juin",),
                ("juillet", "juil"),
                ("août", "aout"),
                ("septembre", "sept"),
                ("octobre", "oct"),
                ("novembre", "nov"),
                ("décembre", "decembre", "déc", "dec"),
            ],
            start=1,
        )
        for name in names
    },
    "es": {
        name: i
        for i, names in enumerate(
            [
                ("enero", "ene"),
                ("febrero", "feb"),
                ("marzo", "mar"),
                ("abril", "abr"),
                ("mayo", "may"),
                ("junio", "jun"),
                ("julio", "jul"),
                ("agosto", "ago"),
                ("septiembre", "setiembre", "sept", "sep", "set"),
                ("octubre", "oct"),
                ("noviembre", "nov"),
                ("diciembre", "dic"),
            ],
            start=1,
        )
        for name in names
    },
    "it": {
        name: i
        for i, names in enumerate(
            [
                ("gennaio", "gen"),
                ("febbraio", "feb"),
                ("marzo", "mar"),
                ("aprile", "apr"),
                ("maggio", "mag"),
                ("giugno", "giu"),
                ("luglio", "lug"),
                ("agosto", "ago"),
                ("settembre", "set"),
                ("ottobre", "ott"),
                ("novembre", "nov"),
                ("dicembre", "dic"),
            ],
            start=1,
        )
        for name in names
    },
    "nl": {
        name: i
        for i, names in enumerate(
            [
                ("januari", "jan"),
                ("februari", "feb"),
                ("maart", "mrt"),
                ("april", "apr"),
                ("mei",),
                ("juni", "jun"),
                ("juli", "jul"),
                ("augustus", "aug"),
                ("september", "sept", "sep"),
                ("oktober", "okt"),
                ("november", "nov"),
                ("december", "dec"),
            ],
            start=1,
        )
        for name in names
    },
}
"""Lower-case month names and abbreviations by language. Dates are matched with English
names plus the page language's; ``parse_date`` reads every language's (a name means the
same month in every language that has it)."""

ALL_MONTH_NAMES: dict[str, int] = {
    name: month for names in MONTH_NAMES.values() for name, month in names.items()
}

MULTIPLIERS: dict[str, dict[str, int]] = {
    "de": {
        "tsd": 10**3,
        "tausend": 10**3,
        "mio": 10**6,
        "million": 10**6,
        "millionen": 10**6,
        "mrd": 10**9,
        "milliarde": 10**9,
        "milliarden": 10**9,
    },
    "fr": {
        "million": 10**6,
        "millions": 10**6,
        "md": 10**9,
        "mds": 10**9,
        "mrd": 10**9,
        "milliard": 10**9,
        "milliards": 10**9,
    },
    "es": {
        "millón": 10**6,
        "millon": 10**6,
        "millones": 10**6,
        "mil millones": 10**9,
    },
    "it": {
        "mila": 10**3,
        "milione": 10**6,
        "milioni": 10**6,
        "mln": 10**6,
        "miliardo": 10**9,
        "miliardi": 10**9,
        "mld": 10**9,
        "mrd": 10**9,
    },
    "nl": {
        "duizend": 10**3,
        "miljoen": 10**6,
        "mln": 10**6,
        "miljard": 10**9,
        "mld": 10**9,
        "mrd": 10**9,
    },
}
"""Lower-case amount multipliers by language, beyond English's (k, m, mn, bn, thousand,
million, billion). Money is matched with English ones plus the page language's, each
optionally followed by a dot ("1,5 Mio. €"); ``parse_money`` reads every language's, so a
word must mean the same in every language that has it and nothing else after a number in
English (which is why Spanish "mil" isn't here)."""

ALL_MULTIPLIERS: dict[str, int] = {
    word: factor for words in MULTIPLIERS.values() for word, factor in words.items()
}

LONG_SCALE_BILLION = frozenset({"de", "fr"})
"""Languages whose "billion" is 10^12: on their pages the English word isn't a multiplier."""


@dataclass(frozen=True)
class RangeWords:
    """How a language writes a range in words: "4 ``to`` 5", "``between`` 4 ``and_`` 5"."""

    to: tuple[str, ...]
    between: tuple[str, ...]
    and_: tuple[str, ...]


RANGE_WORDS: dict[str, RangeWords] = {
    "de": RangeWords(to=("bis",), between=("zwischen",), and_=("und",)),
    "fr": RangeWords(to=("à",), between=("entre",), and_=("et",)),
    "es": RangeWords(to=("a",), between=("entre",), and_=("y",)),
    "it": RangeWords(to=("a",), between=("tra", "fra"), and_=("e",)),
    "nl": RangeWords(to=("tot",), between=("tussen",), and_=("en",)),
}
"""Range words by language, matched on top of English "to" and "between … and"."""

ALL_RANGE_JOINS: frozenset[str] = frozenset(
    {"to", "and"} | {w for words in RANGE_WORDS.values() for w in (*words.to, *words.and_)}
)
"""Every language's words joining a range's two numbers, English "to" and "and" included
(``parse_range`` reads them whatever the page's language)."""


def _subtags(locale: str) -> tuple[str, str | None]:
    """The language and region of a tag: ``de-DE`` → ("de", "DE"), ``zh-Hant-TW`` →
    ("zh", "TW"), ``fr`` → ("fr", None)."""
    parts = locale.replace("_", "-").split("-")
    region = next(
        (p.upper() for p in parts[1:] if len(p) == 2 or (len(p) == 3 and p.isdigit())), None
    )
    return parts[0].lower(), region


def canonical_locale(tag: str) -> str:
    """``tag`` in BCP 47's conventional form, so one locale is always written one way:
    ``de_DE``, ``de-de`` and ``DE-de`` are all ``de-DE``, ``zh-hant-tw`` is ``zh-Hant-TW``.

    Subtags are joined by ``-``; the language is lower case, a script title case and a
    region upper case, and everything from the first singleton (``-u-``, ``-x-``) on is
    lower case. Doesn't check that ``tag`` is one (see
    :data:`~jevex.document.LOCALE_TAG`).
    """
    parts = tag.replace("_", "-").split("-")
    out = [parts[0].lower()]
    extension = False
    for part in parts[1:]:
        extension = extension or len(part) == 1
        if extension:
            out.append(part.lower())
        elif len(part) == 4 and part.isalpha():
            out.append(part.title())
        elif len(part) == 2 and part.isalpha():
            out.append(part.upper())
        else:
            out.append(part.lower())
    return "-".join(out)


def checked_locale(tag: str) -> str:
    """``tag`` made canonical (:func:`canonical_locale`), after checking it's a BCP 47
    language tag (:data:`~jevex.document.LOCALE_TAG`). Raises ``ValueError`` if it isn't.

    Every option that sets the extractor's ``locale`` checks its value with this
    (``Extractor(locale=)``, ``jevex eval``/``jevex serve --locale``, the Scrapy
    pipeline's ``JEVEX_LOCALE``), so all of them accept the same tags.
    """
    if not _TAG.fullmatch(tag):
        raise ValueError(f"locale must be a BCP 47 language tag such as 'en-GB', got {tag!r}")
    return canonical_locale(tag)


def locale_conventions(locale: str | None) -> LocaleConventions:
    """The conventions for a BCP 47 tag; en-GB's when it's ``None``, empty or unknown.

    Languages that write a decimal comma (German, French, Spanish, Dutch, ...) get it with
    day-first dates and the symbol allowed after an amount, except in regions that write
    a decimal point (``de-CH``, ``es-MX``). A US region (``en-US``, ``es-US``) means a
    decimal point, month-first dates and US gallons. A decimal point in Switzerland or
    Liechtenstein (``de-CH``, ``it-CH``) comes with apostrophe grouping ("1’250.50").
    "$" and "¥" are the region's own dollar or yen (``en-AU``: AUD, ``zh-CN``: CNY), else
    USD and JPY.
    """
    if not locale:
        return EN_GB
    language, region = _subtags(locale)
    dollar = _DOLLARS.get(region, "USD") if region else "USD"
    yen = _YENS.get(region, "JPY") if region else "JPY"
    if region in _US_REGIONS:
        return LocaleConventions(date_order="mdy", gallon="us", language=language)
    if language in _DECIMAL_COMMA_LANGUAGES and region not in _POINT_DECIMAL_REGIONS.get(
        language, frozenset()
    ):
        return LocaleConventions(
            decimal=",", currency_after=True, language=language, dollar=dollar, yen=yen
        )
    return LocaleConventions(
        language=language,
        apostrophe_groups=region in _APOSTROPHE_REGIONS,
        dollar=dollar,
        yen=yen,
    )


_DECIMAL_STEPS = frozenset({"parse_number", "parse_range", "parse_money"})


def localise_steps(
    steps: Iterable[NormaliserStep], conventions: LocaleConventions
) -> list[NormaliserStep]:
    """``steps`` with the arguments ``conventions`` need that the steps don't already set.

    Number, range and money steps get ``decimal``, ``parse_date`` gets ``order`` and
    ``unit`` gets ``gallon``, each only when the locale's differs from en-GB's (the
    normalisers' defaults). en-GB chains come back unchanged.
    """
    out: list[NormaliserStep] = []
    for step in steps:
        extra: dict[str, object] = {}
        if step.name in _DECIMAL_STEPS and conventions.decimal != EN_GB.decimal:
            extra["decimal"] = conventions.decimal
        elif step.name == "parse_date" and conventions.date_order != EN_GB.date_order:
            extra["order"] = conventions.date_order
        elif step.name == "unit" and conventions.gallon != EN_GB.gallon:
            extra["gallon"] = conventions.gallon
        missing = {k: v for k, v in extra.items() if k not in step.args}
        out.append(
            NormaliserStep(name=step.name, args={**step.args, **missing}) if missing else step
        )
    return out


_TAG = re.compile(LOCALE_TAG)

_HEAD_TAGS = frozenset(
    {"html", "head", "meta", "title", "link", "base", "style", "script", "noscript", "template"}
)
"""Elements that can come before the body; the first other one ends the head."""


def document_locale(document: Document) -> str | None:
    """The document's own locale, or ``None`` when nothing says.

    The caller's :attr:`~jevex.Document.locale` wins. Then, for HTML, the root element's
    ``lang`` (or ``xml:lang``), then a ``<meta http-equiv="Content-Language">``, then the
    HTTP :attr:`~jevex.Document.content_language`, as browsers read a page's language. A
    header or pragma naming several languages counts by its first. Values that aren't a
    language tag (``lang=""``, ``lang="English"``) are skipped, so the next source counts.
    The tag comes back canonical (:func:`canonical_locale`): ``lang="en-gb"`` is ``en-GB``.
    """
    if document.locale:
        return canonical_locale(document.locale)
    if document.is_html:
        found = html_language(document.content)
        if found:
            return canonical_locale(found)
    found = _first_tag(document.content_language)
    return canonical_locale(found) if found else None


def html_language(content: bytes) -> str | None:
    """The language an HTML page declares before its body: the root element's ``lang``
    or ``xml:lang``, else a ``Content-Language`` pragma. ``None`` when neither is a
    language tag."""
    reader = _HeadReader()
    try:
        reader.feed(html_text_of(content))
        reader.close()
    except _HeadEnded:
        pass
    return _first_tag(reader.lang) or _first_tag(reader.pragma)


def _first_tag(value: str | None) -> str | None:
    """The first language in a header-style list (``de-DE, en``), if it's a tag."""
    tag = (value or "").split(",", 1)[0].strip()
    return tag if _TAG.match(tag) else None


class _HeadEnded(Exception):
    """Stops :class:`_HeadReader` at the body: the page's language is declared by then."""


class _HeadReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang: str | None = None
        self.pragma: str | None = None
        self.noscript = 0
        """Open ``<noscript>`` elements: a tracking pixel in one doesn't end the head."""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "noscript":
            self.noscript += 1
        if self.noscript:
            return
        if tag not in _HEAD_TAGS:
            raise _HeadEnded
        values = dict(attrs)
        if tag == "html" and self.lang is None:
            self.lang = _first_tag(values.get("lang")) or _first_tag(values.get("xml:lang"))
        elif (
            tag == "meta"
            and self.pragma is None
            and (values.get("http-equiv") or "").strip().lower() == "content-language"
        ):
            self.pragma = values.get("content")

    def handle_endtag(self, tag: str) -> None:
        if tag == "noscript" and self.noscript:
            self.noscript -= 1

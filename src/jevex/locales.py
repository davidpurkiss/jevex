"""How a locale writes numbers, dates and units (spec: *Open questions*, locale handling).

Generators match a page's numbers and dates the way its locale writes them, and the
normaliser chains they emit read them back the same way. :func:`locale_conventions`
maps a BCP 47 tag (``de-DE``, ``en_US``, ``fr``) to :class:`LocaleConventions`; unknown
or missing tags get en-GB's, which is what jevex assumed before locales were handled.

:func:`localise_steps` turns an en-GB normaliser chain into one for a locale by adding
the arguments that differ (``decimal``, ``order``, ``gallon``), never overriding ones the
chain already sets. Built-in generators use it with the document's locale; a declarative
generator uses it with its own scope's locale, so a generator scoped ``de-DE`` reads
"1.234,5" as 1234.5 without spelling that out in every step.

:func:`document_locale` finds a document's own locale (the caller's, the page's
``<html lang>``, its ``Content-Language``), so one pipeline reads each page by its own
conventions and runs only the locale-scoped generators meant for it.
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
    may put the symbol after the number ("18.495 €"). ``language`` picks the month names
    dates are matched with, on top of English ones.
    """

    decimal: DecimalMark = "."
    date_order: DateOrder = "dmy"
    gallon: Gallon = "uk"
    currency_after: bool = False
    language: str = "en"

    @property
    def thousands(self) -> str:
        """The characters that group thousands."""
        return "," if self.decimal == "." else THOUSANDS_AFTER_DECIMAL_COMMA


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
}
"""Lower-case month names and abbreviations by language. Dates are matched with English
names plus the page language's; ``parse_date`` reads every language's."""

ALL_MONTH_NAMES: dict[str, int] = {
    name: month for names in MONTH_NAMES.values() for name, month in names.items()
}


def _subtags(locale: str) -> tuple[str, str | None]:
    """The language and region of a tag: ``de-DE`` → ("de", "DE"), ``zh-Hant-TW`` →
    ("zh", "TW"), ``fr`` → ("fr", None)."""
    parts = locale.replace("_", "-").split("-")
    region = next(
        (p.upper() for p in parts[1:] if len(p) == 2 or (len(p) == 3 and p.isdigit())), None
    )
    return parts[0].lower(), region


def locale_conventions(locale: str | None) -> LocaleConventions:
    """The conventions for a BCP 47 tag; en-GB's when it's ``None``, empty or unknown.

    Languages that write a decimal comma (German, French, Spanish, Dutch, ...) get it with
    day-first dates and the symbol allowed after an amount, except in regions that write
    a decimal point (``de-CH``, ``es-MX``). A US region (``en-US``, ``es-US``) means a
    decimal point, month-first dates and US gallons. Swiss apostrophe grouping ("1’250")
    isn't read.
    """
    if not locale:
        return EN_GB
    language, region = _subtags(locale)
    if region in _US_REGIONS:
        return LocaleConventions(date_order="mdy", gallon="us", language=language)
    if language in _DECIMAL_COMMA_LANGUAGES and region not in _POINT_DECIMAL_REGIONS.get(
        language, frozenset()
    ):
        return LocaleConventions(decimal=",", currency_after=True, language=language)
    return LocaleConventions(language=language)


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
    """
    if document.locale:
        return document.locale
    if document.is_html:
        found = html_language(document.content)
        if found:
            return found
    return _first_tag(document.content_language)


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

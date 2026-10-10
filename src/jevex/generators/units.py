"""The unit lexicon used by ``number_with_unit``.

Each canonical unit lists the spellings found in the wild. Matching is longest-first so
that "km/h" wins over "km" and "lb ft" over "lb". Spellings flagged case-sensitive only
match exactly ("PS", "L"); the rest ignore case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    """A canonical unit and the spellings that mean it."""

    canonical: str
    spellings: tuple[str, ...]
    case_sensitive: tuple[str, ...] = ()


UNITS: tuple[Unit, ...] = (
    Unit("s", ("seconds", "second", "secs", "sec", "s")),
    Unit("mph", ("mph", "miles per hour")),
    Unit("km/h", ("km/h", "kmh", "kph", "km per hour")),
    Unit("bhp", ("bhp",)),
    Unit("hp", ("hp",)),
    Unit("PS", ("PS",), case_sensitive=("PS",)),
    Unit("kW", ("kw",)),
    Unit("kWh", ("kwh",)),
    Unit("Nm", ("nm", "newton metres", "newton meters")),
    Unit("lb ft", ("lb ft", "lb-ft", "lbft", "lb.ft", "ft lb", "ft-lb")),
    Unit("mpg", ("mpg",)),
    Unit("l/100km", ("l/100km", "l/100 km", "litres/100km", "liters/100km")),
    Unit("g/km", ("g/km",)),
    Unit("kg", ("kg", "kgs", "kilograms")),
    Unit("mm", ("mm", "millimetres", "millimeters")),
    Unit("cm", ("cm",)),
    Unit("m", ("metres", "meters", "m")),
    Unit("cc", ("cc", "cm3", "cm³")),
    Unit("l", ("litres", "liters", "litre", "liter", "ltr", "L"), case_sensitive=("L",)),
    Unit("miles", ("miles", "mile", "mi")),
    Unit("km", ("km", "kilometres", "kilometers")),
)


def spellings() -> list[tuple[str, str, bool]]:
    """(spelling, canonical, case_sensitive), longest spelling first."""
    out: list[tuple[str, str, bool]] = []
    for unit in UNITS:
        for spelling in unit.spellings:
            out.append((spelling, unit.canonical, spelling in unit.case_sensitive))
    return sorted(out, key=lambda item: len(item[0]), reverse=True)


def alternation() -> str:
    """A regex alternation of every spelling, longest first; a space in one matches any
    whitespace or none, and only case-sensitive spellings match case."""
    parts: list[str] = []
    for spelling, _canonical, case_sensitive in spellings():
        escaped = re.escape(spelling).replace(r"\ ", r"\s?")
        parts.append(escaped if case_sensitive else f"(?i:{escaped})")
    return "|".join(parts)


_TO_CANONICAL = {
    (spelling if case_sensitive else spelling.lower()): canonical
    for spelling, canonical, case_sensitive in spellings()
}


def canonical(found: str) -> str:
    """The canonical unit of a spelling :func:`alternation` matched ("secs" → "s"); an
    unknown one comes back as it is."""
    compact = re.sub(r"\s+", " ", found)
    return _TO_CANONICAL.get(compact) or _TO_CANONICAL.get(compact.lower(), compact)


_MENTION = re.compile(rf"(?<![A-Za-z])(?:{alternation()})(?![A-Za-z0-9])")


def mentioned(text: str) -> list[str]:
    """The canonical units ``text`` spells out anywhere, not only after a number ("Power
    (kW)"), in order of first mention."""
    return list(dict.fromkeys(canonical(m.group()) for m in _MENTION.finditer(text)))

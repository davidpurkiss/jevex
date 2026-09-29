"""The unit lexicon used by ``number_with_unit``.

Each canonical unit lists the spellings found in the wild. Matching is longest-first so
that "km/h" wins over "km" and "lb ft" over "lb". Spellings flagged case-sensitive only
match exactly ("PS", "L"); the rest ignore case.
"""

from __future__ import annotations

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

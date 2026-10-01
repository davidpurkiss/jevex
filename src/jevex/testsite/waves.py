"""Wave schedules: which template families the test site releases, and in what order.

The learning demo (spec: *Synthetic test site*) replays the site in ``truth.json`` order
(``jevex eval --replay``, #47). :func:`~jevex.testsite.build` orders the pages by wave, so
each wave's families arrive together: LLM calls spike when a wave starts, then fall away
as generators are learned for it. A family the schedule leaves out isn't built at all.

On the command line a schedule is written ``table,listing;kv,grid;prose``: waves split by
``;``, families within a wave by ``,``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jevex.testsite.render import FAMILIES

if TYPE_CHECKING:
    from collections.abc import Sequence

Waves = tuple[tuple[str, ...], ...]
"""A schedule: the families of wave 1, then of wave 2, and so on."""

DEFAULT_WAVES: Waves = (
    ("table", "listing"),
    ("kv", "grid"),
    ("prose",),
    ("pdf",),
    ("scanned", "infographic"),
)
"""Every family, easiest first: HTML tables and one-car listings, then other HTML layouts,
free prose, spec-sheet PDFs, and last the images that need OCR."""


def check_waves(waves: Sequence[Sequence[str]]) -> Waves:
    """``waves`` as a :data:`Waves` tuple, or ``ValueError`` if it isn't a schedule.

    A schedule has at least one wave, no empty waves, only families from
    :data:`~jevex.testsite.render.FAMILIES`, and each of them at most once.
    """
    if isinstance(waves, str):
        raise TypeError("waves must be a sequence of waves; use parse_waves for a string")
    schedule = tuple(tuple(wave) for wave in waves)
    if not schedule:
        raise ValueError("a wave schedule needs at least one wave")
    seen: dict[str, int] = {}  # family -> its wave
    for number, wave in enumerate(schedule, start=1):
        if isinstance(waves[number - 1], str) or not wave:
            raise ValueError(f"wave {number} must be a non-empty list of families")
        for family in wave:
            if family not in FAMILIES:
                raise ValueError(
                    f"wave {number}: unknown family {family!r}; they're {', '.join(FAMILIES)}"
                )
            if (first := seen.get(family)) == number:
                raise ValueError(f"wave {number} lists {family!r} twice")
            if first is not None:
                raise ValueError(f"wave {number}: {family!r} is already in wave {first}")
            seen[family] = number
    return schedule


def parse_waves(text: str) -> Waves:
    """Read a schedule written ``table,listing;kv,grid;prose``. Spaces are ignored."""
    waves = [[f.strip() for f in wave.split(",") if f.strip()] for wave in text.split(";")]
    return check_waves(waves)


def format_waves(waves: Waves) -> str:
    """The inverse of :func:`parse_waves`."""
    return ";".join(",".join(wave) for wave in waves)


def wave_numbers(waves: Waves) -> dict[str, int]:
    """Each scheduled family's wave, counting from 1."""
    return {family: n for n, wave in enumerate(waves, start=1) for family in wave}

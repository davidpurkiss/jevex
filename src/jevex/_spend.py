"""A spend ledger shared by processes, for the process-wide spend caps.

``JEVEX_JEV_MAX_COST_USD`` and ``JEVEX_LLM_MAX_COST_USD`` cap one process. A live agent
run starts several (pytest, scripts), and the weekly caps span many runs, so when
``JEVEX_SPEND_LEDGER`` names a file, every process appends what it spends there and both
caps count the file's total instead of the process's own spend. ``scripts/agent-loop.sh``
gives each run its own ledger and sets the caps to what the run and the week have left.

The file is plain text, one ``<kind> <usd>`` line per charge (``kind`` is ``jev`` or
``llm``), so shell tools can add it up. Lines this short are appended in one ``write``,
which POSIX keeps whole even with several writers. Reading it before each call is a small
local file read, which is why these helpers are sync.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

LEDGER_ENV = "JEVEX_SPEND_LEDGER"

Kind = Literal["jev", "llm"]


def ledger_path() -> Path | None:
    """The shared ledger named by ``JEVEX_SPEND_LEDGER``, or ``None`` when unset."""
    raw = os.environ.get(LEDGER_ENV, "").strip()
    return Path(raw) if raw else None


def ledger_total(path: Path, kind: Kind, error: type[Exception]) -> float:
    """USD of ``kind`` recorded in ``path`` by every process (creating an empty ledger).

    Raises ``error`` when the ledger can't be written or has a line that isn't a finite,
    non-negative charge: a cap that can't keep its ledger mustn't let calls through, and
    a check before each call is the last point where refusing costs nothing.
    """
    try:
        with path.open("a+", encoding="utf-8") as f:
            f.seek(0)
            text = f.read()
    except OSError as exc:
        raise error(f"can't use the {LEDGER_ENV} file {path}: {exc}") from exc
    total = 0.0
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        amount = _charge(parts)
        if amount is None:
            raise error(f"{LEDGER_ENV} file {path} has a bad line {number}: {line!r}")
        if parts[0] == kind:
            total += amount
    return total


def _charge(parts: list[str]) -> float | None:
    """The USD of a ``<kind> <usd>`` line, or ``None`` unless it's finite and >= 0."""
    if len(parts) != 2:
        return None
    try:
        amount = float(parts[1])
    except ValueError:
        return None
    return amount if math.isfinite(amount) and amount >= 0 else None


def ledger_add(path: Path, kind: Kind, usd: float, error: type[Exception]) -> None:
    """Append one charge to the ledger."""
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{kind} {usd:.9f}\n")
    except OSError as exc:
        raise error(f"can't write to the {LEDGER_ENV} file {path}: {exc}") from exc

"""The spend ledger: what jevex spent, shared by every worker that keeps a run budget
(spec: *Budgets*, #238).

A run budget (:class:`~jevex.budgets.RunBudget`) is kept in a :class:`SpendLedger`. It's
its own protocol, not part of :class:`~jevex.store.Store`, so a ledger can live somewhere
else than the learned state: a Redis counter, a billing system, a per-process cap. The
built-in stores (:class:`~jevex.store.SQLiteStore`, ``PostgresStore``) are ledgers too,
and an extractor uses its store as its ledger unless it's given another
(``Extractor(ledger=)``). :class:`MemoryLedger` keeps one in the process.

A ledger that raises isn't trusted with the spend it was asked about: the LLM call that
needed it isn't made, and the failure is reported on the document's result (kind
``ledger``). Jev and generators carry on. Retrying, buffering or failing open in an
outage is the ledger's own business (a wrapper around it), not jevex's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from jevex.store.base import SpendEntry

if TYPE_CHECKING:
    from datetime import datetime

    from jevex.store.base import SpendKind


class LedgerError(Exception):
    """The spend ledger couldn't be read or written, so spend can't be confirmed. The
    ledger's own exception is the ``__cause__``."""


@runtime_checkable
class SpendLedger(Protocol):
    """Spend records shared by every worker keeping the same run budget.

    Entries are :class:`~jevex.store.SpendEntry` charges (never refunds). Sums should be
    exact at a nano-dollar (Jev charges a few per token), so a budget's spend cap is hit
    exactly.
    """

    async def record_spend(self, entry: SpendEntry) -> None:
        """Add ``entry``."""
        ...

    async def spend(
        self,
        *,
        since: datetime | None = None,
        kind: SpendKind | None = None,
        run_id: str | None = None,
    ) -> float:
        """Total USD of the entries matching every filter given (``since``: at or after)."""
        ...

    async def try_spend(
        self,
        entry: SpendEntry,
        *,
        max_count: int,
        since: datetime | None = None,
        kind: SpendKind | None = None,
    ) -> bool:
        """Record ``entry`` only if that keeps the entries at or after ``since`` whose kind
        is ``kind`` (every kind when ``None``), ``entry`` included, to at most
        ``max_count``: a rate limit, such as LLM calls over the last minute.

        The check and the write must be atomic, so workers sharing a limit can't overshoot
        it together. Returns whether the entry was recorded. Raises ``ValueError`` for a
        negative ``max_count`` or an ``entry`` of another kind than ``kind``.
        """
        ...

    async def spend_entries(
        self, *, since: datetime | None = None, kind: SpendKind | None = None
    ) -> list[SpendEntry]:
        """The entries at or after ``since`` (of ``kind``), oldest first."""
        ...


_NANO = 1_000_000_000


def _nano(usd: float) -> int:
    return round(usd * _NANO)


def _aware(at: datetime) -> datetime:
    if at.tzinfo is None:
        raise ValueError(f"naive datetime {at!r}: pass a timezone-aware datetime")
    return at


@dataclass
class MemoryLedger:
    """A :class:`SpendLedger` kept in this process: for one worker's run budget, or tests.

    Limits are shared by everything using this object (an extractor's documents and its
    learner), not by other processes. No method awaits, so :meth:`try_spend` is atomic on
    the event loop. Entries are kept until the ledger is
    dropped.
    """

    entries: list[SpendEntry] = field(default_factory=list[SpendEntry])

    def _matching(
        self, since: datetime | None, kind: SpendKind | None, run_id: str | None = None
    ) -> list[SpendEntry]:
        if since is not None:
            _aware(since)
        return [
            e
            for e in self.entries
            if (since is None or e.at >= since)
            and (kind is None or e.kind == kind)
            and (run_id is None or e.run_id == run_id)
        ]

    def _add(self, entry: SpendEntry) -> None:
        _aware(entry.at)
        self.entries.append(entry)

    async def record_spend(self, entry: SpendEntry) -> None:
        self._add(entry)

    async def spend(
        self,
        *,
        since: datetime | None = None,
        kind: SpendKind | None = None,
        run_id: str | None = None,
    ) -> float:
        return sum(_nano(e.amount_usd) for e in self._matching(since, kind, run_id)) / _NANO

    async def try_spend(
        self,
        entry: SpendEntry,
        *,
        max_count: int,
        since: datetime | None = None,
        kind: SpendKind | None = None,
    ) -> bool:
        if max_count < 0:
            raise ValueError(f"max_count must be non-negative, not {max_count}")
        if kind is not None and entry.kind != kind:
            raise ValueError(f"a {entry.kind} entry can't be checked against {kind} limits")
        if len(self._matching(since, kind)) + 1 > max_count:
            return False
        self._add(entry)
        return True

    async def spend_entries(
        self, *, since: datetime | None = None, kind: SpendKind | None = None
    ) -> list[SpendEntry]:
        return sorted(self._matching(since, kind), key=lambda e: e.at)

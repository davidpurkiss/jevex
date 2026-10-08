"""Runtime signals that span documents: drift, budget headroom and health.

A result reports one document (:mod:`jevex.errors`); these watch a running service.
``jevex serve`` exposes them on ``/metrics`` and ``/health``, and the stats UI shows
drift from the store (:mod:`jevex.stats`). ``docs/monitoring.md`` suggests alerts.

- **Drift** (:class:`DriftWindow`): over the last ``size`` documents, per
  ``"Schema.field"``, how often a record had no value (the "none" rate), how often the
  LLM fallback gave the value, and Jev's mean confidence. A site that changes its layout
  shows as a rising "none" or fallback rate long before anyone reads the records.
- **Budget headroom** (:func:`budget_headroom`): what's left of the run budget's LLM and
  Jev caps this period (``RunBudget``), and of the process caps
  (``JEVEX_*_MAX_COST_USD``).
- **Health** (:func:`store_error`, :attr:`GeneratorLearner.alive
  <jevex.learn.GeneratorLearner.alive>`): whether the store answers and the learner's
  worker is still running.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from jevex.jev import process_cap
from jevex.llm import process_llm_cap
from jevex.schema import SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pydantic import BaseModel

    from jevex.budgets import RunLedger
    from jevex.extractor import ExtractionResult
    from jevex.results import Extracted
    from jevex.store import Store

DRIFT_WINDOW = 200
"""Documents :class:`DriftWindow` keeps by default."""

STORE_TIMEOUT_S = 5.0
"""How long :func:`store_error` waits for the store."""


@dataclass(frozen=True)
class Observation:
    """One field of one record: whether it had a value, how, and how surely."""

    field: str
    found: bool
    method: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class FieldDrift:
    """One ``"Schema.field"`` over a window of documents."""

    field: str
    records: int
    """Records in the window with this field (each record of its schema)."""
    found: int
    """Those with a value."""
    llm: int
    """Values the LLM fallback gave (``method="llm"``)."""
    mean_confidence: float | None
    """Over the values that have a confidence; ``None`` if none has."""

    @property
    def none_rate(self) -> float:
        """The share of records with no value."""
        return 1 - self.found / self.records if self.records else 0.0

    @property
    def fallback_rate(self) -> float:
        """The share of values the LLM fallback gave."""
        return self.llm / self.found if self.found else 0.0


def observations(records: Iterable[Extracted[BaseModel]]) -> list[Observation]:
    """Every field of every record, children's too (``"Parent.nested_field.field"``),
    but not the nested-model fields themselves, whose values are the children's."""
    out: list[Observation] = []
    for r in records:
        nested = _nested_fields(r.model)
        for name, meta in r.meta.items():
            if name in nested:
                continue
            out.append(
                Observation(
                    field=f"{r.schema_name}.{name}",
                    found=meta.found,
                    method=meta.method if meta.found else None,
                    confidence=meta.confidence if meta.found else None,
                )
            )
        for children in r.children.values():
            out += observations(children)
    return out


def _nested_fields(model: type[BaseModel]) -> frozenset[str]:
    """The names of ``model``'s nested-model fields."""
    found = _NESTED.get(model)
    if found is None:
        found = _NESTED[model] = frozenset(
            f.name for f in SchemaSpec.from_model(model).child_fields
        )
    return found


_NESTED: dict[type[BaseModel], frozenset[str]] = {}


@dataclass
class DriftWindow:
    """The fields of the last ``size`` documents' records (see the module docstring).

    A record is observed once per field. A document whose schema was active but gave no
    record of it (it found nothing) counts as one record with none of the schema's
    fields, when the schema is among ``schemas``. A failed document adds nothing: its
    missing values are an error, not drift.
    """

    size: int = DRIFT_WINDOW
    schemas: Sequence[SchemaSpec] = ()
    _documents: deque[list[Observation]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"size must be at least 1, got {self.size}")
        self._documents = deque(maxlen=self.size)

    def add(self, result: ExtractionResult) -> None:
        """Add a document's records (dropping the oldest document once full)."""
        if result.status == "failed":
            return
        seen = observations(result.records)
        given = {r.schema_name for r in result.records}
        for spec in self.schemas:
            if spec.name in result.meta.active_schemas and spec.name not in given:
                nested = {f.name for f in spec.child_fields}
                seen += [
                    Observation(field=f"{spec.name}.{f.name}", found=False)
                    for f in spec.fields
                    if f.name not in nested
                ]
        self._documents.append(seen)

    @property
    def documents(self) -> int:
        """Documents in the window now."""
        return len(self._documents)

    def fields(self) -> list[FieldDrift]:
        """Each field seen in the window, sorted by name."""
        by_field: dict[str, list[Observation]] = {}
        for document in self._documents:
            for o in document:
                by_field.setdefault(o.field, []).append(o)
        return [_drift(name, by_field[name]) for name in sorted(by_field)]


def _drift(name: str, seen: Sequence[Observation]) -> FieldDrift:
    confident = [o.confidence for o in seen if o.confidence is not None]
    return FieldDrift(
        field=name,
        records=len(seen),
        found=sum(o.found for o in seen),
        llm=sum(o.method == "llm" for o in seen),
        mean_confidence=sum(confident) / len(confident) if confident else None,
    )


@dataclass(frozen=True)
class Headroom:
    """One spend cap and what's left of it."""

    scope: Literal["run", "process"]
    """``run``: the run budget (``RunBudget``), shared through the store's ledger.
    ``process``: ``JEVEX_JEV_MAX_COST_USD`` or ``JEVEX_LLM_MAX_COST_USD``."""
    kind: Literal["llm", "jev"]
    period: str
    """The run budget's period (``hour``, ``day``, ``week``, ``month``, ``run``);
    ``process`` for a process cap."""
    limit_usd: float
    spent_usd: float

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)


async def budget_headroom(ledger: RunLedger | None) -> list[Headroom]:
    """Every spend cap that is set, with what's spent of it: the run budget's (through
    ``ledger``, read from its spend ledger) and the process caps. Raises
    :class:`~jevex.store.LedgerError` if the ledger can't be read."""
    out: list[Headroom] = []
    run = ledger.budget if ledger is not None and ledger.active else None
    if ledger is not None and run is not None:
        caps: tuple[tuple[Literal["llm", "jev"], float | None], ...] = (
            ("llm", run.max_spend),
            ("jev", run.max_jev_spend),
        )
        for kind, limit in caps:
            if limit is not None:
                spent = await ledger.spent(kind)
                out.append(Headroom("run", kind, run.period, limit, spent))
    process: tuple[tuple[Literal["llm", "jev"], tuple[float, float] | None], ...] = (
        ("jev", process_cap()),
        ("llm", process_llm_cap()),
    )
    for kind, capped in process:
        if capped is not None:
            limit, spent = capped
            out.append(Headroom("process", kind, "process", limit, spent))
    return out


async def store_error(store: Store, *, wait_s: float = STORE_TIMEOUT_S) -> str | None:
    """Why ``store`` doesn't answer a small read (its disabled generator ids) within
    ``wait_s`` seconds, or ``None`` when it does (a health check: it never raises for the
    store's own failures)."""
    try:
        async with asyncio.timeout(wait_s):
            await store.disabled_generator_ids()
    except TimeoutError:
        return f"the store didn't answer within {wait_s:g}s"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None

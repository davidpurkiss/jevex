"""Budgets: limits on LLM and Jev use, per document and per run (spec: *Budgets*).

.. code-block:: python

    Extractor(
        schemas=[VehicleSpec],
        budgets=Budgets(
            per_document=DocBudget(max_llm_calls=5, max_spend=0.05, timeout_s=30),
            run=RunBudget(max_spend=5.00, period="day", llm_rpm=60),
        ),
        store="sqlite:///jevex.db",  # shares the run budget across workers
    )

When an LLM budget is hit, LLM use stops for the document and Jev and generators carry
on: affected fields keep the best Jev answer or stay ``None``. Every hit is reported in
``meta.budget_events``.

- **Per document** (:class:`DocBudget`): LLM calls, LLM spend and a deadline for LLM use
  (``timeout_s`` from the start of the document), plus an optional cap on Jev requests.
  Reaching the Jev cap stops the document; its results so far are returned.
- **Per run** (:class:`RunBudget`, kept by a :class:`RunLedger`): LLM spend per period and
  an LLM rate limit, shared by every worker on the same store through its spend ledger,
  plus an optional Jev spend cap per period. Without a store, the ledger is kept in
  memory for the extractor.

**For stage authors** the one entry point is :meth:`DocumentBudget.call_llm` (on
``ctx.budget``): it checks the budgets, makes the call and records its cost, and returns
``None`` when a budget says no. Code that runs outside a document (the learner, #38) uses
:meth:`RunLedger.call_llm`, which applies only the run budget. Every other method is used
by the extractor and changes the ledger.

**Limits and their bounds:**

- Call counts are reserved when a call starts, so concurrent calls can't pass a count cap
  together. Spend is only known after a call, so a spend cap can be overshot by the call
  that crosses it plus any calls already in flight.
- A call whose cost is unknown (the model isn't in the price table) counts as $0 against
  spend caps; the first such call under a spend cap is reported as an ``unpriced_llm``
  event so the gap is visible.
- ``llm_rpm`` refusals skip the call rather than wait for a slot; the event counts skips.
- ``max_jev_spend`` is checked when a document starts and each document's Jev spend is
  recorded when it ends, so documents already running when the cap is reached finish:
  the overshoot is at most (documents in flight across workers) × (Jev spend per
  document).
- ``period="run"`` counts the ledger entries of this run's ``run_id``. Workers share a
  run by passing the same ``run_id`` to their extractors.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevex.llm import LLMBudgetExceededError
from jevex.store import SpendEntry

if TYPE_CHECKING:
    from jevex.llm import LLM, LLMResponse
    from jevex.store import SpendKind, Store

Period = Literal["hour", "day", "week", "month", "run"]
Scope = Literal["document", "run", "process"]


class DocBudget(BaseModel):
    """Limits for one document. ``None`` means unlimited."""

    model_config = ConfigDict(frozen=True)

    max_llm_calls: int | None = Field(default=None, ge=0)
    max_spend: float | None = Field(default=None, ge=0, description="LLM spend, in USD")
    timeout_s: float | None = Field(
        default=None, gt=0, description="No LLM call starts later than this after the start"
    )
    max_jev_requests: int | None = Field(default=None, ge=0)


class RunBudget(BaseModel):
    """Limits shared by every document (and worker) using the same store."""

    model_config = ConfigDict(frozen=True)

    max_spend: float | None = Field(default=None, ge=0, description="LLM spend per period")
    period: Period = "day"
    llm_rpm: int | None = Field(default=None, ge=1, description="LLM calls per minute")
    max_jev_spend: float | None = Field(default=None, ge=0, description="Jev spend per period")


class Budgets(BaseModel):
    """Every limit an extractor applies: ``per_document`` for each document, and an
    optional ``run`` budget shared through the store's spend ledger. The defaults set no
    limits at all."""

    model_config = ConfigDict(frozen=True)

    per_document: DocBudget = Field(default_factory=DocBudget)
    run: RunBudget | None = None


class BudgetEvent(BaseModel):
    """A budget that was hit, as reported in ``meta.budget_events``.

    ``scope`` is ``"process"`` for the ``JEVEX_LLM_MAX_COST_USD`` backstop.
    """

    model_config = ConfigDict(frozen=True)

    scope: Scope
    limit: str
    message: str


def period_start(period: Period, now: datetime, run_started: datetime) -> datetime:
    """The start of the budget period containing ``now`` (UTC calendar periods; weeks
    start on Monday). ``"run"`` gives ``run_started``, though the ledger counts a run by
    its ``run_id`` rather than by time."""
    now = now.astimezone(UTC)
    if period == "run":
        return run_started
    if period == "hour":
        return now.replace(minute=0, second=0, microsecond=0)
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "day":
        return day
    if period == "week":
        return day - timedelta(days=day.weekday())
    return day.replace(day=1)


@dataclass(frozen=True)
class Refusal:
    """Why the ledger said no to an LLM call. ``stops`` is False for a rate limit, which
    only skips the call."""

    limit: str
    message: str
    stops: bool = True


@dataclass
class RunLedger:
    """The run budget and the store's spend ledger it is kept in.

    Owned by the extractor and shared by its documents. ``store`` is ``None`` when no
    run budget is set and no store was given; the ledger then allows everything.
    """

    budget: RunBudget | None = None
    store: Store | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def active(self) -> bool:
        return self.budget is not None and self.store is not None

    async def spent(self, kind: SpendKind) -> float:
        """This period's spend of ``kind`` (this run's, for ``period="run"``)."""
        if self.budget is None or self.store is None:
            return 0.0
        if self.budget.period == "run":
            return await self.store.spend(kind=kind, run_id=self.run_id)
        since = period_start(self.budget.period, datetime.now(UTC), self.started)
        return await self.store.spend(since=since, kind=kind)

    async def refuse_llm(self) -> Refusal | None:
        """Check the run's LLM limits before a call. Mutates: a call that passes the rate
        limit takes one of its slots."""
        run = self.budget
        if run is None or self.store is None:
            return None
        if run.max_spend is not None:
            spent = await self.spent("llm")
            if spent >= run.max_spend:
                return Refusal(
                    "max_spend",
                    f"${spent:.4f} of LLM spend this {run.period} (limit ${run.max_spend:.2f})",
                )
        if run.llm_rpm is not None:
            allowed = await self.store.try_spend(
                SpendEntry(amount_usd=0, kind="llm_call", run_id=self.run_id),
                max_count=run.llm_rpm,
                since=datetime.now(UTC) - timedelta(minutes=1),
                kind="llm_call",
            )
            if not allowed:
                return Refusal("llm_rpm", f"over {run.llm_rpm} LLM calls a minute", stops=False)
        return None

    async def refuse_document(self) -> Refusal | None:
        """Check the run's Jev spend cap before a document starts."""
        run = self.budget
        if run is None or run.max_jev_spend is None or self.store is None:
            return None
        spent = await self.spent("jev")
        if spent >= run.max_jev_spend:
            return Refusal(
                "max_jev_spend",
                f"${spent:.4f} of Jev spend this {run.period} (limit ${run.max_jev_spend:.2f})",
            )
        return None

    async def record(self, kind: SpendKind, cost: float | None) -> None:
        """Add spend to the ledger (only while a run budget is kept, and only real cost)."""
        if self.active and cost and self.store is not None:
            await self.store.record_spend(
                SpendEntry(amount_usd=cost, kind=kind, run_id=self.run_id)
            )

    async def call_llm[T: BaseModel](
        self, llm: LLM, prompt: str, schema: type[T]
    ) -> LLMResponse[T] | None:
        """An LLM call under the run budget only, for work outside a document (the learner).
        ``None`` when the run budget (or the process backstop) says no."""
        if await self.refuse_llm() is not None:
            return None
        try:
            response = await llm.structured(prompt, schema)
        except LLMBudgetExceededError:
            return None
        await self.record("llm", response.usage.cost)
        return response


@dataclass
class DocumentBudget:
    """The budgets as they stand for one document; the extractor creates one per document
    and puts it on ``ctx.budget``. Stages use only :meth:`call_llm`."""

    budgets: Budgets = field(default_factory=Budgets)
    ledger: RunLedger = field(default_factory=RunLedger)
    started: float = field(default_factory=time.monotonic)
    llm_calls: int = 0
    """Calls started (reserved), including ones in flight and ones that failed."""
    llm_spend: float = 0.0
    unpriced_calls: int = 0
    rpm_skips: int = 0
    llm_stopped: bool = False
    events: list[BudgetEvent] = field(default_factory=list[BudgetEvent])
    _pending: int = field(default=0, repr=False)
    """Reserved slots still waiting on the run ledger's checks."""

    def record_hit(self, scope: Scope, limit: str, message: str) -> None:
        """Report a budget hit in ``meta.budget_events`` (once per scope and limit; a
        later hit of the same limit replaces the message)."""
        self.events = [e for e in self.events if not (e.scope == scope and e.limit == limit)]
        self.events.append(BudgetEvent(scope=scope, limit=limit, message=message))

    def _stop_llm(self, scope: Scope, limit: str, message: str) -> None:
        self.llm_stopped = True
        self.record_hit(scope, limit, message)

    def _document_refusal(self) -> bool:
        """Whether a document limit refuses a call now. Only a limit that is really used up
        stops LLM use: a call slot held by a call still waiting on the run ledger may yet be
        given back, so while any are pending a full call count refuses without stopping."""
        doc = self.budgets.per_document
        if doc.max_llm_calls is not None and self.llm_calls >= doc.max_llm_calls:
            if self.llm_calls - self._pending < doc.max_llm_calls:
                return True
            self._stop_llm("document", "max_llm_calls", f"{self.llm_calls} LLM calls (the limit)")
        elif doc.max_spend is not None and self.llm_spend >= doc.max_spend:
            self._stop_llm(
                "document",
                "max_spend",
                f"${self.llm_spend:.4f} of LLM spend (limit ${doc.max_spend:.2f})",
            )
        elif doc.timeout_s is not None and time.monotonic() - self.started >= doc.timeout_s:
            self._stop_llm("document", "timeout_s", f"past the {doc.timeout_s:g} s LLM deadline")
        return self.llm_stopped

    async def _reserve(self) -> bool:
        """Take a call slot if every budget allows one. The document checks and the slot
        happen before any ``await``, so concurrent calls can't pass a count cap together."""
        if self.llm_stopped or self._document_refusal():
            return False
        self.llm_calls += 1
        self._pending += 1
        try:
            refusal = await self.ledger.refuse_llm()
        finally:
            self._pending -= 1
        if refusal is None:
            return True
        self.llm_calls -= 1  # give the slot back
        if refusal.stops:
            self._stop_llm("run", refusal.limit, refusal.message)
        else:
            self.rpm_skips += 1
            self.record_hit(
                "run", refusal.limit, f"{self.rpm_skips} calls skipped: {refusal.message}"
            )
        return False

    async def _settle(self, cost: float | None) -> None:
        self.llm_spend += cost or 0.0
        if cost is None:
            self.unpriced_calls += 1
            capped = self.budgets.per_document.max_spend is not None or (
                self.ledger.budget is not None and self.ledger.budget.max_spend is not None
            )
            if capped:
                self.record_hit(
                    "document",
                    "unpriced_llm",
                    f"{self.unpriced_calls} LLM calls with unknown cost counted as $0 "
                    "against spend caps (add the model's price)",
                )
        await self.ledger.record("llm", cost)

    async def call_llm[T: BaseModel](
        self, llm: LLM, prompt: str, schema: type[T]
    ) -> LLMResponse[T] | None:
        """Make an LLM call if the budgets allow it; ``None`` when they don't.

        A failed call still counts (the provider may bill it), though its cost is
        unknown. Hitting the process backstop (``JEVEX_LLM_MAX_COST_USD``) stops LLM use
        for the document like any other budget, rather than failing it.
        """
        if not await self._reserve():
            return None
        try:
            response = await llm.structured(prompt, schema)
        except LLMBudgetExceededError as exc:
            self.llm_calls -= 1  # refused before any request was made
            self._stop_llm("process", "JEVEX_LLM_MAX_COST_USD", str(exc))
            return None
        except BaseException:
            # Counted as a call (the provider may bill it), but its cost is unknown for a
            # different reason than a missing price, so it isn't reported as unpriced.
            raise
        await self._settle(response.usage.cost)
        return response

    async def start_document(self) -> bool:
        """Called by the extractor before the pipeline: whether the run's Jev spend cap
        allows the document. A refusal is recorded."""
        refusal = await self.ledger.refuse_document()
        if refusal is None:
            return True
        self.record_hit("run", refusal.limit, refusal.message)
        return False

    async def finish_document(self, jev_cost: float) -> None:
        """Called by the extractor after the pipeline (even a failed one): records the
        document's Jev spend in the ledger."""
        await self.ledger.record("jev", jev_cost)

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
- **Per run** (:class:`RunBudget`): LLM spend per period and an LLM rate limit, shared by
  every worker on the same store through its spend ledger, plus an optional Jev spend cap
  per period, checked when each document starts. Without a store, the run budget is kept
  in memory for this extractor.

LLM calls go through :meth:`DocumentBudget.call_llm`, which checks the budgets, makes the
call and records its cost. Spend is only known after a call, so a spend cap can be
overshot by the call that crosses it (and by calls already in flight).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevex.store import SpendEntry

if TYPE_CHECKING:
    from jevex.llm import LLM, LLMResponse
    from jevex.store import Store

Period = Literal["hour", "day", "week", "month", "run"]


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
    model_config = ConfigDict(frozen=True)

    per_document: DocBudget = Field(default_factory=DocBudget)
    run: RunBudget | None = None


class BudgetEvent(BaseModel):
    """A budget that was hit, as reported in ``meta.budget_events``."""

    model_config = ConfigDict(frozen=True)

    scope: Literal["document", "run"]
    limit: str
    message: str


def period_start(period: Period, now: datetime, run_started: datetime) -> datetime:
    """The start of the budget period containing ``now`` (UTC calendar periods)."""
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


@dataclass
class DocumentBudget:
    """The budgets as they stand for one document. Created by the extractor per document."""

    budgets: Budgets
    store: Store | None = None
    run_id: str | None = None
    run_started: datetime = field(default_factory=lambda: datetime.now(UTC))
    started: float = field(default_factory=time.monotonic)
    llm_calls: int = 0
    llm_spend: float = 0.0
    llm_stopped: bool = False
    events: list[BudgetEvent] = field(default_factory=list[BudgetEvent])

    def hit(self, scope: Literal["document", "run"], limit: str, message: str) -> None:
        """Record a budget hit (once per scope and limit)."""
        if not any(e.scope == scope and e.limit == limit for e in self.events):
            self.events.append(BudgetEvent(scope=scope, limit=limit, message=message))

    def _stop_llm(self, scope: Literal["document", "run"], limit: str, message: str) -> bool:
        self.llm_stopped = True
        self.hit(scope, limit, message)
        return False

    async def allow_llm(self) -> bool:
        """Whether an LLM call may start now. A refusal is recorded as a budget event.

        A document-level refusal (or a spent run budget) stops LLM use for the rest of the
        document; a rate-limit refusal only skips this call.
        """
        if self.llm_stopped:
            return False
        doc = self.budgets.per_document
        if doc.max_llm_calls is not None and self.llm_calls >= doc.max_llm_calls:
            return self._stop_llm(
                "document", "max_llm_calls", f"{self.llm_calls} LLM calls (the limit)"
            )
        if doc.max_spend is not None and self.llm_spend >= doc.max_spend:
            return self._stop_llm(
                "document",
                "max_spend",
                f"${self.llm_spend:.4f} of LLM spend (limit ${doc.max_spend:.2f})",
            )
        if doc.timeout_s is not None and time.monotonic() - self.started >= doc.timeout_s:
            return self._stop_llm(
                "document", "timeout_s", f"past the {doc.timeout_s:g} s LLM deadline"
            )
        run = self.budgets.run
        if run is None or self.store is None:
            return True
        now = datetime.now(UTC)
        if run.max_spend is not None:
            since = period_start(run.period, now, self.run_started)
            spent = await self.store.spend(since=since, kind="llm")
            if spent >= run.max_spend:
                return self._stop_llm(
                    "run",
                    "max_spend",
                    f"${spent:.4f} of LLM spend this {run.period} (limit ${run.max_spend:.2f})",
                )
        if run.llm_rpm is not None:
            allowed = await self.store.try_spend(
                SpendEntry(amount_usd=0, kind="llm_call", run_id=self.run_id),
                max_count=run.llm_rpm,
                since=now - timedelta(minutes=1),
                kind="llm_call",
            )
            if not allowed:
                self.hit("run", "llm_rpm", f"over {run.llm_rpm} LLM calls a minute")
                return False
        return True

    async def charge_llm(self, cost: float | None) -> None:
        """Count one LLM call and its cost (``None`` when the model's price is unknown)."""
        self.llm_calls += 1
        self.llm_spend += cost or 0.0
        if self.store is not None and self.budgets.run is not None and cost:
            await self.store.record_spend(
                SpendEntry(amount_usd=cost, kind="llm", run_id=self.run_id)
            )

    async def call_llm[T: BaseModel](
        self, llm: LLM, prompt: str, schema: type[T]
    ) -> LLMResponse[T] | None:
        """Make an LLM call if the budgets allow it; ``None`` when they don't.

        The call is counted even when it fails (the provider may still bill it); its
        cost is known only when it succeeds.
        """
        if not await self.allow_llm():
            return None
        try:
            response = await llm.structured(prompt, schema)
        except BaseException:
            await self.charge_llm(None)
            raise
        await self.charge_llm(response.usage.cost)
        return response

    async def allow_document(self) -> bool:
        """Whether a document may start, given the run's Jev spend cap."""
        run = self.budgets.run
        if run is None or run.max_jev_spend is None or self.store is None:
            return True
        since = period_start(run.period, datetime.now(UTC), self.run_started)
        spent = await self.store.spend(since=since, kind="jev")
        if spent >= run.max_jev_spend:
            self.hit(
                "run",
                "max_jev_spend",
                f"${spent:.4f} of Jev spend this {run.period} (limit ${run.max_jev_spend:.2f})",
            )
            return False
        return True

    async def finish(self, jev_cost: float) -> None:
        """Record the document's Jev spend in the ledger (when a run budget is kept)."""
        if self.store is not None and self.budgets.run is not None and jev_cost > 0:
            await self.store.record_spend(
                SpendEntry(amount_usd=jev_cost, kind="jev", run_id=self.run_id)
            )

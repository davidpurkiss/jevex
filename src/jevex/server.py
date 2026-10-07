"""``jevex serve``: the extraction microservice (spec: *Integration*, ``server`` extra).

A FastAPI app for callers that aren't Python, such as the car finder's Go crawler:

- ``POST /extract`` takes an :class:`ExtractRequest` (a document, its bytes base64, and
  the names of the registered schemas to extract) and answers with the records, as
  ``jevex extract`` prints them (``"meta": true``: with per-field and document meta).
- ``GET /health`` says the service is up and which schemas it serves; it answers 503
  when the store doesn't answer or a learner's worker has died.
- ``GET /metrics`` is Prometheus text: documents, records, the resolution mix, Jev and
  LLM calls, retries, rate limits and spend, budget hits and headroom, errors by stage
  and part, extraction and per-stage time, learner outcomes and liveness, store errors
  and per-field drift (:mod:`jevex.monitoring`) since the process started.
- Opt-in (``stats=True``, ``jevex serve --stats``), because it shows URLs and spend: the
  stats UI over the service's store, with the routes ``jevex stats`` serves (``/stats/``,
  ``/stats/api/<view>``, ``/stats/api/chart/<view>.svg``).

Schemas are registered by module path (``module:Class``, as ``jevex extract`` takes them).
Each set of schemas a request asks for gets its own :class:`~jevex.extractor.Extractor`,
made on first use, so a document is only asked about the schemas it's for; they all share
one Jev client, one store (and so the run budget's ledger and the learned generators) and
the LLMs.

This module imports FastAPI, so nothing in core imports it: ``jevex serve`` loads it when
it runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from pydantic import BaseModel, ConfigDict, Field

from jevex import __version__
from jevex.budgets import RunLedger
from jevex.document import LOCALE_TAG, Document
from jevex.extractor import Extractor, document_stat
from jevex.jev import JevBudgetExceededError, JevClient
from jevex.learn import LearnStatus
from jevex.llm import LLMBudgetExceededError
from jevex.logs import get_logger
from jevex.monitoring import DRIFT_WINDOW, DriftWindow, budget_headroom, store_error
from jevex.schema import SchemaSpec
from jevex.stats import CHART_VIEWS, VIEWS, chart_svg, from_store, render_page, to_json
from jevex.stats.server import redact
from jevex.store import StoreError, open_store

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from jevex.budgets import Budgets
    from jevex.errors import PartError
    from jevex.extractor import ExtractionResult
    from jevex.learn import GeneratorLearner
    from jevex.llm import LLM
    from jevex.monitoring import Headroom
    from jevex.stats import Stats
    from jevex.store import Store

DEFAULT_PORT = 8080

log = get_logger(__name__)


class UnknownSchemaError(ValueError):
    """A request named a schema the service doesn't serve."""


class DocumentIn(BaseModel):
    """A document as ``POST /extract`` takes it. ``content`` is base64; without a
    ``content_type`` it's sniffed from the bytes (:meth:`Document.from_bytes`, which also
    drops parameters such as ``; charset=utf-8``). ``content_language`` is the page's
    ``Content-Language`` header and ``locale`` the caller's override (see
    :class:`~jevex.Document`)."""

    model_config = ConfigDict(frozen=True, val_json_bytes="base64")

    content: bytes
    content_type: str | None = None
    url: str | None = None
    fetched_at: datetime | None = None
    site: str | None = None
    content_language: str | None = None
    locale: str | None = Field(default=None, pattern=LOCALE_TAG)

    def to_document(self) -> Document:
        return Document.from_bytes(
            self.content,
            url=self.url,
            content_type=self.content_type,
            fetched_at=self.fetched_at,
            site=self.site,
            content_language=self.content_language,
            locale=self.locale,
        )


class ExtractRequest(BaseModel):
    """The body of ``POST /extract``: a document and the schema (or schemas) to extract
    from it, by name. ``meta`` adds per-field metadata and the document's meta."""

    model_config = ConfigDict(frozen=True)

    document: DocumentIn
    schemas: Annotated[str, Field(min_length=1)] | Annotated[list[str], Field(min_length=1)] = (
        Field(alias="schema")
    )
    meta: bool = False

    def schema_names(self) -> list[str]:
        """The schemas asked for, in the order given, without repeats."""
        names = [self.schemas] if isinstance(self.schemas, str) else self.schemas
        return list(dict.fromkeys(names))


type Outcome = Literal["ok", "partial", "stopped", "error"]
OUTCOMES: tuple[Outcome, ...] = ("ok", "partial", "stopped", "error")


@dataclass
class Metrics:
    """Counters since the process started, for ``/metrics``.

    A document's outcome is its result's status (:mod:`jevex.errors`): ``ok``,
    ``partial`` (a part failed and was skipped) or ``error`` (it failed), with ``stopped``
    for an otherwise ok one a stage or budget stopped. Spend, calls and retries are those
    of the documents with a result; one whose extraction raised (a process spend cap)
    counts only as an ``error``. ``errors`` counts the documents with each error, by
    stage, kind and part (``""`` for none). ``store_errors`` counts store failures:
    those in results and those the service met itself (health checks, headroom and stats
    reads). ``drift`` keeps the last documents' fields (:class:`~jevex.monitoring.DriftWindow`).
    """

    documents: Counter[Outcome] = field(default_factory=Counter[Outcome])
    in_progress: int = 0
    records: Counter[str] = field(default_factory=Counter[str])
    values: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])
    budget_events: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])
    jev_requests: int = 0
    jev_questions: int = 0
    jev_tokens: int = 0
    jev_cost: float = 0.0
    llm_calls: int = 0
    llm_cost: float = 0.0
    jev_retries: int = 0
    llm_retries: int = 0
    jev_rate_limited: int = 0
    llm_rate_limited: int = 0
    errors: Counter[tuple[str, str, str]] = field(default_factory=Counter[tuple[str, str, str]])
    store_errors: int = 0
    seconds: float = 0.0
    timed: int = 0
    stage_seconds: dict[str, float] = field(default_factory=dict[str, float])
    stage_runs: Counter[str] = field(default_factory=Counter[str])
    drift: DriftWindow = field(default_factory=DriftWindow)

    def add(self, result: ExtractionResult, seconds: float) -> None:
        """Count a document's result (see the class docstring for its outcome)."""
        stat = document_stat(result, doc_id="", run_id=None, seconds=seconds)
        meta = result.meta
        self.documents[_outcome(result)] += 1
        self.errors.update((e.stage, e.kind, e.part or "") for e in result.errors)
        self.store_errors += sum(e.kind == "store" for e in result.errors)
        self.jev_retries += meta.jev.retries
        self.llm_retries += meta.llm.retries
        self.jev_rate_limited += meta.jev.rate_limited
        self.llm_rate_limited += meta.llm.rate_limited
        self.records.update(r.schema_name for r in result.records)
        self.values.update((v.field.split(".")[0], v.method or "unknown") for v in stat.values)
        self.budget_events.update((e.scope, e.limit) for e in meta.budget_events)
        self.jev_requests += stat.jev_requests
        self.jev_questions += stat.jev_questions
        self.jev_tokens += stat.jev_tokens
        self.jev_cost += stat.jev_cost
        self.llm_calls += stat.llm_calls
        self.llm_cost += stat.llm_cost
        self.seconds += seconds
        self.timed += 1
        for stage, took in meta.timings.items():
            self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + took
            self.stage_runs[stage] += 1
        self.drift.add(result)

    def render(
        self,
        *,
        learners: Sequence[GeneratorLearner] = (),
        headroom: Sequence[Headroom] = (),
    ) -> str:
        """The Prometheus text exposition format (version 0.0.4). ``learners`` are the
        service's running learners (outcomes summed; alive only if every one is) and
        ``headroom`` its spend caps; their series are left out when there are none."""
        lines: list[str] = []

        def metric(
            name: str, kind: str, help_: str, samples: list[tuple[dict[str, str], float]]
        ) -> None:
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} {kind}")
            for labels, value in samples:
                lines.append(f"{name}{_labels(labels)} {_number(value)}")

        def summary(
            name: str, help_: str, samples: list[tuple[dict[str, str], float, int]]
        ) -> None:
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} summary")
            for labels, total, count in samples:
                lines.append(f"{name}_sum{_labels(labels)} {_number(total)}")
                lines.append(f"{name}_count{_labels(labels)} {count}")

        metric("jevex_info", "gauge", "The jevex version.", [({"version": __version__}, 1)])
        metric(
            "jevex_documents_total",
            "counter",
            "Documents extracted, by outcome (partial: a part failed and was skipped; "
            "stopped: a stage or budget stopped it; error: it failed).",
            [({"outcome": o}, self.documents[o]) for o in OUTCOMES],
        )
        metric(
            "jevex_documents_in_progress",
            "gauge",
            "Documents being extracted now.",
            [({}, self.in_progress)],
        )
        metric(
            "jevex_records_total",
            "counter",
            "Records extracted, by schema.",
            [({"schema": s}, n) for s, n in sorted(self.records.items())],
        )
        metric(
            "jevex_values_total",
            "counter",
            "Field values found, by schema and how they were resolved.",
            [({"schema": s, "method": m}, n) for (s, m), n in sorted(self.values.items())],
        )
        metric(
            "jevex_budget_events_total",
            "counter",
            "Budget limits hit, by scope and limit.",
            [({"scope": s, "limit": lim}, n) for (s, lim), n in sorted(self.budget_events.items())],
        )
        metric(
            "jevex_errors_total",
            "counter",
            "Documents with a failure, by stage, kind and part (jevex.errors).",
            [
                ({"stage": st, "kind": k, "part": p}, n)
                for (st, k, p), n in sorted(self.errors.items())
            ],
        )
        for name, help_, value in (
            ("jevex_jev_requests_total", "Jev requests.", self.jev_requests),
            ("jevex_jev_questions_total", "Jev questions.", self.jev_questions),
            ("jevex_jev_input_tokens_total", "Jev input tokens.", self.jev_tokens),
            ("jevex_jev_cost_usd_total", "Estimated Jev spend in USD.", self.jev_cost),
            ("jevex_llm_calls_total", "LLM calls (fallback and verification).", self.llm_calls),
            ("jevex_llm_cost_usd_total", "LLM spend in USD.", self.llm_cost),
            ("jevex_jev_retries_total", "Jev requests retried.", self.jev_retries),
            ("jevex_llm_retries_total", "LLM calls retried (as SDKs report).", self.llm_retries),
            (
                "jevex_jev_rate_limited_total",
                "Jev requests answered 429 (retried or not).",
                self.jev_rate_limited,
            ),
            (
                "jevex_llm_rate_limited_total",
                "LLM calls that failed on a 429 after the SDK's retries.",
                self.llm_rate_limited,
            ),
            (
                "jevex_store_errors_total",
                "Store failures, in documents and in the service's own reads.",
                self.store_errors,
            ),
        ):
            metric(name, "counter", help_, [({}, value)])
        summary(
            "jevex_extract_seconds",
            "Time to extract a finished document.",
            [({}, self.seconds, self.timed)],
        )
        summary(
            "jevex_stage_seconds",
            "Time spent in each pipeline stage.",
            [
                ({"stage": st}, self.stage_seconds[st], self.stage_runs[st])
                for st in sorted(self.stage_seconds)
            ],
        )
        if learners:
            counts: Counter[LearnStatus] = Counter()
            for learner in learners:
                counts.update(learner.outcome_counts())
            metric(
                "jevex_learner_outcomes_total",
                "counter",
                "Examples the learner finished, by outcome (jevex.learn.LearnStatus).",
                [({"status": st}, counts[st]) for st in get_args(LearnStatus)],
            )
            metric(
                "jevex_learner_alive",
                "gauge",
                "1 while every learner's worker can take examples; 0 once one has died "
                "(until its error is raised and a new worker starts).",
                [({}, int(all(learner.alive for learner in learners)))],
            )
            metric(
                "jevex_learner_worker_deaths_total",
                "counter",
                "Times a learner's worker died on an unexpected error.",
                [({}, sum(learner.deaths for learner in learners))],
            )
        if headroom:
            metric(
                "jevex_budget_remaining_usd",
                "gauge",
                "What's left of each spend cap (scope run: the run budget this period; "
                "process: JEVEX_*_MAX_COST_USD).",
                [
                    ({"scope": h.scope, "kind": h.kind, "period": h.period}, h.remaining_usd)
                    for h in headroom
                ],
            )
            metric(
                "jevex_budget_limit_usd",
                "gauge",
                "Each spend cap.",
                [
                    ({"scope": h.scope, "kind": h.kind, "period": h.period}, h.limit_usd)
                    for h in headroom
                ],
            )
        drift = self.drift.fields()
        metric(
            "jevex_drift_documents",
            "gauge",
            f"Documents in the drift window (the last {self.drift.size}).",
            [({}, self.drift.documents)],
        )
        for name, help_, values in (
            (
                "jevex_field_none_rate",
                "Share of the window's records with no value for the field.",
                [d.none_rate for d in drift],
            ),
            (
                "jevex_field_fallback_rate",
                "Share of the window's values for the field the LLM fallback gave.",
                [d.fallback_rate for d in drift],
            ),
            (
                "jevex_field_records",
                "Records in the window with the field.",
                [d.records for d in drift],
            ),
        ):
            metric(
                name,
                "gauge",
                help_,
                [({"field": d.field}, v) for d, v in zip(drift, values, strict=True)],
            )
        metric(
            "jevex_field_confidence_mean",
            "gauge",
            "Mean confidence of the window's values for the field (those that have one).",
            [
                ({"field": d.field}, d.mean_confidence)
                for d in drift
                if d.mean_confidence is not None
            ],
        )
        return "\n".join(lines) + "\n"


def _outcome(result: ExtractionResult) -> Outcome:
    if result.status == "failed":
        return "error"
    if result.status == "partial":
        return "partial"
    return "stopped" if result.meta.stopped else "ok"


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    escaped = (
        f'{k}="{v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")}"'
        for k, v in labels.items()
    )
    return "{" + ",".join(escaped) + "}"


def _number(value: float) -> str:
    return str(value) if isinstance(value, int) else repr(float(value))


class Service:
    """What ``jevex serve``'s app runs on: the registered schemas, an extractor per set of
    schemas asked for, the shared Jev client, store and LLMs, and the :class:`Metrics`.

    ``store`` (a :class:`~jevex.store.Store` or a URL) is opened by :meth:`start` and,
    if it was a URL, closed by :meth:`aclose`. Without one, the extractors share an
    in-memory store, so the run budget and learned generators are still shared, and no
    document stats are recorded (nothing could read them). ``jev`` defaults to a client
    from the ``TYPESAFE_*`` environment variables. The extractors don't close the LLMs
    they're given; ``close_llms`` has :meth:`aclose` close them (for adapters built just
    for the service). The other options are :class:`~jevex.extractor.Extractor`'s.

    ``stats`` mounts the stats UI over the store (it needs a ``store``);
    ``stats_budget_usd`` draws its budget line. ``drift_window`` is how many recent
    documents ``/metrics``' per-field drift covers (:class:`~jevex.monitoring.DriftWindow`).
    """

    def __init__(
        self,
        schemas: Sequence[type[BaseModel]],
        *,
        jev: JevClient | None = None,
        store: Store | str | Path | None = None,
        budgets: Budgets | None = None,
        threshold: float = 0.0,
        extraction_llm: LLM | None = None,
        generator_llm: LLM | None = None,
        close_llms: bool = False,
        stats: bool = False,
        stats_budget_usd: float | None = None,
        drift_window: int = DRIFT_WINDOW,
    ) -> None:
        if not schemas:
            raise ValueError("register at least one schema")
        self.models: dict[str, type[BaseModel]] = {}
        for model in schemas:
            name = SchemaSpec.from_model(model).name
            if name in self.models:
                raise ValueError(f"schema names must be unique: {name!r} is registered twice")
            self.models[name] = model
        if stats and store is None:
            raise ValueError("the stats UI reads the store: pass store=")
        self.stats = stats
        self.stats_budget_usd = stats_budget_usd
        self.threshold = threshold
        self.budgets = budgets
        self.extraction_llm = extraction_llm
        self.generator_llm = generator_llm
        self.close_llms = close_llms
        self.metrics = Metrics(drift=DriftWindow(drift_window))
        self.run_id = uuid.uuid4().hex[:12]
        self._jev = jev
        self._store_source = store if isinstance(store, str | Path) else None
        self._store = None if isinstance(store, str | Path) else store
        self._owns_store = False
        # Stats go to the caller's store only, not to an in-memory one nothing else reads.
        self._record_stats = store is not None
        self._extractors: dict[tuple[str, ...], Extractor] = {}

    @property
    def schema_names(self) -> list[str]:
        return list(self.models)

    @property
    def store(self) -> Store | None:
        """The store the extractors share, once :meth:`start` has opened it."""
        return self._store

    @property
    def store_label(self) -> str:
        """The store as the stats page shows it: its URL without a password."""
        if self._store_source is not None:
            return redact(str(self._store_source))
        return type(self._store).__name__

    async def start(self) -> None:
        """Open the store and the Jev client. Raises :class:`~jevex.store.StoreError` if
        the store can't be opened."""
        if self._store is None:
            self._store = await asyncio.to_thread(open_store, self._store_source or ":memory:")
            self._owns_store = True
        if self._jev is None:
            self._jev = JevClient.from_env()

    def extractor(self, names: Sequence[str]) -> Extractor:
        """The extractor for these schemas, made on first use. Raises
        :class:`UnknownSchemaError` for a name that isn't registered."""
        unknown = [n for n in names if n not in self.models]
        if unknown:
            raise UnknownSchemaError(
                f"unknown schema {', '.join(map(repr, unknown))}; this service has "
                f"{', '.join(self.models)}"
            )
        key = tuple(n for n in self.models if n in names)
        if key not in self._extractors:
            if self._store is None or self._jev is None:
                raise RuntimeError("the service isn't started")
            self._extractors[key] = Extractor(
                [self.models[n] for n in key],
                jev=self._jev,
                store=self._store,
                run_id=self.run_id,
                budgets=self.budgets,
                threshold=self.threshold,
                extraction_llm=self.extraction_llm,
                generator_llm=self.generator_llm,
                record_stats=self._record_stats,
            )
        return self._extractors[key]

    async def extract(self, document: Document, names: Sequence[str]) -> ExtractionResult:
        """Extract ``names`` from ``document``, counting it in :attr:`metrics`."""
        extractor = self.extractor(names)
        self.metrics.in_progress += 1
        started = time.perf_counter()
        try:
            result = await extractor.extract(document)
        except Exception:  # a process spend cap (not a cancellation: a client gone)
            self.metrics.documents["error"] += 1
            raise
        finally:
            self.metrics.in_progress -= 1
        self.metrics.add(result, time.perf_counter() - started)
        return result

    async def read_stats(self) -> Stats:
        if self._store is None:
            raise RuntimeError("the service isn't started")
        try:
            return await from_store(
                self._store, source=self.store_label, budget_usd=self.stats_budget_usd
            )
        except StoreError:
            self.metrics.store_errors += 1
            raise

    @property
    def learners(self) -> list[GeneratorLearner]:
        """The extractors' running learners (each is made by its first document)."""
        return [e.running_learner for e in self._extractors.values() if e.running_learner]

    async def headroom(self) -> list[Headroom]:
        """What's left of each spend cap (:func:`~jevex.monitoring.budget_headroom`):
        none when the store can't be read, which is counted in ``store_errors``."""
        run = self.budgets.run if self.budgets else None
        ledger = RunLedger(run, self._store, run_id=self.run_id) if run else None
        try:
            return await budget_headroom(ledger)
        except Exception as exc:
            self.metrics.store_errors += 1
            log.warning("can't read the spend ledger for /metrics: %s", exc, exc_info=exc)
            return []

    async def render_metrics(self) -> str:
        """``/metrics``: :attr:`metrics` with the learners and the spend caps."""
        return self.metrics.render(learners=self.learners, headroom=await self.headroom())

    async def health(self) -> dict[str, Any]:
        """``/health``'s answer: ``status`` is ``ok``, or ``unhealthy`` when the store
        doesn't answer (:func:`~jevex.monitoring.store_error`) or a learner's worker has
        died; ``checks`` says which."""
        checks: dict[str, str] = {}
        if self._store is not None:
            problem = await store_error(self._store)
            if problem is not None:
                self.metrics.store_errors += 1
                log.warning("health check: %s", problem)
            checks["store"] = problem or "ok"
        learners = self.learners
        if learners:
            dead = sum(not learner.alive for learner in learners)
            checks["learner"] = f"{dead} worker(s) died" if dead else "ok"
        healthy = all(v == "ok" for v in checks.values())
        return {
            "status": "ok" if healthy else "unhealthy",
            "version": __version__,
            "schemas": self.schema_names,
            "checks": checks,
        }

    async def aclose(self) -> None:
        """Close the extractors (stopping their learners), the LLMs with ``close_llms``,
        then the store if the service opened it."""
        extractors, self._extractors = list(self._extractors.values()), {}
        try:
            for extractor in extractors:
                await extractor.aclose()  # closes the shared Jev client too
            if self._jev is not None and not extractors:
                close = getattr(self._jev.backend, "aclose", None)
                if close is not None:
                    await close()
            if self.close_llms:
                llms = {id(m): m for m in (self.extraction_llm, self.generator_llm) if m}
                for llm in llms.values():
                    close = getattr(llm, "aclose", None)
                    if close is not None:
                        await close()
        finally:
            store, owned = self._store, self._owns_store
            if owned:
                self._store, self._owns_store = None, False
            if owned and store is not None:
                await store.aclose()


def _failure(error: PartError) -> HTTPException:
    """The HTTP error for a failed result's core failure."""
    if error.kind == "document":
        return HTTPException(422, detail=f"can't read the document: {error.message}")
    if error.kind == "jev":
        return HTTPException(502, detail=f"Jev: {error.message}")
    return HTTPException(500, detail=f"extraction failed: {error.describe()}")


def create_app(service: Service) -> FastAPI:
    """The FastAPI app for ``service`` (also on ``app.state.service``). Its lifespan
    starts the service and closes it.

    ``POST /extract`` answers 422 for a body that doesn't validate or names an unknown
    schema, 503 when a process spend cap (``JEVEX_*_MAX_COST_USD``) is reached, and for a
    failed result (:mod:`jevex.errors`) 422 if the document can't be read (a broken PDF or
    image), 502 if Jev failed and 500 otherwise. A ``partial`` result is a 200 whose
    ``status`` and ``errors`` say what was skipped.

    ``GET /health`` answers 503 when the store doesn't answer or a learner's worker has
    died (:meth:`Service.health`), so an orchestrator restarts the service.
    """

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.aclose()

    app = FastAPI(
        title="jevex",
        version=__version__,
        description="Extract typed records from web pages and PDFs using Jev.",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.post("/extract")
    async def extract(request: ExtractRequest) -> dict[str, Any]:
        """Extract the named schemas' records from the document."""
        try:
            result = await service.extract(request.document.to_document(), request.schema_names())
        except UnknownSchemaError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except (JevBudgetExceededError, LLMBudgetExceededError) as exc:
            raise HTTPException(503, detail=str(exc)) from exc
        fatal = next((e for e in result.errors if e.fatal), None)
        if fatal is not None:
            raise _failure(fatal)
        return result.to_dict() if request.meta else result.to_plain_dict()

    @app.get("/health")
    async def health() -> Response:
        """The service's version and schemas, and whether its store answers and its
        learners are alive: 200 when healthy, 503 when not."""
        answer = await service.health()
        return JSONResponse(answer, status_code=200 if answer["status"] == "ok" else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> Response:
        """Prometheus metrics since the process started."""
        return PlainTextResponse(
            await service.render_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    if service.stats:
        _mount_stats(app, service)
    return app


def _mount_stats(app: FastAPI, service: Service) -> None:
    """The stats UI's routes, as :func:`jevex.stats.stats_server` serves them."""

    async def load() -> Stats:
        try:
            return await service.read_stats()
        except StoreError as exc:
            raise HTTPException(500, detail=f"can't read stats: {exc}") from exc

    @app.get("/stats", include_in_schema=False)
    async def stats_root() -> Response:
        return RedirectResponse("/stats/", status_code=302)

    @app.get("/stats/", response_class=HTMLResponse)
    async def stats_page() -> Response:
        """The stats page (reloads itself)."""
        return HTMLResponse(render_page(await load(), live=True), headers=_NO_STORE)

    @app.get("/stats/api/chart/{name}")
    async def stats_chart(name: str, x: str | None = None, animate: str = "0") -> Response:
        """A view as a standalone SVG chart."""
        view = name.removesuffix(".svg")
        if view not in CHART_VIEWS or not name.endswith(".svg"):
            raise HTTPException(404, detail=f"no chart {view!r}")
        stats = await load()
        axis = x or stats.default_axis()
        if axis not in ("docs", "time"):
            raise HTTPException(400, detail="x must be docs or time")
        try:
            svg = chart_svg(
                stats,
                view,
                "time" if axis == "time" else "docs",
                standalone=True,
                animate=animate == "1",
            )
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        return Response(svg, media_type="image/svg+xml; charset=utf-8", headers=_NO_STORE)

    @app.get("/stats/api/{view}")
    async def stats_view(view: str) -> Response:
        """A view's data as JSON."""
        if view not in VIEWS:
            raise HTTPException(404, detail=f"no view {view!r}")
        return JSONResponse(to_json(await load(), view), headers=_NO_STORE)


_NO_STORE = {"Cache-Control": "no-store"}

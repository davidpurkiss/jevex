"""The consumer entry point: ``Extractor(schemas=[...]).extract(document)``."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, get_args, overload

from pydantic import BaseModel, ConfigDict, Field

from jevex.budgets import BudgetEvent, Budgets, DocumentBudget, RunLedger
from jevex.categorise import CategoriseStage
from jevex.clean import CleanStage
from jevex.component_gate import ComponentGateStage
from jevex.fallback import FALLBACK_THRESHOLD, FallbackStage
from jevex.gate import DocumentGateStage
from jevex.generators import GeneratorRegistry
from jevex.housekeeping import PRUNE_AFTER, DuplicateGenerator, Housekeeper
from jevex.images import ImageStage
from jevex.interfaces import GateDecision
from jevex.jev import JevClient, JevRequestCapError
from jevex.keypaths import StructuredStage
from jevex.layout import LayoutStage
from jevex.learn import (
    LEARN_THRESHOLD,
    ExampleLogger,
    GeneratorLearner,
    LearnedGenerators,
    LearnMode,
    LearnStage,
    PackDiff,
    compile_pack,
)
from jevex.normalise import BUILTIN_NORMALISERS, NormaliseStage
from jevex.pipeline import Context, Pipeline
from jevex.resolve import EntityStage
from jevex.results import Extracted, FieldMeta, build_extracted, inherit, select_records
from jevex.schema import ReservedFieldNameError, SchemaSpec, UnsupportedFieldError
from jevex.select import CandidateStage, JevCandidateSelector, SelectStage
from jevex.split import StatementStage
from jevex.store import Store, open_store

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from jevex.document import Document
    from jevex.generators import GeneratorSpec
    from jevex.llm import LLM
    from jevex.pipeline import SchemaRun, Stage

# The spec's stage order (see jevex.interfaces). Default stages must use these names; they
# are sorted into this order, so issues can add their stage without coordinating position.
STAGE_ORDER: tuple[str, ...] = (
    "fetch",
    "clean",
    "document_gate",
    "structured",
    "layout",
    "images",
    "component_gate",
    "statements",
    "entities",
    "categorise",
    "candidates",
    "select",
    "normalise",
    "fallback",
    "learn",
)

# Default stages, one per line to keep merges clean. Order here doesn't matter: see
# STAGE_ORDER.
DEFAULT_STAGES: tuple[Stage, ...] = (
    CleanStage(),
    DocumentGateStage(),
    StructuredStage(),
    LayoutStage(),
    ImageStage(),
    ComponentGateStage(),
    StatementStage(),
    EntityStage(),
    CategoriseStage(),
    CandidateStage(),
    SelectStage(),
    NormaliseStage(),
    FallbackStage(),
    LearnStage(),
)


def default_pipeline() -> Pipeline:
    """The pipeline ``Extractor`` builds when none is given, in :data:`STAGE_ORDER`.

    Stateful stages (the structured stage's in-memory mappings) are new in each call.
    """
    unknown = [s.name for s in DEFAULT_STAGES if s.name not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"default stages {unknown} are not in STAGE_ORDER {STAGE_ORDER}")
    # The structured stage's mapper remembers mappings; each pipeline gets its own.
    stages = [
        StructuredStage(mode=s.mode) if isinstance(s, StructuredStage) else s
        for s in DEFAULT_STAGES
    ]
    return Pipeline(sorted(stages, key=lambda s: STAGE_ORDER.index(s.name)))


class JevUsageSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: int
    questions: int
    input_tokens: int
    cost: float
    seconds: float
    models: list[str]


class LLMUsageSummary(BaseModel):
    """The document's LLM calls through ``ctx.budget.call_llm``. ``unpriced_calls`` had no
    known price, so ``cost`` leaves them out."""

    model_config = ConfigDict(frozen=True)

    calls: int = 0
    cost: float = 0.0
    unpriced_calls: int = 0


class EventInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    kind: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict[str, Any])


class DocumentMeta(BaseModel):
    """Document-level metadata: gates, Jev usage, timings and events."""

    model_config = ConfigDict(frozen=True)

    url: str | None
    content_type: str
    gates: dict[str, GateDecision]
    active_schemas: list[str]
    jev: JevUsageSummary
    llm: LLMUsageSummary = Field(default_factory=LLMUsageSummary)
    timings: dict[str, float]
    events: list[EventInfo]
    stopped: bool
    budget_events: list[BudgetEvent] = Field(default_factory=list[BudgetEvent])
    generator_snapshot: int | None = None
    """The version of the learned-generator snapshot the document ran with (``None``
    without a store or learner); see :class:`~jevex.learn.GeneratorSnapshot`."""


@dataclass(frozen=True)
class ExtractionResult:
    """What ``extract`` returns: typed records plus document-level metadata.

    ``records`` holds one :class:`~jevex.results.Extracted` per entity per schema, in schema
    registration order. Use ``for_schema(Model)`` for typed access, or ``one()`` for
    single-entity documents. Child entities (``ParentChild``) aren't records of their own:
    they're in their parent's nested field and ``children``.
    """

    records: list[Extracted[BaseModel]]
    meta: DocumentMeta

    @overload
    def for_schema[T: BaseModel](self, model: type[T]) -> list[Extracted[T]]: ...
    @overload
    def for_schema(self, model: None = None) -> list[Extracted[BaseModel]]: ...
    def for_schema(self, model: type[BaseModel] | None = None) -> list[Extracted[Any]]:
        return select_records(self.records, model)

    @overload
    def one[T: BaseModel](self, model: type[T]) -> Extracted[T]: ...
    @overload
    def one(self, model: None = None) -> Extracted[BaseModel]: ...
    def one(self, model: type[BaseModel] | None = None) -> Extracted[Any]:
        """The only record (of ``model``, if given). Raises ``LookupError`` otherwise."""
        found = select_records(self.records, model)
        if len(found) != 1:
            which = f" for {model.__name__}" if model else ""
            raise LookupError(f"expected exactly one record{which}, found {len(found)}")
        return found[0]

    @property
    def values(self) -> dict[str, dict[str, dict[str, Any]]]:
        """The records' values (as coerced into them), by schema, entity and field.

        Only fields that made it into a record: not filtered, not rejected by the type.
        """
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for r in self.records:
            found = {name: getattr(r.record, name) for name in r.record.model_fields_set}
            if found:
                out.setdefault(r.schema_name, {})[r.entity] = found
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "meta": self.meta.model_dump(mode="json"),
        }

    @classmethod
    def from_context(
        cls,
        ctx: Context,
        *,
        threshold: float = 0.0,
        thresholds: Mapping[str, float] | None = None,
    ) -> ExtractionResult:
        usage = ctx.jev.usage
        records: list[Extracted[BaseModel]] = []
        for run in ctx.schemas.values():
            if run.parent is None:
                records.extend(_records(ctx, run, threshold, thresholds))
        return cls(
            records=records,
            meta=DocumentMeta(
                url=ctx.document.url,
                content_type=ctx.document.content_type,
                gates={name: run.gate for name, run in ctx.schemas.items() if run.gate},
                active_schemas=[
                    name for name, run in ctx.schemas.items() if run.active and run.parent is None
                ],
                jev=JevUsageSummary(
                    requests=usage.requests,
                    questions=usage.questions,
                    input_tokens=usage.input_tokens,
                    cost=usage.cost,
                    seconds=usage.seconds,
                    models=sorted(usage.models),
                ),
                llm=LLMUsageSummary(
                    calls=ctx.budget.llm_calls,
                    cost=ctx.budget.llm_spend,
                    unpriced_calls=ctx.budget.unpriced_calls,
                )
                if ctx.budget
                else LLMUsageSummary(),
                timings=dict(ctx.timings),
                events=[
                    EventInfo(stage=e.stage, kind=e.kind, message=e.message, data=e.data)
                    for e in ctx.events
                ],
                stopped=ctx.stopped,
                budget_events=list(ctx.budget.events) if ctx.budget else [],
                generator_snapshot=ctx.generators.version if ctx.generators else None,
            ),
        )


def _threshold_keys(specs: Sequence[SchemaSpec]) -> set[str]:
    return {f.name for s in specs for f in s.fields} | {
        f"{s.name}.{f.name}" for s in specs for f in s.fields
    }


def _child_specs(specs: Sequence[SchemaSpec]) -> list[SchemaSpec]:
    """Specs of the nested models jevex can extract (a nested model it can't is only an
    error if ``ParentChild`` is asked to fill it)."""
    out: list[SchemaSpec] = []
    for spec in specs:
        for f in spec.child_fields:
            try:
                out.append(spec.child(f.name))
            except (UnsupportedFieldError, ReservedFieldNameError):
                continue
    return out


def _records(
    ctx: Context, run: SchemaRun, threshold: float, thresholds: Mapping[str, float] | None
) -> list[Extracted[BaseModel]]:
    """A record per entity that found anything or has children, in scope order."""
    child_runs = [r for r in ctx.schemas.values() if r.parent == run.name]
    out: list[Extracted[BaseModel]] = []
    for entity, metas in _entity_metas(run).items():
        children = {
            r.parent_field: _children(r, entity, threshold, thresholds)
            for r in child_runs
            if r.parent_field is not None
        }
        if _has_values(metas) or any(children.values()):
            out.append(
                build_extracted(
                    run.spec,
                    entity,
                    metas,
                    threshold=threshold,
                    thresholds=thresholds,
                    children=children,
                )
            )
    return out


def _children(
    run: SchemaRun, parent: str, threshold: float, thresholds: Mapping[str, float] | None
) -> list[Extracted[BaseModel]]:
    """``parent``'s child records from a nested model's run: each child scope's own
    values, plus whatever the run found in the parent's scope (inherited, ``shared``)."""
    metas = _entity_metas(run)
    inherited = metas.get(parent, {})
    out: list[Extracted[BaseModel]] = []
    for scope in run.scopes:
        if scope.parent != parent:
            continue
        found = inherit(metas.get(scope.label, {}), inherited)
        if _has_values(found):
            out.append(
                build_extracted(
                    run.spec, scope.label, found, threshold=threshold, thresholds=thresholds
                )
            )
    return out


def _entity_metas(run: SchemaRun) -> dict[str, dict[str, FieldMeta]]:
    """Field metadata per entity, in scope order; bare ``values`` fill any gaps."""
    order = [s.label for s in run.scopes]
    labels = [*order, *(k for k in [*run.fields, *run.values] if k not in order)]
    out: dict[str, dict[str, FieldMeta]] = {}
    for label in dict.fromkeys(labels):
        metas = dict(run.fields.get(label, {}))
        for name, value in run.values.get(label, {}).items():
            metas.setdefault(name, FieldMeta(value=value))
        out[label] = metas
    return out


def _stage[S](pipeline: Pipeline, name: str, kind: type[S]) -> S | None:
    found = next((s for s in pipeline if s.name == name), None)
    return found if isinstance(found, kind) else None


def _has_values(metas: Mapping[str, FieldMeta]) -> bool:
    return any(m.found or m.alternatives for m in metas.values())


class Extractor:
    """Extracts records for the registered schemas from documents.

    The Jev client is created from ``TYPESAFE_*`` environment variables on first use
    unless one is passed in. Use as an async context manager (or call ``aclose``) to
    release its connections.
    """

    def __init__(
        self,
        schemas: Sequence[type[BaseModel]],
        *,
        jev: JevClient | None = None,
        pipeline: Pipeline | None = None,
        threshold: float = 0.0,
        thresholds: Mapping[str, float] | None = None,
        budgets: Budgets | None = None,
        store: Store | str | Path | None = None,
        run_id: str | None = None,
        extraction_llm: LLM | None = None,
        generator_llm: LLM | None = None,
        learn_threshold: float = LEARN_THRESHOLD,
        learn_mode: LearnMode = "inline",
        prune_after: int | None = PRUNE_AFTER,
    ) -> None:
        """``threshold`` (default 0: keep everything) and per-field ``thresholds`` (keys
        ``"field"`` or ``"Schema.field"``, and ``"Schema.nested_field.field"`` for a nested
        model's children) set the confidence below which a value becomes ``None`` in the
        record. It stays in ``meta`` with ``filtered=True``.

        ``budgets`` limits LLM and Jev use (:mod:`jevex.budgets`). ``store`` (a
        :class:`~jevex.store.Store` or a URL such as ``"sqlite:///jevex.db"``) holds learned
        state and the spend ledger that shares the run budget across workers; a URL is
        opened on first use and closed by ``aclose``. With a run budget but no store, the
        ledger is kept in memory. ``run_id`` labels this run's ledger entries; workers
        that should share a ``period="run"`` budget pass the same one.

        ``extraction_llm`` turns on the LLM fallback (:mod:`jevex.fallback`): where Jev's
        selection fails, the LLM is asked for the value and its evidence, and Jev verifies
        the answer before it is used. Off (``None``) by default.

        ``generator_llm`` turns on learning (:mod:`jevex.learn`): fallback answers Jev
        verified with probability ``>= learn_threshold`` are turned into generators in
        the background, and documents started after one is accepted use it. Learned
        generators and examples go to the store (an in-memory one if none is given); a
        store's learned generators are used whether or not learning is on.

        ``learn_mode`` says when learning runs (:data:`~jevex.learn.LearnMode`). In
        ``"compile"`` mode documents only log those examples to the store, with or without a
        ``generator_llm``, and :meth:`compile_pack` (``jevex learn``) learns from them
        later. ``"hybrid"`` learns inline like ``"inline"``; :meth:`compile_pack` then
        gathers what was learned into a reviewable pack diff. Both need a ``store``.

        With a store, each document's generator counts are added to its stats
        (:mod:`jevex.housekeeping`), and a learned generator with no wins after
        ``prune_after`` scoped documents is disabled (``None``: never)."""
        if not schemas:
            raise ValueError("register at least one schema")
        self.schemas = [SchemaSpec.from_model(m) for m in schemas]
        names = [s.name for s in self.schemas]
        if len(set(names)) != len(names):
            raise ValueError(f"schema names must be unique: {names}")
        self.pipeline = pipeline if pipeline is not None else default_pipeline()
        self.threshold = threshold
        self.thresholds = dict(thresholds or {})
        unknown = sorted(set(self.thresholds) - _threshold_keys(self.schemas))
        if unknown:
            # Nested models' fields too ("CarModel.trims.power_ps"): ParentChild's children.
            unknown = sorted(set(unknown) - _threshold_keys(_child_specs(self.schemas)))
        if unknown:
            raise ValueError(f"thresholds for unknown fields: {unknown}")
        self._jev = jev
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self.budgets = budgets or Budgets()
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.run_started = datetime.now(UTC)
        self._ledger: RunLedger | None = None
        self._store_source = store if isinstance(store, str | Path) else None
        self._store: Store | None = None if isinstance(store, str | Path) else store
        self._owns_store = False
        self._store_lock: asyncio.Lock | None = None
        self.extraction_llm = extraction_llm
        self.generator_llm = generator_llm
        self.learn_threshold = learn_threshold
        self._learned: LearnedGenerators | None = None
        self._learner: GeneratorLearner | None = None
        self._learn_lock: asyncio.Lock | None = None
        self.learn_mode: LearnMode = learn_mode
        if learn_mode not in get_args(LearnMode):
            raise ValueError(f"learn_mode must be one of {get_args(LearnMode)}, got {learn_mode!r}")
        if learn_mode != "inline" and store is None:
            # Without one, the examples and generators would be lost with the process.
            raise ValueError(
                f"learn_mode={learn_mode!r} keeps what it learns in a store: pass store="
            )
        if not 0 <= learn_threshold <= 1:
            raise ValueError(f"learn_threshold must be between 0 and 1, got {learn_threshold}")
        if prune_after is not None and prune_after < 1:
            raise ValueError(f"prune_after must be at least 1, got {prune_after}")
        self.prune_after = prune_after
        self._housekeeper: Housekeeper | None = None

    @property
    def jev(self) -> JevClient:
        if self._jev is None:
            self._jev = JevClient.from_env()
        return self._jev

    async def store(self) -> Store | None:
        """The store, opened on first use (off the event loop: opening can wait on locks)."""
        if self._store is not None:
            return self._store
        if self._store_source is None and self.budgets.run is None and self.generator_llm is None:
            return None
        if self._store_lock is None:
            self._store_lock = asyncio.Lock()
        async with self._store_lock:
            if self._store is None:
                source = self._store_source or ":memory:"
                self._store = await asyncio.to_thread(open_store, source)
                self._owns_store = True
        return self._store

    async def ledger(self) -> RunLedger:
        """The run ledger shared by this extractor's documents (and by the learner, #38)."""
        store = await self.store()
        if self._ledger is None or self._ledger.store is not store:
            self._ledger = RunLedger(
                self.budgets.run, store, run_id=self.run_id, started=self.run_started
            )
        return self._ledger

    async def learned_generators(self) -> LearnedGenerators | None:
        """The store's learned generators, loaded on first use (``None`` without a store)."""
        store = await self.store()
        if store is None:
            return None
        if self._learn_lock is None:
            self._learn_lock = asyncio.Lock()
        async with self._learn_lock:
            if self._learned is None or self._learned.store is not store:
                learned = LearnedGenerators(store)
                await learned.load()
                self._learned, self._learner = learned, None
        return self._learned

    async def housekeeper(self) -> Housekeeper | None:
        """The housekeeper documents report their generator counts to (``None`` without a
        store). Its ``pruned`` lists the generators it disabled."""
        learned = await self.learned_generators()
        if learned is None or learned.store is None:
            return None
        keeper = self._housekeeper
        if keeper is None or keeper.store is not learned.store or keeper.generators is not learned:
            keeper = Housekeeper(learned.store, learned, prune_after=self.prune_after)
            self._housekeeper = keeper
        return keeper

    async def dedupe_generators(self) -> list[DuplicateGenerator]:
        """Disable stored generators that duplicate another (same field, scope and
        candidates on every stored example; :meth:`~jevex.housekeeping.Housekeeper.dedupe`)
        and return them. Later documents don't run them. Needs a ``store``."""
        keeper = await self.housekeeper()
        if keeper is None:
            raise ValueError("dedupe_generators needs a store: pass store=")
        return await keeper.dedupe()

    async def learner(self) -> GeneratorLearner | None:
        """The inline learner (created on first use), or ``None`` without a
        ``generator_llm`` or in ``"compile"`` mode.

        Its ``outcomes`` say what became of each queued example. It tests generators as
        the pipeline's default candidate, select, normalise and fallback stages would run
        them (their generators, locale, selector, normalisers and fallback threshold).
        """
        if self.generator_llm is None or self.learn_mode == "compile":
            return None
        learned = await self.learned_generators()
        if self._learner is None:
            self._learner = await self._new_learner(
                self.generator_llm, learned or LearnedGenerators()
            )
        return self._learner

    async def _new_learner(self, llm: LLM, generators: LearnedGenerators) -> GeneratorLearner:
        candidates = _stage(self.pipeline, "candidates", CandidateStage)
        select = _stage(self.pipeline, "select", SelectStage)
        norm = _stage(self.pipeline, "normalise", NormaliseStage)
        fallback = _stage(self.pipeline, "fallback", FallbackStage)
        return GeneratorLearner(
            self.schemas,
            llm,
            self.jev,
            generators=generators,
            ledger=await self.ledger(),
            base=candidates.registry if candidates else GeneratorRegistry(),
            locale=candidates.locale if candidates else None,
            selector=select.selector if select else JevCandidateSelector(),
            normalisers=norm.registry if norm else BUILTIN_NORMALISERS,
            fallback_threshold=fallback.fallback_threshold if fallback else FALLBACK_THRESHOLD,
            learn_threshold=self.learn_threshold,
        )

    async def compile_pack(self, pack: Sequence[GeneratorSpec] = ()) -> PackDiff:
        """Learn from the store's logged examples in one batch and return what to add to
        ``pack`` for review (:func:`~jevex.learn.compile_pack`; ``jevex learn``).

        Needs a ``store`` and a ``generator_llm``. Generators are tested as the inline
        learner would test them, but nothing is published to the store or used by this
        extractor's documents: they reach documents once the reviewed diff is imported.
        """
        if self.generator_llm is None:
            raise ValueError("compile_pack needs a generator_llm")
        if self._store_source is None and (self._store is None or self._owns_store):
            # Not the in-memory store an extractor opens for itself: it has no examples.
            raise ValueError("compile_pack learns from a store's examples: pass store=")
        store = await self.store()
        assert store is not None
        learned = LearnedGenerators(store, persist=False)
        return await compile_pack(await self._new_learner(self.generator_llm, learned), pack)

    async def wait_for_learning(self) -> None:
        """Wait until every example queued so far has been learned from (or rejected)."""
        if self._learner is not None:
            await self._learner.drain()

    async def extract(self, document: Document) -> ExtractionResult:
        """Run the pipeline over one document."""
        budget = DocumentBudget(self.budgets, await self.ledger())
        jev = self.jev.metered(max_requests=self.budgets.per_document.max_jev_requests)
        ctx = Context.create(document, self.schemas, jev)
        ctx.budget = budget
        ctx.store = await self.store()
        ctx.extraction_llm = self.extraction_llm
        learner = await self.learner()
        if learner is None and self.learn_mode == "compile":
            store = await self.store()
            assert store is not None  # checked in __init__
            ctx.learner = ExampleLogger(store, self.learn_threshold)
        else:
            ctx.learner = learner
        learned = await self.learned_generators()
        ctx.generators = learned.current if learned else None
        ctx.housekeeper = await self.housekeeper()
        try:
            if not await budget.start_document():
                ctx.stop("budget", "the run's Jev spend cap is reached")
            else:
                try:
                    await self.pipeline.run(ctx)
                except JevRequestCapError as exc:
                    budget.record_hit("document", "max_jev_requests", str(exc))
                    ctx.stop("budget", str(exc))
        finally:
            # Every branch has settled (fan-outs cancel on failure, and a request cancelled
            # mid-flight is counted at its estimate), so this is the document's whole Jev
            # spend, recorded even when a stage failed.
            await budget.finish_document(ctx.jev.usage.cost)
        return ExtractionResult.from_context(
            ctx, threshold=self.threshold, thresholds=self.thresholds
        )

    def extract_sync(self, document: Document) -> ExtractionResult:
        """Blocking wrapper for scripts and notebooks without a running event loop.

        Reuses one private event loop so HTTP connections stay valid between calls.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("extract_sync() called inside an event loop; use await extract()")
        if self._sync_loop is None or self._sync_loop.is_closed():
            self._sync_loop = asyncio.new_event_loop()
        return self._sync_loop.run_until_complete(self.extract(document))

    async def aclose(self) -> None:
        """Stop the learner (examples still queued stay in the store, unlearned; call
        :meth:`wait_for_learning` first to finish them), then close Jev and the store."""
        learner, self._learner, self._learned, self._learn_lock = self._learner, None, None, None
        self._housekeeper = None
        try:
            if learner is not None:
                await learner.aclose()
        finally:
            close = getattr(self._jev.backend, "aclose", None) if self._jev else None
            try:
                if close is not None:
                    await close()
            finally:
                store, owned = self._store, self._owns_store
                if owned:
                    self._store, self._owns_store, self._ledger = None, False, None
                # The lock binds to the loop that used it; a reopen may be on another loop.
                self._store_lock = None
                if owned and store is not None:
                    await store.aclose()

    def close(self) -> None:
        """Close the Jev client and the private loop used by ``extract_sync``."""
        loop = self._sync_loop
        if loop is not None and not loop.is_closed():
            loop.run_until_complete(self.aclose())
            loop.close()
        self._sync_loop = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

"""The pipeline: an ordered list of stages run over a shared per-document context.

Every stage has the same shape, ``async run(ctx)``. It reads what earlier stages left
on the :class:`Context` and adds its own results. Stages are replaced, removed or
added by name with :class:`Pipeline`'s helpers.

Work fans out inside a stage, not across stages: a stage handles every active schema
and entity scope concurrently (see :func:`for_each_scope`), so each level's Jev
questions go out together.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from jevex._tasks import gather
from jevex.errors import PartErrors
from jevex.locales import canonical_locale, document_locale
from jevex.logs import get_logger, log_context
from jevex.results import Conflict
from jevex.tracing import record_failure, set_attributes, trace_span

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence

    from opentelemetry.trace import Span as OtelSpan
    from opentelemetry.trace import Tracer

    from jevex.budgets import DocumentBudget
    from jevex.document import Document
    from jevex.entities import EntityScope
    from jevex.errors import PartKind
    from jevex.housekeeping import Housekeeper
    from jevex.interfaces import GateDecision, Learner, ParsedDocument, Selection
    from jevex.jev import ChoiceAnswer, JevClient
    from jevex.keypaths import StructuredItem
    from jevex.learn import GeneratorSnapshot
    from jevex.llm import LLM
    from jevex.packs import Pack
    from jevex.results import FieldMeta, Method, Source
    from jevex.schema import FieldSpec, SchemaSpec
    from jevex.statements import Candidate, Span, Statement
    from jevex.store import Store, VerifiedExample

log = get_logger(__name__)


@dataclass(frozen=True)
class VisionValue:
    """A value (a list field's item) that only a vision model's statement gave, waiting
    for the fallback stage to verify it. ``span`` is the candidate's, when it had one."""

    statement_id: str
    value: Any
    span: Span | None = None


def vision_values(given: Iterable[tuple[Statement, Any, Span | None]]) -> list[VisionValue]:
    """Of the values a route offers, as ``(statement, value, span)`` for each statement
    that gave one, those only vision statements gave, once each, in order: a value that
    a document's own text states too needs no check."""
    given = list(given)
    stated = [value for statement, value, _ in given if statement.kind != "vision"]
    out: list[VisionValue] = []
    for statement, value, span in given:
        if statement.kind != "vision" or value in stated:
            continue
        if not any(v.statement_id == statement.id and v.value == value for v in out):
            out.append(VisionValue(statement.id, value, span))
    return out


@dataclass(frozen=True)
class ValuePick:
    """One pick a route took into a list field's value: the items it gave, and how the
    field's meta describes the value when this pick is the best one in it."""

    items: tuple[Any, ...]
    method: Method
    source: Source
    confidence: float | None
    generator_id: str | None = None
    shared: bool = False


@runtime_checkable
class Stage(Protocol):
    """One step of the pipeline. ``name`` identifies it for replace/remove/insert."""

    name: str

    async def run(self, ctx: Context) -> None: ...


@dataclass
class SchemaRun:
    """Per-schema working state for one document."""

    spec: SchemaSpec
    active: bool = True
    gate: GateDecision | None = None
    component_ids: dict[str, list[str]] | None = None
    """Component-gate result: group → ids of components relevant to it. ``None`` when no
    component gate ran, which means every component is relevant."""
    child_component_ids: dict[str, dict[str, list[str]]] = field(
        default_factory=dict[str, dict[str, list[str]]]
    )
    """Component-gate results for the nested models' own field groups, keyed by the
    nested-model field, then group. The entity stage gives them to that field's child run
    as its ``component_ids``."""
    ungated_groups: set[str] = field(default_factory=set[str])
    """Field groups the component gate didn't ask about because another route (embedded
    data, in ``fill_gaps`` mode) had found all their fields. They stay categorise options
    (:meth:`relevant_fields`), so a statement about one isn't put down to another field."""
    scopes: list[EntityScope] = field(default_factory=list["EntityScope"])
    categories: dict[str, ChoiceAnswer] = field(default_factory=dict[str, "ChoiceAnswer"])
    candidates: dict[tuple[str, str], list[Candidate]] = field(
        default_factory=dict[tuple[str, str], list["Candidate"]]
    )
    selections: dict[tuple[str, str, str], Selection] = field(
        default_factory=dict[tuple[str, str, str], "Selection"]
    )
    """Keyed by (scope label, field name, statement id)."""
    fields: dict[str, dict[str, FieldMeta]] = field(
        default_factory=dict[str, dict[str, "FieldMeta"]]
    )
    """What was found, keyed by scope label, then field name. Records are built from this."""
    value_generators: dict[tuple[str, str], set[str]] = field(
        default_factory=dict[tuple[str, str], set[str]]
    )
    """Keyed by (scope label, field name): ids of the generators whose candidates are in
    the value the normalise stage offered (every accepted pick for a list field). Whether
    that value stood is up to :attr:`fields`."""
    vision_values: dict[tuple[str, str], list[VisionValue]] = field(
        default_factory=dict[tuple[str, str], list[VisionValue]]
    )
    """Keyed by (scope label, field name): the values (a list field's items) in what the
    select or normalise stage offered that only vision statements gave. The fallback
    stage verifies them, if the offered value stood (:mod:`jevex.fallback`)."""
    value_picks: dict[tuple[str, str], list[ValuePick]] = field(
        default_factory=dict[tuple[str, str], list[ValuePick]]
    )
    """Keyed by (scope label, field name), for a list field with :attr:`vision_values`:
    the picks in the value the select or normalise stage offered, best first. When the
    fallback stage drops some of the vision items, the best pick left in the value
    describes it (method, source, confidence) and :attr:`value_generators` keeps only
    the generators of the picks left."""
    values: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """Bare values by scope label, then field name, for stages with no metadata to give.
    Used only when ``fields`` has no entry for that field."""
    parent: str | None = None
    """For a nested model's run (the entity stage adds one per field ``ParentChild``
    fills): the parent schema's name. Its records become the parent records' children,
    not records of their own."""
    parent_field: str | None = None
    """The parent schema's field this run's records fill."""
    finished: bool = False
    """Set when the schema needs nothing from later stages: the structured stage found its
    values and its mode skips the layout route. Unlike :meth:`deactivate`, its records are
    still built; it just leaves ``Context.active``."""
    merge: bool = False
    """Set by the structured stage in ``merge`` mode: later routes look for every field
    again, and :meth:`offer_field` settles disagreements instead of keeping the first
    value found."""
    structured_items: list[StructuredItem] = field(default_factory=list["StructuredItem"])
    """Named objects in the embedded data's arrays and the values each gives (set by the
    structured stage), so the entity stage can give an entity the values of the object
    naming it."""
    structured_rest: dict[str, FieldMeta] = field(default_factory=dict[str, "FieldMeta"])
    """The embedded data's values from outside :attr:`structured_items`."""

    def relevant_components(self) -> set[str] | None:
        """Components that passed the gate for any group; ``None`` if nothing was gated."""
        if self.component_ids is None:
            return None
        return {cid for ids in self.component_ids.values() for cid in ids}

    def relevant_fields(self, component_id: str) -> list[FieldSpec]:
        """The fields a component can state: those whose group it passed the gate for, and
        those of :attr:`ungated_groups` if it passed any group.

        Every field when no component gate ran.
        """
        if self.component_ids is None:
            return list(self.spec.fields)
        passed = {g for g, ids in self.component_ids.items() if component_id in ids}
        if passed:
            passed |= self.ungated_groups
        return [f for f in self.spec.fields if (f.group or f.name) in passed]

    def shared_statements(self, scope: str) -> set[str]:
        """Ids of the statements ``scope`` shares with every other entity ("all of them").

        A value from one of these is recorded with ``shared=True``, and only when none of
        the scope's own statements gives the field.
        """
        return {sid for s in self.scopes if s.label == scope for sid in s.shared_statement_ids}

    def set_field(self, scope: str, name: str, meta: FieldMeta) -> None:
        self.fields.setdefault(scope, {})[name] = meta

    def needs(self, scope: str, name: str) -> bool:
        """Whether a route should still look for ``name`` in ``scope``: no route found it
        yet, only as a value every entity shares, or routes are merged."""
        existing = self.fields.get(scope, {}).get(name)
        return self.merge or existing is None or not existing.found or existing.shared

    def offer_field(self, scope: str, name: str, meta: FieldMeta) -> None:
        """Record a route's ``meta`` unless another route already found the field.

        A value of the entity's own replaces one it shares with every entity (in ``merge``
        mode the shared one, if it differs, goes on its ``conflicts``). In ``merge`` mode,
        two found values are otherwise weighed: the more confident one wins (a value with
        no confidence, such as a direct read of embedded data, counts as certain; a tie
        keeps the earlier route's), and the other, if it differs, goes on the winner's
        ``conflicts``.
        """
        existing = self.fields.get(scope, {}).get(name)
        if existing is None or not existing.found:
            self.set_field(scope, name, meta)
            return
        if existing.shared and meta.found and not meta.shared:
            conflicts = list(meta.conflicts)
            if self.merge and existing.value != meta.value:
                conflicts.append(_conflict(existing))
            self.set_field(scope, name, meta.model_copy(update={"conflicts": conflicts}))
            return
        if not self.merge or not meta.found:
            return
        winner, loser = existing, meta
        if _certainty(meta) > _certainty(existing):
            winner, loser = meta, existing
        conflicts = [*winner.conflicts, *loser.conflicts]
        if loser.value != winner.value:
            conflicts.append(_conflict(loser))
        self.set_field(scope, name, winner.model_copy(update={"conflicts": conflicts}))

    @property
    def name(self) -> str:
        return self.spec.name

    def deactivate(self) -> None:
        """Stop later stages working on this schema (e.g. the document gate said no)."""
        self.active = False

    def finish(self) -> None:
        """Stop later stages working on this schema because it has what it needs."""
        self.finished = True


def _conflict(meta: FieldMeta) -> Conflict:
    return Conflict(
        value=meta.value, method=meta.method, confidence=meta.confidence, source=meta.source
    )


def _certainty(meta: FieldMeta) -> float:
    return 1.0 if meta.confidence is None else meta.confidence


@dataclass
class Event:
    """Something worth reporting in document meta: a skipped stage, a budget hit..."""

    stage: str
    kind: str
    message: str
    data: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class Context:
    """Everything known about one document as it moves through the pipeline."""

    document: Document
    jev: JevClient
    schemas: dict[str, SchemaRun]
    parsed: ParsedDocument | None = None
    structured: list[Statement] = field(default_factory=list["Statement"])
    headed_tables: frozenset[str] = frozenset()
    """Tables without header cells that the component gate found headers in, by component
    id: the statement stage reads them with those marked (:func:`~jevex.tables.infer_headers`)."""
    timings: dict[str, float] = field(default_factory=dict[str, float])
    events: list[Event] = field(default_factory=list[Event])
    stopped: bool = False
    budget: DocumentBudget | None = None
    """The document's budgets; LLM calls go through ``budget.call_llm`` (see
    :mod:`jevex.budgets`). ``None`` outside an extractor, meaning unlimited."""
    store: Store | None = None
    """The extractor's store (learned state: key mappings, generators...), ``None`` without
    one. Stages that learn read and write it here."""
    packs: Sequence[Pack] = ()
    """The extractor's packs, project packs then community packs (:mod:`jevex.packs`): the
    layers under :attr:`store` that the structured stage looks key mappings up in."""
    pipeline: Pipeline | None = None
    """The pipeline running this context (set by :meth:`Pipeline.run`), for a stage whose
    work depends on how a later stage is configured. ``None`` when stages are run by hand."""
    extraction_llm: LLM | None = None
    """The extractor's ``extraction_llm``: the fallback stage asks it where Jev's selection
    failed. ``None`` turns the fallback off."""
    vision_llm: LLM | None = None
    """The extractor's ``vision_llm``: the image stage adds a
    :class:`~jevex.images.VisionProcessor` over it. ``None``: only the stage's own
    processors read images."""
    verified: list[VerifiedExample] = field(default_factory=list["VerifiedExample"])
    """LLM and vision answers that passed Jev verification on this document, queued for
    learning."""
    learner: Learner | None = None
    """The extractor's learner (a ``GeneratorLearner`` with a ``generator_llm``, or an
    ``ExampleLogger`` in ``compile`` mode): the learn stage hands it :attr:`verified`.
    ``None`` turns learning off."""
    generators: GeneratorSnapshot | None = None
    """The learned generators this document runs with, taken when it starts: generators
    learned meanwhile are for later documents. ``None``: only the stages' own."""
    generators_ran: set[str] = field(default_factory=set[str])
    """Ids of the generators the candidate stage ran on at least one statement (the
    housekeeper counts these documents towards each one's stats)."""
    housekeeper: Housekeeper | None = None
    """The extractor's :class:`~jevex.housekeeping.Housekeeper` (when it has a store): the
    learn stage gives it the document's generator counts. ``None``: none are kept."""
    default_locale: str | None = None
    """The extractor's ``locale``: the locale of a document that doesn't say its own.
    ``None``: such a document has none (stages fall back to their own ``locale``)."""
    errors: PartErrors = field(default_factory=PartErrors)
    """What failed on this document (:mod:`jevex.errors`): parts skipped
    (:meth:`part_failed`) and the core failure that ended it, if any."""
    stage: str | None = None
    """The stage running now (set by :meth:`Pipeline.run`); after a stage raised, the
    one that raised."""
    tracer: Tracer | None = None
    """The extractor's OpenTelemetry tracer (:mod:`jevex.tracing`): :meth:`Pipeline.run`
    traces each stage under the document's span. ``None``: no tracing."""
    span: OtelSpan | None = None
    """The span of the stage running now (when tracing), which part failures are recorded
    on."""

    @classmethod
    def create(cls, document: Document, schemas: Sequence[SchemaSpec], jev: JevClient) -> Context:
        return cls(
            document=document,
            jev=jev,
            schemas={s.name: SchemaRun(s) for s in schemas},
        )

    @cached_property
    def locale(self) -> str | None:
        """The document's own locale (:func:`~jevex.locales.document_locale`), else
        :attr:`default_locale`, canonical (:func:`~jevex.locales.canonical_locale`).
        ``None`` when neither says. Read once: cleaning keeps what it's read from."""
        own = document_locale(self.document)
        if own:
            return own
        return canonical_locale(self.default_locale) if self.default_locale else None

    @property
    def active(self) -> list[SchemaRun]:
        """The runs later stages work on: those still active and not finished."""
        return [run for run in self.schemas.values() if run.active and not run.finished]

    def stop(self, stage: str, reason: str) -> None:
        """End the run early; later stages are skipped."""
        self.stopped = True
        self.event(stage, "stopped", reason)

    def event(self, stage: str, kind: str, message: str, **data: Any) -> None:
        """Report something in the document's meta (and log it: ``stopped`` at ``INFO``,
        the rest at ``DEBUG``)."""
        self.events.append(Event(stage, kind, message, data))
        level = logging.INFO if kind == "stopped" else logging.DEBUG
        log.log(level, "%s: %s", kind, message, extra={"stage": stage})

    def part_failed(self, stage: str, kind: PartKind, part: str | None, exc: Exception) -> None:
        """Record that a pluggable part raised on one input and was skipped for it. The
        result is ``partial``; the pipeline carries on (:mod:`jevex.errors`). It's logged
        and, when tracing, recorded on the stage's span."""
        self.errors.add(stage, kind, part, exc)
        record_failure(self.span, exc, stage=stage, kind=kind, part=part, fatal=False)


async def for_each_scope[T](
    ctx: Context, fn: Callable[[SchemaRun, EntityScope], Awaitable[T]]
) -> list[T]:
    """Run ``fn`` for every entity scope of every active schema, concurrently.

    The first failure cancels the other calls before it propagates.
    """
    return await gather(fn(run, scope) for run in ctx.active for scope in run.scopes)


async def for_each_schema[T](ctx: Context, fn: Callable[[SchemaRun], Awaitable[T]]) -> list[T]:
    """Run ``fn`` for every active schema, concurrently (a failure cancels the rest)."""
    return await gather(fn(run) for run in ctx.active)


class Pipeline:
    """An immutable, ordered list of stages. The helpers return new pipelines."""

    def __init__(self, stages: Sequence[Stage] = ()) -> None:
        names = [s.name for s in stages]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate stage names: {sorted(duplicates)}")
        self._stages = tuple(stages)

    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._stages]

    def __iter__(self) -> Iterator[Stage]:
        return iter(self._stages)

    def __len__(self) -> int:
        return len(self._stages)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __repr__(self) -> str:
        return f"Pipeline({self.names})"

    def replace(self, name: str, stage: Stage) -> Pipeline:
        i = self._index(name)
        return Pipeline([*self._stages[:i], stage, *self._stages[i + 1 :]])

    def without(self, *names: str) -> Pipeline:
        for name in names:
            self._index(name)
        return Pipeline([s for s in self._stages if s.name not in names])

    def insert_before(self, name: str, stage: Stage) -> Pipeline:
        i = self._index(name)
        return Pipeline([*self._stages[:i], stage, *self._stages[i:]])

    def insert_after(self, name: str, stage: Stage) -> Pipeline:
        i = self._index(name)
        return Pipeline([*self._stages[: i + 1], stage, *self._stages[i + 1 :]])

    def append(self, stage: Stage) -> Pipeline:
        return Pipeline([*self._stages, stage])

    async def run(self, ctx: Context) -> Context:
        """Run each stage in order until the context stops or no schema is active."""
        ctx.pipeline = self
        for stage in self._stages:
            if ctx.stopped:
                break
            if not ctx.active:
                # Finished schemas need nothing more; only all-inactive ones stop the run.
                if not any(run.active for run in ctx.schemas.values()):
                    ctx.stop(stage.name, "no schema is still active")
                break
            start = time.perf_counter()
            ctx.stage = stage.name
            with (
                log_context(stage=stage.name),
                trace_span(
                    ctx.tracer, f"jevex.stage {stage.name}", {"jevex.stage": stage.name}
                ) as now,
            ):
                ctx.span = now
                before = _usage(ctx)
                log.debug("stage %s started", stage.name)
                try:
                    await stage.run(ctx)
                finally:
                    elapsed = ctx.timings[stage.name] = time.perf_counter() - start
                    ctx.span = None
                    _trace_stage(now, ctx, before)
                log.debug("stage %s finished in %.3fs", stage.name, elapsed)
        ctx.stage = None
        return ctx

    def _index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no stage named {name!r} in {self!r}") from None


def _usage(ctx: Context) -> tuple[int, float, int, float]:
    """The document's Jev requests and cost and LLM calls and cost so far."""
    budget = ctx.budget
    return (
        ctx.jev.usage.requests,
        ctx.jev.usage.cost,
        budget.llm_calls if budget else 0,
        budget.llm_spend if budget else 0.0,
    )


def _trace_stage(
    current: OtelSpan | None, ctx: Context, before: tuple[int, float, int, float]
) -> None:
    """A stage span's attributes: what's active when it ends and what the stage used."""
    if current is None or not current.is_recording():
        return
    after = _usage(ctx)
    active = ctx.active
    set_attributes(
        current,
        {
            "jevex.schemas": [run.name for run in active],
            "jevex.entities": sum(len(run.scopes) for run in active),
            "jevex.jev.requests": after[0] - before[0],
            "jevex.jev.cost_usd": after[1] - before[1],
            "jevex.llm.calls": after[2] - before[2],
            "jevex.llm.cost_usd": after[3] - before[3],
        },
    )

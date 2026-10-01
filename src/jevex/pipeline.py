"""The pipeline: an ordered list of stages run over a shared per-document context.

Every stage has the same shape, ``async run(ctx)``. It reads what earlier stages left
on the :class:`Context` and adds its own results. Stages are replaced, removed or
added by name with :class:`Pipeline`'s helpers.

Work fans out inside a stage, not across stages: a stage handles every active schema
and entity scope concurrently (see :func:`for_each_scope`), so each level's Jev
questions go out together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from jevex._tasks import gather
from jevex.locales import document_locale
from jevex.results import Conflict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Sequence

    from jevex.budgets import DocumentBudget
    from jevex.document import Document
    from jevex.entities import EntityScope
    from jevex.housekeeping import Housekeeper
    from jevex.interfaces import GateDecision, Learner, ParsedDocument, Selection
    from jevex.jev import ChoiceAnswer, JevClient
    from jevex.learn import GeneratorSnapshot
    from jevex.llm import LLM
    from jevex.results import FieldMeta
    from jevex.schema import FieldSpec, SchemaSpec
    from jevex.statements import Candidate, Statement
    from jevex.store import Store, VerifiedExample


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
        yet, or routes are merged."""
        existing = self.fields.get(scope, {}).get(name)
        return self.merge or existing is None or not existing.found

    def offer_field(self, scope: str, name: str, meta: FieldMeta) -> None:
        """Record a route's ``meta`` unless another route already found the field.

        In ``merge`` mode, two found values are weighed instead: the more confident one
        wins (a value with no confidence, such as a direct read of embedded data, counts
        as certain; a tie keeps the earlier route's), and the other, if it differs, goes
        on the winner's ``conflicts``.
        """
        existing = self.fields.get(scope, {}).get(name)
        if existing is None or not existing.found:
            self.set_field(scope, name, meta)
            return
        if not self.merge or not meta.found:
            return
        winner, loser = existing, meta
        if _certainty(meta) > _certainty(existing):
            winner, loser = meta, existing
        conflicts = [*winner.conflicts, *loser.conflicts]
        if loser.value != winner.value:
            conflicts.append(
                Conflict(
                    value=loser.value,
                    method=loser.method,
                    confidence=loser.confidence,
                    source=loser.source,
                )
            )
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
    timings: dict[str, float] = field(default_factory=dict[str, float])
    events: list[Event] = field(default_factory=list[Event])
    stopped: bool = False
    budget: DocumentBudget | None = None
    """The document's budgets; LLM calls go through ``budget.call_llm`` (see
    :mod:`jevex.budgets`). ``None`` outside an extractor, meaning unlimited."""
    store: Store | None = None
    """The extractor's store (learned state: key mappings, generators...), ``None`` without
    one. Stages that learn read and write it here."""
    pipeline: Pipeline | None = None
    """The pipeline running this context (set by :meth:`Pipeline.run`), for a stage whose
    work depends on how a later stage is configured. ``None`` when stages are run by hand."""
    extraction_llm: LLM | None = None
    """The extractor's ``extraction_llm``: the fallback stage asks it where Jev's selection
    failed. ``None`` turns the fallback off."""
    verified: list[VerifiedExample] = field(default_factory=list["VerifiedExample"])
    """LLM answers that passed Jev verification on this document, queued for learning."""
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

    @classmethod
    def create(cls, document: Document, schemas: Sequence[SchemaSpec], jev: JevClient) -> Context:
        return cls(
            document=document,
            jev=jev,
            schemas={s.name: SchemaRun(s) for s in schemas},
        )

    @cached_property
    def locale(self) -> str | None:
        """The document's own locale (:func:`~jevex.locales.document_locale`), ``None``
        when it doesn't say. Read once: cleaning keeps what it's read from."""
        return document_locale(self.document)

    @property
    def active(self) -> list[SchemaRun]:
        """The runs later stages work on: those still active and not finished."""
        return [run for run in self.schemas.values() if run.active and not run.finished]

    def stop(self, stage: str, reason: str) -> None:
        """End the run early; later stages are skipped."""
        self.stopped = True
        self.event(stage, "stopped", reason)

    def event(self, stage: str, kind: str, message: str, **data: Any) -> None:
        self.events.append(Event(stage, kind, message, data))


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
            try:
                await stage.run(ctx)
            finally:
                ctx.timings[stage.name] = time.perf_counter() - start
        return ctx

    def _index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no stage named {name!r} in {self!r}") from None

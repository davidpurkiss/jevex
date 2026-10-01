"""The learner: verified LLM answers become generators (spec: *Learning loop*; stage 15).

Opt-in through ``Extractor(generator_llm=...)``. The fallback stage (#33) puts every LLM
answer Jev verified on ``ctx.verified``; :class:`LearnStage` hands them to the learner
and the document carries on. :class:`GeneratorLearner` then works through them one at a
time, in the background:

1. **Queue.** Only examples verified with ``probability >= learn_threshold`` (or from a
   human) are queued, and they are stored as verified examples first, so they serve as
   regression tests for later generators even if learning is cut short.
2. **Covered?** If the generators in use already propose a span that normalises to the
   value, the miss wasn't one of recall and there's nothing to learn.
3. **Synthesise.** ``generator_llm`` writes the pattern and normalisers as structured
   output (:class:`GeneratorDraft`), asked for recall, not for the only match. The
   learner fills in the id, field, scope and provenance.
4. **Validate.** :meth:`~jevex.generators.GeneratorSpec.parse`: the pattern compiles
   under RE2 within the length cap, and the normalisers are built in.
5. **Test.** The generator must find the value in the triggering statement, and Jev
   must choose it there among every generator's candidates. On up to ``sample_size`` of
   the field's stored examples (the newest), it must not lower accuracy: wherever it
   changes the candidates, Jev picks from the old and the new set (one request per
   example) and the new set must be right at least as often.
6. **Hot-swap.** An accepted spec is put in the store and published as a new
   :class:`GeneratorSnapshot`. A document takes the current snapshot when it starts and
   keeps it to the end; later documents get the new one. Other processes sharing the
   store pick it up when they next refresh (:meth:`LearnedGenerators.refresh`).

Every example ends in a :class:`LearnOutcome` on :attr:`GeneratorLearner.outcomes`.
Expected failures (a budget saying no, an LLM or Jev error, an invalid spec, a failed
test) are outcomes, not exceptions. Anything else stops the worker; the error is raised
by the next :meth:`~GeneratorLearner.submit`, :meth:`~GeneratorLearner.drain` or
:meth:`~GeneratorLearner.aclose`.

That is the ``inline`` mode (:data:`LearnMode`). In ``compile`` mode documents only log
examples (:class:`ExampleLogger`), and :func:`compile_pack` (``jevex learn``) runs steps 2
to 5 over them in a batch. It publishes nothing: the generators it accepts, plus any the
store learned inline (``hybrid`` mode), become a :class:`PackDiff` written out for review.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from jevex._tasks import gather
from jevex.budgets import RunLedger
from jevex.fallback import FALLBACK_THRESHOLD, field_type_text
from jevex.generators import (
    GeneratorRegistry,
    GeneratorSpec,
    InvalidGeneratorError,
    default_registry,
)
from jevex.generators.regex import MAX_PATTERN_LENGTH
from jevex.generators.spec import NORMALISE_JSON_SCHEMA
from jevex.jev import JevBudgetExceededError, JevError
from jevex.layout import DomLocation, section_text
from jevex.llm import LLMError
from jevex.normalise import BUILTIN_NORMALISERS, NormaliseError, NormaliserRegistry, normalise
from jevex.packs import (
    PACK_GENERATORS,
    generator_record,
    layered_generators,
    stored_generators,
)
from jevex.select import JevCandidateSelector, statement_state, unique_spans
from jevex.statements import Statement, StatementKind

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from jevex.housekeeping import Housekeeper
    from jevex.interfaces import CandidateSelector, Learner, Selection
    from jevex.jev import Answer, JevClient
    from jevex.llm import LLM
    from jevex.packs import Pack
    from jevex.pipeline import Context
    from jevex.schema import FieldSpec, SchemaSpec
    from jevex.statements import Candidate
    from jevex.store import Store, VerifiedExample

LEARN_THRESHOLD = 0.9
"""Verification probability at or above which an LLM answer is learned from. Provisional:
the spec's open questions set the defaults from eval runs."""
SAMPLE_SIZE = 20
"""How many of a field's stored examples a new generator is tested on (all, if fewer)."""
REFRESH_GENERATORS = 30.0
"""Default seconds between an extractor's checks for generators other processes sharing its
store published or disabled (:meth:`LearnedGenerators.refresh`)."""


# --- snapshots ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorSnapshot:
    """The learned generators at one moment. Never changes: publishing makes a new one.

    ``version`` counts changes in this process (0: what the store held at start):
    publishes, withdrawals, and refreshes that found the store changed.
    """

    version: int
    registry: GeneratorRegistry

    def on(self, base: GeneratorRegistry) -> GeneratorRegistry:
        """``base`` followed by these generators (``base`` wins ties and keeps its ids)."""
        return base.extended(self.registry)


class LearnedGenerators:
    """Holds the current :class:`GeneratorSnapshot`, loaded from and published to a store.

    ``store`` is the local learned layer: :meth:`load` reads its enabled generators, and
    :meth:`publish` writes there before swapping the snapshot. With ``persist=False``,
    :meth:`publish` only swaps the snapshot: a batch compile (:func:`compile_pack`) reads
    the store but leaves what it learns for review.

    Other processes sharing the store (Scrapy workers, ``jevex serve``) publish and
    disable generators too. :meth:`refresh` reloads at most every ``refresh_after``
    seconds (``None``: never) and swaps the snapshot if the store changed. ``clock``
    gives the seconds ``refresh_after`` is measured in (monotonic; tests pass a fake one).

    ``packs`` are the layers below the store: project packs, then community packs
    (:func:`~jevex.packs.layered_generators`). The store's disable list applies to them.
    """

    def __init__(
        self,
        store: Store | None = None,
        *,
        persist: bool = True,
        packs: Sequence[Pack] = (),
        refresh_after: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if refresh_after is not None and refresh_after < 0:
            raise ValueError(f"refresh_after must be at least 0, got {refresh_after}")
        self.store = store
        self.persist = persist
        self.packs = list(packs)
        self.refresh_after = refresh_after
        self.clock = clock
        self.current = GeneratorSnapshot(0, GeneratorRegistry())
        self._loaded_at: float | None = None
        self._refreshing = False

    async def load(self, also: Sequence[GeneratorSpec] = ()) -> GeneratorSnapshot:
        """Replace the snapshot with the store's enabled generators (oldest first), then
        the packs' (through the layers), then ``also`` (a pack diff's base; an id already
        in wins).

        Raises :class:`~jevex.store.StoreError` for a stored spec that doesn't validate.
        """
        self._loaded_at = self.clock()
        registry = (await self._layered()).extended(s.to_generator() for s in also)
        self.current = GeneratorSnapshot(self.current.version, registry)
        return self.current

    async def refresh(self) -> GeneratorSnapshot:
        """Reload if ``refresh_after`` seconds have passed since the last load, and make a
        new snapshot if the store (or a pack) now gives different generators: ones another
        process published, or without ones it disabled. Documents already running keep
        their snapshot. Returns the current snapshot.

        Does nothing without a store, with ``persist`` off (the snapshot holds what
        :meth:`publish` didn't store), with ``refresh_after=None``, or while another
        refresh is running (the caller takes the current snapshot rather than wait). If
        this process publishes or withdraws during the reload, the reload is dropped and
        the next call tries again. Raises :class:`~jevex.store.StoreError` for a stored
        spec that doesn't validate.
        """
        if self.store is None or not self.persist or self.refresh_after is None:
            return self.current
        now = self.clock()
        due = self._loaded_at is None or now - self._loaded_at >= self.refresh_after
        if not due or self._refreshing:
            return self.current
        self._refreshing = True
        before = self.current
        try:
            registry = await self._layered()
        finally:
            self._refreshing = False
        if self.current is not before:
            return self.current  # stale: it may lack what was published meanwhile
        self._loaded_at = now
        if list(registry) != list(before.registry):
            self.current = GeneratorSnapshot(before.version + 1, registry)
        return self.current

    async def _layered(self) -> GeneratorRegistry:
        specs = await self.stored()
        if self.packs:
            disabled = await self.store.disabled_generator_ids() if self.store else set[str]()
            specs = layered_generators(specs, disabled, self.packs)
        return GeneratorRegistry([s.to_generator() for s in specs])

    async def stored(self) -> list[GeneratorSpec]:
        """The store's enabled generator specs, oldest first (none without a store).

        Raises :class:`~jevex.store.StoreError` for a stored spec that doesn't validate.
        """
        return await stored_generators(self.store) if self.store is not None else []

    async def publish(self, spec: GeneratorSpec) -> GeneratorSnapshot:
        """Store ``spec`` (unless ``persist`` is off) and make a new snapshot holding it
        (replacing one with its id)."""
        if self.store is not None and self.persist:
            await self.store.put_generator(generator_record(spec))
        self.current = GeneratorSnapshot(
            self.current.version + 1, self.current.registry.with_generator(spec.to_generator())
        )
        return self.current

    def withdraw(self, generator_ids: Sequence[str]) -> GeneratorSnapshot:
        """Make a new snapshot without these generators (housekeeping disabled them in the
        store). Ids it doesn't hold are ignored; with none held, the snapshot stays."""
        registry = self.current.registry
        held = [gid for gid in generator_ids if gid in registry]
        if not held:
            return self.current
        for gid in held:
            registry = registry.without(gid)
        self.current = GeneratorSnapshot(self.current.version + 1, registry)
        return self.current


# --- synthesis -------------------------------------------------------------------------

NORMALISERS = """\
  - strip: trim whitespace and surrounding punctuation (text values)
  - parse_number: the first number in the text ("1,395" -> 1395)
  - parse_range: a range of numbers ("5-7" -> [5, 7])
  - unit: {from: <unit in the text>, to: <unit wanted>} converts a number; omit "to" to
    convert to the field's unit
  - parse_money: an amount with its currency ("£18,495"); {currency: GBP} when the text
    has none
  - parse_date: a date; {order: dmy|mdy|ymd} for numeric dates, {precision: day|month|year}"""

PROMPT = """\
Write a regular expression that finds one field's value in statements from documents.

Field: {name}
Description: {description}
Type: {type}

Example statement: {statement}
{section}Its value: {value}
The words that state it: {evidence}

The pattern will run on other statements phrased like this one, from documents of the
same kind. Aim for recall: it should find this kind of value wherever a statement like
this states it, not only in this statement. It may match other spans too; a later step
chooses between them.

- "regex" uses RE2 syntax (no lookaround or backreferences) and is at most {max_length}
  characters.
- "group" is the capture group holding the value's text (0 for the whole match).
- "normalise" turns that text into the value, using only these steps, in order:
{normalisers}"""
"""The default synthesis prompt. Placeholders: ``name``, ``description``, ``type`` (with
the unit, if any), ``statement``, ``section`` (``"Section: ...\\n"`` or empty), ``value``,
``evidence``, ``max_length`` and ``normalisers`` (:data:`NORMALISERS`)."""


class GeneratorDraft(BaseModel):
    """What ``generator_llm`` writes: a spec's ``match`` and ``normalise`` parts.

    Kept loose so a bad pattern reaches :meth:`GeneratorSpec.parse` and is reported as an
    invalid spec, rather than failing the LLM call.
    """

    regex: str = Field(description="An RE2 pattern that finds the value")
    group: int = Field(default=0, description="The capture group holding the value's text")
    normalise: Annotated[list[Any], WithJsonSchema(NORMALISE_JSON_SCHEMA)] = Field(
        default_factory=list[Any]
    )


def draft_spec(draft: GeneratorDraft, field: str, example_id: str) -> GeneratorSpec:
    """The spec for a draft: its id is a hash of what it does, so a second draft of the
    same generator gets the same id. Raises :class:`InvalidGeneratorError`."""
    body = json.dumps(
        [field, draft.regex, draft.group, draft.normalise], sort_keys=True, default=str
    )
    return GeneratorSpec.parse(
        {
            "id": f"gen-{hashlib.sha256(body.encode()).hexdigest()[:12]}",
            "field": field,
            "match": {"regex": draft.regex, "group": draft.group},
            "normalise": draft.normalise,
            "provenance": {
                "learned_from": [example_id],
                "synthesised_by": "generator_llm",
                "created": datetime.now(UTC).date(),
            },
        }
    )


# --- outcomes ----------------------------------------------------------------------------

LearnStatus = Literal[
    "accepted",
    "covered",
    "unlearnable",
    "budget",
    "llm_error",
    "jev_error",
    "invalid_spec",
    "missed_trigger",
    "regressed",
]
"""``accepted``: published. ``covered``: the generators in use already find the value.
``unlearnable``: no registered candidate field, or an example whose value doesn't fit it.
``budget``: the run budget (or a process cap) refused the LLM or Jev. ``llm_error``,
``jev_error``: a call failed. ``invalid_spec``: the draft didn't validate.
``missed_trigger``: it didn't give the value on the triggering statement.
``regressed``: it lowered accuracy on the field's stored examples."""


class LearnOutcome(BaseModel):
    """What became of one queued example."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    field: str
    status: LearnStatus
    message: str = ""
    spec: GeneratorSpec | None = None
    """The spec that was tested (accepted or not)."""
    snapshot: int | None = None
    """The snapshot version that published it (``accepted`` only)."""


class _Rejected(Exception):
    """Ends one example's learning with an outcome."""

    def __init__(self, status: LearnStatus, message: str, spec: GeneratorSpec | None = None):
        super().__init__(message)
        self.status: LearnStatus = status
        self.message = message
        self.spec = spec


# --- the learner ---------------------------------------------------------------------------


@dataclass
class GeneratorLearner:
    """The default :class:`~jevex.interfaces.Learner`: synthesise, test and hot-swap.

    ``schemas`` are the extractor's (an example's ``field`` is ``"Schema.field"``, or
    ``"Parent.nested_field.field"`` for a nested model's field).
    ``base``, ``locale``, ``selector``, ``normalisers`` and ``fallback_threshold`` should
    be what the candidate, select, normalise and fallback stages use, so a generator is
    tested as documents will run it. ``ledger`` applies the run budget to
    the learner's LLM and Jev calls and records their spend.
    """

    schemas: Sequence[SchemaSpec]
    llm: LLM
    jev: JevClient
    generators: LearnedGenerators = field(default_factory=LearnedGenerators)
    ledger: RunLedger = field(default_factory=RunLedger)
    base: GeneratorRegistry = field(default_factory=default_registry)
    selector: CandidateSelector = field(default_factory=JevCandidateSelector)
    learn_threshold: float = LEARN_THRESHOLD
    sample_size: int = SAMPLE_SIZE
    normalisers: NormaliserRegistry = BUILTIN_NORMALISERS
    locale: str | None = None
    fallback_threshold: float = FALLBACK_THRESHOLD
    """Jev must choose the value on the triggering statement at least this confidently,
    or the fallback would still ask the LLM there."""
    prompt: str = PROMPT
    outcomes: list[LearnOutcome] = field(default_factory=list[LearnOutcome])
    _queue: asyncio.Queue[VerifiedExample] | None = field(default=None, init=False, repr=False)
    _worker: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for label in ("learn_threshold", "fallback_threshold"):
            value = getattr(self, label)
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be between 0 and 1, got {value}")
        if self.sample_size < 0:
            raise ValueError(f"sample_size must not be negative, got {self.sample_size}")

    @property
    def snapshot(self) -> GeneratorSnapshot:
        return self.generators.current

    def wants(self, example: VerifiedExample) -> bool:
        """Whether ``example`` is learned from: a human's, or verified with
        ``probability >= learn_threshold``."""
        return _wants(example, self.learn_threshold)

    async def submit(self, example: VerifiedExample) -> None:
        """Store and queue ``example`` if :meth:`wants` it; return without waiting."""
        self._raise_error()
        if not self.wants(example):
            return
        if self.generators.store is not None:
            await self.generators.store.add_example(example)
        self._ensure_worker().put_nowait(example)

    async def drain(self) -> None:
        """Wait until every queued example has its outcome."""
        self._raise_error()
        if self._queue is not None and self._loop is asyncio.get_running_loop():
            join = asyncio.ensure_future(self._queue.join())
            assert self._worker is not None
            # The worker can die before the queue empties; then join() would never return.
            await asyncio.wait({join, self._worker}, return_when=asyncio.FIRST_COMPLETED)
            join.cancel()
        self._raise_error()

    async def aclose(self) -> None:
        """Stop the worker. Examples still queued are dropped (they stay in the store)."""
        worker, self._worker, self._queue, self._loop = self._worker, None, None, None
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                if not worker.cancelled():
                    raise  # this task was cancelled, not just the worker
        self._raise_error()

    async def learn(self, example: VerifiedExample) -> LearnOutcome:
        """Learn from one example now (the worker calls this for each queued one)."""
        jev = self.jev.metered()
        try:
            outcome = await self._learn(example, jev)
        except _Rejected as exc:
            outcome = LearnOutcome(
                example_id=example.id,
                field=example.field,
                status=exc.status,
                message=exc.message,
                spec=exc.spec,
            )
        finally:
            await self.ledger.record("jev", jev.usage.cost)
        self.outcomes.append(outcome)
        return outcome

    # -- the steps ----------------------------------------------------------------------

    async def _learn(self, example: VerifiedExample, jev: JevClient) -> LearnOutcome:
        schema, spec = self._field(example)
        statement = example_statement(example)
        expected = self._expected(example, spec)
        source = example.document_source
        current = self.snapshot.on(self.base)
        if self._finds(self._generate(current, statement, spec, schema, source), spec, expected):
            raise _Rejected("covered", "the generators in use already find the value")
        draft = await self._synthesise(example, statement, spec)
        try:
            generator_spec = draft_spec(draft, example.field, example.id)
        except InvalidGeneratorError as exc:
            raise _Rejected("invalid_spec", str(exc)) from None
        generator = generator_spec.to_generator()
        if not self._finds(generator.generate(statement), spec, expected):
            raise _Rejected("missed_trigger", "it finds no span with the value", generator_spec)
        if await self.ledger.refuse_document() is not None:
            raise _Rejected("budget", "the run's Jev spend cap is reached", generator_spec)
        candidate = current.with_generator(generator)
        try:
            if not await self._chosen(jev, statement, spec, schema, source, candidate, expected):
                raise _Rejected("missed_trigger", "Jev doesn't choose its value", generator_spec)
            await self._regression(example, jev, spec, schema, current, candidate, generator_spec)
        except JevBudgetExceededError as exc:
            raise _Rejected("budget", str(exc), generator_spec) from None
        except JevError as exc:
            raise _Rejected("jev_error", f"{type(exc).__name__}: {exc}", generator_spec) from None
        snapshot = await self.generators.publish(generator_spec)
        return LearnOutcome(
            example_id=example.id,
            field=example.field,
            status="accepted",
            spec=generator_spec,
            snapshot=snapshot.version,
        )

    @cached_property
    def all_schemas(self) -> tuple[SchemaSpec, ...]:
        """``schemas`` and their nested models' specs (:meth:`~jevex.SchemaSpec.children`,
        named ``"Parent.nested_field"``): every schema an example's ``field`` can name."""
        return (*self.schemas, *(child for s in self.schemas for child in s.children()))

    def _field(self, example: VerifiedExample) -> tuple[str, FieldSpec]:
        schema, _, name = example.field.rpartition(".")
        spec = next((s for s in self.all_schemas if s.name == schema), None)
        found = next((f for f in spec.fields if f.name == name), None) if spec else None
        if found is None or not found.needs_candidates:
            raise _Rejected("unlearnable", f"{example.field} isn't a candidate field")
        return schema, found

    async def _synthesise(
        self, example: VerifiedExample, statement: Statement, spec: FieldSpec
    ) -> GeneratorDraft:
        section = section_text(statement.heading_trail)
        value = json.dumps(example.value, default=str, ensure_ascii=False)
        evidence = example.evidence
        prompt = self.prompt.format(
            name=spec.name,
            description=spec.description,
            type=field_type_text(spec) + (f", in {spec.unit}" if spec.unit else ""),
            statement=statement.text,
            section=f"Section: {section}\n" if section else "",
            value=value,
            evidence=statement.text[evidence[0] : evidence[1]] if evidence else value,
            max_length=MAX_PATTERN_LENGTH,
            normalisers=NORMALISERS,
        )
        try:
            response = await self.ledger.call_llm(self.llm, prompt, GeneratorDraft)
        except LLMError as exc:
            raise _Rejected("llm_error", f"{type(exc).__name__}: {exc}") from None
        if response is None:
            raise _Rejected("budget", "the run budget refused the generator_llm call")
        return response.output

    async def _chosen(
        self,
        jev: JevClient,
        statement: Statement,
        spec: FieldSpec,
        schema: str,
        source: str | None,
        registry: GeneratorRegistry,
        expected: Any,
    ) -> bool:
        candidates = self._generate(registry, statement, spec, schema, source)
        questions = self.selector.questions(statement, spec, candidates)
        answers = await jev.ask(statement_state(statement), questions)
        selection = self._right(spec, candidates, answers, expected)
        return selection is not None and selection.confidence >= self.fallback_threshold

    async def _regression(
        self,
        example: VerifiedExample,
        jev: JevClient,
        spec: FieldSpec,
        schema: str,
        old: GeneratorRegistry,
        new: GeneratorRegistry,
        generator_spec: GeneratorSpec,
    ) -> None:
        """Raise ``regressed`` if ``new`` gets fewer of the stored examples right."""
        if self.sample_size == 0 or self.generators.store is None:
            return
        stored = await self.generators.store.examples(example.field, limit=self.sample_size + 1)
        cases: list[tuple[Statement, str | None, Any]] = []
        for other in stored:
            if other.id == example.id or len(cases) == self.sample_size:
                continue
            try:
                expected = self._expected(other, spec)
            except _Rejected:
                continue  # a stored example this field can no longer read tests nothing
            cases.append((example_statement(other), other.document_source, expected))
        results = await gather(
            self._compare(jev, statement, spec, schema, source, old, new, expected)
            for statement, source, expected in cases
        )
        changed = [r for r in results if r is not None]
        before = sum(old_right for old_right, _ in changed)
        after = sum(new_right for _, new_right in changed)
        if after < before:
            raise _Rejected(
                "regressed",
                f"right on {after} of the {len(changed)} stored examples it changes, "
                f"down from {before}",
                generator_spec,
            )

    async def _compare(
        self,
        jev: JevClient,
        statement: Statement,
        spec: FieldSpec,
        schema: str,
        source: str | None,
        old: GeneratorRegistry,
        new: GeneratorRegistry,
        expected: Any,
    ) -> tuple[bool, bool] | None:
        """Whether the old and the new candidates each lead to ``expected``; ``None`` when
        the generator doesn't change the candidates (Jev isn't asked). Both sets go in one
        request."""
        before = self._generate(old, statement, spec, schema, source)
        after = self._generate(new, statement, spec, schema, source)
        if unique_spans(before).keys() == unique_spans(after).keys():
            return None
        old_q = self.selector.questions(statement, spec, before)
        new_q = self.selector.questions(statement, spec, after)
        replies = await jev.ask(
            statement_state(statement),
            {
                **{f"old/{k}": q for k, q in old_q.items()},
                **{f"new/{k}": q for k, q in new_q.items()},
            },
        )
        return (
            self._right(spec, before, _strip(replies, "old/"), expected) is not None,
            self._right(spec, after, _strip(replies, "new/"), expected) is not None,
        )

    def _right(
        self,
        spec: FieldSpec,
        candidates: list[Candidate],
        answers: dict[str, Answer],
        expected: Any,
    ) -> Selection | None:
        """The selection, if it picked ``expected``."""
        if not answers:
            return None
        selection = self.selector.selection(spec, candidates, answers)
        picks = selection.accepted if spec.many else []
        if not picks and selection.candidate is not None:
            picks = [selection.candidate]
        return selection if self._finds(picks, spec, expected) else None

    def _generate(
        self,
        registry: GeneratorRegistry,
        statement: Statement,
        spec: FieldSpec,
        schema: str,
        source: str | None,
    ) -> list[Candidate]:
        """The candidates a document from ``source`` (an example's ``document_source``)
        would get: source-scoped generators run only when it matches."""
        return registry.generate(statement, spec, schema=schema, locale=self.locale, source=source)

    def _expected(self, example: VerifiedExample, spec: FieldSpec) -> Any:
        """The example's value as the field types it (a stored date comes back a string)."""
        try:
            return normalise(example.value, [], spec, registry=self.normalisers)
        except NormaliseError as exc:
            raise _Rejected("unlearnable", f"its value doesn't fit {spec.name}: {exc}") from None

    def _finds(self, candidates: list[Candidate], spec: FieldSpec, expected: Any) -> bool:
        """Whether any candidate normalises to ``expected`` (or, for a list field, to a
        list holding it)."""
        for candidate in candidates:
            try:
                value = normalise(
                    candidate.raw, candidate.normalise, spec, registry=self.normalisers
                )
            except NormaliseError:
                continue
            if value == expected or (spec.many and isinstance(value, list) and expected in value):
                return True
        return False

    # -- the worker -----------------------------------------------------------------------

    def _ensure_worker(self) -> asyncio.Queue[VerifiedExample]:
        loop = asyncio.get_running_loop()
        if self._queue is None or self._worker is None or self._loop is not loop:
            # A new event loop (asyncio.run per batch) can't use the old loop's queue.
            self._queue = asyncio.Queue()
            self._loop = loop
            self._worker = loop.create_task(self._work(self._queue))
        return self._queue

    async def _work(self, queue: asyncio.Queue[VerifiedExample]) -> None:
        while True:
            item = await queue.get()
            try:
                await self.learn(item)
            except Exception as exc:
                # Not an expected failure (those are outcomes): stop, and let the next
                # submit, drain or aclose raise it.
                self._error = exc
                return
            finally:
                queue.task_done()

    def _raise_error(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            self._worker, self._queue, self._loop = None, None, None
            raise RuntimeError("the learner worker failed") from error


def _strip(answers: dict[str, Answer], prefix: str) -> dict[str, Answer]:
    return {k.removeprefix(prefix): a for k, a in answers.items() if k.startswith(prefix)}


_KINDS: frozenset[str] = frozenset(get_args(StatementKind))
_LOCATION = DomLocation(dom_path="")


def example_statement(example: VerifiedExample) -> Statement:
    """The example's statement, rebuilt with the heading trail and kind it was seen with
    (what the learner tests generators on)."""
    kind = example.context.get("kind")
    trail: object = example.context.get("heading_trail")
    headings = (
        [h for h in cast("list[object]", trail) if isinstance(h, str)]
        if isinstance(trail, list)
        else []
    )
    return Statement(
        id=example.id,
        text=example.statement,
        kind=cast("StatementKind", kind) if kind in _KINDS else "sentence",
        component_id="learn",
        heading_trail=headings,
        location=_LOCATION,
    )


# --- modes and batch compiles -------------------------------------------------------------

LearnMode = Literal["inline", "compile", "hybrid"]
"""When learning runs (spec: *Learning loop*). ``inline`` (default): the extractor's
:class:`GeneratorLearner` learns in the background and documents use what it accepts.
``compile``: documents only log verified examples (:class:`ExampleLogger`); ``jevex
learn`` (:func:`compile_pack`) synthesises and tests them in a batch and writes a pack
diff for review. ``hybrid``: documents learn inline into the store (the local layer), and
periodic ``jevex learn`` runs compile that layer, plus anything still unlearned, into a
reviewable pack diff."""


def _wants(example: VerifiedExample, learn_threshold: float) -> bool:
    if example.source == "human":
        return True
    return example.probability is not None and example.probability >= learn_threshold


@dataclass
class ExampleLogger:
    """The ``compile`` mode's :class:`~jevex.interfaces.Learner`: stores the examples a
    :class:`GeneratorLearner` would learn from, and learns nothing.

    :func:`compile_pack` (``jevex learn``) learns from them later, in a batch.
    """

    store: Store
    learn_threshold: float = LEARN_THRESHOLD

    def __post_init__(self) -> None:
        if not 0 <= self.learn_threshold <= 1:
            raise ValueError(f"learn_threshold must be between 0 and 1, got {self.learn_threshold}")

    def wants(self, example: VerifiedExample) -> bool:
        """The same rule as :meth:`GeneratorLearner.wants`."""
        return _wants(example, self.learn_threshold)

    async def submit(self, example: VerifiedExample) -> None:
        """Store ``example`` if :meth:`wants` it."""
        if self.wants(example):
            await self.store.add_example(example)


class PackDiff(BaseModel):
    """What a batch compile proposes adding to a pack, for review.

    ``generators``: the local layer's generators the pack doesn't have (what ``hybrid``
    mode learned inline), then the ones this compile accepted. ``outcomes``: what became
    of each example it learned from.
    """

    model_config = ConfigDict(frozen=True)

    generators: list[GeneratorSpec]
    outcomes: list[LearnOutcome]

    def write(self, directory: Path) -> list[Path]:
        """Write each generator to ``directory/generators/<id>.yaml`` and return the paths.

        ``directory`` must not exist or be empty, so a review never picks up files from an
        earlier diff (:class:`FileExistsError` otherwise).
        """
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f"{directory} already exists and isn't an empty directory")
        folder = directory / PACK_GENERATORS
        folder.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for spec in self.generators:
            path = folder / f"{spec.id}.yaml"
            path.write_text(spec.to_yaml(), encoding="utf-8")
            paths.append(path)
        return paths


async def compile_pack(learner: GeneratorLearner, pack: Sequence[GeneratorSpec] = ()) -> PackDiff:
    """Learn from the store's verified examples in one batch (``jevex learn``) and return
    what to add to ``pack``.

    ``learner.generators`` must be ``LearnedGenerators(store, persist=False)``: the store
    gives the examples and the local layer's generators, and nothing is published to it.
    The learner runs on the store's generators plus ``pack``'s (less those the store
    disables), so an example they already find costs nothing, and each generator accepted
    is in use for the examples after it. It learns from the candidate fields' examples
    (nested models' fields too: :attr:`~GeneratorLearner.all_schemas`) that it
    :meth:`~GeneratorLearner.wants`, oldest first. Running it again is safe: covered
    examples are skipped without an LLM call, but ones rejected before are tried again.
    """
    learned = learner.generators
    store = learned.store
    if store is None or learned.persist:
        raise ValueError(
            "compile_pack needs a learner with LearnedGenerators(store, persist=False)"
        )
    disabled = await store.disabled_generator_ids()
    in_pack = {spec.id for spec in pack}
    stored = await learned.stored()
    await learned.load(also=[spec for spec in pack if spec.id not in disabled])
    outcomes: list[LearnOutcome] = []
    for schema in learner.all_schemas:
        for spec in schema.fields:
            if not spec.needs_candidates:
                continue
            for example in reversed(await store.examples(f"{schema.name}.{spec.name}")):
                if learner.wants(example):
                    outcomes.append(await learner.learn(example))
    added: dict[str, GeneratorSpec] = {s.id: s for s in stored if s.id not in in_pack}
    for outcome in outcomes:
        if outcome.status == "accepted" and outcome.spec and outcome.spec.id not in in_pack:
            added.setdefault(outcome.spec.id, outcome.spec)
    return PackDiff(generators=list(added.values()), outcomes=outcomes)


# --- the stage ---------------------------------------------------------------------------


@dataclass
class LearnStage:
    """Hands the document's verified examples (``ctx.verified``) to the learner, then its
    generator counts to the housekeeper (:mod:`jevex.housekeeping`).

    ``learner=None`` uses ``ctx.learner`` (the extractor's: a :class:`GeneratorLearner`
    with a ``generator_llm``, or an :class:`ExampleLogger` in ``compile`` mode), and
    ``housekeeper=None`` uses ``ctx.housekeeper`` (the extractor's, when it has a store).
    Without either, that part is skipped.
    """

    learner: Learner | None = None
    housekeeper: Housekeeper | None = None
    name: str = "learn"

    async def run(self, ctx: Context) -> None:
        learner = self.learner if self.learner is not None else ctx.learner
        if learner is not None:
            for example in ctx.verified:
                await learner.submit(example)
        housekeeper = self.housekeeper if self.housekeeper is not None else ctx.housekeeper
        if housekeeper is not None:
            await housekeeper.record(ctx)

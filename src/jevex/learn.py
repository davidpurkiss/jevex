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
   keeps it to the end; later documents get the new one.

Every example ends in a :class:`LearnOutcome` on :attr:`GeneratorLearner.outcomes`.
Expected failures (a budget saying no, an LLM or Jev error, an invalid spec, a failed
test) are outcomes, not exceptions. Anything else stops the worker; the error is raised
by the next :meth:`~GeneratorLearner.submit`, :meth:`~GeneratorLearner.drain` or
:meth:`~GeneratorLearner.aclose`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from jevex._tasks import gather
from jevex.budgets import RunLedger
from jevex.fallback import field_type_text
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
from jevex.normalise import NormaliseError, normalise
from jevex.select import JevCandidateSelector, statement_state, unique_spans
from jevex.statements import Statement, StatementKind
from jevex.store import GeneratorRecord, StoreError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jevex.interfaces import CandidateGenerator, CandidateSelector, Learner
    from jevex.jev import Answer, JevClient
    from jevex.llm import LLM
    from jevex.pipeline import Context
    from jevex.schema import FieldSpec, SchemaSpec
    from jevex.statements import Candidate
    from jevex.store import Store, VerifiedExample

LEARN_THRESHOLD = 0.9
"""Verification probability at or above which an LLM answer is learned from. Provisional:
the spec's open questions set the defaults from eval runs."""
SAMPLE_SIZE = 20
"""How many of a field's stored examples a new generator is tested on (all, if fewer)."""


# --- snapshots ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorSnapshot:
    """The learned generators at one moment. Never changes: publishing makes a new one.

    ``version`` counts publishes in this process (0: what the store held at start).
    """

    version: int
    registry: GeneratorRegistry

    def on(self, base: GeneratorRegistry) -> GeneratorRegistry:
        """``base`` followed by these generators (``base`` wins ties and keeps its ids)."""
        return base.extended(self.registry)


class LearnedGenerators:
    """Holds the current :class:`GeneratorSnapshot`, loaded from and published to a store.

    ``store`` is the local learned layer: :meth:`load` reads its enabled generators, and
    :meth:`publish` writes there before swapping the snapshot. Generators other processes
    publish later aren't picked up until the next load.
    """

    def __init__(self, store: Store | None = None) -> None:
        self.store = store
        self.current = GeneratorSnapshot(0, GeneratorRegistry())

    async def load(self) -> GeneratorSnapshot:
        """Replace the snapshot with the store's enabled generators (oldest first).

        Raises :class:`~jevex.store.StoreError` for a stored spec that doesn't validate.
        """
        if self.store is None:
            return self.current
        generators: list[CandidateGenerator] = []
        for record in await self.store.generators():
            try:
                generators.append(GeneratorSpec.parse(record.spec).to_generator())
            except InvalidGeneratorError as exc:
                raise StoreError(f"stored generator {record.id!r} is invalid: {exc}") from exc
        self.current = GeneratorSnapshot(self.current.version, GeneratorRegistry(generators))
        return self.current

    async def publish(self, spec: GeneratorSpec) -> GeneratorSnapshot:
        """Store ``spec`` and make a new snapshot holding it (replacing one with its id)."""
        if self.store is not None:
            scope = {"locale": spec.scope.locale} if spec.scope.locale else {}
            await self.store.put_generator(
                GeneratorRecord(id=spec.id, field=spec.field, spec=spec.to_data(), scope=scope)
            )
        self.current = GeneratorSnapshot(
            self.current.version + 1, self.current.registry.with_generator(spec.to_generator())
        )
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
Description: {label}
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
"""The default synthesis prompt. Placeholders: ``name``, ``label``, ``type`` (with the
unit, if any), ``statement``, ``section`` (``"Section: ...\\n"`` or empty), ``value``,
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

    ``schemas`` are the extractor's (an example's ``field`` is ``"Schema.field"``).
    ``base`` and ``selector`` should be what the candidate and select stages use, so a
    generator is tested as documents will run it. ``ledger`` applies the run budget to
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
    prompt: str = PROMPT
    outcomes: list[LearnOutcome] = field(default_factory=list[LearnOutcome])
    _queue: asyncio.Queue[object] | None = field(default=None, init=False, repr=False)
    _worker: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.learn_threshold <= 1:
            raise ValueError(f"learn_threshold must be between 0 and 1, got {self.learn_threshold}")
        if self.sample_size < 0:
            raise ValueError(f"sample_size must not be negative, got {self.sample_size}")

    @property
    def snapshot(self) -> GeneratorSnapshot:
        return self.generators.current

    def wants(self, example: VerifiedExample) -> bool:
        """Whether ``example`` is learned from: a human's, or verified with
        ``probability >= learn_threshold``."""
        if example.source == "human":
            return True
        return example.probability is not None and example.probability >= self.learn_threshold

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
        statement = _statement(example)
        expected = _expected(example, spec)
        current = self.snapshot.on(self.base)
        if _finds(current.generate(statement, spec, schema=schema), spec, expected):
            raise _Rejected("covered", "the generators in use already find the value")
        draft = await self._synthesise(example, statement, spec)
        try:
            generator_spec = draft_spec(draft, example.field, example.id)
        except InvalidGeneratorError as exc:
            raise _Rejected("invalid_spec", str(exc)) from None
        generator = generator_spec.to_generator()
        if not _finds(generator.generate(statement), spec, expected):
            raise _Rejected("missed_trigger", "it finds no span with the value", generator_spec)
        if await self.ledger.refuse_document() is not None:
            raise _Rejected("budget", "the run's Jev spend cap is reached", generator_spec)
        candidate = current.with_generator(generator)
        try:
            if not await self._chosen(jev, statement, spec, schema, candidate, expected):
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

    def _field(self, example: VerifiedExample) -> tuple[str, FieldSpec]:
        schema, _, name = example.field.partition(".")
        spec = next((s for s in self.schemas if s.name == schema), None)
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
            label=spec.label,
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
        registry: GeneratorRegistry,
        expected: Any,
    ) -> bool:
        candidates = registry.generate(statement, spec, schema=schema)
        questions = self.selector.questions(statement, spec, candidates)
        answers = await jev.ask(statement_state(statement), questions)
        return self._right(spec, candidates, answers, expected)

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
        cases: list[tuple[Statement, Any]] = []
        for other in stored:
            if other.id == example.id or len(cases) == self.sample_size:
                continue
            try:
                cases.append((_statement(other), _expected(other, spec)))
            except _Rejected:
                continue  # a stored example this field can no longer read tests nothing
        results = await gather(
            self._compare(jev, statement, spec, schema, old, new, expected)
            for statement, expected in cases
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
        old: GeneratorRegistry,
        new: GeneratorRegistry,
        expected: Any,
    ) -> tuple[bool, bool] | None:
        """Whether the old and the new candidates each lead to ``expected``; ``None`` when
        the generator doesn't change the candidates (Jev isn't asked). Both sets go in one
        request."""
        before = old.generate(statement, spec, schema=schema)
        after = new.generate(statement, spec, schema=schema)
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
            self._right(spec, before, _strip(replies, "old/"), expected),
            self._right(spec, after, _strip(replies, "new/"), expected),
        )

    def _right(
        self,
        spec: FieldSpec,
        candidates: list[Candidate],
        answers: dict[str, Answer],
        expected: Any,
    ) -> bool:
        if not answers:
            return False
        selection = self.selector.selection(spec, candidates, answers)
        picks = selection.accepted if spec.many else []
        if not picks and selection.candidate is not None:
            picks = [selection.candidate]
        return _finds(picks, spec, expected)

    # -- the worker -----------------------------------------------------------------------

    def _ensure_worker(self) -> asyncio.Queue[object]:
        loop = asyncio.get_running_loop()
        if self._queue is None or self._worker is None or self._loop is not loop:
            # A new event loop (asyncio.run per batch) can't use the old loop's queue.
            self._queue = asyncio.Queue()
            self._loop = loop
            self._worker = loop.create_task(self._work(self._queue))
        return self._queue

    async def _work(self, queue: asyncio.Queue[object]) -> None:
        while True:
            item = await queue.get()
            try:
                await self.learn(cast("VerifiedExample", item))
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


def _statement(example: VerifiedExample) -> Statement:
    """The example's statement, rebuilt with the heading trail and kind it was seen with."""
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


def _expected(example: VerifiedExample, spec: FieldSpec) -> Any:
    """The example's value as the field types it (a stored date comes back a string)."""
    try:
        return normalise(example.value, [], spec)
    except NormaliseError as exc:
        raise _Rejected("unlearnable", f"its value doesn't fit {spec.name}: {exc}") from None


def _finds(candidates: list[Candidate], spec: FieldSpec, expected: Any) -> bool:
    """Whether any candidate normalises to ``expected`` (or, for a list field, to a list
    holding it)."""
    for candidate in candidates:
        try:
            value = normalise(candidate.raw, candidate.normalise, spec)
        except NormaliseError:
            continue
        if value == expected or (spec.many and isinstance(value, list) and expected in value):
            return True
    return False


# --- the stage ---------------------------------------------------------------------------


@dataclass
class LearnStage:
    """Hands the document's verified examples (``ctx.verified``) to the learner.

    ``learner=None`` uses ``ctx.learner`` (set by an extractor with a ``generator_llm``);
    with neither, the stage does nothing.
    """

    learner: Learner | None = None
    name: str = "learn"

    async def run(self, ctx: Context) -> None:
        learner = self.learner if self.learner is not None else ctx.learner
        if learner is None:
            return
        for example in ctx.verified:
            await learner.submit(example)

"""LLM fallback extraction, verified by Jev (spec: *Value extraction*, steps 5 to 7; stage 14).

Opt-in: the stage does nothing unless an LLM is configured, as
``Extractor(extraction_llm=...)`` or ``FallbackStage(extractor=...)``. After selection and
normalising, it looks at every categorised statement of a candidate field (numbers,
dates, strings; enums and bools are answered directly) and asks the LLM when:

- no candidates were generated,
- Jev chose "none", but the statement was categorised as the field with probability
  ``>= category_threshold``, or
- the selection's confidence is below ``fallback_threshold``.

A field another route already found with confidence ``>= fallback_threshold`` (or with
no confidence, such as embedded data) is left alone, so nothing is asked about it.

The LLM must return the value **and** the verbatim evidence: words of the statement that
state it. An answer whose evidence isn't in the statement, or whose value doesn't fit the
field, is dropped. The rest are verified with one Jev Noul each (the field's
:meth:`~jevex.schema.FieldSpec.verify_question`; for ``list[...]`` fields, its
:meth:`~jevex.schema.FieldSpec.member_question` per item), with every check about one
statement in one request:

- ``p >= verify_threshold``: the value is recorded with ``method="llm"``,
  ``verified=True`` and ``confidence=p``, replacing the low-confidence Jev answer (which
  becomes an alternative). It is also added to ``Context.verified`` for learning (#38).
- Otherwise the answer is discarded: the field keeps the best Jev answer, which is
  low-confidence by construction (or stays empty), and the rejected value is listed in
  its ``alternatives`` with its verification probability.

Every dropped or rejected answer is reported as a ``fallback`` event. LLM calls go
through ``ctx.budget.call_llm``, so when a budget is hit the remaining fields keep their
Jev answers and the hit is in ``meta.budget_events``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field, create_model

from jevex._tasks import gather
from jevex.budgets import DocumentBudget
from jevex.interfaces import LLMAnswer
from jevex.jev import NoulAnswer
from jevex.layout import section_text
from jevex.llm import LLMError
from jevex.normalise import NormaliseError, normalise
from jevex.results import Alternative, FieldMeta, Source
from jevex.select import field_statements, statement_state, unique_spans
from jevex.statements import Span
from jevex.store import VerifiedExample

if TYPE_CHECKING:
    from jevex.interfaces import LLMExtractor
    from jevex.jev import Question
    from jevex.llm import LLM
    from jevex.pipeline import Context, SchemaRun
    from jevex.schema import FieldSpec
    from jevex.statements import Statement

CATEGORY_THRESHOLD = 0.5
"""Category probability at or above which Jev's "none" sends the statement to the LLM.
Provisional: the spec's open questions set the defaults from eval runs."""
FALLBACK_THRESHOLD = 0.5
"""Selection confidence below which the LLM is asked. Provisional, like the others."""
VERIFY_THRESHOLD = 0.8
"""Verification probability at or above which an LLM answer is accepted. Provisional."""

Trigger = Literal["no_candidates", "none_chosen", "low_confidence"]


# --- the default extractor -------------------------------------------------------------

PROMPT = """\
Read one statement from a document and extract a single field from it.

{section}Statement: {statement}

Field: {name}
Description: {label}
Type: {type}

If the statement states the {phrase}, set "stated" to true, give the value{unit}, and copy
the words of the statement that state it into "evidence", exactly as they appear. If it
doesn't state it, set "stated" to false and leave the rest empty. Never guess."""
"""The default prompt. Placeholders: ``section`` (``"Section: ...\\n"`` or empty),
``statement``, ``name``, ``label``, ``phrase``, ``type`` and ``unit`` (``" in <unit>"`` or
empty)."""


class LLMOutput(BaseModel):
    """What the default extractor asks the LLM for. Each field gets a subclass with
    ``value`` typed for it, so structured output constrains the value too."""

    stated: bool = Field(description="Whether the statement states the field")
    value: Any = None
    evidence: str = Field(
        default="", description="The exact words of the statement that state the value"
    )


_OUTPUTS: dict[tuple[str, bool, tuple[str, ...]], type[LLMOutput]] = {}


def output_model(spec: FieldSpec) -> type[LLMOutput]:
    """The structured-output model for one field: :class:`LLMOutput` with ``value`` as a
    JSON-friendly version of the field's type (dates are ISO strings)."""
    key = (spec.kind, spec.many, spec.options)
    if key not in _OUTPUTS:
        item: Any = {"number": float, "date": str, "str": str, "bool": bool}.get(spec.kind, str)
        if spec.kind == "enum":
            item = Literal[spec.options]  # pyright: ignore[reportInvalidTypeForm]
        value_type: Any = list[item] if spec.many else item
        _OUTPUTS[key] = create_model(
            f"LLMOutput_{spec.kind}{'_list' if spec.many else ''}",
            __base__=LLMOutput,
            value=(value_type | None, None),
        )
    return _OUTPUTS[key]


def _type_text(spec: FieldSpec) -> str:
    item = {
        "number": "a number",
        "date": "a date, as YYYY-MM-DD",
        "str": "text",
        "bool": "true or false",
        "enum": "one of " + ", ".join(f'"{o}"' for o in spec.options),
    }.get(spec.kind, "text")
    return f"a list, each item {item}" if spec.many else item


@dataclass(frozen=True)
class LLMFieldExtractor:
    """The default :class:`~jevex.interfaces.LLMExtractor`: one structured-output call per
    statement and field, with the statement, its heading trail and the field's name,
    description, unit and type. Returns ``None`` when the LLM says the statement doesn't
    state the field, or a budget refuses the call."""

    llm: LLM
    prompt: str = PROMPT

    async def extract(
        self, statement: Statement, field: FieldSpec, budget: DocumentBudget
    ) -> LLMAnswer | None:
        section = section_text(statement.heading_trail)
        prompt = self.prompt.format(
            section=f"Section: {section}\n" if section else "",
            statement=statement.text,
            name=field.name,
            label=field.label,
            phrase=field.phrase,
            type=_type_text(field),
            unit=f" in {field.unit}" if field.unit else "",
        )
        response = await budget.call_llm(self.llm, prompt, output_model(field))
        if response is None:
            return None
        out = response.output
        if not out.stated or out.value is None or out.value == []:
            return None
        return LLMAnswer(value=out.value, evidence=out.evidence)


# --- the stage -------------------------------------------------------------------------


@dataclass
class _Ask:
    """One (schema, field) about one statement: why it falls back and who wants it."""

    run: SchemaRun
    spec: FieldSpec
    statement: Statement
    trigger: Trigger
    scopes: list[str] = field(default_factory=list[str])
    answer: LLMAnswer | None = None
    values: list[Any] = field(default_factory=list[Any])
    """The answer's value, validated against the field: one item, or a list's items."""
    span: Span | None = None
    item_p: dict[int, float] = field(default_factory=dict[int, float])
    """Verification probability per item of ``values``."""
    kept: list[Any] = field(default_factory=list[Any])
    """The items Jev verified."""
    p: float = 0.0
    """The least certain kept item's verification probability."""
    rejected: list[Alternative] = field(default_factory=list[Alternative])

    @property
    def prefix(self) -> str:
        return f"{self.run.name}.{self.spec.name}/"

    def questions(self) -> dict[str, Question]:
        if self.spec.many:
            return {
                f"{self.prefix}member{i}": self.spec.member_question(v)
                for i, v in enumerate(self.values)
            }
        return {f"{self.prefix}verify": self.spec.verify_question(self.values[0])}


@dataclass
class FallbackStage:
    """Asks the LLM where Jev's selection failed, and keeps only what Jev verifies."""

    extractor: LLMExtractor | None = None
    """``None``: an :class:`LLMFieldExtractor` over ``ctx.extraction_llm``, if one is set."""
    category_threshold: float = CATEGORY_THRESHOLD
    fallback_threshold: float = FALLBACK_THRESHOLD
    verify_threshold: float = VERIFY_THRESHOLD
    name: str = "fallback"

    def __post_init__(self) -> None:
        for label in ("category_threshold", "fallback_threshold", "verify_threshold"):
            value = getattr(self, label)
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be between 0 and 1, got {value}")

    async def run(self, ctx: Context) -> None:
        extractor = self.extractor
        if extractor is None and ctx.extraction_llm is not None:
            extractor = LLMFieldExtractor(ctx.extraction_llm)
        if extractor is None:
            return
        asks = self._plan(ctx)
        if not asks:
            return
        budget = ctx.budget or DocumentBudget()
        answers = await gather(self._extract(ctx, extractor, a, budget) for a in asks)
        for ask, answer in zip(asks, answers, strict=True):
            if answer is not None:
                self._check(ctx, ask, answer)
        checked = [a for a in asks if a.values]
        await self._verify(ctx, checked)
        self._record(ctx, checked)

    def needs(self, run: SchemaRun, scope: str, name: str) -> bool:
        """Whether a field still wants the LLM: not found yet, or found by a route with a
        confidence below ``fallback_threshold``."""
        existing = run.fields.get(scope, {}).get(name)
        if existing is None or not existing.found:
            return True
        return existing.confidence is not None and existing.confidence < self.fallback_threshold

    def trigger(
        self, run: SchemaRun, scope: str, statement: Statement, spec: FieldSpec
    ) -> Trigger | None:
        """Why the statement falls back for ``spec``, if it does."""
        if not unique_spans(run.candidates.get((statement.id, spec.name), [])):
            return "no_candidates"
        selection = run.selections.get((scope, spec.name, statement.id))
        if selection is None:
            return None  # the selector asked nothing, so there's no answer to second-guess
        if selection.candidate is None:
            category = run.categories.get(statement.id)
            p = category.probabilities.get(spec.name, 0.0) if category else 0.0
            return "none_chosen" if p >= self.category_threshold else None
        if selection.confidence < self.fallback_threshold:
            return "low_confidence"
        return None

    def _plan(self, ctx: Context) -> list[_Ask]:
        """One ask per (schema, statement, field) that falls back, in document order."""
        asks: dict[tuple[str, str, str], _Ask] = {}
        for run in ctx.active:
            for scope in run.scopes:
                for statement, spec in field_statements(ctx, run, scope, include_found=True):
                    if not spec.needs_candidates or not self.needs(run, scope.label, spec.name):
                        continue
                    why = self.trigger(run, scope.label, statement, spec)
                    if why is None:
                        continue
                    key = (run.name, statement.id, spec.name)
                    ask = asks.setdefault(key, _Ask(run, spec, statement, why))
                    ask.scopes.append(scope.label)
        return list(asks.values())

    async def _extract(
        self, ctx: Context, extractor: LLMExtractor, ask: _Ask, budget: DocumentBudget
    ) -> LLMAnswer | None:
        try:
            return await extractor.extract(ask.statement, ask.spec, budget)
        except LLMError as exc:
            # The fallback is optional: a failed call leaves the Jev answer in place.
            _event(ctx, ask, "llm_error", f"{type(exc).__name__}: {exc}")
            return None

    def _check(self, ctx: Context, ask: _Ask, answer: LLMAnswer) -> None:
        """Keep an answer whose evidence is in the statement and whose value fits."""
        ask.answer = answer
        start = ask.statement.text.find(answer.evidence) if answer.evidence.strip() else -1
        if start < 0:
            _event(
                ctx, ask, "llm_no_evidence", f"evidence {answer.evidence!r} is not in the statement"
            )
            return
        raw: Any = answer.value
        items: list[Any] = (
            cast("list[Any]", raw) if ask.spec.many and isinstance(raw, list) else [raw]
        )
        values: list[Any] = []
        try:
            for item in items:
                value = normalise(item, [], ask.spec)
                if value not in values:
                    values.append(value)
        except NormaliseError as exc:
            _event(ctx, ask, "llm_invalid", str(exc))
            return
        if not values:
            return
        ask.values = values
        ask.span = Span(start=start, end=start + len(answer.evidence))

    async def _verify(self, ctx: Context, asks: list[_Ask]) -> None:
        """One Jev request per statement, holding every check about it."""
        by_statement: dict[str, list[_Ask]] = {}
        for ask in asks:
            by_statement.setdefault(ask.statement.id, []).append(ask)
        groups = list(by_statement.values())
        replies = await gather(
            ctx.jev.ask(
                statement_state(group[0].statement),
                {k: q for a in group for k, q in a.questions().items()},
            )
            for group in groups
        )
        for group, reply in zip(groups, replies, strict=True):
            for ask in group:
                for i, key in enumerate(ask.questions()):
                    answer = reply[key]
                    assert isinstance(answer, NoulAnswer)
                    ask.item_p[i] = answer.p

    def _record(self, ctx: Context, asks: list[_Ask]) -> None:
        """Judge each answer once, then settle each scope's field from its answers."""
        grouped: dict[tuple[str, str, str], list[_Ask]] = {}
        for ask in asks:
            passed = {i for i, p in ask.item_p.items() if p >= self.verify_threshold}
            ask.kept = [v for i, v in enumerate(ask.values) if i in passed]
            if ask.kept:
                ask.p = min(ask.item_p[i] for i in passed)
                ctx.verified.extend(_examples(ask))
            ask.rejected = [
                _rejected(ctx, ask, v, ask.item_p[i])
                for i, v in enumerate(ask.values)
                if i not in passed
            ]
            for scope in ask.scopes:
                grouped.setdefault((ask.run.name, scope, ask.spec.name), []).append(ask)
        for (schema, scope, name), found in grouped.items():
            _settle(ctx, ctx.schemas[schema], scope, name, found)


def _settle(ctx: Context, run: SchemaRun, scope: str, name: str, found: list[_Ask]) -> None:
    """The best verified answer replaces the Jev answer; with none, the rejected values
    become alternatives of whatever Jev found (low-confidence, or nothing)."""
    spec = run.spec.field(name)
    shared = run.shared_statements(scope)
    existing = run.fields.get(scope, {}).get(name)
    rejected = [alt for ask in found for alt in ask.rejected]
    accepted = [ask for ask in found if ask.kept]
    if not accepted:
        if rejected:
            base = existing or FieldMeta()
            alternatives = sorted([*base.alternatives, *rejected], key=lambda a: -a.p)
            run.set_field(scope, name, base.model_copy(update={"alternatives": alternatives}))
        return
    order = {sid: i for i, sid in enumerate(ctx.parsed.statements)} if ctx.parsed else {}
    # The entity's own statements beat shared ones, then the more certain, then the earlier.
    accepted.sort(
        key=lambda a: (a.statement.id in shared, -a.p, order.get(a.statement.id, len(order)))
    )
    best = accepted[0]
    value: Any = best.kept if spec.many else best.kept[0]
    alternatives = list(rejected)
    if existing is not None and existing.found:
        alternatives.append(
            Alternative(
                value=existing.value,
                raw=_raw(existing),
                p=existing.confidence if existing.confidence is not None else 0.0,
            )
        )
    for other in accepted[1:]:
        other_value = other.kept if spec.many else other.kept[0]
        if other_value != value and other.answer is not None:
            alternatives.append(
                Alternative(value=other_value, raw=other.answer.evidence, p=other.p)
            )
    run.set_field(
        scope,
        name,
        FieldMeta(
            value=value,
            confidence=best.p,
            method="llm",
            source=_source(ctx, best),
            alternatives=sorted(alternatives, key=lambda a: -a.p),
            verified=True,
            shared=best.statement.id in shared,
            conflicts=existing.conflicts if existing is not None else [],
        ),
    )


def _raw(meta: FieldMeta) -> str | None:
    source = meta.source
    if source is None or source.span is None or source.statement is None:
        return None
    return source.span.of(source.statement)


def _source(ctx: Context, ask: _Ask) -> Source:
    statement = ask.statement
    return Source(
        url=ctx.document.url,
        component_id=statement.component_id,
        statement_id=statement.id,
        statement=statement.text,
        span=ask.span,
        location=statement.location,
    )


def _rejected(ctx: Context, ask: _Ask, value: Any, p: float) -> Alternative:
    evidence = ask.answer.evidence if ask.answer else None
    _event(
        ctx,
        ask,
        "llm_rejected",
        f"Jev didn't verify {value!r} (p={p:.2f})",
        value=value,
        p=p,
    )
    return Alternative(value=value, raw=evidence, p=p)


def _examples(ask: _Ask) -> list[VerifiedExample]:
    """The verified answer as an example for the learner, one per value."""
    statement = ask.statement
    field_key = f"{ask.run.name}.{ask.spec.name}"
    evidence = (ask.span.start, ask.span.end) if ask.span else None
    out: list[VerifiedExample] = []
    for value in ask.kept:
        digest = hashlib.sha256(f"{field_key}\0{statement.text}\0{value!r}".encode()).hexdigest()
        out.append(
            VerifiedExample(
                id=f"ex-{digest[:12]}",
                field=field_key,
                statement=statement.text,
                value=value,
                evidence=evidence,
                context={"heading_trail": statement.heading_trail, "kind": statement.kind},
                source="llm",
                probability=ask.p,
            )
        )
    return out


def _event(ctx: Context, ask: _Ask, kind: str, message: str, **data: Any) -> None:
    ctx.event(
        "fallback",
        kind,
        f"{ask.run.name}.{ask.spec.name} on {ask.statement.id}: {message}",
        schema=ask.run.name,
        field=ask.spec.name,
        statement_id=ask.statement.id,
        trigger=ask.trigger,
        **data,
    )

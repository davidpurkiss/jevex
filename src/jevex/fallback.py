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

**Vision values** (spec: *Image stage*) are verified the same way, with or without an
LLM, before anything is asked of one. A value Jev picked from a vision model's statement
(``method="vision"``; for a list field, each item only vision statements gave,
:attr:`~jevex.pipeline.SchemaRun.vision_values`) gets the same Noul against that
statement, one request per statement:

- ``p >= verify_threshold``: the value stays, with ``verified=True`` and its confidence
  lowered to ``p`` if that's less, and is added to ``Context.verified`` for learning
  (``source="vision"``).
- Otherwise it is dropped and listed in the field's ``alternatives`` with its
  verification probability (a ``vision_rejected`` event). A field left empty is then
  one the LLM fallback may ask about.

Jev reads only the statement, not the image, so this checks that the value is what the
model's statement says, not that the statement is true of the image.
"""

from __future__ import annotations

import re
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
from jevex.store import VerifiedExample, example_context, example_id

if TYPE_CHECKING:
    from jevex.interfaces import LLMExtractor
    from jevex.jev import Question
    from jevex.llm import LLM
    from jevex.pipeline import Context, SchemaRun, VisionValue
    from jevex.schema import FieldSpec
    from jevex.statements import Statement

CATEGORY_THRESHOLD = 0.5
"""Category probability at or above which Jev's "none" sends the statement to the LLM.
Provisional: the spec's open questions set the defaults from eval runs."""
FALLBACK_THRESHOLD = 0.5
"""Selection confidence below which the LLM is asked. Provisional, like the others."""
VERIFY_THRESHOLD = 0.8
"""Verification probability at or above which an LLM answer (or a vision value) is
accepted. Provisional."""

SELECTED = frozenset({"jev", "generator", "vision"})
"""Methods of the values the select and normalise stages record: a vision value is only
checked while one of theirs stands (not, say, embedded data that won a merge)."""

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


def field_type_text(spec: FieldSpec) -> str:
    """The field's type in words, for prompts: "a number", "a list, each item text"..."""
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
            type=field_type_text(field),
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
        await self._verify_vision(ctx)
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

    async def _verify_vision(self, ctx: Context) -> None:
        """Verify the vision values that stand (see the module docstring)."""
        checks = _vision_checks(ctx)
        if not checks or ctx.parsed is None:
            return
        statements = ctx.parsed.statements
        questions: dict[str, dict[str, Question]] = {}
        keys: dict[tuple[str, str, str, str], str] = {}
        for check in checks:
            for item in check.items:
                ident = check.ident(item)
                if ident in keys:
                    continue
                asked = questions.setdefault(item.statement_id, {})
                key = f"{check.run.name}.{check.spec.name}/vision{len(asked)}"
                spec = check.spec
                asked[key] = (
                    spec.member_question(item.value)
                    if spec.many
                    else spec.verify_question(item.value)
                )
                keys[ident] = key
        ids = list(questions)
        replies = await gather(
            ctx.jev.ask(statement_state(statements[sid]), questions[sid]) for sid in ids
        )
        p: dict[tuple[str, str, str, str], float] = {}
        for sid, reply in zip(ids, replies, strict=True):
            for ident, key in keys.items():
                if ident[0] == sid:
                    answer = reply[key]
                    assert isinstance(answer, NoulAnswer)
                    p[ident] = answer.p
        queued: set[str] = set()
        for check in checks:
            item_p = [p[check.ident(item)] for item in check.items]
            for example in self._settle_vision(ctx, check, item_p):
                if example.id not in queued:
                    queued.add(example.id)
                    ctx.verified.append(example)

    def _settle_vision(
        self, ctx: Context, check: _VisionCheck, item_p: list[float]
    ) -> list[VerifiedExample]:
        """Keep or drop one field's vision values (``item_p``: each one's verification
        probability); the examples of the ones kept. A value one vision statement verified
        stays, even if another that gave it too wasn't verified."""
        meta, spec = check.meta, check.spec
        statements = ctx.parsed.statements if ctx.parsed else {}
        kept: list[tuple[VisionValue, float]] = []
        dropped: list[tuple[VisionValue, float]] = []
        for item, p in zip(check.items, item_p, strict=True):
            (kept if p >= self.verify_threshold else dropped).append((item, p))
        verified = [item.value for item, _ in kept]
        rejected: list[Alternative] = []
        for item, p in dropped:
            if item.value not in verified:
                statement = statements[item.statement_id]
                raw = item.span.of(statement.text) if item.span else None
                rejected.append(Alternative(value=item.value, raw=raw, p=p))
            ctx.event(
                self.name,
                "vision_rejected",
                f"{check.run.name}.{spec.name} on {item.statement_id}: Jev didn't verify "
                f"{item.value!r} (p={p:.2f})",
                schema=check.run.name,
                field=spec.name,
                statement_id=item.statement_id,
                value=item.value,
                p=p,
            )
        current: list[Any] = cast("list[Any]", meta.value) if spec.many else [meta.value]
        gone = [item.value for item, _ in dropped if item.value not in verified]
        value: Any = [v for v in current if v not in gone]
        if not spec.many:
            value = value[0] if value else None
        if value is None or value == []:
            check.run.set_field(
                check.scope,
                spec.name,
                FieldMeta(
                    alternatives=_ranked([*meta.alternatives, *rejected], []),
                    conflicts=meta.conflicts,
                ),
            )
            return []
        update: dict[str, Any] = {"value": value}
        if rejected:
            update["alternatives"] = _ranked([*meta.alternatives, *rejected], value)
        if kept:
            least = min(p for _, p in kept)
            confidence = meta.confidence
            update["confidence"] = least if confidence is None else min(confidence, least)
            update["verified"] = True
        check.run.set_field(check.scope, spec.name, meta.model_copy(update=update))
        field_key = f"{check.run.name}.{spec.name}"
        examples: list[VerifiedExample] = []
        for item, p in kept:
            statement = statements[item.statement_id]
            examples.append(
                VerifiedExample(
                    id=example_id(field_key, statement.text, item.value),
                    field=field_key,
                    statement=statement.text,
                    value=item.value,
                    evidence=(item.span.start, item.span.end) if item.span else None,
                    context=example_context(statement, ctx.locale),
                    source="vision",
                    probability=p,
                    document_source=ctx.document.source,
                )
            )
        return examples

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
        found = _find_evidence(ask.statement.text, answer.evidence)
        if found is None:
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
        ask.span = Span(start=found[0], end=found[1])

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
                ctx.verified.extend(_examples(ask, ctx.document.source, ctx.locale))
            ask.rejected = [
                _rejected(ctx, ask, v, ask.item_p[i])
                for i, v in enumerate(ask.values)
                if i not in passed
            ]
            for scope in ask.scopes:
                grouped.setdefault((ask.run.name, scope, ask.spec.name), []).append(ask)
        for (schema, scope, name), found in grouped.items():
            _settle(ctx, ctx.schemas[schema], scope, name, found)


@dataclass
class _VisionCheck:
    """One field of one scope whose value has vision values to verify."""

    run: SchemaRun
    scope: str
    spec: FieldSpec
    meta: FieldMeta
    items: list[VisionValue]

    def ident(self, item: VisionValue) -> tuple[str, str, str, str]:
        """The question an item needs, the same for every scope that shares it."""
        return (item.statement_id, self.run.name, self.spec.name, repr(item.value))


def _vision_checks(ctx: Context) -> list[_VisionCheck]:
    """The fields whose standing value holds vision values not yet verified."""
    statements = ctx.parsed.statements if ctx.parsed else {}
    checks: list[_VisionCheck] = []
    for run in ctx.active:
        for (scope, name), items in run.vision_values.items():
            meta = run.fields.get(scope, {}).get(name)
            if meta is None or not meta.found or meta.verified is not None:
                continue
            if meta.method not in SELECTED:
                continue
            spec = run.spec.field(name)
            current: list[Any] = cast("list[Any]", meta.value) if spec.many else [meta.value]
            pending = [i for i in items if i.value in current and i.statement_id in statements]
            if pending:
                checks.append(_VisionCheck(run, scope, spec, meta, pending))
    return checks


def _settle(ctx: Context, run: SchemaRun, scope: str, name: str, found: list[_Ask]) -> None:
    """The best verified answer replaces the Jev answer; with none, the rejected values
    become alternatives of whatever Jev found (low-confidence, or nothing).

    Among verified answers the entity's own statements beat shared ones, as in the other
    routes, but any verified answer beats the unverified Jev answer, even a shared one
    over the entity's own. A list field takes every verified item, in document order,
    from the own statements that gave any (else from shared ones), with the least
    certain contributing answer's probability.
    """
    spec = run.spec.field(name)
    shared = run.shared_statements(scope)
    existing = run.fields.get(scope, {}).get(name)
    rejected = [alt for ask in found for alt in ask.rejected]
    accepted = [ask for ask in found if ask.kept]
    if not accepted:
        if rejected:
            base = existing or FieldMeta()
            alternatives = [*base.alternatives, *rejected]
            if base.found:
                alternatives = _ranked(alternatives, base.value)
            else:
                alternatives.sort(key=lambda a: -a.p)
            run.set_field(scope, name, base.model_copy(update={"alternatives": alternatives}))
        return
    order = {sid: i for i, sid in enumerate(ctx.parsed.statements)} if ctx.parsed else {}
    accepted.sort(
        key=lambda a: (a.statement.id in shared, -a.p, order.get(a.statement.id, len(order)))
    )
    best = accepted[0]
    if spec.many:
        used = [a for a in accepted if (a.statement.id in shared) == (best.statement.id in shared)]
        items: list[Any] = []
        for ask in sorted(used, key=lambda a: order.get(a.statement.id, len(order))):
            items.extend(v for v in ask.kept if v not in items)
        value: Any = items
        confidence = min(a.p for a in used)
        others: list[_Ask] = []
    else:
        value, confidence = best.kept[0], best.p
        others = [a for a in accepted[1:] if a.kept[0] != value]
    alternatives = [*(existing.alternatives if existing else []), *rejected]
    if existing is not None and existing.found:
        alternatives.append(
            Alternative(
                value=existing.value,
                raw=_raw(existing),
                p=existing.confidence if existing.confidence is not None else 0.0,
            )
        )
    alternatives.extend(
        Alternative(value=a.kept[0], raw=a.answer.evidence if a.answer else None, p=a.p)
        for a in others
    )
    run.set_field(
        scope,
        name,
        FieldMeta(
            value=value,
            confidence=confidence,
            method="llm",
            source=_source(ctx, best),
            alternatives=_ranked(alternatives, value),
            verified=True,
            shared=best.statement.id in shared,
            conflicts=existing.conflicts if existing is not None else [],
        ),
    )


def _ranked(alternatives: list[Alternative], value: Any) -> list[Alternative]:
    """Most likely first, one per value, never the chosen value (or a chosen list item)."""
    chosen: list[Any] = cast("list[Any]", value) if isinstance(value, list) else [value]
    out: dict[str, Alternative] = {}
    for alt in sorted(alternatives, key=lambda a: -a.p):
        if alt.value == value or alt.value in chosen:
            continue
        out.setdefault(repr(alt.value), alt)
    return list(out.values())


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


def _examples(ask: _Ask, document_source: str | None, locale: str | None) -> list[VerifiedExample]:
    """The verified answer as an example for the learner, one per value. ``locale`` is the
    document's own (none when it doesn't say), which the learner scopes generators to."""
    statement = ask.statement
    field_key = f"{ask.run.name}.{ask.spec.name}"
    evidence = (ask.span.start, ask.span.end) if ask.span else None
    out: list[VerifiedExample] = []
    for value in ask.kept:
        out.append(
            VerifiedExample(
                id=example_id(field_key, statement.text, value),
                field=field_key,
                statement=statement.text,
                value=value,
                evidence=evidence,
                context=example_context(statement, locale),
                source="llm",
                probability=ask.p,
                document_source=document_source,
            )
        )
    return out


def _find_evidence(text: str, evidence: str) -> tuple[int, int] | None:
    """Where ``evidence`` is in ``text``, any run of whitespace matching any other: a
    statement keeps no-break spaces ("9,1\u00a0s") that an LLM writes as plain ones."""
    words = evidence.split()
    if not words:
        return None
    m = re.search(r"\s+".join(re.escape(w) for w in words), text)
    return (m.start(), m.end()) if m else None


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

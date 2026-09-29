"""Candidates and selection (spec: *Value extraction*, stages 11 and 12).

After categorising, each statement is assigned to a field (or none). Then:

- :class:`CandidateStage` runs the generator registry on statements whose field needs
  candidates (numbers, dates, strings) and stores them in ``SchemaRun.candidates``.
- :class:`SelectStage` asks Jev, once per statement, every question that statement needs:
  a Choice over candidate spans plus "none", a Choice over an enum's options plus
  "not stated", or a Noul for a bool. Candidate picks go to ``SchemaRun.selections`` for
  the normalise stage; enum and bool answers are values already, so they're recorded as
  :class:`~jevex.results.FieldMeta` directly (``method="jev"``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from jevex.generators import GeneratorRegistry, default_registry
from jevex.interfaces import Selection
from jevex.jev import ChoiceAnswer, JSONContent, NoulAnswer, Question
from jevex.results import Alternative, FieldMeta, Source
from jevex.schema import NONE_OPTION, NOT_STATED_OPTION

if TYPE_CHECKING:
    from jevex.entities import EntityScope
    from jevex.pipeline import Context, SchemaRun
    from jevex.schema import FieldSpec
    from jevex.statements import Candidate, Statement

BOOL_TRUE_AT = 0.5


def field_statements(
    ctx: Context, run: SchemaRun, scope: EntityScope
) -> list[tuple[Statement, FieldSpec]]:
    """(statement, field) pairs in scope that the classifier assigned to a field."""
    if ctx.parsed is None:
        return []
    out: list[tuple[Statement, FieldSpec]] = []
    names = {f.name for f in run.spec.fields}
    for statement in ctx.parsed.statements_in(scope.component_ids):
        answer = run.categories.get(statement.id)
        if answer is not None and answer.choice in names:
            out.append((statement, run.spec.field(answer.choice)))
    return out


def statement_state(statement: Statement) -> JSONContent:
    """What Jev sees for one statement: its text plus the headings above it."""
    state: dict[str, Any] = {"statement": statement.text}
    if statement.heading_trail:
        state["section"] = " › ".join(statement.heading_trail)
    return state


@dataclass
class CandidateStage:
    """Generates candidate spans for every categorised statement whose field needs them."""

    registry: GeneratorRegistry = field(default_factory=default_registry)
    locale: str | None = None
    name: str = "candidates"

    async def run(self, ctx: Context) -> None:
        for run in ctx.active:
            for scope in run.scopes:
                for statement, spec in field_statements(ctx, run, scope):
                    if not spec.needs_candidates:
                        continue
                    run.candidates[(statement.id, spec.name)] = self.registry.generate(
                        statement, spec, schema=run.name, locale=self.locale
                    )


@dataclass
class SelectStage:
    """Asks Jev to pick values, one batched request per statement."""

    name: str = "select"

    async def run(self, ctx: Context) -> None:
        jobs = [
            self._statement(ctx, run, scope, statement, spec)
            for run in ctx.active
            for scope in run.scopes
            for statement, spec in field_statements(ctx, run, scope)
        ]
        await asyncio.gather(*jobs)

    async def _statement(
        self,
        ctx: Context,
        run: SchemaRun,
        scope: EntityScope,
        statement: Statement,
        spec: FieldSpec,
    ) -> None:
        questions: dict[str, Question] = {}
        by_raw: dict[str, Candidate] = {}
        if spec.kind == "enum":
            questions[spec.name] = spec.enum_question()
        elif spec.kind == "bool":
            questions[spec.name] = spec.bool_question()
        elif spec.needs_candidates:
            candidates = run.candidates.get((statement.id, spec.name), [])
            for cand in candidates:
                by_raw.setdefault(cand.raw, cand)
            by_raw.pop(NONE_OPTION, None)  # a literal "none" span can't be an option
            if not by_raw:
                return  # nothing to choose from; the LLM fallback (#33) handles this
            questions[spec.name] = spec.select_question(list(by_raw))
        else:
            return

        answer = (await ctx.jev.ask(statement_state(statement), questions))[spec.name]
        key = (scope.label, spec.name, statement.id)
        if isinstance(answer, NoulAnswer):
            self._record_direct(
                ctx, run, scope, statement, spec, answer.p >= BOOL_TRUE_AT, answer.p
            )
        elif isinstance(answer, ChoiceAnswer) and spec.kind == "enum":
            if answer.choice != NOT_STATED_OPTION:
                self._record_direct(
                    ctx, run, scope, statement, spec, answer.choice, answer.confidence, answer
                )
        elif isinstance(answer, ChoiceAnswer):
            chosen = by_raw.get(answer.choice)
            run.selections[key] = Selection(
                candidate=chosen,
                confidence=answer.confidence,
                alternatives={
                    raw: p
                    for raw, p in answer.probabilities.items()
                    if raw != answer.choice and raw != NONE_OPTION
                },
            )

    def _record_direct(
        self,
        ctx: Context,
        run: SchemaRun,
        scope: EntityScope,
        statement: Statement,
        spec: FieldSpec,
        value: Any,
        p: float,
        choice: ChoiceAnswer | None = None,
    ) -> None:
        """Enum and bool answers are values: keep the most confident per field and scope.

        For ``list[...]`` enum fields, every stated option is collected instead.
        """
        confidence = p if choice is not None or value else 1 - p
        alternatives = (
            [
                Alternative(value=o, raw=o, p=q)
                for o, q in sorted(choice.probabilities.items(), key=lambda kv: -kv[1])
                if o != choice.choice
            ]
            if choice is not None
            else []
        )
        existing = run.fields.get(scope.label, {}).get(spec.name)
        if spec.many:
            previous: object = existing.value if existing is not None else None
            values: list[Any] = (
                list(cast("list[Any]", previous)) if isinstance(previous, list) else []
            )
            if value not in values:
                values.append(value)
            value = values
        elif existing is not None and existing.found and (existing.confidence or 0) >= confidence:
            return
        run.set_field(
            scope.label,
            spec.name,
            FieldMeta(
                value=value,
                confidence=confidence,
                method="jev",
                source=Source(
                    url=ctx.document.url,
                    component_id=statement.component_id,
                    statement_id=statement.id,
                    statement=statement.text,
                    location=statement.location,
                ),
                alternatives=alternatives,
            ),
        )

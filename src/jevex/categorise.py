"""Statement categorisation: which field, if any, each statement states (spec: *Pipeline
architecture*, stage 10).

One Jev Choice per statement and schema, over the schema's field descriptions plus "none
of these", with every active schema's Choice about a statement in one request. When the
component gate ran, a statement's options are only the fields whose group its component
passed for, so Jev never weighs fields the section can't contain; a statement with no
such field isn't asked at all.

Every answer is kept on ``SchemaRun.categories`` with its full probability distribution,
including "none" answers. The selector routes a statement to its top field and to any
other field with at least :data:`jevex.select.ALSO_CATEGORY_P` ("In stock (22
available)" states both ``in_stock`` and ``stock_count``). The LLM fallback (#33) reads
the distributions too. With gating, a distribution covers only the offered fields: a
field missing from it wasn't offered.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jevex.jev import Choice, ChoiceAnswer, UnexpectedAnswerError
from jevex.select import statement_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jevex.interfaces import StatementClassifier
    from jevex.jev import JevClient
    from jevex.pipeline import Context, SchemaRun
    from jevex.schema import SchemaSpec
    from jevex.statements import Statement

SKIPPED_KINDS = frozenset({"structured"})
"""Statement kinds that aren't categorised: structured data is mapped by key path (#30)."""


@dataclass(frozen=True)
class ToClassify:
    """One statement and the schemas to categorise it for.

    Each entry is ``(schema, fields)``: ``fields`` lists the field names it may be (from
    the component gate), or ``None`` for any field.
    """

    statement: Statement
    schemas: tuple[tuple[SchemaSpec, tuple[str, ...] | None], ...]


class JevStatementClassifier:
    """The default :class:`~jevex.interfaces.StatementClassifier`.

    One request per statement: its state (text plus heading trail) with one Choice per
    schema, keyed by schema name. A schema whose options would be only "none" (no
    allowed field, or only nested-model fields) isn't asked.
    """

    async def classify(
        self, items: Sequence[ToClassify], jev: JevClient
    ) -> dict[str, dict[str, ChoiceAnswer]]:
        out: dict[str, dict[str, ChoiceAnswer]] = {}

        async def one(item: ToClassify) -> None:
            questions: dict[str, Choice] = {}
            for schema, fields in item.schemas:
                question = schema.categorise_question(fields)
                if len(question.options) > 1:  # more than "none"
                    questions[schema.name] = question
            if not questions:
                return
            answers = await jev.ask(statement_state(item.statement), questions)
            for name, answer in answers.items():
                if not isinstance(answer, ChoiceAnswer):
                    raise UnexpectedAnswerError(
                        f"expected a Choice answer for {name!r}, got {answer.type}"
                    )
                out.setdefault(name, {})[item.statement.id] = answer

        await asyncio.gather(*(one(item) for item in items))
        return out


def _allowed(run: SchemaRun, statement: Statement) -> tuple[str, ...] | None:
    if run.component_ids is None:
        return None
    return tuple(f.name for f in run.relevant_fields(statement.component_id) if f.kind != "model")


@dataclass
class CategoriseStage:
    """Categorises every in-scope statement for each active schema (stage 10).

    A schema's statements are those in any of its scopes' components, asked once even if
    scopes share them; statements already categorised are skipped. Results land on
    ``SchemaRun.categories``.
    """

    classifier: StatementClassifier = field(default_factory=JevStatementClassifier)
    name: str = "categorise"

    async def run(self, ctx: Context) -> None:
        parsed = ctx.parsed
        if parsed is None:
            return
        by_statement: dict[str, list[tuple[SchemaSpec, tuple[str, ...] | None]]] = {}
        statements: dict[str, Statement] = {}
        for run in ctx.active:
            seen: set[str] = set()
            for scope in run.scopes:
                for statement in parsed.statements_in(scope.component_ids):
                    sid = statement.id
                    if sid in seen or sid in run.categories or statement.kind in SKIPPED_KINDS:
                        continue
                    seen.add(sid)
                    allowed = _allowed(run, statement)
                    if allowed is not None and not allowed:
                        continue
                    statements[sid] = statement
                    by_statement.setdefault(sid, []).append((run.spec, allowed))
        if not by_statement:
            return
        items = [ToClassify(statements[sid], tuple(pairs)) for sid, pairs in by_statement.items()]
        answers = await self.classifier.classify(items, ctx.jev)
        for run in ctx.active:
            run.categories.update(answers.get(run.name, {}))

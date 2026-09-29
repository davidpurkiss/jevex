"""Statement categorisation: which field, if any, each statement states (spec: *Pipeline
architecture*, stage 10).

One Jev Choice per statement over the schema's field descriptions plus "none of these".
When the component gate ran, a statement's options are only the fields whose group its
component passed for, so Jev never weighs fields the section can't contain, and a
statement whose component passed for no group isn't asked at all.

Every answer is kept on ``SchemaRun.categories`` with its full probability distribution,
including "none" answers. The selector uses the top choice; the LLM fallback (#33) uses
the distribution to decide which statements are worth an LLM call for a field that
selection left empty.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jevex.jev import ChoiceAnswer, UnexpectedAnswerError
from jevex.pipeline import for_each_schema
from jevex.select import statement_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jevex.interfaces import StatementClassifier
    from jevex.jev import JevClient
    from jevex.pipeline import Context, SchemaRun
    from jevex.schema import SchemaSpec
    from jevex.statements import Statement

SKIPPED_KINDS = frozenset({"structured"})
"""Statement kinds that aren't categorised: structured data is mapped by key path (#30)."""


class JevStatementClassifier:
    """The default :class:`~jevex.interfaces.StatementClassifier`: one Choice each.

    ``fields`` maps a statement id to the field names it may be categorised as; missing
    ids may be any field, and an empty list means the statement isn't asked. Each
    statement is its own state (text plus heading trail), so answers are independent and
    Jev can batch them.
    """

    async def classify(
        self,
        statements: list[Statement],
        schema: SchemaSpec,
        jev: JevClient,
        *,
        fields: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, ChoiceAnswer]:
        key = schema.name

        async def one(statement: Statement) -> tuple[str, ChoiceAnswer] | None:
            allowed = None if fields is None else fields.get(statement.id)
            if allowed is not None and not allowed:
                return None
            question = schema.categorise_question(allowed)
            answers = await jev.ask(statement_state(statement), {key: question})
            answer = answers[key]
            if not isinstance(answer, ChoiceAnswer):
                raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
            return statement.id, answer

        results = await asyncio.gather(*(one(s) for s in statements))
        return dict(r for r in results if r is not None)


@dataclass
class CategoriseStage:
    """Categorises every in-scope statement of each active schema (stage 10).

    Statements are those in any scope's components, each asked once per schema even if
    several scopes share it. Results land on ``SchemaRun.categories``.
    """

    classifier: StatementClassifier = field(default_factory=JevStatementClassifier)
    name: str = "categorise"

    async def run(self, ctx: Context) -> None:
        parsed = ctx.parsed
        if parsed is None:
            return

        async def categorise(run: SchemaRun) -> None:
            seen: dict[str, Statement] = {}
            for scope in run.scopes:
                for statement in parsed.statements_in(scope.component_ids):
                    if statement.kind not in SKIPPED_KINDS and statement.id not in run.categories:
                        seen.setdefault(statement.id, statement)
            if not seen:
                return
            fields = (
                None
                if run.component_ids is None
                else {
                    sid: [f.name for f in run.relevant_fields(s.component_id)]
                    for sid, s in seen.items()
                }
            )
            answers = await _classify(self.classifier, list(seen.values()), run, ctx, fields)
            run.categories.update(answers)

        await for_each_schema(ctx, categorise)


async def _classify(
    classifier: StatementClassifier,
    statements: list[Statement],
    run: SchemaRun,
    ctx: Context,
    fields: Mapping[str, Sequence[str]] | None,
) -> dict[str, ChoiceAnswer]:
    if fields is None:
        return await classifier.classify(statements, run.spec, ctx.jev)
    # Only statements with at least one relevant field are asked.
    asked = [s for s in statements if fields.get(s.id)]
    if not asked:
        return {}
    return await classifier.classify(asked, run.spec, ctx.jev, fields=fields)

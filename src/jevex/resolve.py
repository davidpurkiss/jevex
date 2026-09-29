"""Entity resolution: which parts of a document belong to which record (spec: *Entity models*).

The stage asks the configured :class:`~jevex.interfaces.EntityResolver` for each active
schema's scopes. Everything downstream (statements, categorising, candidates, selection)
then runs once per scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jevex.entities import EntityScope
from jevex.pipeline import for_each_schema

if TYPE_CHECKING:
    from jevex.interfaces import EntityResolver, ParsedDocument
    from jevex.jev import JevClient
    from jevex.pipeline import Context, SchemaRun
    from jevex.schema import SchemaSpec

SINGLE_ENTITY_LABEL = "document"


@dataclass(frozen=True)
class SingleEntity:
    """The whole document is one record: the cheapest resolver, asking no questions.

    Use it when you know a page holds one record (a listing detail page, one product).
    The scope covers every component; later stages still skip components the component
    gate rejected. Entities are resolved before statements are split (stage 8 vs 9), so
    ``statement_ids`` holds only statements that already exist (e.g. structured data).
    Later stages should use ``parsed.statements_in(scope.component_ids)``.
    """

    label: str = SINGLE_ENTITY_LABEL

    async def resolve(
        self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
    ) -> list[EntityScope]:
        return [
            EntityScope(
                label=self.label,
                component_ids=[c.id for c in parsed.root.walk()],
                statement_ids=list(parsed.statements),
            )
        ]


@dataclass
class EntityStage:
    """Sets ``SchemaRun.scopes`` for every active schema using ``resolver``.

    When the component gate ran, the resolver is given only the components that passed
    (``ParsedDocument.restricted_to``), and scopes are filtered to them as well, so
    statements from irrelevant parts of the page never reach categorisation.

    Without a parsed document (no layout stage ran), every schema gets one empty scope,
    labelled as the resolver would label a single entity, so structured-data-only
    pipelines still produce a record.
    """

    resolver: EntityResolver = field(default_factory=SingleEntity)
    name: str = "entities"

    async def run(self, ctx: Context) -> None:
        parsed = ctx.parsed

        async def resolve(run: SchemaRun) -> None:
            if parsed is None:
                label = getattr(self.resolver, "label", SINGLE_ENTITY_LABEL)
                run.scopes = [
                    EntityScope(label=label if isinstance(label, str) else SINGLE_ENTITY_LABEL)
                ]
                return
            relevant = run.relevant_components()
            # The resolver sees only what passed the component gate, so a resolver that
            # asks Jev (MultiEntity, #27) never pays for gated-out components.
            view = parsed if relevant is None else parsed.restricted_to(relevant)
            run.scopes = await self.resolver.resolve(view, run.spec, ctx.jev)
            if relevant is not None:
                # Only components that passed the component gate go downstream.
                for scope in run.scopes:
                    scope.component_ids = [c for c in scope.component_ids if c in relevant]
            if not run.scopes:
                ctx.event(self.name, "no_entities", f"{run.name}: the resolver found no entities")

        await for_each_schema(ctx, resolve)

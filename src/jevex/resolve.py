"""Entity resolution: which parts of a document belong to which record (spec: *Entity models*).

The stage asks the configured :class:`~jevex.interfaces.EntityResolver` for each active
schema's scopes. Everything downstream (categorising, candidates, selection) then runs
once per scope. Entities are resolved after statements are split, so a resolver can
assign single statements: a comparison table's cells column by column, or a sentence
Jev says is about one trim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jevex._tasks import gather
from jevex.entities import EntityScope
from jevex.jev import MAX_CHOICE_OPTIONS, ChoiceAnswer, NoulAnswer, UnexpectedAnswerError
from jevex.layout import section_text
from jevex.pipeline import for_each_schema
from jevex.schema import ALL_OPTION
from jevex.select import statement_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jevex.interfaces import EntityResolver, ParsedDocument
    from jevex.jev import JevClient, JSONContent
    from jevex.layout import Component
    from jevex.pipeline import Context, SchemaRun
    from jevex.schema import SchemaSpec
    from jevex.statements import Statement

SINGLE_ENTITY_LABEL = "document"

MAX_ENTITY_LABEL_CHARS = 80
"""Entity labels (a column header, a card's heading) are shortened to this: they are
Choice options and record labels, not content."""

BOUNDARY_P = 0.5
"""Boundary Noul probability at or above which a label is taken as an entity."""

# Components whose children can each be one entity: a card, a list item, a section.
_CONTAINERS = frozenset({"section", "list_item", "breakout", "column"})


@dataclass(frozen=True)
class SingleEntity:
    """The whole document is one record: the cheapest resolver, asking no questions.

    Use it when you know a page holds one record (a listing detail page, one product).
    The scope covers every component; later stages still skip components the component
    gate rejected. ``statement_ids`` holds every statement, including structured data's.
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


@dataclass(frozen=True)
class _Group:
    """Entities one structural rule proposes: a table's columns, or a run of siblings.

    ``rule`` is its priority (0 table headers, 1 repeated siblings, 2 headed sections)
    and ``depth`` how deep it sits: a statement two groups claim goes to the innermost,
    then to the higher priority.
    """

    rule: int
    depth: int
    section: list[str]
    members: list[tuple[str, frozenset[str]]]
    """(label, ids of the statements it claims) per proposed entity."""

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.members]

    def state(self) -> JSONContent:
        """What Jev sees when asked which labels name entities: the labels together, so it
        can tell "SE, SE L" (trims) from "Performance, Dimensions" (topics)."""
        state: dict[str, JSONContent] = {"names": self.labels}
        if section := section_text(self.section):
            state["section"] = section
        return state


@dataclass(frozen=True)
class MultiEntity:
    """Finds each entity (a trim, a listing) a document holds: the car finder's default.

    Boundary detection, in the spec's priority order:

    1. **Table headers:** in a table whose cells have both row and column headers, each
       column label is proposed as an entity, holding the cells below it (a cell spanning
       columns goes to each), and so is each row's label (its headers joined: "Kestrova
       SE"), holding its row's cells.
    2. **Repeated sibling structures:** children of one parent with the same shape (their
       type and their children's types), such as listing cards, each labelled by their
       first heading (or first text).
    3. **Headed sections:** sibling sections that each start with a heading ("SE",
       "SE L"), each with its subtree.

    Structure alone can't tell trims from topics ("Performance" and "Dimensions" sections
    look just like "SE" and "SE L" ones, and a table's rows are as likely to be trims as
    its columns), so each proposed label is checked with one Noul
    (:meth:`~jevex.schema.SchemaSpec.boundary_question`), a group's labels in one
    request. Labels at ``boundary_p`` or above are entities; the same label in two places
    (the "SE" column of two tables) is one entity. With ``confirm=False`` nothing is
    asked and every proposed label is an entity, except table rows (only columns are
    proposed): use it only for pages whose structure is known to be one entity per
    group, since otherwise topic sections split the record.

    When accepted groups overlap, a statement goes to the innermost (listing cards under
    a trim heading are each an entity; a table's columns beat the section around it),
    then to the higher priority above (a table's columns beat its rows). A label left
    with nothing but its own heading is dropped.

    With two or more entities, every statement no boundary claimed is asked one Choice
    (:meth:`~jevex.schema.SchemaSpec.entity_question`): which entity it applies to, or
    "all of them". Those go on every scope's ``shared_statement_ids``: a value found only
    in them is copied into each record with ``meta.shared`` set. Past 254 entities there
    are too many options for a Choice, so those statements are left out (the entity stage
    reports them as ``unassigned_statements``). With fewer than two entities, the
    document is one entity labelled ``label``, as with :class:`SingleEntity`, and nothing
    more is asked.
    """

    confirm: bool = True
    boundary_p: float = BOUNDARY_P
    label: str = SINGLE_ENTITY_LABEL

    def __post_init__(self) -> None:
        if not 0.0 <= self.boundary_p <= 1.0:
            raise ValueError(f"boundary_p must be between 0 and 1, got {self.boundary_p}")

    async def resolve(
        self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
    ) -> list[EntityScope]:
        groups = [*_table_groups(parsed, rows=self.confirm), *_component_groups(parsed)]
        accepted = await self._accepted(groups, schema, jev)
        # The innermost group first, then the higher priority. A statement goes to every
        # label of the first group claiming it: a table cell spanning two columns is on
        # both.
        ranked = sorted(accepted, key=lambda a: (-a[0].depth, a[0].rule))
        owners: dict[str, list[str]] = {}
        winner: dict[str, _Group] = {}
        for group, label, ids in ranked:
            for sid in ids:
                if winner.setdefault(sid, group) is group and label not in owners.get(sid, []):
                    owners.setdefault(sid, []).append(label)
        _drop_bare_headings(parsed, owners)
        in_tree = {c.id for c in parsed.root.walk()}
        statements = [s for s in parsed.statements.values() if s.component_id in in_tree]
        labels = list(dict.fromkeys(label for s in statements for label in owners.get(s.id, [])))
        if len(labels) < 2:
            return await SingleEntity(label=self.label).resolve(parsed, schema, jev)
        shared = await self._assign(
            [s for s in statements if s.id not in owners], labels, owners, schema, jev
        )
        return _scopes(statements, labels, owners, shared)

    async def _accepted(
        self, groups: list[_Group], schema: SchemaSpec, jev: JevClient
    ) -> list[tuple[_Group, str, frozenset[str]]]:
        """(group, label, claimed ids) for every label taken as an entity."""
        if not self.confirm:
            return [(g, label, ids) for g in groups for label, ids in g.members]

        async def ask(group: _Group) -> list[tuple[_Group, str, frozenset[str]]]:
            questions = {
                f"label{i}": schema.boundary_question(label) for i, label in enumerate(group.labels)
            }
            answers = await jev.ask(group.state(), questions)
            out: list[tuple[_Group, str, frozenset[str]]] = []
            for i, (label, ids) in enumerate(group.members):
                answer = answers[f"label{i}"]
                if not isinstance(answer, NoulAnswer):
                    raise UnexpectedAnswerError(f"expected a Noul answer, got {answer.type}")
                if answer.p >= self.boundary_p:
                    out.append((group, label, ids))
            return out

        found = await gather(ask(g) for g in groups)
        return [a for per_group in found for a in per_group]

    async def _assign(
        self,
        ambiguous: list[Statement],
        labels: list[str],
        owners: dict[str, list[str]],
        schema: SchemaSpec,
        jev: JevClient,
    ) -> set[str]:
        """Ask which entity each unclaimed statement applies to. Fills ``owners``; returns
        the ids of statements that apply to all of them."""
        if len(labels) >= MAX_CHOICE_OPTIONS:
            return set()
        question = schema.entity_question(labels)

        async def ask(statement: Statement) -> tuple[str, str]:
            answer = (await jev.ask(statement_state(statement), {"entity": question}))["entity"]
            if not isinstance(answer, ChoiceAnswer):
                raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
            return statement.id, answer.choice

        shared: set[str] = set()
        for sid, choice in await gather(ask(s) for s in ambiguous):
            if choice == ALL_OPTION:
                shared.add(sid)
            else:
                owners[sid] = [choice]
        return shared


def _table_groups(parsed: ParsedDocument, *, rows: bool) -> list[_Group]:
    """Per table whose cells have row and column headers: its column labels as one group,
    and (with ``rows``) its row labels as another, each when there are two or more."""
    depths = _depths(parsed.root)
    by_table: dict[str, list[Statement]] = {}
    for s in parsed.statements.values():
        if s.table is not None and s.component_id in depths:
            by_table.setdefault(s.component_id, []).append(s)
    groups: list[_Group] = []
    for table_id, cells in by_table.items():
        # Only cells with headers on both axes: a header on one axis alone says what the
        # cell states (a key/value table), not which entity it's about.
        refs = [(c.id, c.table) for c in cells if c.table is not None]
        both = [(sid, ref) for sid, ref in refs if ref.row_headers and ref.col_headers]
        axes = [[(sid, ref.col_headers) for sid, ref in both]]
        if rows:
            # A row's headers are levels ("Kestrova" spanning "SE" and "SE L"), joined into
            # one label as stacked column headers are; a column label per covered column.
            axes.append([(sid, [" ".join(ref.row_headers)]) for sid, ref in both])
        for axis in axes:
            claims: dict[str, set[str]] = {}
            for sid, headers in axis:
                for header in headers:
                    if label := _short(header):
                        claims.setdefault(label, set()).add(sid)
            if len(claims) < 2:
                continue
            unique = _unique(list(claims))
            members = [(u, frozenset(ids)) for u, ids in zip(unique, claims.values(), strict=True)]
            groups.append(_Group(0, depths[table_id], list(cells[0].heading_trail), members))
    return groups


def _component_groups(parsed: ParsedDocument) -> list[_Group]:
    """Runs of same-shaped siblings, and sibling sections that each start with a heading."""
    by_component: dict[str, list[str]] = {}
    for s in parsed.statements.values():
        by_component.setdefault(s.component_id, []).append(s.id)
    groups: list[_Group] = []

    def group(rule: int, depth: int, members: list[Component]) -> None:
        found: list[tuple[str, frozenset[str]]] = []
        for member in members:
            label = _label(member)
            ids = frozenset(sid for c in member.walk() for sid in by_component.get(c.id, ()))
            if label and ids:
                found.append((label, ids))
        if len(found) >= 2:
            labels = _unique([label for label, _ in found])
            members_ = [(u, ids) for u, (_, ids) in zip(labels, found, strict=True)]
            groups.append(_Group(rule, depth, list(members[0].heading_trail), members_))

    def visit(parent: Component, depth: int) -> None:
        shapes: dict[tuple[str, tuple[str, ...]], list[Component]] = {}
        for child in parent.children:
            if child.type in _CONTAINERS and len(child.children) >= 2:
                shapes.setdefault(_shape(child), []).append(child)
        runs = [same for same in shapes.values() if len(same) >= 2]
        for same in runs:
            group(1, depth + 1, same)
        # Headed sections of different shapes ("SE" with one paragraph, "SE L" with two).
        # When they're all one shape, the run above already asks about them.
        headed = [c for c in parent.children if _is_headed(c)]
        ids = {c.id for c in headed}
        if len(headed) >= 2 and not any(ids <= {c.id for c in same} for same in runs):
            group(2, depth + 1, headed)
        for child in parent.children:
            visit(child, depth + 1)

    visit(parsed.root, 0)
    return groups


def _scopes(
    statements: list[Statement],
    labels: list[str],
    owners: dict[str, list[str]],
    shared: set[str],
) -> list[EntityScope]:
    """One scope per label, in document order. A component is on a scope when all its
    statements are on it alone; a table split by column is on none (its cells are)."""
    by_label: dict[str, list[str]] = {label: [] for label in labels}
    component_labels: dict[str, set[str | None]] = {}
    for s in statements:
        found = owners.get(s.id, [])
        for label in found:
            by_label[label].append(s.id)
        component_labels.setdefault(s.component_id, set()).add(
            found[0] if len(found) == 1 else None
        )
    shared_ids = [s.id for s in statements if s.id in shared]
    return [
        EntityScope(
            label=label,
            component_ids=[c for c, found in component_labels.items() if found == {label}],
            statement_ids=ids,
            shared_statement_ids=list(shared_ids),
        )
        for label, ids in by_label.items()
    ]


def _drop_bare_headings(parsed: ParsedDocument, owners: dict[str, list[str]]) -> None:
    """Drop labels that own nothing but headings (a trim heading whose listing cards are
    entities of their own), leaving those headings unclaimed."""
    headings = {c.id for c in parsed.root.walk() if c.type == "heading"}
    kept = {
        label
        for sid, labels in owners.items()
        if parsed.statements[sid].component_id not in headings
        for label in labels
    }
    for sid in list(owners):
        owners[sid] = [label for label in owners[sid] if label in kept]
        if not owners[sid]:
            del owners[sid]


def _depths(root: Component) -> dict[str, int]:
    out: dict[str, int] = {}

    def visit(c: Component, depth: int) -> None:
        out[c.id] = depth
        for child in c.children:
            visit(child, depth + 1)

    visit(root, 0)
    return out


def _shape(component: Component) -> tuple[str, tuple[str, ...]]:
    return component.type, tuple(c.type for c in component.children)


def _is_headed(component: Component) -> bool:
    return (
        component.type in _CONTAINERS
        and len(component.children) >= 2
        and component.children[0].type == "heading"
    )


def _label(component: Component) -> str:
    """A member's label: its first heading, else its first text."""
    texts = [c for c in component.walk() if c.text.strip()]
    heading = next((c for c in texts if c.type == "heading"), None)
    first = heading or (texts[0] if texts else None)
    return _short(first.text) if first else ""


def _short(text: str) -> str:
    return section_text([text], max_heading_chars=MAX_ENTITY_LABEL_CHARS)


def _unique(labels: Sequence[str]) -> list[str]:
    """``labels`` with repeats numbered ("Golf", "Golf (2)"); none is "all of them"."""
    seen: set[str] = {ALL_OPTION}
    out: list[str] = []
    for label in labels:
        unique, n = label, 1
        while unique in seen:
            n += 1
            unique = f"{label} ({n})"
        seen.add(unique)
        out.append(unique)
    return out


@dataclass
class EntityStage:
    """Sets ``SchemaRun.scopes`` for every active schema using ``resolver``.

    When the component gate ran, the resolver is given only the components that passed
    (``ParsedDocument.restricted_to``), and scopes are filtered to them as well, so
    statements from irrelevant parts of the page never reach categorisation.

    Statements the resolver left out of every scope are reported in an
    ``unassigned_statements`` event.

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
                # Only components (and statements) that passed the component gate go
                # downstream.
                for scope in run.scopes:
                    scope.component_ids = [c for c in scope.component_ids if c in relevant]
                    scope.statement_ids = [s for s in scope.statement_ids if s in view.statements]
                    scope.shared_statement_ids = [
                        s for s in scope.shared_statement_ids if s in view.statements
                    ]
            if not run.scopes:
                ctx.event(self.name, "no_entities", f"{run.name}: the resolver found no entities")
                return
            if left := _unassigned(view, run.scopes):
                ctx.event(
                    self.name,
                    "unassigned_statements",
                    f"{run.name}: {len(left)} statement(s) belong to no entity",
                    statement_ids=left,
                )

        await for_each_schema(ctx, resolve)


def _unassigned(parsed: ParsedDocument, scopes: list[EntityScope]) -> list[str]:
    """Ids of statements in the tree that no scope holds (e.g. ``MultiEntity`` past the
    Choice option limit)."""
    components = {cid for scope in scopes for cid in scope.component_ids}
    ids = {sid for s in scopes for sid in [*s.statement_ids, *s.shared_statement_ids]}
    in_tree = {c.id for c in parsed.root.walk()}
    return [
        s.id
        for s in parsed.statements.values()
        if s.component_id in in_tree and s.component_id not in components and s.id not in ids
    ]

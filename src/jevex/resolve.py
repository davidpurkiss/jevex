"""Entity resolution: which parts of a document belong to which record (spec: *Entity models*).

The stage asks the configured :class:`~jevex.interfaces.EntityResolver` for each active
schema's scopes. Everything downstream (categorising, candidates, selection) then runs
once per scope. Entities are resolved after statements are split, so a resolver can
assign single statements: a comparison table's cells column by column, or a sentence
Jev says is about one trim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from jevex._tasks import gather
from jevex.entities import EntityScope
from jevex.jev import MAX_CHOICE_OPTIONS, ChoiceAnswer, NoulAnswer, UnexpectedAnswerError
from jevex.layout import section_text
from jevex.pipeline import SchemaRun, for_each_schema
from jevex.schema import ALL_OPTION, ReservedFieldNameError, UnsupportedFieldError
from jevex.select import statement_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jevex.interfaces import EntityResolver, ParsedDocument
    from jevex.jev import JevClient, JSONContent
    from jevex.layout import Component
    from jevex.pipeline import Context
    from jevex.results import FieldMeta
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
    axis: int = 0
    """For a table's group: 0 for its column labels, 1 for its row labels."""

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
       SE"), holding its row's cells (a cell spanning rows goes to each).
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


ChildPlace = Literal[
    "table_columns", "table_rows", "section", "list_item", "column", "breakout", "table"
]
"""Where ``ParentChild`` finds children: a table's columns or rows, or components of one
type."""

_CHILD_PLACES: frozenset[str] = frozenset(get_args(ChildPlace))


@dataclass(frozen=True)
class ParentChild:
    """A parent record with child records in one of its nested-model fields: a model page
    with a column per trim. The caller says where the children live, so no questions are
    asked about structure.

    ``field`` is the parent schema's nested ``BaseModel`` field the children fill
    (``variants: list[Variant]``); by default, the schema's only one. A schema without
    that field (or, by default, without any) is resolved as one parent, as with
    :class:`SingleEntity`. ``children`` says where the children live:

    - ``"table_columns"``: in each table whose cells have row and column headers, each
      column label is a child holding the cells below it (a cell spanning columns goes to
      each); ``"table_rows"`` does the same for each row's label (its headers joined).
    - A component type (``"section"``, ``"list_item"``, ``"column"``, ``"breakout"``,
      ``"table"``): each outermost component of that type is a child, labelled by its
      first heading (or first text), with its subtree. A lone one holding others of its
      type is a wrapper (a page-wide section around the trim sections, or the one under
      the page's first heading), so the ones inside it are the children instead. Only
      when a heading comes before it (the page's title names the parent) is it the one
      child ("SE", with "Performance" and "Dimensions" sections inside). A one-child page
      with no title before the child can't be told from a wrapper, so its sub-sections
      become the children.

    Every other statement is the parent's (labelled ``label``). Children inherit what it
    states: a child field found only in the parent's statements ("Every Kestrova has 5
    doors") is copied into each child record with ``meta.shared`` set, and a child's own
    value wins. Children are extracted with the nested model's fields and questions (see
    :meth:`~jevex.schema.SchemaSpec.child`); nested models inside a child aren't filled.
    """

    children: ChildPlace = "table_columns"
    field: str | None = None
    label: str = SINGLE_ENTITY_LABEL

    def __post_init__(self) -> None:
        if self.children not in _CHILD_PLACES:
            raise ValueError(
                f"children must be one of {sorted(_CHILD_PLACES)}, got {self.children!r}"
            )

    async def resolve(
        self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
    ) -> list[EntityScope]:
        name = self._field(schema)
        if name is None:
            return await SingleEntity(label=self.label).resolve(parsed, schema, jev)
        if self.children in ("table_columns", "table_rows"):
            axis = 0 if self.children == "table_columns" else 1
            groups = _table_groups(parsed, rows=axis == 1, min_labels=1)
            # The same label in two tables (the "SE" column of each) is one child.
            members = [m for g in groups if g.axis == axis for m in g.members]
        else:
            found = _type_members(parsed, self.children)
            unique = _unique([label for label, _ in found])
            members = [(u, ids) for u, (_, ids) in zip(unique, found, strict=True)]
        claims: dict[str, set[str]] = {}
        for label, ids in members:
            claims.setdefault(label, set()).update(ids)
        labels = _unique(list(claims), reserved=self.label)
        owners: dict[str, list[str]] = {}
        for label, ids in zip(labels, claims.values(), strict=True):
            for sid in ids:
                owners.setdefault(sid, []).append(label)  # a spanning cell is on each
        statements = list(parsed.statements.values())
        for s in statements:
            owners.setdefault(s.id, [self.label])
        parent, *children = _scopes(statements, [self.label, *labels], owners, set())
        return [
            parent,
            *(c.model_copy(update={"parent": self.label, "field": name}) for c in children),
        ]

    def _field(self, schema: SchemaSpec) -> str | None:
        """The nested-model field the children fill, or ``None`` if the schema has none."""
        nested = [f.name for f in schema.child_fields]
        if self.field is not None:
            if self.field not in {f.name for f in schema.fields}:
                return None
            if self.field not in nested:
                raise ChildFieldError(f"{schema.name}.{self.field} is not a nested model field")
            name = self.field
        elif len(nested) > 1:
            raise ChildFieldError(
                f"{schema.name} has several nested model fields ({', '.join(nested)}); "
                "say which one holds the children with ParentChild(field=...)"
            )
        elif not nested:
            return None
        else:
            name = nested[0]
        try:
            schema.child(name)
        except (UnsupportedFieldError, ReservedFieldNameError) as exc:
            raise ChildFieldError(f"{schema.name}.{name} can't hold children: {exc}") from exc
        return name


class ChildFieldError(ValueError):
    """``ParentChild`` can't fill the field for children: the one named isn't a nested
    model, none is named and the schema has several, or jevex can't extract the nested
    model's fields."""


def _type_members(parsed: ParsedDocument, kind: str) -> list[tuple[str, frozenset[str]]]:
    """(label, statement ids) for each outermost component of type ``kind``, looking
    inside a lone one that wraps others of its type."""
    by_component: dict[str, list[str]] = {}
    for s in parsed.statements.values():
        by_component.setdefault(s.component_id, []).append(s.id)

    def outermost(component: Component) -> list[Component]:
        found: list[Component] = []
        for child in component.children:
            found.extend([child] if child.type == kind else outermost(child))
        return found

    found = outermost(parsed.root)
    # A lone one under the page's title is a child with sub-parts ("SE" with "Performance"
    # inside); one that holds the title (or has no heading) wraps the children. The
    # heading trail is the layout's, so a gated-out title still counts.
    while len(found) == 1 and (inner := outermost(found[0])):
        lone = found[0]
        if _starts_with_heading(lone) and lone.children[0].heading_trail:
            break
        found = inner
    members: list[tuple[str, frozenset[str]]] = []
    for component in found:
        ids = frozenset(sid for c in component.walk() for sid in by_component.get(c.id, ()))
        if (label := _label(component)) and ids:
            members.append((label, ids))
    return members


def _table_groups(parsed: ParsedDocument, *, rows: bool, min_labels: int = 2) -> list[_Group]:
    """Per table whose cells have row and column headers: its column labels as one group,
    and (with ``rows``) its row labels as another, each when there are ``min_labels`` or
    more."""
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
            # one label as stacked column headers are. A cell spanning rows goes to each.
            axes.append([(sid, ref.row_labels) for sid, ref in both])
        for axis_no, axis in enumerate(axes):
            claims: dict[str, set[str]] = {}
            for sid, headers in axis:
                for header in headers:
                    if label := _short(header):
                        claims.setdefault(label, set()).add(sid)
            if len(claims) < min_labels:
                continue
            unique = _unique(list(claims))
            members = [(u, frozenset(ids)) for u, ids in zip(unique, claims.values(), strict=True)]
            trail = list(cells[0].heading_trail)
            groups.append(_Group(0, depths[table_id], trail, members, axis=axis_no))
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


def _starts_with_heading(component: Component) -> bool:
    return bool(component.children) and component.children[0].type == "heading"


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


def _unique(labels: Sequence[str], reserved: str | None = None) -> list[str]:
    """``labels`` with repeats numbered ("Golf", "Golf (2)"); none is "all of them" or
    ``reserved`` (a parent's label)."""
    seen: set[str] = {ALL_OPTION} if reserved is None else {ALL_OPTION, reserved}
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

    Values recorded for the whole document before entities were known (embedded data)
    move onto the scopes when none of them is the document
    (:func:`place_document_values`, a ``document_values_placed`` event): each to the
    entity an embedded object names, else shared by every entity.

    Without a parsed document (no layout stage ran), every schema gets one empty scope,
    labelled as the resolver would label a single entity, so structured-data-only
    pipelines still produce a record.

    Child scopes (from :class:`ParentChild`: ``field`` set) don't stay on the schema's
    run. Each nested-model field with children gets a run of its own in ``ctx.schemas``
    (named ``"<Parent>.<field>"``, with ``SchemaRun.parent`` set), holding a copy of every
    parent scope and the children, so later stages extract the nested model's fields
    from both without knowing about children; records then inherit the parent scope's
    values (see :class:`~jevex.extractor.ExtractionResult`). Its ``component_ids`` are
    the component gate's results for the nested model's own field groups
    (``SchemaRun.child_component_ids``), so a component is categorised only for the nested
    fields it passed for; without them (no gate ran), every nested field may come from any
    component. A field holding one model (not a list) keeps each parent's first child and
    reports the rest in an ``extra_children`` event.
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
                self._place_document_values(ctx, run)
                return
            relevant = run.relevant_components()
            # The resolver sees only what passed the component gate, so a resolver that
            # asks Jev (MultiEntity, #27) never pays for gated-out components.
            view = parsed if relevant is None else parsed.restricted_to(relevant)
            scopes = await self.resolver.resolve(view, run.spec, ctx.jev)
            if relevant is not None:
                # Only components (and statements) that passed the component gate go
                # downstream.
                for scope in scopes:
                    scope.component_ids = [c for c in scope.component_ids if c in relevant]
                    scope.statement_ids = [s for s in scope.statement_ids if s in view.statements]
                    scope.shared_statement_ids = [
                        s for s in scope.shared_statement_ids if s in view.statements
                    ]
            run.scopes = [s for s in scopes if s.field is None]
            self._place_document_values(ctx, run)
            children.extend(self._child_runs(ctx, run, scopes))
            if not scopes:
                ctx.event(self.name, "no_entities", f"{run.name}: the resolver found no entities")
                return
            if left := _unassigned(view, scopes):
                ctx.event(
                    self.name,
                    "unassigned_statements",
                    f"{run.name}: {len(left)} statement(s) belong to no entity",
                    statement_ids=left,
                )

        children: list[SchemaRun] = []
        await for_each_schema(ctx, resolve)
        for child in children:
            ctx.schemas[child.name] = child

    def _place_document_values(self, ctx: Context, run: SchemaRun) -> None:
        """Give the entities what earlier routes found for the whole document (embedded
        data), so it isn't a record of its own: see :func:`place_document_values`."""
        placed = place_document_values(run)
        if placed is not None:
            matched, shared = placed
            ctx.event(
                self.name,
                "document_values_placed",
                f"{run.name}: {matched} value(s) matched to an entity, "
                f"{shared} shared by every entity",
                matched=matched,
                shared=shared,
            )

    def _child_runs(
        self, ctx: Context, run: SchemaRun, scopes: list[EntityScope]
    ) -> list[SchemaRun]:
        """A run per nested-model field the resolver found children for."""
        parents = [s for s in scopes if s.field is None]
        by_field: dict[str, list[EntityScope]] = {}
        for scope in scopes:
            if scope.field is not None:
                by_field.setdefault(scope.field, []).append(scope)
        out: list[SchemaRun] = []
        for name, found in by_field.items():
            _check_children(run, name, parents, found)
            spec = run.spec.child(name)
            if not run.spec.field(name).many:
                found = self._first_children(ctx, spec.name, found)
            child = SchemaRun(
                spec,
                component_ids=run.child_component_ids.get(name),
                parent=run.name,
                parent_field=name,
            )
            child.scopes = [*(p.model_copy(deep=True) for p in parents), *found]
            out.append(child)
        return out

    def _first_children(
        self, ctx: Context, run_name: str, children: list[EntityScope]
    ) -> list[EntityScope]:
        """Each parent's first child, for a field that holds one model."""
        by_parent: dict[str | None, list[EntityScope]] = {}
        for child in children:
            by_parent.setdefault(child.parent, []).append(child)
        for kids in by_parent.values():
            if len(kids) > 1:
                ctx.event(
                    self.name,
                    "extra_children",
                    f"{run_name} holds one record: kept {kids[0].label!r}, "
                    f"left out {len(kids) - 1} more",
                    labels=[k.label for k in kids[1:]],
                )
        return [kids[0] for kids in by_parent.values()]


def place_document_values(run: SchemaRun) -> tuple[int, int] | None:
    """Move the values recorded for the whole document (on :data:`SINGLE_ENTITY_LABEL`,
    by the structured stage, before entities are known) onto ``run``'s scopes, when
    none of them is the document.

    With one scope, they become its own. With more, each field's value goes:

    - to an entity that one of :attr:`~jevex.pipeline.SchemaRun.structured_items` names
      (:func:`match_label`), from the first such item that gives the field (an item
      holding another matched item isn't matched itself);
    - to every other entity, marked ``shared`` (so a value of the entity's own replaces
      it, :meth:`~jevex.pipeline.SchemaRun.offer_field`): the document's value, unless it
      came from an item matched to an entity; then the value from outside every item
      (:attr:`~jevex.pipeline.SchemaRun.structured_rest`), else from the first item
      matched to no entity (and not inside one that was).

    Returns how many values went to an entity of its own and how many were shared (each
    entity counted), or ``None`` when there was nothing to move.
    """
    labels = [s.label for s in run.scopes]
    page = run.fields.get(SINGLE_ENTITY_LABEL)
    if not page or not labels or SINGLE_ENTITY_LABEL in labels:
        return None
    del run.fields[SINGLE_ENTITY_LABEL]
    if len(labels) == 1:
        for name, meta in page.items():
            run.set_field(labels[0], name, meta)
        return sum(m.found for m in page.values()), 0
    owner = {item.path: match_label(item.names, labels) for item in run.structured_items}
    named = [i for i in run.structured_items if owner[i.path] is not None]
    # An object holding a matched object (a vehicle and its offers) isn't matched itself:
    # its other values are the page's, and the inner objects' are theirs.
    matched = [i for i in named if not any(o.statement_ids < i.statement_ids for o in named)]
    for outer in {i.path for i in named} - {i.path for i in matched}:
        owner[outer] = None
    taken = {sid for item in matched for sid in item.statement_ids}
    loose = [
        i for i in run.structured_items if owner[i.path] is None and not i.statement_ids & taken
    ]
    counts = [0, 0]
    for name, meta in page.items():
        own: dict[str, FieldMeta] = {}
        for item in matched:
            found = item.fields.get(run.name, {}).get(name)
            label = owner[item.path]
            if found is not None and found.found and label is not None:
                own.setdefault(label, found)
        shared: FieldMeta | None = meta
        if meta.found and meta.source is not None and meta.source.statement_id in taken:
            others = [
                run.structured_rest.get(name),
                *(i.fields.get(run.name, {}).get(name) for i in loose),
            ]
            shared = next((m for m in others if m is not None and m.found), None)
        for label in labels:
            if label in own:
                run.set_field(label, name, own[label])
                counts[0] += 1
            elif shared is not None:
                run.set_field(label, name, shared.model_copy(update={"shared": True}))
                counts[1] += shared.found
    return counts[0], counts[1]


def match_label(names: Sequence[str], labels: Sequence[str]) -> str | None:
    """The entity label one of ``names`` names, if exactly one is best.

    Words are compared ignoring case and punctuation. A name equal to a label is best;
    otherwise the longest label found whole within a name ("SE L" in "Kestrova SE L",
    over "SE"). A tie, or no label in any name, gives ``None``: a name inside a label
    ("SE" in "Kestrova SE") isn't enough, since a model's name is inside all of them.
    """
    best: tuple[bool, int] | None = None
    found: set[str] = set()
    for name in names:
        words = _words(name)
        for label in labels:
            target = _words(label)
            if not target or not _contains(words, target):
                continue
            rank = (words == target, len(target))
            if best is None or rank > best:
                best, found = rank, {label}
            elif rank == best:
                found.add(label)
    return next(iter(found)) if len(found) == 1 else None


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _contains(words: list[str], part: list[str]) -> bool:
    n = len(part)
    return any(words[i : i + n] == part for i in range(len(words) - n + 1))


class InvalidScopeError(ValueError):
    """A resolver returned child scopes that can't be extracted: the field isn't one of
    the schema's nested models, the parent isn't one of its scopes, or labels repeat."""


def _check_children(
    run: SchemaRun, name: str, parents: list[EntityScope], children: list[EntityScope]
) -> None:
    if name not in {f.name for f in run.spec.child_fields}:
        raise InvalidScopeError(f"{run.name}.{name} is not a nested model field")
    labels = [s.label for s in [*parents, *children]]
    if len(set(labels)) < len(labels):
        raise InvalidScopeError(f"{run.name}: repeated entity labels in {labels}")
    known = {s.label for s in parents}
    for child in children:
        if child.parent not in known:
            raise InvalidScopeError(
                f"{run.name}: child {child.label!r} names parent {child.parent!r}, "
                f"which isn't one of its scopes"
            )


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

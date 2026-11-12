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
from jevex.schema import (
    ALL_OPTION,
    MAX_AXIS_LABELS_CHARS,
    NONE_OPTION,
    ReservedFieldNameError,
    UnsupportedFieldError,
)
from jevex.select import statement_state
from jevex.tables import header_prefix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jevex.interfaces import EntityResolver, ParsedDocument
    from jevex.jev import JevClient, JSONContent
    from jevex.keypaths import StructuredItem
    from jevex.layout import Component
    from jevex.pipeline import Context
    from jevex.results import FieldMeta
    from jevex.schema import SchemaSpec
    from jevex.statements import Statement, TableCellRef

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


@dataclass(frozen=True, eq=False)
class _Member:
    """One entity a structural rule proposes: a table's column or row, a card, a section."""

    label: str
    ids: frozenset[str]
    """The statements it claims."""
    values: dict[tuple[str, ...], str] = field(default_factory=dict[tuple[str, ...], str])
    """For a table's column or row, its cells' text (without spacing, case or the
    punctuation joining a word's parts) by their
    headers on the other axis: what it states, so two that state different things can be
    told apart (:func:`_conflict`)."""


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
    members: list[_Member]
    at: str
    """The component the group sits at: its table, or its siblings' parent."""
    axis: int = 0
    """For a table's group: 0 for its column labels, 1 for its row labels."""
    table: str | None = None
    """For a table's group: the table's component id."""

    @property
    def labels(self) -> list[str]:
        return [member.label for member in self.members]

    def state(self) -> JSONContent:
        """What Jev sees when asked which labels name entities: the labels together, so it
        can tell "SE, SE L" (trims) from "Performance, Dimensions" (topics)."""
        state: dict[str, JSONContent] = {"names": self.labels}
        if section := section_text(self.section):
            state["section"] = section
        return state


@dataclass(eq=False)
class _Entity:
    """A label taken as an entity, with every place that proposed it. ``kinds`` are the
    coarser entities it is a kind of (the "SE" a price list's "SE 1.5 TSI" row belongs
    to, the section its table sits in): only an entity nothing is a kind of becomes a
    record, and it shares what the coarser ones state."""

    label: str
    members: list[_Member]
    section: list[str]
    """The headings over the first place that proposed it."""
    values: dict[tuple[str, ...], str]
    """What its members state (:attr:`_Member.values`)."""
    kinds: list[_Entity] = field(default_factory=list["_Entity"])

    def add(self, member: _Member) -> None:
        self.members.append(member)
        self.values = {**member.values, **self.values}

    def is_kind_of(self, other: _Entity) -> bool:
        """Whether ``other`` is this entity or one of the coarser ones above it."""
        seen: set[int] = set()
        stack: list[_Entity] = [self]
        while stack:
            entity = stack.pop()
            if entity is other:
                return True
            if id(entity) not in seen:
                seen.add(id(entity))
                stack.extend(entity.kinds)
        return False


def _conflict(a: dict[tuple[str, ...], str], b: dict[tuple[str, ...], str]) -> bool:
    """Whether two table rows (or columns) state different things under one header: two
    price-list rows with their own "Price" are two entities, whatever their labels."""
    return any(key in b and b[key] != text for key, text in a.items())


@dataclass(frozen=True)
class MultiEntity:
    """Finds each entity (a trim, a listing) a document holds: the car finder's default.

    Boundary detection, in the spec's priority order:

    1. **Table headers:** in a table whose cells have both row and column headers, each
       column label is proposed as an entity, holding the cells below it (a cell spanning
       columns goes to each). In a table with column headers, so is each row's label,
       holding its row's cells (a cell spanning rows goes to each): its headers joined
       ("Kestrova SE"), then as many of its cells as it takes to tell it from the other
       rows, so a price list's rows are proposed whether or not they have header cells
       ("Icon", "ALLURE HYBRID 136 e-DSC6"). Rows with one label under different bands
       are one row ("Power" under "Performance" in two parts of a table) unless they
       state different things under one column header, as a price list's do (an engine's
       row under each grade's band, each with its own price): then the band is part of
       each row's label ("EXPRESSION TCe 110").
    2. **Repeated sibling structures:** children of one parent with the same shape (their
       type and their children's types), such as listing cards, each labelled by their
       first heading (or first text).
    3. **Headed sections:** sibling sections that each start with a heading ("SE",
       "SE L"), each with its subtree.

    Structure alone can't tell trims from topics ("Performance" and "Dimensions" sections
    look just like "SE" and "SE L" ones, and a table's rows are as likely to be trims as
    its columns), so each proposed label is checked with one Noul
    (:meth:`~jevex.schema.SchemaSpec.boundary_question`), a group's labels in one
    request. Labels at ``boundary_p`` or above are entities. A label judged among its
    axis's labels alone can't tell a price list's rows (variants) from an engine table's
    (shared by the variants), nor which axis holds the entities where labels on both were
    accepted (trims across the top, paint colours down the side), so one Choice per
    table with accepted labels (:meth:`~jevex.schema.SchemaSpec.axis_question`), the
    table as its state, keeps the columns, the rows or neither. A table with labels
    accepted on both axes is asked about after the others, with the entities found
    elsewhere that its labels name in the state too. With ``confirm=False``
    nothing is asked and every proposed label is an entity, except table rows (only
    columns are proposed): use it only for pages whose structure is known to be one
    entity per group, since otherwise topic sections split the record.

    **One entity in several places.** The same label in two places (the "SE" column of
    two tables) is one entity: a page that splits one comparison over several tables
    repeats its headers in each. Not when the two state different things under one
    header (two models' price lists, each with a "techno" row and its own price): those
    are two entities, told apart by the heading over each, else numbered (as two cards
    or sections with the same label in one run are, "Golf (2)"). A place whose label fits
    several such entities (an equipment table's "techno" column, with nothing to say
    which model's) is neither of them: it is an entity over them, and they are kinds of
    it.

    **Kinds.** A document often names its entities at several levels: a price list's rows
    are the variants ("SE 1.5 TSI 150 DSG"), an equipment table's columns their trims
    ("SE") and a specification table's their engines ("1.5 TSI 150"). A trim or an
    engine isn't a record of its own: what its column states is true of each variant of
    that kind. So an entity that another is a kind of is not a record, and its
    statements go on the ``shared_statement_ids`` of the entities that are kinds of it
    (through however many levels). An entity is a kind of the section or card it sits
    inside. Which label describes which is a judgement, so otherwise it's Jev's: one
    Choice (:meth:`~jevex.schema.SchemaSpec.kind_question`) per entity and group of
    labels from elsewhere, its pick taken at ``KIND_P`` or above. Jev says the two are
    related more reliably than which is the narrower (asked about a trim, it picks one
    of the trim's variants), so where one label has the other's words and more, that one
    is the kind ("SE 1.5 TSI" of "SE"), whichever picked the other; two with the same
    words are one entity, unless others that picked one of them state different things
    (two rows with their own prices picking one column): then that one is over them, and
    they are kinds of it. Otherwise the one picking is the kind, and of two that pick
    each other, the one more others pick is the coarser; with nothing between them they
    are one entity ("techno+ (incl canvas roof)" and "techno+ plein sud canvas roof").
    Entities that state different things under one header are never asked about each
    other (two price lists' rows). The search goes outwards from each part of the page,
    and a label that something in its own part is a kind of isn't offered outside it:
    one model's trims aren't the next model's, whatever they're called.

    When accepted groups overlap, a statement goes to the innermost (a table's columns
    beat the section around it), then to the higher priority above (a table beats headed
    sections). A label left with nothing but its own heading or header, and nothing that
    is a kind of it, is dropped.

    With two or more entities, a statement no boundary claimed applies to the entities
    of the nearest part of the page around it that has any (a model's section, else the
    page): a heading to all of them, and for any other, one Choice
    (:meth:`~jevex.schema.SchemaSpec.entity_question`) says which one, or "all of them",
    unless there is only one. A table's header statements (``table_header``) join their
    column's or row's entity; one no boundary claimed in a table whose other axis is the
    entities (a row label when the columns are) applies to all of them without asking.
    Those go on the entities' ``shared_statement_ids``: a value found only in them is
    copied into each record with ``meta.shared`` set. Where there are too many entities
    for a Choice's options (254), those statements are left out (the entity stage
    reports them as ``unassigned_statements``). With fewer than two entities, the
    document is one entity labelled ``label``, as with :class:`SingleEntity`, and
    nothing more is asked.
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
        if self.confirm:
            accepted = await self._one_axis_per_table(parsed, accepted, schema, jev)
        kinds = _Kinds(accepted, schema if self.confirm else None, jev)
        await kinds.relate(parsed.root)
        # The innermost group first, then the higher priority. A statement goes to every
        # label of the first group claiming it: a table cell spanning two columns is on
        # both.
        ranked = sorted(accepted, key=lambda a: (-a[0].depth, a[0].rule))
        owners: dict[str, list[_Entity]] = {}
        winner: dict[str, _Group] = {}
        for group, member in ranked:
            entity = kinds.entity_of[member]
            for sid in member.ids:
                if winner.setdefault(sid, group) is group and entity not in owners.get(sid, []):
                    owners.setdefault(sid, []).append(entity)
        in_tree = {c.id for c in parsed.root.walk()}
        statements = [s for s in parsed.statements.values() if s.component_id in in_tree]
        records = _records(parsed, statements, owners)
        if len(records) < 2:
            return await SingleEntity(label=self.label).resolve(parsed, schema, jev)
        _name_apart(records)
        own: dict[str, list[_Entity]] = {}
        shared: dict[str, list[_Entity]] = {}
        for s in statements:
            for entity in owners.get(s.id, []):
                if entity in records:
                    own.setdefault(s.id, []).append(entity)
                else:  # a coarser entity: what it states is true of each record of its kind
                    held = shared.setdefault(s.id, [])
                    held.extend(r for r in records if r.is_kind_of(entity) and r not in held)
        # (table, axis) for each table axis whose labels are entities: 0 columns, 1 rows.
        entity_axes = {(g.table, g.axis) for g, _ in accepted if g.table is not None}
        loose = [s for s in statements if not own.get(s.id) and not shared.get(s.id)]
        await self._assign(parsed, loose, records, own, shared, schema, jev, entity_axes)
        return _scopes(
            statements,
            [r.label for r in records],
            {sid: [r.label for r in found] for sid, found in own.items()},
            {sid: [r.label for r in found] for sid, found in shared.items()},
        )

    async def _accepted(
        self, groups: list[_Group], schema: SchemaSpec, jev: JevClient
    ) -> list[tuple[_Group, _Member]]:
        """(group, member) for every label taken as an entity."""
        if not self.confirm:
            return [(g, member) for g in groups for member in g.members]

        async def ask(group: _Group) -> list[tuple[_Group, _Member]]:
            questions = {
                f"label{i}": schema.boundary_question(label) for i, label in enumerate(group.labels)
            }
            answers = await jev.ask(group.state(), questions)
            out: list[tuple[_Group, _Member]] = []
            for i, member in enumerate(group.members):
                answer = answers[f"label{i}"]
                if not isinstance(answer, NoulAnswer):
                    raise UnexpectedAnswerError(f"expected a Noul answer, got {answer.type}")
                if answer.p >= self.boundary_p:
                    out.append((group, member))
            return out

        found = await gather(ask(g) for g in groups)
        return [a for per_group in found for a in per_group]

    async def _one_axis_per_table(
        self,
        parsed: ParsedDocument,
        accepted: list[tuple[_Group, _Member]],
        schema: SchemaSpec,
        jev: JevClient,
    ) -> list[tuple[_Group, _Member]]:
        """``accepted`` with at most one axis per table, the one Jev says names the
        entities. Each label was judged among its axis's labels alone, which can't tell a
        price list's rows (variants) from an engine table's or an options table's (shared by
        the variants), nor, with labels accepted on both axes (trims across the top, paint
        colours down the side), which axis holds the entities; the losing axis would
        otherwise claim the cells of the columns the winning one rejected (a paint's
        price). So one Choice per table with accepted labels
        (:meth:`~jevex.schema.SchemaSpec.axis_question`), with the table's text as the
        state, keeps one axis or neither.

        Where both axes have accepted labels, Jev is torn between two sets of plausible
        names (and takes a paint table's long row labels over its bare trims). What tells
        them apart is on the rest of the page: the trims are the trims of the price list.
        So those tables are asked about after the others, with the entities found
        elsewhere that their labels name (:func:`_named_elsewhere`) in the state."""
        axes: dict[str, dict[int, list[str]]] = {}
        for group, member in accepted:
            if group.table is not None:
                axes.setdefault(group.table, {}).setdefault(group.axis, []).append(member.label)
        if not axes:
            return accepted

        async def ask(table_id: str, elsewhere: Sequence[str] = ()) -> tuple[str, set[int]]:
            labels = axes[table_id]
            question = schema.axis_question(labels.get(0, []), labels.get(1, []))
            table = parsed.root.find(table_id)
            assert table is not None  # a group's table is in the tree it was found in
            section = section_text(table.heading_trail)
            named = _named_elsewhere([*labels.get(0, []), *labels.get(1, [])], elsewhere)

            def state(text: str) -> JSONContent:
                around: dict[str, JSONContent] = {"table": text}
                if section:
                    around["section"] = section
                if named:
                    around[schema.entities_elsewhere] = list(named)
                return around

            text = jev.fit_state(table.text, {"axis": question}, within=state)
            answer = (await jev.ask(state(text), {"axis": question}))["axis"]
            if not isinstance(answer, ChoiceAnswer):
                raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
            entities = {"columns": {0}, "rows": {1}}.get(answer.choice, set[int]())
            return table_id, {0, 1} - entities

        def kept(dropped: dict[str, set[int]]) -> list[tuple[_Group, _Member]]:
            # A table not asked about yet has neither axis.
            return [
                (group, member)
                for group, member in accepted
                if group.table is None or group.axis not in dropped.get(group.table, {0, 1})
            ]

        dropped = dict(await gather(ask(t) for t in axes if len(axes[t]) == 1))
        elsewhere = [member.label for _, member in kept(dropped)]
        dropped.update(await gather(ask(t, elsewhere) for t in axes if len(axes[t]) == 2))
        return kept(dropped)

    async def _assign(
        self,
        parsed: ParsedDocument,
        loose: list[Statement],
        records: list[_Entity],
        own: dict[str, list[_Entity]],
        shared: dict[str, list[_Entity]],
        schema: SchemaSpec,
        jev: JevClient,
        entity_axes: set[tuple[str, int]],
    ) -> None:
        """Give each unclaimed statement to the records it applies to, filling ``own`` and
        ``shared``. They are among the records of the nearest part of the page around it
        that has any: its own component, else each one above it in turn.

        A heading applies to all of them, as does a header on the axis across a table's
        entities (``entity_axes``: a row label, "Power", over every trim's column), and
        any statement where there is only one. Every other statement is asked about: one
        of them (its own), or all of them. So is any other unclaimed header (one on the
        entities' own axis that Jev didn't accept, or in a table with no entity axis)."""
        parents = {child.id: c.id for c in parsed.root.walk() for child in c.children}
        headings = {c.id for c in parsed.root.walk() if c.type == "heading"}
        # The records each component's statements are on, with those of everything in it.
        inside: dict[str, list[_Entity]] = {}
        for sid in [*own, *shared]:
            found = [*own.get(sid, []), *shared.get(sid, [])]
            cid: str | None = parsed.statements[sid].component_id
            while cid is not None:
                held = inside.setdefault(cid, [])
                held.extend(r for r in found if r not in held)
                cid = parents.get(cid)

        def around(statement: Statement) -> list[_Entity]:
            cid: str | None = statement.component_id
            while cid is not None and not inside.get(cid):
                cid = parents.get(cid)
            found = records if cid is None else inside[cid]
            return [r for r in records if r in found]  # in document order

        asked: list[tuple[Statement, list[_Entity]]] = []
        for s in loose:
            scope = around(s)
            if len(scope) == 1 or s.component_id in headings or _across_entities(s, entity_axes):
                shared[s.id] = scope
            elif len(scope) < MAX_CHOICE_OPTIONS:
                asked.append((s, scope))

        async def ask(statement: Statement, scope: list[_Entity]) -> str:
            question = schema.entity_question([r.label for r in scope])
            answer = (await jev.ask(statement_state(statement), {"entity": question}))["entity"]
            if not isinstance(answer, ChoiceAnswer):
                raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
            return answer.choice

        choices = await gather(ask(s, scope) for s, scope in asked)
        for (s, scope), choice in zip(asked, choices, strict=True):
            if choice == ALL_OPTION:
                shared[s.id] = scope
            else:
                own[s.id] = [r for r in scope if r.label == choice]


KIND_P = 0.7
"""The probability of the kind Choice's pick at or above which an entity is taken as a
kind of the one picked. On the real spec-sheets corpus's 18 development documents
(#326), right picks were almost all above 0.8, and those under 0.7 were mostly a
variant picking one of a run of page headings, or a trim one of its own variants."""


class _Kinds:
    """Works out which accepted labels are one entity, and which is a kind of which (see
    :class:`MultiEntity`), part by part from the innermost outwards. Without a ``schema``
    nothing is asked: only what the structure says."""

    def __init__(
        self, accepted: list[tuple[_Group, _Member]], schema: SchemaSpec | None, jev: JevClient
    ) -> None:
        self.schema = schema
        self.jev = jev
        self.entity_of: dict[_Member, _Entity] = {}
        by_group: dict[int, tuple[_Group, list[_Member]]] = {}
        for group, member in accepted:
            by_group.setdefault(id(group), (group, []))[1].append(member)
        self._at: dict[str, list[tuple[_Group, list[_Member]]]] = {}
        for group, members in by_group.values():
            self._at.setdefault(group.at, []).append((group, members))
        self._coarse: set[_Entity] = set()
        self._asked: set[tuple[_Entity, _Entity]] = set()
        self._into: dict[_Entity, _Entity] = {}
        """Each entity made part of another (:meth:`_same`), and which."""

    async def relate(self, component: Component) -> list[list[_Entity]]:
        """The sets of entities in ``component`` that nothing in it is a kind of, once
        everything in it is related: what its parent relates to the rest of the page. A
        set is one group's entities (a table's columns, a run of cards)."""
        below = await gather(self.relate(c) for c in component.children)
        sets = [found for child in below for found in child]
        here = self._at.get(component.id, [])
        holders: list[_Member] = []
        for group, members in here:
            found = [_Entity(m.label, [m], group.section, dict(m.values)) for m in members]
            self.entity_of.update(zip(members, found, strict=True))
            if group.table is None:
                holders += members
            sets.append(found)
        self._unify(sets)
        # A card or section holds whatever its statements hold.
        for member in holders:
            outer = self.entity_of[member]
            for inner in dict.fromkeys(e for found in sets for e in found):
                held = frozenset(sid for m in inner.members for sid in m.ids)
                if inner is not outer and held <= member.ids:
                    self._kind(inner, outer)
        sets = _distinct(sets)
        # Only where parts of the page meet is there anything new to ask.
        if self.schema is not None and len([c for c in below if c]) + len(here) > 1:
            await self._ask(sets, self.schema)
        return _distinct([[e for e in found if e not in self._coarse] for found in sets])

    def _unify(self, sets: list[list[_Entity]]) -> None:
        """Make the entities in ``sets`` with one label one entity, unless they state
        different things under one header. One that fits several that differ from each
        other (an equipment table's "SE" column beside two models' "SE" rows, each with
        its own price) can't be said to be either of them: it stays an entity of its own,
        and they are kinds of it."""
        by_label: dict[str, list[_Entity]] = {}
        for entity in dict.fromkeys(e for found in sets for e in found):
            by_label.setdefault(entity.label, []).append(entity)
        for same in by_label.values():
            # Those that differ from another first, so the rest are placed knowing them
            # all, whatever order the page has them in.
            apart = [e for e in same if any(_conflict(e.values, o.values) for o in same)]
            distinct: list[_Entity] = []
            over: list[_Entity] = []  # those that fit several of the distinct ones
            for entity in [*apart, *(e for e in same if e not in apart)]:
                fits = [e for e in distinct if not _conflict(e.values, entity.values)]
                if len(fits) == 1:
                    self._same(sets, fits[0], entity)
                elif not fits:
                    distinct.append(entity)
                else:
                    held = next((e for e in over if not _conflict(e.values, entity.values)), None)
                    if held is None:
                        over.append(held := entity)
                    else:
                        self._same(sets, held, entity)
                    for narrower in fits:
                        self._kind(narrower, held)

    def _same(self, sets: list[list[_Entity]], entity: _Entity, other: _Entity) -> None:
        """Make ``other`` part of ``entity``: two places, or two labels, for one thing."""
        for member in other.members:
            entity.add(member)
            self.entity_of[member] = entity
        self._into[other] = entity
        for e in set(self.entity_of.values()):
            kinds = dict.fromkeys(entity if k is other else k for k in e.kinds)
            e.kinds = [k for k in kinds if k is not e]
        for of in other.kinds:
            if of is not entity:
                self._kind(entity, of)
        if other in self._coarse:
            self._coarse.add(entity)
        self._asked |= {
            (entity if a is other else a, entity if b is other else b)
            for a, b in self._asked
            if other in (a, b)
        }
        for found in sets:
            found[:] = list(dict.fromkeys(entity if e is other else e for e in found))

    def _now(self, entity: _Entity) -> _Entity:
        """The entity ``entity`` is part of by now."""
        while entity in self._into:
            entity = self._into[entity]
        return entity

    def _kind(self, entity: _Entity, of: _Entity) -> None:
        if of not in entity.kinds and not of.is_kind_of(entity):
            entity.kinds.append(of)
            self._coarse.add(of)

    async def _ask(self, sets: list[list[_Entity]], schema: SchemaSpec) -> None:
        """Ask Jev, of each entity in ``sets``, which entity of each other set it is a kind
        of, and record the answers."""
        asks: list[tuple[_Entity, list[list[_Entity]]]] = []
        for entity in dict.fromkeys(e for found in sets for e in found):
            offered: list[list[_Entity]] = []
            for found in sets:
                # Not its own set, nor one it is already related to (the cards inside
                # the section it is).
                if entity in found or any(
                    e.is_kind_of(entity) or entity.is_kind_of(e) for e in found
                ):
                    continue
                options = [
                    e
                    for e in found
                    if (entity, e) not in self._asked and not _conflict(entity.values, e.values)
                ]
                if options and len(options) < MAX_CHOICE_OPTIONS:
                    offered.append(options)
                    self._asked.update((entity, e) for e in options)
            if offered:
                asks.append((entity, offered))

        async def ask(entity: _Entity, offered: list[list[_Entity]]) -> list[_Entity]:
            questions = {
                f"kind{i}": schema.kind_question([e.label for e in options])
                for i, options in enumerate(offered)
            }
            answers = await self.jev.ask({schema.entity_name: entity.label}, questions)
            picked: list[_Entity] = []
            for i, options in enumerate(offered):
                answer = answers[f"kind{i}"]
                if not isinstance(answer, ChoiceAnswer):
                    raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
                if answer.probabilities.get(answer.choice, 0.0) >= KIND_P:
                    picked.extend(e for e in options if e.label == answer.choice)
            return picked

        answers = await gather(ask(entity, offered) for entity, offered in asks)
        picks = [(e, of) for (e, _), found in zip(asks, answers, strict=True) for of in found]
        pickers: dict[_Entity, int] = {}
        for _, of in picks:
            pickers[of] = pickers.get(of, 0) + 1

        def differing(entity: _Entity, of: _Entity) -> set[_Entity]:
            """Of the two and the others that picked either, those that state something
            different from another of them."""
            near = {self._now(e) for e in (entity, of)}
            near |= {self._now(a) for a, b in picks if b in (entity, of)}
            return {e for e in near if any(_conflict(e.values, o.values) for o in near)}

        for entity, of in picks:
            now, now_of = self._now(entity), self._now(of)
            # Nothing new where they are already one or related, and what was put
            # together since they were asked about may state something different.
            if now.is_kind_of(now_of) or now_of.is_kind_of(now):
                continue
            if _conflict(now.values, now_of.values):
                continue
            words, words_of = _words(entity.label), _words(of.label)
            if words == words_of:
                # One label, written two ways: one entity, unless one of them is over
                # several that differ (two rows with their own prices, "techno" and
                # "techno plus", both picking a "TECHNO" column).
                differ = differing(entity, of)
                if now in differ and now_of not in differ:
                    self._kind(now, now_of)
                elif now_of in differ and now not in differ:
                    self._kind(now_of, now)
                elif now not in differ:
                    self._same(sets, now, now_of)
            elif words < words_of:
                self._kind(now_of, now)  # "SE" picked "SE 1.5 TSI": that is the narrower
            elif words > words_of or (of, entity) not in picks:
                self._kind(now, now_of)
            elif pickers[of] > pickers[entity]:
                self._kind(now, now_of)  # others pick it too, so it is the coarser
            elif pickers[of] == pickers[entity]:
                self._same(sets, now, now_of)  # each picks only the other: one entity


_WORD = re.compile(r"[^\W_]+")


def _named_elsewhere(labels: Sequence[str], elsewhere: Sequence[str]) -> list[str]:
    """The entities' labels in ``elsewhere`` that one of ``labels`` names: one with the
    label's words, or those and more, or fewer ("techno" names "techno 1.2 TCe"). In
    order, each once, those that fit in ``MAX_AXIS_LABELS_CHARS`` between them.

    This only chooses what goes beside the axis question: a document can hold hundreds of
    entities, and the ones that bear on a table are those called what its headers are.
    Which axis they speak for is still Jev's to say. A label with no letters ("5") is
    left out of the matching: it is in any label with that number."""
    words = [found for found in map(_words, labels) if any(c.isalpha() for c in "".join(found))]
    named: list[str] = []
    room = MAX_AXIS_LABELS_CHARS
    for other in dict.fromkeys(elsewhere):
        theirs = _words(other)
        fits = len(other) <= room
        if fits and theirs and any(mine <= theirs or theirs <= mine for mine in words):
            room -= len(other)
            named.append(other)
    return named


def _words(label: str) -> frozenset[str]:
    """A label's words, whatever their case: of two labels Jev relates, the one with the
    other's words and more ("SE 1.5 TSI" beside "SE") names the narrower thing."""
    return frozenset(_WORD.findall(label.casefold()))


def _distinct(sets: list[list[_Entity]]) -> list[list[_Entity]]:
    """``sets`` without the empty ones or the repeats (the same columns in two tables)."""
    seen: set[frozenset[_Entity]] = set()
    out: list[list[_Entity]] = []
    for found in sets:
        key = frozenset(found)
        if key and key not in seen:
            seen.add(key)
            out.append(found)
    return out


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
      each); ``"table_rows"`` does the same for each row's label in a table with column
      headers (its headers joined, then its cells as far as they tell it from the other
      rows, and its band's name first where same-named rows under different bands state
      different things: a grade's band over each engine's row and price).
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
            members = [(m.label, m.ids) for g in groups if g.axis == axis for m in g.members]
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
        parent, *children = _scopes(statements, [self.label, *labels], owners, {})
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
    """Per table: its column labels as one group, when its cells have row and column
    headers, and (with ``rows``) its rows' labels as another, when its cells have column
    headers; each when there are ``min_labels`` or more. A row's label is its headers
    joined, then as many of its cells as it takes to tell it from the table's other rows
    (:attr:`TableCellRef.row_labels <jevex.statements.TableCellRef.row_labels>`): in a
    price list (``Trim | Engine | Price``) each row is a variant, header cells or not.
    Rows with one label under different bands are one member, unless two of them state
    different things under a column header: then every row's label starts with its band.
    A header's own statement (``table_header``) joins its label's group, but only a label
    some cell already gives: an empty column proposes no entity."""
    depths = _depths(parsed.root)
    by_table: dict[str, list[Statement]] = {}
    for s in parsed.statements.values():
        if s.table is not None and s.component_id in depths:
            by_table.setdefault(s.component_id, []).append(s)
    groups: list[_Group] = []
    for table_id, cells in by_table.items():
        # Columns only with headers on both axes: a column header alone says what the cell
        # states (a price list's "Price"), not which entity it's about. Rows need column
        # headers: a row header alone labels a key/value table's value ("Power").
        refs = [(c, c.table) for c in cells if c.table is not None]
        headed = [(s, r) for s, r in refs if s.kind == "table_cell" and r.col_headers]
        heads = [(s, r) for s, r in refs if s.kind == "table_header"]
        columns: dict[str, _Claim] = {}
        for s, ref in headed:
            if ref.row_headers:
                across = ("row", ref.group or "", *ref.row_headers)
                for header in ref.col_headers:
                    if label := _short(header):
                        columns.setdefault(label, _Claim()).cell(s, ref, across)
        for s, ref in heads:
            for header in ref.col_headers:
                if (label := _short(header)) in columns:
                    columns[label].ids.add(s.id)
        axes = [columns, _row_claims(headed, heads)] if rows else [columns]
        for axis_no, claims in enumerate(axes):
            if len(claims) < min_labels:
                continue
            unique = _unique(list(claims))
            members = [
                _Member(u, frozenset(claim.ids), claim.values)
                for u, claim in zip(unique, claims.values(), strict=True)
            ]
            trail = list(cells[0].heading_trail)
            depth = depths[table_id]
            groups.append(
                _Group(0, depth, trail, members, at=table_id, axis=axis_no, table=table_id)
            )
    return groups


_JOINER = re.compile(r"(?<=[^\W_])[-/.'’]+(?=[^\W\d_])")
"""A hyphen, slash, full stop or apostrophe joining a word to the letters after it, which
cells are compared without, as they are without spacing or case: "6 speed manual" and
"6-Speed Manual" state one thing. Before a digit it stays, where it may be a sign, a
range or a decimal point ("-5", "7-8", "1.5"), and so does every other mark."""


@dataclass
class _Claim:
    """What one label of a table's axis claims: statements, and its cells' text by their
    headers on the other axis (:attr:`_Member.values`)."""

    ids: set[str] = field(default_factory=set[str])
    values: dict[tuple[str, ...], str] = field(default_factory=dict[tuple[str, ...], str])

    def cell(self, statement: Statement, ref: TableCellRef, across: tuple[str, ...]) -> None:
        self.ids.add(statement.id)
        value = statement.text.removeprefix(header_prefix(ref))
        self.values.setdefault(across, _JOINER.sub("", "".join(value.split())).casefold())


def _row_claims(
    headed: list[tuple[Statement, TableCellRef]], heads: list[tuple[Statement, TableCellRef]]
) -> dict[str, _Claim]:
    """Each row label's claim, from a table's cells with column headers (a cell spanning
    rows goes to each) and its header statements."""
    banded: dict[tuple[str | None, str], _Claim] = {}
    for s, ref in headed:
        for row in ref.row_labels:
            if label := _short(row):
                banded.setdefault((ref.group, label), _Claim()).cell(
                    s, ref, ("column", *ref.col_headers)
                )
    for s, ref in heads:
        for row in ref.row_labels:
            if (key := (ref.group, _short(row))) in banded:
                banded[key].ids.add(s.id)
    by_label: dict[str, list[_Claim]] = {}
    for (_, label), claim in banded.items():
        by_label.setdefault(label, []).append(claim)
    apart = any(
        _conflict(a.values, b.values)
        for same in by_label.values()
        for i, a in enumerate(same)
        for b in same[i + 1 :]
    )
    claims: dict[str, _Claim] = {}
    for (band, label), claim in banded.items():
        merged = claims.setdefault(_short(f"{band} {label}") if apart and band else label, _Claim())
        merged.ids |= claim.ids
        merged.values = {**claim.values, **merged.values}
    return claims


def _component_groups(parsed: ParsedDocument) -> list[_Group]:
    """Runs of same-shaped siblings, and sibling sections that each start with a heading."""
    by_component: dict[str, list[str]] = {}
    for s in parsed.statements.values():
        by_component.setdefault(s.component_id, []).append(s.id)
    groups: list[_Group] = []

    def group(rule: int, depth: int, parent: Component, members: list[Component]) -> None:
        found: list[tuple[str, frozenset[str]]] = []
        for member in members:
            label = _label(member)
            ids = frozenset(sid for c in member.walk() for sid in by_component.get(c.id, ()))
            if label and ids:
                found.append((label, ids))
        if len(found) >= 2:
            labels = _unique([label for label, _ in found])
            members_ = [_Member(u, ids) for u, (_, ids) in zip(labels, found, strict=True)]
            trail = list(members[0].heading_trail)
            groups.append(_Group(rule, depth, trail, members_, at=parent.id))

    def visit(parent: Component, depth: int) -> None:
        shapes: dict[tuple[str, tuple[str, ...]], list[Component]] = {}
        for child in parent.children:
            if child.type in _CONTAINERS and len(child.children) >= 2:
                shapes.setdefault(_shape(child), []).append(child)
        runs = [same for same in shapes.values() if len(same) >= 2]
        for same in runs:
            group(1, depth + 1, parent, same)
        # Headed sections of different shapes ("SE" with one paragraph, "SE L" with two).
        # When they're all one shape, the run above already asks about them.
        headed = [c for c in parent.children if _is_headed(c)]
        ids = {c.id for c in headed}
        if len(headed) >= 2 and not any(ids <= {c.id for c in same} for same in runs):
            group(2, depth + 1, parent, headed)
        for child in parent.children:
            visit(child, depth + 1)

    visit(parsed.root, 0)
    return groups


def _records(
    parsed: ParsedDocument, statements: list[Statement], owners: dict[str, list[_Entity]]
) -> list[_Entity]:
    """The entities that become records, in document order: those owning more than
    headings and table headers (a trim heading whose listing cards claimed the rest names
    no record of its own) that no other such entity is a kind of."""
    headings = {c.id for c in parsed.root.walk() if c.type == "heading"}
    solid = list(
        dict.fromkeys(
            entity
            for s in statements
            if s.component_id not in headings and s.kind != "table_header"
            for entity in owners.get(s.id, [])
        )
    )
    return [e for e in solid if not any(o is not e and o.is_kind_of(e) for o in solid)]


def _name_apart(records: list[_Entity]) -> None:
    """Give records that share a label one each: with the innermost heading over it that
    isn't over them all ("RAFALE techno", "AUSTRAL techno"), else numbered."""
    by_label: dict[str, list[_Entity]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    for label, same in by_label.items():
        if len(same) > 1:
            for record in same:
                apart = [h for h in record.section if not all(h in r.section for r in same)]
                if apart:
                    record.label = _short(f"{apart[-1]} {label}")
    for record, label in zip(records, _unique([r.label for r in records]), strict=True):
        record.label = label


def _scopes(
    statements: list[Statement],
    labels: list[str],
    own: dict[str, list[str]],
    shared: dict[str, list[str]],
) -> list[EntityScope]:
    """One scope per label: the statements that are its ``own``, and those it shares with
    others (``shared``). A component is on a scope when all its statements are on it
    alone; a table split by column is on none (its cells are)."""
    ids: dict[str, list[str]] = {label: [] for label in labels}
    shared_ids: dict[str, list[str]] = {label: [] for label in labels}
    owners: dict[str, set[str | None]] = {}
    for s in statements:
        found = own.get(s.id, [])
        for label in found:
            ids[label].append(s.id)
        for label in shared.get(s.id, []):
            if label not in found:
                shared_ids[label].append(s.id)
        owners.setdefault(s.component_id, set()).add(found[0] if len(found) == 1 else None)
    return [
        EntityScope(
            label=label,
            component_ids=[c for c, found in owners.items() if found == {label}],
            statement_ids=ids[label],
            shared_statement_ids=shared_ids[label],
        )
        for label in labels
    ]


def _across_entities(statement: Statement, entity_axes: set[tuple[str, int]]) -> bool:
    """Whether ``statement`` is a header on the axis across its table's entities: a row
    header when the columns are entities, a column header when the rows are."""
    ref = statement.table
    if statement.kind != "table_header" or ref is None:
        return False
    across = 0 if ref.row_headers else 1
    return (statement.component_id, across) in entity_axes


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
    """A member's label: its first heading, else its first text.

    The first text is a guess at a name (a card's could be its price). It's only
    proposed: :class:`MultiEntity` asks Jev whether it names an entity (unless
    ``confirm=False``), and :class:`ParentChild`'s caller has said the member is one."""
    texts = [c for c in component.walk() if c.text.strip()]
    heading = next((c for c in texts if c.type == "heading"), None)
    first = heading or (texts[0] if texts else None)
    return _short(first.text) if first else ""


def _short(text: str) -> str:
    return section_text([text], max_heading_chars=MAX_ENTITY_LABEL_CHARS)


def _unique(labels: Sequence[str], reserved: str | None = None) -> list[str]:
    """``labels`` with repeats numbered ("Golf", "Golf (2)"); none is a Choice's reserved
    option ("all of them", "none") or ``reserved`` (a parent's label)."""
    seen: set[str] = {ALL_OPTION, NONE_OPTION} if reserved is None else {ALL_OPTION, reserved}
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
    entity Jev says its embedded object describes, else shared by every entity.

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
                await self._place_document_values(ctx, run)
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
            await self._place_document_values(ctx, run)
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

    async def _place_document_values(self, ctx: Context, run: SchemaRun) -> None:
        """Give the entities what earlier routes found for the whole document (embedded
        data), so it isn't a record of its own: see :func:`place_document_values`."""
        placed = await place_document_values(run, ctx.jev)
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


async def place_document_values(run: SchemaRun, jev: JevClient) -> tuple[int, int] | None:
    """Move the values recorded for the whole document (on :data:`SINGLE_ENTITY_LABEL`,
    by the structured stage, before entities are known) onto ``run``'s scopes, when
    none of them is the document.

    With one scope, they become its own. With more, Jev is asked which entity each of
    :attr:`~jevex.pipeline.SchemaRun.structured_items` giving a value describes: one
    :meth:`~jevex.schema.SchemaSpec.entity_question` Choice per object, with its leaves'
    statements as the state: the question :class:`MultiEntity` asks about an unclaimed
    statement. Then each field's value goes:

    - to the entity an object describes, from the first such object that gives the field
      (an object holding another placed object isn't placed itself: a vehicle and its
      offers);
    - to every other entity, marked ``shared`` (so a value of the entity's own replaces
      it, :meth:`~jevex.pipeline.SchemaRun.offer_field`): the document's value, unless it
      came from an object placed on an entity, or Jev found none of the document's values
      to be it while some entity has one of its own (the values disagreed because they're
      the entities'). Then it's the value from outside every object
      (:attr:`~jevex.pipeline.SchemaRun.structured_rest`, settled by Jev on its own when
      those values disagree), else from the first object that applies to every entity
      (and isn't inside a placed one), else, after Jev's "none", the document's unfound
      value.

    Past 254 entities there are too many options for a Choice, so nothing is asked and
    every object applies to every entity.

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
    items = [
        i for i in run.structured_items if any(m.found for m in i.fields.get(run.name, {}).values())
    ]
    owners = await _item_owners(items, labels, run.spec, jev)
    placed = [(i, label) for i, label in zip(items, owners, strict=True) if label is not None]
    # An object holding a placed object (a vehicle and its offers) isn't placed itself:
    # its other values are the page's, and the inner objects' are theirs.
    matched = [
        (i, label)
        for i, label in placed
        if not any(o.statement_ids < i.statement_ids for o, _ in placed)
    ]
    taken = {sid for item, _ in matched for sid in item.statement_ids}
    loose = [
        i
        for i, label in zip(items, owners, strict=True)
        if label is None and not i.statement_ids & taken
    ]
    counts = [0, 0]
    for name, meta in page.items():
        own: dict[str, FieldMeta] = {}
        for item, label in matched:
            found = item.fields[run.name].get(name)
            if found is not None and found.found:
                own.setdefault(label, found)
        shared: FieldMeta | None = meta
        placed_value = meta.found and meta.source is not None and meta.source.statement_id in taken
        if placed_value or (not meta.found and own):
            others = [
                run.structured_rest.get(name),
                *(i.fields[run.name].get(name) for i in loose),
            ]
            fallback = None if placed_value else meta
            shared = next((m for m in others if m is not None and m.found), fallback)
        for label in labels:
            if label in own:
                run.set_field(label, name, own[label])
                counts[0] += 1
            elif shared is not None:
                run.set_field(label, name, shared.model_copy(update={"shared": True}))
                counts[1] += shared.found
    return counts[0], counts[1]


async def _item_owners(
    items: list[StructuredItem], labels: list[str], schema: SchemaSpec, jev: JevClient
) -> list[str | None]:
    """The entity label Jev says each object describes, or ``None`` when it applies to
    every entity (or there are too many entities to ask). The state is the object's text
    alone, cut to fit one request (:meth:`~jevex.jev.JevClient.fit_state`, which sizes
    plain text): its first leaves say what it is."""
    if len(labels) >= MAX_CHOICE_OPTIONS:
        return [None] * len(items)
    questions = {"entity": schema.entity_question(labels)}

    async def ask(item: StructuredItem) -> str | None:
        answer = (await jev.ask(jev.fit_state(item.text, questions), questions))["entity"]
        if not isinstance(answer, ChoiceAnswer):
            raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
        return None if answer.choice == ALL_OPTION else answer.choice

    return await gather(ask(i) for i in items)


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

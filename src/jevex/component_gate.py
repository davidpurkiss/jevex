"""The component gate: which parts of a document are worth reading for each field group
(spec: *Pipeline architecture*, stage 7).

The layout tree is cut into **gate units**: each container's run of direct content
blocks (headings, paragraphs, lists, captions, images), chunked to at most ``max_chars``,
plus each table on its own. Jev is asked one Noul per unit × field group ("Does this
section contain the price?"), with every schema's questions about a unit in one request.

A unit that passes for a group passes all its components and their descendants (and
their ancestors, so the tree stays connected). The result lands on
``SchemaRun.component_ids`` (group → component ids). The entity stage then keeps only
passing components in each scope, so statements from irrelevant parts of the page never
reach categorisation, and :meth:`~jevex.pipeline.SchemaRun.relevant_fields` tells the
classifier which fields a component can state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jevex.jev import NoulAnswer, UnexpectedAnswerError

if TYPE_CHECKING:
    from jevex.interfaces import ComponentGate, ParsedDocument
    from jevex.jev import JevClient
    from jevex.layout import Component
    from jevex.pipeline import Context
    from jevex.schema import SchemaSpec

DEFAULT_THRESHOLD = 0.3
"""Lower than the document gate's: a wrongly passed section costs a few statements, a
wrongly failed one loses its values. Tuned from eval runs in #49."""

DEFAULT_MAX_CHARS = 2000

_CONTAINERS = frozenset({"section", "column", "breakout"})


@dataclass(frozen=True)
class GateUnit:
    """One thing the gate asks about: some blocks of one container, or one table."""

    id: str
    component_ids: tuple[str, ...]
    text: str
    heading_trail: tuple[str, ...]

    def state(self) -> dict[str, str]:
        state = {"content": self.text}
        if self.heading_trail:
            state["section"] = " › ".join(self.heading_trail)
        return state


def _text(component: Component) -> str:
    """A block's text including its descendants' (list items, image alt text...)."""
    parts = [c.text.strip() for c in component.walk() if c.text.strip()]
    return "\n".join(parts)


def gate_units(root: Component, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[GateUnit]:
    """The units the gate asks about, in reading order. Texts are cut at ``max_chars``."""
    units: list[GateUnit] = []

    def emit(unit_id: str, blocks: list[Component]) -> None:
        text = "\n".join(t for b in blocks if (t := _text(b)))
        if text:
            units.append(
                GateUnit(
                    id=unit_id,
                    component_ids=tuple(c.id for b in blocks for c in b.walk()),
                    text=text[:max_chars],
                    heading_trail=tuple(blocks[0].heading_trail),
                )
            )

    def visit(container: Component) -> None:
        chunk: list[Component] = []
        size = 0
        part = 0

        def close_chunk() -> None:
            nonlocal chunk, size, part
            if chunk:
                emit(f"{container.id}#{part}", chunk)
                part += 1
            chunk, size = [], 0

        for child in container.children:
            if child.type in _CONTAINERS or child.type == "table":
                close_chunk()
                if child.type == "table":
                    emit(child.id, [child])
                else:
                    visit(child)
                continue
            length = len(_text(child)) + 1
            if chunk and size + length > max_chars:
                close_chunk()
            chunk.append(child)
            size += length
        close_chunk()

    if root.type in _CONTAINERS:
        visit(root)
    else:
        emit(root.id, [root])
    return units


class NoulComponentGate:
    """The default :class:`~jevex.interfaces.ComponentGate`: one Noul per unit × group.

    A group passes a unit when ``p >= threshold``. Question text comes from
    :meth:`~jevex.schema.SchemaSpec.component_gate_questions`, so ``Questions`` overrides
    apply.
    """

    def __init__(
        self, *, threshold: float = DEFAULT_THRESHOLD, max_chars: int = DEFAULT_MAX_CHARS
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        if max_chars < 1:
            raise ValueError(f"max_chars must be positive, got {max_chars}")
        self.threshold = threshold
        self.max_chars = max_chars

    async def gate(
        self, parsed: ParsedDocument, schemas: list[SchemaSpec], jev: JevClient
    ) -> dict[str, dict[str, list[str]]]:
        units = gate_units(parsed.root, max_chars=self.max_chars)
        questions = {
            f"{s.name}.{group}": (s.name, group, q)
            for s in schemas
            for group, q in s.component_gate_questions().items()
        }
        out: dict[str, dict[str, list[str]]] = {
            s.name: {group: [] for group in s.groups} for s in schemas
        }
        if not units or not questions:
            return out
        parents = _parents(parsed.root)

        async def ask(unit: GateUnit) -> None:
            answers = await jev.ask(unit.state(), {k: q for k, (_, _, q) in questions.items()})
            for key, answer in answers.items():
                if not isinstance(answer, NoulAnswer):
                    raise UnexpectedAnswerError(
                        f"expected a Noul answer for {key!r}, got {answer.type}"
                    )
                if answer.p >= self.threshold:
                    schema, group, _ = questions[key]
                    out[schema][group].extend(unit.component_ids)

        await asyncio.gather(*(ask(u) for u in units))
        order = {c.id: i for i, c in enumerate(parsed.root.walk())}
        for groups in out.values():
            for group, ids in groups.items():
                passed = set(ids)
                for cid in ids:
                    parent = parents.get(cid)
                    while parent is not None and parent not in passed:
                        passed.add(parent)
                        parent = parents.get(parent)
                groups[group] = sorted(passed, key=order.__getitem__)
        return out


def _parents(root: Component) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in root.walk():
        for child in c.children:
            out[child.id] = c.id
    return out


@dataclass
class ComponentGateStage:
    """Runs a :class:`~jevex.interfaces.ComponentGate` for every active schema.

    Sets ``SchemaRun.component_ids``. Without a parsed document the stage does nothing,
    and ``component_ids`` stays ``None`` (nothing gated). A schema where no component
    passes any group gets a ``no_relevant_components`` event.
    """

    gate: ComponentGate = field(default_factory=NoulComponentGate)
    name: str = "component_gate"

    async def run(self, ctx: Context) -> None:
        parsed = ctx.parsed
        runs = ctx.active
        if parsed is None or not runs:
            return
        decisions = await self.gate.gate(parsed, [r.spec for r in runs], ctx.jev)
        for run in runs:
            groups = decisions.get(run.name, {})
            run.component_ids = groups
            if not any(groups.values()):
                ctx.event(
                    self.name,
                    "no_relevant_components",
                    f"{run.name}: no component passed the component gate",
                    schema=run.name,
                )

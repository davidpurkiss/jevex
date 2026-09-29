"""The component gate: which parts of a document are worth reading for each field group
(spec: *Pipeline architecture*, stage 7).

The layout tree is cut into **gate units**: each container's run of direct content
blocks (headings, paragraphs, lists, captions, images), chunked to at most ``max_chars``,
plus each table on its own. A block longer than ``max_chars`` is split across several
units (tables by row groups with their headers repeated) rather than cut, and headings
with nothing after them join the table or section they introduce. Jev is asked one Noul
per unit × field group ("Does this section contain the price?"), with every schema's
questions about a unit in one request.

A unit that passes for a group passes all its components and their descendants (and
their ancestors, so the tree stays connected). The result lands on
``SchemaRun.component_ids`` (group → component ids). The entity stage then keeps only
passing components in each scope, so statements from irrelevant parts of the page never
reach categorisation, and :meth:`~jevex.pipeline.SchemaRun.relevant_fields` tells the
classifier which fields a component can state.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jevex.jev import NoulAnswer, UnexpectedAnswerError

if TYPE_CHECKING:
    from jevex.interfaces import ComponentGate, ParsedDocument
    from jevex.jev import JevClient
    from jevex.layout import Component, TableCell
    from jevex.pipeline import Context
    from jevex.schema import SchemaSpec

DEFAULT_THRESHOLD = 0.3
"""Lower than the document gate's: a wrongly passed section costs a few statements, a
wrongly failed one loses its values. Tuned from eval runs in #49."""

DEFAULT_MAX_CHARS = 2000

_CONTAINERS = frozenset({"section", "column", "breakout"})


@dataclass(frozen=True)
class GateUnit:
    """One thing the gate asks about: some blocks of one container, or one table.

    A block too long for one state is split into several units that share its
    ``component_ids``, so the block passes if any piece does.
    """

    id: str
    component_ids: tuple[str, ...]
    text: str
    heading_trail: tuple[str, ...]

    def state(self) -> dict[str, str]:
        """What Jev sees: ``content`` is the unit's text; ``section``, when there is one,
        is its heading trail joined with " › "."""
        state = {"content": self.text}
        if self.heading_trail:
            state["section"] = " › ".join(self.heading_trail)
        return state


def _text(component: Component) -> str:
    """A block's text including its descendants' (list items, image alt text...)."""
    parts = [c.text.strip() for c in component.walk() if c.text.strip()]
    return "\n".join(parts)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _windows(text: str, max_chars: int) -> list[str]:
    """Overlapping character windows, so a value on a boundary is whole in one of them."""
    overlap = min(200, max_chars // 10)
    step = max(max_chars - overlap, 1)
    return [text[i : i + max_chars] for i in range(0, max(len(text) - overlap, 1), step)]


def _split_line(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    parts: list[str] = []
    for sentence in _SENTENCE_END.split(line):
        parts.extend(_windows(sentence, max_chars) if len(sentence) > max_chars else [sentence])
    return _pack(parts, max_chars, sep=" ")


def _pack(lines: list[str], max_chars: int, *, sep: str = "\n", head: str = "") -> list[str]:
    """Lines packed greedily into pieces of at most ``max_chars``, each starting ``head``."""
    pieces: list[str] = []
    current = head
    for line in lines:
        candidate = f"{current}{sep}{line}" if current else line
        if len(candidate) > max_chars and current and current != head:
            pieces.append(current)
            candidate = f"{head}{sep}{line}" if head else line
        current = candidate
    if current and current != head:
        pieces.append(current)
    return pieces


def _table_rows(table: Component) -> tuple[list[str], list[str]]:
    """(header rows, other rows) as text lines. Captions count as header lines."""
    captions = [c.text.strip() for c in table.children if c.text.strip()]
    if not table.cells:
        return captions, [line for line in table.text.split("\n") if line.strip()]
    rows: dict[int, list[TableCell]] = {}
    for cell in sorted(table.cells, key=lambda c: (c.row, c.col)):
        rows.setdefault(cell.row, []).append(cell)
    header: list[str] = list(captions)
    body: list[str] = []
    for cells in rows.values():
        line = " | ".join(c.text for c in cells)
        (header if all(c.header for c in cells) else body).append(line)
    return header, body


def _pieces(block: Component, max_chars: int) -> list[str]:
    """``block``'s text in pieces of at most ``max_chars``.

    Tables split by row groups with their header rows (and captions) repeated; other
    blocks by line (list items), then sentence, then overlapping character windows.
    """
    text = _text(block)
    if len(text) <= max_chars:
        return [text] if text else []
    if block.type == "table":
        header, body = _table_rows(block)
        head = "\n".join(header)
        if len(head) <= max_chars // 2:
            return _pack(
                [p for row in body for p in _split_line(row, max_chars - len(head) - 1)],
                max_chars,
                head=head,
            )
        text = "\n".join([*header, *body])
    lines = [p for line in text.split("\n") for p in _split_line(line, max_chars)]
    return _pack(lines, max_chars)


def gate_units(root: Component, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[GateUnit]:
    """The units the gate asks about, in reading order.

    A run of headings with nothing after it in its container (a heading followed by a
    table or a nested section) joins the next unit, so it's asked about with the content
    it introduces. No unit's text is longer than ``max_chars``: oversized blocks are split.
    """
    units: list[GateUnit] = []

    def emit(unit_id: str, blocks: list[Component], pieces: list[str] | None = None) -> None:
        """One unit for ``blocks``, or one per piece when ``pieces`` is given (the last
        block split; any blocks before it lead the first piece)."""
        ids = tuple(c.id for b in blocks for c in b.walk())
        trail = tuple(blocks[0].heading_trail)
        if pieces is None:
            text = "\n".join(t for b in blocks if (t := _text(b)))
            if text:
                units.append(GateUnit(unit_id, ids, text, trail))
            return
        lead = "\n".join(t for b in blocks[:-1] if (t := _text(b)))
        for i, piece in enumerate(pieces):
            text = f"{lead}\n{piece}" if lead and i == 0 else piece
            piece_id = unit_id if len(pieces) == 1 else f"{unit_id}:{i}"
            units.append(GateUnit(piece_id, ids, text, trail))

    def only_headings(blocks: list[Component]) -> bool:
        return bool(blocks) and all(b.type == "heading" for b in blocks)

    def visit(container: Component, carry: list[Component]) -> list[Component]:
        """Emit ``container``'s units; returns trailing headings for what follows."""
        chunk: list[Component] = list(carry)
        size = sum(len(_text(b)) + 1 for b in chunk)
        part = 0

        def close() -> None:
            nonlocal chunk, size, part
            if chunk:
                emit(f"{container.id}#{part}", chunk)
                part += 1
            chunk, size = [], 0

        for child in container.children:
            is_table = child.type == "table"
            length = len(_text(child)) + 1
            if child.type in _CONTAINERS:
                lead = chunk if only_headings(chunk) else []
                if not lead:
                    close()
                chunk, size = [], 0
                chunk = visit(child, lead)
                size = sum(len(_text(b)) + 1 for b in chunk)
                continue
            headings_first = only_headings(chunk)
            if is_table or length > max_chars or (headings_first and size + length > max_chars):
                # The block is split into pieces; leading headings open the first one.
                lead = chunk if headings_first else []
                if not lead:
                    close()
                lead_size = sum(len(_text(b)) + 1 for b in lead)
                if lead_size > max_chars // 2:
                    close()  # too many headings to share a piece: they get their own unit
                    lead, lead_size = [], 0
                pieces = _pieces(child, max_chars - lead_size)
                if pieces:
                    emit(child.id if is_table else f"{container.id}#{part}", [*lead, child], pieces)
                    part += 0 if is_table else 1
                elif lead:
                    continue  # an empty table: the headings wait for what follows
                chunk, size = [], 0
                continue
            if chunk and size + length > max_chars:
                close()
            chunk.append(child)
            size += length
        if only_headings(chunk):
            return chunk
        close()
        return []

    if root.type in _CONTAINERS:
        leftover = visit(root, [])
        if leftover:
            emit(f"{root.id}#end", leftover)
    else:
        pieces = _pieces(root, max_chars)
        if pieces:
            emit(root.id, [root], pieces)
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

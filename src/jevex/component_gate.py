"""The component gate: which parts of a document are worth reading for each field group
(spec: *Pipeline architecture*, stage 7).

The layout tree is cut into **gate units**: each container's run of direct content
blocks (headings, paragraphs, lists, captions, images), chunked to at most ``max_chars``,
plus each table on its own. An image with text found in it (OCR from the image stage, or
text Docling found in a PDF picture) is cut like a section: its alt text, then the
paragraphs and headed sections found in it. One inside a paragraph or list item (an HTML
inline image) is lifted out after that block's own text. A block longer than ``max_chars`` is split
across several units (tables by row groups with their headers repeated) rather than cut,
and headings with nothing after them join the table or section they introduce. Jev is
asked one Noul per unit × field group ("Does this section contain the price?"), with every
schema's questions about a unit in one request.

A unit that passes for a group passes all its components and their descendants (and
their ancestors, so the tree stays connected). The result lands on
``SchemaRun.component_ids`` (group → component ids). Nested models are gated per field
too, for the child runs ``ParentChild`` makes. The entity stage then keeps only
passing components in each scope, so statements from irrelevant parts of the page never
reach categorisation, and :meth:`~jevex.pipeline.SchemaRun.relevant_fields` tells the
classifier which fields a component can state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jevex._tasks import gather
from jevex.jev import NoulAnswer, UnexpectedAnswerError
from jevex.layout import section_text
from jevex.schema import ReservedFieldNameError, UnsupportedFieldError
from jevex.tables import infer_headers, row_roles

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


def _opens(component: Component) -> bool:
    """Whether the gate cuts ``component``'s children into units rather than reading it as
    one block: containers, and images with text found in them (a scanned page's OCR has its
    own paragraphs and headings; a PDF picture can hold text and footnotes). An image with
    only its caption is one block, so a figure still shares a unit with the text around
    it."""
    if component.type == "image":
        return any(c.type != "caption" for c in component.children)
    return component.type in _CONTAINERS


def _blocks(container: Component) -> list[Component]:
    """What ``container`` is cut into: its children, after an image's own alt text (as a
    block without children, so its unit carries only the image's id), each with the read
    images inside it lifted out (:func:`_lifted`). A root that's a block is cut into its
    own text and the read images inside it."""
    if not _opens(container):
        return _lifted(container)
    children = container.children
    if container.type == "image" and container.text.strip():
        children = [container.model_copy(update={"children": []}), *children]
    return [b for child in children for b in _lifted(child)]


def _read_images(block: Component) -> list[Component]:
    """The images with text found in them below ``block``, in reading order: an HTML
    inline image under its paragraph or list item. A container inside a block (a product
    card in a list) is still read as part of the block."""
    out: list[Component] = []
    for child in block.children:
        read = child.type == "image" and _opens(child)
        out.extend([child] if read else _read_images(child))
    return out


def _lifted(block: Component) -> list[Component]:
    """``block`` with the read images inside it lifted out: the block without them (when it
    has text of its own), then each image, so an image is cut like a section and the
    block's own text is still gated with its neighbours."""
    if _opens(block):
        return [block]
    images = _read_images(block)
    if not images:
        return [block]
    ids = {image.id for image in images}

    def without(component: Component) -> Component:
        children = [without(c) for c in component.children if c.id not in ids]
        return component.model_copy(update={"children": children})

    own = without(block)
    return [own, *images] if _text(own) else images


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
        is its heading trail (:func:`~jevex.layout.section_text`)."""
        state = {"content": self.text}
        if section := section_text(self.heading_trail):
            state["section"] = section
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


def _pack(lines: list[str], max_chars: int, *, sep: str = "\n") -> list[str]:
    """Lines packed greedily into pieces of at most ``max_chars``."""
    pieces: list[str] = []
    current = ""
    for line in lines:
        candidate = f"{current}{sep}{line}" if current else line
        if len(candidate) > max_chars and current:
            pieces.append(current)
            candidate = line
        current = candidate
    if current:
        pieces.append(current)
    return pieces


def _table_lines(table: Component) -> list[tuple[str, str]]:
    """``table``'s text lines in reading order, each with the context a piece starting at
    it repeats first: the captions, then the header rows and the band in force.

    Rows are read as :func:`~jevex.tables.table_statements` reads them
    (:func:`~jevex.tables.row_roles`): the leading header rows stack, a header row repeated
    after the first body row replaces them, and a band ("Economy") holds until the next
    one. Bands and repeated header rows stay in place, so a band is only repeated in
    pieces that hold rows it groups.
    """
    captions = [c.text.strip() for c in table.children if c.text.strip()]
    lines = [(caption, "\n".join(captions[:i])) for i, caption in enumerate(captions)]
    if not table.cells:
        head = "\n".join(captions)
        return lines + [(line, head) for line in table.text.split("\n") if line.strip()]
    table = infer_headers(table)  # a comparison table's first row, as its statements have
    roles = row_roles(table)
    rows: dict[int, list[TableCell]] = {}
    for cell in sorted(table.cells, key=lambda c: (c.row, c.col)):
        if cell.text.strip():
            rows.setdefault(cell.row, []).append(cell)
    header: list[str] = []
    band: str | None = None
    in_body = False
    for r, cells in rows.items():
        line = " | ".join(c.text for c in cells)
        role = roles[r]
        if role == "band":
            band = None
        elif role == "header" and in_body:
            header = []
        lines.append((line, "\n".join([*captions, *header, *([band] if band else [])])))
        if role == "band":
            band = line
        elif role == "header":
            header.append(line)
        else:
            in_body = True
    return lines


def _pack_table(lines: list[tuple[str, str]], max_chars: int) -> list[str]:
    """Lines packed greedily into pieces of at most ``max_chars``, each piece after the
    first starting with its first line's context (dropped when over half a piece)."""
    pieces: list[str] = []
    current = head = ""
    for line, context in lines:
        if len(context) > max_chars // 2:
            context = ""
        width = max_chars - len(context) - 1 if context else max_chars
        for part in _split_line(line, width):
            candidate = f"{current}\n{part}" if current else part
            if len(candidate) > max_chars and current != head:
                pieces.append(current)
                head = context
                candidate = f"{context}\n{part}" if context else part
            current = candidate
    if current != head:
        pieces.append(current)
    return pieces


def _pieces(block: Component, max_chars: int) -> list[str]:
    """``block``'s text in pieces of at most ``max_chars``.

    Tables split by row groups, each piece repeating the captions, header rows and band
    in force (:func:`_table_lines`); other blocks by line (list items), then sentence,
    then overlapping character windows.
    """
    text = _text(block)
    if len(text) <= max_chars:
        return [text] if text else []
    if block.type == "table":
        return _pack_table(_table_lines(block), max_chars)
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

        for child in _blocks(container):
            is_table = child.type == "table"
            length = len(_text(child)) + 1
            if _opens(child):
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

    if _opens(root) or _read_images(root):
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

        await gather(ask(u) for u in units)
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

    Each nested-model field's model (:meth:`~jevex.schema.SchemaSpec.child`, named
    ``"<Parent>.<field>"``) is gated too, in the same requests, and its results land on
    ``SchemaRun.child_component_ids`` for the child run ``ParentChild`` makes, so a
    component is categorised only for the nested fields it passed for. A component that
    passes any of them also passes the nested-model field's group, so the entity resolver
    sees it.
    """

    gate: ComponentGate = field(default_factory=NoulComponentGate)
    name: str = "component_gate"

    async def run(self, ctx: Context) -> None:
        parsed = ctx.parsed
        runs = ctx.active
        if parsed is None or not runs:
            return
        children = {run.name: _child_specs(run.spec) for run in runs}
        specs = [r.spec for r in runs] + [s for kids in children.values() for s in kids.values()]
        decisions = await self.gate.gate(parsed, specs, ctx.jev)
        order = {c.id: i for i, c in enumerate(parsed.root.walk())}
        for run in runs:
            groups = decisions.get(run.name, {})
            for name, spec in children[run.name].items():
                child_groups = decisions.get(spec.name)
                if child_groups is None:
                    continue  # a gate that doesn't gate nested models: every field may be anywhere
                run.child_component_ids[name] = child_groups
                group = run.spec.field(name).group or name
                passed = {
                    *groups.get(group, ()),
                    *(c for ids in child_groups.values() for c in ids),
                }
                groups[group] = sorted(passed, key=lambda c: order.get(c, len(order)))
            run.component_ids = groups
            if not any(groups.values()):
                ctx.event(
                    self.name,
                    "no_relevant_components",
                    f"{run.name}: no component passed the component gate",
                    schema=run.name,
                )


def _child_specs(spec: SchemaSpec) -> dict[str, SchemaSpec]:
    """The nested models' specs by field. One jevex can't extract is left out: that's only
    an error if ``ParentChild`` is asked to fill it."""
    out: dict[str, SchemaSpec] = {}
    for f in spec.child_fields:
        try:
            out[f.name] = spec.child(f.name)
        except (UnsupportedFieldError, ReservedFieldNameError):
            continue
    return out

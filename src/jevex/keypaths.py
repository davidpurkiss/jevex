"""Key-path mapping: embedded data → field values (spec: *Structured-data stage*, steps 1–5).

1. **Flatten** each blob (:class:`~jevex.structured.StructuredBlob`) into key-path
   statements, ``vehicle.engine.displacement: 1498``. Array indices appear in the path
   (``offers[1].price``); arrays of objects are the blob's **entity candidates**
   (:attr:`FlatBlob.entities`).
2. **Fingerprint** the blob's shape: a hash of its sorted key paths with indices collapsed
   (``offers[].price``), so two pages from the same template share a fingerprint.
3. **Look up** learned mappings for the fingerprint in the store. A hit is a pure lookup:
   no Jev call.
4. **On a miss**, ask Jev one batched request per blob and schema: one Choice per unmapped
   key path over the schema's fields plus "none", with the flattened blob as the state.
5. **Store** confident answers (including "none", so a later hit needs no call) as
   :class:`~jevex.store.KeyMapping` records.

Mapped values are normalised like any candidate (the chain comes from the built-in
generators, which read "1,498 cc" or "£24,995" as they would in text) and recorded on the
default entity with ``method="structured"``. Later stages don't overwrite them; whether
the layout route runs at all is the structured mode's call (#31).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jevex.generators import GeneratorRegistry, default_registry
from jevex.jev import Choice, ChoiceAnswer, UnexpectedAnswerError
from jevex.layout import DomLocation
from jevex.normalise import NormaliseError, normalise
from jevex.resolve import SINGLE_ENTITY_LABEL
from jevex.results import FieldMeta, Source
from jevex.schema import NONE_OPTION
from jevex.statements import Statement
from jevex.store import KeyMapping
from jevex.structured import EmbeddedDataReader, StructuredBlob

if TYPE_CHECKING:
    from jevex.document import Document
    from jevex.interfaces import StructuredExtractor
    from jevex.jev import JevClient
    from jevex.pipeline import Context
    from jevex.schema import FieldSpec, SchemaSpec
    from jevex.store import Store

ACCEPT_AT = 0.5
"""A Jev answer at or above this confidence becomes a stored mapping (field or "none")."""

MAX_PATHS = 300
"""Key paths asked about per blob; the rest are skipped (with an event)."""

MAX_VALUE_CHARS = 200
"""Values longer than this are cut in the state Jev sees (never in what is extracted)."""

_SKIP_KEYS = frozenset({"@context", "@id"})


@dataclass(frozen=True)
class Leaf:
    path: str
    """With indices: ``offers[1].price``."""
    shape: str
    """With indices collapsed: ``offers[].price``."""
    value: str | int | float | bool


@dataclass(frozen=True)
class FlatBlob:
    blob: StructuredBlob
    index: int
    leaves: tuple[Leaf, ...]
    entities: tuple[str, ...]
    """Collapsed paths of arrays of objects, e.g. ``offers[]`` (entity candidates)."""

    @property
    def fingerprint(self) -> str:
        shapes = sorted({leaf.shape for leaf in self.leaves})
        return hashlib.sha256("\n".join(shapes).encode()).hexdigest()[:16]

    def shapes(self) -> dict[str, list[Leaf]]:
        """Leaves grouped by collapsed path, in document order."""
        out: dict[str, list[Leaf]] = {}
        for leaf in self.leaves:
            out.setdefault(leaf.shape, []).append(leaf)
        return out


def flatten(blob: StructuredBlob, index: int = 0) -> FlatBlob:
    """Leaves of ``blob.data`` with their key paths. ``@context``/``@id`` are skipped."""
    leaves: list[Leaf] = []
    entities: list[str] = []

    def walk(value: Any, path: str, shape: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():  # pyright: ignore[reportUnknownVariableType]
                if key in _SKIP_KEYS:
                    continue
                name = str(key)  # pyright: ignore[reportUnknownArgumentType]
                walk(
                    child, f"{path}.{name}" if path else name, f"{shape}.{name}" if shape else name
                )
        elif isinstance(value, list):
            items: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
            if any(isinstance(item, dict) for item in items) and f"{shape}[]" not in entities:
                entities.append(f"{shape}[]")
            for i, item in enumerate(items):
                walk(item, f"{path}[{i}]", f"{shape}[]")
        elif isinstance(value, str | int | float | bool) and not (
            isinstance(value, str) and not value.strip()
        ):
            leaves.append(Leaf(path or "$", shape or "$", value))

    walk(blob.data, "", "")
    return FlatBlob(blob, index, tuple(leaves), tuple(entities))


def _text(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass(frozen=True)
class StructuredResult:
    """What the structured route found: statements for every leaf, values per schema."""

    statements: list[Statement]
    fields: dict[str, dict[str, FieldMeta]]
    """Schema name → field name → meta (on the default entity)."""
    events: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """``(kind, message)`` pairs for document meta."""


class KeyPathMapper:
    """The default :class:`~jevex.interfaces.StructuredExtractor`.

    ``store`` persists mappings across documents and processes; without one, mappings
    are remembered by this mapper for its lifetime only. ``registry`` supplies the
    normaliser chains for values.
    """

    def __init__(
        self,
        *,
        store: Store | None = None,
        reader: EmbeddedDataReader | None = None,
        registry: GeneratorRegistry | None = None,
        accept_at: float = ACCEPT_AT,
        max_paths: int = MAX_PATHS,
    ) -> None:
        self.store = store
        self.reader = reader or EmbeddedDataReader()
        self.registry = registry or default_registry()
        self.accept_at = accept_at
        self.max_paths = max_paths
        # (fingerprint, schema) → path → field (None: no field). Used without a store.
        self._memory: dict[tuple[str, str], dict[str, str | None]] = {}

    async def extract(
        self, document: Document, schemas: list[SchemaSpec], jev: JevClient
    ) -> StructuredResult:
        data = self.reader.read(document)
        statements: list[Statement] = []
        fields: dict[str, dict[str, FieldMeta]] = {s.name: {} for s in schemas}
        events: list[tuple[str, str]] = []
        for i, blob in enumerate(data.blobs):
            flat = flatten(blob, i)
            if not flat.leaves:
                continue
            ids = {leaf.path: f"structured.{i}.{n}" for n, leaf in enumerate(flat.leaves)}
            statements.extend(
                Statement(
                    id=ids[leaf.path],
                    text=f"{leaf.path}: {_text(leaf.value)}",
                    kind="structured",
                    component_id=f"structured.{i}",
                    location=blob.location,
                )
                for leaf in flat.leaves
            )
            for schema in schemas:
                mapping = await self._mapping(flat, schema, jev, events)
                for shape, leaves in flat.shapes().items():
                    name = mapping.get(shape)
                    if name is None or name in fields[schema.name]:
                        continue
                    meta = self._meta(schema.field(name), leaves, ids, document)
                    if meta is not None:
                        fields[schema.name][name] = meta
        return StructuredResult(statements, fields, events)

    async def _mapping(
        self,
        flat: FlatBlob,
        schema: SchemaSpec,
        jev: JevClient,
        events: list[tuple[str, str]],
    ) -> dict[str, str | None]:
        """Collapsed path → field name (or None) for every path Jev or the store knows."""
        fingerprint = flat.fingerprint
        known = await self._known(fingerprint, schema.name)
        shapes = flat.shapes()
        unknown = [s for s in shapes if s not in known]
        if len(unknown) > self.max_paths:
            events.append(
                (
                    "structured_paths_skipped",
                    f"{schema.name}: asked about {self.max_paths} of {len(unknown)} key paths "
                    f"in {flat.blob.source} blob {flat.index}",
                )
            )
            unknown = unknown[: self.max_paths]
        if not unknown:
            return known
        fieldable = [f for f in schema.fields if f.kind != "model"]
        if not fieldable:
            return known
        answers = await self._ask(flat, schema, unknown, jev)
        learned: dict[str, str | None] = {}
        for shape, answer in answers.items():
            if answer.confidence < self.accept_at:
                continue
            learned[shape] = None if answer.choice == NONE_OPTION else answer.choice
        await self._remember(fingerprint, schema.name, learned)
        return known | learned

    async def _known(self, fingerprint: str, schema: str) -> dict[str, str | None]:
        if self.store is None:
            return dict(self._memory.get((fingerprint, schema), {}))
        return {m.path: m.field for m in await self.store.key_mappings(fingerprint, schema=schema)}

    async def _remember(
        self, fingerprint: str, schema: str, learned: dict[str, str | None]
    ) -> None:
        if not learned:
            return
        if self.store is None:
            self._memory.setdefault((fingerprint, schema), {}).update(learned)
            return
        for path, name in learned.items():
            await self.store.put_key_mapping(
                KeyMapping(fingerprint=fingerprint, schema=schema, path=path, field=name)
            )

    async def _ask(
        self, flat: FlatBlob, schema: SchemaSpec, shapes: list[str], jev: JevClient
    ) -> dict[str, ChoiceAnswer]:
        by_shape = flat.shapes()
        lines = [f"{leaf.path}: {_text(leaf.value)[:MAX_VALUE_CHARS]}" for leaf in flat.leaves]
        state: dict[str, Any] = {"data": "\n".join(lines)}
        if flat.blob.types:
            state["type"] = ", ".join(flat.blob.types)
        options = schema.categorise_question().options
        questions = {
            f"path{i}": Choice(
                instructions=(
                    f'Which detail does the key path "{shape}" hold '
                    f"(e.g. {_text(by_shape[shape][0].value)[:MAX_VALUE_CHARS]!r})?"
                ),
                options=options,
            )
            for i, shape in enumerate(shapes)
        }
        answers = await jev.ask(state, questions)
        out: dict[str, ChoiceAnswer] = {}
        for i, shape in enumerate(shapes):
            answer = answers[f"path{i}"]
            if not isinstance(answer, ChoiceAnswer):
                raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
            out[shape] = answer
        return out

    def _meta(
        self,
        spec: FieldSpec,
        leaves: list[Leaf],
        ids: dict[str, str],
        document: Document,
    ) -> FieldMeta | None:
        """The field's value from its leaves: the first that normalises (every one, for a
        list field). ``None`` when none do; the error is kept on the meta."""
        values: list[Any] = []
        errors: list[str] = []
        first: Leaf | None = None
        for leaf in leaves:
            try:
                value = self._value(spec, leaf.value)
            except NormaliseError as exc:
                errors.append(f"{leaf.path}: {exc}")
                continue
            if first is None:
                first = leaf
            for item in value if isinstance(value, list) and spec.many else [value]:  # pyright: ignore[reportUnknownVariableType]
                if item not in values:
                    values.append(item)
            if not spec.many:
                break
        leaf = first or leaves[0]
        source = Source(
            url=document.url,
            component_id=ids[leaf.path].rsplit(".", 1)[0],
            statement_id=ids[leaf.path],
            statement=f"{leaf.path}: {_text(leaf.value)}",
        )
        if first is None:
            return FieldMeta(method="structured", source=source, error="; ".join(errors))
        return FieldMeta(
            value=values if spec.many else values[0], method="structured", source=source
        )

    def _value(self, spec: FieldSpec, raw: str | int | float | bool) -> Any:
        if not isinstance(raw, str) or spec.kind in ("enum", "bool"):
            return normalise(raw, [], spec)
        statement = Statement(
            id="value",
            text=raw,
            kind="key_value",
            component_id="value",
            location=_NOWHERE,
        )
        # The candidate covering most of the value decides the chain ("1,498 cc" → number
        # with unit cc); without one, the value is taken as it is.
        candidates = sorted(
            self.registry.generate(statement, spec, schema=""),
            key=lambda c: c.span.start - c.span.end,
        )
        errors: list[str] = []
        for candidate in candidates[:3]:
            try:
                return normalise(candidate.raw, candidate.normalise, spec)
            except NormaliseError as exc:
                errors.append(str(exc))
        try:
            return normalise(raw.strip(), [], spec)
        except NormaliseError:
            if errors:
                raise NormaliseError("; ".join(errors)) from None
            raise


_NOWHERE = DomLocation(dom_path="/")


@dataclass
class StructuredStage:
    """Runs a :class:`~jevex.interfaces.StructuredExtractor` (stage 4).

    Values are recorded on the default entity of each active schema, and statements on
    ``ctx.structured``. Fields later routes fill are left alone where this one found them.
    """

    extractor: StructuredExtractor = field(default_factory=KeyPathMapper)
    name: str = "structured"

    async def run(self, ctx: Context) -> None:
        runs = ctx.active
        if not runs:
            return
        result = await self.extractor.extract(ctx.document, [r.spec for r in runs], ctx.jev)
        ctx.structured.extend(result.statements)
        for kind, message in result.events:
            ctx.event(self.name, kind, message)
        for run in runs:
            for name, meta in result.fields.get(run.name, {}).items():
                existing = run.fields.get(SINGLE_ENTITY_LABEL, {}).get(name)
                if existing is None or not existing.found:
                    run.set_field(SINGLE_ENTITY_LABEL, name, meta)
            found = [n for n, m in result.fields.get(run.name, {}).items() if m.found]
            if found:
                ctx.event(
                    self.name,
                    "structured_fields",
                    f"{run.name}: {', '.join(found)} from embedded data",
                    schema=run.name,
                )

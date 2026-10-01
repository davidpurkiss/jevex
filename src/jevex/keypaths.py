"""Key-path mapping: embedded data → field values (spec: *Structured-data stage*, steps 1–5).

1. **Flatten** each blob (:class:`~jevex.structured.StructuredBlob`) into key-path
   statements, ``vehicle.engine.displacement: 1498``. Array indices appear in the path
   (``offers[1].price``), and keys holding ``.``, ``[`` or ``]`` are quoted
   (``messages["cta.buy"]``). Arrays of objects are the blob's **entity candidates**
   (:attr:`FlatBlob.entities`).
2. **Fingerprint** the blob's shape: a hash of its source, its schema.org types and its
   sorted key paths with indices collapsed (``offers[].price``), so two pages from the
   same template share a fingerprint but a ``Brand`` and a ``Person`` node don't.
3. **Look up** learned mappings for the fingerprint in the store. A hit is a pure lookup:
   no Jev call.
4. **On a miss**, ask Jev about the unmapped key paths: one Choice per path and schema,
   over the schema's fields plus "none", every schema's questions in one request. The
   state shows a few example values per path (never the whole blob), chunked so each
   state fits Jev's budget.
5. **Store** confident answers (including "none", so a later hit needs no call) as
   :class:`~jevex.store.KeyMapping` records. A path Jev stays unsure about is asked again
   on the template's next page, but after :data:`UNSURE_LIMIT` unsure answers it is stored
   as an ``unsure`` "none", so every template reaches the zero-call steady state.

Mapped values are normalised like any candidate: numbers, money and dates take the chain
the built-in generators find ("1,498 cc" → 1498), strings are taken whole, and enum or
bool values that don't read directly are asked of Jev as the field's own question. They
are recorded on the default entity with ``method="structured"``.

The stage's :data:`StructuredMode` decides what the layout route (stages 5–13) does next:

- ``structured_only`` (default): a schema the embedded data gave any value is finished;
  the layout route runs only for schemas it gave nothing.
- ``fill_gaps``: the layout route runs while a field is left empty, and selects values
  only for those fields (a schema with none left is finished). The component gate and
  the categoriser still consider every field: they run before entities are resolved, and
  a found field stays a categorise option so statements about it aren't misrouted.
- ``merge``: the layout route looks for every field, and
  :meth:`~jevex.pipeline.SchemaRun.offer_field` settles each disagreement by confidence,
  recording the losing value in ``meta.conflicts``.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from jevex._tasks import gather
from jevex.generators import GeneratorRegistry, default_registry
from jevex.jev import (
    MAX_STATE_TOKENS,
    ChoiceAnswer,
    NoulAnswer,
    UnexpectedAnswerError,
    estimate_tokens,
)
from jevex.layout import DomLocation
from jevex.normalise import NormaliseError, normalise
from jevex.resolve import SINGLE_ENTITY_LABEL
from jevex.results import FieldMeta, Source
from jevex.schema import NONE_OPTION, NOT_STATED_OPTION
from jevex.statements import Statement
from jevex.store import KeyMapping
from jevex.structured import EmbeddedDataReader, StructuredBlob, schema_type

if TYPE_CHECKING:
    from jevex.document import Document
    from jevex.interfaces import StructuredExtractor
    from jevex.jev import JevClient, Question, ScoreAnswer
    from jevex.pipeline import Context, SchemaRun
    from jevex.schema import FieldSpec, SchemaSpec
    from jevex.store import Store

StructuredMode = Literal["structured_only", "fill_gaps", "merge"]
"""How structured data combines with the layout route (spec: *Structured-data stage*)."""

ACCEPT_AT = 0.5
"""A Jev answer at or above this confidence becomes a stored mapping (field or "none")."""

UNSURE_LIMIT = 3
"""Unsure answers (below :data:`ACCEPT_AT`) about one (fingerprint, schema, path) before
it is stored as "none". A review or re-learn can replace that with a confident mapping."""

MAX_PATHS = 300
"""Key paths asked about per blob; the rest are skipped (with an event)."""

MAX_FALLBACK_VALUES = 3
"""Distinct values of a single-value enum/bool field asked of Jev per document when none
reads directly; the first Jev can read wins."""

MAX_LIST_FALLBACK_VALUES = 20
"""Distinct values of a list enum field asked of Jev per document (one request each)."""

MAX_BLOBS = 20
"""Blobs mapped per document, in reading order (JSON-LD first); a miss costs Jev requests
per blob, and pages can carry hundreds of ``data-*`` blobs."""

MAX_VALUE_CHARS = 200
"""Values longer than this are cut in the state Jev sees (never in what is extracted)."""

EXAMPLES_PER_PATH = 3
"""Example values shown per key path in the state."""

STATE_TOKEN_BUDGET = MAX_STATE_TOKENS // 4
"""Estimated tokens per state; unknown paths beyond it go in further states. Kept well
under Jev's limit because the longest question counts against it too."""

MEMORY_SIZE = 1024
"""Fingerprint × schema entries a store-less mapper remembers (least recently used go)."""

_SKIP_KEYS = frozenset({"@context", "@id"})
_QUOTE = frozenset(".[]")


@dataclass(frozen=True)
class Leaf:
    """One scalar in a blob: where it sits and its value."""

    index: int
    """Position among the blob's leaves; statement ids are built from it."""
    path: str
    """With indices: ``offers[1].price``."""
    shape: str
    """With indices collapsed: ``offers[].price``."""
    value: str | int | float | bool


@dataclass(frozen=True)
class FlatBlob:
    """A blob flattened into leaves, with its shape fingerprint and entity candidates."""

    blob: StructuredBlob
    index: int
    leaves: tuple[Leaf, ...]
    entities: tuple[str, ...]
    """Collapsed paths of arrays of objects, e.g. ``offers[]`` (entity candidates)."""

    @property
    def fingerprint(self) -> str:
        shapes = sorted({leaf.shape for leaf in self.leaves})
        key = "\n".join([self.blob.source, ",".join(sorted(self.blob.types)), *shapes])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def shapes(self) -> dict[str, list[Leaf]]:
        """Leaves grouped by collapsed path, in document order."""
        out: dict[str, list[Leaf]] = {}
        for leaf in self.leaves:
            out.setdefault(leaf.shape, []).append(leaf)
        return out

    def statement_id(self, leaf: Leaf) -> str:
        return f"structured.{self.index}.{leaf.index}"


def _key(parent: str, key: str) -> str:
    if _QUOTE.intersection(key):
        return f"{parent}[{json.dumps(key, ensure_ascii=False)}]"
    return f"{parent}.{key}" if parent else key


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
                walk(child, _key(path, name), _key(shape, name))
        elif isinstance(value, list):
            items: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
            if any(isinstance(item, dict) for item in items) and f"{shape}[]" not in entities:
                entities.append(f"{shape}[]")
            for i, item in enumerate(items):
                walk(item, f"{path}[{i}]", f"{shape}[]")
        elif isinstance(value, str | int | float | bool) and not (
            isinstance(value, str) and not value.strip()
        ):
            leaves.append(Leaf(len(leaves), path or "$", shape or "$", value))

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


def _mappable(schema: SchemaSpec) -> set[str]:
    return {f.name for f in schema.fields if f.kind != "model"}


class KeyPathMapper:
    """The default :class:`~jevex.interfaces.StructuredExtractor`.

    ``store`` persists mappings across documents and processes; without one, mappings
    are remembered by this mapper (the :data:`MEMORY_SIZE` most recent fingerprints) for
    its lifetime only. ``registry`` supplies the normaliser chains for values. A path
    with ``unsure_limit`` unsure answers is stored as "none" (see :data:`UNSURE_LIMIT`).
    """

    def __init__(
        self,
        *,
        store: Store | None = None,
        reader: EmbeddedDataReader | None = None,
        registry: GeneratorRegistry | None = None,
        accept_at: float = ACCEPT_AT,
        unsure_limit: int = UNSURE_LIMIT,
        max_paths: int = MAX_PATHS,
        max_blobs: int = MAX_BLOBS,
        state_tokens: int = STATE_TOKEN_BUDGET,
        memory_size: int = MEMORY_SIZE,
    ) -> None:
        self.store = store
        self.reader = reader or EmbeddedDataReader()
        self.registry = registry or default_registry()
        if unsure_limit < 1:
            raise ValueError(f"unsure_limit must be at least 1, not {unsure_limit}")
        self.accept_at = accept_at
        self.unsure_limit = unsure_limit
        self.max_paths = max_paths
        self.max_blobs = max_blobs
        self.state_tokens = state_tokens
        self.memory_size = memory_size
        # (fingerprint, schema) → path → field (None: no field). Used without a store.
        self._memory: OrderedDict[tuple[str, str], dict[str, str | None]] = OrderedDict()
        # (fingerprint, schema) → path → unsure answers so far. Used without a store.
        self._unsure: OrderedDict[tuple[str, str], dict[str, int]] = OrderedDict()
        # The exact state and questions sent → Jev's answers, for enum/bool value fallbacks.
        self._values: OrderedDict[str, dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]] = (
            OrderedDict()
        )

    async def extract(
        self,
        document: Document,
        schemas: list[SchemaSpec],
        jev: JevClient,
        *,
        store: Store | None = None,
    ) -> StructuredResult:
        """``store`` (the extractor's) is used when the mapper wasn't given one."""
        store = self.store if self.store is not None else store
        data = self.reader.read(document)
        events: list[tuple[str, str]] = []
        blobs = data.blobs
        if len(blobs) > self.max_blobs:
            events.append(
                (
                    "structured_blobs_skipped",
                    f"mapped {self.max_blobs} of {len(blobs)} embedded data blobs",
                )
            )
            blobs = blobs[: self.max_blobs]
        flats = [f for i, b in enumerate(blobs) if (f := flatten(b, i)).leaves]
        # Blobs from one template share a fingerprint: ask about it once per document.
        first: dict[str, FlatBlob] = {}
        for flat in flats:
            first.setdefault(flat.fingerprint, flat)
        resolved = await gather(
            self._mappings(flat, schemas, jev, events, store) for flat in first.values()
        )
        mappings = dict(zip(first, resolved, strict=True))
        per_blob = await gather(
            self._blob(flat, schemas, mappings[flat.fingerprint], document, jev) for flat in flats
        )
        statements: list[Statement] = []
        fields: dict[str, dict[str, FieldMeta]] = {s.name: {} for s in schemas}
        for flat, found in zip(flats, per_blob, strict=True):
            statements.extend(
                Statement(
                    id=flat.statement_id(leaf),
                    text=f"{leaf.path}: {_text(leaf.value)}",
                    kind="structured",
                    component_id=f"structured.{flat.index}",
                    location=flat.blob.location,
                )
                for leaf in flat.leaves
            )
            for schema_name, metas in found.items():
                for name, meta in metas.items():
                    _keep_best(fields[schema_name], name, meta)
        return StructuredResult(statements, fields, events)

    async def _blob(
        self,
        flat: FlatBlob,
        schemas: list[SchemaSpec],
        mappings: dict[str, dict[str, str | None]],
        document: Document,
        jev: JevClient,
    ) -> dict[str, dict[str, FieldMeta]]:
        """One blob's values per schema, given its mappings."""
        shapes = flat.shapes()
        found: dict[str, dict[str, FieldMeta]] = {}
        for schema in schemas:
            mapping = mappings[schema.name]
            wanted = [(shape, name) for shape, name in mapping.items() if name and shape in shapes]
            metas = await gather(
                self._meta(schema.field(name), flat, shapes[shape], document, jev)
                for shape, name in wanted
            )
            mine: dict[str, FieldMeta] = {}
            for (_, name), meta in zip(wanted, metas, strict=True):
                _keep_best(mine, name, meta)
            found[schema.name] = mine
        return found

    async def _mappings(
        self,
        flat: FlatBlob,
        schemas: list[SchemaSpec],
        jev: JevClient,
        events: list[tuple[str, str]],
        store: Store | None,
    ) -> dict[str, dict[str, str | None]]:
        """Per schema: collapsed path → field name (or None) for every path known."""
        fingerprint = flat.fingerprint
        shapes = flat.shapes()
        known: dict[str, dict[str, str | None]] = {}
        unknown: dict[str, list[str]] = {}
        for schema in schemas:
            fields = _mappable(schema)
            # A mapping to a field the schema no longer has counts as unknown: re-ask it.
            mine = {
                path: name
                for path, name in (await self._known(fingerprint, schema.name, store)).items()
                if name is None or name in fields
            }
            known[schema.name] = mine
            if not fields:
                continue
            todo = [s for s in shapes if s not in mine]
            if len(todo) > self.max_paths:
                events.append(
                    (
                        "structured_paths_skipped",
                        f"{schema.name}: asked about {self.max_paths} of {len(todo)} key "
                        f"paths in {flat.blob.source} blob {flat.index}",
                    )
                )
                todo = todo[: self.max_paths]
            if todo:
                unknown[schema.name] = todo
        if not unknown:
            return known
        by_name = {s.name: s for s in schemas}
        answers = await self._ask(flat, by_name, unknown, jev, events)
        for schema_name, paths in answers.items():
            learned: dict[str, str | None] = {}
            unsure: list[str] = []
            for shape, answer in paths.items():
                if answer.confidence < self.accept_at:
                    unsure.append(shape)
                    continue
                learned[shape] = None if answer.choice == NONE_OPTION else answer.choice
            counts = await self._count_unsure(fingerprint, schema_name, unsure, store)
            given_up = [path for path in unsure if counts[path] >= self.unsure_limit]
            if given_up:
                events.append(
                    (
                        "structured_paths_unsure",
                        f"{schema_name}: stored {len(given_up)} key path(s) as none after "
                        f"{self.unsure_limit} unsure answers: {', '.join(given_up)}",
                    )
                )
            learned |= dict.fromkeys(given_up)
            await self._remember(fingerprint, schema_name, learned, set(given_up), store)
            known[schema_name] |= learned
        return known

    async def _known(
        self, fingerprint: str, schema: str, store: Store | None
    ) -> dict[str, str | None]:
        if store is None:
            key = (fingerprint, schema)
            if key in self._memory:
                self._memory.move_to_end(key)
            return dict(self._memory.get(key, {}))
        return {m.path: m.field for m in await store.key_mappings(fingerprint, schema=schema)}

    async def _count_unsure(
        self, fingerprint: str, schema: str, paths: list[str], store: Store | None
    ) -> dict[str, int]:
        """Each path's unsure answers so far, this one included."""
        if not paths:
            return {}
        if store is not None:
            return await store.count_unsure_key_paths(fingerprint, schema, paths)
        key = (fingerprint, schema)
        counts = self._unsure.setdefault(key, {})
        self._unsure.move_to_end(key)
        for path in paths:
            counts[path] = counts.get(path, 0) + 1
        found = {path: counts[path] for path in paths}
        while len(self._unsure) > self.memory_size:
            self._unsure.popitem(last=False)
        return found

    async def _remember(
        self,
        fingerprint: str,
        schema: str,
        learned: dict[str, str | None],
        unsure: set[str],
        store: Store | None,
    ) -> None:
        """Store ``learned``; paths in ``unsure`` are "none" because Jev stayed unsure."""
        if not learned:
            return
        if store is None:
            key = (fingerprint, schema)
            self._memory.setdefault(key, {}).update(learned)
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_size:
                self._memory.popitem(last=False)
            counts = self._unsure.get(key, {})
            for path in learned:
                counts.pop(path, None)
            return
        for path, name in learned.items():
            await store.put_key_mapping(
                KeyMapping(
                    fingerprint=fingerprint,
                    schema=schema,
                    path=path,
                    field=name,
                    unsure=path in unsure,
                )
            )

    async def _ask(
        self,
        flat: FlatBlob,
        schemas: dict[str, SchemaSpec],
        unknown: dict[str, list[str]],
        jev: JevClient,
        events: list[tuple[str, str]],
    ) -> dict[str, dict[str, ChoiceAnswer]]:
        """Every schema's questions about the blob's unknown paths, as few states as fit.

        Each state lists example values for its paths only (not the whole blob), so a
        large ``__NEXT_DATA__`` with thousands of leaves but few shapes stays small.
        """
        by_shape = flat.shapes()
        paths = list(dict.fromkeys(p for todo in unknown.values() for p in todo))
        chunks: list[list[tuple[str, list[str]]]] = [[]]
        used = 0
        for path in paths:
            lines = [
                f"{leaf.path}: {_text(leaf.value)[:MAX_VALUE_CHARS]}"
                for leaf in by_shape[path][:EXAMPLES_PER_PATH]
            ]
            cost = estimate_tokens("\n".join(lines))
            if cost > self.state_tokens:
                events.append(("structured_path_too_large", f"skipped key path {path!r}"))
                continue
            if chunks[-1] and used + cost > self.state_tokens:
                chunks.append([])
                used = 0
            chunks[-1].append((path, lines))
            used += cost

        async def ask(chunk: list[tuple[str, list[str]]]) -> dict[str, dict[str, ChoiceAnswer]]:
            state: dict[str, Any] = {
                "data": "\n".join(line for _, lines in chunk for line in lines)
            }
            if flat.blob.types:
                state["type"] = ", ".join(flat.blob.types)
            questions: dict[str, Question] = {}
            keys: dict[str, tuple[str, str]] = {}
            in_chunk = {path for path, _ in chunk}
            for schema_name, todo in unknown.items():
                for path in todo:
                    if path not in in_chunk:
                        continue
                    key = f"{schema_name}:{len(keys)}"
                    example = _text(by_shape[path][0].value)[:MAX_VALUE_CHARS]
                    questions[key] = schemas[schema_name].key_path_question(path, example)
                    keys[key] = (schema_name, path)
            answers = await jev.ask(state, questions)
            out: dict[str, dict[str, ChoiceAnswer]] = {}
            for key, (schema_name, path) in keys.items():
                answer = answers[key]
                if not isinstance(answer, ChoiceAnswer):
                    raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
                out.setdefault(schema_name, {})[path] = answer
            return out

        results = await gather(ask(c) for c in chunks if c)
        merged: dict[str, dict[str, ChoiceAnswer]] = {}
        for result in results:
            for schema_name, answers in result.items():
                merged.setdefault(schema_name, {}).update(answers)
        return merged

    async def _meta(
        self,
        spec: FieldSpec,
        flat: FlatBlob,
        leaves: list[Leaf],
        document: Document,
        jev: JevClient,
    ) -> FieldMeta:
        """The field's value from its leaves: the first that reads directly (every one, for
        a list field). An enum or bool value that doesn't read directly ("Plug-in hybrid",
        "Automatic") is asked of Jev as the field's own question about the key-path
        statement: once per distinct value (remembered by the mapper), concurrently, and
        for a single-value field only when no leaf reads directly, for at most
        :data:`MAX_FALLBACK_VALUES` values (a list enum for at most
        :data:`MAX_LIST_FALLBACK_VALUES`). When nothing works, the error is kept."""
        direct: dict[int, Any] = {}
        errors: dict[int, str] = {}
        for i, leaf in enumerate(leaves):
            try:
                direct[i] = self._value(spec, leaf.value)
            except NormaliseError as exc:
                errors[i] = f"{leaf.path}: {exc}"
        asked: dict[str, tuple[Any, float] | None] = {}
        if errors and spec.kind in ("enum", "bool") and (spec.many or not direct):
            limit = MAX_FALLBACK_VALUES if not spec.many else MAX_LIST_FALLBACK_VALUES
            raws = list(dict.fromkeys(_text(leaves[i].value) for i in errors))[:limit]
            by_raw = {_text(leaves[i].value): leaves[i] for i in reversed(list(errors))}
            results = await gather(self._ask_value_cached(spec, by_raw[r], jev) for r in raws)
            asked = dict(zip(raws, results, strict=True))

        values: list[Any] = []
        first: Leaf | None = None
        confidence: float | None = None
        for i, leaf in enumerate(leaves):
            if i in direct:
                value = direct[i]
            else:
                answer = asked.get(_text(leaf.value))
                if answer is None:
                    continue
                value, p = answer
                confidence = p if confidence is None else min(confidence, p)
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
            component_id=f"structured.{flat.index}",
            statement_id=flat.statement_id(leaf),
            statement=f"{leaf.path}: {_text(leaf.value)}",
        )
        if first is None or not values:
            error = "; ".join(errors.values()) or None
            return FieldMeta(method="structured", source=source, error=error)
        return FieldMeta(
            value=values if spec.many else values[0],
            confidence=confidence,
            method="structured",
            source=source,
        )

    async def _ask_value_cached(
        self, spec: FieldSpec, leaf: Leaf, jev: JevClient
    ) -> tuple[Any, float] | None:
        """Jev's reading of an enum or bool value; ``None`` for other kinds or "not stated".

        Jev's raw answers are remembered, keyed by the exact state and questions sent, so
        a value the template repeats on every page is asked once for the mapper's lifetime
        and fields that merely share a name can't reuse each other's answers. The reading
        is normalised to this field after the lookup.

        A list enum asks one Noul per option (as the selector does) and keeps those at
        p ≥ 0.5; its confidence is the lowest accepted p.
        """
        state = {"statement": f"{leaf.path}: {_text(leaf.value)}"}
        questions: dict[str, Question]
        options = [str(o) for o in spec.options]
        if spec.kind == "enum" and spec.many:
            questions = {f"member{i}": spec.member_question(o) for i, o in enumerate(options)}
        elif spec.kind == "enum":
            questions = {"enum": spec.enum_question()}
        elif spec.kind == "bool":
            questions = {"bool": spec.bool_question()}
        else:
            return None
        key = json.dumps(
            {
                "state": state,
                "questions": {k: q.model_dump(mode="json") for k, q in questions.items()},
            },
            sort_keys=True,
            default=str,
        )
        if key in self._values:
            self._values.move_to_end(key)
            answers = self._values[key]
        else:
            answers = await jev.ask(state, questions)
            self._values[key] = answers
            while len(self._values) > self.memory_size * 4:
                self._values.popitem(last=False)

        if spec.kind == "enum" and spec.many:
            accepted: list[tuple[str, float]] = []
            for i, option in enumerate(options):
                answer = answers[f"member{i}"]
                if not isinstance(answer, NoulAnswer):
                    raise UnexpectedAnswerError(f"expected a Noul answer, got {answer.type}")
                if answer.p >= 0.5:
                    accepted.append((option, answer.p))
            if not accepted:
                return None
            return [normalise(o, [], spec) for o, _ in accepted], min(p for _, p in accepted)
        if spec.kind == "enum":
            answer = answers["enum"]
            if not isinstance(answer, ChoiceAnswer):
                raise UnexpectedAnswerError(f"expected a Choice answer, got {answer.type}")
            if answer.choice == NOT_STATED_OPTION:
                return None
            return normalise(answer.choice, [], spec), answer.confidence
        answer = answers["bool"]
        if not isinstance(answer, NoulAnswer):
            raise UnexpectedAnswerError(f"expected a Noul answer, got {answer.type}")
        value = answer.p >= 0.5
        return value, answer.p if value else 1 - answer.p

    def _value(self, spec: FieldSpec, raw: str | int | float | bool) -> Any:
        if spec.kind == "str":
            # A key path holds one value: take it whole ("Red, Blue" is one colour string;
            # a model called 3008 is text, not a number).
            return normalise(_text(raw).strip(), [], spec)
        if spec.kind == "enum" and isinstance(raw, str):
            return normalise(_enum_option(spec, raw), [], spec)
        if not isinstance(raw, str) or spec.kind == "bool":
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


def _keep_best(metas: dict[str, FieldMeta], name: str, meta: FieldMeta) -> None:
    """Keep the first found value; an unfound meta (an error) is kept only until one is."""
    existing = metas.get(name)
    if existing is None or (not existing.found and meta.found):
        metas[name] = meta


def _enum_option(spec: FieldSpec, raw: str) -> str:
    """The option ``raw`` names, ignoring case and a schema.org prefix
    (``"https://schema.org/InStock"``, ``"schema:InStock"`` → ``InStock``) or any other URL
    path; ``raw`` itself if none matches."""
    text = schema_type(raw).rstrip("/").rsplit("/", 1)[-1].strip()
    by_lower = {str(o).lower(): str(o) for o in spec.options}
    return by_lower.get(text.lower(), raw)


@dataclass
class StructuredStage:
    """Runs a :class:`~jevex.interfaces.StructuredExtractor` (stage 4).

    Values are recorded on the default entity of each active schema, and statements on
    ``ctx.structured``. ``mode`` (a :data:`StructuredMode`) decides whether the layout
    route then runs for each schema, and for which fields; a schema it skips is
    finished (:meth:`~jevex.pipeline.SchemaRun.finish`) with a ``layout_route_skipped`` event.
    :func:`~jevex.extractor.default_pipeline` builds a fresh stage each time, so the
    mapper's in-memory mappings belong to one pipeline.
    """

    extractor: StructuredExtractor = field(default_factory=KeyPathMapper)
    mode: StructuredMode = "structured_only"
    name: str = "structured"

    def __post_init__(self) -> None:
        if self.mode not in get_args(StructuredMode):
            raise ValueError(
                f"unknown structured mode {self.mode!r}; use one of {get_args(StructuredMode)}"
            )

    async def run(self, ctx: Context) -> None:
        runs = ctx.active
        if not runs:
            return
        result = await self.extractor.extract(
            ctx.document, [r.spec for r in runs], ctx.jev, store=ctx.store
        )
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
            self._apply_mode(ctx, run, found)

    def _apply_mode(self, ctx: Context, run: SchemaRun, found: list[str]) -> None:
        if self.mode == "merge":
            run.merge = True
            return
        if self.mode == "structured_only":
            done = bool(found)
        else:
            done = all(not run.needs(SINGLE_ENTITY_LABEL, f.name) for f in run.spec.fields)
        if done:
            run.finish()
            ctx.event(
                self.name,
                "layout_route_skipped",
                f"{run.name}: embedded data gave what {self.mode} needs; no layout route",
                schema=run.name,
            )

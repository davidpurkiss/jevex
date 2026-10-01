"""What consumers get back: plain records with per-field metadata alongside.

Each extracted record is an instance of a *partial* variant of the consumer's model: a
generated model with the same fields, every one optional. It keeps each field's own
constraints, ``Annotated`` validators and alias, but **not** the model's model-level or
field validators, computed fields or serializers, which usually assume every field is
present. A missing value therefore never raises. Because it's a separate class,
``isinstance(item.record, VehicleSpec)`` is false. ``item.complete`` and ``item.strict()``
check the found values against the real model, validators and all.

Stages record what they found as :class:`FieldMeta` on ``SchemaRun.fields``; the
extractor turns those into :class:`Extracted` records, applying confidence thresholds.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import copy
from dataclasses import dataclass, field
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    Literal,
    Optional,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

from pydantic import BaseModel, ConfigDict, Discriminator, Field, ValidationError, create_model

from jevex.layout import Location
from jevex.statements import Span

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from jevex.schema import SchemaSpec

Method = Literal["structured", "jev", "generator", "llm", "vision"]


class Source(BaseModel):
    """Where a value came from."""

    model_config = ConfigDict(frozen=True)

    url: str | None = None
    component_id: str | None = None
    statement_id: str | None = None
    statement: str | None = None
    span: Span | None = None
    location: Location | None = None


class Alternative(BaseModel):
    """Another candidate the deciding question considered, with its probability."""

    model_config = ConfigDict(frozen=True)

    value: Any
    raw: str | None = None
    p: float


class Conflict(BaseModel):
    """A disagreeing value from another route (``merge`` mode)."""

    model_config = ConfigDict(frozen=True)

    value: Any
    method: Method | None = None
    confidence: float | None = None
    source: Source | None = None


class FieldMeta(BaseModel):
    """Everything known about one field of one record (spec: *Results API › FieldMeta*).

    ``value`` is the best answer even when a threshold kept it out of the record
    (``filtered``). ``confidence`` is Jev's confidence for the deciding question, or the
    probability for a Noul. It's ``None`` for methods that have none (e.g. ``structured``).
    """

    model_config = ConfigDict(frozen=True)

    value: Any = None
    confidence: float | None = None
    method: Method | None = None
    generator_id: str | None = None
    source: Source | None = None
    alternatives: list[Alternative] = Field(default_factory=list[Alternative])
    verified: bool | None = None
    shared: bool = False
    conflicts: list[Conflict] = Field(default_factory=list[Conflict])
    filtered: bool = False
    error: str | None = None
    """Why the value didn't fit the field's type, if it didn't."""

    @property
    def found(self) -> bool:
        return self.value is not None


class FieldMetas(Mapping[str, FieldMeta]):
    """Per-field metadata with attribute access: ``item.meta.zero_to_62_s``."""

    __slots__ = ("_by_name",)

    def __init__(self, by_name: Mapping[str, FieldMeta]) -> None:
        object.__setattr__(self, "_by_name", dict(by_name))

    _by_name: dict[str, FieldMeta]

    def __getitem__(self, name: str) -> FieldMeta:
        return self._by_name[name]

    def __getattr__(self, name: str) -> FieldMeta:
        try:
            return self._by_name[name]
        except KeyError:
            raise AttributeError(f"no field named {name!r}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def __repr__(self) -> str:
        found = [k for k, v in self._by_name.items() if v.found]
        return f"FieldMetas(found={found})"


_PARTIALS: dict[type[BaseModel], type[BaseModel]] = {}

# What a user validator on the real model may raise besides ValidationError when it meets
# an incomplete record (e.g. comparing a field with None).
_VALIDATOR_ERRORS = (ValidationError, ValueError, TypeError, AssertionError, AttributeError)


def partial_model[M: BaseModel](model: type[M]) -> type[M]:
    """A model with ``model``'s fields, every one optional and defaulting to ``None``.

    Each field keeps its constraints, ``Annotated`` validators, alias and description;
    only the default and optionality change, and a nested model (``Variant``,
    ``list[Variant]``) becomes its partial too, so a child record missing a field fits
    (except in a discriminated union). That includes a model nested in itself, or models
    nested in each other: their partials refer to each other, whichever is asked for first.
    Model-level behaviour (model and field validators, computed fields, serializers,
    ``extra="forbid"``) is left behind so a partial record can always be built. It's
    typed as ``type[M]`` for convenient attribute access, but it isn't a subclass.
    Cached per model.
    """
    if model in _PARTIALS:
        return cast("type[M]", _PARTIALS[model])
    # Build the partials of every model reachable from this one together: each refers to
    # the others by a forward reference, resolved once they all exist, so a cycle gets
    # partials all the way round.
    refs: dict[type[BaseModel], str] = {}
    _collect_models(model, refs)
    built = {m: _build_partial(m, refs) for m in refs}
    namespace = {refs[m]: partial for m, partial in built.items()}
    for partial in built.values():
        partial.model_rebuild(_types_namespace=namespace)
    _PARTIALS.update(built)
    return cast("type[M]", built[model])


def _collect_models(model: type[BaseModel], refs: dict[type[BaseModel], str]) -> None:
    """Add ``model`` and every model its fields nest (that has no partial yet) to ``refs``,
    each with a forward-reference name for its partial."""
    if model in _PARTIALS or model in refs:
        return
    refs[model] = f"_jevex_partial_{len(refs)}"
    for info in model.model_fields.values():
        if not _discriminated(info):
            for nested in _nested_models(info.annotation):
                _collect_models(nested, refs)


def _build_partial(model: type[BaseModel], refs: Mapping[type[BaseModel], str]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        optional = copy(info)
        optional.default = None
        optional.default_factory = None
        optional.validate_default = False  # a None default must never be validated
        # A discriminated union's members keep their real type: pydantic needs each
        # one's discriminator to stay a required Literal.
        annotation = (
            info.annotation if _discriminated(info) else _partial_annotation(info.annotation, refs)
        )
        fields[name] = (Optional[annotation], optional)  # noqa: UP045 - built at runtime
    # Keep the model's value-shaping config (strict, str_strip_whitespace, use_enum_values,
    # ...) so records hold what the real model would; drop extra="forbid", which is about
    # the whole model, and allow population by field name (stages use field names).
    config = cast("ConfigDict", {k: v for k, v in model.model_config.items() if k != "extra"})
    config["populate_by_name"] = True
    # create_model's overloads don't accept a dynamic **fields mapping alongside __config__.
    return create_model(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        f"Partial{model.__name__}",
        __config__=config,
        __module__=model.__module__,
        **fields,
    )


def _discriminated(info: FieldInfo) -> bool:
    return info.discriminator is not None or any(
        isinstance(m, Discriminator) for m in info.metadata
    )


def _nested_models(annotation: Any) -> Iterator[type[BaseModel]]:
    """The models in ``annotation``: itself, or in a list or a union."""
    model: object = annotation
    if isinstance(model, type) and issubclass(model, BaseModel):
        yield model
    elif get_origin(annotation) in (list, Union, UnionType):
        for arg in get_args(annotation):
            yield from _nested_models(arg)


def _partial_annotation(annotation: Any, refs: Mapping[type[BaseModel], str]) -> Any:
    """``annotation`` with every nested model (in a list or a union too) made partial: the
    cached partial, or a forward reference to one being built."""
    model: object = annotation
    if isinstance(model, type) and issubclass(model, BaseModel):
        return ForwardRef(refs[model]) if model in refs else _PARTIALS[model]
    origin = get_origin(annotation)
    if origin is list:
        (item,) = get_args(annotation) or (Any,)
        return list[_partial_annotation(item, refs)]
    if origin in (Union, UnionType):
        return Union[tuple(_partial_annotation(a, refs) for a in get_args(annotation))]  # noqa: UP007
    return annotation


@dataclass(frozen=True)
class Extracted[T: BaseModel]:
    """One extracted record: the values plus their metadata.

    ``record`` is an instance of the partial variant of ``T`` (see :func:`partial_model`):
    the same fields, all optional, and **only** the fields. The model's methods, properties
    and computed fields aren't on it; call :meth:`strict` for a real ``T``. Values below
    the extractor's thresholds, or that don't fit the field, are ``None`` there but still
    in ``meta``.
    """

    schema_name: str
    entity: str
    record: T
    meta: FieldMetas
    model: type[T] = field(repr=False)
    children: Mapping[str, list[Extracted[BaseModel]]] = field(
        default_factory=dict[str, list["Extracted[BaseModel]"]]
    )
    """Child records (from ``ParentChild``) by nested-model field, each with its own
    ``meta``. The record holds them too, in that field."""

    @property
    def complete(self) -> bool:
        """Whether every found value (including any the type rejected) validates as ``T``."""
        try:
            self.strict()
        except _VALIDATOR_ERRORS:
            return False
        return True

    def strict(self) -> T:
        """The record as the real model, with all its validators.

        Validates the original found values once (not the partial's already-coerced ones,
        so field validators don't run twice), including values the partial rejected, so a
        bad value fails here rather than silently falling back. Filtered and unfound fields
        use the model's own defaults. Raises ``ValidationError`` if anything required is
        missing or invalid, or whatever the model's own validators raise.
        """
        return self.model.model_validate(self.found_values(), by_name=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: the record's values plus every field's metadata."""
        return {
            "schema": self.schema_name,
            "entity": self.entity,
            "record": self.record.model_dump(mode="json"),
            "meta": {name: m.model_dump(mode="json") for name, m in self.meta.items()},
            "children": {
                name: [child.to_dict() for child in kids] for name, kids in self.children.items()
            },
        }

    def found_values(self) -> dict[str, Any]:
        """The values :meth:`strict` validates: each found, unfiltered value, as found."""
        return {name: m.value for name, m in self.meta.items() if m.found and not m.filtered}


def threshold_for(
    thresholds: Mapping[str, float], default: float, schema: str, field_name: str
) -> float:
    """Per-field threshold: ``"Schema.field"`` beats ``"field"`` beats the default."""
    return thresholds.get(f"{schema}.{field_name}", thresholds.get(field_name, default))


def build_extracted(
    spec: SchemaSpec,
    entity: str,
    metas: Mapping[str, FieldMeta],
    *,
    threshold: float = 0.0,
    thresholds: Mapping[str, float] | None = None,
    children: Mapping[str, list[Extracted[BaseModel]]] | None = None,
) -> Extracted[BaseModel]:
    """Build one record from the field metadata stages produced for an entity scope.

    ``children`` holds the child records built for its nested-model fields. Such a
    field's meta has the children's values (a list, or the one child's), and no
    confidence of its own: thresholds apply inside each child.
    """
    thresholds = thresholds or {}
    children = {name: kids for name, kids in (children or {}).items() if kids}
    by_name: dict[str, FieldMeta] = {}
    data: dict[str, Any] = {}
    for f in spec.fields:
        if (kids := children.get(f.name)) is not None:
            if f.many:
                by_name[f.name] = FieldMeta(value=[k.found_values() for k in kids])
                data[f.name] = [k.record for k in kids]
            else:
                by_name[f.name] = FieldMeta(value=kids[0].found_values())
                data[f.name] = kids[0].record
            continue
        meta = metas.get(f.name, FieldMeta())
        limit = threshold_for(thresholds, threshold, spec.name, f.name)
        below = meta.found and meta.confidence is not None and meta.confidence < limit
        if below:
            meta = meta.model_copy(update={"filtered": True})
        elif meta.found:
            data[f.name] = meta.value
        by_name[f.name] = meta

    partial = partial_model(spec.model)
    record: BaseModel | None = None
    while record is None:
        try:
            record = partial.model_validate(data)
        except _VALIDATOR_ERRORS as exc:
            errors = _field_errors(exc, data)
            if not errors:  # nothing attributable to a field: keep what's valid on its own
                errors = _isolate_errors(partial, data)
            if not errors:  # can't blame anything (shouldn't happen): keep no values
                record = partial.model_construct()
                break
            for name, message in errors.items():
                by_name[name] = by_name[name].model_copy(update={"error": message})
                del data[name]

    return Extracted(
        schema_name=spec.name,
        entity=entity,
        record=record,
        meta=FieldMetas(by_name),
        model=spec.model,
        children=children,
    )


def inherit(own: Mapping[str, FieldMeta], parent: Mapping[str, FieldMeta]) -> dict[str, FieldMeta]:
    """A child's metadata: its own, plus each value only its parent found, marked
    ``shared``. A child's own value always wins."""
    out = dict(own)
    for name, meta in parent.items():
        mine = out.get(name)
        if meta.found and (mine is None or not mine.found):
            out[name] = meta.model_copy(update={"shared": True})
    return out


def _field_errors(exc: Exception, data: Mapping[str, Any]) -> dict[str, str]:
    """Error messages per top-level field, keeping any nested path ("cyl: Field required")."""
    if not isinstance(exc, ValidationError):
        return {}
    out: dict[str, list[str]] = {}
    for e in exc.errors():
        loc = e["loc"]
        if loc and loc[0] in data:
            rest = ".".join(str(part) for part in loc[1:])
            out.setdefault(str(loc[0]), []).append(f"{rest}: {e['msg']}" if rest else e["msg"])
    return {name: "; ".join(messages) for name, messages in out.items()}


def _isolate_errors(partial: type[BaseModel], data: Mapping[str, Any]) -> dict[str, str]:
    """Validate each field alone to find which ones fail (or blame them all)."""
    errors: dict[str, str] = {}
    for name, value in data.items():
        try:
            partial.model_validate({name: value})
        except _VALIDATOR_ERRORS as exc:
            errors[name] = str(exc).splitlines()[0]
    if errors or not data:
        return errors
    return dict.fromkeys(data, "record failed validation")


@overload
def select_records[T: BaseModel](
    records: list[Extracted[BaseModel]], model: type[T]
) -> list[Extracted[T]]: ...
@overload
def select_records(
    records: list[Extracted[BaseModel]], model: None = None
) -> list[Extracted[BaseModel]]: ...
def select_records(
    records: list[Extracted[BaseModel]], model: type[BaseModel] | None = None
) -> list[Extracted[Any]]:
    if model is None:
        return list(records)
    return [r for r in records if r.model is model]

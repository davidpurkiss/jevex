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
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, overload

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from jevex.layout import Location
from jevex.statements import Span

if TYPE_CHECKING:
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
    method: Method
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
    only the default and optionality change. Model-level behaviour (model and field
    validators, computed fields, serializers, ``extra="forbid"``) is left behind so a
    partial record can always be built. It's typed as ``type[M]`` for convenient
    attribute access, but it isn't a subclass. Cached per model.
    """
    if model in _PARTIALS:
        return cast("type[M]", _PARTIALS[model])
    fields: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        optional = copy(info)
        optional.default = None
        optional.default_factory = None
        optional.validate_default = False  # a None default must never be validated
        fields[name] = (Optional[info.annotation], optional)  # noqa: UP045 - built at runtime
    # Keep the model's value-shaping config (strict, str_strip_whitespace, use_enum_values,
    # ...) so records hold what the real model would; drop extra="forbid", which is about
    # the whole model, and allow population by field name (stages use field names).
    config = cast("ConfigDict", {k: v for k, v in model.model_config.items() if k != "extra"})
    config["populate_by_name"] = True
    # create_model's overloads don't accept a dynamic **fields mapping alongside __config__.
    partial = create_model(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        f"Partial{model.__name__}",
        __config__=config,
        __module__=model.__module__,
        **fields,
    )
    _PARTIALS[model] = partial
    return cast("type[M]", partial)


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
        values = {name: m.value for name, m in self.meta.items() if m.found and not m.filtered}
        return self.model.model_validate(values, by_name=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: the record's values plus every field's metadata."""
        return {
            "schema": self.schema_name,
            "entity": self.entity,
            "record": self.record.model_dump(mode="json"),
            "meta": {name: m.model_dump(mode="json") for name, m in self.meta.items()},
        }


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
) -> Extracted[BaseModel]:
    """Build one record from the field metadata stages produced for an entity scope."""
    thresholds = thresholds or {}
    by_name: dict[str, FieldMeta] = {}
    data: dict[str, Any] = {}
    for f in spec.fields:
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
    )


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

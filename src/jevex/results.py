"""What consumers get back: plain records with per-field metadata alongside.

Each extracted record is an instance of a *partial* variant of the consumer's model: a
generated subclass in which every field is optional. A missing or rejected value
therefore never raises, and ``isinstance(item.record, VehicleSpec)`` still holds.
``item.complete`` / ``item.strict()`` check the record against the real model.

Stages record what they found as :class:`FieldMeta` on ``SchemaRun.fields``; the
extractor turns those into :class:`Extracted` records, applying confidence thresholds.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
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


def partial_model[M: BaseModel](model: type[M]) -> type[M]:
    """A subclass of ``model`` in which every field is optional and defaults to ``None``.

    Built once per model and cached (a dict, not ``functools.cache``, which would erase
    the generic return type).
    """
    if model in _PARTIALS:
        return cast("type[M]", _PARTIALS[model])
    overrides: dict[str, Any] = {
        name: (Optional[info.annotation], Field(default=None, description=info.description))  # noqa: UP045 - annotation is a runtime object, not syntax
        for name, info in model.model_fields.items()
    }
    partial = create_model(
        f"Partial{model.__name__}",
        __base__=model,
        __module__=model.__module__,
        **overrides,
    )
    _PARTIALS[model] = partial
    return partial


@dataclass(frozen=True)
class Extracted[T: BaseModel]:
    """One extracted record: the values plus their metadata.

    ``record`` is an instance of the partial variant of ``T`` (see :func:`partial_model`).
    Values below the extractor's thresholds are ``None`` there but still in ``meta``.
    """

    schema_name: str
    entity: str
    record: T
    meta: FieldMetas
    model: type[T] = field(repr=False)

    @property
    def complete(self) -> bool:
        """Whether the record validates against the real (strict) model."""
        try:
            self.strict()
        except ValidationError:
            return False
        return True

    def strict(self) -> T:
        """The record as the real model. Raises ``ValidationError`` if anything's missing.

        Fields that weren't found fall back to the model's own defaults.
        """
        return self.model.model_validate(self.record.model_dump(exclude_unset=True))

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
    while True:
        try:
            record = partial.model_validate(data)
            break
        except ValidationError as exc:
            bad = {str(e["loc"][0]) for e in exc.errors() if e["loc"] and e["loc"][0] in data}
            if not bad:
                raise
            for name in bad:
                messages = [e["msg"] for e in exc.errors() if e["loc"] and e["loc"][0] == name]
                by_name[name] = by_name[name].model_copy(update={"error": "; ".join(messages)})
                del data[name]

    return Extracted(
        schema_name=spec.name,
        entity=entity,
        record=record,
        meta=FieldMetas(by_name),
        model=spec.model,
    )


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

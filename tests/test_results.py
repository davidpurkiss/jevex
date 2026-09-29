from dataclasses import dataclass
from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel, ValidationError

from jevex import Context, Document, EntityScope, Extractor, Field, Pipeline
from jevex.results import (
    Alternative,
    Extracted,
    FieldMeta,
    FieldMetas,
    Source,
    build_extracted,
    partial_model,
)
from jevex.schema import SchemaSpec
from jevex.testing import FakeJev


class VehicleSpec(BaseModel):
    """A car's technical specification."""

    model: str = Field(description="Model name")
    fuel_type: Literal["petrol", "diesel", "ev"] = Field(description="Fuel type")
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    seats: int = Field(default=5, description="Seats")
    features: list[str] = Field(default_factory=list, description="Features")

    @property
    def is_fast(self) -> bool:
        # Truthiness, not "is not None": partial records can hold None here.
        return bool(self.zero_to_62_s) and self.zero_to_62_s < 7


class Book(BaseModel):
    title: str = Field(description="Title")


SPEC = SchemaSpec.from_model(VehicleSpec)


def build(entity: str, metas: dict[str, FieldMeta], **kw: Any) -> Extracted[VehicleSpec]:
    """build_extracted, typed for VehicleSpec (the public accessors are typed already)."""
    return cast("Extracted[VehicleSpec]", build_extracted(SPEC, entity, metas, **kw))


def meta(value: object, confidence: float | None = 0.9, **kw: object) -> FieldMeta:
    return FieldMeta(value=value, confidence=confidence, method="jev", **kw)  # pyright: ignore[reportArgumentType]


# --- partial model ---------------------------------------------------------------------


def test_partial_model_is_a_subclass_with_every_field_optional() -> None:
    partial = partial_model(VehicleSpec)
    empty = partial()
    assert isinstance(empty, VehicleSpec)
    assert empty.model is None
    assert empty.seats is None
    assert partial is partial_model(VehicleSpec)  # cached
    assert partial.__name__ == "PartialVehicleSpec"
    assert partial.model_fields["zero_to_62_s"].description == "0-62 mph time"


def test_partial_record_keeps_model_behaviour() -> None:
    record = partial_model(VehicleSpec)(zero_to_62_s=6.2)
    assert record.is_fast


# --- building records ------------------------------------------------------------------


def test_build_extracted_with_full_meta() -> None:
    source = Source(url="https://example.com", statement="0-62 mph in 9.1 s")
    item = build(
        "SE",
        {
            "model": meta("Golf"),
            "fuel_type": meta("petrol"),
            "zero_to_62_s": meta(
                9.1, source=source, alternatives=[Alternative(value=62.0, raw="62 mph", p=0.1)]
            ),
        },
    )
    assert item.record.model == "Golf"
    assert item.record.zero_to_62_s == 9.1
    assert item.meta.zero_to_62_s.source == source
    assert item.meta["zero_to_62_s"].alternatives[0].raw == "62 mph"
    assert item.entity == "SE"
    assert item.schema_name == "VehicleSpec"


def test_missing_fields_are_none_and_have_empty_meta() -> None:
    item = build("doc", {"model": meta("Golf")})
    assert item.record.fuel_type is None
    assert not item.meta.fuel_type.found
    assert set(item.meta) == {"model", "fuel_type", "zero_to_62_s", "seats", "features"}
    with pytest.raises(AttributeError, match="no field named 'nope'"):
        _ = item.meta.nope


def test_complete_and_strict() -> None:
    partial = build("doc", {"model": meta("Golf")})
    assert not partial.complete
    with pytest.raises(ValidationError):
        partial.strict()

    full = build(
        "doc",
        {"model": meta("Golf"), "fuel_type": meta("ev"), "zero_to_62_s": meta(7.9)},
    )
    assert full.complete
    strict = full.strict()
    assert type(strict) is VehicleSpec
    assert strict.seats == 5  # not found: the model's own default applies
    assert strict.features == []


def test_values_that_dont_fit_the_type_are_dropped_with_an_error() -> None:
    item = build(
        "doc", {"model": meta("Golf"), "fuel_type": meta("hydrogen"), "seats": meta("lots")}
    )
    assert item.record.model == "Golf"
    assert item.record.fuel_type is None
    assert item.record.seats is None
    assert item.meta.fuel_type.value == "hydrogen"
    assert item.meta.fuel_type.error
    assert item.meta.seats.error
    assert item.meta.model.error is None


def test_values_are_coerced_by_the_model() -> None:
    item = build("doc", {"seats": meta("7"), "zero_to_62_s": meta("9.1")})
    assert item.record.seats == 7
    assert item.record.zero_to_62_s == 9.1


# --- thresholds ------------------------------------------------------------------------


def test_default_threshold_keeps_everything() -> None:
    item = build("doc", {"model": meta("Golf", confidence=0.01)})
    assert item.record.model == "Golf"
    assert not item.meta.model.filtered


def test_threshold_filters_low_confidence_values_but_keeps_meta() -> None:
    item = build(
        "doc",
        {"model": meta("Golf", confidence=0.5), "fuel_type": meta("ev", confidence=0.95)},
        threshold=0.8,
    )
    assert item.record.model is None
    assert item.meta.model.value == "Golf"
    assert item.meta.model.filtered
    assert item.record.fuel_type == "ev"


def test_values_without_confidence_pass_thresholds() -> None:
    item = build("doc", {"model": FieldMeta(value="Golf", method="structured")}, threshold=0.99)
    assert item.record.model == "Golf"


def test_per_field_thresholds_qualified_beats_bare_beats_default() -> None:
    metas = {"model": meta("Golf", confidence=0.7), "fuel_type": meta("ev", confidence=0.7)}
    item = build(
        "doc",
        metas,
        threshold=0.5,
        thresholds={"model": 0.9, "VehicleSpec.fuel_type": 0.6, "fuel_type": 0.99},
    )
    assert item.record.model is None  # bare 0.9 beats default 0.5
    assert item.record.fuel_type == "ev"  # qualified 0.6 beats bare 0.99


# --- serialisation ---------------------------------------------------------------------


def test_to_dict_is_json_ready() -> None:
    item = build("SE", {"model": meta("Golf")})
    d = item.to_dict()
    assert d["schema"] == "VehicleSpec"
    assert d["record"]["model"] == "Golf"
    assert d["meta"]["model"]["confidence"] == 0.9
    assert d["meta"]["fuel_type"]["value"] is None


def test_field_metas_is_a_mapping() -> None:
    metas = FieldMetas({"a": FieldMeta(value=1)})
    assert dict(metas) == {"a": FieldMeta(value=1)}
    assert len(metas) == 1
    assert "found=['a']" in repr(metas)


# --- through the Extractor -------------------------------------------------------------


@dataclass
class Record:
    """Test stage: two SE/SEL scopes for VehicleSpec, one record for Book."""

    name: str = "record"

    async def run(self, ctx: Context) -> None:
        car = ctx.schemas["VehicleSpec"]
        car.scopes = [EntityScope(label="SE"), EntityScope(label="SEL")]
        car.set_field("SEL", "model", meta("Golf", confidence=0.4))
        car.set_field("SE", "model", meta("Golf", confidence=0.95))
        car.values["SE"] = {"seats": 5}  # bare values still work
        ctx.schemas["Book"].set_field("doc", "title", meta("Dune"))


def extractor(**kwargs: object) -> Extractor:
    return Extractor(
        [VehicleSpec, Book],
        jev=FakeJev().client(),
        pipeline=Pipeline([Record()]),
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


def doc() -> Document:
    return Document.from_bytes(b"<p>x</p>", url="https://example.com")


async def test_records_come_back_in_schema_then_scope_order() -> None:
    result = await extractor().extract(doc())
    assert [(r.schema_name, r.entity) for r in result.records] == [
        ("VehicleSpec", "SE"),
        ("VehicleSpec", "SEL"),
        ("Book", "doc"),
    ]
    se = result.for_schema(VehicleSpec)[0]
    assert se.record.seats == 5
    assert se.meta.seats.method is None  # a bare value has no metadata


async def test_one_and_for_schema_are_typed_accessors() -> None:
    result = await extractor().extract(doc())
    book = result.one(Book)
    assert book.record.title == "Dune"
    assert [r.entity for r in result.for_schema(VehicleSpec)] == ["SE", "SEL"]
    assert len(result.for_schema()) == 3
    with pytest.raises(LookupError, match="exactly one record for VehicleSpec, found 2"):
        result.one(VehicleSpec)
    with pytest.raises(LookupError, match="found 3"):
        result.one()


async def test_extractor_thresholds_apply_to_records() -> None:
    result = await extractor(threshold=0.5).extract(doc())
    se, sel = result.for_schema(VehicleSpec)
    assert se.record.model == "Golf"
    assert sel.record.model is None
    assert sel.meta.model.filtered
    assert result.values["VehicleSpec"] == {"SE": {"model": "Golf", "seats": 5}}


async def test_result_to_dict() -> None:
    d = (await extractor().extract(doc())).to_dict()
    assert [r["entity"] for r in d["records"]] == ["SE", "SEL", "doc"]
    assert d["meta"]["url"] == "https://example.com"

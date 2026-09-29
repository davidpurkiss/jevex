from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal, cast

import pytest
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

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


class Book(BaseModel):
    title: str = Field(description="Title")


SPEC = SchemaSpec.from_model(VehicleSpec)


def build(entity: str, metas: dict[str, FieldMeta], **kw: Any) -> Extracted[VehicleSpec]:
    """build_extracted, typed for VehicleSpec (the public accessors are typed already)."""
    return cast("Extracted[VehicleSpec]", build_extracted(SPEC, entity, metas, **kw))


def meta(value: object, confidence: float | None = 0.9, **kw: Any) -> FieldMeta:
    return FieldMeta(value=value, confidence=confidence, method="jev", **kw)


# --- partial model ---------------------------------------------------------------------


def test_partial_model_has_every_field_optional() -> None:
    partial = partial_model(VehicleSpec)
    empty = partial()
    assert not isinstance(empty, VehicleSpec)  # a separate model, same fields
    assert empty.model is None
    assert empty.seats is None
    assert partial is partial_model(VehicleSpec)  # cached
    assert partial.__name__ == "PartialVehicleSpec"
    assert partial.model_fields["zero_to_62_s"].description == "0-62 mph time"


# --- user models with validators, constraints, aliases, computed fields ----------------


def upper(v: str) -> str:
    return v.upper()


class Tricky(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lo: int = Field(description="Low")
    hi: int = Field(description="High")
    seats: int = Field(default=5, ge=1, le=9, description="Seats")
    name: Annotated[str, AfterValidator(upper)] = Field(default="", description="Name")
    zero_to_62: float = Field(default=0.0, alias="zeroTo62", description="0-62")
    registered: date | None = Field(default=None, description="Registered")

    @model_validator(mode="after")
    def ordered(self) -> "Tricky":
        if self.lo > self.hi:
            raise ValueError("lo must not exceed hi")
        return self

    @field_validator("hi")
    @classmethod
    def hi_after_lo(cls, v: int, info: Any) -> int:
        if v < info.data["lo"]:  # reads another field: fails when lo is missing
            raise ValueError("hi < lo")
        return v

    @computed_field
    @property
    def span(self) -> int:
        return self.hi - self.lo

    @field_serializer("registered")
    def fmt(self, v: date | None) -> str | None:
        return v.strftime("%d/%m/%Y") if v else None


TRICKY = SchemaSpec.from_model(Tricky)


def tricky(metas: dict[str, FieldMeta]) -> Extracted[Tricky]:
    return cast("Extracted[Tricky]", build_extracted(TRICKY, "doc", metas))


def test_user_validators_never_break_partial_records() -> None:
    item = tricky({"lo": meta(3)})  # hi missing: the model validator would compare with None
    assert item.record.lo == 3
    assert not item.complete  # hi is required
    item = tricky({"hi": meta(2)})  # the field validator would read a missing lo
    assert item.record.hi == 2


def test_model_level_validation_failures_surface_in_strict_not_extraction() -> None:
    item = tricky({"lo": meta(5), "hi": meta(9)})
    assert item.complete
    bad = tricky({"lo": meta(9), "hi": meta(5)})
    assert bad.record.lo == 9  # the partial record is still built
    assert not bad.complete
    with pytest.raises(ValidationError, match=r"lo must not exceed hi|hi < lo"):
        bad.strict()


def test_field_constraints_and_annotated_validators_are_kept() -> None:
    item = tricky({"seats": meta(-3), "name": meta("golf")})
    assert item.record.seats is None
    assert item.meta.seats.error is not None
    assert "greater than or equal to 1" in item.meta.seats.error
    assert item.record.name == "GOLF"


def test_aliased_fields_work_by_name() -> None:
    item = tricky({"lo": meta(1), "hi": meta(2), "zero_to_62": meta(9.1)})
    assert item.record.zero_to_62 == 9.1
    strict = item.strict()
    assert strict.zero_to_62 == 9.1
    assert strict.span == 1  # computed fields work on the strict model


def test_strict_validates_original_values_once() -> None:
    item = tricky({"lo": meta(1), "hi": meta(2), "name": meta("golf")})
    assert item.strict().name == "GOLF"


def test_serializers_and_forbid_dont_break_strict_or_to_dict() -> None:
    item = tricky({"lo": meta(1), "hi": meta(2), "registered": meta("2024-03-12")})
    assert item.complete
    assert item.strict().registered == date(2024, 3, 12)
    assert item.to_dict()["record"]["registered"] == "2024-03-12"
    empty = tricky({})
    assert empty.to_dict()["record"]["lo"] is None  # no computed field to crash


class Engine(BaseModel):
    cyl: int
    power: int


class WithNested(BaseModel):
    engine: Engine | None = Field(default=None, description="Engine")
    tags: list[int] = Field(default_factory=list, description="Tags")


def test_nested_errors_keep_their_path() -> None:
    item = cast(
        "Extracted[WithNested]",
        build_extracted(
            SchemaSpec.from_model(WithNested),
            "doc",
            {"engine": meta({"power": 100}), "tags": meta([1, "x", 3])},
        ),
    )
    assert item.record.engine is None
    assert item.meta.engine.error == "cyl: Field required"
    assert item.meta.tags.error is not None
    assert item.meta.tags.error.startswith("1: ")


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


def extractor(**kwargs: Any) -> Extractor:
    return Extractor(
        [VehicleSpec, Book], jev=FakeJev().client(), pipeline=Pipeline([Record()]), **kwargs
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


async def test_values_view_matches_the_records() -> None:
    @dataclass
    class Messy:
        name: str = "record"

        async def run(self, ctx: Context) -> None:
            car = ctx.schemas["VehicleSpec"]
            car.set_field("doc", "seats", meta("7"))  # coerced to 7 in the record
            car.set_field("doc", "fuel_type", meta("hydrogen"))  # rejected by the type

    ex = Extractor([VehicleSpec], jev=FakeJev().client(), pipeline=Pipeline([Messy()]))
    result = await ex.extract(doc())
    assert result.values == {"VehicleSpec": {"doc": {"seats": 7}}}


def test_unknown_threshold_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="feul"):
        extractor(thresholds={"feul": 0.9})
    extractor(thresholds={"VehicleSpec.model": 0.9, "title": 0.5})


async def test_result_to_dict() -> None:
    d = (await extractor().extract(doc())).to_dict()
    assert [r["entity"] for r in d["records"]] == ["SE", "SEL", "doc"]
    assert d["meta"]["url"] == "https://example.com"

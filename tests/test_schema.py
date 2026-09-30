import enum
from datetime import date
from decimal import Decimal
from typing import Literal, Optional

import pytest
from pydantic import BaseModel

from jevex import Field, Questions, SchemaConfig, SchemaSpec
from jevex.jev import Choice, Noul
from jevex.schema import ALL_OPTION, ReservedFieldNameError, UnsupportedFieldError


class VehicleSpec(BaseModel):
    """A manufacturer's technical specification for one vehicle variant."""

    __jevex__ = SchemaConfig(
        document_question="Does this document contain a car's technical specification?",
    )

    model: str = Field(description="Model name, e.g. Golf")
    trim: str = Field(description="Trim or grade name, e.g. SE L")
    fuel_type: Literal["petrol", "diesel", "hybrid", "phev", "ev"] = Field(
        description="Fuel or powertrain type",
    )
    engine_size_cc: int | None = Field(description="Engine displacement", unit="cc")
    zero_to_62_s: float = Field(
        description="0-62 mph acceleration time",
        unit="s",
        questions=Questions(select="Which value is the 0-62 mph time in seconds?"),
    )


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Owner(BaseModel):
    name: str


class Listing(BaseModel):
    price: Decimal = Field(description="Asking price", group="price")
    currency: Literal["GBP", "EUR"] = Field(description="Currency", group="price")
    first_registered: date = Field(description="First registration date")
    colour: Colour = Field(description="Exterior colour")
    automatic: bool = Field(description="has an automatic gearbox")
    features: list[str] = Field(default_factory=list, description="Listed features")
    previous_owners: list[Owner] = Field(default_factory=list, description="Previous owners")
    mileage: Optional[int] = None  # noqa: UP045 - checks typing.Optional too


def test_field_is_plain_pydantic() -> None:
    spec = VehicleSpec(
        model="Golf", trim="SE L", fuel_type="petrol", engine_size_cc=1498, zero_to_62_s=9.1
    )
    assert spec.model_dump()["zero_to_62_s"] == 9.1
    schema = VehicleSpec.model_json_schema()
    assert schema["properties"]["engine_size_cc"]["jevex"] == {"unit": "cc"}
    assert schema["properties"]["zero_to_62_s"]["jevex"] == {
        "unit": "s",
        "questions": {"select": "Which value is the 0-62 mph time in seconds?"},
    }
    assert "jevex" not in schema["properties"]["model"]


def test_field_keeps_other_json_schema_extra() -> None:
    class M(BaseModel):
        x: int = Field(description="X", unit="mm", json_schema_extra={"examples": [1]})

    assert M.model_json_schema()["properties"]["x"]["examples"] == [1]
    assert M.model_json_schema()["properties"]["x"]["jevex"] == {"unit": "mm"}


def test_field_specs_for_vehicle() -> None:
    spec = SchemaSpec.from_model(VehicleSpec)
    assert spec.name == "VehicleSpec"
    assert [(f.name, f.kind, f.required, f.nullable) for f in spec.fields] == [
        ("model", "str", True, False),
        ("trim", "str", True, False),
        ("fuel_type", "enum", True, False),
        ("engine_size_cc", "number", True, True),
        ("zero_to_62_s", "number", True, False),
    ]
    assert spec.field("fuel_type").options == ("petrol", "diesel", "hybrid", "phev", "ev")
    assert spec.field("engine_size_cc").label == "Engine displacement (cc)"
    assert not spec.field("fuel_type").needs_candidates
    assert spec.field("zero_to_62_s").needs_candidates


def test_field_kinds_for_listing() -> None:
    spec = SchemaSpec.from_model(Listing)
    kinds = {f.name: (f.kind, f.many) for f in spec.fields}
    assert kinds == {
        "price": ("number", False),
        "currency": ("enum", False),
        "first_registered": ("date", False),
        "colour": ("enum", False),
        "automatic": ("bool", False),
        "features": ("str", True),
        "previous_owners": ("model", True),
        "mileage": ("number", False),
    }
    assert spec.field("colour").options == ("red", "blue")
    assert spec.field("previous_owners").model is Owner
    assert spec.field("mileage").nullable
    assert spec.field("mileage").description == "mileage"
    assert not spec.field("features").required


def test_unsupported_type_is_rejected() -> None:
    class Bad(BaseModel):
        blob: bytes = Field(description="Raw bytes")

    with pytest.raises(UnsupportedFieldError, match="blob"):
        SchemaSpec.from_model(Bad)


def test_bad_config_is_rejected() -> None:
    class Bad(BaseModel):
        __jevex__ = {"document_question": "?"}
        x: int

    with pytest.raises(TypeError, match="SchemaConfig"):
        SchemaSpec.from_model(Bad)


# --- Generated questions (snapshots) ---------------------------------------------------


def test_document_gate_uses_override() -> None:
    assert SchemaSpec.from_model(VehicleSpec).document_gate_question() == Noul(
        instructions="Does this document contain a car's technical specification?"
    )


def test_document_gate_from_docstring() -> None:
    class Spec(BaseModel):
        """A manufacturer's technical specification for one vehicle variant.

        Longer notes that should not end up in the question.
        """

        x: int

    assert SchemaSpec.from_model(Spec).document_gate_question().instructions == (
        "Does this document describe a manufacturer's technical specification "
        "for one vehicle variant?"
    )


def test_document_gate_keeps_acronyms_and_falls_back_to_class_name() -> None:
    class EVCharger(BaseModel):
        """EV charger listing."""

        x: int

    class UsedCarListing(BaseModel):
        x: int

    assert "describe EV charger listing?" in str(
        SchemaSpec.from_model(EVCharger).document_gate_question().instructions
    )
    assert SchemaSpec.from_model(UsedCarListing).document_gate_question().instructions == (
        "Does this document describe used car listing?"
    )


def test_component_gates_per_group() -> None:
    gates = SchemaSpec.from_model(Listing).component_gate_questions()
    assert list(gates) == [
        "price",
        "first_registered",
        "colour",
        "automatic",
        "features",
        "previous_owners",
        "mileage",
    ]
    assert gates["price"] == Noul(
        instructions="Does this section contain the asking price or currency?"
    )
    assert gates["first_registered"] == Noul(
        instructions="Does this section contain the first registration date?"
    )


def test_component_gate_override() -> None:
    class M(BaseModel):
        x: int = Field(description="X", questions=Questions(component_gate="Is there an X here?"))

    assert SchemaSpec.from_model(M).component_gate_questions() == {
        "x": Noul(instructions="Is there an X here?")
    }


def test_categorise_question() -> None:
    assert SchemaSpec.from_model(VehicleSpec).categorise_question() == Choice(
        instructions="Which detail does this statement state?",
        options={
            "model": "Model name, e.g. Golf",
            "trim": "Trim or grade name, e.g. SE L",
            "fuel_type": "Fuel or powertrain type",
            "engine_size_cc": "Engine displacement (cc)",
            "zero_to_62_s": "0-62 mph acceleration time (s)",
            "none": "None of these details",
        },
    )


def test_categorise_skips_child_models() -> None:
    options = SchemaSpec.from_model(Listing).categorise_question().options
    assert "previous_owners" not in options


def test_select_question() -> None:
    spec = SchemaSpec.from_model(VehicleSpec)
    assert spec.field("zero_to_62_s").select_question(["9.1 s", "62 mph", "9.1 s"]) == Choice(
        instructions="Which value is the 0-62 mph time in seconds?",
        options={
            "9.1 s": None,
            "62 mph": None,
            "none": "None of these is the 0-62 mph acceleration time",
        },
    )
    assert spec.field("engine_size_cc").select_instructions() == (
        "Which of these is the engine displacement (cc)?"
    )
    with pytest.raises(ValueError, match="reserved"):
        spec.field("model").select_question(["none"])


def test_enum_bool_and_verify_questions() -> None:
    vehicle = SchemaSpec.from_model(VehicleSpec)
    listing = SchemaSpec.from_model(Listing)
    assert vehicle.field("fuel_type").enum_question() == Choice(
        instructions="What is the fuel or powertrain type?",
        options={
            "petrol": None,
            "diesel": None,
            "hybrid": None,
            "phev": None,
            "ev": None,
            "not stated": "The statement does not state the fuel or powertrain type",
        },
    )
    assert listing.field("automatic").bool_question() == Noul(
        instructions="Does the statement say it has an automatic gearbox?"
    )
    assert vehicle.field("zero_to_62_s").verify_question(9.1) == Noul(
        instructions="The statement states that the 0-62 mph acceleration time (s) is 9.1."
    )
    with pytest.raises(ValueError, match="not an enum"):
        vehicle.field("model").enum_question()


def test_first_registered_is_date_kind() -> None:
    assert SchemaSpec.from_model(Listing).field("first_registered").annotation is date


def test_member_question_default_and_override() -> None:
    class M(BaseModel):
        tags: list[str] = Field(default_factory=list, description="Tags")
        colours: list[str] = Field(
            default_factory=list,
            description="Colours",
            questions=Questions(member="Is {value} a {description}?"),
        )

    spec = SchemaSpec.from_model(M)
    assert spec.field("tags").member_question("red") == Noul(
        instructions='Does the statement give "red" as one of the tags?'
    )
    assert spec.field("colours").member_question("red") == Noul(instructions="Is red a Colours?")


class GatedCar(BaseModel):
    """A car's specification."""

    price: Decimal = Field(description="Price", unit="GBP")
    power_kw: float = Field(description="Engine power", unit="kW", group="performance")
    zero_to_62_s: float = Field(
        description="0-62 mph time",
        unit="s",
        group="performance",
        questions=Questions(categorise="How long it takes to reach 62 mph"),
    )


def test_categorise_question_can_be_limited_to_some_fields() -> None:
    spec = SchemaSpec.from_model(GatedCar)
    assert list(spec.categorise_question().options) == [
        "price",
        "power_kw",
        "zero_to_62_s",
        "none",
    ]
    limited = spec.categorise_question(["zero_to_62_s"])
    assert limited.options == {
        "zero_to_62_s": "How long it takes to reach 62 mph",
        "none": "None of these details",
    }


def test_a_field_named_none_is_reserved() -> None:
    class Odd(BaseModel):
        none: str = Field(description="Nothing")

    with pytest.raises(ReservedFieldNameError, match="reserved"):
        SchemaSpec.from_model(Odd)


class Gated(BaseModel):
    """A car."""

    price: Decimal = Field(description="Asking price", unit="GBP", group="price")
    vat: bool = Field(description="VAT is included", group="price")
    sunroof: bool = Field(description="Sunroof", group="kit")
    tow_bar: bool = Field(description="Tow bar", group="kit")
    heated: bool = Field(description="The seats are heated", group="comfort")
    cooled: bool = Field(description="The seats are cooled", group="comfort")
    automatic: bool = Field(
        description="has an automatic gearbox",
        questions=Questions(component_gate="Is the gearbox described here?"),
    )
    in_stock: bool = Field(description="The book is in stock")


def test_gate_questions_for_bools_and_mixed_groups() -> None:
    questions = {
        k: q.instructions
        for k, q in SchemaSpec.from_model(Gated).component_gate_questions().items()
    }
    assert questions == {
        "price": "Does this section contain the asking price (GBP) or say whether VAT is included?",
        "kit": "Does this section mention sunroof or tow bar?",
        "comfort": (
            "Does this section say whether the seats are heated or whether the seats are cooled?"
        ),
        "automatic": "Is the gearbox described here?",
        "in_stock": "Does this section say whether the book is in stock?",
    }


def test_bool_select_and_verify_questions_read_as_claims() -> None:
    spec = SchemaSpec.from_model(Gated)
    stock = spec.field("in_stock")
    assert stock.bool_question().instructions == "Does the statement say the book is in stock?"
    assert stock.verify_question(True).instructions == "The statement says the book is in stock."
    assert stock.verify_question(False).instructions == (
        "The statement says it is not the case that the book is in stock."
    )
    auto = spec.field("automatic")
    assert auto.bool_question().instructions == (
        "Does the statement say it has an automatic gearbox?"
    )
    sunroof = spec.field("sunroof")
    # The gate asks about relevance ("mention"); the value question asks about presence,
    # so "No sunroof" can't make it True.
    assert sunroof.bool_question().instructions == "Does the statement say it has sunroof?"
    assert sunroof.verify_question(True).instructions == "The statement says it has sunroof."
    assert sunroof.verify_question(False).instructions == "The statement says there is no sunroof."


def test_base_form_first_words_are_nouns_not_missing_subjects() -> None:
    class H(BaseModel):
        pool: bool = Field(description="Use of pool")
        opener: bool = Field(description="Can opener included")
        uses: bool = Field(description="uses premium fuel")

    spec = SchemaSpec.from_model(H)
    assert spec.field("pool").bool_question().instructions == (
        "Does the statement say it has use of pool?"
    )
    assert spec.field("opener").bool_question().instructions == (
        "Does the statement say it has can opener included?"
    )
    assert spec.field("uses").bool_question().instructions == (
        "Does the statement say it uses premium fuel?"
    )


def test_phrase_keeps_acronyms_and_letter_names() -> None:
    class P(BaseModel):
        vin: str = Field(description="VIN")
        ev_range: int = Field(description="EV range", unit="miles")
        price: Decimal = Field(description="Price", unit="GBP")
        pillar: str = Field(description="A-pillar colour")
        x: str = Field(description="X")
        short: str = Field(description="A short title")
        phone: str = Field(description="iPhone model")
        fuel: str = Field(description="Électrique range")

    spec = SchemaSpec.from_model(P)
    assert [f.phrase for f in spec.fields] == [
        "VIN",
        "EV range (miles)",
        "price (GBP)",
        "A-pillar colour",
        "X",
        "a short title",
        "iPhone model",
        "électrique range",
    ]


def test_custom_templates_get_the_description_as_written() -> None:
    class C(BaseModel):
        price: Decimal = Field(
            description="Price",
            unit="GBP",
            questions=Questions(verify="Is {value} the {description}?"),
        )
        ok: bool = Field(
            description="The car is OK",
            questions=Questions(verify="Does it hold that {description}: {value}?"),
        )

    spec = SchemaSpec.from_model(C)
    assert spec.field("price").verify_question(9).instructions == "Is 9 the Price (GBP)?"
    assert spec.field("ok").verify_question(True).instructions == (
        "Does it hold that The car is OK: True?"
    )


def test_entity_questions_name_the_record_after_the_model() -> None:
    spec = SchemaSpec.from_model(VehicleSpec)
    assert spec.entity_name == "vehicle spec"
    assert spec.boundary_question("SE L") == Noul(
        instructions='Does "SE L" name a separate vehicle spec?'
    )
    assert spec.entity_question(["SE", "SE L"]) == Choice(
        instructions="Which vehicle spec does this statement apply to?",
        options={"SE": None, "SE L": None, ALL_OPTION: "It applies to every vehicle spec"},
    )


def test_entity_questions_can_be_overridden() -> None:
    class Listing(BaseModel):
        __jevex__ = SchemaConfig(
            entity_name="car",
            boundary_question="Is {label} one {entity} for sale?",
            entity_question="Which {entity} is this about?",
        )
        price: int = Field(description="Price")

    spec = SchemaSpec.from_model(Listing)
    assert spec.boundary_question("Golf").instructions == "Is Golf one car for sale?"
    question = spec.entity_question(["Golf"])
    assert question.instructions == "Which car is this about?"
    assert question.options[ALL_OPTION] == "It applies to every car"


def test_entity_question_rejects_the_reserved_label() -> None:
    with pytest.raises(ValueError, match="reserved"):
        SchemaSpec.from_model(VehicleSpec).entity_question(["SE", ALL_OPTION])

from __future__ import annotations

from jevex.jev import Choice, Noul
from jevex.schema import ALL_OPTION, SchemaSpec
from jevex.testsite.schemas import Listing, VehicleSpec


def test_vehicle_spec_entities_are_vehicle_variants() -> None:
    spec = SchemaSpec.from_model(VehicleSpec)
    assert spec.boundary_question("SE L") == Noul(
        instructions='Does "SE L" name a separate vehicle variant?'
    )
    assert spec.entity_question(["SE", "SE L"]) == Choice(
        instructions="Which vehicle variant does this statement apply to?",
        options={"SE": None, "SE L": None, ALL_OPTION: "It applies to every vehicle variant"},
    )


def test_a_listing_keeps_its_name_as_the_entity() -> None:
    spec = SchemaSpec.from_model(Listing)
    assert spec.boundary_question("Kestrova SE") == Noul(
        instructions='Does "Kestrova SE" name a separate listing?'
    )


def test_a_vehicle_spec_price_is_the_on_the_road_one_before_a_grant() -> None:
    price = SchemaSpec.from_model(VehicleSpec).field("price_gbp")
    assert price.select_question(["£9,354.17", "£11,990"]) == Choice(
        instructions=(
            "Which of these is the vehicle's recommended on-the-road price, before any grant?"
        ),
        options={"£9,354.17": None, "£11,990": None, "none": "None of these is the price"},
    )


def test_a_vehicle_specs_names_are_asked_for_without_what_a_label_adds_to_them() -> None:
    spec = SchemaSpec.from_model(VehicleSpec)
    asked = {
        name: spec.field(name).select_question(["Premium 1.2 79PS"]).instructions
        for name in ("model", "trim", "engine_size_cc")
    }
    assert asked == {
        "model": (
            "Which of these is the name of the model alone, not its code, without the make, "
            "trim level or engine?"
        ),
        "trim": (
            "Which of these is the trim level alone, e.g. SE L, without the engine, power or "
            "body style?"
        ),
        "engine_size_cc": "Which of these is the engine displacement in cc, not a size in litres?",
    }


def test_a_listing_asks_for_its_names_by_their_descriptions() -> None:
    spec = SchemaSpec.from_model(Listing)
    assert spec.field("model").select_question(["Kestrova"]).instructions == (
        "Which of these is the model name?"
    )


def test_the_document_gate_asks_for_a_manufacturers_car_variant_with_price_and_specs() -> None:
    assert SchemaSpec.from_model(VehicleSpec).document_gate_question() == Noul(
        instructions=(
            "Does this document include a new car variant a manufacturer offers, with its "
            "price and specifications? It may include several."
        )
    )

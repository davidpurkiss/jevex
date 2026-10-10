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

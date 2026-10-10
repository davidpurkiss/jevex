"""The schemas the test site's ground truth is written in (and eval extracts with)."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from jevex.schema import Field

FuelType = Literal["petrol", "diesel", "hybrid", "phev", "ev"]


class VehicleSpec(BaseModel):
    """A manufacturer's technical specification for one vehicle variant."""

    make: str = Field(description="Manufacturer (make) name")
    model: str = Field(description="Model name")
    trim: str = Field(description="Trim or variant name, e.g. SE L")
    fuel_type: FuelType = Field(description="Fuel or powertrain type")
    engine_size_cc: int | None = Field(description="Engine displacement", unit="cc")
    power_kw: float = Field(description="Maximum power output", unit="kW")
    zero_to_62_s: float = Field(description="0-62 mph acceleration time", unit="s")
    top_speed_mph: int = Field(description="Top speed", unit="mph")
    co2_g_km: int | None = Field(description="CO2 emissions", unit="g/km")
    # Not "On-the-road price": Jev reads that literally, and a "Price from" or "Price" row,
    # which the truth counts, passed the component gate at p 0.25-0.48 (#297).
    price_gbp: Decimal = Field(description="Price", unit="GBP")
    seats: int = Field(description="Number of seats")
    automatic: bool = Field(description="has an automatic gearbox")


# The docstring is the document gate's question, so it names what a listing has and a spec
# page that quotes a price doesn't: "A used car offered for sale." gave the kv and prose spec
# pages p up to 0.80 (#261, benchmarks/gate_wording/listing_docstring.py).
class Listing(BaseModel):
    """A used car advertised for sale, with its mileage and year of registration."""

    make: str = Field(description="Manufacturer (make) name")
    model: str = Field(description="Model name")
    year: int = Field(description="Year of first registration")
    mileage_miles: int = Field(description="Mileage", unit="miles")
    price_gbp: Decimal = Field(description="Asking price", unit="GBP")
    fuel_type: FuelType = Field(description="Fuel or powertrain type")
    colour: str = Field(description="Exterior colour")

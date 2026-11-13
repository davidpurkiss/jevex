"""The schemas the test site's ground truth is written in (and eval extracts with)."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from jevex.schema import Field, Questions, SchemaConfig

FuelType = Literal["petrol", "diesel", "hybrid", "phev", "ev"]


# The docstring is the document gate's question. Real price lists and press releases are
# as much this record's source as spec sheets, and "a manufacturer's technical
# specification" rejected them (p 0.23-0.42, #324); this wording passed every real and
# test-site spec page (p 0.60 and up) and kept listings and off-topic pages below 0.41
# (benchmarks/gate_wording/vehicle_docstring.py).
class VehicleSpec(BaseModel):
    """A new car variant a manufacturer offers, with its price and specifications."""

    # Named after what one record is. As "vehicle spec" (the class name), MultiEntity's
    # boundary Noul read a spec table's row labels and a price list's equipment headings as
    # specs of their own: on the spec-sheets corpus it accepted 1,355 labels naming no
    # labelled trim, against 257 as "vehicle variant" (#322).
    __jevex__ = SchemaConfig(entity_name="vehicle variant")

    # A price list also names the makers of its accessories (a dash camera's, a sound
    # system's). Asked "Which of these is the manufacturer (make) name?", Jev took those
    # too; asked for the vehicle's, it turned down 32 of 84 sampled picks that weren't a
    # truth make and kept 308 of 310 that were (#326).
    make: str = Field(
        description="Manufacturer (make) name",
        questions=Questions(select="Which of these is the vehicle's manufacturer (make)?"),
    )
    # A price list names a variant in one label ("Born V1 59 kWh 230PS", "Premium 1.2 79PS
    # 5MT", "GR Sport AWD"). Asked "Which of these is the model name?" or "... the trim or
    # variant name, e.g. SE L?", Jev took the whole label as either, and a "Model Code" as
    # the model. Told what to leave out, it takes the part: of 250 sampled picks on real
    # price lists that weren't a truth trim, 84 became one and 30 "none", and 4 of 250 that
    # were one stopped being one; for the model, 32 and 95 of 250 (every model code among
    # them), and 1 lost (#326).
    model: str = Field(
        description="Model name",
        questions=Questions(
            select=(
                "Which of these is the name of the model alone, not its code, without the make, "
                "trim level or engine?"
            )
        ),
    )
    trim: str = Field(
        description="Trim or variant name, e.g. SE L",
        questions=Questions(
            select=(
                "Which of these is the trim level alone, e.g. SE L, without the engine, power "
                "or body style?"
            )
        ),
    )
    fuel_type: FuelType = Field(description="Fuel or powertrain type")
    # A variant's name carries its engine's nominal size ("1.0L EcoBoost", "1.2 79PS"), which
    # isn't its displacement (999 cc, 1,197 cc): the truth gives one only where the page
    # does. Asked "Which of these is the engine displacement (cc)?", Jev took the litre
    # figure in 263 of 446 real statements; asked this, in none, and it kept all 54
    # statements giving a truth displacement (#326).
    engine_size_cc: int | None = Field(
        description="Engine displacement",
        unit="cc",
        questions=Questions(
            select="Which of these is the engine displacement in cc, not a size in litres?"
        ),
    )
    power_kw: float = Field(description="Maximum power output", unit="kW")
    zero_to_62_s: float = Field(description="0-62 mph acceleration time", unit="s")
    top_speed_mph: int = Field(description="Top speed", unit="mph")
    co2_g_km: int | None = Field(description="CO2 emissions", unit="g/km")
    # The description isn't "On-the-road price": Jev reads that literally, and a "Price
    # from" or "Price" row, which the truth counts, passed the component gate at p
    # 0.25-0.48 (#297). The select question does name it: a price list's row gives several
    # money figures (basic, VAT, total retail, on the road, on the road less a grant), and
    # the truth's is the on-the-road one before the grant. Over 174 such rows of real price
    # lists, "Which of these is the price a buyer pays for the vehicle?" took it in 114
    # (elsewhere the total retail or basic price) and this wording in 172, and of 286
    # single statements giving a truth price ("Price", "Price from" rows among them) it
    # still took 284, while turning down 601 other figures (VED, delivery, finance) the
    # first wording took (#326).
    price_gbp: Decimal = Field(
        description="Price",
        unit="GBP",
        questions=Questions(
            select=(
                "Which of these is the vehicle's recommended on-the-road price, before any grant?"
            )
        ),
    )
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

"""The phrasing bank: varied wordings for the same fact (spec: *Synthetic test site*).

Every family draws from it, so one fact reads differently from page to page: a sentence
template from :data:`SENTENCES`, a row or tile label from :data:`LABELS`, a value in one
of several units or formats (:func:`cell`), and a used-car fact from :data:`LISTING_FACTS`.
Each choice takes the page's ``random.Random``, so a seed always picks the same words.

Power (whole kW in the truth) may be shown as rounded PS or bhp, within about 0.37 kW of
the truth, and top speed (whole mph) as rounded km/h, within about 0.31 mph. Everything
else is shown exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random
    from collections.abc import Callable
    from decimal import Decimal

    from jevex.testsite.schemas import Listing, VehicleSpec

PS_PER_KW = 1 / 0.73549875
BHP_PER_KW = 1 / 0.745699872
KMH_PER_MPH = 1.609344
FUEL_WORDS = {
    "petrol": ("Petrol", "petrol"),
    "diesel": ("Diesel", "diesel"),
    "hybrid": ("Hybrid", "self-charging hybrid"),
    "phev": ("Plug-in hybrid", "plug-in hybrid"),
    "ev": ("Electric", "fully electric"),
}
NUMBER_WORDS = {4: "four", 5: "five", 7: "seven"}


def power(rng: random.Random, kw: float) -> str:
    """Power in kW, or rounded to whole PS or bhp."""
    unit = rng.choice(("kW", "PS", "bhp"))
    if unit == "kW":
        return f"{kw:g} kW"
    return f"{round(kw * (PS_PER_KW if unit == 'PS' else BHP_PER_KW))}{rng.choice(('', ' '))}{unit}"


def price(rng: random.Random, gbp: Decimal) -> str:
    amount = f"{int(gbp):,}"
    return rng.choice((f"£{amount}", f"£{amount}", f"{amount} GBP"))


def engine(rng: random.Random, cc: int | None) -> str:
    if cc is None:
        return "None (electric)"
    return rng.choice((f"{cc:,} cc", f"{cc}cc", f"{cc} cc"))


def top_speed(rng: random.Random, mph: int) -> str:
    """Top speed in mph, or (three times in ten) rounded to whole km/h."""
    if rng.random() < 0.3:
        return f"{round(mph * KMH_PER_MPH)} km/h"
    return f"{mph} mph"


def seats(rng: random.Random, n: int) -> str:
    """A seat count as a digit or, sometimes, a word."""
    return NUMBER_WORDS.get(n, str(n)) if rng.random() < 0.3 else str(n)


SENTENCES: dict[str, tuple[Callable[[random.Random, VehicleSpec], str], ...]] = {
    "zero_to_62_s": (
        lambda r, v: f"It reaches 62 mph from rest in {v.zero_to_62_s} seconds.",
        lambda r, v: f"0-62 mph takes {v.zero_to_62_s} s.",
        lambda r, v: f"The sprint from 0–62mph is over in {v.zero_to_62_s}s.",
        lambda r, v: f"Accelerating from 0 to 62 mph takes just {v.zero_to_62_s} seconds.",
        lambda r, v: f"Give it {v.zero_to_62_s} seconds and it's doing 62 mph.",
    ),
    "power_kw": (
        lambda r, v: f"Power comes in at {power(r, v.power_kw)}.",
        lambda r, v: f"It develops {power(r, v.power_kw)} of maximum power.",
        lambda r, v: f"Peak output is {power(r, v.power_kw)}.",
        lambda r, v: f"Under your right foot sits {power(r, v.power_kw)}.",
    ),
    "top_speed_mph": (
        lambda r, v: f"Top speed is {top_speed(r, v.top_speed_mph)}.",
        lambda r, v: f"It will run on to a maximum of {top_speed(r, v.top_speed_mph)}.",
        lambda r, v: f"Flat out, it tops out at {top_speed(r, v.top_speed_mph)}.",
    ),
    "price_gbp": (
        lambda r, v: f"On-the-road prices start at {price(r, v.price_gbp)}.",
        lambda r, v: f"The {v.trim} costs {price(r, v.price_gbp)} on the road.",
        lambda r, v: f"Expect to pay {price(r, v.price_gbp)} for this version.",
    ),
    "seats": (
        lambda r, v: f"There's room for {seats(r, v.seats)} people.",
        lambda r, v: f"It seats {seats(r, v.seats)}.",
        lambda r, v: f"It's a {v.seats}-seater.",
    ),
    "automatic": (
        lambda r, v: (
            "An automatic gearbox is standard."
            if v.automatic
            else "It comes with a six-speed manual gearbox."
        ),
        lambda r, v: f"Transmission is {'automatic' if v.automatic else 'manual'}.",
        lambda r, v: (
            "It changes gear for you with an automatic transmission."
            if v.automatic
            else "You change gear yourself: there's a manual stick shift."
        ),
    ),
    "fuel_type": (
        lambda r, v: f"It's a {FUEL_WORDS[v.fuel_type][1]} model.",
        lambda r, v: f"The powertrain is {FUEL_WORDS[v.fuel_type][1]}.",
        lambda r, v: f"This version is {FUEL_WORDS[v.fuel_type][1]}.",
    ),
    "engine_size_cc": (
        lambda r, v: (
            f"The engine displaces {engine(r, v.engine_size_cc)}."
            if v.engine_size_cc
            else "There's no combustion engine."
        ),
        lambda r, v: (
            f"Its engine has a capacity of {engine(r, v.engine_size_cc)}."
            if v.engine_size_cc
            else "As an EV, it has no engine displacement to quote."
        ),
        lambda r, v: (
            f"Displacement is {engine(r, v.engine_size_cc)}."
            if v.engine_size_cc
            else "Engine size: None (electric)."
        ),
    ),
    "co2_g_km": (
        lambda r, v: (
            f"CO2 emissions are {v.co2_g_km} g/km."
            if v.co2_g_km
            else "Tailpipe CO2 emissions are 0 g/km."
        ),
        lambda r, v: f"It emits {v.co2_g_km} g/km of CO2.",
        lambda r, v: f"CO2 output is rated at {v.co2_g_km}g/km.",
    ),
}
"""For each ``VehicleSpec`` field, sentences stating it (make, model and trim go in titles)."""

LABELS: dict[str, tuple[str, ...]] = {
    "fuel_type": ("Fuel type", "Powertrain", "Fuel", "Energy source"),
    "engine_size_cc": ("Engine size", "Displacement", "Engine capacity"),
    "power_kw": ("Maximum power", "Power output", "Power", "Peak power"),
    "zero_to_62_s": ("0-62 mph (s)", "Acceleration 0–62mph", "0-62mph"),
    "top_speed_mph": ("Top speed", "Maximum speed", "Max speed"),
    "co2_g_km": ("CO2 emissions (g/km)", "CO2", "CO2 (g/km)"),
    "price_gbp": ("OTR price", "Price from", "Price", "On-the-road price"),
    "seats": ("Seats", "Seating capacity", "Seating"),
    "automatic": ("Transmission", "Gearbox"),
}
"""For each ``VehicleSpec`` field, labels for a table row, key/value pair or tile."""


def cell(rng: random.Random, name: str, v: VehicleSpec) -> str:
    """The value text for a table cell, key/value pair or tile."""
    match name:
        case "fuel_type":
            return FUEL_WORDS[v.fuel_type][0]
        case "engine_size_cc":
            return engine(rng, v.engine_size_cc)
        case "power_kw":
            return power(rng, v.power_kw)
        case "zero_to_62_s":
            return f"{v.zero_to_62_s}" if rng.random() < 0.5 else f"{v.zero_to_62_s} s"
        case "top_speed_mph":
            return top_speed(rng, v.top_speed_mph)
        case "co2_g_km":
            return str(v.co2_g_km)
        case "price_gbp":
            return price(rng, v.price_gbp)
        case "seats":
            return str(v.seats)
        case "automatic":
            return "Automatic" if v.automatic else "Manual"
        case _:
            raise KeyError(name)


LISTING_FACTS: dict[str, tuple[Callable[[Listing], str], ...]] = {
    "price_gbp": (
        lambda item: f"£{int(item.price_gbp):,}",
        lambda item: f"Price: £{int(item.price_gbp):,}",
        lambda item: f"Yours for £{int(item.price_gbp):,}",
    ),
    "mileage_miles": (
        lambda item: f"{item.mileage_miles:,} miles",
        lambda item: f"Mileage: {item.mileage_miles:,} mi",
        lambda item: f"{item.mileage_miles:,} miles on the clock",
    ),
    "fuel_type": (
        lambda item: FUEL_WORDS[item.fuel_type][0],
        lambda item: f"Fuel: {FUEL_WORDS[item.fuel_type][0]}",
    ),
    "colour": (
        lambda item: item.colour,
        lambda item: f"Colour: {item.colour}",
        lambda item: f"Finished in {item.colour}",
    ),
    "year": (
        lambda item: f"Registered {item.year}",
        lambda item: f"{item.year} reg",
        lambda item: f"First registered in {item.year}",
    ),
}
"""For each used-car fact (make and model go in the card title), short wordings of it."""


def listing_facts(rng: random.Random, item: Listing) -> list[str]:
    """One wording of each fact in :data:`LISTING_FACTS`, in its order."""
    return [rng.choice(wordings)(item) for wordings in LISTING_FACTS.values()]

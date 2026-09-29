"""A seeded, fictional car dataset: makes, models, variants and used-car listings.

Everything comes from one ``random.Random(seed)``, so a seed always gives the same data.
Values are realistic but invented; no real manufacturer or model is named.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from jevex.testsite.schemas import FuelType, Listing, VehicleSpec

MAKES = ("Aurel", "Brantis", "Corvane", "Delmaro", "Esquel", "Fenwick", "Galvor", "Halden")
MODEL_NAMES = (
    "Vento",
    "Lumo",
    "Strada",
    "Orbis",
    "Kestrel",
    "Tamsin",
    "Quill",
    "Ember",
    "Solace",
    "Rivet",
    "Marlow",
    "Nimbus",
    "Parhelia",
    "Sable",
    "Tessa",
    "Wren",
)
TRIMS = ("S", "SE", "SE L", "Sport", "GT", "Vision", "Edition", "Signature")
COLOURS = (
    "Moonstone Grey",
    "Pure White",
    "Deep Black",
    "Racing Red",
    "Atlantic Blue",
    "Forest Green",
    "Silver Frost",
    "Copper Orange",
)


@dataclass(frozen=True)
class Model:
    make: str
    name: str
    variants: tuple[VehicleSpec, ...]

    @property
    def slug(self) -> str:
        return f"{self.make}-{self.name}".lower()


@dataclass(frozen=True)
class Dataset:
    seed: int
    models: tuple[Model, ...]
    listings: tuple[Listing, ...]

    @property
    def variants(self) -> list[VehicleSpec]:
        return [v for m in self.models for v in m.variants]


def _variant(
    rng: random.Random, make: str, model: str, trim: str, fuel: FuelType, step: int
) -> VehicleSpec:
    electric = fuel == "ev"
    power = rng.randint(70, 120) + step * rng.randint(15, 35) + (40 if electric else 0)
    zero_to_62 = round(max(3.2, 12.5 - power / 22 + rng.uniform(-0.4, 0.4)), 1)
    return VehicleSpec(
        make=make,
        model=model,
        trim=trim,
        fuel_type=fuel,
        engine_size_cc=None if electric else rng.choice((999, 1197, 1395, 1498, 1598, 1968, 1984)),
        power_kw=float(power),
        zero_to_62_s=zero_to_62,
        top_speed_mph=int(100 + power / 2.2 + rng.randint(-4, 4)),
        co2_g_km=None
        if electric
        else rng.randint(95, 175) - (40 if fuel in ("hybrid", "phev") else 0),
        price_gbp=Decimal(
            rng.randint(17, 30) * 1000 + step * rng.randint(2, 5) * 1000 + rng.choice((0, 495, 995))
        ),
        seats=rng.choice((5, 5, 5, 7, 4)),
        automatic=electric or rng.random() < 0.6,
    )


def generate(seed: int = 42, *, n_models: int = 16, n_listings: int = 72) -> Dataset:
    """The dataset for ``seed``. The same seed always gives identical data."""
    rng = random.Random(seed)
    makes = list(MAKES)
    rng.shuffle(makes)
    names = list(MODEL_NAMES)
    rng.shuffle(names)
    models: list[Model] = []
    for i in range(n_models):
        make, name = makes[i % len(makes)], names[i % len(names)]
        fuel: FuelType = rng.choice(("petrol", "petrol", "diesel", "hybrid", "phev", "ev"))
        trims = sorted(rng.sample(TRIMS, rng.randint(2, 4)), key=TRIMS.index)
        variants = tuple(
            _variant(rng, make, name, trim, fuel, step) for step, trim in enumerate(trims)
        )
        models.append(Model(make=make, name=name, variants=variants))

    listings: list[Listing] = []
    for _ in range(n_listings):
        model = rng.choice(models)
        variant = rng.choice(model.variants)
        year = rng.randint(2016, 2025)
        age = 2026 - year
        listings.append(
            Listing(
                make=model.make,
                model=model.name,
                year=year,
                mileage_miles=max(500, int(age * rng.randint(4000, 11000) / 10) * 10),
                price_gbp=Decimal(
                    max(3000, int(variant.price_gbp) - age * rng.randint(1200, 2600)) // 5 * 5
                ),
                fuel_type=variant.fuel_type,
                colour=rng.choice(COLOURS),
            )
        )
    return Dataset(seed=seed, models=tuple(models), listings=tuple(listings))

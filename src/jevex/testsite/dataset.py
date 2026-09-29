"""A seeded, fictional car dataset: makes, models, variants and used-car listings.

Everything comes from one ``random.Random(seed)``, so a seed always gives the same data.
Values are realistic but invented; no real manufacturer or model is named.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from jevex.testsite.schemas import FuelType, Listing, VehicleSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

MAKES = ("Aurel", "Brantis", "Corvane", "Delmaro", "Esquel", "Farlan", "Galvor", "Halden")
MODEL_NAMES = (
    "Tovani",
    "Lumo",
    "Quorra",
    "Orbisa",
    "Kestrova",
    "Tamsin",
    "Quill",
    "Emberly",
    "Solace",
    "Rivette",
    "Marlow",
    "Brixa",
    "Parhelia",
    "Selvyn",
    "Tessaro",
    "Wrenna",
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
    """One fictional model: its make, name and variants (one per trim, cheapest first)."""

    make: str
    name: str
    variants: tuple[VehicleSpec, ...]

    @property
    def slug(self) -> str:
        return f"{self.make}-{self.name}".lower()


@dataclass(frozen=True)
class Dataset:
    """Everything the site renders: models with their variants, and used-car listings."""

    seed: int
    models: tuple[Model, ...]
    listings: tuple[Listing, ...]

    @property
    def variants(self) -> list[VehicleSpec]:
        return [v for m in self.models for v in m.variants]


@dataclass(frozen=True)
class _Base:
    """What a model's trims share; each trim steps up from it."""

    fuel: FuelType
    power: int
    price: int
    seats: int
    cc: int | None
    co2: int


def _base(rng: random.Random) -> _Base:
    fuel: FuelType = rng.choice(("petrol", "petrol", "diesel", "hybrid", "phev", "ev"))
    electric = fuel == "ev"
    return _Base(
        fuel=fuel,
        power=rng.randint(70, 110) + (40 if electric else 0),
        price=rng.randint(17, 28) * 1000 + (6000 if fuel in ("phev", "ev") else 0),
        seats=rng.choice((5, 5, 5, 7, 4)),
        cc=None if electric else rng.choice((999, 1197, 1395, 1498, 1598, 1968, 1984)),
        co2=0 if electric else rng.randint(110, 150) - (45 if fuel in ("hybrid", "phev") else 0),
    )


def _variant(
    rng: random.Random, make: str, model: str, trim: str, base: _Base, power: int, price: int
) -> VehicleSpec:
    """One trim at the given power and price; seats, fuel and engine are the model's."""
    electric = base.fuel == "ev"
    top_speed = min(155, int(100 + power / 3) if electric else int(95 + power / 2.1))
    return VehicleSpec(
        make=make,
        model=model,
        trim=trim,
        fuel_type=base.fuel,
        engine_size_cc=base.cc,
        power_kw=float(power),
        zero_to_62_s=round(max(3.2, 12.5 - power / 22 + rng.uniform(-0.3, 0.3)), 1),
        top_speed_mph=top_speed,
        co2_g_km=0 if electric else base.co2 + (power - base.power) // 10,
        price_gbp=Decimal(price),
        seats=base.seats,
        # Hybrids, plug-ins and EVs are always automatic.
        automatic=base.fuel in ("hybrid", "phev", "ev") or rng.random() < 0.5,
    )


def _variants(
    rng: random.Random, make: str, model: str, trims: Sequence[str]
) -> tuple[VehicleSpec, ...]:
    """A model's trims, cheapest first: each one more powerful and dearer than the last."""
    base = _base(rng)
    power, price = base.power, base.price
    out: list[VehicleSpec] = []
    for trim in trims:
        out.append(_variant(rng, make, model, trim, base, power, price + rng.choice((0, 495, 995))))
        power += rng.randint(12, 30)
        price += rng.randint(2, 4) * 1000
    return tuple(out)


def generate(seed: int = 42, *, n_models: int = 16, n_listings: int = 72) -> Dataset:
    """The dataset for ``seed``. The same seed always gives identical data.

    ``n_models`` is at most the number of model names, so every model (and page path)
    is unique.
    """
    if not 1 <= n_models <= len(MODEL_NAMES):
        raise ValueError(f"n_models must be between 1 and {len(MODEL_NAMES)}, not {n_models}")
    if n_listings < 0:
        raise ValueError(f"n_listings can't be negative, not {n_listings}")
    rng = random.Random(seed)
    makes = list(MAKES)
    rng.shuffle(makes)
    names = list(MODEL_NAMES)
    rng.shuffle(names)
    models: list[Model] = []
    for i in range(n_models):
        make, name = makes[i % len(makes)], names[i]
        trims = sorted(rng.sample(TRIMS, rng.randint(2, 4)), key=TRIMS.index)
        variants = _variants(rng, make, name, trims)
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

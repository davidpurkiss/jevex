"""Render the dataset as a static site, with ground truth for every page.

Template families (the ``family`` in the ground truth):

- ``table``: one page per model; a spec table with a column per trim (multi-entity).
- ``prose``: one page per variant; sentences drawn from a phrasing bank. Some pages also
  embed schema.org ``Car`` JSON-LD.
- ``kv``: one page per model; a section per trim with ``<dl>`` key/value pairs.
- ``grid``: used-car listing grids, one card per listing (multi-entity).
- ``listing``: one page per used-car listing.

Each page gets its own ``random.Random(f"{seed}:{path}")``, so pages don't change when
others are added. Pages carry realistic boilerplate (nav, cookie banner, footer) for the
cleaner. Displayed values are rounded as a real page would round them (150 PS for
110.3 kW), so eval compares numbers within a tolerance.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from jevex.testsite.dataset import Dataset, Model
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


@dataclass
class Page:
    path: str
    family: str
    schema: str
    html: str
    records: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    json_ld: bool = False

    def truth(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "family": self.family,
            "schema": self.schema,
            "json_ld": self.json_ld,
            "records": self.records,
        }


# --- phrasing bank ---------------------------------------------------------------------


def _power(rng: random.Random, kw: float) -> str:
    unit = rng.choice(("kW", "PS", "bhp"))
    if unit == "kW":
        return f"{kw:g} kW"
    return f"{round(kw * (PS_PER_KW if unit == 'PS' else BHP_PER_KW))}{rng.choice(('', ' '))}{unit}"


def _price(rng: random.Random, gbp: object) -> str:
    amount = f"{int(str(gbp)):,}"
    return rng.choice((f"£{amount}", f"£{amount}", f"{amount} GBP"))


def _engine(rng: random.Random, cc: int | None) -> str:
    if cc is None:
        return "None (electric)"
    return rng.choice((f"{cc:,} cc", f"{cc}cc", f"{cc} cc"))


def _top_speed(rng: random.Random, mph: int) -> str:
    if rng.random() < 0.3:
        return f"{round(mph * KMH_PER_MPH)} km/h"
    return f"{mph} mph"


SENTENCES: dict[str, tuple[Callable[[random.Random, VehicleSpec], str], ...]] = {
    "zero_to_62_s": (
        lambda r, v: f"It reaches 62 mph from rest in {v.zero_to_62_s} seconds.",
        lambda r, v: f"0-62 mph takes {v.zero_to_62_s} s.",
        lambda r, v: f"The sprint from 0–62mph is over in {v.zero_to_62_s}s.",
        lambda r, v: f"Accelerating from 0 to 62 mph takes just {v.zero_to_62_s} seconds.",
    ),
    "power_kw": (
        lambda r, v: f"Power comes in at {_power(r, v.power_kw)}.",
        lambda r, v: f"It develops {_power(r, v.power_kw)} of maximum power.",
        lambda r, v: f"Peak output is {_power(r, v.power_kw)}.",
    ),
    "top_speed_mph": (
        lambda r, v: f"Top speed is {_top_speed(r, v.top_speed_mph)}.",
        lambda r, v: f"It will run on to a maximum of {_top_speed(r, v.top_speed_mph)}.",
    ),
    "price_gbp": (
        lambda r, v: f"On-the-road prices start at {_price(r, v.price_gbp)}.",
        lambda r, v: f"The {v.trim} costs {_price(r, v.price_gbp)} on the road.",
    ),
    "seats": (
        lambda r, v: f"There's room for {v.seats} people.",
        lambda r, v: f"It seats {v.seats}.",
    ),
    "automatic": (
        lambda r, v: (
            "An automatic gearbox is standard."
            if v.automatic
            else "It comes with a six-speed manual gearbox."
        ),
        lambda r, v: f"Transmission is {'automatic' if v.automatic else 'manual'}.",
    ),
    "fuel_type": (
        lambda r, v: f"It's a {FUEL_WORDS[v.fuel_type][1]} model.",
        lambda r, v: f"The powertrain is {FUEL_WORDS[v.fuel_type][1]}.",
    ),
    "engine_size_cc": (
        lambda r, v: (
            f"The engine displaces {_engine(r, v.engine_size_cc)}."
            if v.engine_size_cc
            else "There's no combustion engine."
        ),
    ),
    "co2_g_km": (
        lambda r, v: (
            f"CO2 emissions are {v.co2_g_km} g/km."
            if v.co2_g_km
            else "It produces zero tailpipe emissions."
        ),
    ),
}

LABELS: dict[str, tuple[str, ...]] = {
    "fuel_type": ("Fuel type", "Powertrain", "Fuel"),
    "engine_size_cc": ("Engine size", "Displacement", "Engine capacity"),
    "power_kw": ("Maximum power", "Power output", "Power"),
    "zero_to_62_s": ("0-62 mph (s)", "Acceleration 0–62mph", "0-62mph"),
    "top_speed_mph": ("Top speed", "Maximum speed"),
    "co2_g_km": ("CO2 emissions (g/km)", "CO2"),
    "price_gbp": ("OTR price", "Price from", "Price"),
    "seats": ("Seats", "Seating capacity"),
    "automatic": ("Transmission", "Gearbox"),
}


def _cell(rng: random.Random, name: str, v: VehicleSpec) -> str:
    """The value text for a table cell or key/value pair."""
    match name:
        case "fuel_type":
            return FUEL_WORDS[v.fuel_type][0]
        case "engine_size_cc":
            return _engine(rng, v.engine_size_cc)
        case "power_kw":
            return _power(rng, v.power_kw)
        case "zero_to_62_s":
            return f"{v.zero_to_62_s}" if rng.random() < 0.5 else f"{v.zero_to_62_s} s"
        case "top_speed_mph":
            return _top_speed(rng, v.top_speed_mph)
        case "co2_g_km":
            return f"{v.co2_g_km}" if v.co2_g_km else "0"
        case "price_gbp":
            return _price(rng, v.price_gbp)
        case "seats":
            return str(v.seats)
        case "automatic":
            return "Automatic" if v.automatic else "Manual"
        case _:
            raise KeyError(name)


# --- page chrome -----------------------------------------------------------------------


def _layout(title: str, body: str, *, head: str = "") -> str:
    return f"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
{head}
</head>
<body>
<div class="cookie-banner" id="cookie-consent">We use cookies to improve your experience. <button>Accept all</button></div>
<nav class="site-nav"><a href="/">Home</a> <a href="/models.html">Models</a> <a href="/used/page-1.html">Used cars</a> <a href="/contact.html">Contact</a></nav>
<main>
{body}
</main>
<footer class="site-footer">© 2026 Testsite Motors Ltd. All rights reserved. <a href="/privacy.html">Privacy</a></footer>
<script>window.__analytics = {{"page": "{escape(title)}"}};</script>
</body>
</html>
"""


def _truth(v: VehicleSpec, entity: str) -> dict[str, Any]:
    return {"entity": entity, "values": v.model_dump(mode="json")}


def _json_ld(v: VehicleSpec) -> str:
    data: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "Car",
        "brand": {"@type": "Brand", "name": v.make},
        "model": v.model,
        "vehicleConfiguration": v.trim,
        "fuelType": FUEL_WORDS[v.fuel_type][0],
        "vehicleEngine": {
            "@type": "EngineSpecification",
            "enginePower": {"@type": "QuantitativeValue", "value": v.power_kw, "unitCode": "KWT"},
        },
        "accelerationTime": {
            "@type": "QuantitativeValue",
            "value": v.zero_to_62_s,
            "unitCode": "SEC",
        },
        "speed": {"@type": "QuantitativeValue", "value": v.top_speed_mph, "unitCode": "HM"},
        "seatingCapacity": v.seats,
        "vehicleTransmission": "Automatic" if v.automatic else "Manual",
        "offers": {"@type": "Offer", "price": str(v.price_gbp), "priceCurrency": "GBP"},
    }
    if v.engine_size_cc:
        data["vehicleEngine"]["engineDisplacement"] = {
            "@type": "QuantitativeValue",
            "value": v.engine_size_cc,
            "unitCode": "CMQ",
        }
    if v.co2_g_km:
        data["emissionsCO2"] = v.co2_g_km
    return f'<script type="application/ld+json">{json.dumps(data, sort_keys=True)}</script>'


# --- families --------------------------------------------------------------------------

SPEC_ORDER = tuple(LABELS)


def table_page(seed: int, model: Model) -> Page:
    path = f"specs/{model.slug}-table.html"
    rng = random.Random(f"{seed}:{path}")
    head = "".join(f"<th>{escape(v.trim)}</th>" for v in model.variants)
    rows: list[str] = []
    for name in SPEC_ORDER:
        label = rng.choice(LABELS[name])
        cells = "".join(f"<td>{escape(_cell(rng, name, v))}</td>" for v in model.variants)
        rows.append(f"<tr><th>{escape(label)}</th>{cells}</tr>")
    body = f"""<h1>{escape(model.make)} {escape(model.name)}: specifications</h1>
<p>Compare the {escape(model.make)} {escape(model.name)} range below.</p>
<section>
<h2>Specifications</h2>
<table class="specs"><thead><tr><th>Specification</th>{head}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
</section>"""
    return Page(
        path=path,
        family="table",
        schema="VehicleSpec",
        html=_layout(f"{model.make} {model.name} specifications", body),
        records=[_truth(v, v.trim) for v in model.variants],
    )


def prose_page(seed: int, model: Model, v: VehicleSpec) -> Page:
    path = f"specs/{model.slug}-{v.trim.lower().replace(' ', '-')}.html"
    rng = random.Random(f"{seed}:{path}")
    fields = list(SENTENCES)
    rng.shuffle(fields)
    sentences = [rng.choice(SENTENCES[name])(rng, v) for name in fields]
    paragraphs = [" ".join(sentences[i : i + 3]) for i in range(0, len(sentences), 3)]
    with_json_ld = rng.random() < 0.35
    body = f"""<h1>{escape(v.make)} {escape(v.model)} {escape(v.trim)}</h1>
<article>
{"".join(f"<p>{escape(p)}</p>" for p in paragraphs)}
</article>"""
    return Page(
        path=path,
        family="prose",
        schema="VehicleSpec",
        html=_layout(
            f"{v.make} {v.model} {v.trim}", body, head=_json_ld(v) if with_json_ld else ""
        ),
        records=[_truth(v, "document")],
        json_ld=with_json_ld,
    )


def kv_page(seed: int, model: Model) -> Page:
    path = f"specs/{model.slug}-details.html"
    rng = random.Random(f"{seed}:{path}")
    sections: list[str] = []
    for v in model.variants:
        pairs = "".join(
            f"<dt>{escape(rng.choice(LABELS[name]))}</dt><dd>{escape(_cell(rng, name, v))}</dd>"
            for name in SPEC_ORDER
        )
        sections.append(f"<section><h2>{escape(v.trim)}</h2><dl>{pairs}</dl></section>")
    body = f"<h1>{escape(model.make)} {escape(model.name)} trims</h1>\n" + "\n".join(sections)
    return Page(
        path=path,
        family="kv",
        schema="VehicleSpec",
        html=_layout(f"{model.make} {model.name} trims", body),
        records=[_truth(v, v.trim) for v in model.variants],
    )


def _listing_facts(rng: random.Random, item: Listing) -> list[str]:
    return [
        rng.choice((f"£{int(item.price_gbp):,}", f"Price: £{int(item.price_gbp):,}")),
        rng.choice((f"{item.mileage_miles:,} miles", f"Mileage: {item.mileage_miles:,} mi")),
        FUEL_WORDS[item.fuel_type][0],
        item.colour,
        rng.choice((f"Registered {item.year}", f"{item.year} reg")),
    ]


def grid_page(seed: int, page_no: int, items: list[tuple[int, Listing]]) -> Page:
    path = f"used/page-{page_no}.html"
    rng = random.Random(f"{seed}:{path}")
    cards: list[str] = []
    for index, item in items:
        facts = "".join(f"<li>{escape(f)}</li>" for f in _listing_facts(rng, item))
        cards.append(
            f'<article class="listing" id="listing-{index}">'
            f"<h3>{escape(item.make)} {escape(item.model)}</h3><ul>{facts}</ul>"
            f'<a href="/used/{index}.html">View details</a></article>'
        )
    body = (
        f'<h1>Used cars: page {page_no}</h1>\n<div class="grid">\n' + "\n".join(cards) + "\n</div>"
    )
    return Page(
        path=path,
        family="grid",
        schema="Listing",
        html=_layout(f"Used cars page {page_no}", body),
        records=[
            {"entity": f"listing-{i}", "values": item.model_dump(mode="json")} for i, item in items
        ],
    )


def listing_page(seed: int, index: int, item: Listing) -> Page:
    path = f"used/{index}.html"
    rng = random.Random(f"{seed}:{path}")
    facts = _listing_facts(rng, item)
    body = f"""<h1>{item.year} {escape(item.make)} {escape(item.model)}</h1>
<p>{escape(facts[0])}. {escape(facts[1])}.</p>
<dl><dt>Fuel</dt><dd>{escape(facts[2])}</dd><dt>Colour</dt><dd>{escape(item.colour)}</dd>
<dt>First registered</dt><dd>{item.year}</dd></dl>"""
    return Page(
        path=path,
        family="listing",
        schema="Listing",
        html=_layout(f"{item.year} {item.make} {item.model}", body),
        records=[{"entity": "document", "values": item.model_dump(mode="json")}],
    )


def render(dataset: Dataset, *, grid_size: int = 12) -> list[Page]:
    """Every page of the site, in a stable order."""
    pages: list[Page] = []
    for model in dataset.models:
        pages.append(table_page(dataset.seed, model))
        pages.append(kv_page(dataset.seed, model))
        pages.extend(prose_page(dataset.seed, model, v) for v in model.variants)
    listings = list(enumerate(dataset.listings, start=1))
    for page_no, start in enumerate(range(0, len(listings), grid_size), start=1):
        pages.append(grid_page(dataset.seed, page_no, listings[start : start + grid_size]))
    pages.extend(listing_page(dataset.seed, i, item) for i, item in listings)
    return pages

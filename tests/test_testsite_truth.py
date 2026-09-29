"""Every ground-truth value must be what the page actually shows (per entity).

Parses each page's tables, key/value sections, cards and main text, and checks every
field of every record against the text for *that* entity, allowing only the documented
rounding (PS/bhp for kW, km/h for mph).
"""

import json
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any

import pytest

from jevex.testsite import generate, render
from jevex.testsite.render import BHP_PER_KW, FUEL_WORDS, KMH_PER_MPH, PS_PER_KW, Page


class Parts(HTMLParser):
    """Table rows, <dl> pairs per <section>, <article> texts and the <main> text."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.sections: list[tuple[str, dict[str, str]]] = []
        self.articles: dict[str, str] = {}
        self.main = ""
        self._cell: list[str] | None = None
        self._buf: list[str] | None = None
        self._dt = ""
        self._art: list[str] | None = None
        self._art_id = ""
        self._main: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []
        elif tag in ("h2", "dt", "dd"):
            self._buf = []
        elif tag == "section":
            self.sections.append(("", {}))
        elif tag == "article":
            self._art, self._art_id = [], dict(attrs).get("id") or "article"
        elif tag == "main":
            self._main = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(unescape("".join(self._cell)).strip())
            self._cell = None
        elif tag == "h2" and self._buf is not None and self.sections:
            self.sections[-1] = (unescape("".join(self._buf)), self.sections[-1][1])
            self._buf = None
        elif tag == "dt" and self._buf is not None:
            self._dt, self._buf = unescape("".join(self._buf)), None
        elif tag == "dd" and self._buf is not None:
            if not self.sections:
                self.sections.append(("", {}))
            self.sections[-1][1][self._dt] = unescape("".join(self._buf))
            self._buf = None
        elif tag == "article" and self._art is not None:
            self.articles[self._art_id] = unescape(" ".join(self._art))
            self._art = None
        elif tag == "main" and self._main is not None:
            self.main = unescape(" ".join(self._main))
            self._main = None

    def handle_data(self, data: str) -> None:
        for buf in (self._cell, self._buf, self._art, self._main):
            if buf is not None:
                buf.append(data)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def power_shown(text: str, kw: float) -> bool:
    for m in re.finditer(r"([\d.,]+)\s?(kW|PS|bhp)\b", text):
        n, unit = _num(m.group(1)), m.group(2)
        expected = {"kW": kw, "PS": round(kw * PS_PER_KW), "bhp": round(kw * BHP_PER_KW)}[unit]
        if n == expected:
            return True
    return False


def speed_shown(text: str, mph: int) -> bool:
    for m in re.finditer(r"(?<![\d-])(\d+)\s?(mph|km/h)", text):
        n, unit = int(m.group(1)), m.group(2)
        if (unit == "mph" and n == mph) or (unit == "km/h" and n == round(mph * KMH_PER_MPH)):
            return True
    return False


def spec_problems(v: dict[str, Any], text: str, cells: dict[str, str] | None) -> list[str]:
    bad: list[str] = []
    if not power_shown(text, v["power_kw"]):
        bad.append("power_kw")
    if not speed_shown(text, v["top_speed_mph"]):
        bad.append("top_speed_mph")
    if not re.search(rf"(?<![\d.]){re.escape(str(v['zero_to_62_s']))}(?!\d)", text):
        bad.append("zero_to_62_s")
    if f"{int(v['price_gbp']):,}" not in text:
        bad.append("price_gbp")
    cc = v["engine_size_cc"]
    if cc is None and not re.search(r"None \(electric\)|no combustion engine", text):
        bad.append("engine_size_cc")
    if cc is not None and not re.search(rf"(?<!\d)({cc}|{cc:,})\s?cc", text):
        bad.append("engine_size_cc")
    if cells is not None:
        co2 = [c for label, c in cells.items() if "CO2" in label]
        if co2 != [str(v["co2_g_km"])]:
            bad.append("co2_g_km")
        seats = [c for label, c in cells.items() if label in ("Seats", "Seating capacity")]
        if seats != [str(v["seats"])]:
            bad.append("seats")
        gearbox = [c for label, c in cells.items() if label in ("Transmission", "Gearbox")]
        if gearbox != ["Automatic" if v["automatic"] else "Manual"]:
            bad.append("automatic")
        fuel = [c for label, c in cells.items() if label in ("Fuel type", "Powertrain", "Fuel")]
        if fuel != [FUEL_WORDS[v["fuel_type"]][0]]:
            bad.append("fuel_type")
    else:
        if not re.search(rf"(?<!\d){v['co2_g_km']} g/km", text):
            bad.append("co2_g_km")
        if not re.search(rf"room for {v['seats']} people|seats {v['seats']}\.", text):
            bad.append("seats")
        if ("automatic" in text.lower()) != v["automatic"]:
            bad.append("automatic")
        if FUEL_WORDS[v["fuel_type"]][1] not in text:
            bad.append("fuel_type")
    return bad


def listing_problems(v: dict[str, Any], text: str) -> list[str]:
    bad: list[str] = []
    checks = {
        "price_gbp": f"£{int(v['price_gbp']):,}",
        "mileage_miles": f"{v['mileage_miles']:,}",
        "year": str(v["year"]),
        "colour": v["colour"],
        "fuel_type": FUEL_WORDS[v["fuel_type"]][0],
        "make": v["make"],
        "model": v["model"],
    }
    for name, needle in checks.items():
        if needle not in text:
            bad.append(name)
    return bad


def page_problems(page: Page) -> list[str]:
    parts = Parts()
    parts.feed(page.html)
    problems: list[str] = []
    for record in page.records:
        v, entity = record["values"], record["entity"]
        if page.family == "table":
            header = parts.rows[0]
            col = header.index(v["trim"])
            cells = {row[0]: row[col] for row in parts.rows[1:]}
            bad = spec_problems(v, " ".join(f"{k} {c}" for k, c in cells.items()), cells)
        elif page.family == "kv":
            [cells] = [c for title, c in parts.sections if title == v["trim"]]
            bad = spec_problems(v, " ".join(f"{k} {c}" for k, c in cells.items()), cells)
        elif page.family == "prose":
            bad = spec_problems(v, parts.main, None)
        elif page.family == "grid":
            bad = listing_problems(v, parts.articles[entity])
        else:
            bad = listing_problems(v, parts.main)
        problems.extend(f"{page.path} [{entity}] {name}" for name in bad)
    return problems


@pytest.mark.parametrize("seed", [42, 1, 7, 2024])
def test_every_truth_value_is_shown_on_its_page(seed: int) -> None:
    problems = [p for page in render(generate(seed)) for p in page_problems(page)]
    assert problems == []


def test_the_checker_catches_a_wrong_value() -> None:
    page = next(p for p in render(generate(42)) if p.family == "table")
    page.records[0]["values"]["co2_g_km"] = 999
    assert any("co2_g_km" in p for p in page_problems(page))


def json_ld_problems(page: Page) -> list[str]:
    """Every JSON-LD value must equal the page's single record."""
    blob = page.html.split('<script type="application/ld+json">')[1].split("</script>")[0]
    ld = json.loads(blob)
    v = page.records[0]["values"]
    expected: dict[str, Any] = {
        "brand": v["make"],
        "model": v["model"],
        "vehicleConfiguration": v["trim"],
        "fuelType": FUEL_WORDS[v["fuel_type"]][0],
        "enginePower": v["power_kw"],
        "accelerationTime": v["zero_to_62_s"],
        "speed": v["top_speed_mph"],
        "seatingCapacity": v["seats"],
        "vehicleTransmission": "Automatic" if v["automatic"] else "Manual",
        "price": v["price_gbp"],
        "emissionsCO2": v["co2_g_km"],
        "engineDisplacement": v["engine_size_cc"],
    }
    engine = ld["vehicleEngine"]
    actual: dict[str, Any] = {
        "brand": ld["brand"]["name"],
        "model": ld["model"],
        "vehicleConfiguration": ld["vehicleConfiguration"],
        "fuelType": ld["fuelType"],
        "enginePower": engine["enginePower"]["value"],
        "accelerationTime": ld["accelerationTime"]["value"],
        "speed": ld["speed"]["value"],
        "seatingCapacity": ld["seatingCapacity"],
        "vehicleTransmission": ld["vehicleTransmission"],
        "price": ld["offers"]["price"],
        "emissionsCO2": ld.get("emissionsCO2"),
        "engineDisplacement": engine["engineDisplacement"]["value"]
        if "engineDisplacement" in engine
        else None,
    }
    return [f"{page.path} json-ld {k}" for k in expected if str(actual[k]) != str(expected[k])]


@pytest.mark.parametrize("seed", [42, 1, 7, 2024])
def test_json_ld_matches_the_truth(seed: int) -> None:
    pages = [p for p in render(generate(seed)) if p.json_ld]
    assert pages
    assert [p for page in pages for p in json_ld_problems(page)] == []

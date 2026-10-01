"""Every ground-truth value must be what the page actually shows (per entity).

Parses each page's tables, key/value sections, cards and main text, and checks every
field of every record against the text for *that* entity, allowing only the documented
rounding (PS/bhp for kW, km/h for mph). Spec-sheet PDFs are read from the text their
content stream draws; scans and infographics from the drawing they rasterise, and an OCR
pass checks that the pixels say the same.
"""

import io
import json
import random
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any

import pytest

from jevex.testsite import Dataset, generate, render
from jevex.testsite.dataset import Model
from jevex.testsite.drawing import Drawing
from jevex.testsite.phrasing import (
    BHP_PER_KW,
    FUEL_WORDS,
    KMH_PER_MPH,
    LABELS,
    LISTING_FACTS,
    NUMBER_WORDS,
    PS_PER_KW,
    SENTENCES,
)
from jevex.testsite.render import Page, infographic_drawing, scanned_drawing
from jevex.testsite.schemas import Listing


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

    @classmethod
    def main_text(cls, html: str) -> str:
        parts = cls()
        parts.feed(html)
        return parts.main

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
    if cc is None and not re.search(
        r"None \(electric\)|no combustion engine|no engine displacement", text
    ):
        bad.append("engine_size_cc")
    if cc is not None and not re.search(rf"(?<!\d)({cc}|{cc:,})\s?cc", text):
        bad.append("engine_size_cc")
    if cells is not None:
        co2 = [c for label, c in cells.items() if "CO2" in label]
        if co2 != [str(v["co2_g_km"])]:
            bad.append("co2_g_km")
        seats = [c for label, c in cells.items() if label in LABELS["seats"]]
        if seats != [str(v["seats"])]:
            bad.append("seats")
        gearbox = [c for label, c in cells.items() if label in LABELS["automatic"]]
        if gearbox != ["Automatic" if v["automatic"] else "Manual"]:
            bad.append("automatic")
        fuel = [c for label, c in cells.items() if label in LABELS["fuel_type"]]
        if fuel != [FUEL_WORDS[v["fuel_type"]][0]]:
            bad.append("fuel_type")
    else:
        if not re.search(rf"(?<!\d){v['co2_g_km']}\s?g/km", text):
            bad.append("co2_g_km")
        n = f"({v['seats']}|{NUMBER_WORDS[v['seats']]})"
        if not re.search(rf"room for {n} people|seats {n}\.|{v['seats']}-seater", text):
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


Placed = list[tuple[float, float, str]]
"""Text with the x and y it's drawn at (larger y further down the page)."""


def pdf_texts(content: bytes) -> Placed:
    """The text a vector PDF's content stream draws (``Td`` positions, ``Tj`` strings)."""
    stream = content.split(b"stream\n", 1)[1].split(b"\nendstream", 1)[0]
    shows = re.finditer(
        rb"BT /F\d [\d.]+ Tf [\d. ]+ rg ([\d.]+) ([\d.]+) Td \(((?:\\.|[^\\)])*)\) Tj ET",
        stream,
    )
    return [(float(m[1]), -float(m[2]), pdf_unescape(m[3])) for m in shows]


def pdf_unescape(raw: bytes) -> str:
    out, i = bytearray(), 0
    while i < len(raw):
        if raw[i : i + 1] == b"\\" and raw[i + 1 : i + 2].isdigit():
            out.append(int(raw[i + 1 : i + 4], 8))
            i += 4
        elif raw[i : i + 1] == b"\\":
            out += raw[i + 1 : i + 2]
            i += 2
        else:
            out += raw[i : i + 1]
            i += 1
    return out.decode("cp1252")


def drawn_texts(drawing: Drawing) -> Placed:
    return [(t.x, t.y, t.text) for t in drawing.texts]


def table_columns(texts: Placed, trims: list[str]) -> dict[str, dict[str, str]]:
    """Each trim's ``{row label: cell}``, from a table whose header row ends in ``trims``."""
    rows: dict[float, list[tuple[float, str]]] = {}
    for x, y, text in texts:
        rows.setdefault(y, []).append((x, text))
    ordered = [sorted(rows[y]) for y in sorted(rows)]
    [header] = [r for r in ordered if [t for _, t in r[-len(trims) :]] == trims]
    xs = [x for x, _ in header[-len(trims) :]]
    columns: dict[str, dict[str, str]] = {trim: {} for trim in trims}
    for row in ordered[ordered.index(header) + 1 :]:
        if len(row) == len(trims) + 1 and [x for x, _ in row[1:]] == xs:
            for trim, (_, text) in zip(trims, row[1:], strict=True):
                columns[trim][row[0][1]] = text
    return columns


def tile_pairs(texts: Placed) -> dict[str, str]:
    """An infographic's ``{label: value}``: each label sits 30 pt under its value."""
    at = {(x, y): text for x, y, text in texts}
    return {at[(x, y + 30)]: text for (x, y), text in at.items() if (x, y + 30) in at}


def drawn_problems(page: Page, dataset: Dataset) -> list[str]:
    """Problems with a PDF, scan or infographic: what's drawn must match its truth."""
    for model in dataset.models:
        trims = [v.trim for v in model.variants]
        if page.path == f"brochures/{model.slug}-spec-sheet.pdf":
            return table_problems(page, pdf_texts(page.content), model, trims)
        if page.path == f"scans/{model.slug}-spec-sheet-scan.pdf":
            texts = drawn_texts(scanned_drawing(dataset.seed, model))
            return table_problems(page, texts, model, trims)
        path, drawing, _ = infographic_drawing(dataset.seed, model)
        if page.path == path:
            texts = drawn_texts(drawing)
            [record] = page.records
            v, cells = record["values"], tile_pairs(texts)
            bad = spec_problems(v, " ".join(f"{k} {c}" for k, c in cells.items()), cells)
            if f"{v['make']} {v['model']} {v['trim']}" not in [t for _, _, t in texts]:
                bad.append("title")
            return [f"{page.path} [document] {name}" for name in bad]
    raise AssertionError(f"no model draws {page.path}")


def table_problems(page: Page, texts: Placed, model: Model, trims: list[str]) -> list[str]:
    columns = table_columns(texts, trims)
    problems: list[str] = []
    if f"{model.make} {model.name}" not in [t for _, _, t in texts]:
        problems.append(f"{page.path} title")
    for record in page.records:
        v = record["values"]
        cells = columns[v["trim"]]
        bad = spec_problems(v, " ".join(f"{k} {c}" for k, c in cells.items()), cells)
        problems.extend(f"{page.path} [{v['trim']}] {name}" for name in bad)
    return problems


def page_problems(page: Page, dataset: Dataset) -> list[str]:
    if page.content_type != "text/html":
        return drawn_problems(page, dataset)
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
    dataset = generate(seed)
    pages = render(dataset)
    assert {"pdf", "scanned", "infographic"} <= {p.family for p in pages}
    problems = [p for page in pages for p in page_problems(page, dataset)]
    assert problems == []


@pytest.mark.parametrize("family", ["table", "pdf", "scanned", "infographic"])
def test_the_checker_catches_a_wrong_value(family: str) -> None:
    dataset = generate(42)
    page = next(p for p in render(dataset) if p.family == family)
    page.records[0]["values"]["co2_g_km"] = 999
    assert any("co2_g_km" in p for p in page_problems(page, dataset))


def test_banded_and_plain_spec_sheets_both_occur() -> None:
    sheets = [pdf_texts(p.content) for p in render(generate(42)) if p.family == "pdf"]
    banded = ["Performance" in [t for _, _, t in texts] for texts in sheets]
    assert any(banded)
    assert not all(banded)


def test_ocr_reads_the_truth_off_scans_and_infographics() -> None:
    pdfium = pytest.importorskip("pypdfium2")
    pytest.importorskip("rapidocr")
    from jevex.images import RapidOcrEngine

    engine = RapidOcrEngine()
    pages = render(generate(42, n_models=2, n_listings=0))
    read = 0
    for page in pages:
        if page.family == "scanned":
            out = io.BytesIO()
            pdfium.PdfDocument(page.content)[0].render(scale=150 / 72).to_pil().save(out, "PNG")
            image = out.getvalue()
        elif page.family == "infographic":
            image = page.content
        else:
            continue
        text = " ".join(t.text for t in engine.read(image)).lower()
        for record in page.records:
            v = record["values"]
            assert v["model"].lower() in text
            assert v["trim"].lower() in text
            assert f"{int(v['price_gbp']):,}" in text
        read += 1
    assert read == 4


# --- the phrasing bank -----------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 7])
def test_every_wording_states_its_fact(seed: int) -> None:
    rng = random.Random(seed)
    for v in generate(seed).variants:
        values = v.model_dump(mode="json")
        for name, wordings in SENTENCES.items():
            for wording in wordings:
                sentence = wording(rng, v)
                assert name not in spec_problems(values, sentence, None), sentence


def test_each_fact_has_several_distinct_wordings() -> None:
    variants = generate(42).variants
    ev = next(v for v in variants if v.fuel_type == "ev")
    combustion = next(v for v in variants if v.fuel_type != "ev")
    spec_fields = set(type(ev).model_fields) - {"make", "model", "trim"}
    assert set(SENTENCES) == set(LABELS) == spec_fields
    for name, wordings in SENTENCES.items():
        assert len(wordings) >= 3, name
        for v in (ev, combustion):
            sentences = [w(random.Random(0), v) for w in wordings]
            assert len(set(sentences)) == len(sentences), sentences
    assert all(len(labels) >= 2 for labels in LABELS.values())
    listing_fields = set(Listing.model_fields) - {"make", "model"}
    assert set(LISTING_FACTS) == listing_fields
    assert all(len(wordings) >= 2 for wordings in LISTING_FACTS.values())


def test_the_site_uses_every_wording() -> None:
    """Wordings are only varied if pages actually pick them all."""
    main = " ".join(
        Parts.main_text(p.html) for p in render(generate(42)) if p.content_type == "text/html"
    )
    for opening in (
        "Give it",
        "Under your right foot",
        "Flat out",
        "Expect to pay",
        "-seater",
        "You change gear yourself",
        "This version is",
        "capacity of",
        "It emits",
        "rated at",
        "Yours for",
        "on the clock",
        "Finished in",
        "First registered in",
    ):
        assert opening in main, opening
    assert re.search(r"(seats|room for) (four|five|seven)", main)


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

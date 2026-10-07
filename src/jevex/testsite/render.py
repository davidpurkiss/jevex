"""Render the dataset as a static site, with ground truth for every page.

Template families (the ``family`` in the ground truth):

- ``table``: one page per model; a spec table with a column per trim (multi-entity).
- ``prose``: one page per variant; sentences drawn from the phrasing bank. Some pages also
  embed schema.org ``Car`` JSON-LD.
- ``kv``: one page per model; a section per trim with ``<dl>`` key/value pairs.
- ``grid``: used-car listing grids, one card per listing (multi-entity).
- ``listing``: one page per used-car listing.
- ``pdf``: one spec-sheet PDF per model; a ruled table with a column per trim, sometimes
  with band rows (multi-entity).
- ``scanned``: one per model; a spec sheet printed and scanned, an image-only PDF.
- ``infographic``: one PNG per model, for one of its trims; a tile per spec.

Rasterised pages (``scanned``, ``infographic``) write prices as ``GBP 17,000``, not
``£17,000``, because the font they're drawn in has no pound sign.

The site is en-GB (:data:`SITE_LOCALE`). HTML pages say so in ``<html lang>``; PDFs and
images can't, so their ground truth carries a ``locale`` that eval gives the document, and
the server sends it as ``Content-Language``.

Each page gets its own ``random.Random(f"{seed}:{path}")``, so pages don't change when
others are added. HTML pages carry realistic boilerplate (nav, cookie banner, footer) for
the cleaner. Values are worded from :mod:`jevex.testsite.phrasing`, whose docstring says
which ones are rounded (so eval compares them with a small tolerance).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING, Any

from jevex.testsite.drawing import Drawing, text_width, to_pdf, to_png, to_scanned_pdf
from jevex.testsite.phrasing import FUEL_WORDS, LABELS, SENTENCES, cell, listing_facts

if TYPE_CHECKING:
    from collections.abc import Collection

    from jevex.testsite.dataset import Dataset, Model
    from jevex.testsite.drawing import RGB
    from jevex.testsite.schemas import Listing, VehicleSpec

HTML = "text/html"
PDF = "application/pdf"
PNG = "image/png"
SITE_LOCALE = "en-GB"
"""The site's locale: HTML pages declare it, and PDFs and images carry it as
:attr:`Page.locale`."""
FAMILIES = ("table", "kv", "prose", "grid", "listing", "pdf", "scanned", "infographic")
"""Every template family, in the order the module docstring describes them."""


@dataclass
class Page:
    """One rendered page (an HTML page, a PDF or an image) and its ground truth.

    ``records[].entity`` is ``"document"`` for single-record pages, the trim name on
    ``table``/``kv``/``pdf``/``scanned`` pages, and ``listing-N`` on grids (matching the
    card's ``id``).
    """

    path: str
    family: str
    schema: str
    content: bytes
    content_type: str = HTML
    records: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    json_ld: bool = False
    locale: str | None = None
    """The page's locale where its content can't say it (PDFs and images): eval hands it
    to the document as :attr:`~jevex.Document.locale`. HTML pages declare theirs."""

    @property
    def html(self) -> str:
        """The page's HTML. Raises ``ValueError`` for a PDF or an image."""
        if self.content_type != HTML:
            raise ValueError(f"{self.path} is {self.content_type}, not HTML")
        return self.content.decode()

    def truth(self) -> dict[str, Any]:
        """The page's entry in ``truth.json``."""
        return {
            "path": self.path,
            "family": self.family,
            "schema": self.schema,
            "content_type": self.content_type,
            "json_ld": self.json_ld,
            "records": self.records,
        } | ({"locale": self.locale} if self.locale else {})


# --- page chrome -----------------------------------------------------------------------


def _layout(title: str, body: str, *, head: str = "") -> bytes:
    html = f"""<!doctype html>
<html lang="{SITE_LOCALE}">
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
    return html.encode()


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
    if v.co2_g_km is not None:
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
        cells = "".join(f"<td>{escape(cell(rng, name, v))}</td>" for v in model.variants)
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
        content=_layout(f"{model.make} {model.name} specifications", body),
        records=[_truth(v, v.trim) for v in model.variants],
    )


def prose_page(seed: int, model: Model, v: VehicleSpec) -> Page:
    path = f"specs/{model.slug}-{_trim_slug(v)}.html"
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
        content=_layout(
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
            f"<dt>{escape(rng.choice(LABELS[name]))}</dt><dd>{escape(cell(rng, name, v))}</dd>"
            for name in SPEC_ORDER
        )
        sections.append(f"<section><h2>{escape(v.trim)}</h2><dl>{pairs}</dl></section>")
    body = f"<h1>{escape(model.make)} {escape(model.name)} trims</h1>\n" + "\n".join(sections)
    return Page(
        path=path,
        family="kv",
        schema="VehicleSpec",
        content=_layout(f"{model.make} {model.name} trims", body),
        records=[_truth(v, v.trim) for v in model.variants],
    )


def grid_page(seed: int, page_no: int, items: list[tuple[int, Listing]]) -> Page:
    path = f"used/page-{page_no}.html"
    rng = random.Random(f"{seed}:{path}")
    cards: list[str] = []
    for index, item in items:
        facts = "".join(f"<li>{escape(f)}</li>" for f in listing_facts(rng, item))
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
        content=_layout(f"Used cars page {page_no}", body),
        records=[
            {"entity": f"listing-{i}", "values": item.model_dump(mode="json")} for i, item in items
        ],
    )


def listing_page(seed: int, index: int, item: Listing) -> Page:
    path = f"used/{index}.html"
    rng = random.Random(f"{seed}:{path}")
    facts = listing_facts(rng, item)
    body = f"""<h1>{item.year} {escape(item.make)} {escape(item.model)}</h1>
<p>{escape(facts[0])}. {escape(facts[1])}.</p>
<dl><dt>Fuel</dt><dd>{escape(FUEL_WORDS[item.fuel_type][0])}</dd><dt>Colour</dt><dd>{escape(item.colour)}</dd>
<dt>First registered</dt><dd>{item.year}</dd></dl>"""
    return Page(
        path=path,
        family="listing",
        schema="Listing",
        content=_layout(f"{item.year} {item.make} {item.model}", body),
        records=[{"entity": "document", "values": item.model_dump(mode="json")}],
    )


# --- PDFs and images -------------------------------------------------------------------

A4 = (595.0, 842.0)
MARGIN = 50.0
WHITE: RGB = (1.0, 1.0, 1.0)
GREY: RGB = (0.35, 0.35, 0.38)
PALETTE: tuple[RGB, ...] = (
    (0.11, 0.23, 0.42),
    (0.55, 0.10, 0.12),
    (0.08, 0.36, 0.29),
    (0.22, 0.22, 0.25),
    (0.36, 0.18, 0.47),
)
BACKGROUNDS: tuple[RGB, ...] = ((0.97, 0.96, 0.93), (0.93, 0.95, 0.98), (1.0, 1.0, 1.0))
BANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Engine and transmission", ("fuel_type", "engine_size_cc", "power_kw", "automatic")),
    ("Performance", ("zero_to_62_s", "top_speed_mph", "co2_g_km")),
    ("Practicality and price", ("seats", "price_gbp")),
)
"""Band rows grouping a spec sheet's rows, in order; every field is in one band."""
INTROS = (
    "Specifications for every trim in the {make} {model} range.",
    "All figures are manufacturer data for the {model} line-up.",
    "Compare the {model} trims side by side.",
)
SHEET_SUBTITLES = ("Technical specification", "Specifications and prices", "Range data sheet")
FOOTNOTES = (
    "Figures are subject to change. Testsite Motors Ltd, 2026.",
    "All prices include VAT and first registration. E&OE.",
)
TILE_SUBTITLES = ("At a glance", "The key numbers", "Facts and figures")
CELL_SIZE = 9.0
LABEL_WIDTH = 145.0
ROW_HEIGHT = 22.0


def spec_sheet(rng: random.Random, model: Model, *, plain: bool = False) -> Drawing:
    """A one-page A4 spec sheet: a ruled table with a row per spec and a column per trim."""
    width, height = A4
    d = Drawing(width, height, title=f"{model.make} {model.name} specifications", plain=plain)
    accent = rng.choice(PALETTE)
    d.box(0, 0, width, 84, accent)
    d.text(MARGIN, 44, f"{model.make} {model.name}", 22, bold=True, fill=WHITE)
    d.text(MARGIN, 66, rng.choice(SHEET_SUBTITLES), 11, fill=WHITE)
    d.text(MARGIN, 118, rng.choice(INTROS).format(make=model.make, model=model.name), 10)

    variants = model.variants
    col = (width - 2 * MARGIN - LABEL_WIDTH) / len(variants)
    right = width - MARGIN
    columns = [MARGIN + LABEL_WIDTH + i * col for i in range(len(variants))]
    top = y = 140.0
    d.box(MARGIN, y, right - MARGIN, ROW_HEIGHT, (0.9, 0.9, 0.92))
    corner = rng.choice(("Specification", "", "Trim"))
    _row(d, y, [corner, *(v.trim for v in variants)], col, bold=True)
    y += ROW_HEIGHT
    d.rule(MARGIN, y, right, y, 1.0)
    rows = [(top, y)]  # where the column rules go: every row but the bands
    banded = rng.random() < 0.5
    for band, names in BANDS:
        if banded:
            d.box(MARGIN, y, right - MARGIN, ROW_HEIGHT, (0.96, 0.96, 0.97))
            d.text(MARGIN + 4, y + ROW_HEIGHT - 7, band, CELL_SIZE, bold=True)
            y += ROW_HEIGHT
            d.rule(MARGIN, y, right, y)
        start = y
        for name in names:
            texts = [rng.choice(LABELS[name]), *(cell(rng, name, v) for v in variants)]
            _row(d, y, texts, col)
            y += ROW_HEIGHT
            d.rule(MARGIN, y, right, y)
        rows.append((start, y))
    d.rule(MARGIN, top, right, top)
    d.rule(MARGIN, top, MARGIN, y)
    d.rule(right, top, right, y)
    for x in columns:
        for row_top, row_bottom in rows:
            d.rule(x, row_top, x, row_bottom)
    d.text(MARGIN, height - 40, rng.choice(FOOTNOTES), 8, fill=GREY)
    d.text(right - 50, height - 40, "Page 1 of 1", 8, fill=GREY)
    return d


def _row(d: Drawing, y: float, texts: list[str], col: float, *, bold: bool = False) -> None:
    """One table row: a label, then a cell per trim column."""
    for i, text in enumerate(texts):
        x = MARGIN if i == 0 else MARGIN + LABEL_WIDTH + (i - 1) * col
        room = (LABEL_WIDTH if i == 0 else col) - 8
        if text_width(d.shown(text), CELL_SIZE, bold=bold) > room:
            raise ValueError(f"{text!r} doesn't fit a {room:g} pt column")  # a layout bug
        if text:
            d.text(x + 4, y + ROW_HEIGHT - 7, text, CELL_SIZE, bold=bold)


def infographic(rng: random.Random, v: VehicleSpec) -> Drawing:
    """A landscape "at a glance" graphic for one trim: a coloured tile per spec."""
    width, height = 600.0, 444.0
    d = Drawing(width, height, title=f"{v.make} {v.model} {v.trim}", plain=True)
    d.box(0, 0, width, height, rng.choice(BACKGROUNDS))
    accent = rng.choice(PALETTE)
    d.text(30, 50, f"{v.make} {v.model} {v.trim}", 24, bold=True, fill=accent)
    d.text(30, 74, rng.choice(TILE_SUBTITLES), 12, fill=GREY)
    names = list(LABELS)
    rng.shuffle(names)
    tile_w, tile_h, gap = 172.0, 100.0, 12.0
    for i, name in enumerate(names):
        x, y = 30 + (i % 3) * (tile_w + gap), 96 + (i // 3) * (tile_h + gap)
        d.box(x, y, tile_w, tile_h, accent)
        value = d.shown(cell(rng, name, v))
        size = min(22.0, (tile_w - 28) / text_width(value, 1, bold=True))
        d.text(x + 14, y + 50, value, size, bold=True, fill=WHITE)
        d.text(x + 14, y + 80, rng.choice(LABELS[name]), 11, fill=WHITE)
    d.text(30, height - 6, "Manufacturer data. Testsite Motors Ltd, 2026.", 8, fill=GREY)
    return d


def pdf_page(seed: int, model: Model) -> Page:
    path = f"brochures/{model.slug}-spec-sheet.pdf"
    rng = random.Random(f"{seed}:{path}")
    return Page(
        path=path,
        family="pdf",
        schema="VehicleSpec",
        content=to_pdf(spec_sheet(rng, model)),
        content_type=PDF,
        locale=SITE_LOCALE,
        records=[_truth(v, v.trim) for v in model.variants],
    )


def scanned_drawing(seed: int, model: Model) -> Drawing:
    """The spec sheet a ``scanned`` page shows, before it's printed and scanned."""
    drawing = spec_sheet(random.Random(f"{seed}:{_scanned_path(model)}"), model, plain=True)
    drawing.title = "Scanned document"
    return drawing


def _scanned_path(model: Model) -> str:
    return f"scans/{model.slug}-spec-sheet-scan.pdf"


def scanned_page(seed: int, model: Model) -> Page:
    path = _scanned_path(model)
    return Page(
        path=path,
        family="scanned",
        schema="VehicleSpec",
        content=to_scanned_pdf(scanned_drawing(seed, model), f"{seed}:{path}:scan"),
        content_type=PDF,
        locale=SITE_LOCALE,
        records=[_truth(v, v.trim) for v in model.variants],
    )


def infographic_drawing(seed: int, model: Model) -> tuple[str, Drawing, VehicleSpec]:
    """The path, graphic and trim of a model's ``infographic`` page."""
    v = random.Random(f"{seed}:infographics/{model.slug}").choice(model.variants)
    path = f"infographics/{model.slug}-{_trim_slug(v)}.png"
    return path, infographic(random.Random(f"{seed}:{path}"), v), v


def infographic_page(seed: int, model: Model) -> Page:
    path, drawing, v = infographic_drawing(seed, model)
    return Page(
        path=path,
        family="infographic",
        schema="VehicleSpec",
        content=to_png(drawing),
        content_type=PNG,
        locale=SITE_LOCALE,
        records=[_truth(v, "document")],
    )


def _trim_slug(v: VehicleSpec) -> str:
    return v.trim.lower().replace(" ", "-")


def render(
    dataset: Dataset, *, grid_size: int = 12, families: Collection[str] | None = None
) -> list[Page]:
    """The site's pages (only those of ``families``, if given), in a stable order.

    Raises ``ImportError`` without Pillow (the ``testsite`` extra), which draws the
    ``scanned`` and ``infographic`` pages, and ``ValueError`` for an unknown family.
    """
    wanted = set(FAMILIES if families is None else families)
    if unknown := sorted(wanted - set(FAMILIES)):
        raise ValueError(f"unknown template families {unknown}; they're {', '.join(FAMILIES)}")
    seed = dataset.seed
    pages: list[Page] = []
    for model in dataset.models:
        if "table" in wanted:
            pages.append(table_page(seed, model))
        if "kv" in wanted:
            pages.append(kv_page(seed, model))
        if "prose" in wanted:
            pages.extend(prose_page(seed, model, v) for v in model.variants)
        if "pdf" in wanted:
            pages.append(pdf_page(seed, model))
        if "scanned" in wanted:
            pages.append(scanned_page(seed, model))
        if "infographic" in wanted:
            pages.append(infographic_page(seed, model))
    listings = list(enumerate(dataset.listings, start=1))
    if "grid" in wanted:
        for page_no, start in enumerate(range(0, len(listings), grid_size), start=1):
            pages.append(grid_page(seed, page_no, listings[start : start + grid_size]))
    if "listing" in wanted:
        pages.extend(listing_page(seed, i, item) for i, item in listings)
    return pages

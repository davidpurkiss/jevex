"""Compare document-gate wordings against the live API (#242).

Live and billed (about $0.001 a run): run only with a key, under a spend cap::

    (set -a; . .env; set +a; JEVEX_JEV_MAX_COST_USD=0.10 \\
        uv run python benchmarks/gate_wording/compare.py)

Asks every wording for three schemas about 21 pages: two each of the test site's table, kv,
prose, pdf, grid and listing families (seed 42), the three books.toscrape pages in
``tests/fixtures/books``, and six off-topic pages written here. Each page is read as
``NoulDocumentGate`` reads it and asked all 15 questions in one request. Writes
``benchmarks/gate_wording/results-<date>.json`` and prints the misses and false passes per
wording at the gate's threshold.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from jevex import Document
from jevex.gate import DEFAULT_MAX_CHARS, DEFAULT_THRESHOLD, DefaultTextReader
from jevex.jev import JevClient, Noul, NoulAnswer, TypeSafeBackend
from jevex.testsite import generate, render

ROOT = Path(__file__).resolve().parents[2]

SCHEMAS = {
    "VehicleSpec": "a manufacturer's technical specification for one vehicle variant",
    "Listing": "a used car offered for sale",
    "Book": "a book offered for sale",
}
"""Each schema's docstring as the default question uses it (first sentence, lower-cased).
``Book`` is a control: it should pass only the books pages."""

WORDINGS = {
    "A_current": "Does this document describe {d}?",
    "B_one_or_more": "Does this document contain {d}, for one or more items?",
    "C_information": "Does this document contain information about {d}?",
    "D_or_more": "Does this document describe {d}, or more than one?",
    "E_may_include_several": "Does this document include {d}? It may include several.",
}
"""``A_current`` was the default before #242; ``E_may_include_several`` replaced it."""

EXPECTED = {
    "table": "VehicleSpec",
    "kv": "VehicleSpec",
    "prose": "VehicleSpec",
    "pdf": "VehicleSpec",
    "grid": "Listing",
    "listing": "Listing",
    "books": "Book",
}
"""The schema each group of pages should pass; the off-topic pages should pass none."""


def _html(title: str, body: str) -> bytes:
    head = f"<head><title>{title}</title></head>"
    return f"<html>{head}<body><h1>{title}</h1>{body}</body></html>".encode()


OFF_TOPIC = {
    "review": _html(
        "Driving the Delmaro Kestrova: first impressions",
        "<p>We spent a wet weekend with the Kestrova. It feels quicker than you'd expect, the "
        "ride is settled on motorways, and the boot swallowed a week's luggage. The cabin is "
        "plain but solid.</p><p>Rivals are sharper to drive, but few are as easy to live with. "
        "Verdict: a sensible choice.</p>",
    ),
    "dealer_contact": _html(
        "Contact Westbridge Motors",
        "<p>Visit our showroom at 14 Mill Lane, Westbridge. Open Monday to Saturday, 9am to "
        "6pm.</p><p>Call 01632 960 123 or email sales@example.test. "
        "<a href='/used/'>Browse our used stock</a>.</p>",
    ),
    "news": _html(
        "New car registrations rise 4% in September",
        "<p>Registrations of new cars in the UK rose 4% year on year in September, industry "
        "figures show. Battery electric cars took a 22% share of the market, up from 19%.</p>"
        "<p>Analysts expect demand to soften next year as incentives end.</p>",
    ),
    "recipe": _html(
        "Lemon drizzle cake",
        "<ul><li>225g butter</li><li>225g caster sugar</li><li>4 eggs</li><li>2 lemons</li>"
        "</ul><p>Beat the butter and sugar, add the eggs, fold in flour and zest. Bake 45 "
        "minutes at 180C.</p>",
    ),
    "insurance_faq": _html(
        "Car insurance FAQ",
        "<p><b>Does my policy cover courtesy cars?</b> Comprehensive policies include a "
        "courtesy car while yours is repaired at an approved garage.</p><p><b>Can I add a "
        "named driver?</b> Yes, from your online account.</p>",
    ),
    "model_range": _html(
        "The Delmaro range",
        "<p>Discover the Kestrova, the Ostra and the Valdis. Kestrova from £19,000. Ostra from "
        "£24,500. Valdis from £31,250. Book a test drive today.</p>",
    ),
}


def pages() -> list[tuple[str, str, Document]]:
    """``(group, name, document)`` for every page asked about."""
    out: list[tuple[str, str, Document]] = []
    site = render(generate(42))
    for family in ("table", "kv", "prose", "pdf", "grid", "listing"):
        for p in [p for p in site if p.family == family][:2]:
            doc = Document.from_bytes(
                p.content, url=f"https://site.test/{p.path}", content_type=p.content_type
            )
            out.append((family, p.path, doc))
    for f in sorted((ROOT / "tests/fixtures/books").glob("*.html")):
        doc = Document.from_bytes(f.read_bytes(), content_type="text/html")
        out.append(("books", f.name, doc))
    for name, body in OFF_TOPIC.items():
        out.append((name, name, Document.from_bytes(body, content_type="text/html")))
    return out


async def main() -> None:
    reader = DefaultTextReader()
    backend = TypeSafeBackend()
    jev = JevClient(backend)
    questions: dict[str, Noul] = {
        f"{w}|{s}": Noul(instructions=t.format(d=d))
        for w, t in WORDINGS.items()
        for s, d in SCHEMAS.items()
    }
    rows: list[dict[str, object]] = []
    scored: list[tuple[str | None, dict[str, float]]] = []
    try:
        for group, name, doc in pages():
            text = reader.read(doc)
            assert text is not None, name
            state = jev.fit_state(text.text.strip()[:DEFAULT_MAX_CHARS], questions)
            answers = await jev.ask(state, questions)
            ps = {k: round(a.p, 3) for k, a in answers.items() if isinstance(a, NoulAnswer)}
            rows.append({"group": group, "page": name, "expected": EXPECTED.get(group), "p": ps})
            scored.append((EXPECTED.get(group), ps))
            print(group, name, flush=True)
    finally:
        await backend.aclose()
    day = datetime.now(UTC).date().isoformat()
    out = Path(__file__).with_name(f"results-{day}.json")
    results = {"threshold": DEFAULT_THRESHOLD, "schemas": SCHEMAS, "wordings": WORDINGS}
    out.write_text(json.dumps({**results, "rows": rows}, indent=1) + "\n")
    for w in WORDINGS:
        missed = passed = 0
        for expected, ps in scored:
            for s in SCHEMAS:
                p = ps[f"{w}|{s}"]
                if expected == s:
                    missed += p < DEFAULT_THRESHOLD
                else:
                    passed += p >= DEFAULT_THRESHOLD
        print(f"{w:24} missed={missed:2}  false passes={passed:2}")


if __name__ == "__main__":
    asyncio.run(main())

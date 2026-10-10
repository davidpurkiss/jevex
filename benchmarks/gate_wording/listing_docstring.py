"""Compare docstrings for the test site's ``Listing`` schema against the live API (#261).

Live and billed (well under $0.001 a run): run only with a key, under a spend cap::

    (set -a; . .env; set +a; JEVEX_JEV_MAX_COST_USD=0.10 \\
        uv run python benchmarks/gate_wording/listing_docstring.py)

Asks the document gate's question (``compare.py``'s ``E_may_include_several``, the default
since #242) with each docstring below, about the pages ``compare.py`` uses. The old
docstring passed spec pages that quote a price. Writes
``benchmarks/gate_wording/listing-docstring-<date>.json`` and prints, per docstring, the
highest p on a page that isn't a listing and the lowest on a grid or listing page.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from compare import EXPECTED, WORDINGS, pages

from jevex.gate import DEFAULT_MAX_CHARS, DefaultTextReader
from jevex.jev import JevClient, Noul, NoulAnswer, TypeSafeBackend

DOCSTRINGS = {
    "offered": "a used car offered for sale",
    "seller_mileage": "a specific used car advertised for sale by a seller, with its mileage",
    "advert_mileage_price": "a seller's advert for one used car, giving its mileage and asking "
    "price",
    "mileage_year": "a used car advertised for sale, with its mileage and year of registration",
    "miles_done": "a second-hand car a seller is advertising, with how many miles it has done",
    "advert_mileage": "an advert for a particular used car for sale, stating its mileage",
}
"""Each candidate as the question uses it (first sentence, lower-cased). ``offered`` was the
docstring before #261; ``mileage_year`` replaced it."""

WORDING = WORDINGS["E_may_include_several"]


async def main() -> None:
    reader = DefaultTextReader()
    backend = TypeSafeBackend()
    jev = JevClient(backend)
    questions = {k: Noul(instructions=WORDING.format(d=d)) for k, d in DOCSTRINGS.items()}
    rows: list[dict[str, object]] = []
    scored: list[tuple[bool, dict[str, float]]] = []
    try:
        for group, name, doc in pages():
            text = reader.read(doc)
            assert text is not None, name
            state = jev.fit_state(text.text.strip()[:DEFAULT_MAX_CHARS], questions)
            answers = await jev.ask(state, questions)
            ps = {k: round(a.p, 3) for k, a in answers.items() if isinstance(a, NoulAnswer)}
            rows.append({"group": group, "page": name, "expected": EXPECTED.get(group), "p": ps})
            scored.append((EXPECTED.get(group) == "Listing", ps))
            print(group, name, flush=True)
    finally:
        await backend.aclose()
    day = datetime.now(UTC).date().isoformat()
    out = Path(__file__).with_name(f"listing-docstring-{day}.json")
    results = {"wording": WORDING, "docstrings": DOCSTRINGS, "rows": rows}
    out.write_text(json.dumps(results, indent=1) + "\n")
    for k in DOCSTRINGS:
        others = max(ps[k] for listing, ps in scored if not listing)
        listings = min(ps[k] for listing, ps in scored if listing)
        print(f"{k:22} highest elsewhere={others:.2f}  lowest on listings={listings:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

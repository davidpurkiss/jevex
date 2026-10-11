"""Compare docstrings for the test site's ``VehicleSpec`` schema against the live API (#324).

Live and billed (well under $0.01 a run): run only with a key, under a spend cap::

    (set -a; . .env; set +a; JEVEX_JEV_MAX_COST_USD=0.10 \\
        uv run python benchmarks/gate_wording/vehicle_docstring.py)

Asks the document gate's question (``compare.py``'s ``E_may_include_several``, the default
since #242) with each docstring below, about the pages ``compare.py`` uses and, when they're
on this machine, the real ``spec-sheets`` corpus's documents
(``benchmarks/corpora/local/spec-sheets/``, never committed). The old docstring rejected
real price lists and press releases. Writes
``benchmarks/gate_wording/vehicle-docstring-<date>.json`` and prints, per docstring, the
highest p on a page that isn't a vehicle spec and the lowest on a test-site spec page and
on a real one.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from compare import EXPECTED, ROOT, WORDINGS, pages

from jevex import Document
from jevex.gate import DEFAULT_MAX_CHARS, DefaultTextReader
from jevex.jev import JevClient, Noul, NoulAnswer, TypeSafeBackend

DOCSTRINGS = {
    "technical_spec": "a manufacturer's technical specification for one vehicle variant",
    "price_and_spec": "a new car variant a manufacturer offers, with its price and specifications",
    "price_or_spec": "a new car variant a manufacturer offers, with its price or specifications",
    "price_list_or_spec": "a manufacturer's price list or technical specification for new "
    "car variants",
    "variant_details": "the price or specifications of a new car variant, as its maker "
    "publishes them",
}
"""Each candidate as the question uses it (first sentence, lower-cased). ``technical_spec``
is the docstring before #324."""

WORDING = WORDINGS["E_may_include_several"]

REAL = ROOT / "benchmarks/corpora/local/spec-sheets"


def real_pages() -> list[tuple[str, str, Document]]:
    """``("real", name, document)`` for each local spec-sheets document, if there are any."""
    if not REAL.is_dir():
        return []
    return [
        ("real", f.name, Document.from_path(f, url=f.as_posix()))
        for f in sorted(REAL.iterdir())
        if f.suffix in (".pdf", ".html")
    ]


async def main() -> None:
    reader = DefaultTextReader()
    backend = TypeSafeBackend()
    jev = JevClient(backend)
    questions = {k: Noul(instructions=WORDING.format(d=d)) for k, d in DOCSTRINGS.items()}
    rows: list[dict[str, object]] = []
    scored: list[tuple[str, dict[str, float]]] = []
    try:
        for group, name, doc in [*pages(), *real_pages()]:
            text = reader.read(doc)
            assert text is not None, name
            state = jev.fit_state(text.text.strip()[:DEFAULT_MAX_CHARS], questions)
            answers = await jev.ask(state, questions)
            ps = {k: round(a.p, 3) for k, a in answers.items() if isinstance(a, NoulAnswer)}
            expected = "VehicleSpec" if group == "real" else EXPECTED.get(group)
            rows.append({"group": group, "page": name, "expected": expected, "p": ps})
            kind = "real" if group == "real" else "site" if expected == "VehicleSpec" else "other"
            scored.append((kind, ps))
            print(group, name, flush=True)
    finally:
        await backend.aclose()
    day = datetime.now(UTC).date().isoformat()
    out = Path(__file__).with_name(f"vehicle-docstring-{day}.json")
    # The real pages' text is the manufacturers': keep only their names and scores.
    results = {"wording": WORDING, "docstrings": DOCSTRINGS, "rows": rows}
    out.write_text(json.dumps(results, indent=1) + "\n")
    for k in DOCSTRINGS:
        others = max(ps[k] for kind, ps in scored if kind == "other")
        site = min(ps[k] for kind, ps in scored if kind == "site")
        real = [ps[k] for kind, ps in scored if kind == "real"]
        lowest_real = f"{min(real):.2f}" if real else "-"
        print(
            f"{k:20} highest elsewhere={others:.2f}  lowest on test-site specs={site:.2f}  "
            f"lowest on real pages={lowest_real}"
        )


if __name__ == "__main__":
    asyncio.run(main())

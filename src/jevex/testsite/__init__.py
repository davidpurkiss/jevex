"""A deterministic synthetic car site with exact ground truth (spec: *Synthetic test site*).

``build(seed, out_dir)`` writes the site's HTML pages, an index, and ``truth.json``: for
every page its template family, schema and expected records. Eval (#46) scores extraction
against it, and the learning demo replays it (#45, #47).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jevex.testsite.dataset import Dataset, generate
from jevex.testsite.render import Page, render
from jevex.testsite.schemas import Listing, VehicleSpec

TRUTH_FILE = "truth.json"


def build(seed: int = 42, out_dir: str | Path = "testsite/build") -> dict[str, Any]:
    """Generate and write the site for ``seed``. Returns the ground-truth manifest."""
    out = Path(out_dir)
    dataset = generate(seed)
    pages = render(dataset)
    for page in pages:
        target = out / page.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page.html, encoding="utf-8")
    (out / "index.html").write_text(_index(pages), encoding="utf-8")
    manifest: dict[str, Any] = {
        "seed": seed,
        "pages": [p.truth() for p in pages],
        "digest": digest(pages),
    }
    (out / TRUTH_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def digest(pages: list[Page]) -> str:
    """A hash of every page's path and bytes: equal digests mean identical sites."""
    h = hashlib.sha256()
    for page in pages:
        h.update(page.path.encode())
        h.update(page.html.encode())
    return h.hexdigest()


def _index(pages: list[Page]) -> str:
    links = "\n".join(f'<li><a href="{p.path}">{p.path}</a> ({p.family})</li>' for p in pages)
    head = "<!doctype html><html><head><meta charset='utf-8'><title>jevex test site</title>"
    return f"{head}</head><body><h1>jevex test site</h1><ul>\n{links}\n</ul></body></html>\n"


__all__ = ["Dataset", "Listing", "Page", "VehicleSpec", "build", "digest", "generate", "render"]

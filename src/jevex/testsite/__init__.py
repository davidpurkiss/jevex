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
    """Generate and write the site for ``seed``. Returns the ground-truth manifest.

    If ``out_dir`` holds an earlier build, its files (the pages its ``truth.json`` lists,
    ``index.html`` and ``truth.json``) are removed first, so no stale pages survive. Any
    other non-empty directory is refused, and nothing else in it is ever deleted.
    """
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        _remove_previous_build(out)
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


def _remove_previous_build(out: Path) -> None:
    """Delete exactly what an earlier build wrote. Refuse if ``out`` isn't one."""
    refusal = f"{out} isn't empty and isn't a jevex test-site build; refusing to write"
    try:
        manifest = json.loads((out / TRUTH_FILE).read_text())
        paths = [page["path"] for page in manifest["pages"]]
        is_build = isinstance(manifest["seed"], int) and isinstance(manifest["digest"], str)
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(refusal) from None
    root = out.resolve()
    if not is_build or not all(
        isinstance(p, str) and (root / p).resolve().is_relative_to(root) for p in paths
    ):
        raise ValueError(refusal)  # never delete outside out_dir (absolute paths, "..")
    for relative in [*paths, "index.html", TRUTH_FILE]:
        (out / relative).unlink(missing_ok=True)
    for directory in sorted((d for d in out.rglob("*") if d.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()


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

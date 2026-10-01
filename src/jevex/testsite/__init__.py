"""A deterministic synthetic car site with exact ground truth (spec: *Synthetic test site*).

``build(seed, out_dir)`` writes the site's pages (HTML, PDFs and images), an index, and
``truth.json``: for every page its template family, wave, schema, content type and
expected records. Eval (#46) scores extraction against it, and the learning demo replays
it in order (#47), a wave of template families at a time (:mod:`jevex.testsite.waves`).
The phrasing bank (:mod:`jevex.testsite.phrasing`) words each fact several ways.
``server(out_dir)`` serves a build over HTTP (``jevex testsite build|serve``).
"""

from __future__ import annotations

import functools
import hashlib
import json
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jevex.testsite.dataset import Dataset, generate
from jevex.testsite.render import FAMILIES, Page, render
from jevex.testsite.schemas import Listing, VehicleSpec
from jevex.testsite.waves import DEFAULT_WAVES, Waves, check_waves, parse_waves, wave_numbers

if TYPE_CHECKING:
    from collections.abc import Sequence

TRUTH_FILE = "truth.json"
BUILD_DIR = "testsite/build"


def build(
    seed: int = 42,
    out_dir: str | Path = BUILD_DIR,
    *,
    waves: Sequence[Sequence[str]] = DEFAULT_WAVES,
) -> dict[str, Any]:
    """Generate and write the site for ``seed``. Returns the ground-truth manifest.

    Pages are listed (and linked from the index) wave by wave, in ``waves`` order; within
    a wave they keep :func:`render`'s order. Families the schedule leaves out aren't
    built. The manifest records the schedule as ``waves`` and each page's ``wave``.

    If ``out_dir`` holds an earlier build, its files (the pages its ``truth.json`` lists,
    ``index.html`` and ``truth.json``) are removed first, so no stale pages survive. Any
    other non-empty directory is refused, and nothing else in it is ever deleted. The
    earlier build stays if rendering fails (e.g. ``ImportError`` without Pillow).
    """
    schedule = check_waves(waves)
    numbers = wave_numbers(schedule)
    out = Path(out_dir)
    pages = sorted(render(generate(seed), families=numbers), key=lambda p: numbers[p.family])
    if out.exists() and any(out.iterdir()):
        _remove_previous_build(out)
    for page in pages:
        target = out / page.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(page.content)
    (out / "index.html").write_text(_index(pages, schedule), encoding="utf-8")
    manifest: dict[str, Any] = {
        "seed": seed,
        "waves": [list(wave) for wave in schedule],
        "pages": [p.truth() | {"wave": numbers[p.family]} for p in pages],
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
    """A hash of every page's path and bytes: equal digests mean identical sites.

    PDFs and HTML are byte for byte the same everywhere. Rasterised pages (scans and
    infographics) can differ between Pillow versions, which may draw text a pixel apart.
    """
    h = hashlib.sha256()
    for page in pages:
        h.update(page.path.encode())
        h.update(page.content)
    return h.hexdigest()


def _index(pages: list[Page], waves: Waves) -> str:
    sections: list[str] = []
    for number, wave in enumerate(waves, start=1):
        links = "\n".join(
            f'<li><a href="{escape(p.path)}">{escape(p.path)}</a> ({p.family})</li>'
            for p in pages
            if p.family in wave
        )
        sections.append(f"<h2>Wave {number}: {', '.join(wave)}</h2>\n<ul>\n{links}\n</ul>")
    head = "<!doctype html><html><head><meta charset='utf-8'><title>jevex test site</title>"
    body = "\n".join(sections)
    return f"{head}</head><body><h1>jevex test site</h1>\n{body}\n</body></html>\n"


def server(
    directory: str | Path = BUILD_DIR, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """An HTTP server for the build in ``directory``, bound but not yet serving: call
    ``serve_forever()`` (and ``server_close()`` when done). ``port=0`` picks a free port
    (``server_address`` says which).

    Raises ``ValueError`` if ``directory`` holds no build (no ``truth.json``), and
    ``OSError`` if the address can't be bound.
    """
    root = Path(directory)
    if not (root / TRUTH_FILE).is_file():
        raise ValueError(f"{root} has no test site; build one first (jevex testsite build)")
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    return ThreadingHTTPServer((host, port), handler)


__all__ = [
    "DEFAULT_WAVES",
    "FAMILIES",
    "Dataset",
    "Listing",
    "Page",
    "VehicleSpec",
    "Waves",
    "build",
    "check_waves",
    "digest",
    "generate",
    "parse_waves",
    "render",
    "server",
]

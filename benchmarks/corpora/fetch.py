# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jevex",
# ]
#
# [tool.uv.sources]
# jevex = { path = "../..", editable = true }
# ///
"""Fetch a benchmark corpus that the repo holds as a manifest (docs/benchmarks.md, #307).

Run from the repository root::

    uv run --script benchmarks/corpora/fetch.py benchmarks/corpora/spec-sheets/manifest.json \\
        --out DIR
    jevex corpus check DIR benchmarks/corpora/spec-sheets.lock

The repo commits each document's URL and sha256 and our labels, never the documents, which
are their publishers'. This downloads every document in the manifest into ``DIR`` under its
``file`` name, through :class:`jevex.SimpleFetcher` (robots.txt honoured, a second between
requests to one host), checks its sha256 and copies the manifest's sibling ``truth.json``
into ``DIR``. A document already in ``DIR`` with the right hash isn't fetched again, so a
partial run can be resumed.

Each document is reported. A dead link, a robots.txt refusal or a different hash fails the
run (exit 1), as does a file already in ``DIR`` that differs from the manifest's: nothing is
overwritten, and a document whose hash differs isn't written. A manifest may be empty, and
the labels may not exist yet: without a ``truth.json`` the documents are still fetched.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jevex.eval import TRUTH_FILE
from jevex.fetch import FetchError, SimpleFetcher

# Manufacturers' brochures run past SimpleFetcher's 20 MB default. The hash, not the size,
# decides what's kept.
MAX_BYTES = 200 * 1024 * 1024


class ManifestDocument(BaseModel):
    """One document: where to get it and the bytes it must be."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str = Field(pattern=r"^[A-Za-z0-9][\w.-]*$")
    """A bare file name, so a manifest can't write outside the corpus directory."""
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured: date
    kind: Literal["pdf", "html"]


class Manifest(BaseModel):
    """A fetchable corpus: ``manifest.json`` next to its ``truth.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    purpose: str = ""
    documents: tuple[ManifestDocument, ...] = ()

    @model_validator(mode="after")
    def _distinct_files(self) -> Self:
        files = [d.file for d in self.documents]
        clashes = sorted({f for f in files if files.count(f) > 1 or f == TRUTH_FILE})
        if clashes:
            raise ValueError(f"file names used twice or for the labels: {', '.join(clashes)}")
        return self


@dataclass(frozen=True)
class Outcome:
    """What happened to one file of the corpus."""

    file: str
    status: Literal["fetched", "present", "copied", "missing", "failed"]
    detail: str = ""

    def describe(self) -> str:
        line = f"{self.status:<8} {self.file}"
        return f"{line}: {self.detail}" if self.detail else line


async def fetch_corpus(manifest: Path, out: Path, fetcher: SimpleFetcher) -> list[Outcome]:
    """Fetch ``manifest``'s documents and copy its labels into ``out``, one outcome per file.

    Raises ``ValueError`` (a pydantic ``ValidationError``) if the manifest isn't valid.
    """
    corpus = await asyncio.to_thread(_open, manifest, out)
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(_fetch(d, out, fetcher)) for d in corpus.documents]
    truth = await asyncio.to_thread(_copy_truth, manifest.parent / TRUTH_FILE, out)
    return [t.result() for t in tasks] + [truth]


def _open(manifest: Path, out: Path) -> Manifest:
    corpus = Manifest.model_validate_json(manifest.read_bytes())
    out.mkdir(parents=True, exist_ok=True)
    return corpus


async def _fetch(document: ManifestDocument, out: Path, fetcher: SimpleFetcher) -> Outcome:
    there = await asyncio.to_thread(_already_there, document, out)
    if there is not None:
        return there
    try:
        fetched = await fetcher.fetch(document.url)
    except FetchError as exc:
        return Outcome(document.file, "failed", str(exc))
    found = _sha256(fetched.content)
    if found != document.sha256:
        return Outcome(
            document.file,
            "failed",
            f"{document.url} has sha256 {found}, not {document.sha256}; not written",
        )
    await asyncio.to_thread(_write, out / document.file, fetched.content)
    return Outcome(document.file, "fetched")


def _already_there(document: ManifestDocument, out: Path) -> Outcome | None:
    target = out / document.file
    if not target.exists():
        return None
    found = _sha256(target.read_bytes())
    if found == document.sha256:
        return Outcome(document.file, "present")
    return Outcome(
        document.file, "failed", f"already in {out} with sha256 {found}; not overwritten"
    )


def _write(target: Path, content: bytes) -> None:
    # Written whole or not at all, so an interrupted run never leaves a file with a bad hash.
    partial = target.with_name(f".{target.name}.part")
    partial.write_bytes(content)
    partial.replace(target)


def _copy_truth(source: Path, out: Path) -> Outcome:
    target = out / TRUTH_FILE
    if not source.exists():
        return Outcome(TRUTH_FILE, "missing", f"no labels next to the manifest yet ({source})")
    if target.exists():
        if target.read_bytes() == source.read_bytes():
            return Outcome(TRUTH_FILE, "present")
        return Outcome(TRUTH_FILE, "failed", f"{target} differs from {source}; not overwritten")
    shutil.copyfile(source, target)
    return Outcome(TRUTH_FILE, "copied")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


async def _run(manifest: Path, out: Path) -> list[Outcome]:
    async with SimpleFetcher(max_bytes=MAX_BYTES) as fetcher:
        return await fetch_corpus(manifest, out, fetcher)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a benchmark corpus from its manifest.")
    parser.add_argument("manifest", type=Path, help="the corpus's manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="the corpus directory")
    args = parser.parse_args(argv)
    try:
        outcomes = asyncio.run(_run(args.manifest, args.out))
    except (OSError, ValidationError) as exc:
        print(f"can't fetch {args.manifest}: {exc}", file=sys.stderr)
        return 1
    for outcome in outcomes:
        print(outcome.describe())
    counts = Counter(o.status for o in outcomes)
    print(", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

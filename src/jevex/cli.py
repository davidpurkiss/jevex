"""The ``jevex`` command line (spec: *Integration › CLI*).

``jevex extract <file|url> --schema module:Class`` prints extracted records as JSON. The
other commands are placeholders until their issues land. Uses only the standard library
(argparse), so the CLI adds nothing to a core install.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from pydantic import BaseModel

from jevex import __version__
from jevex.document import Document
from jevex.extractor import Extractor
from jevex.fetch import FetchError, SimpleFetcher
from jevex.jev import JevError
from jevex.schema import UnsupportedFieldError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jevex.jev import JevClient

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

PLANNED = {
    "learn": ("Synthesise and test generators from logged examples", 39),
    "pack": ("Export, import and diff generator packs", 41),
    "eval": ("Evaluate extraction against a labelled corpus", 46),
    "testsite": ("Build and serve the synthetic test site", 45),
    "serve": ("Run the extraction microservice", 53),
}


class CliError(Exception):
    """A user-facing error: printed without a traceback, exit code 1."""


def load_schema(spec: str) -> type[BaseModel]:
    """Import ``module:Class`` or ``path/to/file.py:Class`` and check it's a Pydantic model.

    ``Class`` may be dotted (``Outer.Inner``). Modules in the current directory are found
    even when running the installed console script.
    """
    target, sep, name = spec.rpartition(":")
    if not sep or not target or not name:
        raise CliError(f"--schema must look like module:Class or path.py:Class, not {spec!r}")
    module = _load_file(Path(target)) if target.endswith(".py") else _load_module(target)
    obj: object = module
    for part in name.split("."):
        if not hasattr(obj, part):
            raise CliError(f"{target!r} has no attribute {name!r}")
        obj = getattr(obj, part)
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise CliError(f"{spec!r} is not a Pydantic model")
    return obj


def _load_module(name: str) -> object:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise CliError(f"couldn't import {name!r}: {exc}") from exc


def _load_file(path: Path) -> object:
    if not path.is_file():
        raise CliError(f"no such schema file: {path}")
    # A private, path-specific module name, so a schema file called json.py or models.py
    # never replaces a real module. It must be in sys.modules for forward references.
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    name = f"_jevex_schema_{path.stem}_{digest}"
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise CliError(f"couldn't load schema file: {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[name]
        raise CliError(f"couldn't import {str(path)!r}: {exc}") from exc
    return module


async def load_document(source: str) -> Document:
    """A local file, or an http(s) URL fetched politely with :class:`SimpleFetcher`."""
    if source.startswith(("http://", "https://")):
        async with SimpleFetcher() as fetcher:
            try:
                return await fetcher.fetch(source)
            except FetchError as exc:
                raise CliError(str(exc)) from exc
    path = Path(source)
    if not await asyncio.to_thread(path.is_file):
        raise CliError(f"no such file: {source}")
    return await asyncio.to_thread(Document.from_path, path)


async def _extract(args: argparse.Namespace, jev: JevClient | None) -> dict[str, Any]:
    schemas = [load_schema(s) for s in args.schema]
    if jev is None and not os.environ.get("TYPESAFE_API_KEY"):
        raise CliError("TYPESAFE_API_KEY is not set (jevex needs a Jev API key to extract)")
    try:
        extractor = Extractor(schemas, jev=jev, threshold=args.threshold)
    except (ValueError, UnsupportedFieldError) as exc:
        raise CliError(str(exc)) from exc
    # Schemas are checked before the (possibly networked) document load.
    document = await load_document(args.source)
    async with extractor:
        try:
            result = await extractor.extract(document)
        except JevError as exc:
            raise CliError(f"Jev: {exc}") from exc
    if args.meta:
        return result.to_dict()
    return {
        "records": [
            {
                "schema": r.schema_name,
                "entity": r.entity,
                "record": r.record.model_dump(mode="json"),
            }
            for r in result.records
        ]
    }


def _probability(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, not {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for every ``jevex`` command."""
    parser = argparse.ArgumentParser(
        prog="jevex", description="Extract typed records from web pages and PDFs using Jev."
    )
    parser.add_argument("--version", action="version", version=f"jevex {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    extract = commands.add_parser(
        "extract",
        help="Extract records from a file or URL and print them as JSON",
        description="Extract records from a file or URL. Needs TYPESAFE_API_KEY.",
    )
    extract.add_argument("source", help="A local file (HTML, PDF, image) or an http(s) URL")
    extract.add_argument(
        "--schema",
        action="append",
        required=True,
        metavar="MODULE:CLASS",
        help="A Pydantic model, as module:Class or path/to/file.py:Class (repeatable)",
    )
    extract.add_argument(
        "--meta", action="store_true", help="Include per-field metadata and document meta"
    )
    extract.add_argument(
        "--threshold",
        type=_probability,
        default=0.0,
        help="Confidence (0–1) below which values become null (default 0: keep everything)",
    )
    extract.add_argument("--indent", type=int, default=2, help="JSON indent (0 for one line)")

    for name, (summary, issue) in PLANNED.items():
        commands.add_parser(name, help=f"{summary} (not implemented yet, #{issue})")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    jev: JevClient | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Run the CLI and return its exit code.

    0: success. 1: a runtime or user error (printed to stderr as ``jevex: error: ...``).
    2: a usage error, no command, or a command that isn't implemented yet.
    ``jev``, ``out`` and ``err`` are injectable for tests.
    """
    stdout: TextIO = out or sys.stdout
    stderr: TextIO = err or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(stderr)
        return EXIT_USAGE
    if args.command in PLANNED:
        summary, issue = PLANNED[args.command]
        print(f"jevex {args.command}: not implemented yet ({summary}; see #{issue})", file=stderr)
        return EXIT_USAGE
    try:
        payload = asyncio.run(_extract(args, jev))
    except CliError as exc:
        print(f"jevex: error: {exc}", file=stderr)
        return EXIT_ERROR
    json.dump(payload, stdout, indent=args.indent or None, ensure_ascii=False)
    stdout.write("\n")
    return EXIT_OK


def entrypoint() -> None:
    """Console-script entry point."""
    sys.exit(main())

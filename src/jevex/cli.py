"""The ``jevex`` command line (spec: *Integration › CLI*).

``jevex extract <file|url> --schema module:Class`` prints extracted records as JSON,
``jevex eval`` scores a corpus, and ``jevex learn`` compiles logged examples into a pack
diff. The other commands are placeholders until their issues land. Uses only the standard library
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
from jevex.budgets import Budgets, RunBudget
from jevex.document import Document
from jevex.eval import EvalReport, evaluate, load_corpus
from jevex.extractor import Extractor
from jevex.fetch import FetchError, SimpleFetcher
from jevex.generators import InvalidGeneratorError
from jevex.jev import JevError
from jevex.learn import LEARN_THRESHOLD, PACK_GENERATORS, PackDiff, pack_generators
from jevex.llm import ANTHROPIC_MODEL
from jevex.schema import UnsupportedFieldError
from jevex.store import StoreError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jevex.generators import GeneratorSpec
    from jevex.jev import JevClient
    from jevex.llm import LLM

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

PLANNED = {
    "pack": ("Export, import and diff generator packs", 41),
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
    if jev is None and not os.environ.get("TYPESAFE_API_KEY", "").strip():
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


async def _eval(args: argparse.Namespace, jev: JevClient | None) -> EvalReport:
    schemas = [load_schema(s) for s in args.schema]
    try:
        corpus = load_corpus(args.corpus)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    if jev is None and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise CliError("TYPESAFE_API_KEY is not set (jevex needs a Jev API key to extract)")
    try:
        extractor = Extractor(schemas, jev=jev)
    except (ValueError, UnsupportedFieldError) as exc:
        raise CliError(str(exc)) from exc
    async with extractor:
        try:
            return await evaluate(extractor, corpus, concurrency=max(1, args.concurrency))
        except ValueError as exc:
            raise CliError(str(exc)) from exc
        except JevError as exc:  # the spend cap or the API itself: the run can't be scored
            raise CliError(f"Jev: {exc}") from exc


LLM_PROVIDERS = ("anthropic", "openai", "gemini", "litellm")


def load_llm(spec: str) -> LLM:
    """``provider[:model]``: an adapter from :data:`LLM_PROVIDERS` (each needs its extra).

    Only ``anthropic`` has a default model (:data:`~jevex.llm.ANTHROPIC_MODEL`).
    """
    provider, _, model = spec.partition(":")
    if provider not in LLM_PROVIDERS:
        raise CliError(f"--llm must start with one of {', '.join(LLM_PROVIDERS)}, not {spec!r}")
    if not model and provider != "anthropic":
        raise CliError(f"--llm {provider} needs a model, as {provider}:<model>")
    try:
        if provider == "anthropic":
            from jevex.llm.anthropic import AnthropicLLM

            return AnthropicLLM(model or ANTHROPIC_MODEL)
        if provider == "openai":
            from jevex.llm.openai import OpenAILLM

            return OpenAILLM(model)
        if provider == "gemini":
            from jevex.llm.gemini import GeminiLLM

            return GeminiLLM(model)
        from jevex.llm.litellm import LiteLLM

        return LiteLLM(model)
    except ImportError as exc:
        raise CliError(f"--llm {provider} needs the {provider} extra: {exc}") from exc
    except Exception as exc:  # e.g. the provider's client finds no API key
        raise CliError(f"--llm {spec}: {exc}") from exc


def _learn_files(out: Path, pack: str | None) -> list[GeneratorSpec]:
    """Check ``--out`` before anything is spent (``PackDiff.write`` checks again), and read
    the ``--pack`` generators."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise CliError(f"--out {out} already exists and isn't an empty directory")
    try:
        return pack_generators(Path(pack)) if pack else []
    except (FileNotFoundError, InvalidGeneratorError) as exc:
        raise CliError(str(exc)) from exc


async def _learn(args: argparse.Namespace, jev: JevClient | None, llm: LLM | None) -> PackDiff:
    schemas = [load_schema(s) for s in args.schema]
    pack = await asyncio.to_thread(_learn_files, Path(args.out), args.pack)
    if jev is None and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise CliError("TYPESAFE_API_KEY is not set (jevex learn tests generators with Jev)")
    generator_llm = llm or load_llm(args.llm)
    run = None
    if args.max_spend is not None or args.max_jev_spend is not None:
        run = RunBudget(max_spend=args.max_spend, max_jev_spend=args.max_jev_spend, period="run")
    try:
        extractor = Extractor(
            schemas,
            jev=jev,
            store=args.store,
            generator_llm=generator_llm,
            learn_threshold=args.learn_threshold,
            budgets=Budgets(run=run),
        )
    except (ValueError, UnsupportedFieldError) as exc:
        raise CliError(str(exc)) from exc
    async with extractor:
        try:
            diff = await extractor.compile_pack(pack)
        except StoreError as exc:
            raise CliError(f"store: {exc}") from exc
        except JevError as exc:
            raise CliError(f"Jev: {exc}") from exc
    try:
        await asyncio.to_thread(diff.write, Path(args.out))
    except OSError as exc:
        raise CliError(str(exc)) from exc
    return diff


def format_diff(diff: PackDiff, out: str) -> str:
    """A plain-text summary: outcomes by status, then the generators written."""
    counts: dict[str, int] = {}
    for outcome in diff.outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    mix = ", ".join(f"{status} {n}" for status, n in sorted(counts.items())) or "none"
    lines = [f"examples: {len(diff.outcomes)} ({mix})"]
    lines.append(f"wrote {len(diff.generators)} generator(s) to {Path(out) / PACK_GENERATORS}")
    lines += [f"  {spec.id}  {spec.field}  {spec.match.regex}" for spec in diff.generators]
    return "\n".join(lines) + "\n"


def format_report(report: EvalReport) -> str:
    """A plain-text summary: run metrics, then precision/recall per field."""

    def pct(x: float | None) -> str:
        return "   –  " if x is None else f"{x * 100:5.1f}%"

    def secs(x: float | None) -> str:
        return "–" if x is None else f"{x:.2f}s"

    s = report.summary()
    lines = [
        f"documents: {s['documents']}  errors: {s['errors']}",
        f"precision: {pct(s['precision'])}  recall: {pct(s['recall'])}",
        f"per document: ${s['cost_per_document']:.5f}  {s['jev_requests_per_document']:.1f} "
        f"Jev requests ({s['jev_questions_per_document']:.1f} questions)  "
        f"{s['llm_calls_per_document']:.1f} LLM calls",
        f"latency: mean {secs(s['seconds_per_document'])}  p50 {secs(s['latency_p50'])}  "
        f"p95 {secs(s['latency_p95'])}",
        f"resolution mix: {s['resolution_mix'] or 'none'}",
        "",
        f"{'field':40} {'precision':>9} {'recall':>7} "
        f"{'ok':>5} {'wrong':>5} {'miss':>5} {'extra':>5}",
    ]
    for name, f in report.field_scores().items():
        lines.append(
            f"{name:40} {pct(f.precision):>9} {pct(f.recall):>7} "
            f"{f.correct:>5} {f.wrong:>5} {f.missing:>5} {f.spurious:>5}"
        )
    return "\n".join(lines) + "\n"


def _usd(text: str) -> float:
    value = float(text)
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {text}")
    return value


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

    evaluate = commands.add_parser(
        "eval",
        help="Evaluate extraction against a labelled corpus",
        description="Score extraction against a corpus directory with a truth.json. "
        "Needs TYPESAFE_API_KEY.",
    )
    evaluate.add_argument("corpus", help="Directory containing truth.json and the documents")
    evaluate.add_argument(
        "--schema",
        action="append",
        required=True,
        metavar="MODULE:CLASS",
        help="Every schema the corpus uses (repeatable), e.g. jevex.testsite:VehicleSpec",
    )
    evaluate.add_argument("--concurrency", type=int, default=4, help="Documents at a time")
    evaluate.add_argument("--json", action="store_true", help="Print the full report as JSON")

    learn = commands.add_parser(
        "learn",
        help="Synthesise and test generators from logged examples; write a pack diff",
        description="Learn generators from the verified examples a store logged (for "
        "example in learn_mode='compile') and write the ones to add to a pack as "
        "OUT/generators/<id>.yaml for review. Nothing is published to the store. Needs "
        "TYPESAFE_API_KEY and the generator LLM's key.",
    )
    learn.add_argument(
        "--schema",
        action="append",
        required=True,
        metavar="MODULE:CLASS",
        help="A Pydantic model whose examples to learn from (repeatable)",
    )
    learn.add_argument(
        "--store", required=True, help="The store holding the examples, e.g. sqlite:///jevex.db"
    )
    learn.add_argument("--out", required=True, help="A new or empty directory for the pack diff")
    learn.add_argument(
        "--pack", help="The pack to diff against (its generators count as already learned)"
    )
    learn.add_argument(
        "--llm",
        default="anthropic",
        metavar="PROVIDER[:MODEL]",
        help=f"The generator LLM: {', '.join(LLM_PROVIDERS)} (default anthropic, "
        f"{ANTHROPIC_MODEL})",
    )
    learn.add_argument(
        "--learn-threshold",
        type=_probability,
        default=LEARN_THRESHOLD,
        help=f"Verification probability an example needs (default {LEARN_THRESHOLD})",
    )
    learn.add_argument("--max-spend", type=_usd, help="Stop calling the LLM after this many USD")
    learn.add_argument("--max-jev-spend", type=_usd, help="Stop calling Jev after this many USD")
    learn.add_argument("--json", action="store_true", help="Print the diff and outcomes as JSON")

    for name, (summary, issue) in PLANNED.items():
        commands.add_parser(name, help=f"{summary} (not implemented yet, #{issue})")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    jev: JevClient | None = None,
    llm: LLM | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Run the CLI and return its exit code.

    0: success. 1: a runtime or user error (printed to stderr as ``jevex: error: ...``),
    including ``jevex eval`` runs where any document failed (the report is still printed).
    2: a usage error, no command, or a command that isn't implemented yet.
    ``jev``, ``llm`` (``jevex learn``'s generator LLM), ``out`` and ``err`` are injectable
    for tests.
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
        if args.command == "eval":
            report = asyncio.run(_eval(args, jev))
            if args.json:
                json.dump(report.to_dict(), stdout, indent=2, ensure_ascii=False)
                stdout.write("\n")
            else:
                stdout.write(format_report(report))
            for doc in report.failed:
                print(f"jevex: error: {doc.path}: {doc.error}", file=stderr)
            # A run with failed documents isn't a clean measurement, even though it's scored.
            return EXIT_ERROR if report.failed else EXIT_OK
        if args.command == "learn":
            diff = asyncio.run(_learn(args, jev, llm))
            if args.json:
                json.dump(diff.model_dump(mode="json"), stdout, indent=2, ensure_ascii=False)
                stdout.write("\n")
            else:
                stdout.write(format_diff(diff, args.out))
            return EXIT_OK
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

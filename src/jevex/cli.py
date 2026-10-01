"""The ``jevex`` command line (spec: *Integration › CLI*).

``jevex extract <file|url> --schema module:Class`` prints extracted records as JSON,
``jevex eval`` scores a corpus (``--replay``: learning curves; ``--gate``: a regression
gate against a baseline), ``jevex learn`` compiles
logged examples into a pack diff, ``jevex pack export|import|diff`` moves learned state
between stores and packs, ``jevex stats`` serves the stats UI over a store or a replay's
CSV (``jevex stats export --svg <view>`` writes a chart), ``jevex serve`` runs the
extraction microservice (``server`` extra, :mod:`jevex.server`), and ``jevex testsite
build|serve`` writes and serves the synthetic test site. Uses only the standard library
(argparse), so the CLI adds nothing to a core install.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, get_args

from pydantic import BaseModel, ValidationError

from jevex import __version__
from jevex.baseline import (
    Baseline,
    BaselineError,
    GateTolerances,
    check_baseline,
    corpus_digest,
    ensure_comparable,
)
from jevex.budgets import Budgets, Period, RunBudget
from jevex.document import Document
from jevex.eval import evaluate, load_corpus
from jevex.extractor import Extractor
from jevex.fetch import FetchError, SimpleFetcher
from jevex.generators import InvalidGeneratorError
from jevex.jev import JevError
from jevex.learn import LEARN_THRESHOLD, PackDiff
from jevex.llm import ANTHROPIC_MODEL
from jevex.packs import (
    PACK_GENERATORS,
    PackError,
    PackManifest,
    diff_packs,
    export_pack,
    import_pack,
    load_pack,
    pack_generators,
)
from jevex.replay import REPLAY_BATCH_SIZE, LearningStoppedError, replay
from jevex.schema import UnsupportedFieldError
from jevex.stats import CHART_VIEWS, chart_svg, replay_loader, stats_server, store_loader
from jevex.stats.server import DEFAULT_PORT
from jevex.store import StoreError, open_store
from jevex.testsite import BUILD_DIR, build, server
from jevex.testsite.waves import DEFAULT_WAVES, format_waves, parse_waves

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jevex.baseline import EvalMode, GateResult
    from jevex.eval import EvalReport
    from jevex.generators import GeneratorSpec
    from jevex.jev import JevClient
    from jevex.llm import LLM
    from jevex.packs import Pack, PackChanges
    from jevex.replay import ReplayReport
    from jevex.stats.server import Loader
    from jevex.store import Store
    from jevex.testsite.waves import Waves

EVAL_CONCURRENCY = 4

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

SERVE_PORT = 8080


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
        except PackError as exc:  # an installed community pack that doesn't load
            raise CliError(f"pack: {exc}") from exc
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


async def _eval(args: argparse.Namespace, jev: JevClient | None, llm: LLM | None) -> EvalReport:
    schemas = [load_schema(s) for s in args.schema]
    try:
        corpus = load_corpus(args.corpus)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    if jev is None and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise CliError("TYPESAFE_API_KEY is not set (jevex needs a Jev API key to extract)")
    model = (llm or load_llm(args.llm)) if args.llm else None
    try:
        try:
            extractor = Extractor(schemas, jev=jev, extraction_llm=model)
        except (ValueError, UnsupportedFieldError) as exc:
            raise CliError(str(exc)) from exc
        async with extractor:
            try:
                concurrency = EVAL_CONCURRENCY if args.concurrency is None else args.concurrency
                return await evaluate(extractor, corpus, concurrency=max(1, concurrency))
            except ValueError as exc:
                raise CliError(str(exc)) from exc
            except JevError as exc:  # the spend cap or the API itself: the run can't be scored
                raise CliError(f"Jev: {exc}") from exc
    finally:
        # The extractor doesn't close an LLM it's given; an adapter built here is ours.
        close = getattr(model, "aclose", None) if llm is None else None
        if close is not None:
            await close()


async def _replay(args: argparse.Namespace, jev: JevClient | None, llm: LLM | None) -> ReplayReport:
    schemas = [load_schema(s) for s in args.schema]
    try:
        corpus = load_corpus(args.corpus)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    for out in (args.csv, args.html):
        if out is not None and not await asyncio.to_thread(Path(out).parent.is_dir):
            raise CliError(f"no such directory for {out}")  # before anything is spent
    if jev is None and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise CliError("TYPESAFE_API_KEY is not set (jevex needs a Jev API key to extract)")
    model = (llm or load_llm(args.llm)) if args.llm else None
    try:
        try:
            # An empty in-memory store, so key mappings and generators are learned from
            # scratch; no community packs, so the curve starts from nothing.
            extractor = Extractor(
                schemas,
                jev=jev,
                store=":memory:",
                extraction_llm=model,
                generator_llm=model,
                community_packs=False,
            )
        except (ValueError, UnsupportedFieldError) as exc:
            raise CliError(str(exc)) from exc
        async with extractor:
            try:
                return await replay(extractor, corpus, batch_size=args.batch_size)
            except (ValueError, LearningStoppedError) as exc:
                raise CliError(str(exc)) from exc
            except JevError as exc:
                raise CliError(f"Jev: {exc}") from exc
    finally:
        # The extractor doesn't close an LLM it's given; an adapter built here is ours.
        close = getattr(model, "aclose", None) if llm is None else None
        if close is not None:
            await close()


def _write_replay(args: argparse.Namespace, replayed: ReplayReport) -> list[str]:
    """Write ``--csv`` and ``--html``; return a line saying where each went."""
    written: list[str] = []
    for out, text in ((args.csv, replayed.to_csv), (args.html, replayed.to_html)):
        if out is None:
            continue
        try:
            Path(out).write_text(text(), encoding="utf-8")
        except OSError as exc:
            raise CliError(str(exc)) from exc
        written.append(f"wrote {out}")
    return written


TOLERANCE_FLAGS = {
    "max_accuracy_drop": ("--max-accuracy-drop", "accuracy_drop"),
    "max_field_drop": ("--max-field-drop", "field_accuracy_drop"),
    "max_llm_rise": ("--max-llm-rise", "llm_rate_rise"),
}
"""argparse dest → (flag, :class:`~jevex.baseline.GateTolerances` field)."""


@dataclass(frozen=True)
class _GatePlan:
    """What ``--gate`` or ``--write-baseline`` will do after the run, worked out before it
    so a bad baseline or path fails before anything is spent."""

    corpus: str
    mode: EvalMode
    tolerances: GateTolerances
    baseline: Baseline | None
    """The baseline to gate against (``--gate``); ``None`` when writing one."""


def _plan_gate(args: argparse.Namespace) -> _GatePlan | None:
    if args.gate is None and args.write_baseline is None:
        return None
    mode: EvalMode = "replay" if args.replay else "eval"
    try:
        corpus = corpus_digest(args.corpus)
    except (OSError, ValueError) as exc:
        raise CliError(str(exc)) from exc
    overrides = {
        field: getattr(args, dest)
        for dest, (_, field) in TOLERANCE_FLAGS.items()
        if hasattr(args, dest)
    }
    try:
        if args.gate is not None:
            baseline = Baseline.load(args.gate)
            tolerances = baseline.tolerances.model_copy(update=overrides)
            ensure_comparable(baseline, corpus=corpus, mode=mode)
            return _GatePlan(corpus, mode, tolerances, baseline)
        target = Path(args.write_baseline)
        if not target.parent.is_dir():
            raise CliError(f"no such directory for {target}")
        # Re-recording a baseline keeps the tolerances the old one set.
        previous = Baseline.load(target).tolerances if target.exists() else GateTolerances()
    except BaselineError as exc:
        raise CliError(str(exc)) from exc
    return _GatePlan(corpus, mode, previous.model_copy(update=overrides), None)


def _finish_gate(args: argparse.Namespace, plan: _GatePlan, report: EvalReport, err: TextIO) -> int:
    """Check or write the baseline; print what happened to ``err``. Returns the exit code."""
    if plan.baseline is None:
        if report.failed:
            raise CliError("not writing a baseline: documents failed (see above)")
        baseline = Baseline.from_report(
            report, corpus=plan.corpus, mode=plan.mode, tolerances=plan.tolerances
        )
        try:
            baseline.write(args.write_baseline)
        except OSError as exc:
            raise CliError(str(exc)) from exc
        print(f"jevex: wrote baseline {args.write_baseline}", file=err)
        return EXIT_OK
    result = check_baseline(
        report, plan.baseline, corpus=plan.corpus, mode=plan.mode, tolerances=plan.tolerances
    )
    print(format_gate(result, args.gate), file=err, end="")
    return EXIT_OK if result.passed else EXIT_ERROR


def format_gate(result: GateResult, baseline: str) -> str:
    """One line for a passed gate; for a failed one, a line per regression."""
    if result.passed:
        return f"jevex: gate passed against {baseline} ({len(result.checks)} checks)\n"
    lines = [f"jevex: error: gate failed against {baseline}:"]
    lines += [f"  {c.describe()}" for c in result.regressions]
    return "\n".join(lines) + "\n"


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
    _check_out(out)
    try:
        return pack_generators(Path(pack)) if pack else []
    except (FileNotFoundError, InvalidGeneratorError) as exc:
        raise CliError(str(exc)) from exc


async def _learn(args: argparse.Namespace, jev: JevClient | None, llm: LLM | None) -> PackDiff:
    schemas = [load_schema(s) for s in args.schema]
    pack = await asyncio.to_thread(_learn_files, Path(args.out), args.pack)
    if jev is None and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise CliError("TYPESAFE_API_KEY is not set (jevex learn tests generators with Jev)")
    run = None
    if args.max_spend is not None or args.max_jev_spend is not None:
        run = RunBudget(max_spend=args.max_spend, max_jev_spend=args.max_jev_spend, period="run")
    generator_llm = llm or load_llm(args.llm)
    try:
        diff = await _compile(args, schemas, pack, jev, generator_llm, Budgets(run=run))
    finally:
        # The extractor doesn't close an LLM it's given; an adapter built here is ours.
        close = getattr(generator_llm, "aclose", None) if llm is None else None
        if close is not None:
            await close()
    try:
        await asyncio.to_thread(diff.write, Path(args.out))
    except OSError as exc:
        raise CliError(str(exc)) from exc
    return diff


async def _compile(
    args: argparse.Namespace,
    schemas: list[type[BaseModel]],
    pack: list[GeneratorSpec],
    jev: JevClient | None,
    generator_llm: LLM,
    budgets: Budgets,
) -> PackDiff:
    try:
        extractor = Extractor(
            schemas,
            jev=jev,
            store=args.store,
            generator_llm=generator_llm,
            learn_threshold=args.learn_threshold,
            budgets=budgets,
        )
    except (ValueError, UnsupportedFieldError) as exc:
        raise CliError(str(exc)) from exc
    async with extractor:
        try:
            return await extractor.compile_pack(pack)
        except StoreError as exc:
            raise CliError(f"store: {exc}") from exc
        except PackError as exc:
            raise CliError(f"pack: {exc}") from exc
        except JevError as exc:
            raise CliError(f"Jev: {exc}") from exc


_MEMORY_STORES = (":memory:", "sqlite://", "sqlite://:memory:", "sqlite:///:memory:")


def _store_file(url: str) -> Path | None:
    """The SQLite file a store URL names (``None`` for in-memory and other backends)."""
    if url in _MEMORY_STORES:
        return None
    if url.startswith("sqlite:///"):
        return Path(url.removeprefix("sqlite:///"))
    return None if "://" in url else Path(url)


async def _open_store(url: str, *, must_exist: bool) -> Store:
    """Open a store; reading one (``must_exist``) never creates an empty database."""
    path = _store_file(url)
    if must_exist and path is not None and not await asyncio.to_thread(path.is_file):
        raise CliError(f"no such store: {url}")
    try:
        return await asyncio.to_thread(open_store, url)
    except StoreError as exc:
        raise CliError(f"store: {exc}") from exc


def _is_store(source: str) -> bool:
    return "://" in source or Path(source).is_file()


async def _load_pack(source: str) -> Pack:
    try:
        return await asyncio.to_thread(load_pack, source)
    except PackError as exc:
        raise CliError(str(exc)) from exc


def _check_out(out: Path) -> None:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise CliError(f"--out {out} already exists and isn't an empty directory")


async def _pack_export(args: argparse.Namespace) -> str:
    try:
        PackManifest(name=args.name, version=args.version)
    except ValidationError as exc:
        problems = "; ".join(f"--{e['loc'][0]}: {e['msg']}" for e in exc.errors())
        raise CliError(problems) from exc
    out = Path(args.out)
    await asyncio.to_thread(_check_out, out)
    store = await _open_store(args.store, must_exist=True)
    try:
        pack = await export_pack(
            store,
            args.name,
            args.version,
            schemas=args.schema or (),
            locales=args.locale or (),
            description=args.description,
            examples=args.examples,
        )
    except StoreError as exc:
        raise CliError(f"store: {exc}") from exc
    finally:
        await store.aclose()
    try:
        await asyncio.to_thread(pack.write, out)
    except OSError as exc:
        raise CliError(str(exc)) from exc
    return f"exported {_describe(pack)} to {out}\n"


async def _pack_import(args: argparse.Namespace) -> str:
    pack = await _load_pack(args.source)
    store = await _open_store(args.store, must_exist=False)
    try:
        await import_pack(pack, store, examples=not args.no_examples)
    except StoreError as exc:
        raise CliError(f"store: {exc}") from exc
    finally:
        await store.aclose()
    return f"imported {_describe(pack, examples=not args.no_examples)} into {args.store}\n"


async def _diff_side(source: str, examples: bool) -> tuple[Pack, bool]:
    """A pack, or a store's state as one (``True``: it's a store)."""
    if not await asyncio.to_thread(_is_store, source):
        return await _load_pack(source), False
    store = await _open_store(source, must_exist=True)
    try:
        return await export_pack(store, "store", "-", examples=examples), True
    except StoreError as exc:
        raise CliError(f"store: {exc}") from exc
    finally:
        await store.aclose()


async def _pack_diff(args: argparse.Namespace) -> PackChanges:
    old, old_store = await _diff_side(args.old, args.examples)
    new, new_store = await _diff_side(args.new, args.examples)
    return diff_packs(old, new, manifest=not (old_store or new_store), examples=args.examples)


def _describe(pack: Pack, *, examples: bool = True) -> str:
    m = pack.manifest
    parts = [
        _count(len(pack.generators), "generator"),
        _count(len(pack.key_mappings), "key mapping"),
    ]
    if examples:
        parts.append(_count(len(pack.examples), "example"))
    parts.append(_count(len(m.disables), "disable"))
    return f"pack {m.name} {m.version} ({', '.join(parts)})"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def format_changes(changes: PackChanges) -> str:
    """A plain-text summary of :func:`~jevex.packs.diff_packs`: ``+`` added, ``-`` removed,
    ``~`` changed (shown as it is in the new pack)."""
    if changes.empty:
        return "no changes\n"
    lines = [f"manifest: {key} {old!r} -> {new!r}" for key, (old, new) in changes.manifest.items()]

    def section(title: str, rows: list[tuple[str, str]]) -> None:
        if rows:
            lines.append(f"{title}:")
            lines.extend(f"  {sign} {text}" for sign, text in rows)

    g = changes.generators
    section(
        "generators",
        [
            (sign, f"{s.id}  {s.field}  {s.match.regex}")
            for sign, items in (("+", g.added), ("-", g.removed), ("~", g.changed))
            for s in items
        ],
    )
    k = changes.key_mappings
    section(
        "key mappings",
        [
            (sign, f"{m.fingerprint}  {m.schema_name}  {m.path} -> {m.field or 'none'}")
            for sign, items in (("+", k.added), ("-", k.removed), ("~", k.changed))
            for m in items
        ],
    )
    e = changes.examples
    section(
        "examples",
        [
            (sign, f"{x.id}  {x.field}  {json.dumps(x.value, ensure_ascii=False)}")
            for sign, items in (("+", e.added), ("-", e.removed), ("~", e.changed))
            for x in items
        ],
    )
    d = changes.disables
    section("disables", [("+", i) for i in d.added] + [("-", i) for i in d.removed])
    return "\n".join(lines) + "\n"


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
        f"precision: {pct(s['precision'])}  recall: {pct(s['recall'])}  "
        f"accuracy: {pct(s['accuracy'])}",
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


def _testsite_build(args: argparse.Namespace) -> str:
    try:
        manifest = build(args.seed, args.out, waves=args.waves)
    except ValueError as exc:  # an --out that isn't a build
        raise CliError(str(exc)) from exc
    except ImportError as exc:  # no Pillow for the scanned and infographic pages
        raise CliError(f"{exc}, or leave scanned and infographic out of --waves") from exc
    except OSError as exc:
        raise CliError(str(exc)) from exc
    pages: list[dict[str, Any]] = manifest["pages"]
    lines = [f"built {_count(len(pages), 'page')} for seed {args.seed} in {args.out}"]
    for number, wave in enumerate(manifest["waves"], start=1):
        count = sum(1 for p in pages if p["wave"] == number)
        lines.append(f"  wave {number}: {', '.join(wave)} ({_count(count, 'page')})")
    lines.append(f"digest {manifest['digest']}")
    return "\n".join(lines) + "\n"


def _stats_loader(args: argparse.Namespace) -> Loader:
    if args.store is not None:
        return store_loader(args.store, budget_usd=args.budget)
    return replay_loader(args.replay, budget_usd=args.budget)


def _stats_export(args: argparse.Namespace) -> str:
    """Write ``--svg``'s chart to ``--out``; return a line saying where it went."""
    try:
        stats = _stats_loader(args)()
        svg = chart_svg(
            stats,
            args.svg,
            args.x or stats.default_axis(),
            standalone=True,
            animate=not args.static,
        )
        Path(args.out).write_text(svg + "\n", encoding="utf-8")
    except (StoreError, OSError, ValueError) as exc:
        raise CliError(str(exc)) from exc
    return f"wrote {args.out}\n"


def _stats_serve(args: argparse.Namespace, stdout: TextIO) -> None:
    load = _stats_loader(args)
    try:
        load()  # fail now, not on the first request, if the source can't be read
        httpd = stats_server(load, args.host, args.port)
    except (StoreError, OSError, ValueError) as exc:
        raise CliError(str(exc)) from exc
    with httpd:
        host, port = httpd.server_address[:2]
        print(f"serving stats at http://{host}:{port}/stats/ (Ctrl-C to stop)", file=stdout)
        stdout.flush()
        with contextlib.suppress(KeyboardInterrupt):  # Ctrl-C is how serving ends
            httpd.serve_forever()


async def _check_store(url: str) -> None:
    """Open and close the store, so a bad URL fails before the service starts."""
    store = await _open_store(url, must_exist=False)
    await store.aclose()


def _serve(
    args: argparse.Namespace, jev: JevClient | None, llm: LLM | None, stdout: TextIO
) -> None:
    try:
        import uvicorn

        from jevex.server import Service, create_app
    except ImportError as exc:
        raise CliError(
            f"jevex serve needs the server extra: pip install 'jevex[server]' ({exc})"
        ) from exc
    schemas = [load_schema(s) for s in args.schema]
    if jev is None and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise CliError("TYPESAFE_API_KEY is not set (jevex needs a Jev API key to extract)")
    if args.store is not None:
        asyncio.run(_check_store(args.store))
    run = None
    if args.max_spend is not None or args.max_jev_spend is not None:
        period: Period = args.period
        run = RunBudget(max_spend=args.max_spend, max_jev_spend=args.max_jev_spend, period=period)
    model = (llm or load_llm(args.llm)) if args.llm else None
    try:
        service = Service(
            schemas,
            jev=jev,
            store=args.store,
            budgets=Budgets(run=run),
            threshold=args.threshold,
            extraction_llm=model,
            generator_llm=model,
            close_llms=llm is None,  # an adapter built here is the service's to close
            stats=args.stats,
            stats_budget_usd=args.stats_budget,
        )
    except (ValueError, UnsupportedFieldError) as exc:
        raise CliError(str(exc)) from exc
    names = ", ".join(service.schema_names)
    stats = f"; stats at http://{args.host}:{args.port}/stats/" if args.stats else ""
    print(
        f"serving {names} at http://{args.host}:{args.port}/extract{stats} (Ctrl-C to stop)",
        file=stdout,
    )
    stdout.flush()
    uvicorn.run(create_app(service), host=args.host, port=args.port)


def _testsite_serve(args: argparse.Namespace, stdout: TextIO) -> None:
    try:
        httpd = server(args.dir, args.host, args.port)
    except (ValueError, OSError) as exc:
        raise CliError(str(exc)) from exc
    with httpd:
        host, port = httpd.server_address[:2]
        print(f"serving {args.dir} at http://{host}:{port}/ (Ctrl-C to stop)", file=stdout)
        stdout.flush()
        with contextlib.suppress(KeyboardInterrupt):  # Ctrl-C is how serving ends
            httpd.serve_forever()


def _waves(text: str) -> Waves:
    try:
        return parse_waves(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _usd(text: str) -> float:
    value = float(text)
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {text}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {text}")
    return value


def _rate(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value >= 0):
        raise argparse.ArgumentTypeError(f"must be a number, 0 or more, not {text}")
    return value


def _field_drop(text: str) -> float | None:
    return None if text.strip().lower() == "none" else _probability(text)


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
    evaluate.add_argument(
        "--concurrency",
        type=int,
        help=f"Documents at a time (default {EVAL_CONCURRENCY}; a replay runs one at a time)",
    )
    evaluate.add_argument("--json", action="store_true", help="Print the full report as JSON")
    evaluate.add_argument(
        "--replay",
        action="store_true",
        help="Learning curves: start from an empty store, run the documents one at a time "
        "in the corpus's order, and report per batch (CSV on stdout unless --csv/--html)",
    )
    evaluate.add_argument(
        "--batch-size",
        type=_positive_int,
        metavar="N",
        help=f"Documents per replay batch (default {REPLAY_BATCH_SIZE})",
    )
    evaluate.add_argument("--csv", metavar="PATH", help="Write the replay's batches as CSV")
    evaluate.add_argument(
        "--html",
        metavar="PATH",
        help="Write the replay's chart of accuracy, cost and LLM calls per document",
    )
    evaluate.add_argument(
        "--llm",
        metavar="PROVIDER[:MODEL]",
        help="Use this LLM for the fallback (and, with --replay, for learning): "
        f"{', '.join(LLM_PROVIDERS)} (anthropic defaults to {ANTHROPIC_MODEL}). "
        "Without it the run uses Jev alone",
    )
    baseline = evaluate.add_mutually_exclusive_group()
    baseline.add_argument(
        "--gate",
        metavar="BASELINE",
        help="Fail (exit 1) if accuracy drops or the LLM-call rate rises beyond the "
        "baseline's tolerances; the result goes to stderr",
    )
    baseline.add_argument(
        "--write-baseline",
        metavar="PATH",
        help="Write this run as the baseline for --gate (an existing one keeps its "
        "tolerances unless the flags below change them)",
    )
    defaults = GateTolerances()
    for dest, (flag, _), help_text in (
        (
            "max_accuracy_drop",
            TOLERANCE_FLAGS["max_accuracy_drop"],
            "Overall accuracy may fall this much, as a fraction "
            f"(default {defaults.accuracy_drop})",
        ),
        (
            "max_field_drop",
            TOLERANCE_FLAGS["max_field_drop"],
            "Each field's accuracy may fall this much; 'none' checks only the overall "
            f"(default {defaults.field_accuracy_drop})",
        ),
        (
            "max_llm_rise",
            TOLERANCE_FLAGS["max_llm_rise"],
            f"LLM calls per document may rise this much (default {defaults.llm_rate_rise})",
        ),
    ):
        evaluate.add_argument(
            flag,
            dest=dest,
            type={"max_field_drop": _field_drop, "max_llm_rise": _rate}.get(dest, _probability),
            default=argparse.SUPPRESS,
            metavar="X",
            help=f"{help_text}; with --gate it overrides the baseline's",
        )

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

    pack = commands.add_parser(
        "pack",
        help="Export, import and diff packs of learned state",
        description="Packs are reviewable YAML: a manifest.yaml plus generators, key "
        "mappings and optional examples. A pack is a directory or an installed pack's name "
        "(jevex.packs entry point).",
    )
    pack_commands = pack.add_subparsers(dest="pack_command", metavar="<pack command>")
    pack_commands.required = True
    export = pack_commands.add_parser(
        "export",
        help="Write a store's learned state as a pack",
        description="Write a store's enabled generators, key mappings and disable list "
        "(and, with --examples, its verified examples) to a new pack directory.",
    )
    export.add_argument("--store", required=True, help="The store, e.g. sqlite:///jevex.db")
    export.add_argument("--out", required=True, help="A new or empty directory for the pack")
    export.add_argument("--name", required=True, help="The pack's name, e.g. automotive-uk")
    export.add_argument("--version", default="0.1.0", help="The pack's version (default 0.1.0)")
    export.add_argument("--description", help="A line saying what the pack is for")
    export.add_argument(
        "--schema",
        action="append",
        metavar="NAME",
        help="Export only this schema's state, by schema name (repeatable; default: all)",
    )
    export.add_argument(
        "--locale",
        action="append",
        help="A locale the pack is for (repeatable; default: the generators' locales)",
    )
    export.add_argument("--examples", action="store_true", help="Include verified examples")
    imp = pack_commands.add_parser(
        "import",
        help="Copy a pack into a store",
        description="Copy a pack's generators, key mappings, examples and disables into a "
        "store, its top layer: they replace entries with the same keys.",
    )
    imp.add_argument("source", help="A pack directory or an installed pack's name")
    imp.add_argument("--store", required=True, help="The store, e.g. sqlite:///jevex.db")
    imp.add_argument("--no-examples", action="store_true", help="Leave the pack's examples out")
    diff = pack_commands.add_parser(
        "diff",
        help="Show what changes from one pack (or store) to another",
        description="Compare two packs. Either side may be a store URL (or SQLite file), "
        "which is compared as its exported state.",
    )
    diff.add_argument("old", help="A pack directory, an installed pack's name or a store")
    diff.add_argument("new", help="A pack directory, an installed pack's name or a store")
    diff.add_argument("--examples", action="store_true", help="Compare verified examples too")
    diff.add_argument("--json", action="store_true", help="Print the changes as JSON")

    stats = commands.add_parser(
        "stats",
        help="Serve the stats UI, or export a chart as SVG",
        description="Show learning curves, the resolution mix, cost, generators, fields and "
        "budget events from a store (live: every page load reads it again) or a replay's "
        "CSV (jevex eval --replay --csv). 'export' writes one chart as an SVG file, "
        "animated unless --static, for READMEs.",
    )
    stats.add_argument(
        "action",
        nargs="?",
        choices=("serve", "export"),
        default="serve",
        help="serve (the default) or export",
    )
    source = stats.add_mutually_exclusive_group(required=True)
    source.add_argument("--store", help="A store URL, e.g. sqlite:///jevex.db")
    source.add_argument("--replay", metavar="CSV", help="A replay's CSV")
    stats.add_argument("--budget", type=_usd, help="Draw a budget line at this many USD")
    stats.add_argument("--host", default="127.0.0.1", help="Address (default 127.0.0.1)")
    stats.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT})"
    )
    stats.add_argument("--svg", choices=CHART_VIEWS, help="export: the chart to write")
    stats.add_argument("--out", metavar="PATH", help="export: where to write the SVG")
    stats.add_argument(
        "--x",
        choices=("docs", "time"),
        help="export: the x-axis (default: time for a store, documents for a replay)",
    )
    stats.add_argument("--static", action="store_true", help="export: no animation")

    serve = commands.add_parser(
        "serve",
        help="Run the extraction microservice (server extra)",
        description="Serve POST /extract (a document, its bytes base64, and a schema "
        "name), /health and /metrics (Prometheus) over HTTP, until interrupted. Needs "
        "TYPESAFE_API_KEY and the server extra.",
    )
    serve.add_argument(
        "--schema",
        action="append",
        required=True,
        metavar="MODULE:CLASS",
        help="A Pydantic model to serve, as module:Class or path/to/file.py:Class "
        "(repeatable); requests name it by its class name",
    )
    serve.add_argument(
        "--store",
        help="A store URL for learned state, the spend ledger and stats, e.g. "
        "sqlite:///jevex.db (default: in memory, lost on exit)",
    )
    serve.add_argument(
        "--llm",
        metavar="PROVIDER[:MODEL]",
        help="Use this LLM for the fallback and for learning generators: "
        f"{', '.join(LLM_PROVIDERS)} (anthropic defaults to {ANTHROPIC_MODEL}). "
        "Without it the service uses Jev alone",
    )
    serve.add_argument(
        "--threshold",
        type=_probability,
        default=0.0,
        help="Confidence (0–1) below which values become null (default 0: keep everything)",
    )
    serve.add_argument("--max-spend", type=_usd, help="LLM spend allowed per --period, in USD")
    serve.add_argument("--max-jev-spend", type=_usd, help="Jev spend allowed per --period, in USD")
    serve.add_argument(
        "--period",
        choices=get_args(Period),
        default="day",
        help="The period the spend limits cover (default day)",
    )
    serve.add_argument(
        "--stats",
        action="store_true",
        help="Also serve the stats UI at /stats/ (needs --store; it shows URLs and spend)",
    )
    serve.add_argument(
        "--stats-budget", type=_usd, metavar="USD", help="Draw the stats UI's budget line"
    )
    serve.add_argument("--host", default="127.0.0.1", help="Address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=SERVE_PORT, help=f"Port (default {SERVE_PORT})")

    site = commands.add_parser(
        "testsite",
        help="Build and serve the synthetic test site",
        description="A seeded synthetic car site with exact ground truth (truth.json) for "
        "jevex eval. The same seed always builds the same site.",
    )
    site_commands = site.add_subparsers(dest="testsite_command", metavar="<testsite command>")
    site_commands.required = True
    site_build = site_commands.add_parser(
        "build",
        help="Write the site and its truth.json",
        description="Write the site's pages, index.html and truth.json. Pages are listed "
        "wave by wave, so a replay meets each wave's template families together. An "
        "earlier build in --out is replaced; any other non-empty directory is refused.",
    )
    site_build.add_argument("--seed", type=int, default=42, help="The dataset seed (default 42)")
    site_build.add_argument(
        "--out", default=BUILD_DIR, help=f"Where to write (default {BUILD_DIR})"
    )
    site_build.add_argument(
        "--waves",
        type=_waves,
        default=DEFAULT_WAVES,
        metavar="SCHEDULE",
        help="Template families per wave: waves split by ';', families by ','. Families "
        f"left out aren't built (default {format_waves(DEFAULT_WAVES)!r})",
    )
    site_serve = site_commands.add_parser(
        "serve",
        help="Serve a built site over HTTP",
        description="Serve a site jevex testsite build wrote, until interrupted.",
    )
    site_serve.add_argument("--dir", default=BUILD_DIR, help=f"The build (default {BUILD_DIR})")
    site_serve.add_argument("--host", default="127.0.0.1", help="Address (default 127.0.0.1)")
    site_serve.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
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
    including ``jevex eval`` runs (and replays) where any document failed or ``--gate``
    found a regression (the report is still printed).
    2: a usage error or no command.
    ``jev``, ``llm`` (the ``--llm`` of ``jevex learn``, ``jevex eval`` and ``jevex serve``),
    ``out`` and ``err`` are injectable for tests.
    """
    stdout: TextIO = out or sys.stdout
    stderr: TextIO = err or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(stderr)
        return EXIT_USAGE
    if args.command == "serve":
        if args.stats and args.store is None:
            parser.error("--stats reads the store: give --store too")
        if args.stats_budget is not None and not args.stats:
            parser.error("--stats-budget only applies with --stats")
        if args.max_spend is None and args.max_jev_spend is None and args.period != "day":
            parser.error("--period only applies with --max-spend or --max-jev-spend")
    if args.command == "stats":
        export_only = [
            flag
            for flag, value in (("--svg", args.svg), ("--out", args.out), ("--x", args.x))
            if value is not None
        ] + (["--static"] if args.static else [])
        if args.action == "export" and (args.svg is None or args.out is None):
            parser.error("jevex stats export needs --svg and --out")
        if args.action == "serve" and export_only:
            parser.error(f"{', '.join(export_only)} only apply to jevex stats export")
    if args.command == "eval" and args.replay and args.concurrency is not None:
        parser.error("--replay runs one document at a time: leave out --concurrency")
    if args.command == "eval" and not args.replay:
        given = [
            flag
            for flag, value in (
                ("--batch-size", args.batch_size),
                ("--csv", args.csv),
                ("--html", args.html),
            )
            if value is not None
        ]
        if given:
            parser.error(f"{', '.join(given)} only apply with --replay")
    if args.command == "eval" and args.gate is None and args.write_baseline is None:
        given = [flag for dest, (flag, _) in TOLERANCE_FLAGS.items() if hasattr(args, dest)]
        if given:
            parser.error(f"{', '.join(given)} only apply with --gate or --write-baseline")
    try:
        plan = _plan_gate(args) if args.command == "eval" else None
        if args.command == "eval" and args.replay:
            if args.batch_size is None:
                args.batch_size = REPLAY_BATCH_SIZE
            replayed = asyncio.run(_replay(args, jev, llm))
            written = _write_replay(args, replayed)
            if args.json:
                json.dump(replayed.to_dict(), stdout, indent=2, ensure_ascii=False)
                stdout.write("\n")
            elif not written:
                stdout.write(replayed.to_csv())
            else:
                stdout.write(format_report(replayed.report) + "\n".join(written) + "\n")
            for doc in replayed.failed:
                print(f"jevex: error: {doc.path}: {doc.error}", file=stderr)
            gated = EXIT_OK if plan is None else _finish_gate(args, plan, replayed.report, stderr)
            return EXIT_ERROR if replayed.failed else gated
        if args.command == "eval":
            report = asyncio.run(_eval(args, jev, llm))
            if args.json:
                json.dump(report.to_dict(), stdout, indent=2, ensure_ascii=False)
                stdout.write("\n")
            else:
                stdout.write(format_report(report))
            for doc in report.failed:
                print(f"jevex: error: {doc.path}: {doc.error}", file=stderr)
            gated = EXIT_OK if plan is None else _finish_gate(args, plan, report, stderr)
            # A run with failed documents isn't a clean measurement, even though it's scored.
            return EXIT_ERROR if report.failed else gated
        if args.command == "pack":
            if args.pack_command == "diff":
                changes = asyncio.run(_pack_diff(args))
                if args.json:
                    json.dump(changes.model_dump(mode="json"), stdout, indent=2, ensure_ascii=False)
                    stdout.write("\n")
                else:
                    stdout.write(format_changes(changes))
                return EXIT_OK
            run = _pack_export if args.pack_command == "export" else _pack_import
            stdout.write(asyncio.run(run(args)))
            return EXIT_OK
        if args.command == "stats":
            if args.action == "export":
                stdout.write(_stats_export(args))
            else:
                _stats_serve(args, stdout)
            return EXIT_OK
        if args.command == "serve":
            _serve(args, jev, llm, stdout)
            return EXIT_OK
        if args.command == "testsite":
            if args.testsite_command == "build":
                stdout.write(_testsite_build(args))
            else:
                _testsite_serve(args, stdout)
            return EXIT_OK
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

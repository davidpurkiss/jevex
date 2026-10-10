"""Benchmark runs: every system over every corpus, saved and scored (``docs/benchmarks.md``
› *Protocol*, #63).

:func:`run_benchmarks` reads the pinned config (:class:`~jevex.benchmarks.BenchmarkConfig`),
builds or locates each corpus and checks it against its lock, then runs each system in
:data:`SYSTEMS` over it into a results directory:

- ``manifest.json``: the :class:`RunManifest`, rewritten after every step: the pinned
  config (models and the prices the run is costed at), the jevex commit, the hash of
  ``uv.lock``, each step's outcome and the run's spend;
- ``<system>/<corpus>.jsonl``: one :class:`~jevex.baselines.ResultRow` per document, the
  same format for jevex and the baselines, so ``jevex eval --results`` rescores any of them
  (for a corpus published only in ``aggregate``, it's kept in the work directory instead);
- ``<system>/<corpus>.score.json``: its :class:`SystemScore`, the numbers the results page
  reports (:mod:`jevex.bench_report`);
- ``jevex-cold/<corpus>.replay.csv``: jevex's learning curve over the corpus
  (:meth:`~jevex.replay.ReplayReport.to_csv`).

A run is capped by the process spend caps and shared ledger (``JEVEX_JEV_MAX_COST_USD``,
``JEVEX_LLM_MAX_COST_USD``, ``JEVEX_SPEND_LEDGER``), which every process it starts adds
to. A live run needs all three, with caps adding up to no more than the config's
``budget_usd``; once a cap is reached the run stops after the step that reached it, keeping
what was saved. A dry run (fake Jev and LLMs) needs none.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevex._spend import ledger_path, ledger_total
from jevex.baselines import (
    LLMBaseline,
    ResultRow,
    baseline_instructions,
    load_prompt,
    pinned_llm,
    read_inputs,
    read_results,
    result_row,
    run_baseline,
    schema_specs,
    score_results,
    write_inputs,
)
from jevex.benchmarks import (
    BenchmarkConfig,
    CorpusLock,
    CorpusSpec,
    Interval,
    bootstrap_interval,
    verify_lock,
)
from jevex.eval import evaluate, load_corpus, match_records, schema_tolerances, score_value
from jevex.extractor import Extractor, default_pipeline
from jevex.jev import JevBudgetExceededError, JevError, process_cap
from jevex.llm import LLMBudgetExceededError, LLMError, process_llm_cap
from jevex.logs import get_logger
from jevex.replay import replay
from jevex.resolve import EntityStage, MultiEntity
from jevex.store import open_store
from jevex.testsite import build
from jevex.testsite.waves import DEFAULT_WAVES, parse_waves

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jevex.benchmarks import PinnedModel
    from jevex.eval import CorpusItem, DocumentRun, EvalReport, Tolerance
    from jevex.jev import JevClient
    from jevex.llm import LLM
    from jevex.pipeline import Pipeline

log = get_logger(__name__)

SYSTEMS: tuple[str, ...] = (
    "jevex-cold",
    "jevex-warm",
    "jevex-no-llm",
    "llm-fast",
    "llm-strong",
    "llm-gemini",
    "scrapegraphai",
    "crawl4ai",
)
"""Every system compared (``docs/benchmarks.md`` › *Systems compared*), in run order.
``jevex-warm`` is measured on a second pass after ``jevex-cold``'s, so it needs it."""

LLM_BASELINES = ("llm-fast", "llm-strong", "llm-gemini")
"""LLM-only systems: the config's ``models.baseline_fast``, ``_strong`` and ``_gemini``."""

TOOL_SCRIPTS = {
    "scrapegraphai": "baselines/scrapegraphai_baseline.py",
    "crawl4ai": "baselines/crawl4ai_baseline.py",
}
"""Open-source tools → their uv script, relative to the config. Each runs in its own
environment with ``jevex baseline run``'s arguments, on the fast baseline's model."""

PROMPT = "baselines/prompt-v1.md"
"""The baselines' instructions, relative to the config."""

MANIFEST = "manifest.json"

StepStatus = Literal["done", "failed", "skipped", "stopped"]


class BenchRunError(Exception):
    """A benchmark run can't start as asked (an unknown system, missing caps), or one of
    its steps failed (recorded on the step, and the run goes on)."""


# --- the manifest ----------------------------------------------------------------------


class StepRecord(BaseModel):
    """How one system did on one corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str
    corpus: str
    status: StepStatus
    """``done``; ``failed`` (``message`` says why; the run went on); ``skipped`` (nothing to
    run, e.g. no model pinned); ``stopped`` (a spend cap was reached during it: what it saved
    is incomplete, and the run stopped)."""
    message: str | None = None
    seconds: float = 0.0


class RunManifest(BaseModel):
    """What a results directory was produced from (``manifest.json``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    started: datetime
    commit: str | None
    """The jevex commit (``git rev-parse HEAD``), ``None`` outside a checkout."""
    dirty: bool
    """Whether the checkout had uncommitted changes."""
    uv_lock: str | None
    """SHA-256 of the checkout's ``uv.lock``: every library version the run used."""
    dry_run: bool
    """Fake Jev and LLM answers: the numbers check the plumbing and mean nothing."""
    config: BenchmarkConfig
    """The pinned setup, prices included, so old results keep their costs."""
    systems: tuple[str, ...]
    corpora: tuple[str, ...]
    steps: tuple[StepRecord, ...] = ()
    jev_spend: float = 0.0
    """USD the run added to the spend ledger, all processes together."""
    llm_spend: float = 0.0

    @property
    def complete(self) -> bool:
        """Every step ran to the end (``done`` or ``skipped``)."""
        return all(s.status in ("done", "skipped") for s in self.steps)


def read_manifest(directory: str | Path) -> RunManifest:
    """A results directory's manifest. Raises ``ValueError`` naming the file."""
    path = Path(directory) / MANIFEST
    try:
        return RunManifest.model_validate_json(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"can't read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{path} isn't a benchmark manifest: {exc}") from exc


# --- scores ----------------------------------------------------------------------------


class SystemScore(BaseModel):
    """One system's numbers on one corpus: means over documents with bootstrap intervals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str
    corpus: str
    documents: int
    failed: int
    """Documents the system failed on, scored as all missing."""
    accuracy: Interval | None
    """Field accuracy, correct over scored values (``None``: nothing to score)."""
    precision: float | None
    recall: float | None
    complete_records: Interval | None
    """Expected records found with every field right."""
    cost_per_document: Interval
    """USD: Jev, LLM calls and what the learner spent learning from the document."""
    llm_calls_per_document: Interval
    latency_p50: float | None
    """Seconds per document that didn't fail."""
    latency_p95: float | None
    resolution_mix: dict[str, int]
    """Values found per method."""
    fields: dict[str, float | None] = Field(default_factory=dict[str, float | None])
    """Each ``Schema.field``'s accuracy."""


def score_system(
    system: str,
    corpus: str,
    directory: str | Path,
    rows: Sequence[ResultRow],
    schemas: Sequence[type[BaseModel]],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> SystemScore:
    """Score ``rows`` against the corpus in ``directory`` (:func:`~jevex.baselines.score_results`,
    as ``jevex eval --results`` does) and summarise them, with seeded bootstrap intervals.

    Raises what :func:`~jevex.baselines.score_results` raises.
    """
    specs = schema_specs(schemas)
    report = score_results(directory, rows, specs)
    items = load_corpus(directory)
    tolerances = schema_tolerances(specs)
    learning = {r.path: r.learning_cost for r in rows}
    root = Path(directory)

    def interval(values: Sequence[float], weights: Sequence[float] | None = None) -> Interval:
        return bootstrap_interval(
            values, weights=weights, samples=samples, confidence=confidence, seed=seed
        )

    def ratio(hits: Sequence[int], totals: Sequence[int]) -> Interval | None:
        if not any(totals):
            return None
        return interval([h / t if t else 0.0 for h, t in zip(hits, totals, strict=True)], totals)

    correct = [sum(s.correct for s in d.fields.values()) for d in report.documents]
    scored = [
        sum(s.correct + s.wrong + s.missing + s.spurious for s in d.fields.values())
        for d in report.documents
    ]
    complete = [
        _complete_records(item, run, tolerances)
        for item, run in zip(items, report.documents, strict=True)
    ]
    costs = [
        run.cost + learning.get(item.path.relative_to(root).as_posix(), 0.0)
        for item, run in zip(items, report.documents, strict=True)
    ]
    summary = report.summary()
    overall = report.overall()
    return SystemScore(
        system=system,
        corpus=corpus,
        documents=len(report.documents),
        failed=len(report.failed),
        accuracy=ratio(correct, scored),
        precision=overall.precision,
        recall=overall.recall,
        complete_records=ratio(complete, [len(i.records) for i in items]),
        cost_per_document=interval(costs),
        llm_calls_per_document=interval([float(d.llm_calls) for d in report.documents]),
        latency_p50=summary["latency_p50"],
        latency_p95=summary["latency_p95"],
        resolution_mix=summary["resolution_mix"],
        fields={name: s.accuracy for name, s in report.field_scores().items()},
    )


def _complete_records(
    item: CorpusItem, run: DocumentRun, tolerances: dict[str, dict[str, Tolerance]]
) -> int:
    """Expected records paired (as scoring pairs them) with a found one that gets every
    field right: nothing wrong, missing or spurious."""
    if run.error is not None:
        return 0
    own = tolerances[item.schema]
    complete = 0
    for exp, rec in match_records(item.records, run.records.get(item.schema, []), own):
        if exp is None or rec is None:
            continue
        scores = [score_value(exp.values.get(n), rec["values"].get(n), t) for n, t in own.items()]
        complete += all(s.wrong == s.missing == s.spurious == 0 for s in scores)
    return complete


def read_score(path: str | Path) -> SystemScore:
    """A ``.score.json`` file. Raises ``ValueError`` naming the file."""
    try:
        return SystemScore.model_validate_json(Path(path).read_bytes())
    except OSError as exc:
        raise ValueError(f"can't read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{path} isn't a system score: {exc}") from exc


def score_path(results: str | Path, system: str, corpus: str) -> Path:
    return Path(results) / system / f"{corpus}.score.json"


def replay_path(results: str | Path, corpus: str) -> Path:
    """jevex's learning curve over ``corpus``: the cold pass's replay, as CSV."""
    return Path(results) / "jevex-cold" / f"{corpus}.replay.csv"


# --- corpora ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedCorpus:
    """A corpus ready to run: where it is (checked against its lock), its schemas and the
    pipeline jevex runs on it."""

    spec: CorpusSpec
    directory: Path
    schemas: tuple[type[BaseModel], ...]
    pipeline: Pipeline


def prepare_corpus(spec: CorpusSpec, config_dir: Path, work: Path) -> PreparedCorpus:
    """Build the test site from its seed (into ``work``) or find a kept corpus (its
    ``path``, relative to the config, or the directory its ``env`` names), and check it
    against its lock.

    Raises :class:`BenchRunError` for an unset ``env``, a schema or pipeline that doesn't
    load, and :class:`~jevex.benchmarks.CorpusLockError` for a corpus that doesn't match.
    """
    # Imported here: jevex.cli imports jevex, which imports this module.
    from jevex.cli import CliError, load_pipeline, load_schema

    try:
        schemas = tuple(load_schema(s) for s in spec.schemas)
        pipeline = load_pipeline(spec.pipeline) if spec.pipeline else default_pipeline()
    except CliError as exc:
        raise BenchRunError(f"corpus {spec.name}: {exc}") from exc
    if spec.entities == "multi":
        pipeline = pipeline.replace("entities", EntityStage(MultiEntity()))
    if spec.kind == "testsite":
        assert spec.seed is not None  # CorpusSpec checks it
        directory = work / "corpora" / spec.name
        build(spec.seed, directory, waves=parse_waves(spec.waves) if spec.waves else DEFAULT_WAVES)
    elif spec.path is not None:
        directory = config_dir / spec.path
    else:
        assert spec.env is not None  # CorpusSpec checks it
        value = os.environ.get(spec.env, "").strip()
        if not value:
            raise BenchRunError(
                f"corpus {spec.name}: set {spec.env} to its directory (docs/benchmarks.md)"
            )
        directory = Path(value)
    verify_lock(directory, CorpusLock.load(config_dir / spec.lock))
    return PreparedCorpus(spec, directory, schemas, pipeline)


# --- running ---------------------------------------------------------------------------


def check_budget_env(config: BenchmarkConfig) -> None:
    """Refuse a live run that isn't hard-capped at the config's budget: both process caps
    set, adding up to no more than ``budget_usd``, and a shared spend ledger, so the tools'
    processes count against the same caps. Raises :class:`BenchRunError` saying what's
    missing."""
    caps = {name: os.environ.get(name, "").strip() for name in _CAPS}
    missing = [name for name, value in caps.items() if not value]
    if ledger_path() is None:
        missing.append("JEVEX_SPEND_LEDGER")
    if missing:
        raise BenchRunError(
            f"a live run needs {', '.join(missing)} set: caps adding up to at most the "
            f"config's ${config.budget_usd:g}, and a ledger file every process adds to"
        )
    try:
        total = sum(float(v) for v in caps.values())
    except ValueError as exc:
        raise BenchRunError(f"the spend caps must be numbers of US dollars: {exc}") from exc
    if total > config.budget_usd + 1e-9:
        raise BenchRunError(
            f"the spend caps add up to ${total:g}, more than the config's "
            f"${config.budget_usd:g} budget"
        )


_CAPS = ("JEVEX_JEV_MAX_COST_USD", "JEVEX_LLM_MAX_COST_USD")


def _cap_reached() -> str | None:
    """Which process cap is spent, if any (``None`` when none is or none is set)."""
    for name, capped in (("Jev", process_cap()), ("LLM", process_llm_cap())):
        if capped is not None and capped[1] >= capped[0]:
            return f"the {name} spend cap was reached (${capped[1]:.4f} of ${capped[0]:.2f})"
    return None


def _spent() -> tuple[float, float]:
    ledger = ledger_path()
    if ledger is None:
        return 0.0, 0.0
    return ledger_total(ledger, "jev", JevError), ledger_total(ledger, "llm", LLMError)


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def _checkout(config_dir: Path) -> tuple[str | None, bool, str | None]:
    """The commit, whether the checkout is dirty, and ``uv.lock``'s SHA-256."""
    top = _git(config_dir, "rev-parse", "--show-toplevel")
    if top is None:
        return None, False, None
    lock = Path(top) / "uv.lock"
    digest = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
    status = _git(config_dir, "status", "--porcelain")
    return _git(config_dir, "rev-parse", "HEAD"), bool(status), digest


async def run_benchmarks(
    config_path: str | Path,
    out: str | Path,
    *,
    systems: Sequence[str] = SYSTEMS,
    corpora: Sequence[str] | None = None,
    work: str | Path | None = None,
    jev: JevClient | None = None,
    llm: Callable[[PinnedModel], LLM] = pinned_llm,
    tool_args: Sequence[str] = (),
    dry_run: bool = False,
) -> RunManifest:
    """Run ``systems`` over ``corpora`` (default: every corpus in the config) into the
    results directory ``out``, which must not exist; return the final manifest.

    ``work`` holds what isn't published: the built test site, the baselines' prepared
    inputs and an ``aggregate`` corpus's results (default ``<out>.work``). ``jev`` is the
    Jev client every jevex system shares (default: each extractor's own, from
    ``TYPESAFE_API_KEY``), ``llm`` builds an LLM for a pinned model (default
    :func:`~jevex.baselines.pinned_llm`), ``tool_args`` go to every tool script (``--fake``
    for a dry run). ``dry_run`` marks the manifest: fake answers, no spend checks.

    A step that fails is recorded and the run goes on; one during which a spend cap is
    reached stops the run. Raises :class:`BenchRunError` before anything runs for an
    unknown system or corpus, ``jevex-warm`` without ``jevex-cold``, a live run without
    its caps (:func:`check_budget_env`), or an existing ``out``.
    """
    config_file = Path(config_path)
    config = await asyncio.to_thread(BenchmarkConfig.load, config_file)
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise BenchRunError(f"unknown systems {unknown}; choose from {list(SYSTEMS)}")
    if "jevex-warm" in systems and "jevex-cold" not in systems:
        raise BenchRunError("jevex-warm is measured after jevex-cold's pass: run both")
    names = [c.name for c in config.corpora] if corpora is None else list(corpora)
    for name in names:
        try:
            config.corpus(name)
        except KeyError:
            raise BenchRunError(f"{config_file} has no corpus {name!r}") from None
    if not dry_run:
        check_budget_env(config)
    results = Path(out)
    if await asyncio.to_thread(results.exists):
        raise BenchRunError(f"{results} exists; a run writes a new results directory")
    commit, dirty, uv_lock = await asyncio.to_thread(_checkout, config_file.parent)
    run = _Run(
        config=config,
        config_file=config_file,
        out=results,
        work=Path(work) if work is not None else results.with_name(results.name + ".work"),
        jev=jev,
        llm=llm,
        tool_args=tuple(tool_args),
        manifest=RunManifest(
            started=datetime.now(UTC),
            commit=commit,
            dirty=dirty,
            uv_lock=uv_lock,
            dry_run=dry_run,
            config=config,
            systems=tuple(s for s in SYSTEMS if s in systems),
            corpora=tuple(names),
        ),
    )
    await run.all()
    return run.manifest


@dataclass
class _Run:
    config: BenchmarkConfig
    config_file: Path
    out: Path
    work: Path
    jev: JevClient | None
    llm: Callable[[PinnedModel], LLM]
    tool_args: tuple[str, ...]
    manifest: RunManifest

    @property
    def config_dir(self) -> Path:
        return self.config_file.parent

    async def all(self) -> None:
        await asyncio.to_thread(self.out.mkdir, parents=True)
        before = await asyncio.to_thread(_spent)
        try:
            for name in self.manifest.corpora:
                if not await self.corpus(self.config.corpus(name)):
                    break
        finally:
            after = await asyncio.to_thread(_spent)
            self.manifest = self.manifest.model_copy(
                update={"jev_spend": after[0] - before[0], "llm_spend": after[1] - before[1]}
            )
            await asyncio.to_thread(self.write_manifest)

    def write_manifest(self) -> None:
        (self.out / MANIFEST).write_text(self.manifest.model_dump_json(indent=2) + "\n")

    async def record(self, step: StepRecord) -> None:
        log.info(
            "%s on %s: %s%s",
            step.system,
            step.corpus,
            step.status,
            f" ({step.message})" if step.message else "",
        )
        self.manifest = self.manifest.model_copy(update={"steps": (*self.manifest.steps, step)})
        await asyncio.to_thread(self.write_manifest)

    async def corpus(self, spec: CorpusSpec) -> bool:
        """Every system on one corpus. ``False`` once a spend cap stops the run."""
        try:
            corpus = await asyncio.to_thread(prepare_corpus, spec, self.config_dir, self.work)
        except (BenchRunError, ValueError) as exc:  # CorpusLockError is a ValueError
            for system in self.manifest.systems:
                await self.record(
                    StepRecord(system=system, corpus=spec.name, status="failed", message=str(exc))
                )
            return True
        systems = list(self.manifest.systems)
        for system in systems:
            if system == "jevex-warm":
                continue  # measured in jevex-cold's step
            start = time.perf_counter()
            status: StepStatus = "done"
            message: str | None = None
            try:
                ran = await self.step(system, corpus)
                if not ran:
                    status, message = "skipped", f"the config pins no model for {system}"
            except (JevBudgetExceededError, LLMBudgetExceededError) as exc:
                status, message = "stopped", str(exc)
            except Exception as exc:  # recorded on the step; the next system runs
                status, message = "failed", f"{type(exc).__name__}: {exc}"
            reached = _cap_reached()
            if reached is not None and status != "stopped":
                status, message = "stopped", reached
            seconds = time.perf_counter() - start
            done = (
                [system, "jevex-warm"]
                if system == "jevex-cold" and "jevex-warm" in systems
                else [system]
            )
            for name in done:
                await self.record(
                    StepRecord(
                        system=name,
                        corpus=spec.name,
                        status=status,
                        message=message,
                        seconds=seconds,
                    )
                )
            if status == "stopped":
                return False
        return True

    async def step(self, system: str, corpus: PreparedCorpus) -> bool:
        """Run ``system`` on ``corpus`` and score it; ``False`` if there was nothing to run."""
        if system == "jevex-cold":
            await self.jevex_learning(corpus, warm="jevex-warm" in self.manifest.systems)
        elif system == "jevex-no-llm":
            async with Extractor(
                list(corpus.schemas),
                jev=self.jev,
                pipeline=corpus.pipeline,
                community_packs=False,
            ) as extractor:
                report = await evaluate(
                    extractor, load_corpus(corpus.directory), concurrency=self.config.concurrency
                )
            await self.save(system, corpus, _rows(report, corpus.directory))
        elif system in LLM_BASELINES:
            models = self.config.models
            pinned = {
                "llm-fast": models.baseline_fast,
                "llm-strong": models.baseline_strong,
                "llm-gemini": models.baseline_gemini,
            }[system]
            if pinned is None:
                return False
            await self.llm_baseline(system, corpus, pinned)
        else:
            await self.tool(system, corpus)
        return True

    async def jevex_learning(self, corpus: PreparedCorpus, *, warm: bool) -> None:
        """jevex (cold): from an empty store, one document at a time, learning as it goes
        (:func:`~jevex.replay.replay`). jevex (warm): a second pass at the pinned
        concurrency over what the first learned, with no more learning."""
        items = load_corpus(corpus.directory)
        models = self.config.models
        extraction, generator = self.llm(models.extraction), self.llm(models.generator)
        store = open_store(":memory:")
        try:
            async with Extractor(
                list(corpus.schemas),
                jev=self.jev,
                pipeline=corpus.pipeline,
                store=store,
                extraction_llm=extraction,
                generator_llm=generator,
                community_packs=False,
            ) as extractor:
                replayed = await replay(extractor, items)
            rows = [
                result_row(run, corpus.directory, learning_cost=spend.cost)
                for run, spend in zip(replayed.report.documents, replayed.learning, strict=True)
            ]
            await asyncio.to_thread(
                replay_path(self.out, corpus.spec.name).parent.mkdir, parents=True, exist_ok=True
            )
            await asyncio.to_thread(
                replay_path(self.out, corpus.spec.name).write_text, replayed.to_csv()
            )
            await self.save("jevex-cold", corpus, rows)
            if warm:
                async with Extractor(
                    list(corpus.schemas),
                    jev=self.jev,
                    pipeline=corpus.pipeline,
                    store=store,
                    extraction_llm=extraction,
                    community_packs=False,
                ) as extractor:
                    report = await evaluate(extractor, items, concurrency=self.config.concurrency)
                await self.save("jevex-warm", corpus, _rows(report, corpus.directory))
        finally:
            await store.aclose()
            for model in {id(extraction): extraction, id(generator): generator}.values():
                close = getattr(model, "aclose", None)
                if close is not None:
                    await close()

    async def inputs(self, corpus: PreparedCorpus) -> Path:
        """The baselines' prepared inputs for ``corpus``, written on first use."""
        path = self.work / "inputs" / f"{corpus.spec.name}.jsonl"
        if not await asyncio.to_thread(path.exists):
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            await write_inputs(corpus.directory, path, pipeline=corpus.pipeline)
        return path

    async def llm_baseline(self, system: str, corpus: PreparedCorpus, pinned: PinnedModel) -> None:
        specs = schema_specs(corpus.schemas)
        template = load_prompt(self.config_dir / PROMPT)
        inputs = await asyncio.to_thread(read_inputs, await self.inputs(corpus))
        model = self.llm(pinned)
        out = self.rows_path(system, corpus)
        await asyncio.to_thread(out.parent.mkdir, parents=True, exist_ok=True)
        try:
            baseline = LLMBaseline(
                model, specs, baseline_instructions(template, specs), name=system
            )
            rows = await run_baseline(
                baseline,
                corpus.directory,
                out,
                inputs=inputs,
                concurrency=self.config.concurrency,
            )
        finally:
            close = getattr(model, "aclose", None)
            if close is not None:
                await close()
        await self.score(system, corpus, rows)

    async def tool(self, system: str, corpus: PreparedCorpus) -> None:
        """An open-source tool's script, in its own environment (``uv run --script``)."""
        out = self.rows_path(system, corpus)
        await asyncio.to_thread(out.parent.mkdir, parents=True, exist_ok=True)
        command = [
            "uv",
            "run",
            "--script",
            str(self.config_dir / TOOL_SCRIPTS[system]),
            str(corpus.directory),
            *(f"--schema={s}" for s in corpus.spec.schemas),
            "--model",
            "fast",
            "--inputs",
            str(await self.inputs(corpus)),
            "--out",
            str(out),
            "--config",
            str(self.config_file),
            "--prompt",
            str(self.config_dir / PROMPT),
            "--concurrency",
            str(self.config.concurrency),
            *self.tool_args,
        ]
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, err = await process.communicate()
        if process.returncode != 0:
            tail = " ".join(err.decode(errors="replace").strip().splitlines()[-3:])
            raise BenchRunError(f"{TOOL_SCRIPTS[system]} exited {process.returncode}: {tail}")
        await self.score(system, corpus, await asyncio.to_thread(read_results, out))

    def rows_path(self, system: str, corpus: PreparedCorpus) -> Path:
        base = self.out if corpus.spec.publish == "full" else self.work / "results"
        return base / system / f"{corpus.spec.name}.jsonl"

    async def save(self, system: str, corpus: PreparedCorpus, rows: list[ResultRow]) -> None:
        out = self.rows_path(system, corpus)
        text = "".join(row.model_dump_json() + "\n" for row in rows)
        await asyncio.to_thread(out.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(out.write_text, text, encoding="utf-8")
        await self.score(system, corpus, rows)

    async def score(self, system: str, corpus: PreparedCorpus, rows: Sequence[ResultRow]) -> None:
        settings = self.config.bootstrap
        score = await asyncio.to_thread(
            score_system,
            system,
            corpus.spec.name,
            corpus.directory,
            rows,
            corpus.schemas,
            samples=settings.samples,
            confidence=settings.confidence,
            seed=self.config.seed,
        )
        path = score_path(self.out, system, corpus.spec.name)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, score.model_dump_json(indent=2) + "\n")


def _rows(report: EvalReport, corpus: Path) -> list[ResultRow]:
    return [result_row(run, corpus) for run in report.documents]


__all__ = [
    "LLM_BASELINES",
    "MANIFEST",
    "SYSTEMS",
    "TOOL_SCRIPTS",
    "BenchRunError",
    "PreparedCorpus",
    "RunManifest",
    "StepRecord",
    "SystemScore",
    "check_budget_env",
    "prepare_corpus",
    "read_manifest",
    "read_score",
    "replay_path",
    "run_benchmarks",
    "score_path",
    "score_system",
]

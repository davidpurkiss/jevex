"""The benchmark results page (``docs/benchmarks.md`` › *Protocol*, step 4; #63).

:func:`results_page` turns a results directory (:func:`~jevex.bench.run_benchmarks`) into
a markdown page and its charts: the cost comparison across corpora, a table per corpus
(accuracy, complete records, cost, LLM calls, latency), accuracy against cost, jevex's
learning curve and resolution mix (the stats UI's animated SVGs, :mod:`jevex.stats`), the
pinned setup and the steps that didn't finish. It reads only the manifest, the scores
and the replay curves, never the corpora, so anyone can rebuild the page from published
results. :func:`write_results_page` writes it (``benchmarks/report.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jevex.bench import (
    SYSTEMS,
    read_manifest,
    read_score,
    replay_path,
    score_path,
)
from jevex.stats.charts import CostPoint, accuracy_cost_svg, chart_svg, pct
from jevex.stats.data import from_replay_csv

if TYPE_CHECKING:
    from jevex.bench import RunManifest, SystemScore
    from jevex.benchmarks import Interval, PinnedModel

NAMES = {
    "jevex-cold": "jevex (cold)",
    "jevex-warm": "jevex (warm)",
    "jevex-no-llm": "jevex (no LLM)",
    "llm-fast": "LLM-only, fast",
    "llm-strong": "LLM-only, strong",
    "llm-gemini": "LLM-only, Gemini",
    "scrapegraphai": "ScrapeGraphAI",
    "crawl4ai": "Crawl4AI",
}
"""Each system's name on the page, as ``docs/benchmarks.md`` names it."""

REFERENCE = "llm-fast"
"""The system costs are compared with: LLM-only on the model jevex's fallback uses."""

CHARTS_DIR = "benchmark-results"
"""Where the charts go, relative to the page."""


@dataclass(frozen=True)
class ResultsPage:
    """The page's markdown and its charts (file name → SVG), which it links as
    ``<charts_dir>/<name>``."""

    markdown: str
    charts: dict[str, str] = field(default_factory=dict[str, str])


def results_page(results: str | Path, *, charts_dir: str = CHARTS_DIR) -> ResultsPage:
    """The results page for the results directory ``results``.

    Raises ``ValueError`` for a directory without a readable manifest or with a score
    file that can't be read.
    """
    root = Path(results)
    manifest = read_manifest(root)
    scores = {
        (s, c): read_score(path)
        for c in manifest.corpora
        for s in manifest.systems
        if (path := score_path(root, s, c)).is_file()
    }
    charts: dict[str, str] = {}
    lines = _header(manifest)
    lines += _cost_section(manifest, scores)
    for corpus in manifest.corpora:
        lines += _corpus_section(root, manifest, corpus, scores, charts, charts_dir)
    lines += _unfinished(manifest)
    lines += _setup(manifest)
    return ResultsPage("\n".join(lines).rstrip() + "\n", charts)


def write_results_page(
    results: str | Path, out: str | Path, *, charts_dir: str = CHARTS_DIR
) -> list[Path]:
    """Write :func:`results_page` to ``out`` and its charts beside it, in ``charts_dir``;
    returns every file written."""
    page = results_page(results, charts_dir=charts_dir)
    target = Path(out)
    written = [target]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page.markdown, encoding="utf-8")
    if page.charts:
        (target.parent / charts_dir).mkdir(exist_ok=True)
    for name, svg in page.charts.items():
        path = target.parent / charts_dir / name
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    return written


# --- sections --------------------------------------------------------------------------


def _header(manifest: RunManifest) -> list[str]:
    config = manifest.config
    commit = f"`{manifest.commit[:12]}`" if manifest.commit else "an unknown commit"
    if manifest.dirty:
        commit += " with uncommitted changes"
    lock = f", `uv.lock` `{manifest.uv_lock[:12]}`" if manifest.uv_lock else ""
    lines = ["# Benchmark results", ""]
    if manifest.dry_run:
        lines += [
            "> **Dry run.** Jev and the LLMs gave fake answers, so these numbers only show",
            "> that every step runs. They say nothing about jevex or the baselines.",
            "",
        ]
    lines += [
        f"Run on {manifest.started:%Y-%m-%d} at jevex {commit}{lock}, by the agreed "
        "[methodology](benchmarks.md). It spent "
        f"{_usd(manifest.jev_spend)} on Jev and {_usd(manifest.llm_spend)} on LLMs, against a "
        f"{_usd(config.budget_usd)} cap.",
        "",
        f"Every number is a mean over documents, with a "
        f"{config.bootstrap.confidence:.0%} bootstrap interval in brackets "
        f"({config.bootstrap.samples} resamples, seed {config.seed}). Accuracy is correct "
        "values over every value expected or found; a complete record has every field right. "
        "jevex's cost includes what its learner spent.",
        "",
        "To rerun: `uv run benchmarks/run.py --all`, then `uv run benchmarks/report.py "
        "<results>` (see `benchmarks/README.md`). The benchmarks run on demand, not on every "
        "release: when a change should move the numbers, and before they're quoted anywhere "
        "new.",
        "",
    ]
    return lines


def _cost_section(manifest: RunManifest, scores: dict[tuple[str, str], SystemScore]) -> list[str]:
    corpora = manifest.corpora
    lines = [
        "## Cost",
        "",
        "USD per 1,000 documents, and how that compares with "
        f"{NAMES[REFERENCE]} (one call per document on the model jevex's fallback uses).",
        "",
        "| System | " + " | ".join(corpora) + " |",
        "| --- |" + " ---: |" * len(corpora),
    ]
    for system in _ordered(manifest.systems):
        cells: list[str] = []
        for corpus in corpora:
            score = scores.get((system, corpus))
            if score is None:
                cells.append("–")
                continue
            cost = score.cost_per_document.mean
            cell = _usd(cost * 1000)
            reference = scores.get((REFERENCE, corpus))
            if reference is not None and system != REFERENCE:
                cell += f" ({_relative(cost, reference.cost_per_document.mean)})"
            cells.append(cell)
        lines.append(f"| {NAMES[system]} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _relative(cost: float, reference: float) -> str:
    if cost <= 0 or reference <= 0:
        return "–"
    if cost <= reference:
        return f"{reference / cost:.1f}× cheaper"
    return f"{cost / reference:.1f}× the cost"


def _corpus_section(
    root: Path,
    manifest: RunManifest,
    corpus: str,
    scores: dict[tuple[str, str], SystemScore],
    charts: dict[str, str],
    charts_dir: str,
) -> list[str]:
    rows = [(s, scores[(s, corpus)]) for s in _ordered(manifest.systems) if (s, corpus) in scores]
    lines = [f"## {corpus}", ""]
    if not rows:
        return [*lines, "No system finished this corpus.", ""]
    lines += [
        f"{rows[0][1].documents} documents.",
        "",
        "| System | Accuracy | Complete records | Cost per document | LLM calls per document "
        "| Latency p50 / p95 | Failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system, s in rows:
        latency = (
            "–"
            if s.latency_p50 is None or s.latency_p95 is None
            else f"{_seconds(s.latency_p50)} / {_seconds(s.latency_p95)}"
        )
        lines.append(
            f"| {NAMES[system]} | {_share(s.accuracy)} | {_share(s.complete_records)} "
            f"| {_money(s.cost_per_document)} | {s.llm_calls_per_document.mean:.2f} "
            f"| {latency} | {s.failed} |"
        )
    lines += [
        "",
        f"jevex (cold) runs one document at a time, so each learns from the ones before it; "
        f"the others run {manifest.config.concurrency} at a time. A baseline's latency is "
        "its extraction only, since every system reads the same prepared input; jevex's "
        "includes cleaning and layout.",
        "",
    ]
    points = [
        CostPoint(
            NAMES[system],
            s.cost_per_document.mean,
            s.accuracy.mean,
            s.accuracy.low,
            s.accuracy.high,
            jevex=system.startswith("jevex"),
        )
        for system, s in rows
        if s.accuracy is not None
    ]
    name = f"accuracy-cost-{corpus}.svg"
    charts[name] = accuracy_cost_svg(
        points, f"Accuracy against cost: {corpus}", standalone=True, animate=True
    )
    lines += [f"![Accuracy against cost per document on {corpus}]({charts_dir}/{name})", ""]
    curve = replay_path(root, corpus)
    if curve.is_file():
        stats = from_replay_csv(curve.read_text(encoding="utf-8"), source=corpus)
        for view, alt in (
            ("learning", "jevex's learning curve"),
            ("mix", "where jevex's values came from"),
        ):
            name = f"{view}-{corpus}.svg"
            charts[name] = chart_svg(stats, view, standalone=True, animate=True)
            lines += [f"![{alt} on {corpus}, from an empty store]({charts_dir}/{name})", ""]
    return lines


def _unfinished(manifest: RunManifest) -> list[str]:
    steps = [s for s in manifest.steps if s.status != "done"]
    if not steps:
        return []
    lines = [
        "## Steps that didn't finish",
        "",
        "| System | Corpus | Status | Why |",
        "| --- | --- | --- | --- |",
    ]
    for step in steps:
        why = (step.message or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {NAMES[step.system]} | {step.corpus} | {step.status} | {why} |")
    return [*lines, ""]


def _setup(manifest: RunManifest) -> list[str]:
    config = manifest.config
    models = config.models
    roles: list[tuple[str, PinnedModel | None]] = [
        ("Jev", config.jev),
        ("jevex's fallback", models.extraction),
        ("jevex's learner", models.generator),
        (NAMES["llm-fast"] + ", ScrapeGraphAI, Crawl4AI", models.baseline_fast),
        (NAMES["llm-strong"], models.baseline_strong),
        (NAMES["llm-gemini"], models.baseline_gemini),
    ]
    lines = [
        "## The pinned setup",
        "",
        f"Seed {config.seed}; {config.concurrency} documents in flight. Costs are at these "
        "prices, as read on their price date.",
        "",
        "| Role | Model | Input, USD per million tokens | Output | Price date |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for role, pinned in roles:
        if pinned is not None:
            lines.append(
                f"| {role} | `{pinned.model}` | {pinned.input_usd_per_mtok:g} "
                f"| {pinned.output_usd_per_mtok:g} | {pinned.price_date} |"
            )
    return [*lines, ""]


# --- formatting ------------------------------------------------------------------------


def _ordered(systems: tuple[str, ...]) -> list[str]:
    return [s for s in SYSTEMS if s in systems]


def _share(interval: Interval | None) -> str:
    if interval is None:
        return "–"
    return f"{pct(interval.mean)} ({pct(interval.low)}–{pct(interval.high)})"


def _money(interval: Interval) -> str:
    return f"{_usd(interval.mean)} ({_usd(interval.low)}–{_usd(interval.high)})"


def _usd(x: float) -> str:
    """Dollars and cents from $1, else two significant figures: a document can cost a
    thousandth of a cent, and rounding that to $0 would hide the comparison."""
    if x == 0:
        return "$0"
    if abs(x) >= 1:
        return f"${x:,.2f}"
    return f"${x:.{max(2, 1 - math.floor(math.log10(abs(x))))}f}"


def _seconds(x: float) -> str:
    return f"{x:.2f} s" if x < 10 else f"{x:.0f} s"


__all__ = ["CHARTS_DIR", "NAMES", "REFERENCE", "ResultsPage", "results_page", "write_results_page"]

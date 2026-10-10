import csv
import io
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jevex.bench import MANIFEST, RunManifest, StepRecord, SystemScore, replay_path, score_path
from jevex.bench_report import results_page, write_results_page
from jevex.benchmarks import BenchmarkConfig, Interval
from jevex.replay import CSV_COLUMNS

ROOT = Path(__file__).parent.parent
CONFIG = BenchmarkConfig.load(ROOT / "benchmarks" / "config.yaml")


def manifest(**fields: object) -> RunManifest:
    base: dict[str, object] = {
        "started": datetime(2026, 10, 10, 16, 0, tzinfo=UTC),
        "commit": "4c55a45beb3e0409572e6445ebc32f38cfac64be",
        "dirty": False,
        "uv_lock": "92cb5596e4e9e2c357106714ec3992bbdaf9e97e568628040b623a5966b6d854",
        "dry_run": False,
        "config": CONFIG,
        "systems": ("jevex-cold", "jevex-warm", "llm-fast", "crawl4ai"),
        "corpora": ("books", "spec-sheets"),
        "steps": (
            StepRecord(system="jevex-cold", corpus="books", status="done"),
            StepRecord(system="jevex-warm", corpus="books", status="done"),
            StepRecord(system="llm-fast", corpus="books", status="done"),
            StepRecord(
                system="crawl4ai", corpus="books", status="failed", message="exited 1: no | key"
            ),
            StepRecord(
                system="jevex-cold",
                corpus="spec-sheets",
                status="stopped",
                message="the LLM spend cap was reached ($23.0000 of $23.00)",
            ),
        ),
        "jev_spend": 0.42,
        "llm_spend": 3.5,
    }
    return RunManifest.model_validate(base | fields)


def score(system: str, accuracy: float, cost: float, **fields: object) -> SystemScore:
    base: dict[str, object] = {
        "system": system,
        "corpus": "books",
        "documents": 200,
        "failed": 0,
        "accuracy": Interval(accuracy, accuracy - 0.02, accuracy + 0.01),
        "precision": accuracy,
        "recall": accuracy,
        "complete_records": Interval(0.8, 0.75, 0.85),
        "cost_per_document": Interval(cost, cost * 0.9, cost * 1.1),
        "llm_calls_per_document": Interval(0.25, 0.2, 0.3),
        "latency_p50": 1.234,
        "latency_p95": 12.5,
        "resolution_mix": {"jev": 900, "llm": 100},
    }
    return SystemScore.model_validate(base | fields)


def write_results(root: Path, run: RunManifest) -> Path:
    root.mkdir(parents=True)
    (root / MANIFEST).write_text(run.model_dump_json())
    for s in (
        score("jevex-cold", 0.9, 0.0004),
        score("jevex-warm", 0.92, 0.0001, latency_p50=None, latency_p95=None),
        score("llm-fast", 0.95, 0.002, llm_calls_per_document=Interval(1.0, 1.0, 1.0)),
    ):
        path = score_path(root, s.system, s.corpus)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(s.model_dump_json())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, restval=0, lineterminator="\n")
    writer.writeheader()
    for batch, llm in ((1, 0.9), (2, 0.2)):
        writer.writerow(
            {"batch": batch, "documents": batch * 10, "size": 10, "waves": "", "accuracy": 0.9}
            | {"llm_calls_per_document": llm, "jev_cost_per_document": 0.0001}
        )
    replay_path(root, "books").write_text(buffer.getvalue())
    return root


def test_the_page_reports_every_corpus_cost_and_setup(tmp_path: Path) -> None:
    page = results_page(write_results(tmp_path / "results", manifest()))
    lines = page.markdown.splitlines()
    assert lines[:3] == [
        "# Benchmark results",
        "",
        (
            "Run on 2026-10-10 at jevex `4c55a45beb3e`, `uv.lock` `92cb5596e4e9`, by the agreed "
            "[methodology](benchmarks.md). It spent $0.42 on Jev and $3.50 on LLMs, against a "
            "$25.00 cap."
        ),
    ]
    assert "Dry run" not in page.markdown
    cost = lines.index("## Cost")
    assert lines[cost + 4 : cost + 10] == [
        "| System | books | spec-sheets |",
        "| --- | ---: | ---: |",
        "| jevex (cold) | $0.40 (5.0× cheaper) | – |",
        "| jevex (warm) | $0.10 (20.0× cheaper) | – |",
        "| LLM-only, fast | $2.00 | – |",
        "| Crawl4AI | – | – |",
    ]
    books = lines.index("## books")
    assert lines[books + 2 : books + 9] == [
        "200 documents.",
        "",
        "| System | Accuracy | Complete records | Cost per document | LLM calls per document "
        "| Latency p50 / p95 | Failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| jevex (cold) | 90.0% (88.0%–91.0%) | 80.0% (75.0%–85.0%) "
        "| $0.00040 ($0.00036–$0.00044) | 0.25 | 1.23 s / 12 s | 0 |",
        "| jevex (warm) | 92.0% (90.0%–93.0%) | 80.0% (75.0%–85.0%) "
        "| $0.00010 ($0.000090–$0.00011) | 0.25 | – | 0 |",
        "| LLM-only, fast | 95.0% (93.0%–96.0%) | 80.0% (75.0%–85.0%) "
        "| $0.0020 ($0.0018–$0.0022) | 1.00 | 1.23 s / 12 s | 0 |",
    ]
    assert (
        "![Accuracy against cost per document on books]"
        "(benchmark-results/accuracy-cost-books.svg)" in lines
    )
    assert (
        "![jevex's learning curve on books, from an empty store]"
        "(benchmark-results/learning-books.svg)" in lines
    )
    assert set(page.charts) == {"accuracy-cost-books.svg", "learning-books.svg", "mix-books.svg"}
    spec = lines.index("## spec-sheets")
    assert lines[spec + 2] == "No system finished this corpus."
    unfinished = lines.index("## Steps that didn't finish")
    assert lines[unfinished + 4 : unfinished + 6] == [
        "| Crawl4AI | books | failed | exited 1: no \\| key |",
        "| jevex (cold) | spec-sheets | stopped | the LLM spend cap was reached "
        "($23.0000 of $23.00) |",
    ]
    setup = lines.index("## The pinned setup")
    assert lines[setup + 6] == "| Jev | `jev-1.13.0` | 0.042 | 0 | 2026-10-01 |"
    assert (
        lines[setup + 11] == "| LLM-only, Gemini | `gemini-3.8-flash` | 0.75 | 3.75 | 2026-10-01 |"
    )


def test_a_dry_run_says_so_and_a_dirty_checkout_too(tmp_path: Path) -> None:
    run = manifest(dry_run=True, dirty=True, commit=None, uv_lock=None)
    text = results_page(write_results(tmp_path / "results", run)).markdown
    assert "> **Dry run.** Jev and the LLMs gave fake answers" in text
    assert "Run on 2026-10-10 at jevex an unknown commit with uncommitted changes, by the" in text


def test_the_page_and_its_charts_are_written_beside_each_other(tmp_path: Path) -> None:
    results = write_results(tmp_path / "results", manifest())
    out = tmp_path / "docs" / "benchmarks-results.md"
    written = write_results_page(results, out)
    assert written[0] == out
    assert sorted(p.name for p in written[1:]) == [
        "accuracy-cost-books.svg",
        "learning-books.svg",
        "mix-books.svg",
    ]
    assert all(p.parent == tmp_path / "docs" / "benchmark-results" for p in written[1:])
    assert (
        (tmp_path / "docs" / "benchmark-results" / "learning-books.svg")
        .read_text()
        .startswith("<svg xmlns=")
    )


def test_a_directory_without_results_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="can't read"):
        results_page(tmp_path)


def test_the_report_script_writes_the_page(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util
    import sys

    path = ROOT / "benchmarks" / "report.py"
    spec = importlib.util.spec_from_file_location("_bench_report_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results = write_results(tmp_path / "results", manifest())
    out = tmp_path / "page.md"
    assert module.main([str(results), "--out", str(out)]) == 0
    assert capsys.readouterr().out.startswith(f"wrote {out}\n")
    assert module.main([str(tmp_path / "nope"), "--out", str(out)]) == 1
    assert "report.py: error: can't read" in capsys.readouterr().err

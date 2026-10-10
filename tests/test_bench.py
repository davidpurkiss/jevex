import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import jevex
from jevex.baselines import ResultRow, read_results, score_results
from jevex.bench import (
    SYSTEMS,
    BenchRunError,
    check_budget_env,
    prepare_corpus,
    read_manifest,
    read_score,
    replay_path,
    run_benchmarks,
    score_path,
    score_system,
)
from jevex.benchmarks import (
    BenchmarkConfig,
    CorpusLockError,
    CorpusSpec,
    PinnedModel,
    book_values,
    lock_corpus,
)
from jevex.clean import html_text_of
from jevex.examples.books import Book
from jevex.extractor import Extractor
from jevex.resolve import EntityStage, MultiEntity, SingleEntity
from jevex.testing import FakeJev, FakeLLM
from jevex.testsite import build

ROOT = Path(__file__).parent.parent
BOOKS = Path(__file__).parent / "fixtures" / "books"
PAGES = ("a-light-in-the-attic_1000", "sapiens-a-brief-history-of-humankind_996")
CAPS = ("JEVEX_JEV_MAX_COST_USD", "JEVEX_LLM_MAX_COST_USD")


def load_script(name: str) -> Any:
    """A script in ``benchmarks/`` as a module (it isn't part of the package)."""
    path = ROOT / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_corpus(root: Path) -> Path:
    """Two books.toscrape.com pages, labelled from their markup like the books corpus."""
    (root / "pages").mkdir(parents=True)
    pages: list[dict[str, Any]] = []
    for name in PAGES:
        shutil.copy(BOOKS / f"{name}.html", root / "pages" / f"{name}.html")
        values = book_values(html_text_of((BOOKS / f"{name}.html").read_bytes()))
        pages.append(
            {
                "path": f"pages/{name}.html",
                "schema": "Book",
                "records": [{"entity": "document", "values": values}],
            }
        )
    (root / "truth.json").write_text(json.dumps({"pages": pages}))
    return root


def pinned(model: str) -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "model": model,
        "input_usd_per_mtok": 1.0,
        "output_usd_per_mtok": 5.0,
        "price_date": "2026-10-01",
    }


def write_config(root: Path, *corpora: dict[str, Any], budget: float = 25.0) -> Path:
    """A config in ``root`` with a ``books`` corpus (and any ``corpora``), and the
    baselines' prompt beside it."""
    lock_corpus(make_corpus(root / "books"), "books").write(root / "books.lock")
    (root / "baselines").mkdir(parents=True)
    shutil.copy(ROOT / "benchmarks" / "baselines" / "prompt-v1.md", root / "baselines")
    books = {
        "name": "books",
        "kind": "directory",
        "path": "books",
        "lock": "books.lock",
        "schemas": ["jevex.examples.books:Book"],
        "pipeline": "jevex.examples.books:books_pipeline",
    }
    config = {
        "seed": 42,
        "concurrency": 2,
        "budget_usd": budget,
        "bootstrap": {"samples": 200},
        "jev": pinned("jev-1.13.0") | {"provider": "jev", "input_usd_per_mtok": 0.042},
        "models": {
            "extraction": pinned("fallback-model"),
            "generator": pinned("learner-model"),
            "baseline_fast": pinned("fast-model"),
            "baseline_strong": pinned("strong-model"),
        },
        "corpora": [books, *corpora],
    }
    path = root / "config.yaml"
    path.write_text(json.dumps(config))  # JSON is YAML
    return path


def truth_of(prompt: str) -> dict[str, Any]:
    """The fast baseline gets every book right: its labels, as the records model takes them."""
    for name in PAGES:
        values = book_values(html_text_of((BOOKS / f"{name}.html").read_bytes()))
        if f"# {values['title']}\n" in prompt:
            return {"Book": [values | {"price": float(values["price"])}]}
    return {"Book": []}


def fake_llm(price: float = 0.0) -> Any:
    """LLMs by pinned model: the fast baseline is right, the strong one finds nothing and
    jevex's fallback and learner get empty answers."""

    def build_llm(model: PinnedModel) -> FakeLLM:
        if model.model == "fast-model":
            return FakeLLM(lambda prompt, _: truth_of(prompt), model=model.model, price=(price, 0))
        return FakeLLM(lambda _p, _s: {}, model=model.model)

    return build_llm


async def run(config: Path, out: Path, **kwargs: Any) -> Any:
    options: dict[str, Any] = {"jev": FakeJev().client(), "llm": fake_llm(), "dry_run": True}
    return await run_benchmarks(config, out, **(options | kwargs))


# --- a run -----------------------------------------------------------------------------


async def test_a_run_saves_and_scores_every_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict[str, Any]] = []

    class Recorded(Extractor):
        def __init__(self, schemas: Any, **kwargs: Any) -> None:
            built.append(kwargs)
            super().__init__(schemas, **kwargs)

    monkeypatch.setattr("jevex.bench.Extractor", Recorded)
    config = write_config(tmp_path / "bench")
    out = tmp_path / "results"
    systems = ["jevex-cold", "jevex-warm", "jevex-no-llm", "llm-fast", "llm-strong", "llm-gemini"]
    manifest = await run(config, out, systems=systems)
    # The warm pass reads the store the cold pass learned into, and learns no more.
    cold, warm, no_llm = built
    assert warm["store"] is cold["store"]
    assert cold["generator_llm"] is not None
    assert "generator_llm" not in warm
    assert warm["extraction_llm"] is cold["extraction_llm"]
    assert "store" not in no_llm
    assert "extraction_llm" not in no_llm
    assert [(s.system, s.status, s.message) for s in manifest.steps] == [
        ("jevex-cold", "done", None),
        ("jevex-warm", "done", None),
        ("jevex-no-llm", "done", None),
        ("llm-fast", "done", None),
        ("llm-strong", "done", None),
        ("llm-gemini", "skipped", "the config pins no model for llm-gemini"),
    ]
    assert manifest.complete
    assert manifest.dry_run
    assert manifest.corpora == ("books",)
    assert manifest.config == BenchmarkConfig.load(config)
    assert (manifest.commit, manifest.uv_lock) == (None, None)  # the config isn't in a checkout
    assert read_manifest(out) == manifest
    fast = read_score(score_path(out, "llm-fast", "books"))
    assert fast.documents == 2
    assert fast.accuracy is not None
    assert fast.accuracy.mean == 1.0
    assert fast.complete_records is not None
    assert fast.complete_records.mean == 1.0
    assert fast.llm_calls_per_document.mean == 1.0
    assert fast.resolution_mix == {"llm": 10}
    strong = read_score(score_path(out, "llm-strong", "books"))
    assert strong.accuracy is not None
    assert strong.accuracy.mean == 0.0
    cold = read_score(score_path(out, "jevex-cold", "books"))
    assert cold.cost_per_document.mean > 0  # Jev's estimated tokens, even when faked
    assert replay_path(out, "books").read_text().startswith("batch,documents,")
    warm_score = read_score(score_path(out, "jevex-warm", "books"))
    assert warm_score.documents == 2
    assert len(read_results(out / "jevex-warm" / "books.jsonl")) == 2
    assert not score_path(out, "llm-gemini", "books").exists()
    # Every results file is rescored as jevex eval --results would.
    rows = read_results(out / "jevex-cold" / "books.jsonl")
    assert [r.path for r in rows] == [f"pages/{p}.html" for p in PAGES]
    assert all(r.jev_requests > 0 and r.methods == {} for r in rows)
    report = score_results(tmp_path / "bench" / "books", rows, jevex.schema_specs([Book]))
    assert report.summary()["jev_requests_per_document"] > 0
    assert (tmp_path / "results.work" / "inputs" / "books.jsonl").is_file()


async def test_a_run_stops_when_a_spend_cap_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "spend.ledger"
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "1")
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "0.000001")
    config = write_config(tmp_path / "bench")
    out = tmp_path / "results"
    manifest = await run(
        config, out, systems=["llm-fast", "llm-strong"], llm=fake_llm(1e6), dry_run=False
    )
    step = manifest.steps[-1]
    assert (step.system, step.status) == ("llm-fast", "stopped")
    assert step.message is not None
    assert step.message.startswith("LLM spend cap reached")
    assert len(manifest.steps) == 1  # llm-strong never ran
    assert not manifest.complete
    assert manifest.llm_spend > 0
    assert manifest.llm_spend == pytest.approx(
        sum(float(line.split()[1]) for line in ledger.read_text().splitlines())
    )
    assert read_manifest(out) == manifest


async def test_a_run_stops_when_the_jev_cap_refuses_a_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(tmp_path / "spend.ledger"))
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "0.0000001")
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "1")
    config = write_config(tmp_path / "bench")
    manifest = await run(
        config, tmp_path / "results", systems=["jevex-no-llm", "llm-fast"], dry_run=False
    )
    (step,) = manifest.steps  # llm-fast never ran
    assert (step.system, step.status) == ("jevex-no-llm", "stopped")
    assert step.message is not None
    assert step.message.startswith("Jev spend cap reached")


async def test_a_cap_spent_without_an_error_still_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jevex's fallback stops calling the LLM at its cap rather than failing, so the run
    checks the caps after every step too."""
    ledger = tmp_path / "spend.ledger"
    ledger.write_text("llm 0.5\n")
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "1")
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "0.5")
    config = write_config(tmp_path / "bench")
    manifest = await run(
        config, tmp_path / "results", systems=["jevex-no-llm", "llm-fast"], dry_run=False
    )
    (step,) = manifest.steps
    assert (step.system, step.status) == ("jevex-no-llm", "stopped")
    assert step.message == "the LLM spend cap was reached ($0.5000 of $0.50)"


async def test_a_corpus_that_fails_its_lock_fails_its_steps_and_the_run_goes_on(
    tmp_path: Path,
) -> None:
    other = {
        "name": "other",
        "kind": "directory",
        "path": "other",
        "lock": "books.lock",  # the books' lock, which the other corpus doesn't match
        "schemas": ["jevex.examples.books:Book"],
    }
    config = write_config(tmp_path / "bench", other)
    make_corpus(tmp_path / "bench" / "other")
    (tmp_path / "bench" / "other" / "pages" / f"{PAGES[0]}.html").write_text("changed")
    manifest = await run(config, tmp_path / "results", systems=["llm-fast"])
    assert [(s.corpus, s.status) for s in manifest.steps] == [
        ("books", "done"),
        ("other", "failed"),
    ]
    message = manifest.steps[1].message
    assert message is not None
    assert "doesn't match the 'books' lock" in message


async def test_a_failing_tool_is_recorded_and_a_working_one_scored(tmp_path: Path) -> None:
    config = write_config(tmp_path / "bench")
    scripts = tmp_path / "bench" / "baselines"
    (scripts / "scrapegraphai_baseline.py").write_text(
        "import sys\nprint('no key for the model', file=sys.stderr)\nsys.exit(3)\n"
    )
    # A tool that finds nothing: one empty row per document, where --out says.
    rows = [ResultRow(path=f"pages/{p}.html", seconds=0.1).model_dump_json() for p in PAGES]
    (scripts / "crawl4ai_baseline.py").write_text(
        "import sys\n"
        f"assert sys.argv[-1] == '--fake', sys.argv\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        f"open(out, 'w').write({chr(10).join(rows) + chr(10)!r})\n"
    )
    out = tmp_path / "results"
    manifest = await run(config, out, systems=["scrapegraphai", "crawl4ai"], tool_args=["--fake"])
    failed, done = manifest.steps
    assert (failed.system, failed.status) == ("scrapegraphai", "failed")
    assert failed.message == (
        "BenchRunError: baselines/scrapegraphai_baseline.py exited 3: no key for the model"
    )
    assert (done.system, done.status) == ("crawl4ai", "done")
    score = read_score(score_path(out, "crawl4ai", "books"))
    assert score.accuracy is not None
    assert score.accuracy.mean == 0.0


async def test_an_aggregate_corpus_keeps_its_rows_out_of_the_results(tmp_path: Path) -> None:
    config = write_config(tmp_path / "bench")
    data = json.loads(config.read_text())
    data["corpora"][0]["publish"] = "aggregate"
    config.write_text(json.dumps(data))
    out = tmp_path / "results"
    await run(config, out, systems=["llm-fast"])
    assert not (out / "llm-fast" / "books.jsonl").exists()
    assert (tmp_path / "results.work" / "results" / "llm-fast" / "books.jsonl").is_file()
    assert score_path(out, "llm-fast", "books").is_file()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"systems": ["jevex-hot"]}, "unknown systems \\['jevex-hot'\\]"),
        ({"systems": ["jevex-warm"]}, "jevex-warm is measured after jevex-cold's pass"),
        ({"corpora": ["nope"]}, "has no corpus 'nope'"),
        ({"dry_run": False}, "a live run needs JEVEX_JEV_MAX_COST_USD"),
    ],
)
async def test_a_run_that_cannot_start_writes_nothing(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    config = write_config(tmp_path / "bench")
    with pytest.raises(BenchRunError, match=message):
        await run(config, tmp_path / "results", **kwargs)
    assert not (tmp_path / "results").exists()


async def test_a_run_never_overwrites_results(tmp_path: Path) -> None:
    config = write_config(tmp_path / "bench")
    (tmp_path / "results").mkdir()
    with pytest.raises(BenchRunError, match="exists; a run writes a new results directory"):
        await run(config, tmp_path / "results", systems=["llm-fast"])


def test_every_system_has_a_place() -> None:
    assert SYSTEMS[:2] == ("jevex-cold", "jevex-warm")  # warm is measured after cold


# --- the budget ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "needs JEVEX_JEV_MAX_COST_USD, JEVEX_LLM_MAX_COST_USD, JEVEX_SPEND_LEDGER set"),
        (
            {"JEVEX_JEV_MAX_COST_USD": "1", "JEVEX_LLM_MAX_COST_USD": "1"},
            "needs JEVEX_SPEND_LEDGER",
        ),
        ({"JEVEX_JEV_MAX_COST_USD": "1", "JEVEX_SPEND_LEDGER": "x"}, "needs JEVEX_LLM_MAX"),
        (
            {
                "JEVEX_JEV_MAX_COST_USD": "2",
                "JEVEX_LLM_MAX_COST_USD": "24",
                "JEVEX_SPEND_LEDGER": "x",
            },
            "add up to \\$26, more than the config's \\$25 budget",
        ),
        (
            {
                "JEVEX_JEV_MAX_COST_USD": "two",
                "JEVEX_LLM_MAX_COST_USD": "1",
                "JEVEX_SPEND_LEDGER": "x",
            },
            "must be numbers of US dollars",
        ),
    ],
)
def test_a_live_run_must_be_capped_at_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str], message: str
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = BenchmarkConfig.load(write_config(tmp_path))
    with pytest.raises(BenchRunError, match=message):
        check_budget_env(config)


def test_caps_within_the_budget_are_enough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "2")
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "23")
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(tmp_path / "ledger"))
    check_budget_env(BenchmarkConfig.load(write_config(tmp_path)))


# --- corpora ---------------------------------------------------------------------------


def spec(**fields: Any) -> CorpusSpec:
    base = {
        "name": "c",
        "kind": "directory",
        "lock": "c.lock",
        "schemas": ["jevex.examples.books:Book"],
    }
    return CorpusSpec.model_validate(base | fields)


def test_a_kept_corpus_is_found_by_path_or_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_corpus(make_corpus(tmp_path / "books"), "c").write(tmp_path / "c.lock")
    by_path = prepare_corpus(spec(path="books"), tmp_path, tmp_path / "work")
    assert by_path.directory == tmp_path / "books"
    assert by_path.schemas == (Book,)
    stage = next(s for s in by_path.pipeline if s.name == "entities")
    assert isinstance(stage, EntityStage)
    assert isinstance(stage.resolver, SingleEntity)
    monkeypatch.setenv("BOOKS_DIR", str(tmp_path / "books"))
    multi = prepare_corpus(spec(env="BOOKS_DIR", entities="multi"), tmp_path, tmp_path / "w")
    assert multi.directory == tmp_path / "books"
    stage = next(s for s in multi.pipeline if s.name == "entities")
    assert isinstance(stage, EntityStage)
    assert isinstance(stage.resolver, MultiEntity)


def test_the_test_site_is_built_from_its_seed_and_checked(tmp_path: Path) -> None:
    build(7, tmp_path / "site", waves=[["table"]])
    lock_corpus(tmp_path / "site", "t").write(tmp_path / "t.lock")
    schemas = ["jevex.testsite.schemas:VehicleSpec", "jevex.testsite.schemas:Listing"]
    site = spec(name="t", kind="testsite", seed=7, waves="table", lock="t.lock", schemas=schemas)
    prepared = prepare_corpus(site, tmp_path, tmp_path / "work")
    assert prepared.directory == tmp_path / "work" / "corpora" / "t"
    with pytest.raises(CorpusLockError, match="doesn't match the 't' lock"):
        prepare_corpus(site.model_copy(update={"seed": 8}), tmp_path, tmp_path / "work2")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"env": "JEVEX_TEST_UNSET_DIR"}, "corpus c: set JEVEX_TEST_UNSET_DIR to its directory"),
        (
            {"path": "x", "schemas": ["jevex.examples.books:Nope"]},
            "corpus c: .* no attribute 'Nope'",
        ),
        ({"path": "x", "pipeline": "jevex.examples.books:Book"}, "corpus c: .* not a Pipeline"),
    ],
)
def test_a_corpus_that_cannot_be_prepared(
    tmp_path: Path, fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(BenchRunError, match=message):
        prepare_corpus(spec(**fields), tmp_path, tmp_path / "work")


# --- scores ----------------------------------------------------------------------------


class Pair(BaseModel):
    """Two numbers."""

    a: int
    b: int


def test_a_score_weights_documents_by_what_they_hold(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    pages: list[dict[str, Any]] = [
        {"path": "one.html", "schema": "Pair", "records": [{"values": {"a": 1, "b": 2}}]},
        {
            "path": "two.html",
            "schema": "Pair",
            "records": [{"values": {"a": 3, "b": 4}}, {"values": {"a": 5, "b": 6}}],
        },
    ]
    for page in pages:
        (root / page["path"]).write_text("<p>x</p>")
    (root / "truth.json").write_text(json.dumps({"pages": pages}))
    rows = [
        # Both right, and the learner spent 0.5 learning from it.
        ResultRow(
            path="one.html",
            records={"Pair": [{"entity": "1", "values": {"a": 1, "b": 2}}]},
            seconds=1.0,
            cost=0.25,
            jev_cost=0.25,
            learning_cost=0.5,
            methods={"llm": 2},
        ),
        # One record whole, one with b wrong: 3 of 4 values, 1 of 2 records.
        ResultRow(
            path="two.html",
            records={
                "Pair": [
                    {"entity": "1", "values": {"a": 3, "b": 4}},
                    {"entity": "2", "values": {"a": 5, "b": 7}},
                ]
            },
            seconds=3.0,
            calls=2,
            cost=1.0,
            methods={"generator": 4},
        ),
    ]
    score = score_system("jevex-cold", "pairs", root, rows, [Pair], samples=100)
    assert score.accuracy is not None
    assert score.accuracy.mean == pytest.approx(5 / 6)  # micro: 5 of 6 values
    assert score.complete_records is not None
    assert score.complete_records.mean == pytest.approx(2 / 3)  # 2 of 3 records
    assert score.cost_per_document.mean == pytest.approx((1.0 + 1.0) / 2)  # learning counts
    assert score.llm_calls_per_document.mean == 1.0
    assert (score.latency_p50, score.latency_p95) == (2.0, pytest.approx(2.9))
    assert score.resolution_mix == {"generator": 4, "llm": 2}
    assert score.fields == {"Pair.a": 1.0, "Pair.b": pytest.approx(2 / 3)}
    assert score == score_system("jevex-cold", "pairs", root, rows, [Pair], samples=100)
    failed = score_system(
        "x", "pairs", root, [rows[0], rows[1].model_copy(update={"error": "boom"})], [Pair]
    )
    assert failed.failed == 1
    assert failed.complete_records is not None
    assert failed.complete_records.mean == pytest.approx(1 / 3)


def test_unreadable_scores_and_manifests_are_named(tmp_path: Path) -> None:
    (tmp_path / "bad.score.json").write_text("{}")
    with pytest.raises(ValueError, match=r"bad\.score\.json isn't a system score"):
        read_score(tmp_path / "bad.score.json")
    with pytest.raises(ValueError, match="can't read"):
        read_manifest(tmp_path)


def test_the_public_names_are_exported() -> None:
    for name in ("run_benchmarks", "SystemScore", "RunManifest", "results_page"):
        assert name in jevex.__all__
        assert getattr(jevex, name) is not None


def test_the_runner_script_runs_a_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``benchmarks/run.py --dry-run``: fake answers, nothing charged to a real ledger."""
    ledger = tmp_path / "real.ledger"
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    script = load_script("run")
    config = write_config(tmp_path / "bench")
    out = tmp_path / "results"
    argv = ["--system", "jevex-no-llm", "--config", str(config), "--out", str(out), "--dry-run"]
    assert script.main(argv) == 0
    assert read_manifest(out).dry_run
    assert not ledger.exists()

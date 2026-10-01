import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from jevex import (
    Context,
    DomLocation,
    Extractor,
    Field,
    GeneratorRegistry,
    LearningStoppedError,
    LearnStage,
    Pipeline,
    ReplayReport,
    Statement,
    replay,
)
from jevex.cli import main
from jevex.entities import EntityScope
from jevex.eval import DocumentRun, EvalReport, FieldScore, load_corpus
from jevex.fallback import FallbackStage
from jevex.interfaces import ParsedDocument
from jevex.jev import Choice, ChoiceAnswer, JevBudgetExceededError
from jevex.layout import Component
from jevex.normalise import NormaliseStage
from jevex.replay import CSV_COLUMNS, nice_ticks
from jevex.results import FieldMeta
from jevex.select import CandidateStage, SelectStage
from jevex.store import KeyMapping, open_store
from jevex.testing import FakeJev, FakeLLM
from jevex.testsite import build
from jevex.testsite.schemas import VehicleSpec
from jevex.testsite.waves import parse_waves


class Car(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")


LOC = DomLocation(dom_path="/p")
TEXT = "62 mph takes 9.1 seconds"
VERIFY = "The statement states that the 0-62 mph time (s) is 9.1."
DRAFT = {"regex": r"(\d+(?:\.\d+)?) seconds", "group": 1, "normalise": ["parse_number"]}


def car_corpus(root: Path, waves: list[int]) -> Path:
    """One page per wave entry, each stating the 0-62 time as :data:`TEXT`."""
    pages: list[dict[str, Any]] = []
    for i, wave in enumerate(waves):
        name = f"car-{i}.html"
        (root / name).write_text(f"<p>{TEXT}</p>")
        pages.append(
            {
                "path": name,
                "schema": "Car",
                "wave": wave,
                "records": [{"entity": "doc", "values": {"zero_to_62_s": 9.1}}],
            }
        )
    (root / "truth.json").write_text(json.dumps({"pages": pages}))
    return root


@dataclass
class Categorised:
    """Leaves the page's one statement categorised as the 0-62 time."""

    name: str = "categorise"

    async def run(self, ctx: Context) -> None:
        statement = Statement(id="s1", text=TEXT, kind="sentence", component_id="c1", location=LOC)
        ctx.parsed = ParsedDocument(
            document=ctx.document,
            root=Component(id="root", type="section", location=LOC),
            statements={statement.id: statement},
        )
        run = ctx.schemas["Car"]
        run.scopes = [EntityScope(label="doc", component_ids=["c1"])]
        run.categories[statement.id] = ChoiceAnswer(
            choice="zero_to_62_s", confidence=0.9, probabilities={"zero_to_62_s": 0.9}
        )


def pick_91(q: Choice) -> str:
    return "9.1" if "9.1" in q.options else "none"


def learning_extractor(fallback_llm: FakeLLM, generator_llm: FakeLLM) -> Extractor:
    pipeline = Pipeline(
        [
            Categorised(),
            CandidateStage(registry=GeneratorRegistry()),
            SelectStage(),
            NormaliseStage(),
            FallbackStage(),
            LearnStage(),
        ]
    )
    fake = FakeJev().noul(VERIFY, p=0.95).choice(None, pick_91)
    return Extractor(
        [Car],
        jev=fake.client(),
        pipeline=pipeline,
        extraction_llm=fallback_llm,
        generator_llm=generator_llm,
        community_packs=False,
    )


# --- replay ------------------------------------------------------------------------------


async def test_the_llm_call_rate_falls_once_a_generator_is_learned(tmp_path: Path) -> None:
    corpus = load_corpus(car_corpus(tmp_path, [1, 1, 2, 2, 2]))
    fallback_llm = FakeLLM(lambda _p, _s: {"stated": True, "value": 9.1, "evidence": "9.1"})
    generator_llm = FakeLLM([DRAFT])
    async with learning_extractor(fallback_llm, generator_llm) as extractor:
        result = await replay(extractor, corpus, batch_size=2)
    first, second, third = result.batches()
    # The learner finished between documents 1 and 2: only the first needed the LLM.
    assert (first.llm_calls_per_document, second.llm_calls_per_document) == (0.5, 0)
    assert first.methods == {"generator": 1, "llm": 1}
    assert second.methods == {"generator": 2}
    assert [b.accuracy for b in (first, second, third)] == [1.0, 1.0, 1.0]
    assert [b.generators for b in (first, second, third)] == [1, 1, 1]
    assert [(b.documents, b.size) for b in (first, second, third)] == [(2, 2), (4, 2), (5, 1)]
    assert [b.waves for b in (first, second, third)] == [(1,), (2,), (2,)]
    assert len(fallback_llm.calls) == 1
    assert result.generators == [1, 1, 1, 1, 1]
    assert result.wave_starts() == [(2, 2)]


@dataclass
class Spy:
    """Sets the true value and logs each document it sees."""

    log: list[str]
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        self.log.append(Path(ctx.document.url or "").name)
        for run in ctx.active:
            run.set_field("doc", "zero_to_62_s", FieldMeta(value=9.1, method="jev"))


async def test_documents_run_one_at_a_time_in_order_with_learning_finished_between(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = load_corpus(car_corpus(tmp_path, [1, 1, 1]))
    log: list[str] = []
    extractor = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Spy(log)]))

    async def wait() -> None:
        log.append("learned")

    monkeypatch.setattr(extractor, "wait_for_learning", wait)
    result = await replay(extractor, corpus)
    assert log == ["car-0.html", "learned", "car-1.html", "learned", "car-2.html", "learned"]
    # No store and no learner: nothing is learned, and there's one (short) batch.
    assert result.generators == [0, 0, 0]
    [batch] = result.batches()
    assert (batch.number, batch.documents, batch.size, batch.accuracy) == (1, 3, 3, 1.0)


async def test_a_store_that_isnt_empty_is_refused(tmp_path: Path) -> None:
    corpus = load_corpus(car_corpus(tmp_path, [1]))
    store = open_store(":memory:")
    await store.put_key_mapping(
        KeyMapping(fingerprint="f", schema="Car", path="$.x", field="zero_to_62_s")
    )
    extractor = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([]), store=store)
    with pytest.raises(ValueError, match="empty store"):
        await replay(extractor, corpus)
    await store.aclose()


async def test_a_disabled_generator_counts_as_learned_state(tmp_path: Path) -> None:
    corpus = load_corpus(car_corpus(tmp_path, [1]))
    store = open_store(":memory:")
    await store.set_generator_enabled("pack-gen", False)
    extractor = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([]), store=store)
    with pytest.raises(ValueError, match="empty store"):
        await replay(extractor, corpus)
    await store.aclose()


async def test_settings_and_schemas_are_checked(tmp_path: Path) -> None:
    corpus = load_corpus(car_corpus(tmp_path, [1]))
    extractor = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([]))
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        await replay(extractor, corpus, batch_size=0)
    other = Extractor([VehicleSpec], jev=FakeJev().client(), pipeline=Pipeline([]))
    with pytest.raises(ValueError, match="Car"):
        await replay(other, corpus)


@dataclass
class OverBudget:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        raise JevBudgetExceededError("over the cap")


async def test_a_failed_learner_ends_the_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = load_corpus(car_corpus(tmp_path, [1, 1]))
    log: list[str] = []
    extractor = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Spy(log)]))

    async def broken() -> None:
        raise RuntimeError("the learner worker failed") from KeyError("draft")

    monkeypatch.setattr(extractor, "wait_for_learning", broken)
    with pytest.raises(LearningStoppedError, match=r"car-0\.html: 'draft'"):
        await replay(extractor, corpus)
    assert log == ["car-0.html"]


async def test_the_spend_cap_ends_the_replay(tmp_path: Path) -> None:
    corpus = load_corpus(car_corpus(tmp_path, [1, 1]))
    extractor = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([OverBudget()]))
    with pytest.raises(JevBudgetExceededError):
        await replay(extractor, corpus)


@dataclass
class FailSecond:
    seen: list[str] = field(default_factory=list[str])
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        self.seen.append(ctx.document.url or "")
        if len(self.seen) == 2:
            raise RuntimeError("boom")
        for run in ctx.active:
            run.set_field("doc", "zero_to_62_s", FieldMeta(value=9.1, method="jev"))


async def test_a_failing_document_is_scored_missing_and_counted(tmp_path: Path) -> None:
    corpus = load_corpus(car_corpus(tmp_path, [1, 1, 1, 1]))
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([FailSecond()]))
    result = await replay(ex, corpus, batch_size=2)
    first, second = result.batches()
    assert (first.errors, first.accuracy, first.recall) == (1, 0.5, 0.5)
    assert (second.errors, second.accuracy) == (0, 1.0)
    assert [d.path for d in result.failed] == [corpus[1].path.as_posix()]


# --- report ------------------------------------------------------------------------------


def doc_run(correct: int = 1, wrong: int = 0, *, llm_calls: int = 0) -> DocumentRun:
    from collections import Counter

    return DocumentRun(
        path="p",
        schema="Car",
        seconds=0.1,
        jev_requests=2,
        jev_questions=5,
        jev_cost=0.001,
        llm_calls=llm_calls,
        llm_cost=0.002 * llm_calls,
        methods=Counter({"jev": correct}),
        fields={"Car.zero_to_62_s": FieldScore(correct=correct, wrong=wrong)},
    )


def report(runs: list[DocumentRun], waves: list[int | None], batch_size: int) -> ReplayReport:
    return ReplayReport(
        report=EvalReport(documents=runs),
        batch_size=batch_size,
        waves=waves,
        generators=list(range(len(runs))),
    )


def test_batches_carry_per_document_metrics() -> None:
    runs = [doc_run(llm_calls=2), doc_run(0, 1, llm_calls=1), doc_run(), doc_run()]
    first, second = report(runs, [1, 1, 2, None], 2).batches()
    assert first.llm_calls_per_document == 1.5
    assert first.cost_per_document == pytest.approx(0.001 + 0.003)
    assert first.llm_cost_per_document == pytest.approx(0.003)
    assert first.jev_cost_per_document == pytest.approx(0.001)
    assert (first.accuracy, first.precision, first.recall) == (0.5, 0.5, 0.5)
    assert first.jev_requests_per_document == 2
    assert first.generators == 1
    assert second.generators == 3
    assert second.waves == (2,)  # a page with no wave adds none
    assert second.llm_calls_per_document == 0


def test_wave_starts_skip_pages_without_a_wave() -> None:
    runs = [doc_run() for _ in range(5)]
    assert report(runs, [1, None, 1, 2, 3], 2).wave_starts() == [(3, 2), (4, 3)]
    assert report(runs, [None] * 5, 2).wave_starts() == []


def test_csv_has_a_header_and_one_row_per_batch() -> None:
    runs = [doc_run() for _ in range(5)] + [doc_run(0)]
    text = report(runs, [1, 1, 2, 2, 2, 2], 3).to_csv()
    rows = list(csv.DictReader(io.StringIO(text)))
    assert text.splitlines()[0] == ",".join(CSV_COLUMNS)
    assert [r["documents"] for r in rows] == ["3", "6"]
    assert [r["waves"] for r in rows] == ["1 2", "2"]
    assert rows[0]["accuracy"] == "1.0"
    assert rows[0]["values_jev"] == "3"
    assert rows[0]["values_llm"] == "0"
    # The last page scored nothing (no value expected or found): no accuracy, an empty cell.
    lone = report([doc_run(0)], [None], 1).to_csv()
    assert next(csv.DictReader(io.StringIO(lone)))["accuracy"] == ""


def test_to_dict_has_the_summary_and_batches() -> None:
    data = report([doc_run(), doc_run(0, 1)], [1, 2], 1).to_dict()
    assert data["batch_size"] == 1
    assert data["summary"]["accuracy"] == 0.5
    assert [b["accuracy"] for b in data["batches"]] == [1.0, 0.0]
    json.dumps(data)  # JSON types only


def test_the_html_charts_each_metric_marks_waves_and_inlines_the_data() -> None:
    runs = [doc_run(llm_calls=3), doc_run(llm_calls=1), doc_run(), doc_run()]
    page = report(runs, [1, 1, 2, 2], 1).to_html(title="Replay <seed 42>")
    assert page.startswith("<!doctype html>")
    assert page.count("<svg") == 3
    for title in ("Accuracy", "Cost per document (USD)", "LLM calls per document"):
        assert f"<h2>{title}</h2>" in page
    assert page.count(">wave 2</text>") == 3  # one marker per chart
    assert "Replay &lt;seed 42&gt;" in page
    assert "<seed 42>" not in page
    assert page.count('class="dot"') == 12
    assert "batch 1: documents 1–1\nLLM calls per document: 3.00" in page
    assert page.count("<tr>") == 5  # the header and a row per batch
    assert "<script>" not in page  # nothing runs: it's safe to open from anywhere
    data = page.split('id="replay-data">')[1].split("</script>")[0]
    assert json.loads(data)["batches"][0]["llm_calls_per_document"] == 3


def test_the_inlined_data_cant_close_its_script_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    sneaky = {"x": "</script><script>alert(1)</script>"}

    def to_dict(_self: ReplayReport) -> dict[str, Any]:
        return sneaky

    monkeypatch.setattr(ReplayReport, "to_dict", to_dict)
    page = report([doc_run()], [1], 1).to_html()
    assert "</script><script>" not in page
    data = page.split('id="replay-data">')[1].split("</script>")[0]
    assert json.loads(data) == sneaky


def test_a_batch_without_a_value_breaks_the_line() -> None:
    runs = [doc_run(), doc_run(), doc_run(0), doc_run(), doc_run()]
    page = report(runs, [None] * 5, 1).to_html()
    accuracy = page.split("<svg")[1]
    assert accuracy.count("<polyline") == 2
    assert accuracy.count('class="dot"') == 4


def test_an_empty_replay_still_renders() -> None:
    page = report([], [], 10).to_html()
    assert page.count("<svg") == 3
    assert "<polyline" not in page
    assert report([], [], 10).to_csv() == ",".join(CSV_COLUMNS) + "\n"


@pytest.mark.parametrize(
    ("top", "ticks"),
    [
        (0.0, [0.0, 1.0]),
        (3.0, [0.0, 1.0, 2.0, 3.0]),
        (0.0042, [0.0, 0.002, 0.004, 0.006]),
        (7.0, [0.0, 2.0, 4.0, 6.0, 8.0]),
        (10.0, [0.0, 2.5, 5.0, 7.5, 10.0]),
    ],
)
def test_nice_ticks(top: float, ticks: list[float]) -> None:
    assert nice_ticks(top) == ticks


# --- CLI -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("site")
    build(42, root, waves=parse_waves("table;kv"))
    return root


@dataclass
class Oracle:
    """Extracts exactly the ground truth; the first page of each family also calls the
    extraction LLM, if there is one, to stand in for the fallback before learning."""

    pages: dict[str, dict[str, Any]]
    seen: set[str] = field(default_factory=set[str])
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        page = self.pages[Path(ctx.document.url or "").name]
        if page["family"] not in self.seen and ctx.extraction_llm is not None:
            assert ctx.budget is not None
            await ctx.budget.call_llm(ctx.extraction_llm, "What is it?", Car)
        self.seen.add(page["family"])
        for record in page["records"]:
            for run in ctx.active:
                if not set(record["values"]) <= {f.name for f in run.spec.fields}:
                    continue
                for name, value in record["values"].items():
                    if value is not None:
                        meta = FieldMeta(value=value, method="jev")
                        run.set_field(record["entity"], name, meta)


def oracle(site: Path) -> Oracle:
    manifest = json.loads((site / "truth.json").read_text())
    return Oracle(pages={Path(p["path"]).name: p for p in manifest["pages"]})


SCHEMAS = [
    "--schema",
    "jevex.testsite.schemas:VehicleSpec",
    "--schema",
    "jevex.testsite.schemas:Listing",
]


def run_cli(
    site: Path, monkeypatch: pytest.MonkeyPatch, *argv: str, llm: FakeLLM | None = None
) -> tuple[int, str, str]:
    import jevex.extractor as extractor_module

    monkeypatch.setattr(extractor_module, "DEFAULT_STAGES", (oracle(site),))
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["eval", str(site), *SCHEMAS, *argv], jev=FakeJev().client(), llm=llm, out=out, err=err
    )
    return code, out.getvalue(), err.getvalue()


def test_cli_replay_prints_csv(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out, err = run_cli(site, monkeypatch, "--replay", "--batch-size", "5")
    assert (code, err) == (0, "")
    rows = list(csv.DictReader(io.StringIO(out)))
    pages = len(load_corpus(site))
    assert len(rows) == -(-pages // 5)
    assert rows[-1]["documents"] == str(pages)
    assert {r["accuracy"] for r in rows} == {"1.0"}
    assert {r["llm_calls_per_document"] for r in rows} == {"0.0"}  # no --llm: Jev alone
    assert rows[0]["waves"] == "1"
    assert rows[-1]["waves"] == "2"


def test_cli_replay_defaults_to_batches_of_ten(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, out, _ = run_cli(site, monkeypatch, "--replay")
    rows = list(csv.DictReader(io.StringIO(out)))
    assert rows[0]["size"] == "10"


def test_cli_replay_writes_csv_and_html(
    site: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path, html_path = tmp_path / "curve.csv", tmp_path / "curve.html"
    llm = FakeLLM(lambda _p, _s: {"zero_to_62_s": 1.0})
    code, out, err = run_cli(
        site,
        monkeypatch,
        "--replay",
        "--csv",
        str(csv_path),
        "--html",
        str(html_path),
        "--llm",
        "anthropic",
        llm=llm,
    )
    assert (code, err) == (0, "")
    assert "precision: 100.0%  recall: 100.0%  accuracy: 100.0%" in out
    assert f"wrote {csv_path}\nwrote {html_path}\n" in out
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
    # The LLM is the extraction LLM: one call for each family's first page.
    assert sum(float(r["llm_calls_per_document"]) * int(r["size"]) for r in rows) == len(llm.calls)
    assert float(rows[0]["llm_calls_per_document"]) > 0
    assert ">wave 2</text>" in html_path.read_text()


def test_cli_replay_json(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out, _ = run_cli(site, monkeypatch, "--replay", "--json")
    assert code == 0
    data = json.loads(out)
    assert data["batch_size"] == 10
    assert data["summary"]["accuracy"] == 1.0


@pytest.mark.parametrize(
    "flags", [["--csv", "x.csv"], ["--html", "x.html"], ["--batch-size", "3"], ["--llm", "openai"]]
)
def test_cli_replay_flags_need_replay(
    site: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        run_cli(site, monkeypatch, *flags)
    assert info.value.code == 2


def test_cli_replay_runs_one_document_at_a_time(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as info:
        run_cli(site, monkeypatch, "--replay", "--concurrency", "2")
    assert info.value.code == 2


def test_cli_replay_learner_failure_is_a_clean_error(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(_self: Extractor) -> None:
        raise RuntimeError("the learner worker failed") from KeyError("draft")

    monkeypatch.setattr(Extractor, "wait_for_learning", broken)
    code, out, err = run_cli(site, monkeypatch, "--replay")
    assert code == 1
    assert out == ""
    assert err.startswith("jevex: error: learning stopped after ")
    assert err.endswith(": 'draft'\n")


@pytest.mark.parametrize("size", ["0", "-1", "x"])
def test_cli_batch_size_must_be_positive(
    site: Path, monkeypatch: pytest.MonkeyPatch, size: str
) -> None:
    with pytest.raises(SystemExit) as info:
        run_cli(site, monkeypatch, "--replay", "--batch-size", size)
    assert info.value.code == 2


def test_cli_replay_checks_output_directories_first(
    site: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "nope" / "curve.html"
    stage = oracle(site)
    import jevex.extractor as extractor_module

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(extractor_module, "DEFAULT_STAGES", (stage,))
    code = main(
        ["eval", str(site), *SCHEMAS, "--replay", "--html", str(missing)],
        jev=FakeJev().client(),
        out=out,
        err=err,
    )
    assert code == 1
    assert f"no such directory for {missing}" in err.getvalue()
    assert stage.seen == set()  # nothing ran


def test_cli_replay_reports_failed_documents(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jevex.extractor as extractor_module

    @dataclass
    class Flaky:
        name: str = "select"

        async def run(self, ctx: Context) -> None:
            if "details" in (ctx.document.url or ""):  # the kv family
                raise RuntimeError("boom")

    monkeypatch.setattr(extractor_module, "DEFAULT_STAGES", (Flaky(),))
    out, err = io.StringIO(), io.StringIO()
    code = main(["eval", str(site), *SCHEMAS, "--replay"], jev=FakeJev().client(), out=out, err=err)
    assert code == 1
    assert out.getvalue().startswith("batch,documents")  # the CSV is still printed
    lines = err.getvalue().splitlines()
    assert lines
    assert all(line.startswith("jevex: error: ") and "boom" in line for line in lines)


def test_cli_replay_stops_on_the_spend_cap(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jevex.extractor as extractor_module

    monkeypatch.setattr(extractor_module, "DEFAULT_STAGES", (OverBudget(),))
    out, err = io.StringIO(), io.StringIO()
    code = main(["eval", str(site), *SCHEMAS, "--replay"], jev=FakeJev().client(), out=out, err=err)
    assert code == 1
    assert "jevex: error: Jev: over the cap" in err.getvalue()
    assert out.getvalue() == ""


def test_the_cli_replay_starts_from_nothing_and_uses_the_llm_for_both(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jevex.cli as cli

    made: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Extractor:
        made.append(kwargs)
        return Extractor(*args, **kwargs)

    monkeypatch.setattr(cli, "Extractor", spy)
    llm = FakeLLM(lambda _p, _s: {"zero_to_62_s": 1.0})
    code, _, _ = run_cli(site, monkeypatch, "--replay", "--llm", "anthropic", llm=llm)
    assert code == 0
    [kwargs] = made
    assert kwargs["store"] == ":memory:"
    assert kwargs["community_packs"] is False
    assert kwargs["extraction_llm"] is kwargs["generator_llm"] is llm

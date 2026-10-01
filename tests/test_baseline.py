import io
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from jevex import Context, Extractor
from jevex.baseline import (
    Baseline,
    BaselineError,
    Check,
    GateTolerances,
    check_baseline,
    corpus_digest,
    ensure_comparable,
)
from jevex.cli import format_gate, main
from jevex.eval import DocumentRun, EvalReport, FieldScore, evaluate, load_corpus
from jevex.jev import JevClient
from jevex.results import FieldMeta
from jevex.testing import RECORD_ENV, FakeJev, FakeLLM, cassette, llm_cassette
from jevex.testsite import build
from jevex.testsite.schemas import Listing, VehicleSpec

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = f"{FIXTURES / 'cli_schemas.py'}:Book"

# --- the baseline and the check --------------------------------------------------------


def run(
    fields: dict[str, FieldScore], *, llm_calls: int = 0, error: str | None = None
) -> DocumentRun:
    return DocumentRun(
        path="x.html",
        schema="Book",
        seconds=0.1,
        jev_requests=1,
        jev_questions=1,
        jev_cost=0.0,
        llm_calls=llm_calls,
        llm_cost=0.0,
        methods=Counter(),
        fields=fields,
        error=error,
    )


def report(title: tuple[int, int], price: tuple[int, int] = (5, 0), llm: int = 0) -> EvalReport:
    """Two documents: ``title`` and ``price`` are (correct, wrong) counts, ``llm`` the LLM
    calls on the first one."""
    return EvalReport(
        [
            run(
                {
                    "Book.title": FieldScore(correct=title[0], wrong=title[1]),
                    "Book.price": FieldScore(correct=price[0], wrong=price[1]),
                },
                llm_calls=llm,
            ),
            run({"Book.title": FieldScore(), "Book.price": FieldScore()}),
        ]
    )


BASE = Baseline.from_report(report((10, 0)), corpus="abc")


def test_a_baseline_holds_the_run_s_headline_numbers() -> None:
    b = Baseline.from_report(report((6, 4), llm=3), corpus="abc", mode="replay")
    assert b.corpus == "abc"
    assert b.mode == "replay"
    assert b.documents == 2
    assert b.accuracy == pytest.approx(11 / 15)
    assert b.llm_calls_per_document == 1.5
    assert b.fields == {"Book.price": 1.0, "Book.title": 0.6}
    assert b.tolerances == GateTolerances()


def test_baselines_round_trip_as_sorted_json(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    b = Baseline.from_report(
        report((6, 4)), corpus="abc", tolerances=GateTolerances(field_accuracy_drop=None)
    )
    b.write(path)
    text = path.read_text()
    assert text.endswith("}\n")
    data = json.loads(text)
    assert list(data) == sorted(data)
    assert data["version"] == 1
    assert data["tolerances"] == {
        "accuracy_drop": 0.02,
        "field_accuracy_drop": None,
        "llm_rate_rise": 0.1,
    }
    assert Baseline.load(path) == b


def test_a_missing_or_malformed_baseline_is_a_baseline_error(tmp_path: Path) -> None:
    with pytest.raises(BaselineError, match="can't read baseline"):
        Baseline.load(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"corpus": "abc"}')
    with pytest.raises(BaselineError, match="isn't a jevex baseline"):
        Baseline.load(bad)
    BASE.write(bad)
    data = json.loads(bad.read_text())
    bad.write_text(json.dumps({**data, "version": 2}))
    with pytest.raises(BaselineError, match="isn't a jevex baseline"):
        Baseline.load(bad)
    bad.write_text(json.dumps({**data, "acuracy": 0.5}))  # a typo isn't silently dropped
    with pytest.raises(BaselineError):
        Baseline.load(bad)


@pytest.mark.parametrize(
    "kwargs", [{"accuracy_drop": -0.1}, {"field_accuracy_drop": 1.5}, {"llm_rate_rise": -1}]
)
def test_tolerances_must_make_sense(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        GateTolerances.model_validate(kwargs)


def test_the_same_run_passes() -> None:
    result = check_baseline(report((10, 0)), BASE, corpus="abc")
    assert result.passed
    assert [c.metric for c in result.checks] == [
        "accuracy",
        "accuracy:Book.price",
        "accuracy:Book.title",
        "llm_calls_per_document",
    ]


def test_a_drop_up_to_the_tolerance_passes_and_beyond_it_fails() -> None:
    base = Baseline.from_report(report((50, 0), price=(50, 0)), corpus="abc")
    # One wrong title in 50: overall 99%, the field 98%: within 2 and 5 points.
    assert check_baseline(report((49, 1), price=(50, 0)), base, corpus="abc").passed
    # Exactly the overall tolerance (two points) still passes.
    assert check_baseline(report((48, 2), price=(50, 0)), base, corpus="abc").passed
    failed = check_baseline(report((40, 10), price=(50, 0)), base, corpus="abc")
    assert [c.metric for c in failed.regressions] == ["accuracy", "accuracy:Book.title"]


def test_one_field_collapsing_fails_even_when_the_overall_holds() -> None:
    base = Baseline.from_report(report((100, 0), price=(4, 0)), corpus="abc")
    result = check_baseline(report((100, 0), price=(2, 2)), base, corpus="abc")
    assert [c.metric for c in result.regressions] == ["accuracy:Book.price"]
    # With the per-field check off, only the overall counts.
    assert check_baseline(
        report((100, 0), price=(2, 2)),
        base,
        corpus="abc",
        tolerances=GateTolerances(field_accuracy_drop=None),
    ).passed


def test_a_rising_llm_rate_fails() -> None:
    base = Baseline.from_report(report((10, 0), llm=2), corpus="abc")  # 1 call per document
    assert check_baseline(report((10, 0), llm=2), base, corpus="abc").passed
    result = check_baseline(report((10, 0), llm=3), base, corpus="abc")  # 1.5 per document
    [regression] = result.regressions
    assert regression.describe() == "llm_calls_per_document 1.50 (baseline 1.00, at most 1.10)"
    loose = GateTolerances(llm_rate_rise=0.5)
    assert check_baseline(report((10, 0), llm=3), base, corpus="abc", tolerances=loose).passed


def test_improvements_never_fail() -> None:
    base = Baseline.from_report(report((5, 5), price=(1, 4), llm=4), corpus="abc")
    assert check_baseline(report((10, 0), llm=0), base, corpus="abc").passed


def test_a_field_the_run_no_longer_scores_is_a_regression() -> None:
    now = EvalReport([run({"Book.title": FieldScore(correct=10)})])
    result = check_baseline(now, BASE, corpus="abc")
    [regression] = result.regressions
    assert regression.metric == "accuracy:Book.price"
    assert regression.describe() == "accuracy:Book.price none (baseline 100.0%, at least 95.0%)"


def test_a_new_field_is_not_checked() -> None:
    now = report((10, 0))
    now.documents[0].fields["Book.isbn"] = FieldScore(wrong=3)
    result = check_baseline(now, BASE, corpus="abc")
    assert "accuracy:Book.isbn" not in [c.metric for c in result.checks]
    assert [c.metric for c in result.regressions] == ["accuracy"]  # 15/18 overall


def test_an_unscored_baseline_only_gates_the_llm_rate() -> None:
    empty = EvalReport([run({"Book.title": FieldScore(empty=1)})])
    base = Baseline.from_report(empty, corpus="abc")
    assert base.accuracy is None
    assert base.fields == {"Book.title": None}
    result = check_baseline(empty, base, corpus="abc")
    assert [c.metric for c in result.checks] == ["llm_calls_per_document"]
    assert result.passed


def test_check_describes_accuracy_as_percentages() -> None:
    c = Check(metric="accuracy", baseline=0.92, value=0.88, limit=0.9, higher_is_better=True)
    assert not c.passed
    assert c.describe() == "accuracy 88.0% (baseline 92.0%, at least 90.0%)"
    nothing = Check(metric="accuracy", baseline=None, value=None, limit=0, higher_is_better=True)
    assert nothing.passed


def test_a_baseline_from_another_corpus_or_mode_is_refused() -> None:
    with pytest.raises(BaselineError, match="different corpus"):
        check_baseline(report((10, 0)), BASE, corpus="xyz")
    with pytest.raises(BaselineError, match="from a plain eval, not a --replay run"):
        ensure_comparable(BASE, corpus="abc", mode="replay")


def test_format_gate() -> None:
    assert format_gate(check_baseline(report((10, 0)), BASE, corpus="abc"), "b.json") == (
        "jevex: gate passed against b.json (4 checks)\n"
    )
    failed = check_baseline(report((5, 5), llm=1), BASE, corpus="abc")
    assert format_gate(failed, "b.json") == (
        "jevex: error: gate failed against b.json:\n"
        "  accuracy 66.7% (baseline 100.0%, at least 98.0%)\n"
        "  accuracy:Book.title 50.0% (baseline 100.0%, at least 95.0%)\n"
        "  llm_calls_per_document 0.50 (baseline 0.00, at most 0.10)\n"
    )


# --- the corpus digest -----------------------------------------------------------------


def write_corpus(root: Path, titles: dict[str, str], truth: dict[str, str] | None = None) -> Path:
    """Pages ``<name>.html`` with an ``<h1>`` title each; ``truth`` overrides what the
    manifest expects (default: the titles)."""
    root.mkdir(parents=True, exist_ok=True)
    expected = {**titles, **(truth or {})}
    pages: list[dict[str, object]] = []
    for name, title in titles.items():
        (root / f"{name}.html").write_text(f"<html><body><h1>{title}</h1></body></html>")
        pages.append(
            {
                "path": f"{name}.html",
                "schema": "Book",
                "records": [{"entity": "document", "values": {"title": expected[name]}}],
            }
        )
    (root / "truth.json").write_text(json.dumps({"pages": pages}))
    return root


def test_the_corpus_digest_covers_the_truth_and_every_document(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "c", {"a": "Dune", "b": "Emma"})
    digest = corpus_digest(root)
    assert digest == corpus_digest(root)
    assert re.fullmatch("[0-9a-f]{64}", digest)
    # The same corpus somewhere else has the same digest.
    assert corpus_digest(write_corpus(tmp_path / "d", {"a": "Dune", "b": "Emma"})) == digest

    (root / "b.html").write_text("<h1>Emma!</h1>")
    assert corpus_digest(root) != digest
    write_corpus(root, {"a": "Dune", "b": "Emma"}, truth={"b": "Persuasion"})
    assert corpus_digest(root) != digest
    (root / "unlisted.html").write_text("ignored")
    write_corpus(root, {"a": "Dune", "b": "Emma"})
    assert corpus_digest(root) == digest


def test_the_digest_of_a_broken_corpus_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isn't a corpus manifest"):
        corpus_digest(tmp_path)
    root = write_corpus(tmp_path / "c", {"a": "Dune"})
    (root / "a.html").unlink()
    with pytest.raises(ValueError, match="doesn't exist"):
        corpus_digest(root)


# --- jevex eval --gate / --write-baseline ----------------------------------------------


class Car(BaseModel):
    zero_to_62_s: float


@dataclass
class ReadTitle:
    """Stand-in for the real stages: reads the ``<h1>``, getting the titles in ``wrong``
    wrong, and calls the extraction LLM (if any) on the pages in ``ask_llm``."""

    wrong: set[str] = field(default_factory=set[str])
    ask_llm: set[str] = field(default_factory=set[str])
    ran: list[str] = field(default_factory=list[str])
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        html = ctx.document.content.decode()
        match = re.search(r"<h1>(.*)</h1>", html)
        if match is None:
            raise RuntimeError("no title")
        title = match.group(1)
        self.ran.append(title)
        if title in self.ask_llm and ctx.extraction_llm is not None:
            assert ctx.budget is not None
            await ctx.budget.call_llm(ctx.extraction_llm, "What is it?", Car)
        value = f"Not {title}" if title in self.wrong else title
        for schema_run in ctx.active:
            schema_run.set_field("document", "title", FieldMeta(value=value, method="jev"))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "corpus", {"a": "Dune", "b": "Emma", "c": "Ulysses"})


def eval_cli(
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    *argv: str,
    stage: ReadTitle | None = None,
    llm: FakeLLM | None = None,
) -> tuple[int, str, str]:
    import jevex.extractor as extractor_module

    monkeypatch.setattr(extractor_module, "DEFAULT_STAGES", (stage or ReadTitle(),))
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["eval", str(corpus), "--schema", SCHEMA, *argv],
        jev=FakeJev().client(),
        llm=llm,
        out=out,
        err=err,
    )
    return code, out.getvalue(), err.getvalue()


def test_write_a_baseline_then_gate_against_it(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "baseline.json"
    code, out, err = eval_cli(corpus, monkeypatch, "--write-baseline", str(path))
    assert (code, err) == (0, f"jevex: wrote baseline {path}\n")
    assert "accuracy: 100.0%" in out  # the report is printed as usual
    written = Baseline.load(path)
    assert written.corpus == corpus_digest(corpus)
    assert (written.mode, written.documents, written.accuracy) == ("eval", 3, 1.0)

    code, out, err = eval_cli(corpus, monkeypatch, "--gate", str(path))
    assert (code, err) == (0, f"jevex: gate passed against {path} (3 checks)\n")
    assert "accuracy: 100.0%" in out


def test_a_regression_fails_the_gate(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "baseline.json"
    eval_cli(corpus, monkeypatch, "--write-baseline", str(path))
    code, out, err = eval_cli(corpus, monkeypatch, "--gate", str(path), stage=ReadTitle({"Emma"}))
    assert code == 1
    assert "accuracy:  66.7%" in out  # still printed
    assert err == (
        f"jevex: error: gate failed against {path}:\n"
        "  accuracy 66.7% (baseline 100.0%, at least 98.0%)\n"
        "  accuracy:Book.title 66.7% (baseline 100.0%, at least 95.0%)\n"
    )
    # Looser tolerances on the command line override the baseline's.
    code, _, err = eval_cli(
        corpus,
        monkeypatch,
        "--gate",
        str(path),
        "--max-accuracy-drop",
        "0.5",
        "--max-field-drop",
        "none",
        stage=ReadTitle({"Emma"}),
    )
    assert (code, err) == (0, f"jevex: gate passed against {path} (2 checks)\n")


def test_the_gate_counts_the_fallback_s_llm_calls(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "baseline.json"
    llm = FakeLLM(lambda _p, _s: {"zero_to_62_s": 1.0})
    eval_cli(corpus, monkeypatch, "--llm", "anthropic", "--write-baseline", str(path), llm=llm)
    assert Baseline.load(path).llm_calls_per_document == 0
    stage = ReadTitle(ask_llm={"Dune"})
    code, _, err = eval_cli(
        corpus, monkeypatch, "--llm", "anthropic", "--gate", str(path), stage=stage, llm=llm
    )
    assert code == 1
    assert "llm_calls_per_document 0.33 (baseline 0.00, at most 0.10)" in err
    assert len(llm.calls) == 1


def test_json_output_stays_json_with_the_gate(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "baseline.json"
    eval_cli(corpus, monkeypatch, "--write-baseline", str(path))
    code, out, err = eval_cli(corpus, monkeypatch, "--gate", str(path), "--json")
    assert code == 0
    assert json.loads(out)["summary"]["accuracy"] == 1.0
    assert err.startswith("jevex: gate passed")


def test_rewriting_a_baseline_keeps_its_tolerances(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "baseline.json"
    eval_cli(corpus, monkeypatch, "--write-baseline", str(path), "--max-llm-rise", "0.5")
    assert Baseline.load(path).tolerances == GateTolerances(llm_rate_rise=0.5)
    code, _, _ = eval_cli(
        corpus, monkeypatch, "--write-baseline", str(path), stage=ReadTitle({"Emma"})
    )
    assert code == 0
    rewritten = Baseline.load(path)
    assert rewritten.accuracy == pytest.approx(2 / 3)
    assert rewritten.tolerances == GateTolerances(llm_rate_rise=0.5)


def test_a_bad_baseline_fails_before_the_run(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = ReadTitle()
    code, out, err = eval_cli(
        corpus, monkeypatch, "--gate", str(tmp_path / "missing.json"), stage=stage
    )
    assert (code, out) == (1, "")
    assert err.startswith("jevex: error: can't read baseline")

    other = tmp_path / "other.json"
    Baseline.from_report(report((1, 0)), corpus="not this corpus").write(other)
    code, _, err = eval_cli(corpus, monkeypatch, "--gate", str(other), stage=stage)
    assert code == 1
    assert "recorded on a different corpus" in err

    not_a_baseline = tmp_path / "notes.json"
    not_a_baseline.write_text("{}")
    code, _, err = eval_cli(
        corpus, monkeypatch, "--write-baseline", str(not_a_baseline), stage=stage
    )
    assert code == 1
    assert "isn't a jevex baseline" in err
    assert not_a_baseline.read_text() == "{}"  # never overwritten

    code, _, err = eval_cli(
        corpus, monkeypatch, "--write-baseline", str(tmp_path / "no" / "b.json"), stage=stage
    )
    assert code == 1
    assert "no such directory" in err
    assert stage.ran == []  # nothing was extracted, so nothing was spent


def test_no_baseline_is_written_from_a_run_with_failures(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (corpus / "b.html").write_text("<p>no title</p>")
    path = tmp_path / "baseline.json"
    code, _, err = eval_cli(corpus, monkeypatch, "--write-baseline", str(path))
    assert code == 1
    assert "jevex: error: not writing a baseline: documents failed" in err
    assert not path.exists()


def test_a_failed_document_fails_a_run_that_passes_the_gate(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "baseline.json"
    eval_cli(corpus, monkeypatch, "--write-baseline", str(path))
    data = json.loads(path.read_text())
    data["tolerances"] = {"accuracy_drop": 1, "field_accuracy_drop": None, "llm_rate_rise": 1}
    (corpus / "b.html").write_text("<p>no title</p>")
    data["corpus"] = corpus_digest(corpus)
    path.write_text(json.dumps(data))
    code, _, err = eval_cli(corpus, monkeypatch, "--gate", str(path))
    assert code == 1
    assert "RuntimeError: no title" in err
    assert "gate passed" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-accuracy-drop", "0.1"],
        ["--max-field-drop", "none"],
        ["--max-llm-rise", "1"],
        ["--gate", "a.json", "--write-baseline", "b.json"],
        ["--gate", "a.json", "--max-accuracy-drop", "2"],
        ["--gate", "a.json", "--max-field-drop", "lots"],
    ],
)
def test_gate_flag_usage_errors(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        eval_cli(corpus, monkeypatch, *argv)
    assert info.value.code == 2


def test_a_replay_gates_against_a_replay_baseline(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replayed = tmp_path / "replay.json"
    code, _, err = eval_cli(corpus, monkeypatch, "--replay", "--write-baseline", str(replayed))
    assert (code, err) == (0, f"jevex: wrote baseline {replayed}\n")
    assert Baseline.load(replayed).mode == "replay"
    code, out, err = eval_cli(corpus, monkeypatch, "--replay", "--gate", str(replayed))
    assert code == 0
    assert out.startswith("batch,")  # the CSV is still on stdout
    assert err.startswith("jevex: gate passed")
    code, _, err = eval_cli(corpus, monkeypatch, "--gate", str(replayed))
    assert code == 1
    assert "the baseline is from a --replay run, not a plain eval" in err


# --- the gate in CI: the test site, replayed from recordings ---------------------------

GATE = FIXTURES / "testsite_gate"
JEV_CASSETTE = GATE / "jev-cassette.json"
LLM_CASSETTE = GATE / "llm-cassette.json"
GATE_BASELINE = GATE / "baseline.json"
UPDATE_ENV = "JEVEX_UPDATE_BASELINE"
GATE_FAMILIES = ("table", "kv", "prose", "grid", "listing")
"""The HTML families: their pages are the same bytes everywhere, unlike rasterised ones,
so the recordings replay on any machine."""


def gate_corpus(root: Path) -> Path:
    """Seed 42's first page of each HTML family, plus its first page with JSON-LD."""
    manifest = build(42, root, waves=[GATE_FAMILIES])
    pages: list[dict[str, object]] = []
    for family in GATE_FAMILIES:
        pages.append(next(p for p in manifest["pages"] if p["family"] == family))
    if not any(p["json_ld"] for p in pages):
        pages.append(next(p for p in manifest["pages"] if p["json_ld"]))
    (root / "truth.json").write_text(json.dumps({"seed": 42, "pages": pages}, indent=2))
    return root


def test_the_gate_corpus_has_each_html_family_and_json_ld(tmp_path: Path) -> None:
    root = gate_corpus(tmp_path / "corpus")
    items = json.loads((root / "truth.json").read_text())["pages"]
    assert {p["family"] for p in items} == set(GATE_FAMILIES)
    assert any(p["json_ld"] for p in items)
    assert {i.schema for i in load_corpus(root)} == {"VehicleSpec", "Listing"}
    assert corpus_digest(root) == corpus_digest(gate_corpus(tmp_path / "again"))


async def test_the_test_site_passes_the_eval_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI's regression gate: the test-site corpus above through the default pipeline, with
    Jev and the fallback LLM replayed from recordings, against ``baseline.json``.

    ``JEVEX_RECORD=1`` records all three (see ``fixtures/testsite_gate/README.md``);
    ``JEVEX_UPDATE_BASELINE=1`` rewrites only the baseline from the recordings, offline.
    A stale recording xfails for now, like the books smoke test; #129 makes it fail in CI.
    """
    recording = os.environ.get(RECORD_ENV) == "1"
    if not recording and not (JEV_CASSETTE.exists() and GATE_BASELINE.exists()):
        pytest.skip(
            "no test-site recording yet; record with JEVEX_RECORD=1 TYPESAFE_API_KEY=... "
            "ANTHROPIC_API_KEY=... uv run pytest tests/test_baseline.py -k eval_gate"
        )
    # A relative path, so document URLs (and anything resolved against them) are the
    # same on every machine and the recorded requests match.
    monkeypatch.chdir(tmp_path)
    corpus = gate_corpus(Path("corpus"))
    inner = None
    if recording:
        from jevex.llm.anthropic import AnthropicLLM

        inner = AnthropicLLM()
    llm = llm_cassette(LLM_CASSETTE, inner)
    jev = JevClient(cassette(JEV_CASSETTE))
    async with Extractor([VehicleSpec, Listing], jev=jev, extraction_llm=llm) as extractor:
        report = await evaluate(extractor, load_corpus(corpus))
    digest = corpus_digest(corpus)
    if recording or os.environ.get(UPDATE_ENV) == "1":
        assert not report.failed, [d.error for d in report.failed]
        Baseline.from_report(report, corpus=digest).write(GATE_BASELINE)
        return
    baseline = Baseline.load(GATE_BASELINE)
    stale = [d.error for d in report.failed if "no recording" in (d.error or "")]
    if stale or digest != baseline.corpus:
        pytest.xfail(f"the test-site recording is stale; re-record it ({stale[:1]})")
    assert not report.failed, [d.error for d in report.failed]
    result = check_baseline(report, baseline, corpus=digest)
    assert result.passed, format_gate(result, str(GATE_BASELINE))

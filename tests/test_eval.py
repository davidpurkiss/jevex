import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from jevex import Context, Extractor, Pipeline
from jevex.cli import main
from jevex.eval import (
    Expected,
    FieldScore,
    Tolerance,
    evaluate,
    list_scores,
    load_corpus,
    match_records,
    score_value,
    values_match,
)
from jevex.results import FieldMeta
from jevex.testing import FakeJev
from jevex.testsite import build
from jevex.testsite.schemas import Listing, VehicleSpec

# --- comparing values ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "actual", "match"),
    [
        (110.0, 110.3, True),  # within 0.5 absolute
        (1000.0, 1004.0, True),  # within 0.5%
        (1000.0, 1010.0, False),
        ("18495", 18495, True),  # Decimal in JSON is a string
        ("Moonstone  Grey", "moonstone grey", True),
        ("SE", "SE L", False),
        (True, 1, False),  # bools never match numbers
        (None, None, True),
    ],
)
def test_values_match(expected: Any, actual: Any, match: bool) -> None:
    assert values_match(expected, actual) is match


def test_tolerance_is_configurable() -> None:
    assert not values_match(110.0, 110.3, Tolerance(rel=0, abs=0.1))


def test_list_scores_match_each_item_once() -> None:
    assert list_scores(["red", "blue"], ["Blue", "red", "red"]) == (2, 2, 3)


@pytest.mark.parametrize(
    ("expected", "actual", "counts"),
    [
        (5, 5, (1, 0, 0, 0, 0)),
        (5, 6, (0, 1, 0, 0, 0)),
        (5, None, (0, 0, 1, 0, 0)),
        (None, 5, (0, 0, 0, 1, 0)),
        (None, None, (0, 0, 0, 0, 1)),
        (["a", "b"], ["a"], (1, 0, 1, 0, 0)),
        (["a"], ["a", "c"], (1, 0, 0, 1, 0)),
    ],
)
def test_score_value(expected: Any, actual: Any, counts: tuple[int, ...]) -> None:
    s = score_value(expected, actual)
    assert (s.correct, s.wrong, s.missing, s.spurious, s.empty) == counts


def test_precision_and_recall() -> None:
    s = FieldScore(correct=8, wrong=1, missing=1, spurious=2)
    assert s.precision == pytest.approx(8 / 11)
    assert s.recall == pytest.approx(8 / 10)
    assert FieldScore().precision is None


def test_records_pair_by_entity_then_by_agreement() -> None:
    expected = (
        Expected("SE", {"trim": "SE", "power_kw": 110}),
        Expected("GT", {"trim": "GT", "power_kw": 150}),
    )
    found = [
        {"entity": "column-2", "values": {"trim": "GT", "power_kw": 150}},
        {"entity": "SE", "values": {"trim": "SE", "power_kw": 111}},
        {"entity": "extra", "values": {"trim": "X"}},
    ]
    pairs = match_records(expected, found)
    by_expected = {e.entity if e else None: (r or {}).get("entity") for e, r in pairs}
    assert by_expected == {"SE": "SE", "GT": "column-2", None: "extra"}


# --- corpus ----------------------------------------------------------------------------


def test_load_corpus_from_a_test_site_build(tmp_path: Path) -> None:
    manifest = build(42, tmp_path)
    corpus = load_corpus(tmp_path)
    assert len(corpus) == len(manifest["pages"])
    assert {i.schema for i in corpus} == {"VehicleSpec", "Listing"}


def test_load_corpus_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isn't a corpus manifest"):
        load_corpus(tmp_path)
    (tmp_path / "truth.json").write_text(
        json.dumps({"pages": [{"path": "nope.html", "schema": "X", "records": []}]})
    )
    with pytest.raises(ValueError, match="doesn't exist"):
        load_corpus(tmp_path)


# --- end to end with an oracle ---------------------------------------------------------


@dataclass
class Oracle:
    """Test stage: "extracts" exactly the ground truth, optionally with mistakes."""

    truth: dict[str, list[dict[str, Any]]]
    mistakes: dict[str, Any] = field(default_factory=dict[str, Any])
    drop: set[str] = field(default_factory=set[str])
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        assert ctx.document.url is not None
        for record in self.truth.get(Path(ctx.document.url).name, []):
            for run in ctx.active:
                if not set(record["values"]) <= {f.name for f in run.spec.fields}:
                    continue
                for name, value in record["values"].items():
                    if name in self.drop or value is None:
                        continue
                    value = self.mistakes.get(name, value)
                    run.set_field(record["entity"], name, FieldMeta(value=value, method="jev"))


def oracle_for(site: Path, **kwargs: Any) -> Oracle:
    manifest = json.loads((site / "truth.json").read_text())
    truth = {Path(p["path"]).name: p["records"] for p in manifest["pages"]}
    return Oracle(truth=truth, **kwargs)


def extractor(stage: Oracle) -> Extractor:
    return Extractor([VehicleSpec, Listing], jev=FakeJev().client(), pipeline=Pipeline([stage]))


async def test_a_perfect_extractor_scores_100_percent(tmp_path: Path) -> None:
    build(42, tmp_path)
    report = await evaluate(extractor(oracle_for(tmp_path)), load_corpus(tmp_path))
    summary = report.summary()
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["errors"] == 0
    assert summary["resolution_mix"]["jev"] > 0
    assert all(s.wrong == s.missing == s.spurious == 0 for s in report.field_scores().values())


async def test_mistakes_land_in_the_right_buckets(tmp_path: Path) -> None:
    build(42, tmp_path)
    stage = oracle_for(tmp_path, mistakes={"seats": 99}, drop={"colour"})
    report = await evaluate(extractor(stage), load_corpus(tmp_path))
    scores = report.field_scores()
    assert scores["VehicleSpec.seats"].correct == 0
    assert scores["VehicleSpec.seats"].wrong > 0
    assert scores["Listing.colour"].missing > 0
    assert scores["Listing.colour"].recall == 0
    assert scores["VehicleSpec.model"].precision == 1.0


async def test_unknown_corpus_schemas_are_rejected(tmp_path: Path) -> None:
    build(42, tmp_path)
    only_specs = Extractor([VehicleSpec], jev=FakeJev().client(), pipeline=Pipeline([]))
    with pytest.raises(ValueError, match="Listing"):
        await evaluate(only_specs, load_corpus(tmp_path))


async def test_a_failing_document_is_scored_missing_and_the_run_continues(tmp_path: Path) -> None:
    build(42, tmp_path)

    @dataclass
    class Flaky:
        name: str = "select"

        async def run(self, ctx: Context) -> None:
            if "table" in (ctx.document.url or ""):
                raise RuntimeError("boom")

    ex = Extractor([VehicleSpec, Listing], jev=FakeJev().client(), pipeline=Pipeline([Flaky()]))
    report = await evaluate(ex, load_corpus(tmp_path))
    failed = [d for d in report.documents if d.error]
    assert failed
    assert all("boom" in (d.error or "") for d in failed)
    assert report.summary()["errors"] == len(failed)
    assert report.overall().missing > 0


# --- CLI -------------------------------------------------------------------------------


def test_cli_eval_prints_a_summary_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jevex.extractor as extractor_module

    build(42, tmp_path)
    monkeypatch.setattr(extractor_module, "DEFAULT_STAGES", (oracle_for(tmp_path),))
    schemas = [
        "--schema",
        "jevex.testsite.schemas:VehicleSpec",
        "--schema",
        "jevex.testsite.schemas:Listing",
    ]
    out, err = io.StringIO(), io.StringIO()
    code = main(["eval", str(tmp_path), *schemas], jev=FakeJev().client(), out=out, err=err)
    assert (code, err.getvalue()) == (0, "")
    assert "precision: 100.0%" in out.getvalue()
    assert "VehicleSpec.power_kw" in out.getvalue()

    out = io.StringIO()
    code = main(["eval", str(tmp_path), *schemas, "--json"], jev=FakeJev().client(), out=out)
    assert code == 0
    assert json.loads(out.getvalue())["summary"]["recall"] == 1.0


def test_cli_eval_bad_corpus_is_a_clean_error(tmp_path: Path) -> None:
    err = io.StringIO()
    code = main(
        ["eval", str(tmp_path), "--schema", "jevex.testsite.schemas:VehicleSpec"],
        jev=FakeJev().client(),
        out=io.StringIO(),
        err=err,
    )
    assert code == 1
    assert "isn't a corpus manifest" in err.getvalue()

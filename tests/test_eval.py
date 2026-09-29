import asyncio
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from jevex import Context, Extractor, Pipeline
from jevex.cli import main
from jevex.eval import (
    EXACT,
    Expected,
    FieldScore,
    Tolerance,
    evaluate,
    field_tolerance,
    list_scores,
    load_corpus,
    match_records,
    rounding_slack,
    score_value,
    values_match,
)
from jevex.jev import JevBudgetExceededError
from jevex.results import FieldMeta
from jevex.schema import SchemaSpec
from jevex.testing import FakeJev
from jevex.testsite import build
from jevex.testsite.schemas import Listing, VehicleSpec

# --- comparing values ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "actual", "match"),
    [
        (110.0, 110.0, True),
        (110.0, 110.3, False),  # exact by default
        ("18495", 18495, True),  # Decimal in JSON is a string
        ("19000", Decimal("19090"), False),
        ("Moonstone  Grey", "moonstone grey", True),
        ("SE", "SE L", False),
        (True, 1, False),  # bools never match numbers
        (None, None, True),
    ],
)
def test_values_match(expected: Any, actual: Any, match: bool) -> None:
    assert values_match(expected, actual) is match


def test_tolerance_is_configurable() -> None:
    assert values_match(110.0, 110.3, Tolerance(abs=0.5))
    assert values_match(1000.0, 1004.0, Tolerance(rel=0.005))
    assert not values_match(1000.0, 1010.0, Tolerance(rel=0.005))


def tolerances(model: type) -> dict[str, Tolerance]:
    return {f.name: field_tolerance(f) for f in SchemaSpec.from_model(model).fields}


@pytest.mark.parametrize(
    ("model", "name", "expected", "actual", "match"),
    [
        (Listing, "year", 2016, 2017, False),  # counts are exact
        (Listing, "year", 2016, 2016, True),
        (Listing, "price_gbp", "19000", Decimal("19090"), False),  # money is exact
        (Listing, "price_gbp", "19000", Decimal("19000"), True),
        (VehicleSpec, "seats", 5, 6, False),
        (VehicleSpec, "engine_size_cc", 1395, 1400, False),  # no comparable unit to round in
        (VehicleSpec, "power_kw", 70, 70.37, True),  # shown as whole bhp
        (VehicleSpec, "power_kw", 70, 70.5, False),
        (VehicleSpec, "power_kw", 300, 301.4, True),  # within 0.5%
        (VehicleSpec, "top_speed_mph", 128, 128.31, True),  # shown as whole km/h
        (VehicleSpec, "top_speed_mph", 128, 129, False),
        (VehicleSpec, "zero_to_62_s", 6.1, 6.5, False),
        (VehicleSpec, "zero_to_62_s", 6.1, 6.12, True),
    ],
)
def test_field_tolerance_fits_the_field(
    model: type, name: str, expected: Any, actual: Any, match: bool
) -> None:
    assert values_match(expected, actual, tolerances(model)[name]) is match


def test_rounding_slack() -> None:
    assert rounding_slack("kW") == pytest.approx(0.745699872 / 2)  # half a bhp
    assert rounding_slack("mph") == pytest.approx(0.5 / 1.609344)  # half a km/h
    assert rounding_slack("s") == 0
    assert rounding_slack("GBP") == 0  # not a physical unit
    assert tolerances(VehicleSpec)["make"] == EXACT


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
        (["a", "b"], None, (0, 0, 2, 0, 0)),  # every expected item is missing
        (["a", "b"], [], (0, 0, 2, 0, 0)),
        (["a", "b"], ["c"], (0, 1, 1, 0, 0)),  # something wrong isn't worse than nothing
        ([], ["x", "y", "z"], (0, 0, 0, 3, 0)),
        (None, ["x", "y"], (0, 0, 0, 2, 0)),
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


def test_records_pair_by_agreement() -> None:
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


def test_labels_only_break_ties() -> None:
    expected = (Expected("a", {"make": "Kia"}), Expected("b", {"make": "Kia"}))
    found = [
        {"entity": "b", "values": {"make": "Kia"}},
        {"entity": "a", "values": {"make": "Kia"}},
    ]
    pairs = match_records(expected, found)
    assert {(e.entity, r["entity"]) for e, r in pairs if e and r} == {("a", "a"), ("b", "b")}


def test_a_skipped_card_does_not_shift_every_pairing() -> None:
    # Positional labels: the extractor missed card 1, so its "listing-1" is really card 2.
    expected = tuple(
        Expected(f"listing-{i}", {"model": f"M{i}", "year": 2010 + i, "mileage_miles": i * 1000})
        for i in range(1, 13)
    )
    found = [
        {"entity": f"listing-{i}", "values": dict(e.values)}
        for i, e in enumerate(expected[1:], start=1)
    ]
    pairs = match_records(expected, found)
    for exp, rec in pairs:
        if exp and rec:
            assert exp.values == rec["values"]
    assert [e.entity for e, r in pairs if e and r is None] == ["listing-1"]


def test_matching_many_records_is_quick() -> None:
    expected = tuple(Expected(f"r{i}", {"n": i, "s": f"x{i}"}) for i in range(300))
    found = [{"entity": f"r{i}", "values": {"n": i, "s": f"x{i}"}} for i in reversed(range(300))]
    pairs = match_records(expected, found)
    assert all(e and r and e.entity == r["entity"] for e, r in pairs)


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


MALFORMED: list[tuple[Any, str]] = [
    ({"a": 1}, "'pages' must be a list"),
    (["specs/x.html"], "page 0: a page must be an object"),
    ([{"path": "x.html", "schema": "X"}], "page 0: missing 'records'"),
    ([{"path": "x.html", "schema": "X", "records": {}}], "'records' must be a list"),
    (
        [{"path": "x.html", "schema": "X", "records": [{"entity": "a"}]}],
        "page 0: record 0 must be an object with a 'values' object",
    ),
    ([{"schema": "X", "records": []}], "page 0: missing 'path'"),
    ([{"path": "/nope/x.html", "schema": "X", "records": []}], "lists /nope/x.html, which doesn't"),
]


@pytest.mark.parametrize(("pages", "message"), MALFORMED)
def test_malformed_manifests_are_value_errors(tmp_path: Path, pages: Any, message: str) -> None:
    (tmp_path / "x.html").write_text("<p>x</p>")
    (tmp_path / "truth.json").write_text(json.dumps({"pages": pages}))
    with pytest.raises(ValueError, match=message):
        load_corpus(tmp_path)


# --- end to end with an oracle ---------------------------------------------------------


@dataclass
class Oracle:
    """Test stage: "extracts" exactly the ground truth, optionally with mistakes."""

    truth: dict[str, list[dict[str, Any]]]
    mistakes: dict[str, Callable[[Any], Any]] = field(default_factory=dict[str, Any])
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
                    if name in self.mistakes:
                        value = self.mistakes[name](value)
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
    mistakes: dict[str, Callable[[Any], Any]] = {
        "seats": lambda _: 99,
        "year": lambda y: y + 1,
        "power_kw": lambda kw: kw + 0.3,
    }
    stage = oracle_for(tmp_path, mistakes=mistakes, drop={"colour"})
    report = await evaluate(extractor(stage), load_corpus(tmp_path))
    scores = report.field_scores()
    assert scores["VehicleSpec.seats"].correct == 0
    assert scores["VehicleSpec.seats"].wrong > 0
    assert scores["Listing.colour"].missing > 0
    assert scores["Listing.colour"].recall == 0
    assert scores["VehicleSpec.model"].precision == 1.0
    assert scores["Listing.year"].correct == 0  # a year out by one is wrong
    assert scores["VehicleSpec.power_kw"].wrong == 0  # 0.3 kW is rounding


async def test_the_summary_and_json_carry_run_metrics(tmp_path: Path) -> None:
    build(42, tmp_path)
    report = await evaluate(extractor(oracle_for(tmp_path)), load_corpus(tmp_path))
    summary = report.summary()
    assert summary["latency_p50"] is not None
    assert summary["latency_p95"] >= summary["latency_p50"]
    assert summary["jev_questions_per_document"] == 0  # the oracle asks nothing
    doc = report.to_dict()["documents"][0]
    assert doc["methods"] == {"jev": sum(doc["methods"].values())}
    assert "jev_questions" in doc


async def test_records_of_another_schema_are_spurious(tmp_path: Path) -> None:
    build(42, tmp_path)
    oracle = oracle_for(tmp_path)
    spec_pages = {
        Path(p["path"]).name
        for p in json.loads((tmp_path / "truth.json").read_text())["pages"]
        if p["schema"] == "VehicleSpec"
    }

    @dataclass
    class WrongGate:
        name: str = "select"

        async def run(self, ctx: Context) -> None:
            await oracle.run(ctx)
            if Path(ctx.document.url or "").name in spec_pages:
                for run in ctx.active:
                    if run.spec.name == "Listing":
                        run.set_field("bogus", "make", FieldMeta(value="Kia", method="jev"))

    ex = Extractor([VehicleSpec, Listing], jev=FakeJev().client(), pipeline=Pipeline([WrongGate()]))
    report = await evaluate(ex, load_corpus(tmp_path))
    make = report.field_scores()["Listing.make"]
    assert make.spurious == len(spec_pages)
    assert make.precision is not None
    assert make.precision < 1
    assert report.summary()["precision"] < 1


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


@dataclass
class OverBudget:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        raise JevBudgetExceededError("over the cap")


async def test_the_spend_cap_stops_the_run(tmp_path: Path) -> None:
    build(42, tmp_path)
    ex = Extractor(
        [VehicleSpec, Listing], jev=FakeJev().client(), pipeline=Pipeline([OverBudget()])
    )
    with pytest.raises(JevBudgetExceededError):
        await evaluate(ex, load_corpus(tmp_path))


async def test_a_run_error_cancels_the_documents_still_running(tmp_path: Path) -> None:
    build(42, tmp_path)
    finished: list[str] = []

    @dataclass
    class CapOnFirst:
        name: str = "select"

        async def run(self, ctx: Context) -> None:
            if (ctx.document.url or "").endswith("used/page-1.html"):
                raise JevBudgetExceededError("over the cap")
            await asyncio.sleep(0.05)
            finished.append(ctx.document.url or "")

    ex = Extractor(
        [VehicleSpec, Listing], jev=FakeJev().client(), pipeline=Pipeline([CapOnFirst()])
    )
    corpus = sorted(load_corpus(tmp_path), key=lambda i: not i.path.match("used/page-1.html"))
    with pytest.raises(JevBudgetExceededError):
        await evaluate(ex, corpus, concurrency=4)
    await asyncio.sleep(0.2)  # long enough for any survivor to finish
    assert finished == []


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


SCHEMAS = [
    "--schema",
    "jevex.testsite.schemas:VehicleSpec",
    "--schema",
    "jevex.testsite.schemas:Listing",
]


def test_cli_eval_stops_on_the_spend_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jevex.extractor as extractor_module

    build(42, tmp_path)
    monkeypatch.setattr(extractor_module, "DEFAULT_STAGES", (OverBudget(),))
    out, err = io.StringIO(), io.StringIO()
    code = main(["eval", str(tmp_path), *SCHEMAS], jev=FakeJev().client(), out=out, err=err)
    assert code == 1
    assert "jevex: error: Jev: over the cap" in err.getvalue()
    assert out.getvalue() == ""


def test_cli_eval_reports_failed_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jevex.extractor as extractor_module

    @dataclass
    class Flaky:
        name: str = "select"

        async def run(self, ctx: Context) -> None:
            if "table" in (ctx.document.url or ""):
                raise RuntimeError("boom")

    build(42, tmp_path)
    monkeypatch.setattr(extractor_module, "DEFAULT_STAGES", (Flaky(),))
    out, err = io.StringIO(), io.StringIO()
    code = main(["eval", str(tmp_path), *SCHEMAS], jev=FakeJev().client(), out=out, err=err)
    assert code == 1
    assert "precision:" in out.getvalue()  # the report is still printed
    lines = err.getvalue().splitlines()
    assert lines
    assert all(line.startswith("jevex: error: ") and "boom" in line for line in lines)
    assert all("table" in line for line in lines)


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

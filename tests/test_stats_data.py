from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from jevex import Budgets, Document, Extractor, Field, Pipeline, RunBudget
from jevex.jev import Noul
from jevex.pipeline import Context
from jevex.stats import (
    VIEWS,
    Point,
    Stats,
    curve,
    field_stats,
    from_replay_csv,
    from_store,
    summary,
    to_json,
)
from jevex.stats.data import merge, method_shares
from jevex.store import (
    DocumentEvent,
    DocumentStat,
    GeneratorRecord,
    MemoryLedger,
    SpendEntry,
    SQLiteStore,
    ValueStat,
    VerifiedExample,
)
from jevex.testing import FakeJev

T0 = datetime(2026, 9, 30, 8, 0, tzinfo=UTC)


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")


@dataclass
class AsksOnce:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("s", {"q": Noul(instructions="a?")})


@pytest.fixture
async def store() -> AsyncIterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    yield s
    await s.aclose()


def doc(i: int, *, llm: int = 0, values: list[ValueStat] | None = None, **kw: Any) -> DocumentStat:
    return DocumentStat.model_validate(
        {
            "id": f"d{i}",
            "at": T0 + timedelta(minutes=i),
            "url": f"https://cars.test/{i}",
            "jev_cost": 0.001,
            "llm_calls": llm,
            "llm_cost": 0.01 * llm,
            "values": values or [],
            **kw,
        }
    )


def point(documents: int, size: int = 1, **kw: Any) -> Point:
    return Point(
        documents=documents,
        size=size,
        llm_calls_per_document=kw.pop("llm", 0.0),
        jev_cost_per_document=kw.pop("jev_cost", 0.0),
        llm_cost_per_document=kw.pop("llm_cost", 0.0),
        **kw,
    )


# --- from a store --------------------------------------------------------------------


async def test_a_store_gives_one_point_per_document(store: SQLiteStore) -> None:
    jev = ValueStat(field="Car.model", method="jev", confidence=0.9, value="Golf")
    llm = ValueStat(field="Car.power_ps", method="llm", confidence=0.6, value="150")
    await store.record_document(doc(0, llm=2, values=[jev, llm]))
    await store.record_document(doc(1, values=[jev]))
    stats = await from_store(store, source="sqlite:///x.db")
    assert stats.kind == "store"
    assert stats.source == "sqlite:///x.db"
    assert stats.has_time
    assert stats.default_axis() == "time"
    first, second = stats.points
    assert (first.documents, first.size, first.at) == (1, 1, T0)
    assert first.llm_calls_per_document == 2
    assert first.cost_per_document == pytest.approx(0.021)
    assert first.methods == {"jev": 1, "llm": 1}
    assert first.accuracy is None  # a store has no ground truth
    assert second.methods == {"jev": 1}
    # No ledger entries (no run budget): spend comes from the documents.
    assert [(s.documents, s.jev, s.llm) for s in stats.spend] == [
        (1, pytest.approx(0.001), pytest.approx(0.02)),
        (2, pytest.approx(0.002), pytest.approx(0.02)),
    ]


async def test_spend_comes_from_the_ledger_when_it_has_entries(store: SQLiteStore) -> None:
    await store.record_document(doc(0))
    await store.record_document(doc(1))
    await store.record_spend(SpendEntry(amount_usd=1.0, kind="jev", at=T0 - timedelta(hours=1)))
    await store.record_spend(SpendEntry(amount_usd=0.5, kind="jev", at=T0))
    await store.record_spend(SpendEntry(amount_usd=0.0, kind="llm_call", at=T0))
    # The learner spends between documents: the ledger has it, the documents don't.
    await store.record_spend(SpendEntry(amount_usd=0.25, kind="llm", at=T0 + timedelta(seconds=30)))
    stats = await from_store(store, source="s", budget_usd=2.0)
    # From the first document on; llm_call markers aren't spend.
    assert [(s.documents, s.jev, s.llm) for s in stats.spend] == [(1, 0.5, 0.0), (1, 0.5, 0.25)]
    assert stats.budget_usd == 2.0
    assert summary(stats)["spent_usd"] == 0.75


async def test_spend_comes_from_a_ledger_kept_apart_from_the_store(store: SQLiteStore) -> None:
    await store.record_document(doc(0))
    await store.record_spend(SpendEntry(amount_usd=9.0, kind="jev", at=T0))  # not this one's
    ledger = MemoryLedger()
    await ledger.record_spend(SpendEntry(amount_usd=0.5, kind="jev", at=T0))
    stats = await from_store(store, source="s", ledger=ledger)
    assert [(s.documents, s.jev) for s in stats.spend] == [(1, 0.5)]


async def test_a_documents_charges_count_against_it(store: SQLiteStore) -> None:
    # As the extractor writes them: charges while a document runs, its stat when it ends.
    for i in range(2):
        start = T0 + timedelta(seconds=10 * i)
        await store.record_spend(
            SpendEntry(amount_usd=0.5, kind="jev", at=start + timedelta(seconds=1))
        )
        await store.record_document(
            DocumentStat(id=f"d{i}", at=start + timedelta(seconds=2), seconds=1.5)
        )
    stats = await from_store(store, source="s")
    assert [(s.documents, s.jev) for s in stats.spend] == [(1, 0.5), (2, 1.0)]
    assert stats.started == T0 + timedelta(seconds=0.5)


async def test_an_extractors_ledger_spend_is_all_counted(tmp_path: Path) -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([AsksOnce()]),
        budgets=Budgets(run=RunBudget(max_jev_spend=10.0)),
        store=f"sqlite:///{tmp_path / 's.db'}",
    )
    first = await ex.extract(Document.from_bytes(b"<p>Golf</p>"))
    second = await ex.extract(Document.from_bytes(b"<p>Polo</p>"))
    store = await ex.store()
    assert store is not None
    stats = await from_store(store, source="s")
    await ex.aclose()
    assert stats.spend[-1].documents == 2
    assert stats.spend[-1].jev == pytest.approx(first.meta.jev.cost + second.meta.jev.cost)
    assert stats.spend[0].documents == 1


async def test_generators_with_counts_status_and_the_example_they_came_from(
    store: SQLiteStore,
) -> None:
    await store.record_document(doc(0))
    await store.record_document(doc(5))
    await store.add_example(
        VerifiedExample(id="ex1", field="Car.power_ps", statement="Power: 150 PS", value=150)
    )
    spec = {"id": "g1", "provenance": {"learned_from": ["gone", "ex1"]}}
    await store.put_generator(
        GeneratorRecord(
            id="g1", field="Car.power_ps", spec=spec, created_at=T0 + timedelta(minutes=2)
        )
    )
    await store.put_generator(
        GeneratorRecord(
            id="g0", field="Car.model", spec={"id": "g0"}, created_at=T0 - timedelta(days=1)
        )
    )
    await store.set_generator_enabled("g0", False)
    await store.record_generator_stats("g1", documents=4, hits=3, wins=2)
    stats = await from_store(store, source="s")
    old, learned = stats.generators
    assert (old.generator_id, old.status, old.win_rate, old.example) == (
        "g0",
        "disabled",
        None,
        None,
    )
    assert (learned.hits, learned.wins, learned.documents) == (3, 2, 4)
    assert learned.win_rate == pytest.approx(2 / 3)
    assert learned.status == "active"
    assert learned.learned_from == ("gone", "ex1")
    assert learned.example == "Power: 150 PS"
    # Only g1 was learned after the first document started: one tick, one document in.
    assert stats.learned == [1]
    assert stats.learned_at == [T0 + timedelta(minutes=2)]
    tiles = summary(stats)
    assert tiles["generators"] == 1
    assert tiles["generators_last_day"] == 1


async def test_events_carry_where_and_when(store: SQLiteStore) -> None:
    await store.record_document(doc(0))
    await store.record_document(
        doc(1, events=[DocumentEvent(kind="error", message="JevBackendError: down")])
    )
    stats = await from_store(store, source="s")
    [event] = stats.events
    assert (event.kind, event.documents, event.at, event.url) == (
        "error",
        2,
        T0 + timedelta(minutes=1),
        "https://cars.test/1",
    )


async def test_since_and_limit_select_documents(store: SQLiteStore) -> None:
    for i in range(5):
        await store.record_document(doc(i))
    stats = await from_store(store, source="s", limit=2)
    assert [p.at for p in stats.points] == [T0 + timedelta(minutes=3), T0 + timedelta(minutes=4)]
    assert [p.documents for p in stats.points] == [1, 2]
    stats = await from_store(store, source="s", since=T0 + timedelta(minutes=4))
    assert len(stats.points) == 1


async def test_an_empty_store_has_empty_stats(store: SQLiteStore) -> None:
    stats = await from_store(store, source="s")
    assert stats.points == []
    assert stats.documents == 0
    assert not stats.has_time
    assert stats.default_axis() == "docs"
    tiles = summary(stats)
    assert tiles["llm_calls_per_document"] is None
    assert tiles["llm_calls_change"] is None
    assert tiles["spent_usd"] == 0.0


def test_field_stats_count_methods_confidence_and_lowest_values() -> None:
    docs = [
        doc(
            0,
            values=[
                ValueStat(field="Car.trim", method="llm", confidence=0.5, value="GTI"),
                ValueStat(field="Car.model", method="structured", value="Golf"),
            ],
        ),
        doc(1, values=[ValueStat(field="Car.trim", method="jev", confidence=0.9, value="R")]),
        doc(2, values=[ValueStat(field="Car.trim", method="llm", confidence=0.5, value="GTD")]),
    ]
    model, trim = field_stats(docs, lowest=2)
    assert (model.field, model.n, model.mean_confidence, model.lowest) == (
        "Car.model",
        1,
        None,
        (),
    )
    assert trim.n == 3
    assert trim.methods == {"llm": 2, "jev": 1}
    assert trim.mean_confidence == pytest.approx((0.5 + 0.9 + 0.5) / 3)
    assert trim.fallback_rate == pytest.approx(2 / 3)
    assert trim.lowest == (("GTD", 0.5), ("GTI", 0.5))  # the most recent first among equals


# --- from a replay -------------------------------------------------------------------

CSV = """\
batch,documents,size,waves,accuracy,llm_calls_per_document,jev_cost_per_document,\
llm_cost_per_document,errors,generators,values_jev,values_llm,values_generator
1,2,2,1,1.0,2.0,0.001,0.02,0,1,1,3,0
2,4,2,1 2,,0.5,0.001,0.005,1,3,2,1,1
3,6,2,2,0.5,0.0,0.001,0.0,0,3,2,0,2
"""


def test_a_replay_csv_gives_one_point_per_batch() -> None:
    stats = from_replay_csv(CSV, source="curve.csv")
    assert stats.kind == "replay"
    assert not stats.has_time
    assert stats.default_axis() == "docs"
    assert [p.documents for p in stats.points] == [2, 4, 6]
    assert [p.accuracy for p in stats.points] == [1.0, None, 0.5]
    assert stats.points[0].methods == {
        "structured": 0,
        "jev": 1,
        "generator": 0,
        "llm": 3,
        "vision": 0,
    }
    assert stats.points[1].waves == (1, 2)
    assert stats.waves == [(2, 2)]  # wave 2 first appears in the batch starting 2 in
    assert stats.learned == [2, 4, 4]  # one generator after batch 1, two more after batch 2
    assert [(e.kind, e.message, e.documents) for e in stats.events] == [
        ("error", "1 documents failed", 4)
    ]
    assert stats.spend[-1].llm == pytest.approx(0.05)
    assert summary(stats)["generators"] == 3
    assert summary(stats)["accuracy"] == pytest.approx(0.75)
    # A CSV from before the learning columns: the learner's spend wasn't measured.
    assert not any(p.learning_counted for p in stats.points)


LEARNING_CSV = """\
documents,size,llm_calls_per_document,jev_cost_per_document,llm_cost_per_document,\
learning_jev_cost_per_document,learning_llm_cost_per_document,\
learning_llm_calls_per_document,errors
2,2,1.0,0.001,0.02,0.002,0.01,0.5,0
4,2,0.0,0.001,0.0,0.0,0.0,0.0,0
"""


def test_a_replay_csvs_learning_spend_counts_in_its_cost() -> None:
    stats = from_replay_csv(LEARNING_CSV)
    first, second = stats.points
    assert first.learning_counted
    assert (first.learning_jev_cost_per_document, first.learning_llm_cost_per_document) == (
        0.002,
        0.01,
    )
    assert first.cost_per_document == pytest.approx(0.001 + 0.02 + 0.002 + 0.01)
    assert second.cost_per_document == pytest.approx(0.001)
    assert stats.spend[0].jev == pytest.approx(2 * (0.001 + 0.002))
    assert stats.spend[-1].llm == pytest.approx(2 * (0.02 + 0.01))
    assert summary(stats)["spent_usd"] == pytest.approx(2 * 0.033 + 2 * 0.001)
    learning = to_json(stats, "learning")["points"][0]
    assert learning["learning_jev_cost_per_document"] == 0.002
    assert learning["cost_per_document"] == pytest.approx(0.033)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("a,b\n1,2\n", "not a jevex replay CSV: no documents, jev_cost_per_document"),
        (
            "documents,size,llm_calls_per_document,jev_cost_per_document\n2,2,lots,0\n",
            "replay CSV line 2: llm_calls_per_document is 'lots', not a number",
        ),
    ],
)
def test_a_bad_replay_csv_is_refused(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        from_replay_csv(text)


# --- queries -------------------------------------------------------------------------


def test_curve_buckets_documents_weighting_each() -> None:
    points = [point(i + 1, llm=float(i % 2), accuracy=None) for i in range(10)]
    stats = Stats(source="s", kind="store", points=points)
    assert curve(stats, max_points=20) == points
    buckets = curve(stats, max_points=3)
    assert [(b.documents, b.size) for b in buckets] == [(4, 4), (8, 4), (10, 2)]
    assert [b.llm_calls_per_document for b in buckets] == [0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="at least 1"):
        curve(stats, max_points=0)


def test_merge_weights_accuracy_by_scored_documents_and_adds_methods() -> None:
    merged = merge(
        [
            point(2, 2, accuracy=1.0, methods={"llm": 2}, waves=(1,), generators=1),
            point(3, 1, accuracy=None, methods={"llm": 1, "jev": 1}),
            point(5, 2, accuracy=0.25, waves=(2,), generators=4),
        ]
    )
    assert (merged.documents, merged.size, merged.generators) == (5, 5, 4)
    assert merged.accuracy == pytest.approx((2 * 1.0 + 2 * 0.25) / 4)
    assert merged.methods == {"llm": 3, "jev": 1}
    assert merged.waves == (1, 2)
    assert merged.learning_jev_cost_per_document is None  # never measured
    with pytest.raises(ValueError, match="nothing"):
        merge([])


def test_merge_counts_learning_spend_where_any_point_measured_it() -> None:
    merged = merge(
        [
            point(2, 2, learning_jev_cost_per_document=0.004, learning_llm_cost_per_document=0.0),
            point(3, 1),
        ]
    )
    assert merged.learning_counted
    assert merged.learning_jev_cost_per_document == pytest.approx(2 * 0.004 / 3)
    assert merged.learning_llm_cost_per_document == 0


def test_method_shares() -> None:
    assert method_shares({"llm": 1, "generator": 3}) == {
        "structured": 0.0,
        "jev": 0.0,
        "generator": 0.75,
        "llm": 0.25,
        "vision": 0.0,
    }
    assert set(method_shares({}).values()) == {0.0}


def test_summary_compares_the_last_tenth_with_the_first() -> None:
    points = [point(i + 1, llm=2.0 if i < 10 else 0.5, llm_cost=0.01) for i in range(100)]
    tiles = summary(Stats(source="s", kind="store", points=points))
    assert tiles["llm_calls_per_document"] == 0.5
    assert tiles["llm_calls_change"] == pytest.approx(-0.75)
    assert tiles["cost_change"] == pytest.approx(0.0)
    assert tiles["documents"] == 100


def test_the_point_time_axis_needs_a_time() -> None:
    assert point(3).x("docs") == 3.0
    with pytest.raises(ValueError, match="no time"):
        point(3).x("time")
    assert point(3, at=T0).x("time") == T0.timestamp()


def test_every_view_is_json() -> None:
    stats = from_replay_csv(CSV)
    everything = to_json(stats, "all")
    assert set(everything) == set(VIEWS) - {"all"}
    assert everything["mix"][0]["llm"] == 0.75
    assert everything["cost"]["points"][0] == {
        "documents": 2,
        "at": None,
        "jev": 0.002,
        "llm": 0.04,
    }
    with pytest.raises(KeyError):
        to_json(stats, "nope")

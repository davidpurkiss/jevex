import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from jevex import Document, Extractor, Field, Pipeline, RunBudget
from jevex.budgets import RunLedger
from jevex.extractor import ExtractionResult
from jevex.jev import JevBackendError, JevError
from jevex.llm import LLMError
from jevex.monitoring import (
    DriftWindow,
    FieldDrift,
    Headroom,
    budget_headroom,
    observations,
    process_headroom,
    run_headroom,
    store_error,
)
from jevex.pipeline import Context
from jevex.results import FieldMeta, build_extracted
from jevex.schema import SchemaSpec
from jevex.store import LedgerError, SpendEntry, SQLiteStore, StoreError, open_store
from jevex.testing import FakeJev


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")
    power_ps: int | None = Field(None, description="Power in PS")


class Trim(BaseModel):
    """A trim level."""

    name: str = Field(description="Trim name")


class CarModel(BaseModel):
    """A car model with trims."""

    model: str = Field(description="Model name")
    trims: list[Trim] = Field(default_factory=list[Trim], description="Trim levels")


@dataclass
class Sets:
    """Records the given fields on the document's entity."""

    fields: dict[str, FieldMeta]
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        for name, meta in self.fields.items():
            ctx.schemas["Car"].set_field("document", name, meta)


@dataclass
class Fails:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        ctx.schemas["Car"].set_field("document", "model", FieldMeta(value="Golf", method="jev"))
        raise JevBackendError("down")


async def extract(stage: Any) -> ExtractionResult:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([stage]))
    return await ex.extract(Document.from_bytes(b"<p>Golf</p>"))


def jev(value: Any, confidence: float) -> FieldMeta:
    return FieldMeta(value=value, method="jev", confidence=confidence)


async def test_drift_counts_none_fallback_and_confidence_per_field() -> None:
    window = DriftWindow(size=10)
    window.add(await extract(Sets({"model": jev("Golf", 0.9), "power_ps": jev(150, 0.7)})))
    window.add(await extract(Sets({"model": jev("Polo", 0.5)})))
    llm = FieldMeta(value=110, method="llm", confidence=0.95)
    window.add(await extract(Sets({"model": jev("Up", 0.7), "power_ps": llm})))
    assert window.documents == 3
    model, power = window.fields()
    assert (model.field, model.records, model.found, model.llm) == ("Car.model", 3, 3, 0)
    assert model.mean_confidence == pytest.approx(0.7)
    assert (power.field, power.records, power.found, power.llm) == ("Car.power_ps", 3, 2, 1)
    assert power.mean_confidence == pytest.approx(0.825)
    assert (model.none_rate, model.fallback_rate) == (0.0, 0.0)
    assert power.none_rate == pytest.approx(1 / 3)
    assert power.fallback_rate == 0.5


async def test_the_window_keeps_only_the_last_documents() -> None:
    window = DriftWindow(size=2)
    for meta in (jev("Golf", 0.1), jev("Polo", 0.8), jev("Up", 0.6)):
        window.add(await extract(Sets({"model": meta})))
    assert window.documents == 2
    [model, _] = window.fields()
    assert (model.records, model.mean_confidence) == (2, pytest.approx(0.7))


async def test_a_failed_document_is_not_drift() -> None:
    window = DriftWindow()
    result = await extract(Fails())
    assert result.status == "failed"
    window.add(result)
    assert (window.documents, window.fields()) == (0, [])


def test_empty_windows_and_bad_sizes() -> None:
    assert DriftWindow().fields() == []
    empty = FieldDrift(field="Car.model", records=0, found=0, llm=0, mean_confidence=None)
    assert (empty.none_rate, empty.fallback_rate) == (0.0, 0.0)
    with pytest.raises(ValueError, match="size must be at least 1"):
        DriftWindow(size=0)


def test_children_are_observed_but_not_the_nested_field() -> None:
    parent = SchemaSpec.from_model(CarModel)
    [child_spec] = parent.children()
    child = build_extracted(child_spec, "GTI", {"name": FieldMeta(value="GTI", method="jev")})
    record = build_extracted(
        parent, "document", {"model": FieldMeta(value="Golf")}, children={"trims": [child]}
    )
    assert [(o.field, o.found) for o in observations([record])] == [
        ("CarModel.model", True),
        ("CarModel.trims.name", True),
    ]
    # A parent without children doesn't count its nested field as missing.
    childless = build_extracted(parent, "document", {"model": FieldMeta(value="Polo")})
    assert [(o.field, o.found) for o in observations([childless])] == [("CarModel.model", True)]


@dataclass
class Finds:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await asyncio.sleep(0)


async def test_a_document_that_found_nothing_is_all_none() -> None:
    window = DriftWindow(schemas=[SchemaSpec.from_model(Car)])
    empty = await extract(Finds())
    assert (empty.records, empty.meta.active_schemas) == ([], ["Car"])
    window.add(empty)
    window.add(await extract(Sets({"model": jev("Golf", 0.9)})))
    model, power = window.fields()
    assert (model.field, model.records, model.found) == ("Car.model", 2, 1)
    assert model.none_rate == 0.5
    assert (power.field, power.records, power.found) == ("Car.power_ps", 2, 0)
    # Without the schema's spec, an empty document has no fields to count.
    blind = DriftWindow()
    blind.add(empty)
    assert (blind.documents, blind.fields()) == (1, [])


async def test_headroom_of_the_run_budget_and_the_process_caps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SQLiteStore(":memory:")
    ledger = RunLedger(
        RunBudget(max_spend=5.0, period="day", max_jev_spend=1.0), store, run_id="r1"
    )
    await store.record_spend(SpendEntry(amount_usd=2.0, kind="llm", run_id="r1"))
    await store.record_spend(SpendEntry(amount_usd=0.25, kind="jev", run_id="r1"))
    ledger_file = tmp_path / "ledger"
    ledger_file.write_text("jev 0.1\nllm 3\n")
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger_file))
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "0.5")
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "2")
    found = await budget_headroom(ledger)
    assert [(h.scope, h.kind, h.period, h.limit_usd) for h in found] == [
        ("run", "llm", "day", 5.0),
        ("run", "jev", "day", 1.0),
        ("process", "jev", "process", 0.5),
        ("process", "llm", "process", 2.0),
    ]
    assert [h.spent_usd for h in found] == pytest.approx([2.0, 0.25, 0.1, 3.0])
    assert [h.remaining_usd for h in found] == pytest.approx([3.0, 0.75, 0.4, 0.0])
    await store.aclose()


async def test_run_and_process_headroom_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteStore(":memory:")
    ledger = RunLedger(RunBudget(max_jev_spend=1.0, period="run"), store, run_id="r1")
    await store.record_spend(SpendEntry(amount_usd=0.25, kind="jev", run_id="r1"))
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "2")
    monkeypatch.delenv("JEVEX_JEV_MAX_COST_USD", raising=False)
    assert await run_headroom(ledger) == [Headroom("run", "jev", "run", 1.0, 0.25)]
    assert process_headroom("llm") == Headroom("process", "llm", "process", 2.0, 0.0)
    assert process_headroom("jev") is None
    await store.aclose()


async def test_a_misconfigured_process_cap_raises_its_own_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "lots")
    with pytest.raises(JevError, match="must be a number of US dollars, got 'lots'"):
        process_headroom("jev")
    with pytest.raises(JevError):  # budget_headroom raises what its parts raise
        await budget_headroom(None)
    ledger_file = tmp_path / "ledger"
    ledger_file.write_text("llm ???\n")
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger_file))
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "2")
    with pytest.raises(LLMError, match="has a bad line 1"):
        process_headroom("llm")


async def test_no_caps_no_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVEX_JEV_MAX_COST_USD", raising=False)
    monkeypatch.delenv("JEVEX_LLM_MAX_COST_USD", raising=False)
    assert await budget_headroom(None) == []
    assert await budget_headroom(RunLedger()) == []  # no run budget
    store = SQLiteStore(":memory:")
    ledger = RunLedger(RunBudget(llm_rpm=10), store)  # a budget without spend caps
    assert await budget_headroom(ledger) == []
    await store.aclose()


class Failing(SQLiteStore):
    def __init__(self, fail: Callable[[], Any]) -> None:
        super().__init__(":memory:")
        self.fail = fail

    async def spend(self, **_: Any) -> float:
        return await self.fail()

    async def disabled_generator_ids(self) -> set[str]:
        await self.fail()
        return set()


async def test_store_error_says_why_the_store_does_not_answer() -> None:
    store = open_store(":memory:")
    assert await store_error(store) is None
    await store.aclose()

    async def broken() -> float:
        raise StoreError("database is locked")

    failing = Failing(broken)
    assert await store_error(failing) == "StoreError: database is locked"
    with pytest.raises(LedgerError):  # the headroom read raises it for the caller
        await budget_headroom(RunLedger(RunBudget(max_spend=1), failing))
    await failing.aclose()

    async def hangs() -> float:
        await asyncio.sleep(10)
        return 0.0

    slow = Failing(hangs)
    assert await store_error(slow, wait_s=0.01) == "the store didn't answer within 0.01s"
    await slow.aclose()

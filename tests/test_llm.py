from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel

from jevex.llm import (
    LLM,
    LLMBudgetExceededError,
    LLMError,
    ModelPrice,
    check_budget,
    cost,
    gemini_flash_3x_price,
    process_llm_cost,
    reset_process_llm_cost,
)
from jevex.testing import CassetteMissError, FakeLLM, LLMCassette, UnscriptedQuestionError


class Book(BaseModel):
    title: str


@pytest.fixture(autouse=True)
def fresh_spend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("JEVEX_LLM_MAX_COST_USD", raising=False)
    reset_process_llm_cost()
    yield
    reset_process_llm_cost()


# --- costing and budgets ---------------------------------------------------------------


def test_cost_from_the_price_table() -> None:
    assert cost("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert cost("claude-sonnet-5-5", 1_000, 500) == pytest.approx(0.007)
    assert cost("unknown-model", 10, 10) is None
    assert cost("mine", 1_000_000, 0, {"mine": ModelPrice(3, 9)}) == pytest.approx(3.0)
    assert cost("gemini-3.5-flash", 1_000_000, 1_000_000) == pytest.approx(10.5)


def test_dated_snapshot_is_costed_at_its_base_model_price() -> None:
    assert cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == pytest.approx(6.0)
    mine = {"mine": ModelPrice(3, 9)}
    assert cost("mine-20260115", 1_000_000, 0, mine) == pytest.approx(3.0)


def test_exact_entry_wins_over_the_dated_snapshot_base_name() -> None:
    prices = {"mine": ModelPrice(3, 9), "mine-20260115": ModelPrice(1, 2)}
    assert cost("mine-20260115", 1_000_000, 1_000_000, prices) == pytest.approx(3.0)
    assert cost("mine", 1_000_000, 1_000_000, prices) == pytest.approx(12.0)


@pytest.mark.parametrize(
    "model",
    [
        "unknown-model-20251001",  # dated, but the base isn't priced either
        "claude-haiku-4-5-2025100",  # seven digits: not a snapshot date
        "claude-haiku-4-5-20251001-beta",  # the date isn't the suffix
        "claude-haiku-4-5-2025-10-01",  # dashed dates aren't this form
        "claude-haiku-4",  # a prefix of a priced model
    ],
)
def test_unpriced_models_still_cost_none(model: str) -> None:
    assert cost(model, 10, 10) is None


def test_gemini_flash_3x_promotion_ends_with_2026() -> None:
    assert gemini_flash_3x_price(date(2026, 12, 31)) == ModelPrice(0.75, 3.75)
    assert gemini_flash_3x_price(date(2027, 1, 1)) == ModelPrice(1.50, 7.50)


async def test_process_budget_stops_further_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLM(lambda p, s: {"title": "Dune"}, price=(1_000_000, 1_000_000))  # $1 per token
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "5")
    await llm.structured("abc", Book)  # ~2 + ~5 tokens: $7, over the cap afterwards
    assert process_llm_cost() > 5
    with pytest.raises(LLMBudgetExceededError, match="spend cap"):
        await llm.structured("abc", Book)
    assert len(llm.calls) == 1


async def test_ledger_cap_counts_spend_from_other_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "run.ledger"
    ledger.write_text("llm 4\njev 9\n")
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "5")
    llm = FakeLLM(lambda p, s: {"title": "Dune"}, price=(1_000_000, 1_000_000))
    await llm.structured("abc", Book)  # $4 spent elsewhere: still under the cap
    assert ledger.read_text() == "llm 4\njev 9\nllm 6.000000000\n"
    with pytest.raises(LLMBudgetExceededError, match=r"\$10\.0000 of \$5\.00"):
        await llm.structured("abc", Book)
    assert len(llm.calls) == 1
    assert process_llm_cost() == pytest.approx(6)


async def test_free_calls_leave_the_ledger_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "run.ledger"
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    await FakeLLM(lambda p, s: {"title": "Dune"}).structured("abc", Book)
    assert not ledger.exists()


async def test_ledger_in_a_missing_dir_blocks_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(tmp_path / "nope" / "run.ledger"))
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "5")
    llm = FakeLLM(lambda p, s: {"title": "Dune"})
    with pytest.raises(LLMError, match="can't use"):
        await llm.structured("abc", Book)
    assert llm.calls == []


async def test_ledger_never_loosens_the_process_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "5")
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(tmp_path / "a.ledger"))
    llm = FakeLLM(lambda p, s: {"title": "Dune"}, price=(1_000_000, 1_000_000))
    await llm.structured("abc", Book)  # $6
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(tmp_path / "b.ledger"))  # empty
    with pytest.raises(LLMBudgetExceededError):
        await llm.structured("abc", Book)


def test_unreadable_ledger_blocks_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = tmp_path / "run.ledger"
    ledger.write_text("llm\n")
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "5")
    with pytest.raises(LLMError, match="bad line 1"):
        check_budget()


# --- FakeLLM and cassettes -------------------------------------------------------------


async def test_fake_llm_queue_and_function() -> None:
    queued = FakeLLM([{"title": "Dune"}, Book(title="Emma")])
    assert (await queued.structured("x", Book)).output == Book(title="Dune")
    assert (await queued.structured("y", Book)).output == Book(title="Emma")
    with pytest.raises(UnscriptedQuestionError):
        await queued.structured("z", Book)
    echo = FakeLLM(lambda prompt, schema: {"title": prompt.upper()})
    response = await echo.structured("dune", Book)
    assert response.output.title == "DUNE"
    assert response.usage.input_tokens > 0
    assert isinstance(echo, LLM)


async def test_invalid_fake_output_is_an_llm_error() -> None:
    with pytest.raises(LLMError, match="doesn't match Book"):
        await FakeLLM([{"name": "wrong"}]).structured("x", Book)


async def test_llm_cassette_records_then_replays(tmp_path: Path) -> None:
    path = tmp_path / "llm.json"
    inner = FakeLLM([{"title": "Dune"}], model="recorded-model", price=(1, 1))
    recorded = await LLMCassette(path, inner, record=True).structured("Title: Dune", Book)
    replayed = await LLMCassette(path).structured("Title: Dune", Book)
    assert replayed == recorded
    assert len(inner.calls) == 1
    with pytest.raises(CassetteMissError, match="JEVEX_RECORD=1"):
        await LLMCassette(path).structured("Another prompt", Book)


class _Status(Exception):
    def __init__(self, **attrs: object) -> None:
        super().__init__("x")
        for name, value in attrs.items():
            setattr(self, name, value)


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"status_code": 429}, True),  # Anthropic, OpenAI, LiteLLM
        ({"code": 429}, True),  # Gemini
        ({"status_code": 500}, False),
        ({"code": "rate_limit"}, False),
        ({}, False),
    ],
)
def test_rate_limited(attrs: dict[str, object], expected: bool) -> None:
    from jevex.llm import rate_limited

    assert rate_limited(_Status(**attrs)) is expected

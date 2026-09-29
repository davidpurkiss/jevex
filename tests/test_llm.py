from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from jevex.llm import (
    LLM,
    LLMBudgetExceededError,
    LLMError,
    ModelPrice,
    cost,
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


async def test_process_budget_stops_further_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLM(lambda p, s: {"title": "Dune"}, price=(1_000_000, 1_000_000))  # $1 per token
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "5")
    await llm.structured("abc", Book)  # ~2 + ~5 tokens: $7, over the cap afterwards
    assert process_llm_cost() > 5
    with pytest.raises(LLMBudgetExceededError, match="spend cap"):
        await llm.structured("abc", Book)
    assert len(llm.calls) == 1


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

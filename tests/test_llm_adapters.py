"""Adapter tests against mocked HTTP (Anthropic, OpenAI) and LiteLLM's mock_response.

Skipped when the extras aren't installed (``uv sync --all-extras``); CI installs them.
"""

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest

for extra in ("anthropic", "openai", "litellm"):
    pytest.importorskip(extra)

import anthropic  # noqa: E402
import httpx2  # noqa: E402
import openai  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from jevex.llm import (  # noqa: E402
    LLMError,
    LLMRefusalError,
    ModelPrice,
    process_llm_cost,
    reset_process_llm_cost,
)
from jevex.llm.anthropic import FALLBACK_BETA, AnthropicLLM  # noqa: E402
from jevex.llm.litellm import LiteLLM  # noqa: E402
from jevex.llm.openai import OpenAILLM  # noqa: E402


class Book(BaseModel):
    title: str


type Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture(autouse=True)
def fresh_spend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("JEVEX_LLM_MAX_COST_USD", raising=False)
    reset_process_llm_cost()
    yield
    reset_process_llm_cost()


# --- Anthropic -------------------------------------------------------------------------


def anthropic_client(handler: Handler) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def message(
    text: str | None, *, stop: str = "end_turn", model: str = "claude-opus-5-5"
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [] if text is None else [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


async def test_anthropic_structured_output_with_fallbacks_and_effort() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["beta"] = request.headers.get("anthropic-beta")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=message('{"title": "Dune"}'))

    llm = AnthropicLLM(client=anthropic_client(handler), effort="low", system="Extract.")
    response = await llm.structured("Title: Dune", Book)
    assert response.output == Book(title="Dune")
    assert response.model == "claude-opus-5-5"
    assert response.usage.cost == pytest.approx((12 * 4 + 7 * 20) / 1_000_000)
    assert process_llm_cost() == pytest.approx(response.usage.cost)
    assert seen["beta"] == FALLBACK_BETA
    body = seen["body"]
    assert body["fallbacks"] == "default"
    assert body["system"] == "Extract."
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"


async def test_anthropic_without_fallbacks_uses_the_plain_endpoint() -> None:
    urls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        urls.append(str(request.url))
        assert "fallbacks" not in json.loads(request.content)
        return httpx2.Response(200, json=message('{"title": "Dune"}'))

    await AnthropicLLM(client=anthropic_client(handler), fallbacks=False).structured("x", Book)
    assert urls == ["https://api.anthropic.com/v1/messages"]


async def test_anthropic_cost_uses_the_model_that_served() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message('{"title": "Dune"}', model="claude-sonnet-5-5"))

    response = await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)
    assert response.model == "claude-sonnet-5-5"
    assert response.usage.cost == pytest.approx((12 * 2 + 7 * 10) / 1_000_000)


async def test_anthropic_refusal() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message(None, stop="refusal"))

    with pytest.raises(LLMRefusalError, match="declined"):
        await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)


async def test_anthropic_output_that_fails_the_schema_is_an_llm_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message('{"name": "wrong"}'))

    with pytest.raises(LLMError):
        await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)


async def test_anthropic_api_errors_are_llm_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}
        )

    with pytest.raises(LLMError, match="Anthropic API error"):
        await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)


# --- OpenAI ----------------------------------------------------------------------------


def openai_response(text: str) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-test",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
    }


def openai_client(handler: Handler) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )


async def test_openai_structured_output() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=openai_response('{"title": "Dune"}'))

    llm = OpenAILLM(
        "gpt-test", client=openai_client(handler), prices={"gpt-test": ModelPrice(1, 2)}
    )
    response = await llm.structured("Title: Dune", Book)
    assert response.output == Book(title="Dune")
    assert response.usage.cost == pytest.approx((10 * 1 + 4 * 2) / 1_000_000)
    assert seen["body"]["text"]["format"]["type"] == "json_schema"


async def test_openai_unknown_price_is_none() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=openai_response('{"title": "Dune"}'))

    response = await OpenAILLM("gpt-test", client=openai_client(handler)).structured("x", Book)
    assert response.usage.cost is None


async def test_openai_errors_are_llm_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401, json={"error": {"message": "bad key", "type": "invalid_request_error"}}
        )

    with pytest.raises(LLMError, match="OpenAI API error"):
        await OpenAILLM("gpt-test", client=openai_client(handler)).structured("x", Book)


# --- LiteLLM ---------------------------------------------------------------------------


@pytest.fixture
def local_litellm_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use LiteLLM's bundled price map; never fetch it (the network is blocked in tests).
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.mark.usefixtures("local_litellm_prices")
async def test_litellm_structured_output_with_mock_response() -> None:
    llm = LiteLLM("ollama/llama3.1", mock_response='{"title": "Dune"}')
    response = await llm.structured("Title: Dune", Book)
    assert response.output == Book(title="Dune")
    assert response.model == "ollama/llama3.1"


@pytest.mark.usefixtures("local_litellm_prices")
async def test_litellm_explicit_prices() -> None:
    llm = LiteLLM(
        "ollama/mine", mock_response='{"title": "Dune"}', prices={"ollama/mine": ModelPrice(1, 1)}
    )
    response = await llm.structured("Title: Dune", Book)
    assert response.usage.cost is not None


@pytest.mark.usefixtures("local_litellm_prices")
async def test_litellm_invalid_output_is_an_llm_error() -> None:
    with pytest.raises(LLMError):
        await LiteLLM("ollama/llama3.1", mock_response="not json").structured("x", Book)

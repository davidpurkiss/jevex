"""Adapter tests against mocked HTTP (Anthropic, OpenAI, Gemini) and LiteLLM's mock_response.

Skipped when the extras aren't installed (``uv sync --all-extras``); CI installs them.
"""

import base64
import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest

for extra in ("anthropic", "openai", "google.genai", "litellm"):
    pytest.importorskip(extra)

import anthropic  # noqa: E402
import httpx2  # noqa: E402
import openai  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types as genai_types  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from jevex.llm import (  # noqa: E402
    LLMBudgetExceededError,
    LLMError,
    LLMImage,
    LLMRefusalError,
    ModelPrice,
    process_llm_cost,
    reset_process_llm_cost,
)
from jevex.llm.anthropic import FALLBACK_BETA, AnthropicLLM  # noqa: E402
from jevex.llm.gemini import GeminiLLM  # noqa: E402
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


async def test_anthropic_dated_snapshot_is_costed_at_its_base_price() -> None:
    body = message('{"title": "Dune"}', model="claude-haiku-4-5-20251001")
    body["usage"]["iterations"] = [
        {"type": "message", "model": "claude-opus-5-5", "input_tokens": 100, "output_tokens": 5},
        {
            "type": "fallback_message",
            "model": "claude-haiku-4-5-20251001",
            "input_tokens": 12,
            "output_tokens": 7,
        },
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body)

    llm = AnthropicLLM("claude-opus-5-5", client=anthropic_client(handler))
    response = await llm.structured("x", Book)
    assert response.model == "claude-haiku-4-5-20251001"
    assert response.usage.cost == pytest.approx((100 * 4 + 5 * 20 + 12 * 1 + 7 * 5) / 1_000_000)
    assert process_llm_cost() == pytest.approx(response.usage.cost)


async def test_anthropic_cost_sums_fallback_attempts_at_their_own_prices() -> None:
    body = message('{"title": "Dune"}', model="claude-sonnet-5-5")
    body["usage"]["iterations"] = [
        {"type": "message", "model": "claude-opus-5-5", "input_tokens": 100, "output_tokens": 5},
        {
            "type": "fallback_message",
            "model": "claude-sonnet-5-5",
            "input_tokens": 12,
            "output_tokens": 7,
        },
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body)

    response = await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)
    assert response.usage.input_tokens == 112
    assert response.usage.output_tokens == 12
    assert response.usage.cost == pytest.approx((100 * 4 + 5 * 20 + 12 * 2 + 7 * 10) / 1_000_000)


async def test_anthropic_iteration_without_a_model_is_priced_as_the_serving_model() -> None:
    body = message('{"title": "Dune"}')
    body["usage"]["iterations"] = [{"type": "message", "input_tokens": 12, "output_tokens": 7}]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body)

    response = await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)
    assert response.usage.cost == pytest.approx((12 * 4 + 7 * 20) / 1_000_000)


async def test_anthropic_refusal_still_records_usage() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message(None, stop="refusal"))

    with pytest.raises(LLMRefusalError, match="declined"):
        await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)
    assert process_llm_cost() > 0


async def test_anthropic_truncated_output_is_an_llm_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message('{"title": "Du', stop="max_tokens"))

    with pytest.raises(LLMError, match="max_tokens"):
        await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)


async def test_spent_cap_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(str(request.url))
        return httpx2.Response(200, json=message('{"title": "Dune"}'))

    llm = AnthropicLLM(client=anthropic_client(handler))
    await llm.structured("x", Book)
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "0.0000001")
    with pytest.raises(LLMBudgetExceededError):
        await llm.structured("x", Book)
    assert len(calls) == 1


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
    assert process_llm_cost() > 0  # the failed call still counts


async def test_anthropic_api_errors_are_llm_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}
        )

    with pytest.raises(LLMError, match="Anthropic API error"):
        await AnthropicLLM(client=anthropic_client(handler)).structured("x", Book)


# --- OpenAI ----------------------------------------------------------------------------


def openai_response(text: str, *, refusal: bool = False) -> dict[str, Any]:
    content: dict[str, Any] = (
        {"type": "refusal", "refusal": "I can't help with that."}
        if refusal
        else {"type": "output_text", "text": text, "annotations": []}
    )
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
                "content": [content],
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


async def test_openai_invalid_output_is_an_llm_error_and_still_costs() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=openai_response('{"name": "wrong"}'))

    llm = OpenAILLM(
        "gpt-test", client=openai_client(handler), prices={"gpt-test": ModelPrice(1, 2)}
    )
    with pytest.raises(LLMError, match="doesn't match Book"):
        await llm.structured("x", Book)
    assert process_llm_cost() == pytest.approx((10 * 1 + 4 * 2) / 1_000_000)


async def test_openai_refusal() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=openai_response("", refusal=True))

    with pytest.raises(LLMRefusalError, match="can't help"):
        await OpenAILLM("gpt-test", client=openai_client(handler)).structured("x", Book)


async def test_openai_errors_are_llm_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401, json={"error": {"message": "bad key", "type": "invalid_request_error"}}
        )

    with pytest.raises(LLMError, match="OpenAI API error"):
        await OpenAILLM("gpt-test", client=openai_client(handler)).structured("x", Book)


# --- Gemini ----------------------------------------------------------------------------


def gemini_client(handler: Handler) -> genai.Client:
    return genai.Client(
        api_key="test-key",
        http_options=genai_types.HttpOptions(
            httpx_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        ),
    )


def gemini_response(
    text: str | None,
    *,
    finish: str = "STOP",
    model: str = "gemini-3.5-flash",
    thoughts: int = 0,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": "Reading the title.", "thought": True}]
    if text is not None:
        parts.append({"text": text})
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": finish, "index": 0}
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": thoughts,
            "totalTokenCount": 19 + thoughts,
        },
        "modelVersion": model,
        "responseId": "resp_1",
    }


async def test_gemini_structured_output_sends_the_json_schema() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=gemini_response('{"title": "Dune"}', thoughts=5))

    llm = GeminiLLM(
        "gemini-3.5-flash",
        client=gemini_client(handler),
        system_instruction="Extract.",
        thinking_level="low",
        max_output_tokens=256,
    )
    response = await llm.structured("Title: Dune", Book)
    assert response.output == Book(title="Dune")  # the thought part isn't output
    assert response.model == "gemini-3.5-flash"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 12)
    # Thinking tokens are billed as output.
    assert response.usage.cost == pytest.approx((12 * 1.50 + 12 * 9.00) / 1_000_000)
    assert process_llm_cost() == pytest.approx(response.usage.cost)
    assert seen["url"].endswith("/models/gemini-3.5-flash:generateContent")
    body = seen["body"]
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Title: Dune"}]}]
    assert body["systemInstruction"]["parts"] == [{"text": "Extract."}]
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == Book.model_json_schema()
    assert list(config["thinkingConfig"].values()) == ["LOW"]
    assert config["maxOutputTokens"] == 256


async def test_gemini_defaults_send_no_optional_config() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=gemini_response('{"title": "Dune"}'))

    await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)
    assert "systemInstruction" not in seen["body"]
    assert set(seen["body"]["generationConfig"]) == {"responseMimeType", "responseJsonSchema"}


async def test_gemini_cost_uses_the_requested_model_not_the_snapshot() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = gemini_response('{"title": "Dune"}', model="gemini-2.5-flash-preview-09-2025")
        return httpx2.Response(200, json=body)

    llm = GeminiLLM("models/gemini-2.5-flash", client=gemini_client(handler))
    response = await llm.structured("x", Book)
    assert response.model == "gemini-2.5-flash-preview-09-2025"
    assert response.usage.cost == pytest.approx((12 * 0.30 + 7 * 2.50) / 1_000_000)


async def test_gemini_unknown_price_is_none() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=gemini_response('{"title": "Dune"}'))

    response = await GeminiLLM("gemini-test", client=gemini_client(handler)).structured("x", Book)
    assert response.usage.cost is None


async def test_gemini_explicit_prices() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=gemini_response('{"title": "Dune"}'))

    llm = GeminiLLM(
        "gemini-test", client=gemini_client(handler), prices={"gemini-test": ModelPrice(1, 2)}
    )
    response = await llm.structured("x", Book)
    assert response.usage.cost == pytest.approx((12 * 1 + 7 * 2) / 1_000_000)


async def test_gemini_blocked_prompt_is_a_refusal_and_still_costs() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
                "usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 12},
                "modelVersion": "gemini-3.5-flash",
            },
        )

    with pytest.raises(LLMRefusalError, match=r"blocked the prompt.*PROHIBITED_CONTENT"):
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)
    assert process_llm_cost() == pytest.approx(12 * 1.50 / 1_000_000)


@pytest.mark.parametrize(
    "reason", ["SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"]
)
async def test_gemini_safety_stop_is_a_refusal(reason: str) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=gemini_response(None, finish=reason))

    with pytest.raises(LLMRefusalError, match=f"declined.*{reason}"):
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)
    assert process_llm_cost() > 0


async def test_gemini_truncated_output_is_an_llm_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=gemini_response('{"title": "Du', finish="MAX_TOKENS"))

    with pytest.raises(LLMError, match="truncated") as info:
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)
    assert not isinstance(info.value, LLMRefusalError)


async def test_gemini_empty_reply_is_an_llm_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=gemini_response(None))

    with pytest.raises(LLMError, match="no output"):
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)


async def test_gemini_no_candidates_is_an_llm_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"usageMetadata": {"promptTokenCount": 12}})

    with pytest.raises(LLMError, match="no candidates"):
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)


async def test_gemini_invalid_output_is_an_llm_error_and_still_costs() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=gemini_response('{"name": "wrong"}'))

    with pytest.raises(LLMError, match="doesn't match Book"):
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)
    assert process_llm_cost() == pytest.approx((12 * 1.50 + 7 * 9.00) / 1_000_000)


async def test_gemini_api_errors_are_llm_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400,
            json={"error": {"code": 400, "message": "API key not valid", "status": "INVALID"}},
        )

    with pytest.raises(LLMError, match=r"Gemini API error.*API key not valid"):
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)
    assert process_llm_cost() == 0


async def test_gemini_spent_cap_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(str(request.url))
        return httpx2.Response(200, json=gemini_response('{"title": "Dune"}'))

    llm = GeminiLLM("gemini-3.5-flash", client=gemini_client(handler))
    await llm.structured("x", Book)
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "0.0000001")
    with pytest.raises(LLMBudgetExceededError):
        await llm.structured("x", Book)
    assert len(calls) == 1


async def test_gemini_transport_errors_are_llm_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(LLMError, match="connection refused"):
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)


async def test_gemini_non_json_reply_is_an_llm_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text="<html>proxy error</html>")

    with pytest.raises(LLMError, match="Gemini API error"):
        await GeminiLLM("gemini-3.5-flash", client=gemini_client(handler)).structured("x", Book)


async def test_gemini_aclose_closes_the_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []
    client = gemini_client(lambda request: httpx2.Response(200))

    async def aclose() -> None:
        closed.append(True)

    monkeypatch.setattr(client.aio, "aclose", aclose)
    await GeminiLLM("gemini-3.5-flash", client=client).aclose()
    assert closed == [True]


# --- LiteLLM ---------------------------------------------------------------------------
# Not ``ollama/`` ids: LiteLLM asks a local Ollama server for their model info. The price
# map is the bundled copy (conftest sets LITELLM_LOCAL_MODEL_COST_MAP).


async def test_litellm_structured_output_with_mock_response() -> None:
    llm = LiteLLM("openai/gpt-test", mock_response='{"title": "Dune"}')
    response = await llm.structured("Title: Dune", Book)
    assert response.output == Book(title="Dune")
    assert response.model == "openai/gpt-test"
    assert response.usage.cost is None  # not in LiteLLM's price map, no prices given


async def test_litellm_explicit_prices() -> None:
    llm = LiteLLM(
        "openai/mine",
        mock_response='{"title": "Dune"}',
        prices={"openai/mine": ModelPrice(1, 1)},
    )
    response = await llm.structured("Title: Dune", Book)
    # LiteLLM's mock reports 10 prompt and 20 completion tokens.
    assert response.usage.cost == pytest.approx((10 * 1 + 20 * 1) / 1_000_000)


async def test_litellm_invalid_output_is_an_llm_error() -> None:
    with pytest.raises(LLMError):
        await LiteLLM("openai/gpt-test", mock_response="not json").structured("x", Book)


# --- images ------------------------------------------------------------------------------

IMAGE = LLMImage(b"\x89PNG fake", "image/png")
IMAGE_B64 = base64.b64encode(IMAGE.content).decode()


def test_llm_images_are_png_jpeg_or_webp() -> None:
    assert IMAGE.data_uri == f"data:image/png;base64,{IMAGE_B64}"
    with pytest.raises(ValueError, match="image/bmp"):
        LLMImage(b"BM", "image/bmp")


async def test_anthropic_sends_images_before_the_prompt() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=message('{"title": "Dune"}'))

    llm = AnthropicLLM(client=anthropic_client(handler))
    await llm.structured("Title?", Book, images=[IMAGE])
    assert seen["body"]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": IMAGE_B64},
                },
                {"type": "text", "text": "Title?"},
            ],
        }
    ]
    await llm.structured("Title?", Book)
    assert seen["body"]["messages"] == [{"role": "user", "content": "Title?"}]


async def test_openai_sends_images_as_data_uris() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=openai_response('{"title": "Dune"}'))

    llm = OpenAILLM("gpt-test", client=openai_client(handler))
    await llm.structured("Title?", Book, images=[IMAGE])
    assert seen["body"]["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": IMAGE.data_uri, "detail": "auto"},
                {"type": "input_text", "text": "Title?"},
            ],
        }
    ]
    await llm.structured("Title?", Book)
    assert seen["body"]["input"] == "Title?"


async def test_gemini_sends_images_as_inline_parts() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=gemini_response('{"title": "Dune"}'))

    llm = GeminiLLM("gemini-3.5-flash", client=gemini_client(handler))
    await llm.structured("Title?", Book, images=[IMAGE])
    # The SDK puts the image parts and the prompt in one user turn.
    assert seen["body"]["contents"] == [
        {
            "role": "user",
            "parts": [
                {"inlineData": {"data": IMAGE_B64, "mime_type": "image/png"}},
                {"text": "Title?"},
            ],
        }
    ]


async def test_litellm_sends_images_in_the_openai_chat_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    seen: list[Any] = []
    real: Any = litellm.acompletion  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    async def capture(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return await real(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", capture)
    llm = LiteLLM("openai/gpt-test", mock_response='{"title": "Dune"}')
    await llm.structured("Title?", Book, images=[IMAGE])
    await llm.structured("Title?", Book)
    assert seen == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": IMAGE.data_uri}},
                    {"type": "text", "text": "Title?"},
                ],
            }
        ],
        [{"role": "user", "content": "Title?"}],
    ]

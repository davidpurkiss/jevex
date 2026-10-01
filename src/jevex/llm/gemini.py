"""Gemini via Google's ``google-genai`` SDK (``pip install jevex[gemini]``).

Structured output sends the pydantic model's JSON schema as ``response_json_schema``
(with ``response_mime_type="application/json"``), and the reply is validated here, after
usage is recorded. Credentials come from ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` (or the
Vertex AI environment variables) unless a client is passed in.

Costs use :data:`~jevex.llm.PRICES`, the Gemini Developer API's standard paid-tier list
prices. Thinking tokens are billed as output. Not modelled: the higher Pro rates for
prompts over 200k tokens, implicit-cache discounts and Vertex AI's own price list (pass
``prices=`` for those).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from jevex.llm import (
    PRICES,
    LLMError,
    LLMRefusalError,
    LLMResponse,
    LLMUsage,
    ModelPrice,
    check_budget,
    cost,
    record,
    validate_output,
)

if TYPE_CHECKING:
    from google.genai import Client
    from google.genai.types import GenerateContentResponse
    from pydantic import BaseModel

ThinkingLevel = Literal["minimal", "low", "medium", "high"]

# Finish reasons where Gemini stopped because of a policy, not because it was done.
REFUSAL_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}
)


class GeminiLLM:
    """:class:`~jevex.llm.LLM` backed by a Gemini model (the model id is required).

    ``thinking_level`` trades depth for cost on Gemini 3 models; extraction prompts are
    short, so ``minimal`` or ``low`` usually suffice. Gemini 2.5 models don't take it.
    Usage (and cost, thinking tokens included) is recorded before the output is validated,
    so failed calls still count against ``JEVEX_LLM_MAX_COST_USD``.
    """

    def __init__(
        self,
        model: str,
        *,
        client: Client | None = None,
        system_instruction: str | None = None,
        thinking_level: ThinkingLevel | None = None,
        max_output_tokens: int | None = None,
        prices: dict[str, ModelPrice] | None = None,
    ) -> None:
        from google import genai

        self.model = model
        self.system_instruction = system_instruction
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.prices = prices if prices is not None else PRICES
        self._client = client or genai.Client()

    async def structured[T: BaseModel](self, prompt: str, schema: type[T]) -> LLMResponse[T]:
        from google.genai import types

        check_budget()
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            system_instruction=self.system_instruction,
            max_output_tokens=self.max_output_tokens,
            # No tools are sent; this also stops the SDK warning about AFC on every call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel(self.thinking_level.upper())
                )
                if self.thinking_level
                else None
            ),
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        # Not just ``errors.APIError``: transport failures surface as httpx, httpx2 or aiohttp
        # exceptions (whichever client the SDK picked), and a non-JSON body as a ValueError.
        except Exception as exc:
            raise LLMError(f"Gemini API error: {exc}") from exc

        served_by = response.model_version or self.model
        usage = self._usage(response)
        record(usage)
        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason is not None:
            raise LLMRefusalError(
                f"{served_by} blocked the prompt (reason: {feedback.block_reason.value})"
            )
        if not response.candidates:
            raise LLMError(f"{served_by} returned no candidates")
        candidate = response.candidates[0]
        reason = candidate.finish_reason.value if candidate.finish_reason else None
        if reason in REFUSAL_FINISH_REASONS:
            raise LLMRefusalError(f"{served_by} declined the request (finish reason: {reason})")
        if reason == "MAX_TOKENS":
            raise LLMError(f"{served_by} hit its output token limit; the output is truncated")
        parts = candidate.content.parts if candidate.content else None
        text = "".join(p.text for p in parts or [] if p.text and not p.thought)
        if not text:
            raise LLMError(f"{served_by} returned no output (finish reason: {reason})")
        return LLMResponse(output=validate_output(schema, text), usage=usage, model=served_by)

    def _usage(self, response: GenerateContentResponse) -> LLMUsage:
        """Tokens and cost, priced by the requested model.

        ``model_version`` can name a dated snapshot that isn't in the price table, so the
        requested id (without a ``models/`` style prefix) is the one looked up.
        """
        meta = response.usage_metadata
        in_tokens = out_tokens = 0
        if meta is not None:
            in_tokens = (meta.prompt_token_count or 0) + (meta.tool_use_prompt_token_count or 0)
            out_tokens = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
        model_id = self.model.rsplit("/", 1)[-1]
        return LLMUsage(in_tokens, out_tokens, cost(model_id, in_tokens, out_tokens, self.prices))

    async def aclose(self) -> None:
        await self._client.aio.aclose()

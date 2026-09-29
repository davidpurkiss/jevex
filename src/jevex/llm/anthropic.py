"""Claude via the Anthropic SDK (``pip install jevex[anthropic]``).

Structured output uses ``messages.parse(output_format=Model)``, so the response is
validated against the schema. Refusal fallbacks are on by default (server-side,
``fallbacks="default"``): if Claude declines, the API retries on a suitable fallback
model inside the same call. Credentials come from the environment (``ANTHROPIC_API_KEY``
or an ``ant auth login`` profile) unless a client is passed in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jevex.llm import (
    ANTHROPIC_MODEL,
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
    from anthropic import AsyncAnthropic
    from pydantic import BaseModel

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    """:class:`~jevex.llm.LLM` backed by Claude.

    ``effort`` (``low``..``max``) trades depth for cost; extraction prompts are short, so
    ``low`` or ``medium`` usually suffice. ``fallbacks`` needs the first-party Claude API;
    turn it off for proxies or other platforms.
    """

    def __init__(
        self,
        model: str = ANTHROPIC_MODEL,
        *,
        client: AsyncAnthropic | None = None,
        max_tokens: int = 16_000,
        effort: str | None = None,
        system: str | None = None,
        fallbacks: bool = True,
        prices: dict[str, ModelPrice] | None = None,
    ) -> None:
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.system = system
        self.fallbacks = fallbacks
        self.prices = prices if prices is not None else PRICES
        self._client = client or anthropic.AsyncAnthropic()

    async def structured[T: BaseModel](self, prompt: str, schema: type[T]) -> LLMResponse[T]:
        import anthropic
        from pydantic import ValidationError

        check_budget()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_format": schema,
        }
        if self.system:
            kwargs["system"] = self.system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            if self.fallbacks:
                response = await self._client.beta.messages.parse(
                    **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = await self._client.messages.parse(**kwargs)
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        except ValidationError as exc:  # the SDK validates the output against the schema
            raise LLMError(f"Claude's output doesn't match {schema.__name__}: {exc}") from exc

        served_by = str(response.model)
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=cost(
                served_by, response.usage.input_tokens, response.usage.output_tokens, self.prices
            ),
        )
        record(usage)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMRefusalError(f"{served_by} declined the request (category: {category})")
        if response.parsed_output is None:
            raise LLMError(
                f"{served_by} returned no structured output (stop: {response.stop_reason})"
            )
        return LLMResponse(
            output=validate_output(schema, response.parsed_output), usage=usage, model=served_by
        )

    async def aclose(self) -> None:
        await self._client.close()

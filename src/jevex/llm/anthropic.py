"""Claude via the Anthropic SDK (``pip install jevex[anthropic]``).

Structured output uses ``output_config.format`` with the model's JSON schema, and the reply
is validated against the model here, after usage is recorded. Refusal fallbacks are on by
default (server-side, ``fallbacks="default"``): if Claude declines, the API retries on a
suitable fallback model inside the same call. Credentials come from the environment
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile) unless a client is passed in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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


Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AnthropicLLM:
    """:class:`~jevex.llm.LLM` backed by Claude.

    ``effort`` trades depth for cost; extraction prompts are short, so ``low`` or
    ``medium`` usually suffice. ``fallbacks`` needs the first-party Claude API; turn it off
    for proxies or other platforms. Usage (and cost, including declined fallback attempts)
    is recorded before the output is validated, so failed calls still count against
    ``JEVEX_LLM_MAX_COST_USD``.
    """

    def __init__(
        self,
        model: str = ANTHROPIC_MODEL,
        *,
        client: AsyncAnthropic | None = None,
        max_tokens: int = 16_000,
        effort: Effort | None = None,
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

        check_budget()
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
        }
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if self.system:
            kwargs["system"] = self.system
        # ``Any``: the beta and GA responses share the fields read below, and ``**kwargs``
        # defeats the SDK's overloads anyway.
        create: Any = (
            self._client.beta.messages.create if self.fallbacks else self._client.messages.create
        )
        if self.fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        try:
            response: Any = await create(**kwargs)
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        served_by = str(response.model)
        usage = self._usage(response, served_by)
        record(usage)
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise LLMRefusalError(f"{served_by} declined the request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{served_by} hit max_tokens={self.max_tokens}; the output is truncated")
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text:
            raise LLMError(f"{served_by} returned no output (stop: {response.stop_reason})")
        return LLMResponse(output=validate_output(schema, text), usage=usage, model=served_by)

    def _usage(self, response: Any, served_by: str) -> LLMUsage:
        """Tokens and cost, summed over every attempt when fallbacks ran.

        ``usage.iterations`` is the per-attempt billing record: a declined attempt is billed
        at its own model's rates, not the fallback's.
        """
        iterations: list[Any] = getattr(response.usage, "iterations", None) or []
        attempts = [
            it for it in iterations if getattr(it, "type", None) in ("message", "fallback_message")
        ]
        if not attempts:
            in_t, out_t = response.usage.input_tokens, response.usage.output_tokens
            return LLMUsage(in_t, out_t, cost(served_by, in_t, out_t, self.prices))
        costs = [
            cost(str(it.model), it.input_tokens, it.output_tokens, self.prices) for it in attempts
        ]
        return LLMUsage(
            input_tokens=sum(it.input_tokens for it in attempts),
            output_tokens=sum(it.output_tokens for it in attempts),
            cost=None if any(c is None for c in costs) else sum(c for c in costs if c is not None),
        )

    async def aclose(self) -> None:
        await self._client.close()

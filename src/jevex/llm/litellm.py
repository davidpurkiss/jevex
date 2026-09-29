"""Any provider LiteLLM supports, including local Ollama (``pip install jevex[litellm]``).

Model ids use LiteLLM's ``provider/model`` form, e.g. ``"ollama/llama3.1"`` or
``"anthropic/claude-sonnet-5-5"``. The output schema goes in as ``response_format`` and
the JSON reply is validated here. Cost comes from LiteLLM's own price table (``None`` if
it doesn't know the model, e.g. local models) unless ``prices`` is given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from jevex.llm import (
    LLMError,
    LLMResponse,
    LLMUsage,
    ModelPrice,
    check_budget,
    cost,
    record,
    validate_output,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


class LiteLLM:
    """:class:`~jevex.llm.LLM` backed by ``litellm.acompletion``."""

    def __init__(
        self,
        model: str,
        *,
        prices: dict[str, ModelPrice] | None = None,
        **completion_kwargs: Any,
    ) -> None:
        self.model = model
        self.prices = prices
        self.completion_kwargs = completion_kwargs
        """Extra ``litellm.acompletion`` arguments, e.g. ``api_base`` for Ollama."""

    async def structured[T: BaseModel](self, prompt: str, schema: type[T]) -> LLMResponse[T]:
        import litellm

        check_budget()
        try:
            response: Any = await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                **self.completion_kwargs,
            )
        except Exception as exc:  # LiteLLM maps every provider's errors onto its own classes
            raise LLMError(f"LiteLLM error: {exc}") from exc
        in_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        out_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
        usage = LLMUsage(in_tokens, out_tokens, self._cost(response, in_tokens, out_tokens))
        record(usage)
        content = cast("str | None", response.choices[0].message.content)
        if not content:
            raise LLMError(f"{self.model} returned no output")
        return LLMResponse(output=validate_output(schema, content), usage=usage, model=self.model)

    def _cost(self, response: Any, in_tokens: int, out_tokens: int) -> float | None:
        if self.prices is not None:
            return cost(self.model, in_tokens, out_tokens, self.prices)
        import litellm

        try:
            return float(litellm.completion_cost(completion_response=response))  # pyright: ignore[reportUnknownMemberType]
        except Exception:  # unknown model (e.g. local): cost isn't known
            return None

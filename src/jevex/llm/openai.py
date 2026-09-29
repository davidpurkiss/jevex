"""OpenAI models via the OpenAI SDK (``pip install jevex[openai]``).

Structured output uses ``responses.parse(text_format=Model)``. No OpenAI prices are
bundled (they change often): pass ``prices={"model-id": ModelPrice(input, output)}`` to
get costs, otherwise ``usage.cost`` is ``None``. Credentials come from ``OPENAI_API_KEY``
unless a client is passed in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from openai import AsyncOpenAI
    from pydantic import BaseModel


class OpenAILLM:
    """:class:`~jevex.llm.LLM` backed by an OpenAI model (the model id is required)."""

    def __init__(
        self,
        model: str,
        *,
        client: AsyncOpenAI | None = None,
        instructions: str | None = None,
        prices: dict[str, ModelPrice] | None = None,
    ) -> None:
        import openai

        self.model = model
        self.instructions = instructions
        self.prices = prices or {}
        self._client = client or openai.AsyncOpenAI()

    async def structured[T: BaseModel](self, prompt: str, schema: type[T]) -> LLMResponse[T]:
        import openai

        check_budget()
        try:
            response = await self._client.responses.parse(
                model=self.model,
                input=prompt,
                text_format=schema,
                instructions=self.instructions,
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc
        in_tokens = response.usage.input_tokens if response.usage else 0
        out_tokens = response.usage.output_tokens if response.usage else 0
        usage = LLMUsage(
            in_tokens, out_tokens, cost(self.model, in_tokens, out_tokens, self.prices)
        )
        record(usage)
        if response.output_parsed is None:
            raise LLMError(f"{self.model} returned no structured output (it may have refused)")
        return LLMResponse(
            output=validate_output(schema, response.output_parsed), usage=usage, model=self.model
        )

    async def aclose(self) -> None:
        await self._client.close()

"""OpenAI models via the OpenAI SDK (``pip install jevex[openai]``).

Structured output uses ``responses.parse(text_format=Model)``. No OpenAI prices are
bundled (they change often): pass ``prices={"model-id": ModelPrice(input, output)}`` to
get costs, otherwise ``usage.cost`` is ``None``. Credentials come from ``OPENAI_API_KEY``
unless a client is passed in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from jevex.llm import (
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
    from collections.abc import Sequence

    from openai import AsyncOpenAI
    from pydantic import BaseModel

    from jevex.llm import LLMImage


def _input(prompt: str, images: Sequence[LLMImage]) -> Any:
    """The request's input: the prompt alone, or one user message with the images (as
    ``data:`` URIs) then the prompt."""
    if not images:
        return prompt
    content: list[dict[str, Any]] = [
        {"type": "input_image", "image_url": i.data_uri, "detail": "auto"} for i in images
    ]
    content.append({"type": "input_text", "text": prompt})
    return [{"role": "user", "content": content}]


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

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        import openai
        from pydantic import ValidationError

        check_budget()
        try:
            raw: Any = await self._client.responses.with_raw_response.parse(
                model=self.model,
                input=_input(prompt, images),
                text_format=schema,
                instructions=self.instructions,
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc
        # Record usage from the raw body first, so invalid output still counts.
        body = cast("dict[str, Any]", raw.http_response.json())
        tokens = cast("dict[str, Any]", body.get("usage") or {})
        in_tokens, out_tokens = (
            int(tokens.get("input_tokens", 0)),
            int(tokens.get("output_tokens", 0)),
        )
        usage = LLMUsage(
            in_tokens, out_tokens, cost(self.model, in_tokens, out_tokens, self.prices)
        )
        record(usage)
        refusals = [
            str(part.get("refusal", ""))
            for item in cast("list[dict[str, Any]]", body.get("output") or [])
            for part in cast("list[dict[str, Any]]", item.get("content") or [])
            if part.get("type") == "refusal"
        ]
        if refusals:
            raise LLMRefusalError(f"{self.model} declined the request: {' '.join(refusals)}")
        try:
            response = raw.parse()
        except (ValidationError, ValueError) as exc:
            raise LLMError(f"{self.model}'s output doesn't match {schema.__name__}: {exc}") from exc
        if response.output_parsed is None:
            raise LLMError(f"{self.model} returned no structured output")
        return LLMResponse(
            output=validate_output(schema, response.output_parsed), usage=usage, model=self.model
        )

    async def aclose(self) -> None:
        await self._client.close()

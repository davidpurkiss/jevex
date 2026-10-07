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
    from collections.abc import Sequence

    from pydantic import BaseModel

    from jevex.llm import LLMImage


def _content(prompt: str, images: Sequence[LLMImage]) -> str | list[dict[str, Any]]:
    """The user message: the prompt alone, or the images (as ``data:`` URIs) then it, in
    the OpenAI chat format LiteLLM translates for every provider."""
    if not images:
        return prompt
    parts: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": i.data_uri}} for i in images
    ]
    return [*parts, {"type": "text", "text": prompt}]


class LiteLLM:
    """:class:`~jevex.llm.LLM` backed by ``litellm.acompletion``."""

    def __init__(
        self,
        model: str,
        *,
        prices: dict[str, ModelPrice] | None = None,
        max_retries: int | None = None,
        **completion_kwargs: Any,
    ) -> None:
        """``max_retries`` is LiteLLM's ``num_retries`` for each call (``None``: its
        default). LiteLLM doesn't say how many it took, so
        :attr:`~jevex.llm.LLMResponse.retries` stays 0."""
        # Import here, not in the async call: importing litellm takes about a second and
        # would block the event loop on the first request.
        import litellm

        litellm.suppress_debug_info = True
        self.model = model
        self.prices = prices
        if max_retries is not None:
            completion_kwargs = {**completion_kwargs, "num_retries": max_retries}
        self.completion_kwargs = completion_kwargs
        """Extra ``litellm.acompletion`` arguments, e.g. ``api_base`` for Ollama."""

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        import litellm  # already imported by __init__; this is a cheap lookup

        check_budget()
        try:
            response: Any = await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]
                model=self.model,
                messages=[{"role": "user", "content": _content(prompt, images)}],
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

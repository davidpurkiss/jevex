"""LLMs for the optional fallback and for generator synthesis (spec: *LLM adapters and budgets*).

Every adapter implements one small protocol::

    class LLM(Protocol):
        async def structured(
            self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
        ) -> LLMResponse[T]: ...

``LLMResponse`` carries the validated output plus token usage and its cost. ``images``
(:class:`LLMImage`: PNG, JPEG or WebP bytes) go with the prompt to a vision model, for the
opt-in vision processor (:class:`~jevex.images.VisionProcessor`); callers pass them only
when there are some, so an LLM used only for text may leave the argument out. Adapters live
in extras: ``jevex.llm.anthropic`` (``jevex[anthropic]``), ``jevex.llm.openai``
(``jevex[openai]``), ``jevex.llm.gemini`` (``jevex[gemini]``) and ``jevex.llm.litellm``
(``jevex[litellm]``, which covers any provider LiteLLM supports, including local Ollama).

Spend is capped process-wide by ``JEVEX_LLM_MAX_COST_USD``: once this process has spent
the cap, no further call is made. With ``JEVEX_SPEND_LEDGER`` set, the cap counts what
every process wrote to that shared ledger instead (see ``jevex._spend``). Unlike Jev's
cap it can't pre-estimate a call (output size isn't known), so one call can overshoot,
concurrent calls can all pass the check, and calls with unknown prices count as $0.
It's a backstop; per-document and per-run budgets are #34.

Transient API failures are retried by each provider's SDK; every adapter takes
``max_retries`` (``None``: the SDK's default) and the retries it can see are counted on
:attr:`LLMResponse.retries` and in a document's ``meta.llm.retries``.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from jevex._spend import ledger_add, ledger_path, ledger_total

if TYPE_CHECKING:
    from collections.abc import Sequence

ANTHROPIC_MODEL = "claude-opus-5-5"
"""Default model: strongest, for generator synthesis (``generator_llm``)."""
ANTHROPIC_FAST_MODEL = "claude-sonnet-5-5"
"""Suggested cheaper model for bulk fallback extraction (``extraction_llm``)."""

MAX_COST_ENV = "JEVEX_LLM_MAX_COST_USD"


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float


def gemini_flash_3x_price(today: date) -> ModelPrice:
    """Gemini 3.6-3.8 Flash's price on ``today``: half price through 2026-12-31.

    :data:`PRICES` takes it once, at import, so a process running across the new year keeps
    the old rate until it restarts.
    """
    if today < date(2027, 1, 1):
        return ModelPrice(0.75, 3.75)
    return ModelPrice(1.50, 7.50)


_GEMINI_FLASH_3X = gemini_flash_3x_price(datetime.now(UTC).date())

# Anthropic first-party API list prices (docs, 2026-09-25) and Gemini Developer API standard
# paid-tier prices for text (ai.google.dev/gemini-api/docs/pricing, 2026-10-01; Pro models at
# their rate for prompts up to 200k tokens). OpenAI's prices change often and aren't
# bundled: pass ``prices=`` to an adapter, or use LiteLLM's own table.
PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(10.00, 50.00),
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-opus-5-5": ModelPrice(4.00, 20.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    "gemini-3.8-flash": _GEMINI_FLASH_3X,
    "gemini-3.7-flash": _GEMINI_FLASH_3X,
    "gemini-3.6-flash": _GEMINI_FLASH_3X,
    "gemini-3.5-flash": ModelPrice(1.50, 9.00),
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50),
    "gemini-3.1-flash-lite": ModelPrice(0.25, 1.50),
    "gemini-3.1-pro-preview": ModelPrice(2.00, 12.00),
    "gemini-3-flash-preview": ModelPrice(0.50, 3.00),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
}


_DATED_SNAPSHOT = re.compile(r"-\d{8}$")


def cost(
    model: str, input_tokens: int, output_tokens: int, prices: dict[str, ModelPrice] | None = None
) -> float | None:
    """USD for a call, or ``None`` when the model's price isn't known.

    An exact entry in ``prices`` (default :data:`PRICES`) wins. Otherwise a dated snapshot
    ID (``<name>-YYYYMMDD``) is costed at its base name's price, because APIs report the
    snapshot that served a call (Anthropic answers a ``claude-haiku-4-5`` request as
    ``claude-haiku-4-5-20251001``) while price lists name the base model.
    """
    table = prices if prices is not None else PRICES
    price = table.get(model)
    if price is None and _DATED_SNAPSHOT.search(model):
        price = table.get(_DATED_SNAPSHOT.sub("", model))
    if price is None:
        return None
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cost: float | None
    """USD, or ``None`` if the price isn't known."""


@dataclass(frozen=True)
class LLMResponse[T: BaseModel]:
    """A validated structured output plus what it cost.

    ``retries`` counts the requests the adapter's SDK sent again after a transient failure
    (0 when the SDK doesn't say: Gemini's and LiteLLM's don't)."""

    output: T
    usage: LLMUsage
    model: str
    retries: int = 0


IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
"""Image types every adapter can send (Claude, OpenAI and Gemini all read them)."""


@dataclass(frozen=True)
class LLMImage:
    """An image sent with a prompt: encoded bytes and their media type, one of
    :data:`IMAGE_TYPES`."""

    content: bytes
    content_type: str

    def __post_init__(self) -> None:
        if self.content_type not in IMAGE_TYPES:
            raise ValueError(
                f"an LLM image must be one of {sorted(IMAGE_TYPES)}, got {self.content_type!r}"
            )

    @property
    def base64(self) -> str:
        return base64.b64encode(self.content).decode("ascii")

    @property
    def data_uri(self) -> str:
        return f"data:{self.content_type};base64,{self.base64}"


@runtime_checkable
class LLM(Protocol):
    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        """``schema``'s output for ``prompt``, with ``images`` (if any) shown to the model
        before it. Raise :class:`LLMError` when the call fails."""
        ...


class LLMError(Exception):
    """The LLM call failed (API error, invalid output)."""


class LLMRefusalError(LLMError):
    """The model declined the request."""


class LLMBudgetExceededError(LLMError):
    """``JEVEX_LLM_MAX_COST_USD`` is already spent (by this process, or in the ledger)."""


# --- process-wide spend ----------------------------------------------------------------

_process_cost = 0.0


def process_llm_cost() -> float:
    """USD spent on LLM calls by this process (calls with unknown prices count as 0)."""
    return _process_cost


def reset_process_llm_cost() -> None:
    global _process_cost
    _process_cost = 0.0


def check_budget() -> None:
    """Raise before a call if the process cap is already spent."""
    raw = os.environ.get(MAX_COST_ENV)
    if not raw:
        return
    try:
        cap = float(raw)
    except ValueError:
        raise LLMError(f"{MAX_COST_ENV} must be a number of US dollars, got {raw!r}") from None
    ledger = ledger_path()
    spent = _process_cost
    if ledger is not None:  # the ledger can only tighten the cap
        spent = max(spent, ledger_total(ledger, "llm", LLMError))
    if spent >= cap:
        raise LLMBudgetExceededError(
            f"LLM spend cap reached: ${spent:.4f} of ${cap:.2f} ({MAX_COST_ENV})"
        )


def record(usage: LLMUsage) -> None:
    global _process_cost
    _process_cost += usage.cost or 0.0
    ledger = ledger_path()
    if ledger is not None and usage.cost:
        ledger_add(ledger, "llm", usage.cost, LLMError)


def validate_output[T: BaseModel](schema: type[T], data: Any) -> T:
    """Validate an adapter's parsed payload, raising :class:`LLMError` if it doesn't fit."""
    try:
        if isinstance(data, schema):
            return data
        if isinstance(data, str | bytes):
            return schema.model_validate_json(data)
        return schema.model_validate(data)
    except ValueError as exc:
        raise LLMError(f"LLM output doesn't match {schema.__name__}: {exc}") from exc


__all__ = [
    "ANTHROPIC_FAST_MODEL",
    "ANTHROPIC_MODEL",
    "IMAGE_TYPES",
    "LLM",
    "PRICES",
    "LLMBudgetExceededError",
    "LLMError",
    "LLMImage",
    "LLMRefusalError",
    "LLMResponse",
    "LLMUsage",
    "ModelPrice",
    "cost",
    "gemini_flash_3x_price",
    "process_llm_cost",
    "reset_process_llm_cost",
]

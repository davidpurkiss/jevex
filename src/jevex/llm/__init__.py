"""LLMs for the optional fallback and for generator synthesis (spec: *LLM adapters and budgets*).

Every adapter implements one small protocol::

    class LLM(Protocol):
        async def structured(self, prompt: str, schema: type[T]) -> LLMResponse[T]: ...

``LLMResponse`` carries the validated output plus token usage and its cost. Adapters live
in extras: ``jevex.llm.anthropic`` (``jevex[anthropic]``), ``jevex.llm.openai``
(``jevex[openai]``) and ``jevex.llm.litellm`` (``jevex[litellm]``, which covers any
provider LiteLLM supports, including local Ollama).

Spend is capped process-wide by ``JEVEX_LLM_MAX_COST_USD``: once this process has spent
the cap, no further call is made. With ``JEVEX_SPEND_LEDGER`` set, the cap counts what
every process wrote to that shared ledger instead (see ``jevex._spend``). Unlike Jev's
cap it can't pre-estimate a call (output size isn't known), so one call can overshoot,
concurrent calls can all pass the check, and calls with unknown prices count as $0.
It's a backstop; per-document and per-run budgets are #34.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from jevex._spend import ledger_add, ledger_path, ledger_total

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


# Anthropic first-party API list prices (docs, 2026-09-25). Other providers' prices change
# often and aren't bundled: pass ``prices=`` to an adapter, or use LiteLLM's own table.
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
}


def cost(
    model: str, input_tokens: int, output_tokens: int, prices: dict[str, ModelPrice] | None = None
) -> float | None:
    """USD for a call, or ``None`` when the model's price isn't known."""
    price = (prices if prices is not None else PRICES).get(model)
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
    """A validated structured output plus what it cost."""

    output: T
    usage: LLMUsage
    model: str


@runtime_checkable
class LLM(Protocol):
    async def structured[T: BaseModel](self, prompt: str, schema: type[T]) -> LLMResponse[T]: ...


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
    spent = _process_cost if ledger is None else ledger_total(ledger, "llm", LLMError)
    if spent >= cap:
        raise LLMBudgetExceededError(
            f"LLM spend cap reached: ${spent:.4f} of ${cap:.2f} ({MAX_COST_ENV})"
        )


def record(usage: LLMUsage) -> None:
    global _process_cost
    _process_cost += usage.cost or 0.0
    ledger = ledger_path()
    if ledger is not None and usage.cost:
        ledger_add(ledger, "llm", usage.cost)


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
    "LLM",
    "PRICES",
    "LLMBudgetExceededError",
    "LLMError",
    "LLMRefusalError",
    "LLMResponse",
    "LLMUsage",
    "ModelPrice",
    "cost",
    "process_llm_cost",
    "reset_process_llm_cost",
]

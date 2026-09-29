"""Async Jev client: batched questions, request splitting and metering.

Stages build :class:`Noul`, :class:`Choice` and :class:`Score` questions and call
:meth:`JevClient.ask` once per level. Every question about the same state goes into
one request, because Jev evaluates them in parallel and extra questions barely change
latency or cost. The client splits a batch only when it would exceed Jev's context
budget.

The transport is a :class:`JevBackend`, so tests can script answers without the
network. :class:`TypeSafeBackend` talks to the real API through ``typesafe-sdk``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from typesafe_sdk import AsyncTypeSafeClient, NoulCriteria, RetryPolicy

type JSONContent = str | Mapping[str, Any] | Sequence[Any]

# Documented limits for Jev 1.13: https://docs.typesafe.ai (Models page).
MAX_REQUEST_TOKENS = 64_000
MAX_STATE_TOKENS = 32_000  # state plus the longest question
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
PRICE_PER_MILLION_INPUT_TOKENS = 0.042  # USD; output tokens are free


# --- Questions -------------------------------------------------------------------------


class Noul(BaseModel):
    """Is this statement true? Answered with a probability."""

    model_config = ConfigDict(frozen=True)

    type: Literal["noul"] = "noul"
    instructions: JSONContent
    criteria: dict[Literal["true", "false"], JSONContent] | None = None


class Choice(BaseModel):
    """Pick one option. ``options`` maps each label to an optional description."""

    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    instructions: JSONContent
    options: dict[str, JSONContent | None]

    @model_validator(mode="after")
    def _option_count(self) -> Choice:
        if not 1 <= len(self.options) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"a Choice needs 1 to {MAX_CHOICE_OPTIONS} options")
        return self


class Score(BaseModel):
    """Rate against ordered levels, lowest first."""

    model_config = ConfigDict(frozen=True)

    type: Literal["score"] = "score"
    instructions: JSONContent
    levels: list[JSONContent]

    @model_validator(mode="after")
    def _level_count(self) -> Score:
        if not MIN_SCORE_LEVELS <= len(self.levels) <= MAX_SCORE_LEVELS:
            raise ValueError(f"a Score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")
        return self


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


# --- Answers ---------------------------------------------------------------------------


class NoulAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["noul"] = "noul"
    p: float = Field(ge=0, le=1, description="Probability the statement is true")


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["score"] = "score"
    score: float = Field(description="Probability-weighted mean level (0-based)")
    confidence: float
    probabilities: dict[int, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class JevResponse(BaseModel):
    """What a backend returns for one request."""

    answers: dict[str, Answer]
    input_tokens: int | None = None
    model: str | None = None


# --- Errors ----------------------------------------------------------------------------


class JevError(Exception):
    """Base class for jevex's Jev errors."""


class StateTooLargeError(JevError):
    """The state (plus one question) cannot fit in a single Jev request; chunk it first."""


class MissingAnswerError(JevError):
    """The backend returned no answer for a question that was asked."""


class UnexpectedAnswerError(JevError):
    """The backend answered a question with the wrong answer type."""


class JevBackendError(JevError):
    """The Jev API failed: bad or missing key, network error, rejected request, 5xx."""


class JevRequestCapError(JevError):
    """A client's ``max_requests`` is used up (a document's ``DocBudget.max_jev_requests``).

    The extractor stops that document and returns what it found; other documents carry
    on. Unlike :class:`JevBudgetExceededError`, it never ends a run.
    """


class JevBudgetExceededError(JevError):
    """Sending the request would take this process past ``JEVEX_JEV_MAX_COST_USD``."""


# --- Process-wide spend cap ------------------------------------------------------------
#
# A hard stop for live runs (agent loop, live tests, benchmarks): when
# JEVEX_JEV_MAX_COST_USD is set, no request is sent once this process's estimated Jev
# spend would exceed it. Per-document and per-run budgets are a separate, richer
# feature (#34); this cap is the backstop that holds whatever the code above does.

MAX_COST_ENV = "JEVEX_JEV_MAX_COST_USD"
_process_cost = 0.0


def process_cost() -> float:
    """Estimated USD spent on Jev by this process so far."""
    return _process_cost


def reset_process_cost() -> None:
    """Reset the process-wide spend counter (for tests)."""
    global _process_cost
    _process_cost = 0.0


def _max_cost() -> float | None:
    raw = os.environ.get(MAX_COST_ENV)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise JevError(f"{MAX_COST_ENV} must be a number of US dollars, got {raw!r}") from None


def _token_cost(tokens: int) -> float:
    return tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000


# --- Metering --------------------------------------------------------------------------


@dataclass
class JevUsage:
    """Running totals for one document or run."""

    requests: int = 0
    questions: int = 0
    input_tokens: int = 0
    seconds: float = 0.0
    models: set[str] = field(default_factory=set[str])

    @property
    def cost(self) -> float:
        """Estimated spend in USD."""
        return self.input_tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000

    def add(self, other: JevUsage) -> None:
        self.requests += other.requests
        self.questions += other.questions
        self.input_tokens += other.input_tokens
        self.seconds += other.seconds
        self.models |= other.models


# --- Backends --------------------------------------------------------------------------


class JevBackend(Protocol):
    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse: ...


class TypeSafeBackend:
    """The real Jev API via ``typesafe-sdk``.

    Retries (429, 5xx, timeouts, ``Retry-After``) are handled by the SDK. Reads
    ``TYPESAFE_API_KEY``, ``TYPESAFE_BASE_URL`` and ``TYPESAFE_DEFAULT_MODEL`` unless
    a client is passed in.
    """

    def __init__(
        self,
        client: AsyncTypeSafeClient | None = None,
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError

        try:
            self._client = client or AsyncTypeSafeClient(model=model, retry=retry)
        except TypeSafeError as exc:  # e.g. no TYPESAFE_API_KEY
            raise JevBackendError(str(exc)) from exc
        self._model = model

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        from typesafe_sdk import Choice as SdkChoice
        from typesafe_sdk import Noul as SdkNoul
        from typesafe_sdk import Score as SdkScore

        sdk_questions: dict[str, SdkNoul | SdkChoice | SdkScore] = {}
        for key, q in questions.items():
            match q:
                case Noul():
                    criteria = cast("NoulCriteria | None", q.criteria)
                    sdk_questions[key] = SdkNoul(instructions=q.instructions, criteria=criteria)
                case Choice():
                    sdk_questions[key] = SdkChoice(instructions=q.instructions, criteria=q.options)
                case Score():
                    sdk_questions[key] = SdkScore(instructions=q.instructions, criteria=q.levels)

        from typesafe_sdk import TypeSafeError

        try:
            # The SDK's recursive JSON alias reads as partially unknown under strict pyright.
            response = await self._client.system_one(  # pyright: ignore[reportUnknownMemberType]
                state, sdk_questions, model=self._model
            )
        except TypeSafeError as exc:  # after the SDK's own retries
            raise JevBackendError(str(exc)) from exc

        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}
        for key, nou in response.nouls.items():
            answers[key] = NoulAnswer(p=nou.noul)
        for key, cho in response.choices.items():
            answers[key] = ChoiceAnswer(
                choice=cho.choice, confidence=cho.confidence, probabilities=dict(cho.probabilities)
            )
        for key, sco in response.scores.items():
            answers[key] = ScoreAnswer(
                score=sco.score, confidence=sco.confidence, probabilities=dict(sco.probabilities)
            )
        return JevResponse(
            answers=answers, input_tokens=response.usage.input_tokens, model=response.model
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# --- Client ----------------------------------------------------------------------------


class _Limiter:
    """A concurrency cap shared by metered clients, with one semaphore per event loop.

    Semaphores bind to the loop that first uses them, and ``extract_sync`` starts a new
    loop per call, so a single semaphore would fail on the second call.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if loop not in self._by_loop:
            self._by_loop[loop] = asyncio.Semaphore(self.limit)
        return self._by_loop[loop]


def estimate_tokens(content: object) -> int:
    """Rough token count (about 4 characters per token) used to plan requests."""
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return len(text) // 4 + 1


class JevClient:
    """Batches questions per state, splits oversized batches and meters every request."""

    def __init__(
        self,
        backend: JevBackend,
        *,
        max_concurrency: int = 16,
        request_token_budget: int = int(MAX_REQUEST_TOKENS * 0.9),
        state_token_budget: int = int(MAX_STATE_TOKENS * 0.9),
        usage: JevUsage | None = None,
        max_requests: int | None = None,
        _limiter: _Limiter | None = None,
    ) -> None:
        self.backend = backend
        self.usage = usage if usage is not None else JevUsage()
        self.max_requests = max_requests
        self._started = 0  # requests begun, so concurrent sends can't pass the cap together
        self._request_budget = request_token_budget
        self._state_budget = state_token_budget
        self._limiter = _limiter or _Limiter(max_concurrency)

    @classmethod
    def from_env(cls, *, model: str | None = None, max_concurrency: int = 16) -> JevClient:
        """A client for the real API, configured from ``TYPESAFE_*`` environment variables."""
        return cls(TypeSafeBackend(model=model), max_concurrency=max_concurrency)

    def metered(
        self, usage: JevUsage | None = None, *, max_requests: int | None = None
    ) -> JevClient:
        """A client sharing this backend and concurrency limit but with its own usage.

        The pipeline takes one per document, so each result reports its own Jev calls.
        ``max_requests`` caps that client's requests (:class:`JevRequestCapError`).
        """
        return JevClient(
            self.backend,
            request_token_budget=self._request_budget,
            state_token_budget=self._state_budget,
            usage=usage if usage is not None else JevUsage(),
            max_requests=max_requests,
            _limiter=self._limiter,
        )

    async def ask(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]:
        """Ask every question about ``state``, in as few requests as fit the budget."""
        if not questions:
            return {}
        batches = self._plan(state, questions)
        responses = await asyncio.gather(*(self._send(state, batch) for batch in batches))
        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}
        for batch, response in zip(batches, responses, strict=True):
            for key in batch:
                if key not in response.answers:
                    raise MissingAnswerError(f"Jev returned no answer for question {key!r}")
                answers[key] = response.answers[key]
        return answers

    def _plan(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> list[dict[str, Question]]:
        state_tokens = estimate_tokens(state)
        sizes = {
            key: estimate_tokens(q.model_dump(exclude={"type"})) for key, q in questions.items()
        }
        if state_tokens + max(sizes.values()) > self._state_budget:
            raise StateTooLargeError(
                f"state is ~{state_tokens} tokens; the limit for state plus one question is "
                f"~{self._state_budget}. Split the component before asking."
            )
        batches: list[dict[str, Question]] = [{}]
        used = state_tokens
        for key, q in questions.items():
            if batches[-1] and used + sizes[key] > self._request_budget:
                batches.append({})
                used = state_tokens
            batches[-1][key] = q
            used += sizes[key]
        return batches

    async def _send(self, state: JSONContent, batch: dict[str, Question]) -> JevResponse:
        global _process_cost
        estimated = estimate_tokens(state) + sum(
            estimate_tokens(q.model_dump()) for q in batch.values()
        )
        if self.max_requests is not None:
            if self._started >= self.max_requests:
                raise JevRequestCapError(f"the cap of {self.max_requests} Jev requests is used up")
            self._started += 1
        cap = _max_cost()
        if cap is not None and _process_cost + _token_cost(estimated) > cap:
            raise JevBudgetExceededError(
                f"Jev spend cap reached: ${_process_cost:.4f} spent, this request would add "
                f"~${_token_cost(estimated):.4f}, cap is ${cap:.2f} ({MAX_COST_ENV})"
            )
        async with self._limiter.semaphore():
            start = time.perf_counter()
            response = await self.backend.system_one(state, batch)
            elapsed = time.perf_counter() - start
        tokens = response.input_tokens if response.input_tokens is not None else estimated
        _process_cost += _token_cost(tokens)
        self.usage.requests += 1
        self.usage.questions += len(batch)
        self.usage.input_tokens += tokens
        self.usage.seconds += elapsed
        if response.model:
            self.usage.models.add(response.model)
        return response

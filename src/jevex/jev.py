"""Async Jev client: batched questions, request splitting and metering.

Stages build :class:`Noul`, :class:`Choice` and :class:`Score` questions and call
:meth:`JevClient.ask` once per level. Every question about the same state goes into
one request, because Jev evaluates them in parallel and extra questions barely change
latency or cost. The client splits a batch only when it would exceed Jev's context
budget, by a token estimate calibrated on real counts, or when Jev rejects it as too big.

The transport is a :class:`JevBackend`, so tests can script answers without the
network. :class:`TypeSafeBackend` talks to the real API through ``typesafe-sdk``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevex._spend import ledger_add, ledger_path, ledger_total
from jevex._tasks import gather
from jevex.logs import get_logger

if TYPE_CHECKING:
    from typesafe_sdk import AsyncTypeSafeClient, NoulCriteria
    from typesafe_sdk import RetryPolicy as SdkRetryPolicy

log = get_logger(__name__)

type JSONContent = str | Mapping[str, Any] | Sequence[Any]

# Documented limits for Jev 1.13: https://docs.typesafe.ai (Models page).
MAX_REQUEST_TOKENS = 64_000
MAX_STATE_TOKENS = 32_000  # state plus the longest question
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
PRICE_PER_MILLION_INPUT_TOKENS = 0.042  # USD; output tokens are free

# Measured against jev-1.13.0 in #6: what Jev bills on top of the state and question text.
REQUEST_OVERHEAD_TOKENS = 260
"""Tokens every request costs whatever it asks."""
QUESTION_OVERHEAD_TOKENS = 7
"""Tokens each question costs on top of its text."""
OPTION_OVERHEAD_TOKENS = 2
"""Tokens each Choice option, Score level or Noul criterion costs on top of its text (not
measured; set to err high)."""


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


class JevTransientError(JevBackendError):
    """A Jev request failed in a way that may pass on a retry: a timeout, a connection
    error, a 408, 429 or 5xx. :class:`JevClient` retries it by its :class:`RetryPolicy`;
    ``retry_after`` (seconds) is the server's ``Retry-After``, if it sent one, and
    ``status`` the HTTP status (``None`` for a timeout or connection error)."""

    def __init__(
        self, message: str, *, retry_after: float | None = None, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status

    @property
    def rate_limited(self) -> bool:
        """Jev said too many requests (429)."""
        return self.status == 429


class JevTokenLimitError(JevBackendError):
    """Jev rejected a request as over its token limits (``max_tokens_exceeded``).

    A backend raises it and :class:`JevClient` handles it: it asks the questions over more
    requests, or raises :class:`StateTooLargeError` when the state with a single question
    is already too big. It never reaches a stage.
    """


class JevRequestCapError(JevError):
    """A client's ``max_requests`` is used up (a document's ``DocBudget.max_jev_requests``).

    The extractor stops that document and returns what it found; other documents carry
    on. Unlike :class:`JevBudgetExceededError`, it never ends a run.
    """


class JevBudgetExceededError(JevError):
    """Sending the request would take spend past ``JEVEX_JEV_MAX_COST_USD``.

    Spend is this process's, or everything in ``JEVEX_SPEND_LEDGER`` when that's set.
    """


# --- Process-wide spend cap ------------------------------------------------------------
#
# A hard stop for live runs (agent loop, live tests, benchmarks): when
# JEVEX_JEV_MAX_COST_USD is set, no request is sent once this process's estimated Jev
# spend would exceed it. Per-document and per-run budgets are a separate, richer
# feature (#34); this cap is the backstop that holds whatever the code above does.
# With JEVEX_SPEND_LEDGER set, the cap counts every process writing to that ledger
# (jevex._spend), which is how the agent loop caps a whole run and a week.

MAX_COST_ENV = "JEVEX_JEV_MAX_COST_USD"
_process_cost: float = 0.0


def process_cap() -> tuple[float, float] | None:
    """``JEVEX_JEV_MAX_COST_USD`` and the spend it's compared against (this process's, or
    the ``JEVEX_SPEND_LEDGER`` total), or ``None`` when no cap is set."""
    cap = _max_cost()
    return None if cap is None else (cap, _spent())


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


def _spent() -> float:
    """Spend the cap compares against: the shared ledger's Jev total, but never less than
    this process's own (so a ledger can only tighten the cap)."""
    ledger = ledger_path()
    if ledger is None:
        return _process_cost
    return max(_process_cost, ledger_total(ledger, "jev", JevError))


def _charge(usd: float) -> None:
    global _process_cost
    _process_cost += usd
    ledger = ledger_path()
    if ledger is not None:
        ledger_add(ledger, "jev", usd, JevError)


# --- Retries ---------------------------------------------------------------------------


class RetryPolicy(BaseModel):
    """How :class:`JevClient` retries a request that failed transiently
    (:class:`JevTransientError`: timeouts, connection errors, 408, 429, 5xx).

    The delay before retry ``n`` (1-based) is ``backoff_initial * 2**(n-1)``, capped at
    ``backoff_max``, less a random fraction of up to ``backoff_jitter`` of it; a
    ``Retry-After`` from the server is waited instead when it's longer (up to
    ``backoff_max``). ``max_retries=0`` turns retries off. Every retry is counted in the
    usage (``JevUsage.retries``), so results report them.
    """

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(default=2, ge=0)
    backoff_initial: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    backoff_max: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    backoff_jitter: float = Field(default=0.25, ge=0, le=1)

    def delay(self, retry: int, retry_after: float | None = None) -> float:
        """Seconds to wait before retry number ``retry`` (1-based)."""
        base = min(self.backoff_initial * 2 ** (retry - 1), self.backoff_max)
        wait = base * (1 - random.random() * self.backoff_jitter)
        if retry_after is not None:
            wait = max(wait, min(retry_after, self.backoff_max))
        return wait


# --- Metering --------------------------------------------------------------------------


@dataclass
class JevUsage:
    """Running totals for one document or run."""

    requests: int = 0
    questions: int = 0
    input_tokens: int = 0
    seconds: float = 0.0
    models: set[str] = field(default_factory=set[str])
    retries: int = 0
    """Requests sent again after a transient failure (:class:`RetryPolicy`)."""
    rate_limited: int = 0
    """Requests Jev answered with 429 (too many requests), retried or not."""

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
        self.retries += other.retries
        self.rate_limited += other.rate_limited


# --- Backends --------------------------------------------------------------------------


class JevBackend(Protocol):
    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse: ...


class TypeSafeBackend:
    """The real Jev API via ``typesafe-sdk``.

    A transient failure (a timeout, a connection error, a 408, 429 or 5xx) is raised as
    :class:`JevTransientError`, for :class:`JevClient` to retry by its
    :class:`RetryPolicy` and count. So the client this backend makes has the SDK's own
    retries off; ``sdk_retry`` (the SDK's ``RetryPolicy``) turns them back on, uncounted.
    A client passed in keeps its own settings. Reads ``TYPESAFE_API_KEY``,
    ``TYPESAFE_BASE_URL`` and ``TYPESAFE_DEFAULT_MODEL`` unless a client is passed in.
    """

    def __init__(
        self,
        client: AsyncTypeSafeClient | None = None,
        *,
        model: str | None = None,
        sdk_retry: SdkRetryPolicy | None = None,
    ) -> None:
        from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError
        from typesafe_sdk import RetryPolicy as SdkRetry

        retry = sdk_retry if sdk_retry is not None else SdkRetry(max_retries=0)
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

        from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError, TypeSafeError

        try:
            # The SDK's recursive JSON alias reads as partially unknown under strict pyright.
            response = await self._client.system_one(  # pyright: ignore[reportUnknownMemberType]
                state, sdk_questions, model=self._model
            )
        except TypeSafeError as exc:
            if isinstance(exc, TypeSafeAPIError) and _error_type(exc.body) == TOKEN_LIMIT_ERROR:
                raise JevTokenLimitError(str(exc)) from exc
            if isinstance(exc, TypeSafeAPIConnectionError):  # timeouts too
                raise JevTransientError(str(exc)) from exc
            if isinstance(exc, TypeSafeAPIError) and _transient_status(exc.status):
                after = _retry_after(exc.headers.get("retry-after"))
                raise JevTransientError(str(exc), retry_after=after, status=exc.status) from exc
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


def _transient_status(status: int) -> bool:
    return status in (408, 429) or 500 <= status < 600


def _retry_after(header: str | None) -> float | None:
    """A ``Retry-After`` header's seconds (``None`` for an HTTP date or nothing)."""
    try:
        return max(0.0, float(header)) if header else None
    except ValueError:
        return None


TOKEN_LIMIT_ERROR = "max_tokens_exceeded"
"""The ``error_type`` of Jev's 400 for a request over its token limits."""


def _error_type(body: object) -> object:
    """The ``error_type`` of an API error body: ``{"detail": {"error_type": ...}}``, as
    measured in #6, or at the top level."""
    if not isinstance(body, dict):
        return None
    fields = cast("dict[str, object]", body)
    detail = fields.get("detail")
    if isinstance(detail, dict) and "error_type" in detail:
        return cast("dict[str, object]", detail)["error_type"]
    return fields.get("error_type")


# --- Token estimates -------------------------------------------------------------------
#
# Jev's tokeniser isn't published, so requests are planned with an estimate calibrated on
# real counts (#6). A flat 4 characters per token was close for prose but read tables and
# JSON at a third of their size: every digit is a token of its own, and so is almost every
# punctuation mark or separator, while a common English word is one token with the space
# before it. The estimate counts those pieces. Words cost a token per few characters for
# states, so names, codes and units err high, and closer to one per word for questions,
# whose text is plain English.

_TOKEN_PIECES = re.compile(r" ?[A-Za-z]+| ?[^\w\s]|\d|\s+|.", re.DOTALL)
STATE_WORD_CHARS = 7
"""Letters per token in a state's ASCII words: errs high (about 1.1x on prose)."""
QUESTION_WORD_CHARS = 13
"""Letters per token in a question's ASCII words: close (common words are one token)."""


def _piece_tokens(piece: str, word_chars: int) -> int:
    word = piece.lstrip(" ")
    return 1 + (len(word) - 1) // word_chars if word.isascii() and word.isalpha() else 1


def _text_tokens(text: str, word_chars: int) -> int:
    return sum(_piece_tokens(piece, word_chars) for piece in _TOKEN_PIECES.findall(text))


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str, ensure_ascii=False)


def estimate_tokens(content: object) -> int:
    """Estimated tokens for ``content`` as a state (JSON as it is serialised), erring high.

    Calibrated on #6's measurements: within about 10% above the real count for prose, and
    at or just above it for table text and JSON, which a characters-per-token rule reads
    at a third of their size. Non-ASCII letters count a token each.
    """
    return _text_tokens(_content_text(content), STATE_WORD_CHARS)


def estimate_question_tokens(question: Noul | Choice | Score) -> int:
    """Estimated tokens for one question: its text, options and per-question overhead."""
    match question:
        case Noul():
            options = len(question.criteria or {})
            texts: list[object] = list((question.criteria or {}).values())
        case Choice():
            options = len(question.options)
            texts = [*question.options, *(d for d in question.options.values() if d is not None)]
        case Score():
            options = len(question.levels)
            texts = list(question.levels)
    return (
        QUESTION_OVERHEAD_TOKENS
        + options * OPTION_OVERHEAD_TOKENS
        + sum(
            _text_tokens(_content_text(text), QUESTION_WORD_CHARS)
            for text in (question.instructions, *texts)
        )
    )


def estimate_request_tokens(state: JSONContent, questions: Mapping[str, Question]) -> int:
    """Estimated billed tokens for one request: the state once, every question, and Jev's
    fixed overhead. The spend cap and the cost of a request without a reported count use
    it."""
    return (
        REQUEST_OVERHEAD_TOKENS
        + estimate_tokens(state)
        + sum(estimate_question_tokens(q) for q in questions.values())
    )


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """The longest leading part of ``text`` whose :func:`estimate_tokens` is at most
    ``max_tokens``, cut between tokens."""
    tokens = 0
    for match in _TOKEN_PIECES.finditer(text):
        cost = _piece_tokens(match.group(), STATE_WORD_CHARS)
        if tokens + cost > max_tokens:
            return text[: match.start()]
        tokens += cost
    return text


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


class JevClient:
    """Batches questions per state, splits oversized batches and meters every request.

    A request that fails transiently (:class:`JevTransientError`) is sent again by
    ``retry`` (default :class:`RetryPolicy()`: two retries with backoff), and each retry
    is counted in ``usage.retries``. What still fails after the retries raises.
    """

    def __init__(
        self,
        backend: JevBackend,
        *,
        max_concurrency: int = 16,
        request_token_budget: int = int(MAX_REQUEST_TOKENS * 0.9),
        state_token_budget: int = int(MAX_STATE_TOKENS * 0.9),
        usage: JevUsage | None = None,
        max_requests: int | None = None,
        retry: RetryPolicy | None = None,
        _limiter: _Limiter | None = None,
    ) -> None:
        self.backend = backend
        self.usage = usage if usage is not None else JevUsage()
        self.max_requests = max_requests
        self.retry = retry if retry is not None else RetryPolicy()
        self._started = 0  # requests begun, so concurrent sends can't pass the cap together
        self._request_budget = request_token_budget
        self._state_budget = state_token_budget
        self._limiter = _limiter or _Limiter(max_concurrency)

    @classmethod
    def from_env(cls, *, model: str | None = None, max_concurrency: int = 16) -> JevClient:
        """A client for the real API, configured from ``TYPESAFE_*`` environment variables."""
        return cls(TypeSafeBackend(model=model), max_concurrency=max_concurrency)

    def metered(
        self,
        usage: JevUsage | None = None,
        *,
        max_requests: int | None = None,
        retry: RetryPolicy | None = None,
    ) -> JevClient:
        """A client sharing this backend and concurrency limit but with its own usage.

        The pipeline takes one per document, so each result reports its own Jev calls.
        ``max_requests`` caps that client's requests (:class:`JevRequestCapError`).
        ``retry`` replaces this client's retry policy (``None``: keep it).
        """
        return JevClient(
            self.backend,
            request_token_budget=self._request_budget,
            state_token_budget=self._state_budget,
            usage=usage if usage is not None else JevUsage(),
            max_requests=max_requests,
            retry=retry if retry is not None else self.retry,
            _limiter=self._limiter,
        )

    def fit_state(self, text: str, questions: Mapping[str, Question]) -> str:
        """The leading part of ``text`` that fits in one request's state alongside the
        longest of ``questions``, by the same estimate :meth:`ask` plans with.

        For a stage that would rather read less of a long text than split it.
        """
        room = self._state_budget - max(
            (estimate_question_tokens(q) for q in questions.values()), default=0
        )
        if room < 1:
            raise StateTooLargeError(
                f"a question alone is ~{self._state_budget - room} tokens, over the "
                f"~{self._state_budget} for state plus one question"
            )
        return truncate_to_tokens(text, room)

    async def ask(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]:
        """Ask every question about ``state``, in as few requests as fit the budget.

        Raises :class:`StateTooLargeError` when the state and one question don't fit,
        by the estimate or by Jev's own count: a request Jev rejects as too big is asked
        again in smaller parts, and only a state too big for any request raises.
        """
        if not questions:
            return {}
        batches = self._plan(state, questions)
        parts = await gather(self._ask_batch(state, batch) for batch in batches)
        answers = {key: answer for part in parts for key, answer in part.items()}
        return {key: answers[key] for key in questions}

    def _plan(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> list[dict[str, Question]]:
        # The budgets cover the state and questions; their 10% headroom covers Jev's fixed
        # REQUEST_OVERHEAD_TOKENS.
        state_tokens = estimate_tokens(state)
        sizes = {key: estimate_question_tokens(q) for key, q in questions.items()}
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

    async def _ask_batch(
        self, state: JSONContent, batch: dict[str, Question]
    ) -> dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]:
        try:
            response = await self._send(state, batch)
        except JevTokenLimitError as exc:
            return await self._split_rejected(state, batch, exc)
        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}
        for key in batch:
            if key not in response.answers:
                raise MissingAnswerError(f"Jev returned no answer for question {key!r}")
            answers[key] = response.answers[key]
        return answers

    async def _split_rejected(
        self, state: JSONContent, batch: dict[str, Question], rejected: JevTokenLimitError
    ) -> dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]:
        """Answers for a batch Jev rejected as too big.

        The longest question goes alone first. If Jev rejects that too, the state is too
        big, and :class:`StateTooLargeError` tells the stage to chunk it, as it would for
        an estimated overflow. Otherwise the rest go in two halves, each split again if
        it is rejected. So a too-big state costs one more request, not one per question.
        """
        if len(batch) == 1:
            raise StateTooLargeError(
                f"Jev rejected the state with a single question as over its token limits "
                f"(estimated ~{estimate_request_tokens(state, batch)} tokens). Split the "
                "component before asking."
            ) from rejected
        longest = max(batch, key=lambda key: estimate_question_tokens(batch[key]))
        answers = await self._ask_batch(state, {longest: batch[longest]})
        rest = [key for key in batch if key != longest]
        halves = [half for half in (rest[: len(rest) // 2], rest[len(rest) // 2 :]) if half]
        parts = await gather(
            self._ask_batch(state, {key: batch[key] for key in half}) for half in halves
        )
        for part in parts:
            answers.update(part)
        return answers

    async def _send(self, state: JSONContent, batch: dict[str, Question]) -> JevResponse:
        estimated = estimate_request_tokens(state, batch)
        if self.max_requests is not None:
            if self._started >= self.max_requests:
                raise JevRequestCapError(f"the cap of {self.max_requests} Jev requests is used up")
            self._started += 1
        cap = _max_cost()
        if cap is not None and (spent := _spent()) + _token_cost(estimated) > cap:
            raise JevBudgetExceededError(
                f"Jev spend cap reached: ${spent:.4f} spent, this request would add "
                f"~${_token_cost(estimated):.4f}, cap is ${cap:.2f} ({MAX_COST_ENV})"
            )
        retries = 0
        while True:
            try:
                response = await self._attempt(state, batch, estimated)
            except JevTransientError as exc:
                if retries >= self.retry.max_retries:
                    raise
                retries += 1
                self.usage.retries += 1
                delay = self.retry.delay(retries, exc.retry_after)
                log.warning(
                    "Jev request failed (%s), retry %d of %d in %.1fs",
                    exc,
                    retries,
                    self.retry.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            log.debug(
                "Jev request: %d questions, %d tokens", len(batch), response.input_tokens or 0
            )
            return response

    async def _attempt(
        self, state: JSONContent, batch: dict[str, Question], estimated: int
    ) -> JevResponse:
        """One request to the backend, metered."""
        async with self._limiter.semaphore():
            start = time.perf_counter()
            try:
                response = await self.backend.system_one(state, batch)
            except asyncio.CancelledError:
                # Cancelled mid-request (a sibling failed): the request may still be
                # billed, so count it at its estimated size before letting go.
                _charge(_token_cost(estimated))
                self.usage.requests += 1
                self.usage.questions += len(batch)
                self.usage.input_tokens += estimated
                self.usage.seconds += time.perf_counter() - start
                raise
            except (JevTokenLimitError, JevTransientError) as exc:
                # Rejected (or lost) before Jev answered, so nothing is billed, but it was
                # a request.
                self.usage.requests += 1
                self.usage.seconds += time.perf_counter() - start
                if isinstance(exc, JevTransientError) and exc.rate_limited:
                    self.usage.rate_limited += 1
                raise
            elapsed = time.perf_counter() - start
        tokens = response.input_tokens if response.input_tokens is not None else estimated
        _charge(_token_cost(tokens))
        self.usage.requests += 1
        self.usage.questions += len(batch)
        self.usage.input_tokens += tokens
        self.usage.seconds += elapsed
        if response.model:
            self.usage.models.add(response.model)
        return response

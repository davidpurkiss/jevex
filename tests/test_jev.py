import asyncio
import json
from collections.abc import Iterator, Mapping

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jevex.jev import (
    Choice,
    ChoiceAnswer,
    JevClient,
    JevRequestCapError,
    JevResponse,
    JSONContent,
    MissingAnswerError,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    StateTooLargeError,
    TypeSafeBackend,
    estimate_tokens,
)


class RecordingBackend:
    """Answers every Noul with p=0.9 and every Choice with its first option."""

    def __init__(self, *, drop: str | None = None, delay: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._drop = drop
        self._delay = delay

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        self.calls.append(list(questions))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self._delay)
        self.in_flight -= 1
        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}
        for key, q in questions.items():
            if key == self._drop:
                continue
            match q:
                case Noul():
                    answers[key] = NoulAnswer(p=0.9)
                case Choice():
                    first = next(iter(q.options))
                    answers[key] = ChoiceAnswer(
                        choice=first, confidence=1.0, probabilities={first: 1.0}
                    )
                case Score():
                    answers[key] = ScoreAnswer(score=0.0, confidence=1.0, probabilities={0: 1.0})
        return JevResponse(answers=answers, input_tokens=100, model="jev-test")


async def test_all_questions_go_in_one_request() -> None:
    backend = RecordingBackend()
    client = JevClient(backend)
    answers = await client.ask(
        {"statement": "0-62 mph in 9.1 s"},
        {
            "gate": Noul(instructions="Is this about performance?"),
            "field": Choice(
                instructions="Which detail does this state?",
                options={"zero_to_62_s": "0-62 time", "none": None},
            ),
        },
    )
    assert backend.calls == [["gate", "field"]]
    assert answers["gate"] == NoulAnswer(p=0.9)
    assert isinstance(answers["field"], ChoiceAnswer)
    assert answers["field"].choice == "zero_to_62_s"


async def test_oversized_batch_is_split_and_answers_merged() -> None:
    backend = RecordingBackend()
    client = JevClient(backend, request_token_budget=60, state_token_budget=50)
    questions: dict[str, Question] = {
        f"q{i}": Noul(instructions=f"Is question number {i} true?") for i in range(6)
    }
    answers = await client.ask("short state", questions)
    assert len(backend.calls) > 1
    assert sorted(k for call in backend.calls for k in call) == sorted(questions)
    assert set(answers) == set(questions)


async def test_state_too_large_raises() -> None:
    client = JevClient(RecordingBackend(), state_token_budget=10)
    with pytest.raises(StateTooLargeError):
        await client.ask("x" * 200, {"q": Noul(instructions="?")})


async def test_missing_answer_raises() -> None:
    client = JevClient(RecordingBackend(drop="b"))
    with pytest.raises(MissingAnswerError, match="'b'"):
        await client.ask("s", {"a": Noul(instructions="?"), "b": Noul(instructions="?")})


async def test_empty_questions_make_no_request() -> None:
    backend = RecordingBackend()
    assert await JevClient(backend).ask("s", {}) == {}
    assert backend.calls == []


async def test_usage_is_metered_per_client() -> None:
    backend = RecordingBackend()
    shared = JevClient(backend)
    doc_a, doc_b = shared.metered(), shared.metered()
    await doc_a.ask("s", {"a": Noul(instructions="?"), "b": Noul(instructions="?")})
    await doc_b.ask("s", {"c": Noul(instructions="?")})
    assert (doc_a.usage.requests, doc_a.usage.questions, doc_a.usage.input_tokens) == (1, 2, 100)
    assert (doc_b.usage.requests, doc_b.usage.questions) == (1, 1)
    assert doc_a.usage.models == {"jev-test"}
    assert doc_a.usage.cost == pytest.approx(100 * 0.042 / 1_000_000)
    assert shared.usage.requests == 0


async def test_concurrency_limit_is_shared_by_metered_clients() -> None:
    backend = RecordingBackend(delay=0.01)
    shared = JevClient(backend, max_concurrency=2)
    clients = [shared.metered() for _ in range(6)]
    await asyncio.gather(
        *(c.ask(f"s{i}", {"q": Noul(instructions="?")}) for i, c in enumerate(clients))
    )
    assert backend.max_in_flight == 2


def test_client_survives_separate_event_loops() -> None:
    client = JevClient(RecordingBackend(), max_concurrency=1)
    for _ in range(2):
        asyncio.run(client.ask("s", {"q": Noul(instructions="?")}))
    assert client.usage.requests == 2


def test_question_validation() -> None:
    with pytest.raises(ValidationError):
        Choice(instructions="?", options={})
    with pytest.raises(ValidationError):
        Choice(instructions="?", options={str(i): None for i in range(256)})
    with pytest.raises(ValidationError):
        Score(instructions="?", levels=["only one"])
    with pytest.raises(ValidationError):
        Score(instructions="?", levels=[str(i) for i in range(11)])


def test_estimate_tokens() -> None:
    assert estimate_tokens("abcd" * 100) == 101
    assert estimate_tokens({"k": "v"}) >= 1


async def test_typesafe_backend_round_trip() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "gate": {"type": "noul", "noul": 0.97},
                    "field": {
                        "type": "choice",
                        "choice": "price",
                        "confidence": 0.8,
                        "probabilities": {"price": 0.9, "none": 0.1},
                    },
                    "rank": {
                        "type": "score",
                        "score": 1.2,
                        "confidence": 0.7,
                        "legend": {"0": "low", "1": "mid", "2": "high"},
                        "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
                    },
                },
                "usage": {"input_tokens": 321, "output_tokens": 12},
            },
        )

    sdk = AsyncTypeSafeClient(
        api_key="ts-test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )
    backend = TypeSafeBackend(sdk)
    client = JevClient(backend)
    answers = await client.ask(
        {"statement": "Price: £18,495"},
        {
            "gate": Noul(instructions="Is this a price?"),
            "field": Choice(instructions="Which field?", options={"price": "Price", "none": None}),
            "rank": Score(instructions="How relevant?", levels=["low", "mid", "high"]),
        },
    )
    await backend.aclose()

    questions = seen["questions"]
    assert isinstance(questions, dict)
    assert questions["field"] == {
        "type": "choice",
        "instructions": "Which field?",
        "criteria": {"price": "Price", "none": None},
    }
    assert questions["rank"]["criteria"] == ["low", "mid", "high"]
    assert answers["gate"] == NoulAnswer(p=0.97)
    assert answers["field"] == ChoiceAnswer(
        choice="price", confidence=0.8, probabilities={"price": 0.9, "none": 0.1}
    )
    assert answers["rank"] == ScoreAnswer(
        score=1.2, confidence=0.7, probabilities={0: 0.1, 1: 0.6, 2: 0.3}
    )
    assert client.usage.input_tokens == 321
    assert client.usage.models == {"jev-1.13.0"}


# --- Process-wide spend cap ------------------------------------------------------------


@pytest.fixture
def fresh_spend() -> Iterator[None]:
    from jevex.jev import reset_process_cost

    reset_process_cost()
    yield
    reset_process_cost()


@pytest.mark.usefixtures("fresh_spend")
async def test_spend_cap_blocks_request_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    from jevex.jev import JevBudgetExceededError, process_cost

    backend = RecordingBackend()  # reports 100 input tokens per request
    client = JevClient(backend)
    # Each request reports 100 tokens ($0.0000042) and is estimated at ~14 tokens before
    # sending. Two fit under $0.000007; after that, spend is already past the cap.
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "0.0000070")
    await client.ask("s", {"q": Noul(instructions="?")})
    await client.ask("s", {"q": Noul(instructions="?")})
    assert process_cost() == pytest.approx(2 * 100 * 0.042 / 1_000_000)
    with pytest.raises(JevBudgetExceededError, match="spend cap"):
        await client.ask("s", {"q": Noul(instructions="?")})
    assert len(backend.calls) == 2


@pytest.mark.usefixtures("fresh_spend")
async def test_spend_cap_is_shared_across_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    from jevex.jev import JevBudgetExceededError

    # One request spends $0.0000042; a second one would pass $0.0000045.
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "0.0000045")
    await JevClient(RecordingBackend()).ask("s", {"q": Noul(instructions="?")})
    with pytest.raises(JevBudgetExceededError):
        await JevClient(RecordingBackend()).ask("s", {"q": Noul(instructions="?")})


@pytest.mark.usefixtures("fresh_spend")
async def test_no_cap_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVEX_JEV_MAX_COST_USD", raising=False)
    client = JevClient(RecordingBackend())
    for _ in range(5):
        await client.ask("s", {"q": Noul(instructions="?")})
    assert client.usage.requests == 5


@pytest.mark.usefixtures("fresh_spend")
async def test_bad_cap_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from jevex.jev import JevError

    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "five dollars")
    with pytest.raises(JevError, match="must be a number"):
        await JevClient(RecordingBackend()).ask("s", {"q": Noul(instructions="?")})


async def test_sdk_errors_become_jev_backend_errors() -> None:
    from jevex.jev import JevBackendError

    def unauthorised(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"detail": "bad key"})

    sdk = AsyncTypeSafeClient(
        api_key="ts-test-key",
        transport=httpx2.MockTransport(unauthorised),
        retry=RetryPolicy(max_retries=0),
    )
    backend = TypeSafeBackend(sdk)
    with pytest.raises(JevBackendError):
        await JevClient(backend).ask("s", {"q": Noul(instructions="?")})
    await backend.aclose()


def test_missing_key_is_a_jev_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from jevex.jev import JevBackendError

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevBackendError):
        TypeSafeBackend()


async def test_max_requests_caps_a_client_and_counts_requests_as_they_start() -> None:
    capped = JevClient(RecordingBackend(), max_requests=2)
    results = await asyncio.gather(
        *(capped.ask(f"s{i}", {"q": Noul(instructions="a?")}) for i in range(5)),
        return_exceptions=True,
    )
    assert sum(isinstance(r, JevRequestCapError) for r in results) == 3
    assert capped.usage.requests == 2
    metered = JevClient(RecordingBackend()).metered(max_requests=0)
    with pytest.raises(JevRequestCapError, match="0 Jev requests"):
        await metered.ask("s", {"q": Noul(instructions="a?")})


async def test_a_request_cancelled_mid_flight_is_still_counted() -> None:
    class Hangs:
        async def system_one(
            self, state: JSONContent, questions: Mapping[str, Question]
        ) -> JevResponse:
            await asyncio.sleep(10)
            raise AssertionError("unreachable")

    client = JevClient(Hangs())
    task = asyncio.create_task(client.ask("some state", {"q": Noul(instructions="a?")}))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.usage.requests == 1
    assert client.usage.input_tokens > 0

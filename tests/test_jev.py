import asyncio
import json
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

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
    JevTokenLimitError,
    JevTransientError,
    JSONContent,
    MissingAnswerError,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    StateTooLargeError,
    TypeSafeBackend,
    estimate_question_tokens,
    estimate_request_tokens,
    estimate_tokens,
    truncate_to_tokens,
)
from jevex.jev import RetryPolicy as JevRetryPolicy


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


# --- Requests Jev rejects as too big -----------------------------------------------------


class TokenLimitedBackend(RecordingBackend):
    """Rejects, as Jev does with ``max_tokens_exceeded``, any request with more than
    ``max_questions`` questions or a state longer than ``max_state_chars``: a real count
    higher than the client's estimate."""

    def __init__(self, *, max_questions: int = 1000, max_state_chars: int = 10_000) -> None:
        super().__init__()
        self.rejected: list[list[str]] = []
        self._max_questions = max_questions
        self._max_state_chars = max_state_chars

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        if len(questions) > self._max_questions or len(str(state)) > self._max_state_chars:
            self.rejected.append(list(questions))
            raise JevTokenLimitError('400 {"detail":{"error_type":"max_tokens_exceeded"}}')
        return await super().system_one(state, questions)


def _nouls(n: int) -> dict[str, Question]:
    return {f"q{i}": Noul(instructions=f"Is question number {i} true?") for i in range(n)}


async def test_a_rejected_request_is_asked_again_in_parts() -> None:
    backend = TokenLimitedBackend(max_questions=3)
    client = JevClient(backend)
    answers = await client.ask("state", _nouls(10))
    assert list(answers) == list(_nouls(10))  # every answer, in the order asked
    assert all(len(call) <= 3 for call in backend.calls)
    assert sorted(k for call in backend.calls for k in call) == sorted(_nouls(10))
    # The longest question goes alone first, then the rest in halves.
    assert backend.rejected[0] == list(_nouls(10))
    assert backend.calls[0] == ["q0"]
    # Rejected requests count as requests but aren't billed.
    assert client.usage.requests == len(backend.calls) + len(backend.rejected)
    assert client.usage.questions == 10
    assert client.usage.input_tokens == 100 * len(backend.calls)


async def test_a_state_too_big_for_jev_raises_after_one_more_request() -> None:
    backend = TokenLimitedBackend(max_state_chars=100)
    client = JevClient(backend)
    with pytest.raises(StateTooLargeError, match="rejected the state"):
        await client.ask("x " * 100, _nouls(50))
    assert len(backend.rejected) == 2  # the batch, then its longest question alone
    assert len(backend.rejected[1]) == 1
    assert backend.calls == []
    assert client.usage.input_tokens == 0


async def test_a_single_rejected_question_raises_at_once() -> None:
    backend = TokenLimitedBackend(max_state_chars=100)
    with pytest.raises(StateTooLargeError):
        await JevClient(backend).ask("x " * 100, {"q": Noul(instructions="?")})
    assert len(backend.rejected) == 1


async def test_fit_state_cuts_text_to_the_state_budget() -> None:
    client = JevClient(RecordingBackend(), state_token_budget=100)
    questions = {"q": Noul(instructions="Is this a car?")}
    text = _table(80)
    fitted = client.fit_state(text, questions)
    assert text.startswith(fitted)
    assert estimate_tokens(fitted) + estimate_question_tokens(questions["q"]) <= 100
    await client.ask(fitted, questions)  # plans without StateTooLargeError
    with pytest.raises(StateTooLargeError, match="a question alone"):
        JevClient(RecordingBackend(), state_token_budget=5).fit_state(text, questions)


# --- Token estimates, against #6's measurements ------------------------------------------
#
# The states below are the ones #6's probe sent (benchmarks/jev_limits/probe.py, same seeds),
# each with one Noul; the counts are the input tokens Jev billed for them.

_WORDS = (  # noqa: SIM905 - the probe's word list, as it wrote it
    "engine power torque gearbox manual automatic hybrid petrol diesel electric range battery "
    "charge seats boot litres kilowatts emissions mileage warranty trim alloy wheels the a of "
    "and with in on for to from by is are offers delivers includes standard optional"
).split()


def _prose(chars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    n = 0
    while n < chars:
        sentence = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(8, 18))).capitalize()
        sentence += f" {rng.randint(10, 999)} {rng.choice(['kW', 'PS', 'mph', 'g/km', 'l'])}."
        out.append(sentence)
        n += len(sentence) + 1
    return " ".join(out)[:chars]


def _table(rows: int) -> str:
    rng = random.Random(1)
    lines = ["Specification | SE | SE L | GT | R-Line"]
    for i in range(rows):
        lines.append(
            f"Spec {i} ({rng.choice(['kW', 'mph', 's', 'g/km'])}) | "
            + " | ".join(str(rng.randint(1, 999)) for _ in range(4))
        )
    return "\n".join(lines)


def _blob(keys: int) -> dict[str, Any]:
    rng = random.Random(2)
    return {
        "vehicle": {
            f"field_{i}": {"value": rng.randint(1, 9999), "unit": rng.choice(["kW", "mph"])}
            for i in range(keys)
        }
    }


def _spec_noul(i: int) -> Noul:
    return Noul(instructions=f"Does the text state a value for spec {i}?")


MEASURED_STATES: list[tuple[str, JSONContent, int]] = [
    ("prose", _prose(4000), 1159),
    ("table", _table(80), 2522),
    ("json", _blob(60), 1859),
    ("long prose, 22k", _prose(int(28_000 * 3.451), seed=28_000), 21_905),
    ("long prose, 31k", _prose(int(40_000 * 3.451), seed=40_000), 31_178),
    ("largest accepted", _prose(int(32_500 * 4.4), seed=32_500), 32_193),
]


@pytest.mark.parametrize(
    ("kind", "state", "real"), MEASURED_STATES, ids=[kind for kind, _, _ in MEASURED_STATES]
)
def test_state_estimates_err_high_but_not_far(kind: str, state: JSONContent, real: int) -> None:
    estimate = estimate_request_tokens(state, {"q": _spec_noul(0)})
    assert real <= estimate <= real * 1.15, kind


def test_tables_and_json_are_no_longer_read_at_a_third_of_their_size() -> None:
    # The old 4-characters-per-token rule estimated these at 750 and 651.
    assert estimate_tokens(_table(80)) > 2000
    assert estimate_tokens(_blob(60)) > 1500


_LONG_QUESTION = "Does the text state the value for " + "this particular specification " * 40


@pytest.mark.parametrize(
    ("questions", "real"),
    [
        ({f"q{i}": Noul(instructions=f"Is {i} stated?") for i in range(1000)}, 14_160),
        ({f"q{i}": Noul(instructions=f"Is {i} stated?") for i in range(4400)}, 65_160),
        ({f"q{i}": Noul(instructions=f"{_LONG_QUESTION} {i}?") for i in range(200)}, 28_160),
    ],
    ids=["1000 tiny", "4400 tiny (largest accepted request)", "200 long"],
)
def test_question_estimates_are_close(questions: dict[str, Question], real: int) -> None:
    # The state is a few tokens, so the count is almost all questions.
    estimate = estimate_request_tokens("Spec 3 is 120 kW.", questions)
    assert estimate == pytest.approx(real, rel=0.06)


@pytest.mark.parametrize(("n", "real"), [(10, 1321), (50, 2081)])
def test_extra_questions_on_a_state_cost_what_jev_bills(n: int, real: int) -> None:
    state = _prose(4000)
    one = estimate_request_tokens(state, {"q": _spec_noul(0)})
    many = estimate_request_tokens(state, {f"q{i}": _spec_noul(i) for i in range(n)})
    assert many - one == pytest.approx(real - 1159, rel=0.1)


def test_a_short_question_is_estimated_as_measured() -> None:
    # Each extra Noul on the same state added about 18 tokens: (1321 - 1159) / 9.
    assert estimate_question_tokens(_spec_noul(0)) == 18


def test_options_and_levels_count_towards_a_question() -> None:
    bare = estimate_question_tokens(Choice(instructions="Which field?", options={"a": None}))
    described = estimate_question_tokens(
        Choice(instructions="Which field?", options={"a": None, "price": "The price paid"})
    )
    assert described > bare + 4
    score = Score(instructions="How relevant?", levels=["low", "mid", "high"])
    assert estimate_question_tokens(score) > estimate_question_tokens(
        Noul(instructions="How relevant?")
    )


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("the engine") == 2
    assert estimate_tokens("120") == 3  # a token per digit
    assert estimate_tokens("a | b") == 3
    assert estimate_tokens("Größe") == 4  # each non-ASCII letter on its own
    assert estimate_tokens({"k": "ö"}) == estimate_tokens('{"k": "ö"}')


def test_truncate_to_tokens_keeps_the_longest_prefix_that_fits() -> None:
    text = _table(80)
    cut = truncate_to_tokens(text, 500)
    assert text.startswith(cut)
    assert estimate_tokens(cut) <= 500 < estimate_tokens(text[: len(cut) + 1])
    assert truncate_to_tokens("short text", 500) == "short text"
    assert truncate_to_tokens("short text", 0) == ""


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


# RecordingBackend bills 100 tokens per request; the cap is checked against each request's
# estimate before it is sent.
ESTIMATE = estimate_request_tokens("s", {"q": Noul(instructions="?")})


def usd(tokens: float) -> str:
    return f"{tokens * 0.042 / 1_000_000:.12f}"


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
    # Two requests fit (100 tokens spent, plus the next one's estimate); a third doesn't.
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", usd(150 + ESTIMATE))
    await client.ask("s", {"q": Noul(instructions="?")})
    await client.ask("s", {"q": Noul(instructions="?")})
    assert process_cost() == pytest.approx(2 * 100 * 0.042 / 1_000_000)
    with pytest.raises(JevBudgetExceededError, match="spend cap"):
        await client.ask("s", {"q": Noul(instructions="?")})
    assert len(backend.calls) == 2


@pytest.mark.usefixtures("fresh_spend")
async def test_spend_cap_is_shared_across_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    from jevex.jev import JevBudgetExceededError

    # One request spends 100 tokens; a second one's estimate would pass the cap.
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", usd(50 + ESTIMATE))
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


@pytest.mark.usefixtures("fresh_spend")
async def test_ledger_cap_counts_spend_from_other_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jevex.jev import JevBudgetExceededError, process_cost

    ledger = tmp_path / "run.ledger"
    ledger.write_text("jev 0.0000040\nllm 5\n")  # another process's spend: ~95 tokens
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", usd(150 + ESTIMATE))
    backend = RecordingBackend()
    client = JevClient(backend)
    await client.ask("s", {"q": Noul(instructions="?")})  # 95 + the estimate: fits
    with pytest.raises(JevBudgetExceededError, match="spend cap"):
        await client.ask("s", {"q": Noul(instructions="?")})
    assert len(backend.calls) == 1
    assert ledger.read_text() == "jev 0.0000040\nllm 5\njev 0.000004200\n"
    assert process_cost() == pytest.approx(0.0000042)  # this process's own spend


@pytest.mark.usefixtures("fresh_spend")
async def test_ledger_records_spend_without_a_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "run.ledger"
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    monkeypatch.delenv("JEVEX_JEV_MAX_COST_USD", raising=False)
    await JevClient(RecordingBackend()).ask("s", {"q": Noul(instructions="?")})
    assert ledger.read_text() == "jev 0.000004200\n"


@pytest.mark.usefixtures("fresh_spend")
async def test_unreadable_ledger_blocks_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jevex.jev import JevError

    ledger = tmp_path / "run.ledger"
    ledger.write_text("jev ???\n")
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "1")
    backend = RecordingBackend()
    with pytest.raises(JevError, match="bad line 1"):
        await JevClient(backend).ask("s", {"q": Noul(instructions="?")})
    assert backend.calls == []


@pytest.mark.usefixtures("fresh_spend")
async def test_ledger_in_a_missing_dir_blocks_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jevex.jev import JevError

    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(tmp_path / "nope" / "run.ledger"))
    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", "1")
    backend = RecordingBackend()
    with pytest.raises(JevError, match="can't use"):
        await JevClient(backend).ask("s", {"q": Noul(instructions="?")})
    assert backend.calls == []


@pytest.mark.usefixtures("fresh_spend")
async def test_ledger_never_loosens_the_process_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jevex.jev import JevBudgetExceededError

    monkeypatch.setenv("JEVEX_JEV_MAX_COST_USD", usd(50 + ESTIMATE))
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(tmp_path / "a.ledger"))
    await JevClient(RecordingBackend()).ask("s", {"q": Noul(instructions="?")})  # $0.0000042
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(tmp_path / "b.ledger"))  # empty
    with pytest.raises(JevBudgetExceededError):
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


@pytest.mark.parametrize(
    "body",
    [{"detail": {"error_type": "max_tokens_exceeded"}}, {"error_type": "max_tokens_exceeded"}],
)
async def test_max_tokens_exceeded_becomes_a_token_limit_error(body: dict[str, Any]) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        questions = json.loads(request.content)["questions"]
        if len(questions) > 2:
            return httpx2.Response(400, json=body)
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {k: {"type": "noul", "noul": 0.5} for k in questions},
                "usage": {"input_tokens": 300},
            },
        )

    sdk = AsyncTypeSafeClient(
        api_key="ts-test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )
    backend = TypeSafeBackend(sdk)
    with pytest.raises(JevTokenLimitError):
        await backend.system_one("s", _nouls(3))
    client = JevClient(backend)
    answers = await client.ask("s", _nouls(5))
    await backend.aclose()
    assert set(answers) == set(_nouls(5))
    assert client.usage.requests == 4  # rejected; the longest alone; the rest in two halves
    assert client.usage.input_tokens == 3 * 300


async def test_other_bad_requests_are_not_token_limit_errors() -> None:
    from jevex.jev import JevBackendError

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"detail": {"error_type": "invalid_question"}})

    sdk = AsyncTypeSafeClient(
        api_key="ts-test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )
    backend = TypeSafeBackend(sdk)
    with pytest.raises(JevBackendError) as raised:
        await JevClient(backend).ask("s", _nouls(3))
    await backend.aclose()
    assert not isinstance(raised.value, JevTokenLimitError)


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
    assert client.usage.input_tokens == estimate_request_tokens(
        "some state", {"q": Noul(instructions="a?")}
    )


# --- retries ---------------------------------------------------------------------------


class FlakyBackend(RecordingBackend):
    """Fails transiently ``failures`` times (``retry_after`` on each), then answers."""

    def __init__(self, failures: int, *, retry_after: float | None = None) -> None:
        super().__init__()
        self.failures = failures
        self.retry_after = retry_after

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        if self.failures:
            self.failures -= 1
            self.calls.append(list(questions))
            raise JevTransientError("503 overloaded", retry_after=self.retry_after)
        return await super().system_one(state, questions)


NO_WAIT = JevRetryPolicy(backoff_initial=0)


async def test_a_transient_failure_is_retried_and_counted() -> None:
    backend = FlakyBackend(2)
    client = JevClient(backend, retry=NO_WAIT)
    answers = await client.ask("s", {"q": Noul(instructions="?")})
    assert answers == {"q": NoulAnswer(p=0.9)}
    assert len(backend.calls) == 3
    assert (client.usage.retries, client.usage.requests) == (2, 3)


async def test_retries_give_up_after_max_retries() -> None:
    backend = FlakyBackend(5)
    client = JevClient(backend, retry=NO_WAIT.model_copy(update={"max_retries": 1}))
    with pytest.raises(JevTransientError, match="503 overloaded"):
        await client.ask("s", {"q": Noul(instructions="?")})
    assert len(backend.calls) == 2
    assert client.usage.retries == 1


async def test_other_backend_errors_are_not_retried() -> None:
    from jevex.jev import JevBackendError

    class Down(RecordingBackend):
        async def system_one(
            self, state: JSONContent, questions: Mapping[str, Question]
        ) -> JevResponse:
            self.calls.append(list(questions))
            raise JevBackendError("bad key")

    backend = Down()
    client = JevClient(backend, retry=NO_WAIT)
    with pytest.raises(JevBackendError, match="bad key"):
        await client.ask("s", {"q": Noul(instructions="?")})
    assert (len(backend.calls), client.usage.retries) == (1, 0)


async def test_a_retry_waits_by_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    policy = JevRetryPolicy(max_retries=3, backoff_initial=1.0, backoff_jitter=0)
    await JevClient(FlakyBackend(3), retry=policy).ask("s", {"q": Noul(instructions="?")})
    assert [w for w in waits if w] == [1.0, 2.0, 4.0]  # the backend's own sleep(0) aside


def test_retry_delays() -> None:
    policy = JevRetryPolicy(backoff_initial=1.0, backoff_max=3.0, backoff_jitter=0)
    assert [policy.delay(n) for n in (1, 2, 3, 4)] == [1.0, 2.0, 3.0, 3.0]
    assert policy.delay(1, retry_after=2.5) == 2.5  # the server's wait, when longer
    assert policy.delay(1, retry_after=60) == 3.0  # but never past backoff_max
    jittered = JevRetryPolicy(backoff_initial=1.0, backoff_jitter=0.5)
    assert all(0.5 <= jittered.delay(1) <= 1.0 for _ in range(50))
    with pytest.raises(ValidationError):
        JevRetryPolicy(max_retries=-1)
    assert JevClient(RecordingBackend()).retry == JevRetryPolicy()


def test_metered_clients_keep_or_replace_the_retry_policy() -> None:
    client = JevClient(RecordingBackend(), retry=NO_WAIT)
    assert client.metered().retry == NO_WAIT
    other = JevRetryPolicy(max_retries=0)
    assert client.metered(retry=other).retry == other


@pytest.mark.parametrize(
    ("status", "headers", "retry_after"),
    [(503, {}, None), (429, {"retry-after": "2"}, 2.0), (408, {"retry-after": "soon"}, None)],
)
async def test_transient_api_errors_become_transient_errors(
    status: int, headers: dict[str, str], retry_after: float | None
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"detail": "busy"}, headers=headers)

    sdk = AsyncTypeSafeClient(
        api_key="ts-test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )
    backend = TypeSafeBackend(sdk)
    with pytest.raises(JevTransientError) as raised:
        await backend.system_one("s", _nouls(1))
    await backend.aclose()
    assert raised.value.retry_after == retry_after


async def test_connection_errors_are_transient() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    sdk = AsyncTypeSafeClient(
        api_key="ts-test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )
    backend = TypeSafeBackend(sdk)
    client = JevClient(backend, retry=NO_WAIT)
    with pytest.raises(JevTransientError):
        await client.ask("s", _nouls(1))
    await backend.aclose()
    assert client.usage.retries == NO_WAIT.max_retries


def test_the_backend_turns_the_sdks_own_retries_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import typesafe_sdk

    made: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            made.append(kwargs)

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", Client)
    TypeSafeBackend()
    TypeSafeBackend(sdk_retry=RetryPolicy(max_retries=4))
    assert [m["retry"].max_retries for m in made] == [0, 4]

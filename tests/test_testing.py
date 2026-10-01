import json
import re
import socket
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel
from pytest_socket import SocketBlockedError

import jevex.testing
from jevex import Extractor
from jevex.jev import (
    Choice,
    ChoiceAnswer,
    JevClient,
    JevResponse,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
)
from jevex.llm import LLMResponse
from jevex.testing import (
    Cassette,
    CassetteMissError,
    FakeJev,
    FakeLLM,
    LLMCassette,
    UnscriptedQuestionError,
    cassette,
    request_key,
    stale_recording,
)

VEHICLE = Choice(
    instructions="Which detail does this statement state?",
    options={"zero_to_62_s": "0-62 time", "price": "Price", "none": None},
)


async def test_rules_match_on_instructions_and_state() -> None:
    fake = FakeJev()
    fake.noul("Does this document", p=0.95)
    fake.choice("Which detail", "zero_to_62_s", state="9.1")
    fake.choice("Which detail", "price", state=re.compile(r"£\d"))
    client = fake.client()

    a = await client.ask(
        "0-62 mph in 9.1 s", {"gate": Noul(instructions="Does this document?"), "cat": VEHICLE}
    )
    b = await client.ask({"text": "Price £18,495"}, {"cat": VEHICLE})

    assert a["gate"] == NoulAnswer(p=0.95)
    assert isinstance(a["cat"], ChoiceAnswer)
    assert a["cat"].choice == "zero_to_62_s"
    assert isinstance(b["cat"], ChoiceAnswer)
    assert b["cat"].choice == "price"
    assert len(fake.calls) == 2
    assert len(fake.questions) == 3


async def test_latest_rule_wins() -> None:
    fake = FakeJev().noul(None, p=0.1).noul("special", p=0.9)
    answers = await fake.client().ask(
        "s", {"a": Noul(instructions="special?"), "b": Noul(instructions="other?")}
    )
    assert answers == {"a": NoulAnswer(p=0.9), "b": NoulAnswer(p=0.1)}


async def test_defaults_when_unscripted() -> None:
    fake = FakeJev(default_p=0.3)
    answers = await fake.client().ask(
        "s",
        {
            "n": Noul(instructions="?"),
            "c": VEHICLE,
            "e": Choice(instructions="?", options={"petrol": None, "not stated": None}),
            "f": Choice(instructions="?", options={"a": None, "b": None}),
            "s": Score(instructions="?", levels=["low", "high"]),
        },
    )
    assert answers["n"] == NoulAnswer(p=0.3)
    picked: list[str] = []
    for key in "cef":
        answer = answers[key]
        assert isinstance(answer, ChoiceAnswer)
        picked.append(answer.choice)
    assert picked == ["none", "not stated", "a"]
    assert answers["s"] == ScoreAnswer(score=0.0, confidence=1.0, probabilities={0: 1.0, 1: 0.0})


async def test_strict_mode_raises_on_unscripted_question() -> None:
    fake = FakeJev(strict=True)
    with pytest.raises(UnscriptedQuestionError, match="no rule for noul"):
        await fake.client().ask("s", {"q": Noul(instructions="anything?")})


async def test_choice_must_be_an_option_and_can_be_computed() -> None:
    fake = FakeJev().choice("Which", "missing")
    with pytest.raises(UnscriptedQuestionError, match="not an option"):
        await fake.client().ask("s", {"q": VEHICLE})

    fake = FakeJev().choice("Which", lambda q: list(q.options)[1], confidence=0.8)
    answer = (await fake.client().ask("s", {"q": VEHICLE}))["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "price"
    assert answer.probabilities["price"] == 0.8
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


async def test_score_rule() -> None:
    fake = FakeJev().score("How relevant", level=2, confidence=0.7)
    answer = (
        await fake.client().ask(
            "s", {"q": Score(instructions="How relevant?", levels=["a", "b", "c"])}
        )
    )["q"]
    assert answer == ScoreAnswer(score=2.0, confidence=0.7, probabilities={0: 0.0, 1: 0.0, 2: 0.7})


def test_request_key_is_stable_and_order_independent() -> None:
    q = {"a": Noul(instructions="?"), "b": VEHICLE}
    assert request_key({"x": 1, "y": 2}, q) == request_key(
        {"y": 2, "x": 1}, dict(reversed(q.items()))
    )
    assert request_key("s", q) != request_key("t", q)


async def test_cassette_records_then_replays(tmp_path: Path) -> None:
    path = tmp_path / "cassettes" / "golf.json"
    live = FakeJev().noul("Is it", p=0.8)
    questions = {"q": Noul(instructions="Is it fast?")}

    recorder = Cassette(path, record=True, inner=live)
    recorded = await recorder.system_one("state", questions)
    assert path.exists()
    assert len(json.loads(path.read_text())) == 1

    replayed = await Cassette(path).system_one("state", questions)
    assert replayed == recorded
    assert len(live.calls) == 1


async def test_cassette_miss_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(CassetteMissError, match="JEVEX_RECORD=1"):
        await Cassette(tmp_path / "empty.json").system_one("s", {"q": Noul(instructions="?")})


class Car(BaseModel):
    """A car."""

    model: str


class ClosableBackend:
    """A recording backend that notes when it's closed."""

    def __init__(self, fake: FakeJev | None = None, *, fail: bool = False) -> None:
        self.fake = fake or FakeJev()
        self.fail = fail
        self.closed = 0

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        return await self.fake.system_one(state, questions)

    async def aclose(self) -> None:
        self.closed += 1
        if self.fail:
            raise ConnectionError("close failed")


async def test_cassette_closes_its_inner_backend(tmp_path: Path) -> None:
    inner = ClosableBackend(FakeJev().noul("Is it", p=0.8))
    recorder = Cassette(tmp_path / "c.json", record=True, inner=inner)
    await recorder.system_one("state", {"q": Noul(instructions="Is it fast?")})
    await recorder.aclose()
    assert inner.closed == 1


async def test_extractor_closes_a_cassettes_inner_backend(tmp_path: Path) -> None:
    inner = ClosableBackend()
    async with Extractor([Car], jev=JevClient(Cassette(tmp_path / "c.json", inner=inner))):
        pass
    assert inner.closed == 1


async def test_cassette_closes_the_api_backend_it_made_to_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[ClosableBackend] = []

    def api_backend() -> ClosableBackend:
        made.append(ClosableBackend(FakeJev().noul("Is it", p=0.8)))
        return made[-1]

    monkeypatch.setattr(jevex.testing, "TypeSafeBackend", api_backend)
    recorder = Cassette(tmp_path / "c.json", record=True)
    await recorder.system_one("state", {"q": Noul(instructions="Is it fast?")})
    await recorder.aclose()
    assert [b.closed for b in made] == [1]


async def test_cassette_close_without_a_closable_inner_is_a_no_op(tmp_path: Path) -> None:
    await Cassette(tmp_path / "c.json").aclose()  # replay: no inner
    await Cassette(tmp_path / "c.json", record=True, inner=FakeJev()).aclose()  # no aclose
    await LLMCassette(tmp_path / "llm.json").aclose()
    await LLMCassette(tmp_path / "llm.json", FakeLLM([]), record=True).aclose()


async def test_cassette_close_failure_propagates(tmp_path: Path) -> None:
    inner = ClosableBackend(fail=True)
    with pytest.raises(ConnectionError, match="close failed"):
        await Cassette(tmp_path / "c.json", inner=inner).aclose()


class Title(BaseModel):
    title: str


class ClosableLLM:
    def __init__(self) -> None:
        self.llm = FakeLLM([{"title": "Dune"}])
        self.closed = 0

    async def structured[T: BaseModel](self, prompt: str, schema: type[T]) -> LLMResponse[T]:
        return await self.llm.structured(prompt, schema)

    async def aclose(self) -> None:
        self.closed += 1


async def test_llm_cassette_closes_its_inner_llm(tmp_path: Path) -> None:
    inner = ClosableLLM()
    recorder = LLMCassette(tmp_path / "llm.json", inner, record=True)
    assert (await recorder.structured("Title: Dune", Title)).output == Title(title="Dune")
    await recorder.aclose()
    assert inner.closed == 1


def test_cassette_mode_follows_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVEX_RECORD", raising=False)
    assert not cassette(tmp_path / "c.json").record
    monkeypatch.setenv("JEVEX_RECORD", "1")
    assert cassette(tmp_path / "c.json").record


def test_a_stale_recording_xfails_outside_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("JEVEX_CASSETTE_STALE_OK", raising=False)
    with pytest.raises(pytest.xfail.Exception, match="re-record it with JEVEX_RECORD=1"):
        stale_recording("no recording for request abc")


@pytest.mark.parametrize("ci", ["true", "1", "TRUE"])
def test_a_stale_recording_fails_in_ci(ci: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", ci)
    monkeypatch.delenv("JEVEX_CASSETTE_STALE_OK", raising=False)
    with pytest.raises(pytest.fail.Exception) as failed:
        stale_recording("no recording for request abc")
    assert str(failed.value) == (
        "the recording is stale: no recording for request abc; re-record it with "
        "JEVEX_RECORD=1 (or set JEVEX_CASSETTE_STALE_OK=1 to xfail instead)"
    )
    assert not failed.value.pytrace


@pytest.mark.parametrize("ok", ["", "0", "false", "no"])
def test_only_a_set_flag_lets_ci_pass_a_stale_recording(
    ok: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("JEVEX_CASSETTE_STALE_OK", ok)
    with pytest.raises(pytest.fail.Exception):
        stale_recording("x")


def test_the_stale_ok_flag_xfails_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("JEVEX_CASSETTE_STALE_OK", "1")
    with pytest.raises(pytest.xfail.Exception, match="the recording is stale: x;"):
        stale_recording("x")


@pytest.mark.parametrize("ci", ["", "0", "false"])
def test_a_ci_variable_that_is_off_is_not_ci(ci: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", ci)
    monkeypatch.delenv("JEVEX_CASSETTE_STALE_OK", raising=False)
    with pytest.raises(pytest.xfail.Exception):
        stale_recording("x")


def test_network_is_blocked() -> None:
    with (
        pytest.warns(UserWarning, match="socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


@pytest.mark.live
def test_live_marker_is_skipped_by_default() -> None:
    pytest.fail("live tests must not run without --live")

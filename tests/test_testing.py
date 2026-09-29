import json
import re
import socket
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

from jevex.jev import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from jevex.testing import (
    Cassette,
    CassetteMissError,
    FakeJev,
    UnscriptedQuestionError,
    cassette,
    request_key,
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


def test_cassette_mode_follows_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVEX_RECORD", raising=False)
    assert not cassette(tmp_path / "c.json").record
    monkeypatch.setenv("JEVEX_RECORD", "1")
    assert cassette(tmp_path / "c.json").record


def test_network_is_blocked() -> None:
    with (
        pytest.warns(UserWarning, match="socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


@pytest.mark.live
def test_live_marker_is_skipped_by_default() -> None:
    pytest.fail("live tests must not run without --live")

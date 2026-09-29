"""Test helpers: a scripted fake Jev and record/replay cassettes.

Useful for jevex's own tests and for testing your schemas without network or spend::

    fake = FakeJev()
    fake.noul("Does this document", p=0.95)
    fake.choice("Which detail", "zero_to_62_s", state="9.1 s")
    extractor = Extractor([VehicleSpec], jev=fake.client())

For realistic answers, record real responses once and replay them in CI::

    backend = cassette("tests/cassettes/golf.json")  # replay; JEVEX_RECORD=1 records
    extractor = Extractor([VehicleSpec], jev=JevClient(backend))
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from jevex.jev import (
    Choice,
    ChoiceAnswer,
    JevBackend,
    JevClient,
    JevResponse,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    TypeSafeBackend,
)

type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer
type Matcher = str | re.Pattern[str] | Callable[[str], bool] | None


def _matches(matcher: Matcher, text: str) -> bool:
    if matcher is None:
        return True
    if isinstance(matcher, str):
        return matcher in text
    if isinstance(matcher, re.Pattern):
        return matcher.search(text) is not None
    return matcher(text)


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)


@dataclass(frozen=True)
class _Rule:
    type: Literal["noul", "choice", "score"]
    instructions: Matcher
    state: Matcher
    answer: Callable[[Question], Answer]

    def applies(self, state: JSONContent, question: Question) -> bool:
        return (
            question.type == self.type
            and _matches(self.instructions, _text(question.instructions))
            and _matches(self.state, _text(state))
        )


class UnscriptedQuestionError(AssertionError):
    """A strict :class:`FakeJev` was asked a question no rule covers."""


@dataclass(frozen=True)
class FakeCall:
    state: JSONContent
    questions: dict[str, Question]


class FakeJev:
    """A :class:`~jevex.jev.JevBackend` that answers from scripted rules.

    Rules match on question type, a substring/regex/predicate of the instructions and,
    optionally, of the state. The most recently added matching rule wins, so a test can
    override a broad rule with a narrow one.

    Without a matching rule, questions get defaults (Noul ``p=default_p``; Choice picks
    ``"none"``/``"not stated"`` if offered, else the first option; Score picks level 0),
    or raise :class:`UnscriptedQuestionError` when ``strict=True``.
    """

    def __init__(self, *, strict: bool = False, default_p: float = 0.0) -> None:
        self.strict = strict
        self.default_p = default_p
        self.calls: list[FakeCall] = []
        self._rules: list[_Rule] = []

    def noul(self, instructions: Matcher, *, p: float, state: Matcher = None) -> FakeJev:
        """Answer matching Nouls with probability ``p``. ``None`` matches anything."""
        self._rules.append(_Rule("noul", instructions, state, lambda _q: NoulAnswer(p=p)))
        return self

    def choice(
        self,
        instructions: Matcher,
        choice: str | Callable[[Choice], str],
        *,
        confidence: float = 1.0,
        state: Matcher = None,
    ) -> FakeJev:
        """Pick ``choice`` (or ``choice(question)``) for matching Choices."""

        def answer(q: Question) -> Answer:
            assert isinstance(q, Choice)
            picked = choice(q) if callable(choice) else choice
            if picked not in q.options:
                raise UnscriptedQuestionError(
                    f"scripted choice {picked!r} is not an option: {list(q.options)}"
                )
            return _choice_answer(q, picked, confidence)

        self._rules.append(_Rule("choice", instructions, state, answer))
        return self

    def score(
        self,
        instructions: Matcher,
        *,
        level: int,
        confidence: float = 1.0,
        state: Matcher = None,
    ) -> FakeJev:
        def answer(q: Question) -> Answer:
            assert isinstance(q, Score)
            probabilities = {i: (confidence if i == level else 0.0) for i in range(len(q.levels))}
            return ScoreAnswer(
                score=float(level), confidence=confidence, probabilities=probabilities
            )

        self._rules.append(_Rule("score", instructions, state, answer))
        return self

    def client(self, *, max_concurrency: int = 16) -> JevClient:
        """A :class:`~jevex.jev.JevClient` backed by this fake."""
        return JevClient(self, max_concurrency=max_concurrency)

    @property
    def questions(self) -> list[Question]:
        """Every question asked, in order."""
        return [q for call in self.calls for q in call.questions.values()]

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        self.calls.append(FakeCall(state, dict(questions)))
        answers = {key: self._answer(state, q) for key, q in questions.items()}
        return JevResponse(
            answers=answers,
            input_tokens=len(_text(state)) // 4 + 1,
            model="fake-jev",
        )

    def _answer(self, state: JSONContent, question: Question) -> Answer:
        for rule in reversed(self._rules):
            if rule.applies(state, question):
                return rule.answer(question)
        if self.strict:
            raise UnscriptedQuestionError(
                f"no rule for {question.type} {_text(question.instructions)!r} "
                f"with state {_text(state)[:80]!r}"
            )
        match question:
            case Noul():
                return NoulAnswer(p=self.default_p)
            case Choice():
                fallback = next(
                    (o for o in ("none", "not stated") if o in question.options),
                    next(iter(question.options)),
                )
                return _choice_answer(question, fallback, 1.0)
            case Score():
                probabilities = {i: float(i == 0) for i in range(len(question.levels))}
                return ScoreAnswer(score=0.0, confidence=1.0, probabilities=probabilities)


def _choice_answer(question: Choice, picked: str, confidence: float) -> ChoiceAnswer:
    rest = (1.0 - confidence) / max(len(question.options) - 1, 1)
    probabilities = {o: (confidence if o == picked else rest) for o in question.options}
    return ChoiceAnswer(choice=picked, confidence=confidence, probabilities=probabilities)


# --- Cassettes -------------------------------------------------------------------------

RECORD_ENV = "JEVEX_RECORD"
_QUESTIONS: TypeAdapter[dict[str, Question]] = TypeAdapter(dict[str, Question])
_RESPONSE: TypeAdapter[JevResponse] = TypeAdapter(JevResponse)


class CassetteMissError(LookupError):
    """A replayed request has no recording. Re-record with ``JEVEX_RECORD=1``."""


def request_key(state: JSONContent, questions: Mapping[str, Question]) -> str:
    """A stable hash of one request, used to look up recordings."""
    payload = {
        "state": state,
        "questions": _QUESTIONS.dump_python(dict(questions), mode="json"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Cassette:
    """Replays recorded Jev responses from a JSON file; records them when ``record`` is on.

    In record mode, requests go to ``inner`` (by default the real API) and each response
    is saved. In replay mode, an unrecorded request raises :class:`CassetteMissError`,
    so tests never reach the network by accident.
    """

    def __init__(self, path: str | Path, *, record: bool = False, inner: JevBackend | None = None):
        self.path = Path(path)
        self.record = record
        self._inner = inner
        self._entries: dict[str, dict[str, object]] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        key = request_key(state, questions)
        if not self.record:
            if key not in self._entries:
                raise CassetteMissError(
                    f"no recording for request {key} in {self.path}; "
                    f"run with {RECORD_ENV}=1 to record it"
                )
            return _RESPONSE.validate_python(self._entries[key]["response"])
        if self._inner is None:
            self._inner = TypeSafeBackend()
        response = await self._inner.system_one(state, questions)
        self._entries[key] = {
            "request": {
                "state": state,
                "questions": _QUESTIONS.dump_python(dict(questions), mode="json"),
            },
            "response": _RESPONSE.dump_python(response, mode="json"),
        }
        self.save()
        return response

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True, default=str) + "\n"
        )


def cassette(path: str | Path, *, inner: JevBackend | None = None) -> Cassette:
    """A cassette that records when ``JEVEX_RECORD=1`` and replays otherwise."""
    return Cassette(path, record=os.environ.get(RECORD_ENV) == "1", inner=inner)

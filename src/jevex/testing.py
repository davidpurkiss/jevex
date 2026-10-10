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

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from pydantic import BaseModel, TypeAdapter

from jevex.jev import (
    Choice,
    ChoiceAnswer,
    JevBackend,
    JevClient,
    JevResponse,
    JevTokenLimitError,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    TypeSafeBackend,
)
from jevex.llm import LLMImage, LLMResponse, LLMUsage, check_budget, record, validate_output

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from jevex.llm import LLM

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
        probabilities: Mapping[str, float] | None = None,
    ) -> FakeJev:
        """Pick ``choice`` (or ``choice(question)``) for matching Choices.

        The rest of the probability is spread evenly over the other options, unless
        ``probabilities`` gives some of them (options it names that aren't offered are
        ignored).
        """

        def answer(q: Question) -> Answer:
            assert isinstance(q, Choice)
            picked = choice(q) if callable(choice) else choice
            if picked not in q.options:
                raise UnscriptedQuestionError(
                    f"scripted choice {picked!r} is not an option: {list(q.options)}"
                )
            return _choice_answer(q, picked, confidence, probabilities)

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


def _choice_answer(
    question: Choice,
    picked: str,
    confidence: float,
    given: Mapping[str, float] | None = None,
) -> ChoiceAnswer:
    fixed = {o: p for o, p in (given or {}).items() if o in question.options and o != picked}
    others = [o for o in question.options if o != picked and o not in fixed]
    rest = max(1.0 - confidence - sum(fixed.values()), 0.0) / max(len(others), 1)
    probabilities = {o: confidence if o == picked else fixed.get(o, rest) for o in question.options}
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


class _Recording:
    """The requests a recording run has recorded, so each one is asked live only once."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._done: set[str] = set()

    @asynccontextmanager
    async def once(self, key: str) -> AsyncGenerator[bool]:
        """Yields whether ``key`` still needs recording, holding its lock so a duplicate
        in flight waits for the first. It counts as recorded once the block finishes;
        a failed call (a transient error the client retries) leaves it to the next one.
        """
        async with self._locks.setdefault(key, asyncio.Lock()):
            yield key not in self._done
            self._done.add(key)


class Cassette:
    """Replays recorded Jev responses from a JSON file; records them when ``record`` is on.

    In record mode, requests go to ``inner`` (by default the real API) and each response
    is saved, as is a request Jev rejects as too big
    (:class:`~jevex.jev.JevTokenLimitError`), so the client's split replays too. In replay
    mode, an unrecorded request raises :class:`CassetteMissError`, so tests never reach
    the network by accident. :meth:`aclose` closes the API backend recording made, so a
    recording run doesn't leak connections; an ``inner`` passed in is its maker's to close.

    A request asked again while recording (two identical cards on a page, say) gets the
    answer recorded for it in this run, as a replay would: Jev can answer the same request
    differently, and a run scored on one answer but recorded with another replays to other
    numbers than the baseline it wrote.
    """

    def __init__(self, path: str | Path, *, record: bool = False, inner: JevBackend | None = None):
        self.path = Path(path)
        self.record = record
        self._inner = inner
        self._made: TypeSafeBackend | None = None
        self._entries: dict[str, dict[str, object]] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )
        self._recording = _Recording()

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        key = request_key(state, questions)
        if self.record:
            async with self._recording.once(key) as first:
                if first:
                    await self._record(key, state, questions)
        if key not in self._entries:
            raise CassetteMissError(
                f"no recording for request {key} in {self.path}; "
                f"run with {RECORD_ENV}=1 to record it"
            )
        entry = self._entries[key]
        if "rejected" in entry:
            raise JevTokenLimitError(str(entry["rejected"]))
        return _RESPONSE.validate_python(entry["response"])

    async def _record(
        self, key: str, state: JSONContent, questions: Mapping[str, Question]
    ) -> None:
        if self._inner is None:
            self._inner = self._made = TypeSafeBackend()
        request = {
            "state": state,
            "questions": _QUESTIONS.dump_python(dict(questions), mode="json"),
        }
        try:
            response = await self._inner.system_one(state, questions)
        except JevTokenLimitError as exc:
            # Part of a normal run (the client asks again in parts), so replay it too.
            self._entries[key] = {"request": request, "rejected": str(exc)}
        else:
            self._entries[key] = {
                "request": request,
                "response": _RESPONSE.dump_python(response, mode="json"),
            }
        self.save()

    async def aclose(self) -> None:
        """Close the API backend recording made, if it made one."""
        made, self._made = self._made, None
        if made is not None:
            self._inner = None
            await made.aclose()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True, default=str) + "\n"
        )


def cassette(path: str | Path, *, inner: JevBackend | None = None) -> Cassette:
    """A cassette that records when ``JEVEX_RECORD=1`` and replays otherwise."""
    return Cassette(path, record=os.environ.get(RECORD_ENV) == "1", inner=inner)


STALE_OK_ENV = "JEVEX_CASSETTE_STALE_OK"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def stale_recording(reason: str) -> NoReturn:
    """End a replay test whose recording no longer matches what jevex asks (a
    :class:`CassetteMissError`, or inputs that changed since recording).

    Outside CI the test xfails, so a build that can't reach the API isn't broken. In CI
    (``CI`` set, as every CI service does) it fails: an xfail there is easy to miss, and the
    replay test would quietly stop checking anything. ``JEVEX_CASSETTE_STALE_OK=1``
    downgrades that to an xfail, for a change that alters the questions on purpose and
    can't be re-recorded yet. Needs pytest.
    """
    import pytest

    message = f"the recording is stale: {reason}; re-record it with {RECORD_ENV}=1"
    if _env_flag("CI") and not _env_flag(STALE_OK_ENV):
        pytest.fail(f"{message} (or set {STALE_OK_ENV}=1 to xfail instead)", pytrace=False)
    pytest.xfail(message)


# --- LLMs ------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeLLMCall:
    prompt: str
    schema: type[BaseModel]
    images: tuple[LLMImage, ...] = ()


class FakeLLM:
    """A scripted :class:`~jevex.llm.LLM`: answers from a function or a queue of outputs.

    ``answer(prompt, schema)`` returns the output (a model instance or a dict/JSON it can
    validate), or raise to simulate a failure. Token usage is estimated from the text so
    budgets and cost accounting can be tested; ``price`` sets USD per million tokens.
    Images sent with a call are kept on its :class:`FakeLLMCall`.
    """

    def __init__(
        self,
        answer: Callable[[str, type[BaseModel]], object] | list[object],
        *,
        model: str = "fake-llm",
        price: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._queue: list[object] | None = None
        self._answer: Callable[[str, type[BaseModel]], object] | None = None
        if isinstance(answer, list):
            self._queue = list(cast("list[object]", answer))
        else:
            self._answer = answer
        self.model = model
        self.price = price
        self.calls: list[FakeLLMCall] = []

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        check_budget()
        self.calls.append(FakeLLMCall(prompt, schema, tuple(images)))
        if self._queue is not None:
            if not self._queue:
                raise UnscriptedQuestionError("FakeLLM has no more scripted answers")
            raw = self._queue.pop(0)
        else:
            assert self._answer is not None
            raw = self._answer(prompt, schema)
        output = validate_output(schema, raw)
        in_tokens = len(prompt) // 4 + 1
        out_tokens = len(output.model_dump_json()) // 4 + 1
        usage = LLMUsage(
            in_tokens,
            out_tokens,
            (in_tokens * self.price[0] + out_tokens * self.price[1]) / 1_000_000,
        )
        record(usage)
        return LLMResponse(output=output, usage=usage, model=self.model)


class LLMCassette:
    """Record/replay for any :class:`~jevex.llm.LLM`, like :class:`Cassette` for Jev.

    Keyed by the prompt and the schema's JSON schema, plus a digest of each image sent
    with it (so a text-only call's key is the same as before images could be sent). In
    replay mode an unrecorded call
    raises :class:`CassetteMissError`; ``JEVEX_RECORD=1`` (see :func:`llm_cassette`) records.
    A call made again while recording gets the answer recorded for it in this run.
    ``inner`` is its maker's to close.
    """

    def __init__(self, path: str | Path, inner: LLM | None = None, *, record: bool = False):
        self.path = Path(path)
        self.record = record
        self._inner = inner
        self._entries: dict[str, dict[str, object]] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )
        self._recording = _Recording()

    @staticmethod
    def key(prompt: str, schema: type[BaseModel], images: Sequence[LLMImage] = ()) -> str:
        keyed: dict[str, object] = {"prompt": prompt, "schema": schema.model_json_schema()}
        if images:
            keyed["images"] = [
                [i.content_type, hashlib.sha256(i.content).hexdigest()] for i in images
            ]
        blob = json.dumps(keyed, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        key = self.key(prompt, schema, images)
        if self.record:
            async with self._recording.once(key) as first:
                if first:
                    await self._record(key, prompt, schema, images)
        if key not in self._entries:
            raise CassetteMissError(
                f"no LLM recording {key} in {self.path}; run with {RECORD_ENV}=1 to record it"
            )
        entry = self._entries[key]
        usage = cast("dict[str, Any]", entry["usage"])
        return LLMResponse(
            output=schema.model_validate(entry["output"]),
            usage=LLMUsage(usage["input_tokens"], usage["output_tokens"], usage["cost"]),
            model=str(entry["model"]),
        )

    async def _record(
        self, key: str, prompt: str, schema: type[BaseModel], images: Sequence[LLMImage]
    ) -> None:
        if self._inner is None:
            raise CassetteMissError("recording needs an inner LLM")
        response = await (
            self._inner.structured(prompt, schema, images=images)
            if images
            else self._inner.structured(prompt, schema)
        )
        self._entries[key] = {
            "prompt": prompt,
            "output": response.output.model_dump(mode="json"),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost": response.usage.cost,
            },
            "model": response.model,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2, sort_keys=True) + "\n")


def llm_cassette(path: str | Path, inner: LLM | None = None) -> LLMCassette:
    """An LLM cassette that records when ``JEVEX_RECORD=1`` and replays otherwise."""
    return LLMCassette(path, inner, record=os.environ.get(RECORD_ENV) == "1")

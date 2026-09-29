import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from jevex import Context, Document, EntityScope, Extractor, Field, Pipeline, SchemaSpec
from jevex.interfaces import GateDecision
from jevex.jev import (
    ChoiceAnswer,
    JevClient,
    JevResponse,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    ScoreAnswer,
)
from jevex.pipeline import SchemaRun, Stage, for_each_scope
from jevex.results import FieldMeta


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")


class Book(BaseModel):
    """A book."""

    title: str = Field(description="Title")


class YesBackend:
    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        # Yes to every Noul; "none" to every Choice (it's always an option).
        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {
            k: ChoiceAnswer(choice="none", confidence=1.0, probabilities={"none": 1.0})
            if q.type == "choice"
            else NoulAnswer(p=1.0)
            for k, q in questions.items()
        }
        return JevResponse(answers=answers, input_tokens=10, model="fake")


@dataclass
class Record:
    name: str
    log: list[str] = field(default_factory=list[str])

    async def run(self, ctx: Context) -> None:
        self.log.append(self.name)


def stages(*names: str, log: list[str] | None = None) -> list[Stage]:
    shared = log if log is not None else []
    return [Record(n, shared) for n in names]


def doc() -> Document:
    return Document.from_bytes(b"<html><body>Golf</body></html>", url="https://example.com")


def ctx_for(*models: type[BaseModel]) -> Context:
    specs = [SchemaSpec.from_model(m) for m in models]
    return Context.create(doc(), specs, JevClient(YesBackend()))


# --- Composition -----------------------------------------------------------------------


def test_composition_helpers_return_new_pipelines() -> None:
    base = Pipeline(stages("clean", "layout", "select"))
    assert base.replace("layout", Record("pdf")).names == ["clean", "pdf", "select"]
    assert base.without("clean", "select").names == ["layout"]
    assert base.insert_before("layout", Record("ocr")).names == ["clean", "ocr", "layout", "select"]
    assert base.insert_after("layout", Record("ocr")).names == ["clean", "layout", "ocr", "select"]
    assert base.append(Record("learn")).names[-1] == "learn"
    assert base.names == ["clean", "layout", "select"]
    assert "layout" in base
    assert len(base) == 3


def test_unknown_stage_name_raises() -> None:
    with pytest.raises(KeyError, match="nope"):
        Pipeline(stages("a")).without("nope")


def test_duplicate_stage_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Pipeline(stages("a", "a"))


def test_stage_protocol_is_runtime_checkable() -> None:
    assert isinstance(Record("x"), Stage)


# --- Running ---------------------------------------------------------------------------


async def test_stages_run_in_order_and_are_timed() -> None:
    log: list[str] = []
    ctx = await Pipeline(stages("a", "b", "c", log=log)).run(ctx_for(Car))
    assert log == ["a", "b", "c"]
    assert set(ctx.timings) == {"a", "b", "c"}


@dataclass
class Stopper:
    name: str = "stopper"

    async def run(self, ctx: Context) -> None:
        ctx.stop(self.name, "nothing to do")


@dataclass
class GateAll:
    """Deactivates every schema whose name is not in ``keep``."""

    keep: set[str]
    name: str = "gate"

    async def run(self, ctx: Context) -> None:
        for run in ctx.active:
            passed = run.name in self.keep
            run.gate = GateDecision(p=1.0 if passed else 0.0, passed=passed)
            if not passed:
                run.deactivate()


async def test_stop_skips_later_stages() -> None:
    log: list[str] = []
    ctx = await Pipeline([*stages("a", log=log), Stopper(), *stages("b", log=log)]).run(
        ctx_for(Car)
    )
    assert log == ["a"]
    assert ctx.stopped
    assert ctx.events[0].message == "nothing to do"


async def test_run_stops_when_no_schema_is_active() -> None:
    log: list[str] = []
    ctx = await Pipeline([GateAll(keep=set()), *stages("later", log=log)]).run(ctx_for(Car, Book))
    assert log == []
    assert ctx.stopped
    assert ctx.active == []


async def test_timing_recorded_even_when_stage_fails() -> None:
    @dataclass
    class Boom:
        name: str = "boom"

        async def run(self, ctx: Context) -> None:
            raise RuntimeError("boom")

    ctx = ctx_for(Car)
    with pytest.raises(RuntimeError):
        await Pipeline([Boom()]).run(ctx)
    assert "boom" in ctx.timings


async def test_for_each_scope_runs_scopes_concurrently() -> None:
    ctx = ctx_for(Car, Book)
    for run in ctx.schemas.values():
        run.scopes = [EntityScope(label=f"{run.name}-{i}") for i in range(3)]
    ctx.schemas["Book"].deactivate()
    running = 0
    peak = 0

    async def work(run: SchemaRun, scope: EntityScope) -> str:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return scope.label

    labels = await for_each_scope(ctx, work)
    assert labels == ["Car-0", "Car-1", "Car-2"]
    assert peak == 3


# --- Extractor -------------------------------------------------------------------------


@dataclass
class AskAndStore:
    """Asks one Noul per active schema and stores the answer as a value."""

    name: str = "ask"

    async def run(self, ctx: Context) -> None:
        async def one(run: SchemaRun) -> None:
            answers = await ctx.jev.ask("state", {"q": Noul(instructions="?")})
            answer = answers["q"]
            assert isinstance(answer, NoulAnswer)
            run.set_field(
                "only", run.spec.fields[0].name, FieldMeta(value="Golf", confidence=answer.p)
            )

        await asyncio.gather(*(one(run) for run in ctx.active))


def extractor(*stages_: Stage) -> Extractor:
    return Extractor([Car, Book], jev=JevClient(YesBackend()), pipeline=Pipeline(stages_))


async def test_extract_returns_values_and_meta() -> None:
    result = await extractor(GateAll(keep={"Car"}), AskAndStore()).extract(doc())
    assert result.values == {"Car": {"only": {"model": "Golf"}}}
    meta = result.meta
    assert meta.url == "https://example.com"
    assert meta.content_type == "text/html"
    assert meta.active_schemas == ["Car"]
    assert meta.gates["Book"].passed is False
    assert (meta.jev.requests, meta.jev.questions, meta.jev.input_tokens) == (1, 1, 10)
    assert meta.jev.models == ["fake"]
    assert set(meta.timings) == {"gate", "ask"}
    assert not meta.stopped


async def test_each_document_gets_its_own_usage() -> None:
    ex = extractor(AskAndStore())
    first = await ex.extract(doc())
    second = await ex.extract(doc())
    assert first.meta.jev.requests == second.meta.jev.requests == 2
    assert ex.jev.usage.requests == 0


def test_extract_sync_reuses_one_loop() -> None:
    ex = extractor(AskAndStore())
    try:
        assert ex.extract_sync(doc()).values["Car"]["only"] == {"model": "Golf"}
        assert ex.extract_sync(doc()).meta.jev.requests == 2
    finally:
        ex.close()


async def test_extract_sync_inside_event_loop_raises() -> None:
    with pytest.raises(RuntimeError, match="await extract"):
        extractor().extract_sync(doc())


def test_extractor_validates_schemas() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Extractor([])
    with pytest.raises(ValueError, match="unique"):
        Extractor([Car, Car])


async def test_default_pipeline_is_used_when_none_given() -> None:
    async with Extractor([Car], jev=JevClient(YesBackend())) as ex:
        result = await ex.extract(doc())
    assert result.values == {}
    assert result.meta.active_schemas == ["Car"]

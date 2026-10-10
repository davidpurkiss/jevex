import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, get_args

import pytest
from pydantic import BaseModel

from jevex import (
    Context,
    Document,
    DomLocation,
    ExampleLogger,
    Extractor,
    Field,
    GeneratorLearner,
    GeneratorRecord,
    GeneratorRegistry,
    GeneratorSpec,
    Housekeeper,
    LearnedGenerators,
    LearnStage,
    Pack,
    PackDiff,
    PackError,
    PackManifest,
    Pipeline,
    SchemaSpec,
    Statement,
    StoreError,
    VerifiedExample,
    compile_pack,
    pack_generators,
)
from jevex.budgets import RunBudget, RunLedger
from jevex.entities import EntityScope
from jevex.extractor import default_pipeline
from jevex.fallback import FallbackStage
from jevex.generators import InvalidGeneratorError, RegexGenerator, default_registry
from jevex.interfaces import Learner, ParsedDocument, Scope
from jevex.jev import Choice, ChoiceAnswer, JevBackendError
from jevex.layout import Component
from jevex.learn import (
    NORMALISERS,
    PROMPT,
    GeneratorDraft,
    LearningSpend,
    LearnOutcome,
    LearnStatus,
    draft_spec,
    example_statement,
)
from jevex.llm import LLMError
from jevex.logs import log_context
from jevex.normalise import (
    BUILTIN_NORMALISERS,
    FunctionNormaliser,
    NormaliseStage,
    normalise,
    strip,
)
from jevex.select import CandidateStage, JevCandidateSelector, SelectStage
from jevex.statements import NormaliserStep
from jevex.store import MemoryLedger, SpendEntry, SQLiteStore, Store, open_store
from jevex.testing import FakeJev, FakeLLM

LOC = DomLocation(dom_path="/p")


class Car(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    launched: date = Field(description="Launch date")
    trims: list[str] = Field(default_factory=list, description="Trim names")
    gearbox: Literal["manual", "automatic"] = Field(description="Gearbox")


SPEC = SchemaSpec.from_model(Car)
FIELD = "Car.zero_to_62_s"
TEXT = "62 mph takes 9.1 seconds"
DRAFT = {"regex": r"(\d+(?:\.\d+)?) seconds", "group": 1, "normalise": ["parse_number"]}

CARS = "cars.example.com"


def cars_only(pattern: str, gid: str = "cars") -> GeneratorRegistry:
    """A base registry with one generator scoped to the :data:`CARS` source."""
    scope = Scope(sources=frozenset({CARS}))
    return GeneratorRegistry([RegexGenerator(id=gid, pattern=pattern, group=1, scope=scope)])


def example(
    text: str = TEXT,
    value: Any = 9.1,
    *,
    eid: str = "ex-1",
    field: str = FIELD,
    p: float | None = 0.95,
    source: Literal["llm", "vision", "human"] = "llm",
    evidence: tuple[int, int] | None = None,
    context: dict[str, Any] | None = None,
    document_source: str | None = None,
) -> VerifiedExample:
    if evidence is None and str(value) in text:
        start = text.index(str(value))
        evidence = (start, start + len(str(value)))
    return VerifiedExample(
        id=eid,
        field=field,
        statement=text,
        value=value,
        evidence=evidence,
        context=context if context is not None else {"heading_trail": [], "kind": "sentence"},
        source=source,
        probability=p,
        document_source=document_source,
    )


def pick(value: str) -> Callable[[Choice], str]:
    """A Choice answerer: ``value`` when offered, else "none"."""

    def answer(q: Choice) -> str:
        return value if value in q.options else "none"

    return answer


def learner(
    fake: FakeJev,
    llm: FakeLLM,
    *,
    store: Store | None = None,
    base: GeneratorRegistry | None = None,
    **kwargs: Any,
) -> GeneratorLearner:
    return GeneratorLearner(
        [SPEC],
        llm,
        fake.client(),
        generators=LearnedGenerators(store if store is not None else open_store(":memory:")),
        base=base if base is not None else GeneratorRegistry(),
        **kwargs,
    )


async def learned(fake: FakeJev, llm: FakeLLM, ex: VerifiedExample, **kw: Any) -> LearnOutcome:
    return await learner(fake, llm, **kw).learn(ex)


# --- queueing ----------------------------------------------------------------------------


async def test_only_examples_at_or_above_the_learn_threshold_are_queued_and_stored() -> None:
    store = open_store(":memory:")
    llm = FakeLLM(lambda _p, _s: DRAFT)
    lrn = learner(FakeJev().choice(None, pick("9.1")), llm, store=store, learn_threshold=0.9)
    await lrn.submit(example(eid="low", p=0.89))
    await lrn.submit(example(eid="unknown", p=None))
    await lrn.submit(example(eid="at", p=0.9))
    await lrn.submit(example(eid="human", p=None, source="human"))
    await lrn.drain()
    assert [o.example_id for o in lrn.outcomes] == ["at", "human"]
    assert {e.id for e in await store.examples(FIELD)} == {"at", "human"}
    await lrn.aclose()


async def test_submit_returns_before_learning_and_drain_waits_for_it() -> None:
    lrn = learner(FakeJev().choice(None, pick("9.1")), FakeLLM(lambda _p, _s: DRAFT))
    await lrn.submit(example())
    assert lrn.outcomes == []
    await lrn.drain()
    assert [o.status for o in lrn.outcomes] == ["accepted"]
    await lrn.aclose()


async def test_aclose_drops_queued_examples_but_keeps_them_stored() -> None:
    store = open_store(":memory:")
    started = asyncio.Event()

    class SlowLLM(FakeLLM):
        async def structured(
            self, prompt: str, schema: type[Any], *, images: Sequence[Any] = ()
        ) -> Any:
            started.set()
            await asyncio.sleep(10)
            raise AssertionError("cancelled before it answers")

    lrn = learner(FakeJev(), SlowLLM([]), store=store)
    await lrn.submit(example(eid="a"))
    await lrn.submit(example(eid="b"))
    await started.wait()
    await lrn.aclose()
    assert lrn.outcomes == []
    assert {e.id for e in await store.examples(FIELD)} == {"a", "b"}


async def test_an_unexpected_error_stops_the_worker_and_is_raised_by_drain() -> None:
    @dataclass
    class Broken:
        def questions(self, *_: Any) -> dict[str, Any]:
            raise RuntimeError("selector bug")

        def selection(self, *_: Any) -> Any:
            raise AssertionError

    lrn = learner(FakeJev(), FakeLLM(lambda _p, _s: DRAFT), selector=Broken())
    await lrn.submit(example())
    with pytest.raises(RuntimeError, match="learner worker failed") as info:
        await lrn.drain()
    assert str(info.value.__cause__) == "selector bug"
    await lrn.drain()  # raised once; the next submit starts a new worker
    await lrn.aclose()


async def test_a_dead_worker_is_not_alive_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass
    class Broken:
        def questions(self, *_: Any) -> dict[str, Any]:
            raise RuntimeError("selector bug")

        def selection(self, *_: Any) -> Any:
            raise AssertionError

    caplog.set_level(logging.INFO, logger="jevex")
    lrn = learner(FakeJev(), FakeLLM(lambda _p, _s: DRAFT), selector=Broken())
    assert lrn.alive  # no worker yet
    with log_context(url="https://cars.test/a", stage="learn"):
        await lrn.submit(example(eid="ex-1"))
    assert lrn.alive  # queued, not yet learned
    for _ in range(100):  # let the worker run until it dies
        await asyncio.sleep(0)
    assert not lrn.alive
    [died] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert died.getMessage() == "the learner worker died on example ex-1"
    assert died.exc_info is not None
    assert died.exc_info[0] is RuntimeError
    # The worker logs as the learner, not as the document that happened to start it.
    assert (died.__dict__["run_id"], died.__dict__["url"]) == (lrn.ledger.run_id, None)
    with pytest.raises(RuntimeError, match="learner worker failed"):
        await lrn.drain()
    assert lrn.alive  # the error is raised; the next submit starts a new worker
    assert lrn.deaths == 1
    await lrn.aclose()


async def test_outcomes_are_counted_by_status_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jevex")
    lrn = learner(FakeJev().choice(None, pick("9.1")), FakeLLM(lambda _p, _s: DRAFT))
    await lrn.submit(example(eid="a"))
    await lrn.drain()
    await lrn.submit(example(eid="b"))  # the published generator finds it now
    await lrn.drain()
    counts = lrn.outcome_counts()
    assert (counts["accepted"], counts["covered"], counts["budget"]) == (1, 1, 0)
    assert set(counts) == set(get_args(LearnStatus))
    accepted, covered = (r.getMessage() for r in caplog.records)
    assert accepted.startswith("learned generator ")
    assert accepted.endswith(f" for {FIELD} (snapshot 1)")
    assert covered == (
        f"example b for {FIELD} not learned: covered: the generators in use already find the value"
    )
    await lrn.aclose()


def test_settings_are_checked() -> None:
    with pytest.raises(ValueError, match="learn_threshold"):
        GeneratorLearner([SPEC], FakeLLM([]), FakeJev().client(), learn_threshold=1.1)
    with pytest.raises(ValueError, match="fallback_threshold"):
        GeneratorLearner([SPEC], FakeLLM([]), FakeJev().client(), fallback_threshold=-0.1)
    with pytest.raises(ValueError, match="sample_size"):
        GeneratorLearner([SPEC], FakeLLM([]), FakeJev().client(), sample_size=-1)


# --- synthesis ---------------------------------------------------------------------------


async def test_the_synthesis_prompt_asks_for_recall_with_the_example() -> None:
    llm = FakeLLM(lambda _p, _s: DRAFT)
    ex = example(context={"heading_trail": ["Performance"], "kind": "sentence"})
    await learned(FakeJev().choice(None, pick("9.1")), llm, ex)
    assert llm.calls[0].schema is GeneratorDraft
    assert llm.calls[0].prompt == (
        "Write a regular expression that finds one field's value in statements from documents.\n"
        "\n"
        "Field: zero_to_62_s\n"
        "Description: 0-62 mph time\n"
        "Type: a number, in s\n"
        "\n"
        "Example statement: 62 mph takes 9.1 seconds\n"
        "Section: Performance\n"
        "Its value: 9.1\n"
        "The words that state it: 9.1\n"
        "\n"
        "The pattern will run on other statements phrased like this one, from documents of the\n"
        "same kind. Aim for recall: it should find this kind of value wherever a statement like\n"
        "this states it, not only in this statement. It may match other spans too; a later step\n"
        "chooses between them.\n"
        "\n"
        '- "regex" uses RE2 syntax (no lookaround or backreferences) and is at most 500\n'
        "  characters.\n"
        '- "group" is the capture group holding the value\'s text (0 for the whole match).\n'
        '- "normalise" turns that text into the value, using only these steps, in order:\n'
        f"{NORMALISERS}"
    )


async def test_an_example_without_evidence_leaves_the_words_line_out() -> None:
    llm = FakeLLM(lambda _p, _s: DRAFT)
    ex = example("62 mph takes 1,395 seconds", 1395.0, source="human")
    assert ex.evidence is None
    await learned(FakeJev().choice(None, pick("1,395")), llm, ex)
    assert "Its value: 1395.0\n\nThe pattern will run" in llm.calls[0].prompt
    assert "The words that state it" not in llm.calls[0].prompt


async def test_the_prompt_can_be_overridden() -> None:
    llm = FakeLLM(lambda _p, _s: DRAFT)
    await learned(
        FakeJev().choice(None, pick("9.1")), llm, example(), prompt="Find {name} in {statement}"
    )
    assert llm.calls[0].prompt == f"Find zero_to_62_s in {TEXT}"
    assert PROMPT.startswith("Write a regular expression")


def test_draft_specs_get_a_content_id_and_provenance() -> None:
    draft = GeneratorDraft.model_validate(DRAFT)
    spec = draft_spec(draft, FIELD, "ex-1")
    assert spec.id.startswith("gen-")
    assert draft_spec(draft, FIELD, "ex-2").id == spec.id
    assert draft_spec(draft, "Car.launched", "ex-1").id != spec.id
    assert spec.provenance.learned_from == ["ex-1"]
    assert spec.provenance.synthesised_by == "generator_llm"
    assert spec.provenance.created is not None
    assert GeneratorSpec.from_yaml(spec.to_yaml()) == spec


def test_a_draft_from_a_document_with_a_locale_is_scoped_and_written_for_it() -> None:
    draft = GeneratorDraft.model_validate(
        {
            "regex": r"(\d+(?:,\d+)?) Sekunden am (\d+\.\d+\.\d+)",
            "group": 1,
            "normalise": ["parse_number", {"parse_number": {"decimal": "."}}, "parse_date"],
        }
    )
    unscoped = draft_spec(draft, FIELD, "ex-1")
    german = draft_spec(draft, FIELD, "ex-1", "de-DE")
    assert unscoped.scope.locale is None
    assert german.scope.locale == "de-DE"
    # localise_steps adds what de-DE needs and keeps what the draft set itself.
    assert german.to_data()["normalise"] == [
        {"parse_number": {"decimal": ","}},
        {"parse_number": {"decimal": "."}},
        "parse_date",
    ]
    # en-US reads numeric dates month first.
    assert draft_spec(draft, FIELD, "ex-1", "en-US").to_data()["normalise"] == [
        "parse_number",
        {"parse_number": {"decimal": "."}},
        {"parse_date": {"order": "mdy"}},
    ]
    assert german.to_data()["scope"] == {"locale": "de-DE"}
    assert GeneratorSpec.from_yaml(german.to_yaml()) == german
    # Learned on pages that write numbers differently, the same draft is two generators.
    ids = {unscoped.id, german.id, draft_spec(draft, FIELD, "ex-2", "en-GB").id}
    assert len(ids) == 3
    assert draft_spec(draft, FIELD, "ex-2", "de-DE").id == german.id


def test_the_draft_schema_offers_only_built_in_normalisers() -> None:
    schema = GeneratorDraft.model_json_schema()
    names = schema["properties"]["normalise"]["items"]["anyOf"][0]["enum"]
    assert names == ["parse_date", "parse_money", "parse_number", "parse_range", "strip", "unit"]


# --- outcomes ----------------------------------------------------------------------------


async def test_an_accepted_generator_is_stored_and_published_as_a_new_snapshot() -> None:
    store = open_store(":memory:")
    fake = FakeJev(strict=True).choice("Which of these is the 0-62 mph time (s)?", pick("9.1"))
    lrn = learner(fake, FakeLLM(lambda _p, _s: DRAFT), store=store)
    before = lrn.snapshot
    outcome = await lrn.learn(example())
    assert outcome.status == "accepted"
    assert outcome.snapshot == 1
    assert outcome.spec is not None
    gen_id = outcome.spec.id
    # Copy-on-write: the old snapshot is untouched, so documents holding it keep it.
    assert before.version == 0
    assert gen_id not in before.registry
    assert lrn.snapshot.version == 1
    assert lrn.snapshot.registry.ids == [gen_id]
    [record] = await store.generators()
    assert record.id == gen_id
    assert record.field == FIELD
    assert GeneratorSpec.parse(record.spec) == outcome.spec
    # Jev chose among the new candidates on the triggering statement.
    [call] = fake.calls
    assert call.state == {"statement": TEXT}
    [question] = call.questions.values()
    assert isinstance(question, Choice)
    assert list(question.options) == ["9.1", "none"]


async def test_a_value_the_generators_already_find_is_covered_without_an_llm_call() -> None:
    llm = FakeLLM([])
    fake = FakeJev(strict=True)
    outcome = await learned(fake, llm, example(), base=default_registry())
    assert outcome.status == "covered"
    assert llm.calls == []
    assert fake.calls == []


async def test_a_stored_date_value_is_compared_as_a_date() -> None:
    text = "Launched on 12 March 2024"
    ex = example(text, "2024-03-12", field="Car.launched", evidence=(12, 25))
    outcome = await learned(FakeJev(strict=True), FakeLLM([]), ex, base=default_registry())
    assert outcome.status == "covered"


@pytest.mark.parametrize(
    ("field", "value"),
    [("Car.gearbox", "manual"), ("Car.missing", 1), ("Other.zero_to_62_s", 1.0), ("Car", 1)],
)
async def test_fields_without_candidates_are_unlearnable(field: str, value: Any) -> None:
    llm = FakeLLM([])
    outcome = await learned(FakeJev(strict=True), llm, example(value=value, field=field))
    assert outcome.status == "unlearnable"
    assert llm.calls == []


async def test_a_value_that_doesnt_fit_the_field_is_unlearnable() -> None:
    outcome = await learned(FakeJev(strict=True), FakeLLM([]), example(value="fast"))
    assert outcome.status == "unlearnable"
    assert "doesn't fit zero_to_62_s" in outcome.message


# --- nested models -----------------------------------------------------------------------


class Variant(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    gearbox: Literal["manual", "automatic"] = Field(description="Gearbox")


class Range(BaseModel):
    name: str = Field(description="Range name")
    variants: list[Variant] = Field(description="Variants")


RANGE = SchemaSpec.from_model(Range)
CHILD_FIELD = "Range.variants.zero_to_62_s"


def range_learner(fake: FakeJev, llm: FakeLLM, store: Store | None = None) -> GeneratorLearner:
    return GeneratorLearner(
        [RANGE],
        llm,
        fake.client(),
        generators=LearnedGenerators(store if store is not None else open_store(":memory:")),
        base=GeneratorRegistry(),
    )


def test_all_schemas_holds_the_nested_models_specs() -> None:
    lrn = range_learner(FakeJev(), FakeLLM([]))
    assert [s.name for s in lrn.all_schemas] == ["Range", "Range.variants"]


async def test_a_nested_models_field_is_learned_for_its_child_run() -> None:
    store = open_store(":memory:")
    fake = FakeJev(strict=True).choice("Which of these is the 0-62 mph time (s)?", pick("9.1"))
    lrn = range_learner(fake, FakeLLM(lambda _p, _s: DRAFT), store)
    outcome = await lrn.learn(example(field=CHILD_FIELD))
    assert outcome.status == "accepted"
    assert outcome.spec is not None
    assert outcome.spec.field == CHILD_FIELD
    [record] = await store.generators()
    assert record.field == CHILD_FIELD
    # Documents run it on the child run (named "Range.variants"), as the candidate stage does.
    variants = RANGE.child("variants")
    statement = Statement(id="s1", text=TEXT, kind="sentence", component_id="c", location=LOC)
    found = lrn.snapshot.registry.generate(
        statement, variants.field("zero_to_62_s"), schema="Range.variants"
    )
    assert [c.raw for c in found] == ["9.1"]
    assert lrn.snapshot.registry.generate(statement, SPEC.fields[0], schema="Car") == []


@pytest.mark.parametrize(
    "field",
    [
        "Range.variants.missing",
        "Range.variants.gearbox",
        "Range.variants",
        "Range.name.zero_to_62_s",
        "Range.nope.zero_to_62_s",
        "Other.variants.zero_to_62_s",
        "Range.variants.engine.zero_to_62_s",
    ],
)
async def test_an_unknown_nested_path_is_unlearnable(field: str) -> None:
    llm = FakeLLM([])
    outcome = await range_learner(FakeJev(strict=True), llm).learn(example(field=field))
    assert outcome.status == "unlearnable"
    assert outcome.message == f"{field} isn't a candidate field"
    assert llm.calls == []


@pytest.mark.parametrize(
    "draft",
    [
        {"regex": r"(?=\d)(\d+)", "group": 1},  # lookahead isn't RE2
        {"regex": r"(\d+)", "group": 2},
        {"regex": r"(\d+)", "group": 1, "normalise": ["eval"]},
        {"regex": "x" * 501},
    ],
)
async def test_an_invalid_draft_is_rejected(draft: dict[str, Any]) -> None:
    fake = FakeJev(strict=True)
    outcome = await learned(fake, FakeLLM([draft]), example())
    assert outcome.status == "invalid_spec"
    assert outcome.spec is None
    assert fake.calls == []


async def test_a_generator_that_misses_the_value_is_rejected_without_asking_jev() -> None:
    fake = FakeJev(strict=True)
    draft = {"regex": r"(\d+) mph", "group": 1, "normalise": ["parse_number"]}
    outcome = await learned(fake, FakeLLM([draft]), example())
    assert outcome.status == "missed_trigger"
    assert outcome.message == "it finds no span with the value"
    assert outcome.spec is not None
    assert fake.calls == []


async def test_a_generator_whose_value_jev_doesnt_choose_is_rejected() -> None:
    lrn = learner(FakeJev().choice(None, "none"), FakeLLM([DRAFT]))
    outcome = await lrn.learn(example())
    assert outcome.status == "missed_trigger"
    assert outcome.message == "Jev doesn't choose its value"
    assert lrn.snapshot.version == 0


async def test_an_llm_error_is_an_outcome() -> None:
    def fail(_p: str, _s: type[BaseModel]) -> object:
        raise LLMError("overloaded")

    outcome = await learned(FakeJev(), FakeLLM(fail), example())
    assert outcome.status == "llm_error"
    assert outcome.message == "LLMError: overloaded"


async def test_a_jev_error_is_an_outcome() -> None:
    class Down(FakeJev):
        async def system_one(self, state: Any, questions: Any) -> Any:
            raise JevBackendError("503")

    outcome = await learned(Down(), FakeLLM([DRAFT]), example())
    assert outcome.status == "jev_error"
    assert outcome.spec is not None


async def test_the_run_budget_refuses_synthesis() -> None:
    store = SQLiteStore(":memory:")
    llm = FakeLLM([DRAFT])
    ledger = RunLedger(RunBudget(max_spend=0.0, period="run"), store)
    outcome = await learned(FakeJev(), llm, example(), store=store, ledger=ledger)
    assert outcome.status == "budget"
    assert llm.calls == []


async def test_the_run_jev_cap_stops_testing() -> None:
    store = SQLiteStore(":memory:")
    ledger = RunLedger(RunBudget(max_jev_spend=0.0, period="run"), store)
    fake = FakeJev(strict=True)
    outcome = await learned(fake, FakeLLM([DRAFT]), example(), store=store, ledger=ledger)
    assert outcome.status == "budget"
    assert fake.calls == []


async def test_the_learners_jev_spend_goes_in_the_run_ledger() -> None:
    store = SQLiteStore(":memory:")
    ledger = RunLedger(RunBudget(max_spend=5.0, period="run"), store)
    fake = FakeJev().choice(None, pick("9.1"))
    llm = FakeLLM([DRAFT], price=(1.0, 1.0))
    outcome = await learned(fake, llm, example(), store=store, ledger=ledger)
    assert outcome.status == "accepted"
    assert await store.spend(kind="jev", run_id=ledger.run_id) > 0
    assert await store.spend(kind="llm", run_id=ledger.run_id) > 0


class DownLedger(MemoryLedger):
    """A spend ledger that can't be read (``read``) or written (``write``)."""

    def __init__(self, *, read: bool = True, write: bool = True) -> None:
        super().__init__()
        self.read_fails, self.write_fails = read, write

    async def spend(self, **kw: Any) -> float:
        if self.read_fails:
            raise ConnectionError("ledger unreachable")
        return await super().spend(**kw)

    async def record_spend(self, entry: SpendEntry) -> None:
        if self.write_fails:
            raise ConnectionError("ledger is read-only")
        await super().record_spend(entry)


async def test_a_ledger_that_cant_be_read_makes_no_synthesis_call() -> None:
    llm = FakeLLM([DRAFT])
    ledger = RunLedger(RunBudget(max_spend=1.0, period="run"), DownLedger())
    outcome = await learned(FakeJev(strict=True), llm, example(), ledger=ledger)
    assert outcome.status == "ledger_error"
    assert outcome.message == "ConnectionError: ledger unreachable"
    assert llm.calls == []


async def test_a_ledger_that_cant_check_the_jev_cap_stops_testing() -> None:
    fake = FakeJev(strict=True)
    ledger = RunLedger(RunBudget(max_jev_spend=1.0, period="run"), DownLedger(write=False))
    outcome = await learned(fake, FakeLLM([DRAFT]), example(), ledger=ledger)
    assert outcome.status == "ledger_error"
    assert outcome.spec is not None
    assert fake.calls == []


async def test_spend_a_ledger_cant_record_is_noted_on_the_outcome() -> None:
    fake = FakeJev().choice(None, pick("9.1"))
    llm = FakeLLM([DRAFT], price=(1.0, 1.0))
    ledger = RunLedger(RunBudget(max_spend=5.0, period="run"), DownLedger(read=False))
    outcome = await learned(fake, llm, example(), ledger=ledger)
    assert outcome.status == "accepted"
    assert outcome.message == (
        "its LLM spend wasn't recorded: ConnectionError: ledger is read-only; "
        "its Jev spend wasn't recorded: ConnectionError: ledger is read-only"
    )


class NoPublish(SQLiteStore):
    """A store that can't store generators (``publish``) or read examples (``examples``)."""

    def __init__(self, *, publish: bool = True, examples: bool = False) -> None:
        super().__init__(":memory:")
        self.fail_publish, self.fail_examples = publish, examples

    async def put_generator(self, generator: GeneratorRecord) -> None:
        if self.fail_publish:
            raise StoreError("disk full")
        await super().put_generator(generator)

    async def examples(
        self, field: str | None = None, *, limit: int | None = None
    ) -> list[VerifiedExample]:
        if self.fail_examples:
            raise StoreError("connection reset")
        return await super().examples(field, limit=limit)


@pytest.mark.parametrize(
    ("store_kw", "message"),
    [({"publish": True}, "disk full"), ({"publish": False, "examples": True}, "connection reset")],
)
async def test_a_store_failure_while_learning_is_an_outcome_and_the_worker_carries_on(
    store_kw: dict[str, bool], message: str
) -> None:
    store = NoPublish(**store_kw)
    fake = FakeJev().choice(None, pick("9.1"))
    gl = learner(fake, FakeLLM(lambda _p, _s: DRAFT), store=store)
    await gl.submit(example())
    await gl.drain()
    [outcome] = gl.outcomes
    assert (outcome.status, outcome.message) == ("store_error", message)
    assert outcome.spec is not None
    assert len(gl.snapshot.registry) == 0  # nothing was published
    await gl.submit(example("9.5 seconds", 9.5, eid="ex-2"))  # the worker is still running
    await gl.drain()
    assert len(gl.outcomes) == 2
    await gl.aclose()
    await store.aclose()


async def test_spend_keeps_running_totals_without_a_run_budget() -> None:
    fake = FakeJev().choice(None, pick("9.1"))
    llm = FakeLLM([DRAFT], price=(1.0, 1.0))
    gl = learner(fake, llm)
    assert gl.spend == LearningSpend()
    assert (await gl.learn(example())).status == "accepted"
    first = gl.spend
    assert first.llm_calls == 1
    assert first.llm_cost > 0
    assert first.jev_cost > 0
    assert first.cost == first.jev_cost + first.llm_cost
    # The new generator covers the next one: nothing spent, so the difference is zero.
    assert (await gl.learn(example("9.5 seconds", 9.5, eid="ex-2"))).status == "covered"
    assert gl.spend - first == LearningSpend()


async def test_spend_counts_a_failed_llm_call_but_not_a_refused_one() -> None:
    def fail(_p: str, _s: type[BaseModel]) -> object:
        raise LLMError("overloaded")

    gl = learner(FakeJev(strict=True), FakeLLM(fail))
    assert (await gl.learn(example())).status == "llm_error"
    assert gl.spend == LearningSpend(llm_calls=1)
    store = SQLiteStore(":memory:")
    ledger = RunLedger(RunBudget(max_spend=0.0, period="run"), store)
    refused = learner(FakeJev(strict=True), FakeLLM([DRAFT]), store=store, ledger=ledger)
    assert (await refused.learn(example())).status == "budget"
    assert refused.spend == LearningSpend()


async def test_spend_counts_jev_even_when_the_generator_is_rejected() -> None:
    gl = learner(FakeJev().choice(None, pick("none")), FakeLLM([DRAFT]))
    assert (await gl.learn(example())).status == "missed_trigger"
    assert gl.spend.llm_calls == 1
    assert gl.spend.jev_cost > 0


# --- regression tests on stored examples --------------------------------------------------

STORED = "Top speed 120 mph, 0-62 in 7.5 seconds"
WIDE = {"regex": r"(\d+(?:\.\d+)?) (?:mph|seconds)", "group": 1, "normalise": ["parse_number"]}
BASE = GeneratorRegistry([RegexGenerator(id="after-in", pattern=r"in (\d+\.\d+)", group=1)])


async def stored(store: Store, *examples: VerifiedExample) -> Store:
    for ex in examples:
        await store.add_example(ex)
    return store


async def test_a_generator_that_lowers_accuracy_on_stored_examples_is_rejected() -> None:
    store = await stored(open_store(":memory:"), example(STORED, 7.5, eid="old"))
    fake = (
        FakeJev(strict=True)
        .choice(None, pick("9.1"), state=TEXT)
        # On the stored statement Jev is fooled by the new "120" span.
        .choice(None, lambda q: "120" if "120" in q.options else "7.5", state=STORED)
    )
    lrn = learner(fake, FakeLLM([WIDE]), store=store, base=BASE)
    outcome = await lrn.learn(example())
    assert outcome.status == "regressed"
    assert outcome.message == "right on 0 of the 1 stored examples it changes, down from 1"
    assert lrn.snapshot.version == 0
    assert await store.generators() == []
    # The old and new candidate sets were asked about together.
    old_new = [c for c in fake.calls if c.state == {"statement": STORED}]
    assert len(old_new) == 1
    assert {k.split("/")[0] for k in old_new[0].questions} == {"old", "new"}


@pytest.mark.parametrize(("document_source", "status"), [(CARS, "regressed"), (None, "accepted")])
async def test_stored_examples_are_tested_with_their_own_sources_generators(
    document_source: str | None, status: str
) -> None:
    # The source-scoped generator gets the stored example right only on its own source.
    old = example(STORED, 7.5, eid="old", document_source=document_source)
    store = await stored(open_store(":memory:"), old)
    fake = (
        FakeJev(strict=True)
        .choice(None, pick("9.1"), state=TEXT)
        .choice(None, lambda q: "120" if "120" in q.options else "7.5", state=STORED)
    )
    base = cars_only(r"in (\d+\.\d+)", gid="after-in")
    outcome = await learned(fake, FakeLLM([WIDE]), example(), store=store, base=base)
    assert outcome.status == status


async def test_a_generator_that_keeps_accuracy_is_accepted() -> None:
    store = await stored(open_store(":memory:"), example(STORED, 7.5, eid="old"))
    fake = FakeJev(strict=True).choice(None, pick("9.1"), state=TEXT)
    fake.choice(None, pick("7.5"), state=STORED)
    outcome = await learned(fake, FakeLLM([WIDE]), example(), store=store, base=BASE)
    assert outcome.status == "accepted"


async def test_stored_examples_it_doesnt_change_cost_no_questions() -> None:
    same = "Rapid: 0-62 in 7.5 seconds"
    store = await stored(open_store(":memory:"), example(same, 7.5, eid="old"))
    fake = FakeJev(strict=True).choice(None, pick("9.1"), state=TEXT)
    outcome = await learned(fake, FakeLLM([DRAFT]), example(), store=store, base=BASE)
    assert outcome.status == "accepted"
    assert [c.state for c in fake.calls] == [{"statement": TEXT}]


async def test_only_a_sample_of_stored_examples_is_tested() -> None:
    store = open_store(":memory:")
    for i in range(5):
        await store.add_example(example(f"Run {i}: 0-62 in 7.{i} seconds", 7 + i / 10, eid=f"s{i}"))
    fake = FakeJev().choice(None, pick("9.1"), state=TEXT)
    fake.choice(None, lambda q: next(o for o in q.options if o != "none"), state="Run")
    outcome = await learned(fake, FakeLLM([DRAFT]), example(), store=store, sample_size=2)
    assert outcome.status == "accepted"
    assert len([c for c in fake.calls if "Run" in str(c.state)]) == 2


async def test_an_unreadable_stored_example_is_left_out() -> None:
    store = await stored(open_store(":memory:"), example(STORED, "not a number", eid="bad"))
    fake = FakeJev(strict=True).choice(None, pick("9.1"), state=TEXT)
    outcome = await learned(fake, FakeLLM([WIDE]), example(), store=store, base=BASE)
    assert outcome.status == "accepted"


# --- snapshots -----------------------------------------------------------------------------


def spec_record(gen_id: str = "gen-a", regex: str = r"(\d+) secs") -> GeneratorRecord:
    spec = GeneratorSpec.parse(
        {"id": gen_id, "field": FIELD, "match": {"regex": regex, "group": 1}, "normalise": []}
    )
    return GeneratorRecord(id=gen_id, field=FIELD, spec=spec.to_data())


async def test_load_reads_the_stores_enabled_generators() -> None:
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-a"))
    await store.put_generator(spec_record("gen-b").model_copy(update={"enabled": False}))
    snapshot = await LearnedGenerators(store).load()
    assert snapshot.version == 0
    assert snapshot.registry.ids == ["gen-a"]


def spec_pack(*gen_ids: str, disables: list[str] | None = None, regex: str = r"(\d+) secs") -> Pack:
    return Pack(
        manifest=PackManifest(name="cars", version="1", disables=disables or []),
        generators=[GeneratorSpec.parse(spec_record(g, regex).spec) for g in gen_ids],
    )


async def test_load_puts_the_packs_under_the_store() -> None:
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-a"))
    await store.set_generator_enabled("gen-c", False)
    project = spec_pack("gen-a", "gen-b", disables=["gen-d"])
    community = spec_pack("gen-c", "gen-d", "gen-e")
    snapshot = await LearnedGenerators(store, packs=[project, community]).load()
    assert snapshot.registry.ids == ["gen-a", "gen-b", "gen-e"]
    # Without a store, the packs alone.
    assert (await LearnedGenerators(packs=[community]).load()).registry.ids == [
        "gen-c",
        "gen-d",
        "gen-e",
    ]


async def test_an_invalid_stored_spec_is_a_store_error() -> None:
    store = open_store(":memory:")
    await store.put_generator(GeneratorRecord(id="gen-x", field=FIELD, spec={"id": "gen-x"}))
    with pytest.raises(StoreError, match="stored generator 'gen-x' is invalid"):
        await LearnedGenerators(store).load()


def test_a_snapshot_runs_after_the_base_generators() -> None:
    learned_gen = RegexGenerator(id="learned", pattern=r"\d+")
    base = GeneratorRegistry([RegexGenerator(id="base", pattern=r"\d+ s")])
    snapshot = LearnedGenerators().current
    assert snapshot.on(base) is base
    merged = snapshot.registry.with_generator(learned_gen)
    assert base.extended(merged).ids == ["base", "learned"]


def ctx_with(statement: Statement) -> Context:
    ctx = Context.create(Document.from_bytes(b"<p/>"), [SPEC], FakeJev().client())
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={statement.id: statement},
    )
    run = ctx.schemas["Car"]
    run.scopes = [EntityScope(label="doc", component_ids=["c1"])]
    run.categories[statement.id] = ChoiceAnswer(
        choice="zero_to_62_s", confidence=0.9, probabilities={"zero_to_62_s": 0.9}
    )
    return ctx


async def test_withdraw_makes_a_snapshot_without_the_generators() -> None:
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-a"))
    await store.put_generator(spec_record("gen-b"))
    learned_gens = LearnedGenerators(store)
    before = await learned_gens.load()
    after = learned_gens.withdraw(["gen-a", "gen-unknown"])
    assert (after.version, after.registry.ids) == (1, ["gen-b"])
    assert before.registry.ids == ["gen-a", "gen-b"]  # in-flight documents keep theirs
    assert learned_gens.withdraw(["gen-unknown"]) is after
    # Only the snapshot changes: disabling is the store's (the housekeeper's) job.
    assert [r.id for r in await store.generators()] == ["gen-a", "gen-b"]


@dataclass
class Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


async def test_refresh_picks_up_generators_another_process_published() -> None:
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-a"))
    clock = Clock()
    learned_gens = LearnedGenerators(store, refresh_after=30, clock=clock)
    loaded = await learned_gens.load()
    await store.put_generator(spec_record("gen-b"))  # another worker learned it
    clock.now = 29.9
    assert await learned_gens.refresh() is loaded  # not due yet
    clock.now = 30
    refreshed = await learned_gens.refresh()
    assert (refreshed.version, refreshed.registry.ids) == (1, ["gen-a", "gen-b"])
    assert loaded.registry.ids == ["gen-a"]  # in-flight documents keep theirs
    clock.now = 60
    assert await learned_gens.refresh() is refreshed  # the store didn't change


async def test_refresh_drops_generators_another_process_disabled() -> None:
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-a"))
    await store.put_generator(spec_record("gen-b"))
    learned_gens = LearnedGenerators(store, packs=[spec_pack("gen-p")], refresh_after=0)
    assert (await learned_gens.load()).registry.ids == ["gen-a", "gen-b", "gen-p"]
    await store.set_generator_enabled("gen-a", False)  # pruned elsewhere
    await store.set_generator_enabled("gen-p", False)  # a pack's, disabled by the store
    refreshed = await learned_gens.refresh()
    assert (refreshed.version, refreshed.registry.ids) == (1, ["gen-b"])
    await learned_gens.publish(GeneratorSpec.parse(spec_record("gen-c").spec))
    later = await learned_gens.refresh()
    assert (later.version, later.registry.ids) == (2, ["gen-b", "gen-c"])


async def test_refresh_does_nothing_unless_it_can_and_should() -> None:
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-a"))
    for learned_gens in [
        LearnedGenerators(store),  # refresh_after=None
        LearnedGenerators(store, persist=False, refresh_after=0),
        LearnedGenerators(packs=[spec_pack("gen-p")], refresh_after=0),
    ]:
        loaded = await learned_gens.load()
        await store.put_generator(spec_record("gen-b"))
        assert await learned_gens.refresh() is loaded
        await store.set_generator_enabled("gen-b", False)
    with pytest.raises(ValueError, match="refresh_after must be at least 0"):
        LearnedGenerators(store, refresh_after=-1)


async def test_a_refresh_overtaken_by_a_local_change_is_dropped() -> None:
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-a"))
    await store.put_generator(spec_record("gen-b"))
    learned_gens = LearnedGenerators(store, refresh_after=0)
    await learned_gens.load()
    await store.put_generator(spec_record("gen-c"))
    release = asyncio.Event()
    reads = 0
    stored = learned_gens.stored

    async def slow_stored() -> list[GeneratorSpec]:
        nonlocal reads
        reads += 1
        specs = await stored()
        await release.wait()
        return specs

    learned_gens.stored = slow_stored
    running = asyncio.create_task(learned_gens.refresh())
    await asyncio.sleep(0)
    # A second document doesn't wait on (or repeat) the running reload.
    assert (await learned_gens.refresh()).version == 0
    withdrawn = learned_gens.withdraw(["gen-a"])  # the housekeeper, meanwhile
    release.set()
    assert await running is withdrawn  # the reload still held gen-a: dropped
    assert reads == 1
    await store.set_generator_enabled("gen-a", False)
    refreshed = await learned_gens.refresh()  # due again at once
    assert (refreshed.version, refreshed.registry.ids) == (2, ["gen-b", "gen-c"])


async def test_a_failed_refresh_raises_and_can_be_retried() -> None:
    store = open_store(":memory:")
    learned_gens = LearnedGenerators(store, refresh_after=0)
    await learned_gens.load()
    await store.put_generator(GeneratorRecord(id="gen-x", field=FIELD, spec={"id": "gen-x"}))
    with pytest.raises(StoreError, match="stored generator 'gen-x' is invalid"):
        await learned_gens.refresh()
    await store.put_generator(spec_record("gen-x"))
    assert (await learned_gens.refresh()).registry.ids == ["gen-x"]


async def test_a_document_keeps_the_snapshot_it_started_with() -> None:
    learned_gens = LearnedGenerators()
    statement = Statement(id="s1", text=TEXT, kind="sentence", component_id="c1", location=LOC)
    stage = CandidateStage(registry=GeneratorRegistry())
    in_flight = ctx_with(statement)
    in_flight.generators = learned_gens.current
    await learned_gens.publish(draft_spec(GeneratorDraft.model_validate(DRAFT), FIELD, "ex-1"))
    later = ctx_with(statement)
    later.generators = learned_gens.current
    await stage.run(in_flight)
    await stage.run(later)
    assert in_flight.schemas["Car"].candidates[("s1", "zero_to_62_s")] == []
    [cand] = later.schemas["Car"].candidates[("s1", "zero_to_62_s")]
    assert cand.raw == "9.1"
    assert cand.generator_id.startswith("gen-")


# --- the stage and the extractor -----------------------------------------------------------


def test_the_stage_is_last_in_the_default_pipeline() -> None:
    assert default_pipeline().names[-2:] == ["fallback", "learn"]


async def test_without_a_learner_the_stage_does_nothing() -> None:
    ctx = Context.create(Document.from_bytes(b"<p/>"), [SPEC], FakeJev().client())
    ctx.verified.append(example())
    await LearnStage().run(ctx)


async def test_the_stage_submits_every_verified_example() -> None:
    got: list[VerifiedExample] = []

    @dataclass
    class Recorder:
        async def submit(self, example: VerifiedExample) -> None:
            got.append(example)

    assert isinstance(Recorder(), Learner)
    ctx = Context.create(Document.from_bytes(b"<p/>"), [SPEC], FakeJev().client())
    ctx.verified.extend([example(eid="a"), example(eid="b")])
    await LearnStage(learner=Recorder()).run(ctx)
    assert [e.id for e in got] == ["a", "b"]


async def test_the_stage_hands_the_document_to_the_housekeeper_without_a_learner() -> None:
    store = open_store(":memory:")
    ctx = Context.create(Document.from_bytes(b"<p/>"), [SPEC], FakeJev().client())
    ctx.generators_ran.add("gen-a")
    ctx.housekeeper = Housekeeper(store)
    await LearnStage().run(ctx)
    assert (await store.generator_stats("gen-a")).documents == 1


class DownStore(SQLiteStore):
    """A store whose example and generator stats writes fail."""

    def __init__(self) -> None:
        super().__init__(":memory:")

    async def add_example(self, example: VerifiedExample) -> None:
        raise StoreError("disk full")

    async def record_generator_stats(self, generator_id: str, **counts: int) -> None:
        raise StoreError("disk full")


async def test_the_stage_reports_store_failures_and_carries_on() -> None:
    store = DownStore()
    ctx = Context.create(Document.from_bytes(b"<p/>"), [SPEC], FakeJev().client())
    ctx.verified.extend([example(eid="a"), example(eid="b")])
    ctx.generators_ran.add("gen-a")
    ctx.learner = ExampleLogger(store)
    ctx.housekeeper = Housekeeper(store)
    await LearnStage().run(ctx)
    assert [(e.stage, e.kind, e.part, e.count, e.fatal) for e in ctx.errors.errors] == [
        ("learn", "store", "examples", 2, False),
        ("learn", "store", "generator_stats", 1, False),
    ]
    await store.aclose()


@dataclass
class Categorised:
    """Leaves one statement categorised as the 0-62 time, as the stages before would."""

    text: str = TEXT
    name: str = "categorise"

    async def run(self, ctx: Context) -> None:
        statement = Statement(
            id="s1", text=self.text, kind="sentence", component_id="c1", location=LOC
        )
        done = ctx_with(statement)
        ctx.parsed = done.parsed
        ctx.schemas["Car"].scopes = done.schemas["Car"].scopes
        ctx.schemas["Car"].categories = done.schemas["Car"].categories


VERIFY = "The statement states that the 0-62 mph time (s) is 9.1."


def pipeline(text: str = TEXT) -> Pipeline:
    return Pipeline(
        [
            Categorised(text),
            CandidateStage(registry=GeneratorRegistry()),
            SelectStage(),
            NormaliseStage(),
            FallbackStage(),
            LearnStage(),
        ]
    )


async def test_the_extractor_learns_from_the_fallback_and_later_documents_use_it() -> None:
    fallback_llm = FakeLLM(lambda _p, _s: {"stated": True, "value": 9.1, "evidence": "9.1"})
    generator_llm = FakeLLM([DRAFT])
    fake = FakeJev().noul(VERIFY, p=0.95).choice(None, pick("9.1"))
    doc = Document.from_bytes(b"<p/>")
    async with Extractor(
        [Car],
        jev=fake.client(),
        pipeline=pipeline(),
        extraction_llm=fallback_llm,
        generator_llm=generator_llm,
    ) as extractor:
        first = await extractor.extract(doc)
        assert first.one(Car).meta.zero_to_62_s.method == "llm"
        assert first.meta.generator_snapshot == 0
        await extractor.wait_for_learning()
        lrn = await extractor.learner()
        assert lrn is not None
        [outcome] = lrn.outcomes
        assert outcome.status == "accepted"

        second = await extractor.extract(doc)
        meta = second.one(Car).meta.zero_to_62_s
        assert second.meta.generator_snapshot == 1
        assert meta.method == "generator"
        assert outcome.spec is not None
        assert meta.generator_id == outcome.spec.id
        assert second.one(Car).record.zero_to_62_s == 9.1
        assert len(fallback_llm.calls) == 1
        assert len(generator_llm.calls) == 1
        store = await extractor.store()
        assert store is not None
        assert [e.field for e in await store.examples(FIELD)] == [FIELD]


async def test_a_generator_learned_on_a_german_page_doesnt_run_on_an_english_one() -> None:
    fallback_llm = FakeLLM(lambda _p, _s: {"stated": True, "value": 9.1, "evidence": "9,1"})
    generator_llm = FakeLLM(lambda _p, _s: GERMAN_DRAFT)
    fake = FakeJev().noul(VERIFY, p=0.95).choice(None, pick("9,1"))
    de = Document.from_bytes(b'<html lang="de-DE"><p/></html>')
    uk = Document.from_bytes(b"<p/>", locale="en-GB")
    async with Extractor(
        [Car],
        jev=fake.client(),
        pipeline=pipeline(GERMAN),
        extraction_llm=fallback_llm,
        generator_llm=generator_llm,
    ) as extractor:
        await extractor.extract(de)
        await extractor.wait_for_learning()
        lrn = await extractor.learner()
        assert lrn is not None
        [outcome] = lrn.outcomes
        assert outcome.status == "accepted"
        assert outcome.spec is not None
        assert outcome.spec.scope.locale == "de-DE"

        again = (await extractor.extract(de)).one(Car).meta.zero_to_62_s
        assert (again.method, again.generator_id) == ("generator", outcome.spec.id)
        assert again.value == 9.1

        english = (await extractor.extract(uk)).one(Car).meta.zero_to_62_s
        assert english.method == "llm"
        assert english.generator_id is None
        assert len(fallback_llm.calls) == 2
        # The English page's example is tested under en-GB, where "9,1" isn't 9.1.
        await extractor.wait_for_learning()
        assert [o.status for o in lrn.outcomes] == ["accepted", "missed_trigger"]


UK_PAGE = Document.from_bytes(b'<html lang="en-GB"><p/></html>')
UNTAGGED = Document.from_bytes(b"%PDF-1.7", content_type="application/pdf")
DE_PAGE = Document.from_bytes(b'<html lang="de-DE"><p/></html>')


def uk_learner(locale: str | None) -> tuple[Extractor, FakeLLM]:
    fallback_llm = FakeLLM(lambda _p, _s: {"stated": True, "value": 9.1, "evidence": "9.1"})
    fake = FakeJev().noul(VERIFY, p=0.95).choice(None, pick("9.1"))
    extractor = Extractor(
        [Car],
        jev=fake.client(),
        pipeline=pipeline(),
        extraction_llm=fallback_llm,
        generator_llm=FakeLLM(lambda _p, _s: DRAFT),
        locale=locale,
    )
    return extractor, fallback_llm


async def learn_once(extractor: Extractor, document: Document) -> GeneratorSpec:
    await extractor.extract(document)
    await extractor.wait_for_learning()
    lrn = await extractor.learner()
    assert lrn is not None
    [outcome] = lrn.outcomes
    assert outcome.status == "accepted"
    assert outcome.spec is not None
    return outcome.spec


@pytest.mark.parametrize(("locale", "method"), [("en_GB", "generator"), (None, "llm")])
async def test_with_the_extractors_locale_a_tagged_pages_generator_runs_on_untagged_ones(
    locale: str | None, method: str
) -> None:
    # A mixed corpus: an en-GB page, then a PDF that says nothing.
    extractor, fallback_llm = uk_learner(locale)
    async with extractor:
        learned = await learn_once(extractor, UK_PAGE)
        assert learned.scope.locale == "en-GB"
        pdf = (await extractor.extract(UNTAGGED)).one(Car).meta.zero_to_62_s
        assert pdf.method == method
        assert pdf.value == 9.1
        assert (pdf.generator_id == learned.id) is (method == "generator")
        german = (await extractor.extract(DE_PAGE)).one(Car).meta.zero_to_62_s
        assert german.method == "llm"  # a page that says another locale keeps its own
        assert len(fallback_llm.calls) == (2 if method == "generator" else 3)


async def test_a_generator_learned_on_an_untagged_document_is_scoped_to_the_default() -> None:
    extractor, _ = uk_learner("en-gb")
    async with extractor:
        learned = await learn_once(extractor, UNTAGGED)
        assert learned.scope.locale == "en-GB"
        store = await extractor.store()
        assert store is not None
        [example] = await store.examples(FIELD)
        assert example.locale == "en-GB"
        for document, method in [(UK_PAGE, "generator"), (DE_PAGE, "llm")]:
            meta = (await extractor.extract(document)).one(Car).meta.zero_to_62_s
            assert meta.method == method


async def test_without_a_default_an_untagged_documents_generator_is_unscoped() -> None:
    extractor, _ = uk_learner(None)
    async with extractor:
        learned = await learn_once(extractor, UNTAGGED)
        assert learned.scope.locale is None
        meta = (await extractor.extract(DE_PAGE)).one(Car).meta.zero_to_62_s
        assert meta.method == "generator"  # runs everywhere, German pages included


async def test_answers_below_the_learn_threshold_arent_learned() -> None:
    fallback_llm = FakeLLM(lambda _p, _s: {"stated": True, "value": 9.1, "evidence": "9.1"})
    generator_llm = FakeLLM([])
    fake = FakeJev().noul(VERIFY, p=0.85).choice(None, pick("9.1"))
    async with Extractor(
        [Car],
        jev=fake.client(),
        pipeline=pipeline(),
        extraction_llm=fallback_llm,
        generator_llm=generator_llm,
        learn_threshold=0.9,
    ) as extractor:
        result = await extractor.extract(Document.from_bytes(b"<p/>"))
        assert result.one(Car).meta.zero_to_62_s.verified
        await extractor.wait_for_learning()
        lrn = await extractor.learner()
        assert lrn is not None
        assert lrn.outcomes == []
        assert generator_llm.calls == []


async def test_a_stores_learned_generators_are_used_without_a_generator_llm() -> None:
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-a", r"(\d+(?:\.\d+)?) seconds"))
    fake = FakeJev().choice(None, pick("9.1"))
    async with Extractor([Car], jev=fake.client(), pipeline=pipeline(), store=store) as ex:
        assert await ex.learner() is None
        result = await ex.extract(Document.from_bytes(b"<p/>"))
    assert result.meta.generator_snapshot == 0
    assert result.one(Car).meta.zero_to_62_s.generator_id == "gen-a"


async def test_a_project_packs_generators_are_used_without_a_store(tmp_path: Path) -> None:
    spec_pack("gen-p", regex=r"(\d+(?:\.\d+)?) seconds").write(tmp_path / "cars")
    fake = FakeJev().choice(None, pick("9.1"))
    async with Extractor(
        [Car], jev=fake.client(), pipeline=pipeline(), packs=[tmp_path / "cars"]
    ) as ex:
        assert [p.name for p in await ex.packs()] == ["cars"]
        result = await ex.extract(Document.from_bytes(b"<p/>"))
        assert await ex.store() is None
    assert result.meta.generator_snapshot == 0
    assert result.one(Car).meta.zero_to_62_s.generator_id == "gen-p"


async def test_the_store_disables_a_packs_generator_for_the_extractor() -> None:
    store = open_store(":memory:")
    await store.set_generator_enabled("gen-p", False)
    packs = [spec_pack("gen-p", regex=r"(\d+(?:\.\d+)?) seconds")]
    fake = FakeJev().choice(None, pick("9.1"))
    async with Extractor(
        [Car], jev=fake.client(), pipeline=pipeline(), store=store, packs=packs
    ) as ex:
        learned = await ex.learned_generators()
        assert learned is not None
        assert learned.current.registry.ids == []
        result = await ex.extract(Document.from_bytes(b"<p/>"))
    assert result.records == []  # nothing found the value


async def test_documents_pick_up_generators_other_workers_learned(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'jevex.db'}"
    other = open_store(url)  # another worker's connection to the same store
    fake = FakeJev().choice(None, pick("9.1"))
    doc = Document.from_bytes(b"<p/>")
    async with (
        Extractor(
            [Car], jev=fake.client(), pipeline=pipeline(), store=url, refresh_generators=0
        ) as fresh,
        Extractor(
            [Car], jev=fake.client(), pipeline=pipeline(), store=url, refresh_generators=None
        ) as frozen,
    ):
        for ex in (fresh, frozen):
            assert (await ex.extract(doc)).records == []
        await other.put_generator(spec_record("gen-a", r"(\d+(?:\.\d+)?) seconds"))
        result = await fresh.extract(doc)
        assert result.meta.generator_snapshot == 1
        assert result.one(Car).meta.zero_to_62_s.generator_id == "gen-a"
        assert (await frozen.extract(doc)).records == []  # never refreshes

        await other.set_generator_enabled("gen-a", False)
        result = await fresh.extract(doc)
        assert (result.meta.generator_snapshot, result.records) == (2, [])
    await other.aclose()
    with pytest.raises(ValueError, match="refresh_generators must be at least 0"):
        Extractor([Car], refresh_generators=-1)


def test_wait_for_learning_sync_finishes_what_extract_sync_queued() -> None:
    fallback_llm = FakeLLM(lambda _p, _s: {"stated": True, "value": 9.1, "evidence": "9.1"})
    fake = FakeJev().noul(VERIFY, p=0.95).choice(None, pick("9.1"))
    ex = Extractor(
        [Car],
        jev=fake.client(),
        pipeline=pipeline(),
        extraction_llm=fallback_llm,
        generator_llm=FakeLLM([DRAFT]),
    )
    try:
        assert ex.extract_sync(Document.from_bytes(b"<p/>")).meta.generator_snapshot == 0
        ex.wait_for_learning_sync()
        assert ex.extract_sync(Document.from_bytes(b"<p/>")).meta.generator_snapshot == 1
        assert len(fallback_llm.calls) == 1
    finally:
        ex.close()


async def test_wait_for_learning_sync_inside_event_loop_raises() -> None:
    with pytest.raises(RuntimeError, match=r"wait_for_learning_sync\(\) .* await wait_for"):
        Extractor([Car], jev=FakeJev().client()).wait_for_learning_sync()


async def test_the_extractor_takes_community_packs_as_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[list[str] | None] = []

    def community(names: list[str] | None = None) -> list[Pack]:
        asked.append(names)
        return [spec_pack("gen-c")]

    monkeypatch.setattr("jevex.extractor.community_packs", community)
    for setting, expected in ((True, [None]), (["cars"], [["cars"]]), (False, [])):
        asked.clear()
        ex = Extractor([Car], jev=FakeJev().client(), community_packs=setting)
        found = await ex.packs()
        assert asked == expected
        assert len(found) == len(expected)
        await ex.aclose()


async def test_a_pack_that_wont_load_fails_the_extract(tmp_path: Path) -> None:
    async with Extractor(
        [Car], jev=FakeJev().client(), pipeline=pipeline(), packs=[tmp_path / "missing"]
    ) as ex:
        with pytest.raises(PackError, match="no such pack directory"):
            await ex.extract(Document.from_bytes(b"<p/>"))


async def test_without_a_store_or_learner_there_is_no_snapshot() -> None:
    async with Extractor([Car], jev=FakeJev().client(), pipeline=pipeline()) as ex:
        result = await ex.extract(Document.from_bytes(b"<p/>"))
        await ex.wait_for_learning()
    assert result.meta.generator_snapshot is None


def test_the_extractor_checks_the_learn_threshold() -> None:
    with pytest.raises(ValueError, match="learn_threshold"):
        Extractor([Car], generator_llm=FakeLLM([]), learn_threshold=1.5)


async def test_a_value_jev_picks_without_confidence_misses_the_trigger() -> None:
    fake = FakeJev().choice(None, pick("9.1"), confidence=0.4)
    outcome = await learned(fake, FakeLLM([DRAFT]), example(), fallback_threshold=0.5)
    assert outcome.status == "missed_trigger"
    assert outcome.message == "Jev doesn't choose its value"


async def test_generators_are_run_with_the_learners_locale() -> None:
    scoped = RegexGenerator(
        id="uk", pattern=r"(\d+\.\d+) seconds", group=1, scope=Scope(locale="en-GB")
    )
    base = GeneratorRegistry([scoped])
    unknown = await learned(FakeJev(), FakeLLM([DRAFT]), example(), base=base)
    assert unknown.status != "covered"
    uk = await learned(FakeJev(), FakeLLM([]), example(), base=base, locale="en-GB")
    assert uk.status == "covered"


GERMAN = "Von 0 auf 100 in 9,1 Sekunden"
GERMAN_DRAFT = {"regex": r"(\d+(?:,\d+)?) Sekunden", "group": 1, "normalise": ["parse_number"]}


def german(locale: str | None = "de-DE", **kw: Any) -> VerifiedExample:
    context: dict[str, Any] = {"heading_trail": [], "kind": "sentence"}
    if locale is not None:
        context["locale"] = locale
    return example(
        GERMAN, 9.1, evidence=(GERMAN.index("9,1"), GERMAN.index("9,1") + 3), context=context, **kw
    )


async def test_an_example_with_a_locale_gives_a_generator_scoped_to_it() -> None:
    store = open_store(":memory:")
    fake = FakeJev(strict=True).choice("Which of these is the 0-62 mph time (s)?", pick("9,1"))
    lrn = learner(fake, FakeLLM([GERMAN_DRAFT]), store=store)
    outcome = await lrn.learn(german())
    assert outcome.status == "accepted"
    assert outcome.spec is not None
    assert outcome.spec.scope.locale == "de-DE"
    assert outcome.spec.to_data()["normalise"] == [{"parse_number": {"decimal": ","}}]
    [record] = await store.generators()
    assert GeneratorSpec.parse(record.spec) == outcome.spec
    # It runs on German pages, reading "12,5" as they write it, and nowhere else.
    registry = lrn.snapshot.registry
    field = SPEC.field("zero_to_62_s")
    statement = example_statement(example("Von 0 auf 100 in 12,5 Sekunden", 12.5))
    assert len(registry.generate(statement, field, schema="Car", locale="de_de")) == 1
    for locale in ("en-GB", "de", "de-AT", None):
        assert registry.generate(statement, field, schema="Car", locale=locale) == []
    [found] = registry.generate(statement, field, schema="Car", locale="de-DE")
    assert normalise(found.raw, found.normalise, field) == 12.5


async def test_an_example_without_a_locale_gives_an_unscoped_generator() -> None:
    outcome = await learned(FakeJev().choice(None, pick("9.1")), FakeLLM([DRAFT]), example())
    assert outcome.status == "accepted"
    assert outcome.spec is not None
    assert outcome.spec.scope.locale is None
    assert outcome.spec.to_data()["normalise"] == ["parse_number"]


@pytest.mark.parametrize(("locale", "status"), [("de-DE", "accepted"), (None, "missed_trigger")])
async def test_the_trigger_is_read_by_the_examples_locale(locale: str | None, status: str) -> None:
    # Unlocalised, parse_number reads "9,1" as 9 (or 91), not the example's 9.1.
    fake = FakeJev().choice(None, pick("9,1"))
    outcome = await learned(fake, FakeLLM([GERMAN_DRAFT]), german(locale))
    assert outcome.status == status


async def test_the_examples_locale_scopes_the_generators_it_is_tested_with() -> None:
    # A de-DE generator covers the German example, whatever the learner's
    # own locale; on an example with no locale, the learner's is used.
    scoped = RegexGenerator(
        id="de",
        pattern=r"(\d+,\d+) Sekunden",
        group=1,
        normalise=(NormaliserStep(name="parse_number"),),
        scope=Scope(locale="de-DE"),
    )
    base = GeneratorRegistry([scoped])
    for lrn_locale in (None, "en-GB"):
        outcome = await learned(FakeJev(), FakeLLM([]), german(), base=base, locale=lrn_locale)
        assert outcome.status == "covered"
    uk = await learned(
        FakeJev().choice(None, pick("9,1")), FakeLLM([GERMAN_DRAFT]), german("en-GB"), base=base
    )
    assert uk.status != "covered"
    unknown = await learned(FakeJev(), FakeLLM([]), german(None), base=base, locale="de-DE")
    assert unknown.status == "covered"


async def test_stored_examples_are_tested_under_their_own_locales() -> None:
    # The new de-DE generator would change the candidates on both stored statements, but
    # runs only on the German one: the English one costs no question.
    old_german = "Von 0 auf 100 in 7,5 Sekunden"
    english = example("0-62 takes 7,5 Sekunden", 7.5, eid="uk", context={"locale": "en-GB"})
    other = example(old_german, 7.5, eid="de-old", context={"locale": "de-DE"})
    store = await stored(open_store(":memory:"), english, other)
    fake = FakeJev(strict=True).choice(None, lambda q: next(o for o in q.options if o != "none"))
    outcome = await learned(fake, FakeLLM([GERMAN_DRAFT]), german(), store=store)
    assert outcome.status == "accepted"
    states = [c.state for c in fake.calls]
    assert sorted(states, key=str) == [{"statement": old_german}, {"statement": GERMAN}]


@pytest.mark.parametrize("document_source", [CARS, "www.Cars.Example.com"])
async def test_a_source_scoped_generator_covers_an_example_from_its_source(
    document_source: str,
) -> None:
    llm = FakeLLM([])
    fake = FakeJev(strict=True)
    ex = example(document_source=document_source)
    outcome = await learned(fake, llm, ex, base=cars_only(r"(\d+\.\d+) seconds"))
    assert outcome.status == "covered"
    assert llm.calls == []
    assert fake.calls == []


@pytest.mark.parametrize("document_source", ["vans.example.com", None])
async def test_a_source_scoped_generator_doesnt_cover_an_example_from_elsewhere(
    document_source: str | None,
) -> None:
    llm = FakeLLM([DRAFT])
    ex = example(document_source=document_source)
    lrn = learner(FakeJev().choice(None, pick("9.1")), llm, base=cars_only(r"(\d+\.\d+) seconds"))
    outcome = await lrn.learn(ex)
    assert outcome.status == "accepted"
    assert len(llm.calls) == 1


async def test_jev_chooses_on_the_trigger_among_its_sources_candidates() -> None:
    fake = FakeJev().choice(None, pick("9.1"))
    ex = example(document_source=CARS)
    outcome = await learned(fake, FakeLLM([DRAFT]), ex, base=cars_only(r"(\d+) mph"))
    assert outcome.status == "accepted"
    [call] = fake.calls
    [question] = call.questions.values()
    assert isinstance(question, Choice)
    assert sorted(question.options) == ["62", "9.1", "none"]


async def test_values_are_normalised_with_the_learners_normalisers() -> None:
    def tenths(value: Any, **_: Any) -> float:
        return int(value) / 10

    custom = BUILTIN_NORMALISERS.with_normaliser(FunctionNormaliser("tenths", tenths))
    base = GeneratorRegistry(
        [
            RegexGenerator(
                id="t", pattern=r"(\d+) tenths", group=1, normalise=(NormaliserStep(name="tenths"),)
            )
        ]
    )
    ex = example("It takes 91 tenths", 9.1)
    outcome = await learned(FakeJev(), FakeLLM([]), ex, base=base, normalisers=custom)
    assert outcome.status == "covered"


async def test_the_extractors_learner_tests_as_the_pipeline_runs() -> None:
    registry = GeneratorRegistry()
    norm = BUILTIN_NORMALISERS.with_normaliser(FunctionNormaliser("noop", strip))
    selector = JevCandidateSelector(accept_at=0.7)
    stages = (
        default_pipeline()
        .replace("candidates", CandidateStage(registry=registry, locale="en-GB"))
        .replace("select", SelectStage(selector=selector))
        .replace("normalise", NormaliseStage(registry=norm))
        .replace("fallback", FallbackStage(fallback_threshold=0.7))
    )
    async with Extractor(
        [Car], jev=FakeJev().client(), pipeline=stages, generator_llm=FakeLLM([])
    ) as extractor:
        lrn = await extractor.learner()
        assert lrn is not None
        assert lrn.base is registry
        assert lrn.locale == "en-GB"
        assert lrn.selector is selector
        assert lrn.normalisers is norm
        assert lrn.fallback_threshold == 0.7
        assert lrn.learn_threshold == 0.9


# --- learning modes ----------------------------------------------------------------------


async def test_compile_mode_only_logs_examples() -> None:
    fallback_llm = FakeLLM(lambda _p, _s: {"stated": True, "value": 9.1, "evidence": "9.1"})
    generator_llm = FakeLLM([])
    fake = FakeJev().noul(VERIFY, p=0.95).choice(None, pick("9.1"))
    store = open_store(":memory:")
    async with Extractor(
        [Car],
        jev=fake.client(),
        pipeline=pipeline(),
        store=store,
        extraction_llm=fallback_llm,
        generator_llm=generator_llm,
        learn_mode="compile",
    ) as extractor:
        assert await extractor.learner() is None
        first = await extractor.extract(Document.from_bytes(b"<p/>"))
        await extractor.wait_for_learning()
        second = await extractor.extract(Document.from_bytes(b"<p/>"))
        [logged] = await store.examples(FIELD)  # the same statement: one example
    assert first.one(Car).meta.zero_to_62_s.method == "llm"
    assert second.one(Car).meta.zero_to_62_s.method == "llm"  # nothing was learned
    assert generator_llm.calls == []
    assert await store.generators() == []
    assert (logged.field, logged.value, logged.statement) == (FIELD, 9.1, TEXT)


async def test_compile_mode_logs_only_examples_at_the_learn_threshold() -> None:
    store = open_store(":memory:")
    log = ExampleLogger(store, learn_threshold=0.9)
    await log.submit(example(eid="low", p=0.89))
    await log.submit(example(eid="unknown", p=None))
    await log.submit(example(eid="at", p=0.9))
    await log.submit(example(eid="human", p=None, source="human"))
    assert {e.id for e in await store.examples(FIELD)} == {"at", "human"}
    assert isinstance(log, Learner)
    with pytest.raises(ValueError, match="learn_threshold"):
        ExampleLogger(store, learn_threshold=-0.1)


async def test_hybrid_mode_learns_inline() -> None:
    async with Extractor(
        [Car],
        jev=FakeJev().client(),
        store=":memory:",
        generator_llm=FakeLLM([]),
        learn_mode="hybrid",
    ) as extractor:
        assert isinstance(await extractor.learner(), GeneratorLearner)


def test_the_extractor_checks_the_learn_mode() -> None:
    with pytest.raises(ValueError, match="learn_mode must be one of"):
        Extractor([Car], learn_mode="batch")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="learn_mode='compile' keeps what it learns"):
        Extractor([Car], generator_llm=FakeLLM([]), learn_mode="compile")
    with pytest.raises(ValueError, match="learn_mode='hybrid' keeps what it learns"):
        Extractor([Car], generator_llm=FakeLLM([]), learn_mode="hybrid")
    with pytest.raises(ValueError, match="learn_threshold"):
        Extractor([Car], store=":memory:", learn_mode="compile", learn_threshold=2)


async def test_a_learner_that_doesnt_persist_publishes_only_in_memory() -> None:
    store = open_store(":memory:")
    learned = LearnedGenerators(store, persist=False)
    spec = GeneratorSpec.parse(spec_record("gen-a").spec)
    snapshot = await learned.publish(spec)
    assert snapshot.registry.ids == ["gen-a"]
    assert await store.generators() == []


# --- batch compiles ----------------------------------------------------------------------


def compiler(fake: FakeJev, llm: FakeLLM, store: Store, **kwargs: Any) -> GeneratorLearner:
    return GeneratorLearner(
        [SPEC],
        llm,
        fake.client(),
        generators=LearnedGenerators(store, persist=False),
        base=GeneratorRegistry(),
        **kwargs,
    )


def pack_spec(gen_id: str, regex: str = r"(\d+(?:\.\d+)?) seconds") -> GeneratorSpec:
    return GeneratorSpec.parse(spec_record(gen_id, regex).spec)


async def test_compile_pack_learns_from_stored_examples_and_publishes_nothing() -> None:
    store = open_store(":memory:")
    await store.add_example(example(eid="ex-1"))
    await store.add_example(example("0-62 in 8.4 seconds", 8.4, eid="ex-2"))
    llm = FakeLLM([DRAFT])
    diff = await compile_pack(compiler(FakeJev().choice(None, pick("9.1")), llm, store))
    [accepted, covered] = diff.outcomes
    assert (accepted.example_id, accepted.status) == ("ex-1", "accepted")
    # ex-2 is newer: the generator learned from ex-1 is already in use for it.
    assert (covered.example_id, covered.status) == ("ex-2", "covered")
    assert len(llm.calls) == 1
    assert accepted.spec is not None
    assert diff.generators == [accepted.spec]
    assert diff.generators[0].provenance.learned_from == ["ex-1"]
    assert await store.generators() == []


async def test_compile_pack_skips_examples_a_pack_already_covers() -> None:
    store = open_store(":memory:")
    await store.add_example(example())
    llm = FakeLLM([])
    pack = [pack_spec("pack-gen")]
    diff = await compile_pack(compiler(FakeJev(), llm, store), pack)
    assert [o.status for o in diff.outcomes] == ["covered"]
    assert diff.generators == []
    assert llm.calls == []


async def test_a_pack_generator_the_store_disables_isnt_in_use() -> None:
    store = open_store(":memory:")
    await store.add_example(example())
    await store.set_generator_enabled("pack-gen", False)
    llm = FakeLLM([DRAFT])
    pack = [pack_spec("pack-gen")]
    diff = await compile_pack(compiler(FakeJev().choice(None, pick("9.1")), llm, store), pack)
    assert [o.status for o in diff.outcomes] == ["accepted"]
    assert len(llm.calls) == 1


async def test_compile_pack_adds_the_local_layers_generators_the_pack_lacks() -> None:
    # hybrid mode: generators learned inline sit in the store until a compile proposes them.
    store = open_store(":memory:")
    await store.put_generator(spec_record("gen-inline", r"(\d+) secs"))
    await store.put_generator(spec_record("gen-reviewed", r"(\d+) s\b"))
    diff = await compile_pack(
        compiler(FakeJev(), FakeLLM([]), store), [pack_spec("gen-reviewed", r"(\d+) s\b")]
    )
    assert diff.outcomes == []
    assert [s.id for s in diff.generators] == ["gen-inline"]


async def test_compile_pack_takes_only_wanted_examples_of_candidate_fields() -> None:
    store = open_store(":memory:")
    await store.add_example(example(eid="low", p=0.5))
    await store.add_example(example(eid="enum", field="Car.gearbox", value="manual"))
    await store.add_example(example(eid="human", p=None, source="human"))
    diff = await compile_pack(
        compiler(FakeJev().choice(None, pick("9.1")), FakeLLM([DRAFT]), store, learn_threshold=0.9)
    )
    assert [(o.example_id, o.status) for o in diff.outcomes] == [("human", "accepted")]


async def test_compile_pack_learns_from_a_nested_models_examples() -> None:
    store = open_store(":memory:")
    await store.add_example(example(field=CHILD_FIELD))
    lrn = GeneratorLearner(
        [RANGE],
        FakeLLM([DRAFT]),
        FakeJev().choice(None, pick("9.1")).client(),
        generators=LearnedGenerators(store, persist=False),
        base=GeneratorRegistry(),
    )
    diff = await compile_pack(lrn)
    assert [(o.field, o.status) for o in diff.outcomes] == [(CHILD_FIELD, "accepted")]
    assert [g.field for g in diff.generators] == [CHILD_FIELD]


async def test_compile_pack_reports_rejected_examples_and_writes_no_generator() -> None:
    store = open_store(":memory:")
    await store.add_example(example())
    llm = FakeLLM([{"regex": r"(\d+) mph", "group": 1, "normalise": ["parse_number"]}])
    diff = await compile_pack(compiler(FakeJev(), llm, store))
    assert [o.status for o in diff.outcomes] == ["missed_trigger"]
    assert diff.generators == []


async def test_compile_pack_needs_a_store_it_wont_publish_to() -> None:
    with pytest.raises(ValueError, match="persist=False"):
        await compile_pack(learner(FakeJev(), FakeLLM([])))
    no_store = GeneratorLearner(
        [SPEC], FakeLLM([]), FakeJev().client(), generators=LearnedGenerators(persist=False)
    )
    with pytest.raises(ValueError, match="persist=False"):
        await compile_pack(no_store)


async def test_the_extractor_compiles_a_stores_examples() -> None:
    store = open_store(":memory:")
    await store.add_example(example())
    fake = FakeJev().choice(None, pick("9.1"))
    async with Extractor(
        [Car], jev=fake.client(), pipeline=pipeline(), store=store, generator_llm=FakeLLM([DRAFT])
    ) as extractor:
        diff = await extractor.compile_pack()
        result = await extractor.extract(Document.from_bytes(b"<p/>"))
    assert [o.status for o in diff.outcomes] == ["accepted"]
    assert await store.generators() == []
    # Documents don't use unreviewed generators: there's nothing to find the value with.
    assert result.meta.generator_snapshot == 0
    assert result.records == []


async def test_the_extractor_compiles_only_with_a_generator_llm_and_a_store() -> None:
    async with Extractor([Car], jev=FakeJev().client(), store=":memory:") as extractor:
        with pytest.raises(ValueError, match="needs a generator_llm"):
            await extractor.compile_pack()
    async with Extractor([Car], jev=FakeJev().client(), generator_llm=FakeLLM([])) as extractor:
        await extractor.learner()  # opens the extractor's own in-memory store
        with pytest.raises(ValueError, match="pass store="):
            await extractor.compile_pack()


# --- pack diffs ----------------------------------------------------------------------------


def test_a_pack_diff_writes_one_yaml_file_per_generator(tmp_path: Path) -> None:
    specs = [pack_spec("gen-a"), pack_spec("gen-b", r"(\d+) secs")]
    out = tmp_path / "diff"
    paths = PackDiff(generators=specs, outcomes=[]).write(out)
    assert paths == [out / "generators" / "gen-a.yaml", out / "generators" / "gen-b.yaml"]
    assert pack_generators(out) == specs


def test_a_pack_diff_goes_only_into_a_new_or_empty_directory(tmp_path: Path) -> None:
    diff = PackDiff(generators=[pack_spec("gen-a")], outcomes=[])
    empty = tmp_path / "empty"
    empty.mkdir()
    diff.write(empty)
    with pytest.raises(FileExistsError, match="isn't an empty directory"):
        diff.write(empty)
    afile = tmp_path / "file"
    afile.write_text("")
    with pytest.raises(FileExistsError):
        diff.write(afile)


def test_pack_generators_reads_a_packs_generators(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such pack directory"):
        pack_generators(tmp_path / "missing")
    assert pack_generators(tmp_path) == []  # no generators directory
    folder = tmp_path / "generators"
    folder.mkdir()
    (folder / "b.yaml").write_text(pack_spec("gen-b").to_yaml())
    (folder / "a.yaml").write_text(pack_spec("gen-a").to_yaml())
    (folder / "notes.txt").write_text("not a spec")
    assert [s.id for s in pack_generators(tmp_path)] == ["gen-a", "gen-b"]

    (folder / "c.yaml").write_text(pack_spec("gen-a").to_yaml())
    with pytest.raises(InvalidGeneratorError, match=r"c\.yaml: another file already has id"):
        pack_generators(tmp_path)
    (folder / "c.yaml").write_text("id: bad id\n")
    with pytest.raises(InvalidGeneratorError, match=r"c\.yaml: "):
        pack_generators(tmp_path)

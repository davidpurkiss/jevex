import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

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
    LearnedGenerators,
    LearnStage,
    PackDiff,
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
from jevex.learn import NORMALISERS, PROMPT, GeneratorDraft, LearnOutcome, draft_spec
from jevex.llm import LLMError
from jevex.normalise import BUILTIN_NORMALISERS, FunctionNormaliser, NormaliseStage, strip
from jevex.select import CandidateStage, JevCandidateSelector, SelectStage
from jevex.statements import NormaliserStep
from jevex.store import Store, open_store
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


def example(
    text: str = TEXT,
    value: Any = 9.1,
    *,
    eid: str = "ex-1",
    field: str = FIELD,
    p: float | None = 0.95,
    source: Literal["llm", "human"] = "llm",
    evidence: tuple[int, int] | None = None,
    context: dict[str, Any] | None = None,
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
        async def structured(self, prompt: str, schema: type[Any]) -> Any:
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
    store = open_store(":memory:")
    llm = FakeLLM([DRAFT])
    ledger = RunLedger(RunBudget(max_spend=0.0, period="run"), store)
    outcome = await learned(FakeJev(), llm, example(), store=store, ledger=ledger)
    assert outcome.status == "budget"
    assert llm.calls == []


async def test_the_run_jev_cap_stops_testing() -> None:
    store = open_store(":memory:")
    ledger = RunLedger(RunBudget(max_jev_spend=0.0, period="run"), store)
    fake = FakeJev(strict=True)
    outcome = await learned(fake, FakeLLM([DRAFT]), example(), store=store, ledger=ledger)
    assert outcome.status == "budget"
    assert fake.calls == []


async def test_the_learners_jev_spend_goes_in_the_run_ledger() -> None:
    store = open_store(":memory:")
    ledger = RunLedger(RunBudget(max_spend=5.0, period="run"), store)
    fake = FakeJev().choice(None, pick("9.1"))
    llm = FakeLLM([DRAFT], price=(1.0, 1.0))
    outcome = await learned(fake, llm, example(), store=store, ledger=ledger)
    assert outcome.status == "accepted"
    assert await store.spend(kind="jev", run_id=ledger.run_id) > 0
    assert await store.spend(kind="llm", run_id=ledger.run_id) > 0


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


@dataclass
class Categorised:
    """Leaves one statement categorised as the 0-62 time, as the stages before would."""

    name: str = "categorise"

    async def run(self, ctx: Context) -> None:
        statement = Statement(id="s1", text=TEXT, kind="sentence", component_id="c1", location=LOC)
        done = ctx_with(statement)
        ctx.parsed = done.parsed
        ctx.schemas["Car"].scopes = done.schemas["Car"].scopes
        ctx.schemas["Car"].categories = done.schemas["Car"].categories


VERIFY = "The statement states that the 0-62 mph time (s) is 9.1."


def pipeline() -> Pipeline:
    return Pipeline(
        [
            Categorised(),
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

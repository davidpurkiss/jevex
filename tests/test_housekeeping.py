import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from jevex import (
    Candidate,
    Context,
    Document,
    DomLocation,
    DuplicateGenerator,
    Extractor,
    Field,
    FieldMeta,
    GeneratorRecord,
    GeneratorRegistry,
    GeneratorSpec,
    Housekeeper,
    LearnedGenerators,
    LearnStage,
    Pipeline,
    SchemaSpec,
    Span,
    Statement,
    StoreError,
    VerifiedExample,
    generator_use,
)
from jevex.entities import EntityScope
from jevex.housekeeping import PRUNE_AFTER, QUARANTINE_AFTER
from jevex.interfaces import ParsedDocument, Scope, Selection
from jevex.jev import Choice, ChoiceAnswer
from jevex.layout import Component
from jevex.learn import GeneratorSnapshot
from jevex.normalise import NormaliseStage
from jevex.select import CandidateStage, SelectStage
from jevex.store import Store, open_store
from jevex.testing import FakeJev

LOC = DomLocation(dom_path="/p")
TEXT = "62 mph takes 9.1 seconds"
FIELD = "Car.zero_to_62_s"


class Car(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    trims: list[str] = Field(default_factory=list, description="Trim names")
    lap_times_s: list[float] = Field(default_factory=list, description="Lap times", unit="s")


SPEC = SchemaSpec.from_model(Car)


def spec(
    gen_id: str,
    regex: str = r"(\d+(?:\.\d+)?) seconds",
    *,
    field: str = FIELD,
    normalise: list[Any] | None = None,
    locale: str | None = None,
) -> GeneratorSpec:
    data: dict[str, Any] = {
        "id": gen_id,
        "field": field,
        "match": {"regex": regex, "group": 1},
        "normalise": normalise or [],
    }
    if locale:
        data["scope"] = {"locale": locale}
    return GeneratorSpec.parse(data)


def record(s: GeneratorSpec) -> GeneratorRecord:
    return GeneratorRecord(id=s.id, field=s.field, spec=s.to_data())


def example(eid: str, text: str = TEXT, value: Any = 9.1, field: str = FIELD) -> VerifiedExample:
    return VerifiedExample(id=eid, field=field, statement=text, value=value, probability=0.95)


def candidate(gen_id: str, start: int = 13, end: int = 16, text: str = TEXT) -> Candidate:
    return Candidate(span=Span(start=start, end=end), raw=text[start:end], generator_id=gen_id)


def meta(gen_id: str | None, value: Any = 9.1, method: Any = "generator") -> FieldMeta:
    return FieldMeta(value=value, confidence=0.9, method=method, generator_id=gen_id)


_opened: list[Store] = []


def new_store() -> Store:
    store = open_store(":memory:")
    _opened.append(store)
    return store


@pytest.fixture(autouse=True)
async def close_stores() -> AsyncIterator[None]:
    yield
    while _opened:
        await _opened.pop().aclose()


def bare_ctx() -> Context:
    return Context.create(Document.from_bytes(b"<p/>"), [SPEC], FakeJev().client())


# --- what one document did ----------------------------------------------------------------


def test_generator_use_reads_ran_hits_and_wins() -> None:
    ctx = bare_ctx()
    run = ctx.schemas["Car"]
    ctx.generators_ran.update({"winner", "hitter", "idle"})
    run.candidates[("s1", "zero_to_62_s")] = [candidate("winner"), candidate("hitter", 0, 2)]
    run.set_field("doc", "zero_to_62_s", meta("winner"))
    use = generator_use(ctx)
    assert use.ran == {"winner", "hitter", "idle"}
    assert use.hits == {"winner", "hitter"}
    assert use.wins == {"winner"}


def test_values_that_didnt_stand_arent_wins() -> None:
    ctx = bare_ctx()
    run = ctx.schemas["Car"]
    run.candidates[("s1", "zero_to_62_s")] = [candidate("g")]
    run.set_field("llm", "zero_to_62_s", meta("g", method="llm"))
    run.set_field("unfit", "zero_to_62_s", meta("g", value=None))
    assert generator_use(ctx).wins == frozenset()


def test_a_custom_stages_candidates_count_as_ran() -> None:
    ctx = bare_ctx()
    ctx.schemas["Car"].candidates[("s1", "zero_to_62_s")] = [candidate("custom")]
    assert generator_use(ctx).ran == {"custom"}


async def test_every_generator_a_list_value_took_a_pick_from_wins() -> None:
    ctx = bare_ctx()
    run = ctx.schemas["Car"]
    text = "laps of 9.1, 9.4 and abc"
    picks = [
        candidate("first", 8, 11, text),
        candidate("second", 13, 16, text),
        candidate("unfit", 21, 24, text),  # "abc" doesn't normalise to a float
    ]
    run.candidates[("s1", "lap_times_s")] = picks
    run.selections[("doc", "lap_times_s", "s1")] = Selection(
        candidate=picks[0], confidence=0.9, accepted=picks
    )
    await NormaliseStage().run(ctx)
    assert run.fields["doc"]["lap_times_s"].value == [9.1, 9.4]
    assert run.value_generators[("doc", "lap_times_s")] == {"first", "second"}
    assert generator_use(ctx).wins == {"first", "second"}


async def test_a_scalar_wins_only_for_the_pick_it_took() -> None:
    ctx = bare_ctx()
    run = ctx.schemas["Car"]
    best, other = candidate("best"), candidate("other", 0, 2)
    run.candidates[("s1", "zero_to_62_s")] = [best, other]
    run.selections[("doc", "zero_to_62_s", "s1")] = Selection(
        candidate=best, confidence=0.9, accepted=[best, other]
    )
    await NormaliseStage().run(ctx)
    assert generator_use(ctx).wins == {"best"}


# --- stats and pruning --------------------------------------------------------------------


def statement() -> Statement:
    return Statement(id="s1", text=TEXT, kind="sentence", component_id="c1", location=LOC)


@dataclass
class Categorised:
    """Leaves one statement categorised as the 0-62 time, as the stages before would."""

    name: str = "categorise"

    async def run(self, ctx: Context) -> None:
        s = statement()
        ctx.parsed = ParsedDocument(
            document=ctx.document,
            root=Component(id="root", type="section", location=LOC),
            statements={s.id: s},
        )
        run = ctx.schemas["Car"]
        run.scopes = [EntityScope(label="doc", component_ids=["c1"])]
        run.categories[s.id] = ChoiceAnswer(
            choice="zero_to_62_s", confidence=0.9, probabilities={"zero_to_62_s": 0.9}
        )


def pick(value: str) -> Any:
    def answer(q: Choice) -> str:
        return value if value in q.options else "none"

    return answer


def pipeline(base: GeneratorRegistry | None = None) -> Pipeline:
    return Pipeline(
        [
            Categorised(),
            CandidateStage(registry=base if base is not None else GeneratorRegistry()),
            SelectStage(),
            NormaliseStage(),
            LearnStage(),
        ]
    )


async def run_document(
    keeper: Housekeeper, learned: LearnedGenerators, base: GeneratorRegistry | None = None
) -> Context:
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SPEC], FakeJev().choice(None, pick("9.1")).client()
    )
    ctx.generators = learned.current
    ctx.housekeeper = keeper
    await pipeline(base).run(ctx)
    return ctx


async def setup(store: Store, *specs: GeneratorSpec) -> LearnedGenerators:
    for s in specs:
        await store.put_generator(record(s))
    learned = LearnedGenerators(store)
    await learned.load()
    return learned


async def test_a_document_adds_to_each_generators_stats() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"), spec("gen-mph", r"(\d+) mph"))
    keeper = Housekeeper(store, learned)
    await run_document(keeper, learned)
    await run_document(keeper, learned)
    time_stats = await store.generator_stats("gen-time")
    mph_stats = await store.generator_stats("gen-mph")
    assert (time_stats.documents, time_stats.hits, time_stats.wins) == (2, 2, 2)
    assert (mph_stats.documents, mph_stats.hits, mph_stats.wins) == (2, 2, 0)
    assert mph_stats.hit_rate == 1.0
    assert mph_stats.win_rate == 0.0


async def test_a_generator_that_ran_without_a_candidate_counts_a_document_only() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"), spec("gen-km", r"(\d+) km"))
    await run_document(Housekeeper(store, learned), learned)
    stats = await store.generator_stats("gen-km")
    assert (stats.documents, stats.hits, stats.wins) == (1, 0, 0)
    assert stats.win_rate is None


async def test_generators_scoped_elsewhere_arent_counted() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"), spec("gen-de", r"(\d+) km", locale="de"))
    await run_document(Housekeeper(store, learned), learned)
    assert (await store.generator_stats("gen-de")).documents == 0


async def test_a_learned_generator_with_no_wins_after_prune_after_documents_is_disabled() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"), spec("gen-mph", r"(\d+) mph"))
    keeper = Housekeeper(store, learned, prune_after=2)
    first = await run_document(keeper, learned)
    assert keeper.pruned == []
    assert first.events == []
    second = await run_document(keeper, learned)
    assert keeper.pruned == ["gen-mph"]
    [event] = second.events
    assert (event.stage, event.kind) == ("learn", "generator_pruned")
    assert event.message == "generator gen-mph won nothing in 2 documents"
    assert event.data == {"generator_id": "gen-mph", "documents": 2, "hits": 2}
    # Disabled, not deleted, and out of the snapshot later documents take.
    assert await store.disabled_generator_ids() == {"gen-mph"}
    stored = await store.generators(include_disabled=True)
    assert [r.id for r in stored] == ["gen-time", "gen-mph"]
    assert learned.current.registry.ids == ["gen-time"]
    assert learned.current.version == 1
    # The document that pruned it keeps the snapshot it started with.
    assert second.generators is not None
    assert "gen-mph" in second.generators.registry
    third = await run_document(keeper, learned)
    assert "gen-mph" not in third.generators_ran
    assert third.events == []


async def test_documents_finishing_together_prune_a_generator_once() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"), spec("gen-mph", r"(\d+) mph"))
    keeper = Housekeeper(store, learned, prune_after=1)
    docs = await asyncio.gather(*(run_document(keeper, learned) for _ in range(4)))
    assert keeper.pruned == ["gen-mph"]
    assert sum(len(ctx.events) for ctx in docs) == 1


async def test_a_loser_with_earlier_wins_isnt_pruned() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"), spec("gen-mph", r"(\d+) mph"))
    await store.record_generator_stats("gen-mph", documents=5, hits=5, wins=1)
    keeper = Housekeeper(store, learned, prune_after=2)
    await run_document(keeper, learned)
    assert keeper.pruned == []
    assert await store.disabled_generator_ids() == set()


async def test_built_in_generators_are_counted_but_never_pruned() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"))
    base = GeneratorRegistry([spec("stage-mph", r"(\d+) mph").to_generator()])
    keeper = Housekeeper(store, learned, prune_after=1)
    await run_document(keeper, learned, base)
    assert (await store.generator_stats("stage-mph")).documents == 1
    assert keeper.pruned == []


async def test_prune_after_none_never_prunes() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"), spec("gen-mph", r"(\d+) mph"))
    keeper = Housekeeper(store, learned, prune_after=None)
    for _ in range(3):
        await run_document(keeper, learned)
    assert keeper.pruned == []
    assert (await store.generator_stats("gen-mph")).documents == 3


async def test_without_a_snapshot_nothing_is_pruned() -> None:
    store = new_store()
    ctx = bare_ctx()
    ctx.generators_ran.add("gen-x")
    keeper = Housekeeper(store, prune_after=1)
    assert await keeper.record(ctx) == []
    assert (await store.generator_stats("gen-x")).documents == 1


# --- quarantine ----------------------------------------------------------------------


@dataclass
class Crashes:
    """A generator that raises on every statement."""

    id: str
    scope: Scope = field(default_factory=Scope)

    def generate(self, statement: Statement) -> list[Candidate]:
        raise RuntimeError("bad backreference")


async def run_crashing(
    keeper: Housekeeper, learned: LearnedGenerators, base: GeneratorRegistry | None = None
) -> Context:
    """A document whose snapshot has the stored ``gen-bad`` as a generator that raises."""
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SPEC], FakeJev().choice(None, pick("9.1")).client()
    )
    registry = learned.current.registry
    if "gen-bad" in registry:
        registry = registry.with_generator(Crashes("gen-bad"))
    ctx.generators = GeneratorSnapshot(learned.current.version, registry)
    ctx.housekeeper = keeper
    await pipeline(base).run(ctx)
    return ctx


async def test_a_learned_generator_is_quarantined_after_its_third_failure() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"), spec("gen-bad"))
    keeper = Housekeeper(store, learned, prune_after=None)
    for n in (1, 2):
        ctx = await run_crashing(keeper, learned)
        assert (await store.generator_stats("gen-bad")).failures == n
        assert keeper.quarantined == []
        assert [e.kind for e in ctx.errors.errors] == ["generator"]
    third = await run_crashing(keeper, learned)
    assert keeper.quarantined == ["gen-bad"]
    [event] = third.events
    assert (event.stage, event.kind) == ("learn", "generator_quarantined")
    assert event.message == "generator gen-bad failed 3 times and was disabled"
    assert event.data == {"generator_id": "gen-bad", "failures": QUARANTINE_AFTER}
    # Disabled and kept for review; the other generator still covers the field.
    assert await store.disabled_generator_ids() == {"gen-bad"}
    assert [r.id for r in await store.generators(include_disabled=True)] == ["gen-time", "gen-bad"]
    assert learned.current.registry.ids == ["gen-time"]
    assert third.schemas["Car"].fields["doc"]["zero_to_62_s"].value == 9.1
    fourth = await run_crashing(keeper, learned)
    assert "gen-bad" not in fourth.generators_ran
    assert (fourth.events, fourth.errors.errors) == ([], [])


async def test_failures_on_one_document_count_each_statement() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-bad"))
    keeper = Housekeeper(store, learned)
    ctx = bare_ctx()
    ctx.generators = GeneratorSnapshot(1, learned.current.registry)
    for _ in range(QUARANTINE_AFTER):
        ctx.part_failed("candidates", "generator", "gen-bad", RuntimeError("x"))
    await keeper.record(ctx)
    assert keeper.quarantined == ["gen-bad"]


async def test_built_in_generators_that_fail_are_counted_but_never_quarantined() -> None:
    store = new_store()
    learned = await setup(store, spec("gen-time"))
    keeper = Housekeeper(store, learned)
    base = GeneratorRegistry([Crashes("stage-bad")])
    docs = [await run_crashing(keeper, learned, base) for _ in range(QUARANTINE_AFTER + 1)]
    assert (await store.generator_stats("stage-bad")).failures == QUARANTINE_AFTER + 1
    assert keeper.quarantined == []
    assert all(ctx.events == [] for ctx in docs)


def test_prune_after_must_be_positive() -> None:
    with pytest.raises(ValueError, match="prune_after must be at least 1, got 0"):
        Housekeeper(new_store(), prune_after=0)
    assert Housekeeper(new_store()).prune_after == PRUNE_AFTER


# --- dedup --------------------------------------------------------------------------------


OTHER = "it does 0-62 in 7.4 seconds"


async def dedupe_store(*specs: GeneratorSpec, examples: tuple[str, ...] = (TEXT, OTHER)) -> Store:
    store = new_store()
    for s in specs:
        await store.put_generator(record(s))
    for i, text in enumerate(examples):
        await store.add_example(example(f"ex-{i}", text))
    return store


async def test_dedupe_disables_the_newer_of_two_generators_with_the_same_candidates() -> None:
    store = await dedupe_store(
        spec("gen-old", r"(\d+(?:\.\d+)?) seconds"),
        spec("gen-new", r"(\d+\.\d+)\s*seconds"),
        spec("gen-other", r"(\d+) mph"),
    )
    learned = LearnedGenerators(store)
    await learned.load()
    found = await Housekeeper(store, learned).dedupe()
    assert found == [DuplicateGenerator(id="gen-new", kept="gen-old", field=FIELD, examples=2)]
    assert await store.disabled_generator_ids() == {"gen-new"}
    assert learned.current.registry.ids == ["gen-old", "gen-other"]


async def test_dedupe_keeps_the_generator_with_more_wins() -> None:
    store = await dedupe_store(spec("gen-old"), spec("gen-new", r"(\d+\.\d+)\s*seconds"))
    await store.record_generator_stats("gen-new", documents=3, hits=3, wins=2)
    await store.record_generator_stats("gen-old", documents=3, hits=3, wins=1)
    [found] = await Housekeeper(store).dedupe()
    assert (found.id, found.kept) == ("gen-old", "gen-new")


async def test_generators_that_differ_on_one_example_arent_duplicates() -> None:
    # Both find 9.1; only the first finds "7 seconds".
    store = await dedupe_store(
        spec("gen-any"),
        spec("gen-decimal", r"(\d+\.\d+) seconds"),
        examples=(TEXT, "in 7 seconds"),
    )
    assert await Housekeeper(store).dedupe() == []


@pytest.mark.parametrize(
    "other",
    [
        spec("gen-b", r"(\d+\.\d+)\s*seconds", normalise=["parse_number"]),
        spec("gen-b", r"(\d+\.\d+)\s*seconds", locale="en-GB"),
        spec("gen-b", r"(\d+\.\d+)\s*seconds", field="Car.other"),
    ],
    ids=["normalisers", "scope", "field"],
)
async def test_a_different_normaliser_chain_scope_or_field_isnt_a_duplicate(
    other: GeneratorSpec,
) -> None:
    store = await dedupe_store(spec("gen-a"), other)
    assert await Housekeeper(store).dedupe() == []
    assert await store.disabled_generator_ids() == set()


async def test_scopes_written_differently_for_one_locale_are_the_same_scope() -> None:
    store = await dedupe_store(spec("gen-a", locale="de-DE"))
    # Stored as written before tags were canonical.
    other = record(spec("gen-b", r"(\d+\.\d+)\s*seconds", locale="de-DE"))
    await store.put_generator(
        other.model_copy(update={"spec": {**other.spec, "scope": {"locale": "de_de"}}})
    )
    learned = LearnedGenerators(store)
    await learned.load()
    [found] = await Housekeeper(store, learned).dedupe()
    assert (found.id, found.kept) == ("gen-b", "gen-a")


async def test_generators_that_find_nothing_on_any_example_arent_duplicates() -> None:
    store = await dedupe_store(spec("gen-a", r"(\d+) km"), spec("gen-b", r"(\d+)\s*km"))
    assert await Housekeeper(store).dedupe() == []
    store = await dedupe_store(spec("gen-a"), spec("gen-b", r"(\d+\.\d+)\s*seconds"), examples=())
    assert await Housekeeper(store).dedupe() == []


async def test_dedupe_skips_disabled_generators() -> None:
    store = await dedupe_store(spec("gen-a"), spec("gen-b", r"(\d+\.\d+)\s*seconds"))
    await store.set_generator_enabled("gen-a", False)
    assert await Housekeeper(store).dedupe() == []


async def test_dedupe_raises_for_an_invalid_stored_spec() -> None:
    store = await dedupe_store(spec("gen-a"))
    bad = {"id": "gen-bad", "field": FIELD, "match": {"regex": "(", "group": 0}}
    await store.put_generator(GeneratorRecord(id="gen-bad", field=FIELD, spec=bad))
    with pytest.raises(StoreError, match="gen-bad"):
        await Housekeeper(store).dedupe()


# --- the extractor ------------------------------------------------------------------------


async def test_the_extractor_prunes_and_later_documents_dont_run_the_pruned() -> None:
    store = new_store()
    await setup(store, spec("gen-time"), spec("gen-mph", r"(\d+) mph"))
    fake = FakeJev().choice(None, pick("9.1"))
    doc = Document.from_bytes(b"<p/>")
    async with Extractor(
        [Car], jev=fake.client(), pipeline=pipeline(), store=store, prune_after=2
    ) as ex:
        await ex.extract(doc)
        second = await ex.extract(doc)
        third = await ex.extract(doc)
        keeper = await ex.housekeeper()
    assert [e.kind for e in second.meta.events] == ["generator_pruned"]
    assert second.meta.generator_snapshot == 0
    assert third.meta.generator_snapshot == 1
    assert keeper is not None
    assert keeper.pruned == ["gen-mph"]
    assert (await store.generator_stats("gen-mph")).documents == 2
    assert (await store.generator_stats("gen-time")).wins == 3


async def test_the_extractor_dedupes_its_stores_generators() -> None:
    store = await dedupe_store(spec("gen-a"), spec("gen-b", r"(\d+\.\d+)\s*seconds"))
    async with Extractor([Car], jev=FakeJev().client(), store=store) as ex:
        [found] = await ex.dedupe_generators()
        learned = await ex.learned_generators()
    assert found.id == "gen-b"
    assert learned is not None
    assert learned.current.registry.ids == ["gen-a"]


async def test_without_a_store_there_is_no_housekeeping() -> None:
    async with Extractor([Car], jev=FakeJev().client(), pipeline=pipeline()) as ex:
        assert await ex.housekeeper() is None
        with pytest.raises(ValueError, match="dedupe_generators needs a store"):
            await ex.dedupe_generators()


def test_the_extractor_checks_prune_after() -> None:
    with pytest.raises(ValueError, match="prune_after must be at least 1, got 0"):
        Extractor([Car], prune_after=0)

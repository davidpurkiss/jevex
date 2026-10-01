import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from jevex import (
    Context,
    Document,
    DomLocation,
    ExtractionResult,
    Extractor,
    Field,
    KeyPathMapper,
    Pipeline,
    SchemaConfig,
    SchemaSpec,
    StructuredMode,
    StructuredStage,
    flatten,
)
from jevex.extractor import default_pipeline
from jevex.interfaces import StructuredExtractor
from jevex.jev import Choice, JevClient
from jevex.resolve import SINGLE_ENTITY_LABEL
from jevex.results import FieldMeta
from jevex.store import KeyMapping, SQLiteStore
from jevex.structured import EmbeddedDataReader, StructuredBlob
from jevex.testing import FakeJev
from jevex.testsite import VehicleSpec, generate, render

LOC = DomLocation(dom_path="/html/head/script")


def blob(data: object, types: list[str] | None = None) -> StructuredBlob:
    return StructuredBlob(source="json_ld", data=data, types=types or [], location=LOC)


class Car(BaseModel):
    """A car for sale."""

    model: str = Field(description="Model name")
    price: Decimal = Field(description="Price", unit="GBP")
    engine_size_cc: int = Field(description="Engine size", unit="cc")
    fuel: Literal["petrol", "diesel", "ev"] = Field(description="Fuel type")
    colours: list[str] = Field(default_factory=list, description="Colours offered")


def page_blob(data: object) -> StructuredBlob:
    """The blob as the reader finds it on a page (with its declared types)."""
    [found] = EmbeddedDataReader().read(page(data)).blobs
    return found


def page(data: object) -> Document:
    script = f'<script type="application/ld+json">{json.dumps(data)}</script>'
    return Document.from_bytes(
        f"<html><head>{script}</head><body><p>x</p></body></html>".encode(),
        url="https://cars.test/golf",
    )


CAR = {
    "@context": "https://schema.org",
    "@type": "Car",
    "model": "Golf",
    "vehicleEngine": {"engineDisplacement": "1,498 cc"},
    "fuelType": "Petrol",
    "offers": [{"@type": "Offer", "price": 24995}, {"@type": "Offer", "price": 26995}],
    "color": ["Red", "Moonstone Grey"],
    "description": "",
}

MAPPING = {
    "model": "model",
    "offers[].price": "price",
    "vehicleEngine.engineDisplacement": "engine_size_cc",
    "fuelType": "fuel",
    "color[]": "colours",
}


def mapping_jev(mapping: dict[str, str] = MAPPING, confidence: float = 0.9) -> FakeJev:
    fake = FakeJev()
    for path, field_name in mapping.items():
        fake.choice(f'key path "{path}"', _pick(field_name), confidence=confidence)
    return fake


def _pick(name: str) -> Callable[[Choice], str]:
    """The field if this schema's question offers it, else "none"."""
    return lambda q: name if name in q.options else "none"


# --- flattening and fingerprints -----------------------------------------------------


def test_flatten_gives_key_paths_with_indices_and_collapsed_shapes() -> None:
    flat = flatten(blob(CAR))
    assert [(leaf.path, leaf.shape, leaf.value) for leaf in flat.leaves] == [
        ("@type", "@type", "Car"),
        ("model", "model", "Golf"),
        ("vehicleEngine.engineDisplacement", "vehicleEngine.engineDisplacement", "1,498 cc"),
        ("fuelType", "fuelType", "Petrol"),
        ("offers[0].@type", "offers[].@type", "Offer"),
        ("offers[0].price", "offers[].price", 24995),
        ("offers[1].@type", "offers[].@type", "Offer"),
        ("offers[1].price", "offers[].price", 26995),
        ("color[0]", "color[]", "Red"),
        ("color[1]", "color[]", "Moonstone Grey"),
    ]
    assert flat.entities == ("offers[]",)  # arrays of objects are entity candidates


def test_a_bare_scalar_blob_is_one_leaf() -> None:
    assert [(leaf.path, leaf.value) for leaf in flatten(blob("hello")).leaves] == [("$", "hello")]
    assert flatten(blob({"a": None, "b": [], "c": {}})).leaves == ()


def test_fingerprint_depends_on_shape_not_values_or_counts() -> None:
    one = flatten(blob(CAR)).fingerprint
    other_values = {**CAR, "model": "Polo", "offers": [{"@type": "Offer", "price": 1}]}
    assert flatten(blob(other_values)).fingerprint == one
    assert flatten(blob({**CAR, "mileage": 12000})).fingerprint != one
    assert len(one) == 16


# --- mapping -------------------------------------------------------------------------


async def extract(mapper: KeyPathMapper, fake: FakeJev, data: object = CAR) -> dict[str, FieldMeta]:
    result = await mapper.extract(page(data), [SchemaSpec.from_model(Car)], fake.client())
    return result.fields["Car"]


async def test_a_miss_asks_one_batched_request_and_maps_the_values() -> None:
    fake = mapping_jev()
    fields = await extract(KeyPathMapper(), fake)
    assert {name: meta.value for name, meta in fields.items()} == {
        "model": "Golf",
        "price": Decimal("24995"),  # first offer
        "engine_size_cc": 1498,  # "1,498 cc" read like text
        "fuel": "petrol",  # "Petrol", matched ignoring case
        "colours": ["Red", "Moonstone Grey"],  # a list field takes every value
    }
    [call] = fake.calls
    assert set(call.questions) == {f"Car:{i}" for i in range(7)}  # one per collapsed path
    state = call.state
    assert isinstance(state, dict)
    assert "offers[1].price: 26995" in state["data"]
    assert state["type"] == "Car"
    question = call.questions["Car:1"]
    assert isinstance(question, Choice)
    assert question.instructions == "Which detail does the key path \"model\" hold (e.g. 'Golf')?"
    assert question == SchemaSpec.from_model(Car).key_path_question("model", "Golf")
    assert "none" in question.options
    meta = fields["price"]
    assert meta.method == "structured"
    assert meta.source is not None
    assert meta.source.statement == "offers[0].price: 24995"


async def test_a_hit_is_a_pure_lookup(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "jevex.db")
    mapper = KeyPathMapper(store=store)
    await extract(mapper, mapping_jev())
    # Same template, other values, a new mapper: every path is known, so no Jev call.
    fresh = FakeJev(strict=True)
    other = {**CAR, "model": "Polo", "offers": [{"@type": "Offer", "price": 18000}]}
    fields = await extract(KeyPathMapper(store=store), fresh, other)
    assert fresh.calls == []
    assert fields["model"].value == "Polo"
    assert fields["price"].value == Decimal("18000")
    stored = await store.key_mappings(flatten(page_blob(CAR)).fingerprint, schema="Car")
    assert {m.path: m.field for m in stored}["@type"] is None  # "none" is remembered too
    await store.aclose()


async def test_without_a_store_the_mapper_remembers_for_its_lifetime() -> None:
    mapper = KeyPathMapper()
    await extract(mapper, mapping_jev())
    fresh = FakeJev(strict=True)
    await extract(mapper, fresh)
    assert fresh.calls == []


async def test_unsure_answers_arent_stored_so_the_path_is_asked_again() -> None:
    mapper = KeyPathMapper()
    fields = await extract(mapper, mapping_jev(confidence=0.4))
    assert fields == {}
    again = mapping_jev()
    await extract(mapper, again)
    assert len(again.calls) == 1


UNSURE_PATHS = [
    "model",
    "vehicleEngine.engineDisplacement",
    "fuelType",
    "offers[].price",
    "color[]",
]


async def test_after_three_unsure_answers_a_path_is_stored_as_none() -> None:
    mapper = KeyPathMapper()
    car = SchemaSpec.from_model(Car)
    # The first page asks about every path; the "@type"s get a confident "none", so the
    # second asks only about the unsure ones.
    for asked in (7, 5):
        fake = mapping_jev(confidence=0.4)
        result = await mapper.extract(page(CAR), [car], fake.client())
        [call] = fake.calls
        assert len(call.questions) == asked
        assert not [e for e in result.events if e[0] == "structured_paths_unsure"]
    third = mapping_jev(confidence=0.4)
    result = await mapper.extract(page(CAR), [car], third.client())
    assert len(third.calls) == 1
    assert result.events == [
        (
            "structured_paths_unsure",
            f"Car: stored 5 key path(s) as none after 3 unsure answers: {', '.join(UNSURE_PATHS)}",
        )
    ]
    # Steady state: every path is known, so no call (and, being "none", no values).
    fourth = FakeJev(strict=True)
    assert await extract(mapper, fourth) == {}
    assert fourth.calls == []


async def test_unsure_counts_persist_in_the_store_and_a_confident_answer_replaces_none(
    store: SQLiteStore,
) -> None:
    fp = flatten(page_blob(CAR)).fingerprint
    for _ in range(3):  # a fresh mapper each page: the count lives in the store
        await extract(KeyPathMapper(store=store), mapping_jev(confidence=0.4))
    stored = {m.path: m for m in await store.key_mappings(fp, schema="Car")}
    assert stored["model"].field is None
    assert stored["model"].unsure
    assert not stored["@type"].unsure  # a confident "none" isn't marked
    fresh = FakeJev(strict=True)
    await extract(KeyPathMapper(store=store), fresh)
    assert fresh.calls == []
    # A review or re-learn puts a confident mapping in its place.
    await store.put_key_mapping(
        KeyMapping(fingerprint=fp, schema="Car", path="model", field="model")
    )
    fields = await extract(KeyPathMapper(store=store), FakeJev(strict=True))
    assert fields["model"].value == "Golf"


async def test_a_confident_answer_after_unsure_ones_is_stored_and_resets_the_count(
    store: SQLiteStore,
) -> None:
    fp = flatten(page_blob(CAR)).fingerprint
    for _ in range(2):
        await extract(KeyPathMapper(store=store), mapping_jev(confidence=0.4))
    await extract(KeyPathMapper(store=store), mapping_jev(confidence=0.9))
    stored = {m.path: m for m in await store.key_mappings(fp, schema="Car")}
    assert (stored["model"].field, stored["model"].unsure) == ("model", False)
    assert await store.count_unsure_key_paths(fp, "Car", ["model"]) == {"model": 1}


async def test_unsure_counts_are_per_schema() -> None:
    class Other(BaseModel):
        """Something else on the page."""

        name: str = Field(description="Name")

    mapper = KeyPathMapper(unsure_limit=2)
    car, other = SchemaSpec.from_model(Car), SchemaSpec.from_model(Other)
    await mapper.extract(page(CAR), [car], mapping_jev(confidence=0.4).client())
    # Second page: Car reaches the limit; Other's first unsure answers don't.
    result = await mapper.extract(page(CAR), [car, other], mapping_jev(confidence=0.4).client())
    assert [m.split(":")[0] for k, m in result.events if k == "structured_paths_unsure"] == ["Car"]
    third = mapping_jev(confidence=0.4)
    await mapper.extract(page(CAR), [car, other], third.client())
    [call] = third.calls
    assert {k.split(":")[0] for k in call.questions} == {"Other"}


async def test_an_unsure_limit_of_one_stores_none_at_once() -> None:
    mapper = KeyPathMapper(unsure_limit=1)
    await extract(mapper, mapping_jev(confidence=0.4))
    fresh = FakeJev(strict=True)
    await extract(mapper, fresh)
    assert fresh.calls == []


def test_the_unsure_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="unsure_limit must be at least 1"):
        KeyPathMapper(unsure_limit=0)


async def test_enum_values_that_dont_read_directly_are_asked_of_jev() -> None:
    fake = mapping_jev().choice(
        re.compile("(?i)what is the fuel type"), "ev", confidence=0.8, state="Fully electric"
    )
    fields = await extract(KeyPathMapper(), fake, {**CAR, "fuelType": "Fully electric"})
    assert fields["fuel"].value == "ev"
    assert fields["fuel"].confidence == 0.8
    [asked] = [c for c in fake.calls if "enum" in c.questions]
    assert asked.state == {"statement": "fuelType: Fully electric"}
    assert asked.questions == {"enum": SchemaSpec.from_model(Car).field("fuel").enum_question()}


async def test_a_value_that_wont_normalise_keeps_the_error() -> None:
    fields = await extract(
        KeyPathMapper(), mapping_jev(), {**CAR, "vehicleEngine": {"engineDisplacement": "big"}}
    )
    meta = fields["engine_size_cc"]
    assert not meta.found
    assert meta.error


async def test_too_many_paths_are_capped_with_an_event() -> None:
    wide = {f"k{i}": i for i in range(10)}
    result = await KeyPathMapper(max_paths=4).extract(
        page(wide), [SchemaSpec.from_model(Car)], FakeJev().client()
    )
    assert [kind for kind, _ in result.events] == ["structured_paths_skipped"]


async def test_too_many_blobs_are_capped_with_an_event() -> None:
    scripts = "".join(
        f'<script type="application/ld+json">{json.dumps({"name": f"n{i}"})}</script>'
        for i in range(5)
    )
    doc = Document.from_bytes(f"<html><head>{scripts}</head></html>".encode())
    fake = FakeJev()
    result = await KeyPathMapper(max_blobs=2).extract(
        doc, [SchemaSpec.from_model(Car)], fake.client()
    )
    assert [kind for kind, _ in result.events] == ["structured_blobs_skipped"]
    assert len(fake.calls) == 1  # both kept blobs share a shape, so one ask


def test_key_path_mapper_is_a_structured_extractor() -> None:
    assert isinstance(KeyPathMapper(), StructuredExtractor)


# --- the stage -----------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SQLiteStore]:
    s = SQLiteStore(tmp_path / "jevex.db")
    yield s
    await s.aclose()


async def test_stage_records_values_on_the_default_entity_and_keeps_statements() -> None:
    ctx = Context.create(page(CAR), [SchemaSpec.from_model(Car)], mapping_jev().client())
    await StructuredStage().run(ctx)
    run = ctx.schemas["Car"]
    assert run.fields[SINGLE_ENTITY_LABEL]["model"].value == "Golf"
    assert [s.kind for s in ctx.structured] == ["structured"] * 10
    assert ctx.structured[1].text == "model: Golf"
    assert [e.kind for e in ctx.events] == ["structured_fields", "layout_route_skipped"]


async def test_stage_doesnt_overwrite_a_found_field() -> None:
    ctx = Context.create(page(CAR), [SchemaSpec.from_model(Car)], mapping_jev().client())
    run = ctx.schemas["Car"]
    run.set_field(SINGLE_ENTITY_LABEL, "model", FieldMeta(value="Passat", method="jev"))
    await StructuredStage().run(ctx)
    assert run.fields[SINGLE_ENTITY_LABEL]["model"].value == "Passat"


async def test_a_mapping_from_the_store_can_say_none(store: SQLiteStore) -> None:
    fp = flatten(page_blob(CAR)).fingerprint
    for path in ["@type", "model", "vehicleEngine.engineDisplacement", "fuelType"]:
        await store.put_key_mapping(KeyMapping(fingerprint=fp, schema="Car", path=path, field=None))
    for path, name in [("offers[].price", "price"), ("offers[].@type", None), ("color[]", None)]:
        await store.put_key_mapping(KeyMapping(fingerprint=fp, schema="Car", path=path, field=name))
    fake = FakeJev(strict=True)
    fields = await extract(KeyPathMapper(store=store), fake)
    assert fake.calls == []
    assert list(fields) == ["price"]


class Book(BaseModel):
    """A book."""

    title: str = Field(description="Title")


@dataclass
class Look:
    """Stands in for the layout route: notes which schemas it would work on."""

    seen: list[str] = dc_field(default_factory=list[str])
    name: str = "layout"

    async def run(self, ctx: Context) -> None:
        self.seen.extend(run.name for run in ctx.active)


async def run_mode(
    mode: StructuredMode, mapping: dict[str, str] = MAPPING, *models: type[BaseModel]
) -> tuple[ExtractionResult, Look]:
    look = Look()
    ex = Extractor(
        list(models or (Car,)),
        jev=mapping_jev(mapping).client(),
        pipeline=Pipeline([StructuredStage(mode=mode), look]),
    )
    return await ex.extract(page(CAR)), look


async def test_structured_only_skips_the_layout_route_for_a_schema_the_data_gave_values() -> None:
    result, look = await run_mode("structured_only", MAPPING, Car, Book)
    assert look.seen == ["Book"]  # the embedded data gave Book nothing
    assert result.one(Car).record.model == "Golf"
    assert not result.meta.stopped
    [skipped] = [e for e in result.meta.events if e.kind == "layout_route_skipped"]
    assert skipped.message == (
        "Car: embedded data gave what structured_only needs; no layout route"
    )
    assert skipped.data == {"schema": "Car"}


async def test_structured_only_is_the_default_and_one_value_is_enough() -> None:
    assert StructuredStage().mode == "structured_only"
    assert {s.mode for s in default_pipeline() if isinstance(s, StructuredStage)} == {
        "structured_only"
    }
    result, look = await run_mode("structured_only", {"model": "model"})
    assert look.seen == []
    assert result.one(Car).meta.price.found is False


async def test_without_values_from_embedded_data_the_layout_route_runs() -> None:
    for mode in ("structured_only", "fill_gaps", "merge"):
        _, look = await run_mode(mode, {})
        assert look.seen == ["Car"], mode


async def test_fill_gaps_runs_the_layout_route_only_while_a_field_is_empty() -> None:
    partial = {k: v for k, v in MAPPING.items() if v != "colours"}
    result, look = await run_mode("fill_gaps", partial)
    assert look.seen == ["Car"]
    assert "layout_route_skipped" not in [e.kind for e in result.meta.events]
    result, look = await run_mode("fill_gaps", MAPPING)
    assert look.seen == []  # every field was found
    assert result.one(Car).record.colours == ["Red", "Moonstone Grey"]


async def test_merge_runs_the_layout_route_for_every_field() -> None:
    ctx = Context.create(page(CAR), [SchemaSpec.from_model(Car)], mapping_jev().client())
    await StructuredStage(mode="merge").run(ctx)
    run = ctx.schemas["Car"]
    assert run.merge
    assert not run.finished
    assert run.needs(SINGLE_ENTITY_LABEL, "model")
    assert [e.kind for e in ctx.events] == ["structured_fields"]


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown structured mode 'both'"):
        StructuredStage(mode="both")  # pyright: ignore[reportArgumentType]


def test_structured_is_a_default_stage_before_layout() -> None:
    names = default_pipeline().names
    assert names.index("document_gate") < names.index("structured") < names.index("layout")


# --- the test site's JSON-LD, end to end ---------------------------------------------

SITE_MAPPING = {
    "brand.name": "make",
    "model": "model",
    "vehicleConfiguration": "trim",
    "fuelType": "fuel_type",
    "vehicleEngine.engineDisplacement.value": "engine_size_cc",
    "vehicleEngine.enginePower.value": "power_kw",
    "accelerationTime.value": "zero_to_62_s",
    "speed.value": "top_speed_mph",
    "emissionsCO2": "co2_g_km",
    "offers.price": "price_gbp",
    "seatingCapacity": "seats",
    "vehicleTransmission": "automatic",
}

FUEL_WORDS = {
    "Diesel": "diesel",
    "Petrol": "petrol",
    "Hybrid": "hybrid",
    "Plug-in hybrid": "phev",
    "Electric": "ev",
}


def site_jev() -> FakeJev:
    fake = mapping_jev(SITE_MAPPING)
    for word, option in FUEL_WORDS.items():
        fake.choice(
            re.compile("(?i)what is the fuel or powertrain type"),
            option,
            state=f"fuelType: {word}",
        )
    fake.noul("automatic gearbox", p=0.95, state="vehicleTransmission: Automatic")
    fake.noul("automatic gearbox", p=0.05, state="vehicleTransmission: Manual")
    return fake


async def test_the_test_sites_json_ld_maps_to_the_truth() -> None:
    pages = [p for p in render(generate(42)) if p.json_ld]
    assert len(pages) == 15
    mapper = KeyPathMapper()
    fake = site_jev()
    for p in pages:
        doc = Document.from_bytes(p.html.encode(), url=f"https://site.test/{p.path}")
        result = await mapper.extract(doc, [SchemaSpec.from_model(VehicleSpec)], fake.client())
        found = {n: m.value for n, m in result.fields["VehicleSpec"].items() if m.found}
        [record] = p.records
        truth = {k: v for k, v in record["values"].items() if v is not None}
        found["price_gbp"] = str(found["price_gbp"])
        assert found == truth, p.path
    # Every page shares the template, so only the first asked about key paths (EVs add
    # no displacement or CO2, so they have their own fingerprint).
    mapping_requests = [c for c in fake.calls if isinstance(c.state, dict) and "data" in c.state]
    assert len(mapping_requests) == 2


# --- review round 1 ------------------------------------------------------------------


async def test_every_schemas_questions_about_a_blob_go_in_one_request() -> None:
    fake = mapping_jev().choice(
        'key path "model"', lambda q: "title" if "title" in q.options else "model"
    )
    result = await KeyPathMapper().extract(
        page(CAR), [SchemaSpec.from_model(Car), SchemaSpec.from_model(Book)], fake.client()
    )
    [call] = fake.calls
    assert {k.split(":")[0] for k in call.questions} == {"Car", "Book"}
    assert result.fields["Book"]["title"].value == "Golf"
    assert result.fields["Car"]["model"].value == "Golf"


def test_the_key_path_question_can_be_overridden() -> None:
    class Custom(BaseModel):
        """A car."""

        __jevex__ = SchemaConfig(key_path_question="What is {path}? It holds {example}.")
        model: str = Field(description="Model name")

    question = SchemaSpec.from_model(Custom).key_path_question("model", "Golf")
    assert question.instructions == "What is model? It holds 'Golf'."


async def test_a_stale_mapping_to_a_missing_field_is_asked_again(store: SQLiteStore) -> None:
    fp = flatten(page_blob(CAR)).fingerprint
    await store.put_key_mapping(
        KeyMapping(fingerprint=fp, schema="Car", path="model", field="trim")
    )
    fake = mapping_jev()
    fields = await extract(KeyPathMapper(store=store), fake)
    assert fields["model"].value == "Golf"
    [call] = fake.calls
    asked = [q.instructions for q in call.questions.values()]
    assert "Which detail does the key path \"model\" hold (e.g. 'Golf')?" in asked


async def test_a_huge_blob_with_few_shapes_fits_one_small_state() -> None:
    listings = [{"title": f"Car {i}", "price": 1000 + i, "url": f"/cars/{i}"} for i in range(3000)]
    fake = FakeJev()
    await KeyPathMapper().extract(
        page({"props": {"listings": listings}}), [SchemaSpec.from_model(Car)], fake.client()
    )
    [call] = fake.calls
    assert isinstance(call.state, dict)
    assert len(call.state["data"]) < 2000  # 3 examples per shape, not 9000 leaves
    assert len(call.questions) == 3


async def test_many_shapes_are_chunked_across_states() -> None:
    wide = {f"key{i}": "x" * 150 for i in range(60)}
    fake = FakeJev()
    result = await KeyPathMapper(state_tokens=500).extract(
        page(wide), [SchemaSpec.from_model(Car)], fake.client()
    )
    assert len(fake.calls) > 1
    asked = sum(len(c.questions) for c in fake.calls)
    assert asked == 60
    assert result.events == []


def test_dotted_and_bracketed_keys_are_quoted_so_paths_dont_collide() -> None:
    flat = flatten(blob({"a.b": 1, "a": {"b": 2}, "x[0]": 3, "x": [4]}))
    assert [leaf.path for leaf in flat.leaves] == ['["a.b"]', "a.b", '["x[0]"]', "x[0]"]
    ids = [flat.statement_id(leaf) for leaf in flat.leaves]
    assert len(set(ids)) == 4


def test_fingerprint_includes_the_schema_org_type_and_source() -> None:
    brand = flatten(
        StructuredBlob(source="json_ld", data={"name": "x"}, types=["Brand"], location=LOC)
    )
    person = flatten(
        StructuredBlob(source="json_ld", data={"name": "x"}, types=["Person"], location=LOC)
    )
    micro = flatten(
        StructuredBlob(source="microdata", data={"name": "x"}, types=["Brand"], location=LOC)
    )
    assert len({brand.fingerprint, person.fingerprint, micro.fingerprint}) == 3


async def test_a_later_shape_or_blob_fills_a_field_an_earlier_one_failed() -> None:
    both = {"engine": "big", "engineSize": "1,498 cc"}
    fake = mapping_jev({"engine": "engine_size_cc", "engineSize": "engine_size_cc"})
    fields = await extract(KeyPathMapper(), fake, both)
    assert fields["engine_size_cc"].value == 1498
    scripts = "".join(
        f'<script type="application/ld+json">{json.dumps({"engine": v})}</script>'
        for v in ["big", "1,498 cc"]
    )
    doc = Document.from_bytes(f"<html><head>{scripts}</head></html>".encode())
    result = await KeyPathMapper().extract(
        doc, [SchemaSpec.from_model(Car)], mapping_jev({"engine": "engine_size_cc"}).client()
    )
    assert result.fields["Car"]["engine_size_cc"].value == 1498


async def test_str_fields_take_the_whole_value_and_numbers_as_text() -> None:
    data = {"name": "2021 Volkswagen Golf 1.5 TSI Life 5dr, 32,000 miles, £18,995", "sku": 3008}
    fields = await extract(KeyPathMapper(), mapping_jev({"name": "model"}), data)
    assert fields["model"].value == data["name"]
    fields = await extract(KeyPathMapper(), mapping_jev({"sku": "model"}), {"sku": 3008})
    assert fields["model"].value == "3008"


async def test_memory_forgets_the_least_recently_used_fingerprint() -> None:
    mapper = KeyPathMapper(memory_size=1)
    await extract(mapper, mapping_jev(), CAR)
    await extract(mapper, mapping_jev(), {"other": "shape"})
    again = mapping_jev()
    await extract(mapper, again, CAR)
    assert len(again.calls) == 1


def test_each_default_pipeline_gets_its_own_structured_stage() -> None:
    def stage() -> object:
        [s] = [s for s in default_pipeline().stages if s.name == "structured"]
        assert isinstance(s, StructuredStage)
        return s.extractor

    assert stage() is not stage()


class Car2(BaseModel):
    """A car."""

    fuels: list[Literal["petrol", "diesel", "ev"]] = Field(
        default_factory=list, description="Fuel types"
    )
    stock: Literal["InStock", "OutOfStock"] = Field(description="Availability")


async def test_list_enums_are_asked_one_noul_per_option_and_prefixes_are_stripped() -> None:
    fake = mapping_jev({"fuels": "fuels", "availability": "stock"})
    fake.noul(re.compile('"(petrol|ev)"'), p=0.9, state="Petrol or electric")
    result = await KeyPathMapper().extract(
        page({"fuels": "Petrol or electric", "availability": "schema:InStock"}),
        [SchemaSpec.from_model(Car2)],
        fake.client(),
    )
    fields = result.fields["Car2"]
    assert fields["fuels"].value == ["petrol", "ev"]
    assert fields["stock"].value == "InStock"
    [members] = [c for c in fake.calls if "member0" in c.questions]
    spec = SchemaSpec.from_model(Car2).field("fuels")
    assert members.questions == {
        f"member{i}": spec.member_question(o) for i, o in enumerate(["petrol", "diesel", "ev"])
    }


async def test_the_extractors_store_keeps_mappings_across_extractors(tmp_path: Path) -> None:
    from jevex import Extractor, Pipeline

    url = f"sqlite:///{tmp_path / 'jevex.db'}"
    async with Extractor(
        [Car], jev=mapping_jev().client(), pipeline=Pipeline([StructuredStage()]), store=url
    ) as first:
        result = await first.extract(page(CAR))
    assert result.values["Car"]["document"]["model"] == "Golf"

    fresh = FakeJev(strict=True)  # a new process: only the store remembers
    async with Extractor(
        [Car], jev=fresh.client(), pipeline=Pipeline([StructuredStage()]), store=url
    ) as second:
        result = await second.extract(page({**CAR, "model": "Polo"}))
    assert fresh.calls == []
    assert result.values["Car"]["document"]["model"] == "Polo"


async def test_an_enum_value_is_asked_once_per_distinct_value_and_remembered() -> None:
    data = {"items": [{"fuel": "Fully electric", "model": f"M{i}"} for i in range(40)]}
    fake = mapping_jev({"items[].fuel": "fuel"}).choice(
        re.compile("(?i)what is the fuel type"), "ev", state="Fully electric"
    )
    mapper = KeyPathMapper()
    fields = await extract(mapper, fake, data)
    assert fields["fuel"].value == "ev"
    enum_asks = [c for c in fake.calls if "enum" in c.questions]
    assert len(enum_asks) == 1  # 40 leaves, one distinct value
    before = len(fake.calls)
    await extract(mapper, fake, data)  # the next page from the template
    assert len(fake.calls) == before  # mapping and value both remembered: no calls


async def test_a_capped_request_cancels_the_other_blobs_before_extract_returns() -> None:
    from jevex import Budgets, DocBudget, Extractor, Pipeline
    from jevex.jev import JevResponse

    finished: list[str] = []

    class SlowFirst:
        async def system_one(self, state: object, questions: object) -> JevResponse:
            text = json.dumps(state)
            if "slow" in text:
                await asyncio.sleep(0.2)
            finished.append(text)
            return JevResponse(answers={}, input_tokens=1, model="fake")

    scripts = "".join(
        f'<script type="application/ld+json">{json.dumps(d)}</script>'
        for d in ({"name": "slow", "a": 1}, {"title": "fast", "b": 2})
    )
    doc = Document.from_bytes(f"<html><head>{scripts}</head></html>".encode())
    ex = Extractor(
        [Car],
        jev=JevClient(SlowFirst()),
        pipeline=Pipeline([StructuredStage()]),
        budgets=Budgets(per_document=DocBudget(max_jev_requests=1)),
    )
    result = await ex.extract(doc)
    assert result.meta.stopped
    seen = len(finished)
    await asyncio.sleep(0.3)
    assert len(finished) == seen  # nothing finished after the result was built


async def test_fields_sharing_a_name_dont_share_value_answers() -> None:
    import enum

    from jevex import Questions

    class Fuel(enum.Enum):
        PETROL = "petrol"
        EV = "ev"

    class EnumCar(BaseModel):
        """A car."""

        fuel: Fuel = Field(description="Fuel type")

    class LiteralCar(BaseModel):
        """A car."""

        fuel: Literal["petrol", "ev"] = Field(description="Fuel type")

    class ListCar(BaseModel):
        """A car."""

        fuel: list[Literal["petrol", "ev"]] = Field(
            default_factory=list,
            description="Fuel type",
            questions=Questions(member="Is {value} a fuel it can run on?"),
        )

    data = {"fuel": "Fully electric"}
    fake = (
        FakeJev()
        .choice('key path "fuel"', "fuel", confidence=0.9)
        .choice(re.compile("(?i)what is the fuel type"), "ev", state="Fully electric")
        .noul("Is ev a fuel it can run on?", p=0.9)
    )
    mapper = KeyPathMapper()
    specs = [SchemaSpec.from_model(m) for m in (EnumCar, LiteralCar, ListCar)]
    result = await mapper.extract(page(data), specs, fake.client())
    assert result.fields["EnumCar"]["fuel"].value is Fuel.EV
    assert result.fields["LiteralCar"]["fuel"].value == "ev"  # the string, not Fuel.EV
    assert result.fields["ListCar"]["fuel"].value == ["ev"]  # asked its own member question
    member_asks = [c for c in fake.calls if any(k.startswith("member") for k in c.questions)]
    assert len(member_asks) == 1


async def test_list_enum_fallbacks_are_capped_per_document() -> None:
    from jevex.keypaths import MAX_LIST_FALLBACK_VALUES

    class Fleet(BaseModel):
        """A fleet."""

        fuels: list[Literal["petrol", "ev"]] = Field(default_factory=list, description="Fuels")

    data = {"fuels": [f"Blend {i}" for i in range(MAX_LIST_FALLBACK_VALUES + 10)]}
    fake = FakeJev().choice('key path "fuels[]"', "fuels", confidence=0.9)
    result = await KeyPathMapper().extract(
        page(data), [SchemaSpec.from_model(Fleet)], fake.client()
    )
    member_asks = [c for c in fake.calls if any(k.startswith("member") for k in c.questions)]
    assert len(member_asks) == MAX_LIST_FALLBACK_VALUES
    assert not result.fields["Fleet"]["fuels"].found  # FakeJev accepts no member (p=0)


async def test_merge_mode_end_to_end_records_the_layout_routes_disagreement() -> None:
    class Named(BaseModel):
        """A car."""

        model: str = Field(description="Model name")

    data = {"model": "Golf"}
    script = f'<script type="application/ld+json">{json.dumps(data)}</script>'
    doc = Document.from_bytes(
        f"<html><head>{script}</head><body><p>Model: Polo</p></body></html>".encode()
    )
    fake = (
        FakeJev(default_p=0.9)  # every gate passes
        .choice('key path "model"', "model")
        .choice(None, lambda q: "model" if "model" in q.options else "Polo", confidence=0.8)
    )
    pipeline = default_pipeline().replace("structured", StructuredStage(mode="merge"))
    result = await Extractor([Named], jev=fake.client(), pipeline=pipeline).extract(doc)
    meta = result.one(Named).meta.model
    assert (meta.value, meta.method) == ("Golf", "structured")
    [conflict] = meta.conflicts
    assert (conflict.value, conflict.method, conflict.confidence) == ("Polo", "generator", 0.8)

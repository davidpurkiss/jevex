from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from jevex import Component, Context, Document, DomLocation, Field, SchemaSpec, Statement
from jevex.clean import CleanStage
from jevex.entities import EntityScope
from jevex.extractor import DEFAULT_STAGES, STAGE_ORDER, default_pipeline
from jevex.interfaces import EntityResolver, ParsedDocument
from jevex.pipeline import Pipeline
from jevex.resolve import SINGLE_ENTITY_LABEL, EntityStage, SingleEntity
from jevex.testing import FakeJev


class Listing(BaseModel):
    price: int = Field(description="Price")


class Book(BaseModel):
    title: str = Field(description="Title")


def parsed() -> ParsedDocument:
    loc = DomLocation(dom_path="/")
    root = Component(
        id="c0",
        type="section",
        location=loc,
        children=[
            Component(id="c1", type="paragraph", text="Price: 18495", location=loc),
            Component(id="c2", type="paragraph", text="Title: Dune", location=loc),
        ],
    )
    statements = {
        s.id: s
        for s in [
            Statement(
                id="s1", text="Price: 18495", kind="key_value", component_id="c1", location=loc
            ),
            Statement(
                id="s2", text="Title: Dune", kind="key_value", component_id="c2", location=loc
            ),
        ]
    }
    return ParsedDocument(document=Document.from_bytes(b"<p/>"), root=root, statements=statements)


def ctx(*models: type[BaseModel], with_parsed: bool = True) -> Context:
    c = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(m) for m in models], FakeJev().client()
    )
    if with_parsed:
        c.parsed = parsed()
    return c


async def test_single_entity_covers_the_whole_document() -> None:
    fake = FakeJev(strict=True)  # SingleEntity must ask Jev nothing
    [scope] = await SingleEntity().resolve(parsed(), SchemaSpec.from_model(Listing), fake.client())
    assert scope == EntityScope(
        label="document", component_ids=["c0", "c1", "c2"], statement_ids=["s1", "s2"]
    )
    assert fake.calls == []


async def test_single_entity_label_is_configurable() -> None:
    [scope] = await SingleEntity(label="listing").resolve(
        parsed(), SchemaSpec.from_model(Listing), FakeJev().client()
    )
    assert scope.label == "listing"


async def test_entity_stage_sets_scopes_for_every_active_schema() -> None:
    c = ctx(Listing, Book)
    c.schemas["Book"].deactivate()
    await EntityStage().run(c)
    assert [s.label for s in c.schemas["Listing"].scopes] == [SINGLE_ENTITY_LABEL]
    assert c.schemas["Book"].scopes == []


async def test_entity_stage_without_layout_gives_one_empty_scope() -> None:
    c = ctx(Listing, with_parsed=False)
    await EntityStage().run(c)
    assert c.schemas["Listing"].scopes == [EntityScope(label=SINGLE_ENTITY_LABEL)]


async def test_entity_stage_records_an_event_when_nothing_is_found() -> None:
    @dataclass
    class Nobody:
        async def resolve(
            self, parsed: ParsedDocument, schema: SchemaSpec, jev: object
        ) -> list[EntityScope]:
            return []

    c = ctx(Listing)
    await EntityStage(resolver=Nobody()).run(c)
    assert c.schemas["Listing"].scopes == []
    assert [e.kind for e in c.events] == ["no_entities"]


def test_single_entity_satisfies_the_protocol() -> None:
    assert isinstance(SingleEntity(), EntityResolver)


# --- default pipeline ordering ---------------------------------------------------------


def test_default_pipeline_is_in_spec_order() -> None:
    names = default_pipeline().names
    assert names == sorted(names, key=STAGE_ORDER.index)
    assert {"clean", "entities"} <= set(names)
    assert {s.name for s in DEFAULT_STAGES} == set(names)


def test_default_pipeline_rejects_unknown_stage_names(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class Odd:
        name: str = "mystery"

        async def run(self, ctx: Context) -> None: ...

    import jevex.extractor as extractor

    monkeypatch.setattr(extractor, "DEFAULT_STAGES", (*extractor.DEFAULT_STAGES, Odd()))
    with pytest.raises(ValueError, match="mystery"):
        extractor.default_pipeline()


def test_custom_pipelines_keep_the_order_they_are_given() -> None:
    assert Pipeline([EntityStage(), CleanStage()]).names == ["entities", "clean"]


async def test_no_layout_scope_uses_the_resolvers_label() -> None:
    c = ctx(Listing, with_parsed=False)
    await EntityStage(resolver=SingleEntity(label="listing")).run(c)
    assert [s.label for s in c.schemas["Listing"].scopes] == ["listing"]

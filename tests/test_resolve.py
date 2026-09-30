from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from jevex import (
    Component,
    Context,
    Document,
    DomLocation,
    Extractor,
    Field,
    MultiEntity,
    SchemaConfig,
    SchemaSpec,
    Statement,
)
from jevex.clean import CleanStage
from jevex.entities import EntityScope
from jevex.extractor import DEFAULT_STAGES, STAGE_ORDER, default_pipeline
from jevex.interfaces import EntityResolver, ParsedDocument
from jevex.jev import Choice, Noul
from jevex.layout import LayoutStage
from jevex.pipeline import Pipeline
from jevex.resolve import SINGLE_ENTITY_LABEL, EntityStage, SingleEntity
from jevex.schema import ALL_OPTION
from jevex.split import StatementStage
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


def test_statements_are_split_before_entities_are_resolved() -> None:
    # MultiEntity assigns single statements (a table's cells, a sentence about one trim).
    assert STAGE_ORDER.index("statements") < STAGE_ORDER.index("entities")
    names = default_pipeline().names
    assert names.index("component_gate") < names.index("statements") < names.index("entities")


# --- MultiEntity -----------------------------------------------------------------------


class VehicleSpec(BaseModel):
    """A car's technical specification."""

    power_ps: int = Field(description="Power", unit="PS")
    doors: int = Field(description="Number of doors")
    automatic: bool = Field(description="has an automatic gearbox")


VEHICLE = SchemaSpec.from_model(VehicleSpec)
BOUNDARY = "name a separate vehicle spec?"
WHICH = "Which vehicle spec does this statement apply to?"

TABLE_PAGE = """<h1>Kestrova</h1>
<p>Every Kestrova has 5 doors.</p>
<p>The SE L has 3 doors.</p>
<table>
<thead><tr><th></th><th>SE</th><th>SE L</th></tr></thead>
<tbody>
<tr><th>Power</th><td>150PS</td><td>180PS</td></tr>
<tr><th>Warranty</th><td colspan="2">3 years</td></tr>
</tbody>
</table>"""


async def parse(html: str) -> ParsedDocument:
    c = Context.create(
        Document.from_bytes(html.encode(), content_type="text/html"), [], FakeJev().client()
    )
    await LayoutStage().run(c)
    await StatementStage().run(c)
    assert c.parsed is not None
    return c.parsed


def texts(parsed: ParsedDocument, ids: list[str]) -> list[str]:
    return [parsed.statements[i].text for i in ids]


async def test_multi_entity_splits_a_comparison_table_by_column() -> None:
    parsed = await parse(TABLE_PAGE)
    fake = (
        FakeJev(strict=True)
        .noul(BOUNDARY, p=0.9)
        .noul(BOUNDARY, p=0.1, state="Warranty")  # the row labels are rejected
        .choice(WHICH, ALL_OPTION)
        .choice(WHICH, "SE L", state="The SE L")
    )
    se, se_l = await MultiEntity().resolve(parsed, VEHICLE, fake.client())

    assert (se.label, se_l.label) == ("SE", "SE L")
    assert texts(parsed, se.statement_ids) == ["Power · SE: 150PS", "Warranty · SE / SE L: 3 years"]
    assert texts(parsed, se_l.statement_ids) == [
        "The SE L has 3 doors.",
        "Power · SE L: 180PS",
        "Warranty · SE / SE L: 3 years",  # a cell spanning both columns is on both
    ]
    shared = ["Kestrova", "Every Kestrova has 5 doors."]
    assert texts(parsed, se.shared_statement_ids) == shared
    assert texts(parsed, se_l.shared_statement_ids) == shared
    # A component is on a scope only when all its statements are: not the split table.
    assert se.component_ids == []
    assert se_l.component_ids == [parsed.statements[se_l.statement_ids[0]].component_id]

    columns, rows, *assigned = fake.calls
    assert columns.state == {"names": ["SE", "SE L"], "section": "Kestrova"}
    assert columns.questions == {
        "label0": Noul(instructions='Does "SE" name a separate vehicle spec?'),
        "label1": Noul(instructions='Does "SE L" name a separate vehicle spec?'),
    }
    # The rows could be the entities just as well; Jev decides.
    assert rows.state == {"names": ["Power", "Warranty"], "section": "Kestrova"}
    # Every statement no boundary claimed is asked which entity it's about.
    assert [c.state for c in assigned] == [
        {"statement": "Kestrova"},
        {"statement": "Every Kestrova has 5 doors.", "section": "Kestrova"},
        {"statement": "The SE L has 3 doors.", "section": "Kestrova"},
    ]
    assert assigned[0].questions == {
        "entity": Choice(
            instructions=WHICH,
            options={
                "SE": None,
                "SE L": None,
                ALL_OPTION: "It applies to every vehicle spec",
            },
        )
    }


async def test_multi_entity_scopes_give_downstream_stages_their_statements() -> None:
    parsed = await parse(TABLE_PAGE)
    fake = FakeJev().noul(BOUNDARY, p=0.9).choice(WHICH, ALL_OPTION)
    se, _ = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    assert [s.text for s in parsed.scope_statements(se)] == [
        "Kestrova",
        "Every Kestrova has 5 doors.",
        "The SE L has 3 doors.",  # FakeJev said "all of them" to everything here
        "Power · SE: 150PS",
        "Warranty · SE / SE L: 3 years",
    ]


async def test_multi_entity_leaves_topic_sections_as_one_entity() -> None:
    parsed = await parse(
        "<h1>Kestrova</h1><h2>Performance</h2><p>150PS.</p><h2>Dimensions</h2><p>4.2 m.</p>"
    )
    fake = FakeJev(strict=True).noul(BOUNDARY, p=0.1)
    scopes = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    assert scopes == await SingleEntity().resolve(parsed, VEHICLE, fake.client())
    [boundary] = fake.calls  # no entity questions with a single entity
    assert boundary.state == {"names": ["Performance", "Dimensions"], "section": "Kestrova"}


async def test_multi_entity_splits_headed_sections_jev_says_are_entities() -> None:
    parsed = await parse(
        "<h1>Kestrova</h1><h2>SE</h2><p>150PS.</p>"
        "<h2>SE L</h2><p>180PS.</p><p>An automatic gearbox.</p>"
    )
    fake = FakeJev(strict=True).noul(BOUNDARY, p=0.8).choice(WHICH, ALL_OPTION)
    se, se_l = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    assert texts(parsed, se.statement_ids) == ["SE", "150PS."]
    assert texts(parsed, se_l.statement_ids) == ["SE L", "180PS.", "An automatic gearbox."]
    assert texts(parsed, se.shared_statement_ids) == ["Kestrova"]
    assert len(se_l.component_ids) == 3  # its heading and two paragraphs


async def test_multi_entity_splits_repeated_cards_and_numbers_repeated_labels() -> None:
    card = "<article><h3>{}</h3><p>{}</p><p>Automatic.</p></article>"
    parsed = await parse(
        "<h1>Used cars</h1>"
        + card.format("Kestrova SE", "£18,000")
        + card.format("Kestrova SE", "£17,500")
        + card.format("Kestrova R", "£31,000")
    )
    fake = FakeJev(strict=True).noul(BOUNDARY, p=0.9).choice(WHICH, ALL_OPTION)
    scopes = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    assert [s.label for s in scopes] == ["Kestrova SE", "Kestrova SE (2)", "Kestrova R"]
    assert [texts(parsed, s.statement_ids)[1] for s in scopes] == ["£18,000", "£17,500", "£31,000"]
    assert fake.calls[0].state == {
        "names": ["Kestrova SE", "Kestrova SE (2)", "Kestrova R"],
        "section": "Used cars",
    }


async def test_multi_entity_keeps_only_the_labels_jev_accepts() -> None:
    parsed = await parse(
        "<h2>SE</h2><p>150PS.</p><h2>Warranty</h2><p>3 years.</p><h2>SE L</h2><p>180PS.</p>"
    )
    fake = (
        FakeJev(strict=True)
        .noul(BOUNDARY, p=0.9)
        .noul('"Warranty"', p=0.2)
        .choice(WHICH, ALL_OPTION)
    )
    se, se_l = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    assert (se.label, se_l.label) == ("SE", "SE L")
    # The rejected section's statements were asked about like any other.
    assert texts(parsed, se.shared_statement_ids) == ["Warranty", "3 years."]


async def test_multi_entity_gives_nested_groups_to_the_innermost() -> None:
    parsed = await parse(
        "<h2>SE</h2><h3>Engine</h3><p>150PS.</p><h3>Price</h3><p>£20,000.</p>"
        "<h2>SE L</h2><h3>Engine</h3><p>180PS.</p>"
        "<table><tr><th></th><th>Manual</th><th>Automatic</th></tr>"
        "<tr><th>0-62 mph</th><td>8.9 s</td><td>9.2 s</td></tr></table>"
        "<h3>Price</h3><p>£24,000.</p>"
    )
    fake = (
        FakeJev(strict=True)
        .noul(BOUNDARY, p=0.9)
        .noul(BOUNDARY, p=0.1, state='"Engine"')  # topics, not trims
    )
    scopes = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    by_label = {s.label: texts(parsed, s.statement_ids) for s in scopes}
    assert by_label == {
        "SE": ["SE", "Engine", "150PS.", "Price", "£20,000."],
        "SE L": ["SE L", "Engine", "180PS.", "Price", "£24,000."],
        # A table's columns win over the section around it.
        "Manual": ["0-62 mph · Manual: 8.9 s"],
        "Automatic": ["0-62 mph · Automatic: 9.2 s"],
    }


async def test_multi_entity_splits_listing_cards_under_trim_headings() -> None:
    card = "<article><h3>{}</h3><p>{}</p></article>"
    parsed = await parse(
        "<h2>SE</h2>"
        + card.format("2019 SE, 40k miles", "£18,000")
        + card.format("2021 SE, 12k miles", "£21,000")
        + "<h2>SE L</h2>"
        + card.format("2020 SE L, 30k miles", "£22,000")
        + card.format("2022 SE L, 8k miles", "£26,000")
    )
    fake = FakeJev(strict=True).noul(BOUNDARY, p=0.9).choice(WHICH, ALL_OPTION)
    scopes = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    # Each card is a listing; "SE" and "SE L" are left with only their headings, so they
    # aren't entities, and their headings are asked about like any other statement.
    assert [s.label for s in scopes] == [
        "2019 SE, 40k miles",
        "2021 SE, 12k miles",
        "2020 SE L, 30k miles",
        "2022 SE L, 8k miles",
    ]
    assert [texts(parsed, s.statement_ids)[1] for s in scopes] == [
        "£18,000",
        "£21,000",
        "£22,000",
        "£26,000",
    ]
    assert texts(parsed, scopes[0].shared_statement_ids) == ["SE", "SE L"]


ROW_KEYED = (
    "<table><tr><th>Trim</th><th>Power</th><th>Price</th></tr>"
    "<tr><th>SE</th><td>150PS</td><td>£20,000</td></tr>"
    "<tr><th>SE L</th><td>180PS</td><td>£24,000</td></tr></table>"
)


async def test_multi_entity_lets_jev_pick_a_tables_rows_as_the_entities() -> None:
    parsed = await parse(ROW_KEYED)
    fake = FakeJev(strict=True).noul(BOUNDARY, p=0.9).noul(BOUNDARY, p=0.1, state="Power")
    se, se_l = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    assert (se.label, se_l.label) == ("SE", "SE L")
    assert texts(parsed, se_l.statement_ids) == [
        "SE L · Power: 180PS",
        "SE L · Price: £24,000",
    ]


async def test_multi_entity_without_confirm_splits_tables_by_column_only() -> None:
    parsed = await parse(ROW_KEYED)
    scopes = await MultiEntity(confirm=False).resolve(
        parsed, VEHICLE, FakeJev(strict=True).client()
    )
    # Nothing tells a trim per row from a trim per column without asking.
    assert [s.label for s in scopes] == ["Power", "Price"]


async def test_multi_entity_without_confirm_asks_no_boundary_questions() -> None:
    parsed = await parse(TABLE_PAGE)
    fake = FakeJev(strict=True).choice(WHICH, ALL_OPTION)
    scopes = await MultiEntity(confirm=False).resolve(parsed, VEHICLE, fake.client())
    assert [s.label for s in scopes] == ["SE", "SE L"]
    assert all("entity" in c.questions for c in fake.calls)


async def test_multi_entity_with_one_entity_is_a_single_entity() -> None:
    # One column label, and a table without row headers: nothing to split.
    parsed = await parse(
        "<table><tr><th></th><th>SE</th></tr><tr><th>Power</th><td>150PS</td></tr></table>"
        "<table><tr><th>SE</th><th>SE L</th></tr><tr><td>150PS</td><td>180PS</td></tr></table>"
    )
    fake = FakeJev(strict=True)
    [scope] = await MultiEntity(label="car").resolve(parsed, VEHICLE, fake.client())
    assert scope.label == "car"
    assert scope.statement_ids == list(parsed.statements)
    assert fake.calls == []


async def test_multi_entity_leaves_statements_out_past_the_choice_option_limit() -> None:
    columns = "".join(f"<th>Trim {i}</th>" for i in range(255))
    cells = "".join(f"<td>{i}</td>" for i in range(255))
    parsed = await parse(
        f"<p>Every trim has 5 doors.</p><table><tr><th></th>{columns}</tr>"
        f"<tr><th>Power</th>{cells}</tr></table>"
    )
    fake = FakeJev(strict=True)
    scopes = await MultiEntity(confirm=False).resolve(parsed, VEHICLE, fake.client())
    assert len(scopes) == 255
    assert fake.calls == []  # 255 labels + "all of them" don't fit one Choice
    assert all(len(s.statement_ids) == 1 and not s.shared_statement_ids for s in scopes)


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_multi_entity_rejects_a_boundary_p_outside_0_to_1(p: float) -> None:
    with pytest.raises(ValueError, match="boundary_p"):
        MultiEntity(boundary_p=p)


def test_multi_entity_satisfies_the_protocol() -> None:
    assert isinstance(MultiEntity(), EntityResolver)


async def test_multi_entity_uses_the_schemas_entity_questions() -> None:
    class Trim(BaseModel):
        __jevex__ = SchemaConfig(
            entity_name="trim",
            boundary_question="Is {label} a {entity}?",
            entity_question="Which {entity} is this about?",
        )
        power_ps: int = Field(description="Power", unit="PS")

    parsed = await parse(TABLE_PAGE)
    fake = (
        FakeJev(strict=True)
        .noul("Is SE", p=0.9)
        .noul("Is SE L", p=0.9)
        .noul("Is Power", p=0.1)
        .noul("Is Warranty", p=0.1)
        .choice("Which trim is this about?", ALL_OPTION)
    )
    scopes = await MultiEntity().resolve(parsed, SchemaSpec.from_model(Trim), fake.client())
    assert [s.label for s in scopes] == ["SE", "SE L"]
    assert fake.calls[0].questions["label1"] == Noul(instructions="Is SE L a trim?")
    options = fake.calls[2].questions["entity"]
    assert isinstance(options, Choice)
    assert options.options[ALL_OPTION] == "It applies to every trim"


async def test_entity_stage_gives_multi_entity_only_what_passed_the_component_gate() -> None:
    parsed = await parse(TABLE_PAGE)
    c = Context.create(parsed.document, [VEHICLE], FakeJev().client())
    c.parsed = parsed
    table = next(comp.id for comp in parsed.root.walk() if comp.type == "table")
    c.schemas["VehicleSpec"].component_ids = {"power_ps": [parsed.root.id, table]}
    fake = FakeJev(strict=True).noul(BOUNDARY, p=0.9).noul(BOUNDARY, p=0.1, state="Power")
    c.jev = fake.client()
    await EntityStage(resolver=MultiEntity()).run(c)
    se, se_l = c.schemas["VehicleSpec"].scopes
    # The paragraphs were gated out, so no entity question is asked about them.
    assert len(fake.calls) == 2  # the columns and the rows
    assert texts(parsed, se_l.statement_ids) == [
        "Power · SE L: 180PS",
        "Warranty · SE / SE L: 3 years",
    ]
    assert se.shared_statement_ids == se_l.shared_statement_ids == []


async def test_entity_stage_drops_statements_a_resolver_gives_from_gated_out_parts() -> None:
    @dataclass
    class Everything:
        async def resolve(
            self, parsed: ParsedDocument, schema: SchemaSpec, jev: object
        ) -> list[EntityScope]:
            return [
                EntityScope(label="all", statement_ids=["s1", "s2"], shared_statement_ids=["s2"])
            ]

    c = ctx(Listing)
    c.schemas["Listing"].component_ids = {"price": ["c0", "c1"]}
    await EntityStage(resolver=Everything()).run(c)
    [scope] = c.schemas["Listing"].scopes
    assert (scope.statement_ids, scope.shared_statement_ids) == (["s1"], [])


# --- shared values, end to end ---------------------------------------------------------


def first_option(q: Choice) -> str:
    return next(iter(q.options))


async def test_shared_values_are_copied_into_each_record_and_own_values_win() -> None:
    page = TABLE_PAGE.replace(
        "<p>The SE L has 3 doors.</p>",
        "<p>The SE L has 3 doors.</p><p>Every Kestrova has an automatic gearbox.</p>"
        "<p>The SE has a manual gearbox.</p>",
    )
    fake = (
        FakeJev(strict=True)
        .noul("Does this document", p=0.95)
        .noul("Does this section", p=0.9)
        .noul(BOUNDARY, p=0.9)
        .choice(WHICH, ALL_OPTION)
        .choice(WHICH, "SE L", state="The SE L")
        .choice(WHICH, "SE", state="The SE has")
        .choice("Which detail", "none")
        .choice("Which detail", "power_ps", state="Power")
        .choice("Which detail", "doors", state="doors")
        .choice("Which detail", "automatic", state="gearbox")
        .choice("Which of these", first_option, confidence=0.9)
        .noul("Does the statement say it has", p=0.95)
        .noul("Does the statement say it has", p=0.05, state="manual")
    )
    pipeline = default_pipeline().replace("entities", EntityStage(resolver=MultiEntity()))
    async with Extractor([VehicleSpec], jev=fake.client(), pipeline=pipeline) as ex:
        result = await ex.extract(Document.from_bytes(page.encode(), content_type="text/html"))

    se, se_l = result.records
    assert (se.entity, se_l.entity) == ("SE", "SE L")
    assert se.record.model_dump() == {"power_ps": 150, "doors": 5, "automatic": False}
    assert se_l.record.model_dump() == {"power_ps": 180, "doors": 3, "automatic": True}
    # "Every Kestrova..." applies to all of them: SE takes its doors, and SE L its gearbox.
    assert se.meta.doors.shared
    assert se_l.meta.automatic.shared
    # Each entity's own statements win over shared ones.
    assert not se.meta.automatic.shared
    assert not se_l.meta.doors.shared
    assert not se.meta.power_ps.shared


async def test_entity_stage_reports_statements_no_scope_holds() -> None:
    @dataclass
    class OnlyPrice:
        async def resolve(
            self, parsed: ParsedDocument, schema: SchemaSpec, jev: object
        ) -> list[EntityScope]:
            return [EntityScope(label="price", component_ids=["c1"])]

    c = ctx(Listing)
    await EntityStage(resolver=OnlyPrice()).run(c)
    [event] = c.events
    assert (event.kind, event.data) == ("unassigned_statements", {"statement_ids": ["s2"]})
    assert event.message == "Listing: 1 statement(s) belong to no entity"

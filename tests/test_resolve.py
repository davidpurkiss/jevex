from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from jevex import (
    ChildFieldError,
    Component,
    ComponentType,
    Context,
    Document,
    DomLocation,
    ExtractionResult,
    Extractor,
    Field,
    InvalidScopeError,
    MultiEntity,
    ParentChild,
    SchemaConfig,
    SchemaSpec,
    Statement,
)
from jevex.clean import CleanStage
from jevex.entities import EntityScope
from jevex.extractor import DEFAULT_STAGES, STAGE_ORDER, default_pipeline
from jevex.interfaces import EntityResolver, ParsedDocument
from jevex.jev import Choice, Noul
from jevex.keypaths import StructuredItem, StructuredMode, StructuredStage
from jevex.layout import LayoutStage
from jevex.pipeline import Pipeline, SchemaRun
from jevex.resolve import (
    SINGLE_ENTITY_LABEL,
    EntityStage,
    SingleEntity,
    match_label,
    place_document_values,
)
from jevex.results import FieldMeta, Source
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
    # Each column's header statement is its entity's: the trim no cell states.
    assert texts(parsed, se.statement_ids) == [
        "SE",
        "Power · SE: 150PS",
        "Warranty · SE / SE L: 3 years",
    ]
    assert texts(parsed, se_l.statement_ids) == [
        "The SE L has 3 doors.",
        "SE L",
        "Power · SE L: 180PS",
        "Warranty · SE / SE L: 3 years",  # a cell spanning both columns is on both
    ]
    # Row labels head every column, so they're shared without a question.
    shared = ["Kestrova", "Every Kestrova has 5 doors.", "Power", "Warranty"]
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
    # Every statement no boundary claimed is asked which entity it's about, except the
    # table's headers.
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
        "SE",
        "Power",
        "Power · SE: 150PS",
        "Warranty",
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
        # The table's row label isn't a column's, so the section around it claims it.
        "SE L": ["SE L", "Engine", "180PS.", "0-62 mph", "Price", "£24,000."],
        # A table's columns win over the section around it, headers included.
        "Manual": ["Manual", "0-62 mph · Manual: 8.9 s"],
        "Automatic": ["Automatic", "0-62 mph · Automatic: 9.2 s"],
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
        "SE L",
        "SE L · Power: 180PS",
        "SE L · Price: £24,000",
    ]
    # The column labels head every row: shared, and not asked about.
    assert texts(parsed, se_l.shared_statement_ids) == ["Power", "Price"]
    assert all("entity" not in c.questions for c in fake.calls)


async def test_multi_entity_joins_a_rows_stacked_headers_into_one_label() -> None:
    parsed = await parse(
        "<table><tr><th></th><th></th><th>Power</th><th>Price</th></tr>"
        '<tr><th rowspan="2">Kestrova</th><th>SE</th><td>150PS</td><td>£20,000</td></tr>'
        "<tr><th>SE L</th><td>180PS</td><td>£24,000</td></tr></table>"
    )
    fake = FakeJev(strict=True).noul(BOUNDARY, p=0.9).noul(BOUNDARY, p=0.1, state="Power")
    se, se_l = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    # One label per row, not "Kestrova" holding every row's cells.
    assert (se.label, se_l.label) == ("Kestrova SE", "Kestrova SE L")
    assert texts(parsed, se_l.statement_ids) == [
        "Kestrova",  # the outer row header covers both rows, so it's on both
        "SE L",
        "Kestrova · SE L · Power: 180PS",
        "Kestrova · SE L · Price: £24,000",
    ]


async def test_multi_entity_gives_a_cell_spanning_rows_to_each_rows_entity() -> None:
    parsed = await parse(
        "<table><tr><th></th><th></th><th>Power</th><th>Warranty</th></tr>"
        '<tr><th rowspan="2">Kestrova</th><th>SE</th><td>150PS</td>'
        '<td rowspan="2">3 years</td></tr>'
        "<tr><th>SE L</th><td>180PS</td></tr></table>"
    )
    fake = FakeJev(strict=True).noul(BOUNDARY, p=0.9).noul(BOUNDARY, p=0.1, state="Power")
    se, se_l = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    # Not a third "Kestrova SE SE L" entity holding the warranty alone.
    assert [c.state for c in fake.calls if "Kestrova" in str(c.state)] == [
        {"names": ["Kestrova SE", "Kestrova SE L"]}
    ]
    assert (se.label, se_l.label) == ("Kestrova SE", "Kestrova SE L")
    warranty = "Kestrova · SE · SE L · Warranty: 3 years"
    assert texts(parsed, se.statement_ids) == [
        "Kestrova",
        "SE",
        "Kestrova · SE · Power: 150PS",
        warranty,
    ]
    assert texts(parsed, se_l.statement_ids) == [
        "Kestrova",
        warranty,
        "SE L",
        "Kestrova · SE L · Power: 180PS",
    ]


async def test_multi_entity_prefers_a_tables_columns_when_jev_accepts_both_axes() -> None:
    parsed = await parse(TABLE_PAGE)
    fake = FakeJev().noul(BOUNDARY, p=0.9).choice(WHICH, ALL_OPTION)
    scopes = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    assert [s.label for s in scopes] == ["SE", "SE L"]
    # The rows' labels lost their cells to the columns, so they're not entities: they're
    # shared, without a question.
    assert texts(parsed, scopes[0].shared_statement_ids)[-2:] == ["Power", "Warranty"]
    assert not any(c.state == {"statement": "Power"} for c in fake.calls)


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


async def test_headers_of_a_table_whose_axes_are_rejected_are_asked_about() -> None:
    # The trims are sections; the price table's years are rejected on both axes, so its
    # headers aren't known to head every trim, and Jev is asked like for any statement.
    parsed = await parse(
        "<table><tr><th></th><th>2022</th><th>2023</th></tr>"
        "<tr><th>Price</th><td>£20,000</td><td>£21,000</td></tr></table>"
        "<h2>SE</h2><p>150PS.</p><h2>SE L</h2><p>180PS.</p>"
    )
    fake = (
        FakeJev(strict=True)
        .noul(BOUNDARY, p=0.1)
        .noul('"SE', p=0.9)  # the section headings (the last matching rule wins)
        .choice(WHICH, ALL_OPTION)
    )
    se, _ = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    assert [c.state for c in fake.calls if "entity" in c.questions] == [
        {"statement": "2022", "table_headers": "2022, 2023"},
        {"statement": "2023", "table_headers": "2022, 2023"},
        {"statement": "Price", "table_headers": "Price"},
        {"statement": "Price · 2022: £20,000"},
        {"statement": "Price · 2023: £21,000"},
    ]
    assert texts(parsed, se.shared_statement_ids) == [
        "2022",
        "2023",
        "Price",
        "Price · 2022: £20,000",
        "Price · 2023: £21,000",
    ]


async def test_a_header_on_the_entities_axis_that_jev_rejected_is_asked_about() -> None:
    parsed = await parse(
        "<table><tr><th></th><th>SE</th><th>SE L</th><th>Notes</th></tr>"
        "<tr><th>Power</th><td>150PS</td><td>180PS</td><td>est.</td></tr></table>"
    )
    fake = (
        FakeJev(strict=True)
        .noul(BOUNDARY, p=0.9)
        .noul('"Notes"', p=0.1)
        .noul('"Power"', p=0.1)
        .choice(WHICH, ALL_OPTION)
    )
    se, _ = await MultiEntity().resolve(parsed, VEHICLE, fake.client())
    # "Power" heads every entity's column: shared without a question. "Notes" sits among
    # the entities but isn't one, so Jev is asked about it like its cell.
    asked = [c.state for c in fake.calls if "entity" in c.questions]
    assert asked == [
        {"statement": "Notes", "table_headers": "SE, SE L, Notes"},
        {"statement": "Power · Notes: est."},
    ]
    assert texts(parsed, se.shared_statement_ids) == ["Notes", "Power", "Power · Notes: est."]


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
    assert texts(parsed, scopes[7].statement_ids) == ["Trim 7", "Power · Trim 7: 7"]
    # The paragraph is left out; the row label needs no question, so it's still shared.
    assert all(texts(parsed, s.shared_statement_ids) == ["Power"] for s in scopes)


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
        "SE L",
        "Power · SE L: 180PS",
        "Warranty · SE / SE L: 3 years",
    ]
    assert texts(parsed, se.shared_statement_ids) == ["Power", "Warranty"]


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


# --- ParentChild -----------------------------------------------------------------------


class Trim(BaseModel):
    """One trim of a car."""

    power_ps: int = Field(description="Power", unit="PS")
    doors: int = Field(description="Number of doors")


class CarModel(BaseModel):
    """A car model page."""

    model: str = Field(description="Model name")
    trims: list[Trim] = Field(description="Trims")


class Engine(BaseModel):
    size_cc: int = Field(description="Engine size", unit="cc")


class OneEngine(BaseModel):
    model: str = Field(description="Model name")
    engine: Engine | None = Field(default=None, description="Engine")


class TwoNested(BaseModel):
    trims: list[Trim] = Field(description="Trims")
    engine: Engine = Field(description="Engine")


CAR = SchemaSpec.from_model(CarModel)

MODEL_PAGE = """<h1>Kestrova</h1>
<p>Every Kestrova has 5 doors.</p>
<table>
<thead><tr><th></th><th>SE</th><th>SE L</th></tr></thead>
<tbody>
<tr><th>Power</th><td>150PS</td><td>180PS</td></tr>
<tr><th>Doors</th><td>-</td><td>3</td></tr>
<tr><th>Warranty</th><td colspan="2">3 years</td></tr>
</tbody>
</table>"""


async def test_parent_child_splits_table_columns_into_children_asking_nothing() -> None:
    parsed = await parse(MODEL_PAGE)
    fake = FakeJev(strict=True)
    parent, se, se_l = await ParentChild().resolve(parsed, CAR, fake.client())
    assert fake.calls == []

    assert (parent.label, parent.parent, parent.field) == ("document", None, None)
    # Row labels head every child's column, so they're the parent's (children inherit).
    assert texts(parsed, parent.statement_ids) == [
        "Kestrova",
        "Every Kestrova has 5 doors.",
        "Power",
        "Doors",
        "Warranty",
    ]
    assert parent.shared_statement_ids == []
    assert (se.label, se.parent, se.field) == ("SE", "document", "trims")
    assert (se_l.label, se_l.parent, se_l.field) == ("SE L", "document", "trims")
    assert texts(parsed, se.statement_ids) == [
        "SE",
        "Power · SE: 150PS",
        "Doors · SE: -",
        "Warranty · SE / SE L: 3 years",
    ]
    assert texts(parsed, se_l.statement_ids) == [
        "SE L",
        "Power · SE L: 180PS",
        "Doors · SE L: 3",
        "Warranty · SE / SE L: 3 years",  # a cell spanning both columns is on both
    ]
    # The split table is on no scope; the heading and paragraph are the parent's.
    assert se.component_ids == se_l.component_ids == []
    assert len(parent.component_ids) == 2


async def test_parent_child_can_take_a_tables_rows_as_the_children() -> None:
    parsed = await parse(MODEL_PAGE)
    parent, *children = await ParentChild(children="table_rows").resolve(
        parsed, CAR, FakeJev(strict=True).client()
    )
    assert [c.label for c in children] == ["Power", "Doors", "Warranty"]
    assert texts(parsed, children[0].statement_ids) == [
        "Power",
        "Power · SE: 150PS",
        "Power · SE L: 180PS",
    ]
    assert texts(parsed, parent.statement_ids) == [
        "Kestrova",
        "Every Kestrova has 5 doors.",
        "SE",
        "SE L",
    ]


async def test_parent_child_gives_a_cell_spanning_rows_to_each_row_child() -> None:
    parsed = await parse(
        "<table><tr><th></th><th>Warranty</th></tr>"
        '<tr><th>SE</th><td rowspan="2">3 years</td></tr>'
        "<tr><th>SE L</th></tr></table>"
    )
    _, se, se_l = await ParentChild(children="table_rows").resolve(
        parsed, CAR, FakeJev(strict=True).client()
    )
    assert (se.label, se_l.label) == ("SE", "SE L")
    warranty = "SE · SE L · Warranty: 3 years"
    assert texts(parsed, se.statement_ids) == ["SE", warranty]
    assert texts(parsed, se_l.statement_ids) == [warranty, "SE L"]


async def test_parent_child_takes_a_single_column_as_one_child() -> None:
    parsed = await parse(
        "<table><tr><th></th><th>SE</th></tr><tr><th>Power</th><td>150PS</td></tr></table>"
    )
    parent, se = await ParentChild().resolve(parsed, CAR, FakeJev(strict=True).client())
    assert texts(parsed, se.statement_ids) == ["SE", "Power · SE: 150PS"]
    assert texts(parsed, parent.statement_ids) == ["Power"]


async def test_parent_child_joins_the_same_column_label_across_tables() -> None:
    table = (
        "<table><tr><th></th><th>SE</th><th>SE L</th></tr>"
        "<tr><th>{0}</th><td>{1}</td><td>{2}</td></tr></table>"
    )
    parsed = await parse(table.format("Power", "150PS", "180PS") + table.format("Doors", "5", "3"))
    _, se, se_l = await ParentChild().resolve(parsed, CAR, FakeJev(strict=True).client())
    # Each table states the trim in its header; the child holds both.
    assert texts(parsed, se.statement_ids) == ["SE", "Power · SE: 150PS", "SE", "Doors · SE: 5"]
    assert texts(parsed, se_l.statement_ids) == [
        "SE L",
        "Power · SE L: 180PS",
        "SE L",
        "Doors · SE L: 3",
    ]


async def test_parent_child_takes_components_of_a_type_as_children() -> None:
    parsed = await parse(
        "<h1>Kestrova</h1><p>Every Kestrova has 5 doors.</p>"
        "<h2>SE</h2><p>150PS.</p><h2>SE L</h2><p>180PS.</p><p>3 doors.</p>"
    )
    parent, se, se_l = await ParentChild(children="section").resolve(
        parsed, CAR, FakeJev(strict=True).client()
    )
    assert texts(parsed, parent.statement_ids) == ["Kestrova", "Every Kestrova has 5 doors."]
    assert texts(parsed, se.statement_ids) == ["SE", "150PS."]
    assert texts(parsed, se_l.statement_ids) == ["SE L", "180PS.", "3 doors."]
    assert len(se_l.component_ids) == 3  # its heading and two paragraphs


async def test_parent_child_numbers_repeated_labels_and_keeps_the_parents_free() -> None:
    parsed = await parse(
        "<p>Trims:</p><ul><li>SE: 150PS</li><li>SE: 160PS</li><li>Kestrova: 180PS</li></ul>"
    )
    parent, *children = await ParentChild(children="list_item", label="Kestrova").resolve(
        parsed, CAR, FakeJev(strict=True).client()
    )
    assert parent.label == "Kestrova"
    assert [c.label for c in children] == ["SE: 150PS", "SE: 160PS", "Kestrova: 180PS"]
    parsed = await parse("<ul><li>SE</li><li>SE</li><li>Kestrova</li></ul>")
    _, *children = await ParentChild(children="list_item", label="Kestrova").resolve(
        parsed, CAR, FakeJev(strict=True).client()
    )
    assert [c.label for c in children] == ["SE", "SE (2)", "Kestrova (2)"]


def _component(cid: str, kind: ComponentType, *children: Component, text: str = "") -> Component:
    return Component(
        id=cid, type=kind, text=text, children=list(children), location=DomLocation(dom_path="/")
    )


def _statements(root: Component) -> dict[str, Statement]:
    return {
        f"s{c.id}": Statement(
            id=f"s{c.id}", text=c.text, kind="sentence", component_id=c.id, location=c.location
        )
        for c in root.walk()
        if c.text
    }


async def test_parent_child_looks_inside_a_lone_wrapping_component() -> None:
    def trim(n: int) -> Component:
        return _component(
            f"t{n}",
            "section",
            _component(f"h{n}", "heading", text=f"Trim {n}"),
            _component(f"p{n}", "paragraph", text=f"{n}00PS."),
        )

    root = _component(
        "root",
        "section",
        _component("intro", "paragraph", text="Kestrova."),
        _component(
            "main", "section", _component("lead", "paragraph", text="Two trims."), trim(1), trim(2)
        ),
    )
    parsed = ParsedDocument(
        document=Document.from_bytes(b"<p/>"), root=root, statements=_statements(root)
    )
    parent, one, two = await ParentChild(children="section").resolve(
        parsed, CAR, FakeJev(strict=True).client()
    )
    assert texts(parsed, parent.statement_ids) == ["Kestrova.", "Two trims."]
    assert (one.label, two.label) == ("Trim 1", "Trim 2")
    assert texts(parsed, two.statement_ids) == ["Trim 2", "200PS."]


@pytest.mark.parametrize(
    "page",
    [
        "<header><a href='/'>Menu</a></header><main><h1>Kestrova</h1><p>Intro.</p>"
        "<h2>SE</h2><p>150PS.</p><h2>SE L</h2><p>180PS.</p></main>",
        "<article><h1>Kestrova</h1><p>Intro.</p><section><h2>SE</h2><p>150PS.</p></section>"
        "<section><h2>SE L</h2><p>180PS.</p></section></article>",
        "<main><section><h1>Kestrova</h1><p>Intro.</p><section><h2>SE</h2><p>150PS.</p>"
        "</section><section><h2>SE L</h2><p>180PS.</p></section></section></main>",
    ],
)
async def test_parent_child_looks_inside_the_section_under_the_page_title(page: str) -> None:
    parsed = await parse(page)
    parent, se, se_l = await ParentChild(children="section").resolve(
        parsed, CAR, FakeJev(strict=True).client()
    )
    assert texts(parsed, parent.statement_ids)[-2:] == ["Kestrova", "Intro."]
    assert texts(parsed, se.statement_ids) == ["SE", "150PS."]
    assert texts(parsed, se_l.statement_ids) == ["SE L", "180PS."]


async def test_parent_child_takes_a_lone_headed_component_as_the_only_child() -> None:
    parsed = await parse(
        "<h1>Kestrova</h1><p>A family car.</p>"
        "<section><h2>SE</h2><p>The SE.</p>"
        "<section><h3>Performance</h3><p>150PS.</p></section>"
        "<section><h3>Dimensions</h3><p>4.2 m.</p></section></section>"
    )
    parent, se = await ParentChild(children="section").resolve(
        parsed, CAR, FakeJev(strict=True).client()
    )
    assert texts(parsed, parent.statement_ids) == ["Kestrova", "A family car."]
    assert se.label == "SE"
    assert texts(parsed, se.statement_ids) == [
        "SE",
        "The SE.",
        "Performance",
        "150PS.",
        "Dimensions",
        "4.2 m.",
    ]


async def test_parent_child_keeps_a_lone_headed_child_when_the_title_was_gated_out() -> None:
    parsed = await parse(
        "<h1>Kestrova</h1><p>A family car.</p>"
        "<section><h2>SE</h2><p>The SE.</p>"
        "<section><h3>Performance</h3><p>150PS.</p></section>"
        "<section><h3>Dimensions</h3><p>4.2 m.</p></section></section>"
    )
    title = [c.id for c in parsed.root.children[:2]]
    view = parsed.restricted_to(c.id for c in parsed.root.walk() if c.id not in title)
    [parent, se] = await ParentChild(children="section").resolve(
        view, CAR, FakeJev(strict=True).client()
    )
    assert parent.statement_ids == []
    assert se.label == "SE"


async def test_parent_child_without_children_is_the_parent_alone() -> None:
    parsed = await parse("<h1>Kestrova</h1><p>Every Kestrova has 5 doors.</p>")
    [parent] = await ParentChild().resolve(parsed, CAR, FakeJev(strict=True).client())
    assert texts(parsed, parent.statement_ids) == ["Kestrova", "Every Kestrova has 5 doors."]


async def test_parent_child_on_a_schema_without_the_field_is_a_single_entity() -> None:
    parsed = await parse(MODEL_PAGE)
    jev = FakeJev(strict=True).client()
    single = await SingleEntity().resolve(parsed, VEHICLE, jev)
    assert await ParentChild().resolve(parsed, VEHICLE, jev) == single
    assert await ParentChild(field="trims").resolve(parsed, VEHICLE, jev) == single


async def test_parent_child_fills_the_named_field() -> None:
    parsed = await parse(MODEL_PAGE)
    spec = SchemaSpec.from_model(TwoNested)
    _, se, _ = await ParentChild(field="engine").resolve(parsed, spec, FakeJev().client())
    assert se.field == "engine"


async def test_parent_child_needs_to_know_which_nested_field_holds_the_children() -> None:
    parsed = await parse(MODEL_PAGE)
    jev = FakeJev().client()
    with pytest.raises(ChildFieldError, match="several nested model fields"):
        await ParentChild().resolve(parsed, SchemaSpec.from_model(TwoNested), jev)
    with pytest.raises(ChildFieldError, match=r"CarModel\.model is not a nested model field"):
        await ParentChild(field="model").resolve(parsed, CAR, jev)


class Extra(BaseModel):
    tags: dict[str, str]


class PageWithExtra(BaseModel):
    title: str = Field(description="Title")
    extra: Extra | None = Field(default=None, description="Extra")


async def test_parent_child_rejects_a_nested_model_jevex_cannot_extract() -> None:
    c = await entity_ctx(MODEL_PAGE, PageWithExtra)
    with pytest.raises(ChildFieldError, match=r"PageWithExtra\.extra can't hold children: "):
        await EntityStage(resolver=ParentChild()).run(c)


def test_parent_child_rejects_an_unknown_place_for_children() -> None:
    with pytest.raises(ValueError, match="children must be one of"):
        ParentChild(children="paragraph")  # pyright: ignore[reportArgumentType]


def test_parent_child_satisfies_the_protocol() -> None:
    assert isinstance(ParentChild(), EntityResolver)


async def entity_ctx(html: str, *models: type[BaseModel]) -> Context:
    parsed = await parse(html)
    c = Context.create(
        parsed.document, [SchemaSpec.from_model(m) for m in models], FakeJev().client()
    )
    c.parsed = parsed
    return c


async def test_entity_stage_gives_children_a_run_of_their_own() -> None:
    c = await entity_ctx(MODEL_PAGE, CarModel)
    await EntityStage(resolver=ParentChild()).run(c)
    run, child = c.schemas["CarModel"], c.schemas["CarModel.trims"]
    assert [s.label for s in run.scopes] == ["document"]
    assert (child.parent, child.parent_field) == ("CarModel", "trims")
    assert child.spec.model is Trim
    assert child.component_ids is None
    # The parent's scope comes along, so the nested fields are looked for there too.
    assert [(s.label, s.parent) for s in child.scopes] == [
        ("document", None),
        ("SE", "document"),
        ("SE L", "document"),
    ]
    assert child.scopes[0].statement_ids == run.scopes[0].statement_ids
    assert c.events == []  # every statement is on a scope


async def test_entity_stage_keeps_one_child_for_a_field_holding_one_model() -> None:
    c = await entity_ctx(MODEL_PAGE, OneEngine)
    await EntityStage(resolver=ParentChild()).run(c)
    child = c.schemas["OneEngine.engine"]
    assert [s.label for s in child.scopes] == ["document", "SE"]
    [event] = c.events
    assert (event.kind, event.data) == ("extra_children", {"labels": ["SE L"]})
    assert event.message == "OneEngine.engine holds one record: kept 'SE', left out 1 more"


@pytest.mark.parametrize(
    ("scopes", "message"),
    [
        (
            [EntityScope(label="p"), EntityScope(label="c", parent="p", field="model")],
            r"CarModel\.model is not a nested model field",
        ),
        (
            [EntityScope(label="p"), EntityScope(label="c", parent="q", field="trims")],
            "names parent 'q', which isn't one of its scopes",
        ),
        (
            [EntityScope(label="p"), EntityScope(label="p", parent="p", field="trims")],
            "repeated entity labels",
        ),
    ],
)
async def test_entity_stage_rejects_children_it_cannot_extract(
    scopes: list[EntityScope], message: str
) -> None:
    @dataclass
    class Given:
        async def resolve(
            self, parsed: ParsedDocument, schema: SchemaSpec, jev: object
        ) -> list[EntityScope]:
            return scopes

    c = await entity_ctx(MODEL_PAGE, CarModel)
    with pytest.raises(InvalidScopeError, match=message):
        await EntityStage(resolver=Given()).run(c)


# --- ParentChild, end to end -----------------------------------------------------------


def pick(name: str) -> Callable[[Choice], str]:
    """Choose ``name`` where it's offered, else "none" (the parent's and the children's
    categorise Choices go out together)."""
    return lambda q: name if name in q.options else "none"


def car_jev() -> FakeJev:
    return (
        FakeJev(strict=True)
        .noul("Does this document", p=0.95)
        .noul("Does this section", p=0.9)
        .choice("Which detail", "none")
        .choice("Which detail", pick("model"), state="Kestrova")
        .choice("Which detail", pick("doors"), state="oors")
        .choice("Which detail", pick("power_ps"), state="Power")
        .choice("Which of these", first_option, confidence=0.9)
    )


async def extract_car(page: str, fake: FakeJev, **kwargs: Any) -> tuple[ExtractionResult, FakeJev]:
    pipeline = default_pipeline().replace("entities", EntityStage(resolver=ParentChild()))
    async with Extractor([CarModel], jev=fake.client(), pipeline=pipeline, **kwargs) as ex:
        result = await ex.extract(Document.from_bytes(page.encode(), content_type="text/html"))
    return result, fake


async def test_children_inherit_the_parents_values_and_their_own_win() -> None:
    result, fake = await extract_car(MODEL_PAGE, car_jev())

    car = result.one(CarModel)
    assert car.entity == "document"
    assert car.record.model_dump() == {
        "model": "Kestrova",
        "trims": [{"power_ps": 150, "doors": 5}, {"power_ps": 180, "doors": 3}],
    }
    se, se_l = car.children["trims"]
    assert (se.entity, se_l.entity) == ("SE", "SE L")
    assert se.record is car.record.trims[0]
    # "Every Kestrova has 5 doors." is the parent's: SE has no doors of its own, so it
    # inherits them; SE L states its own.
    assert se.meta.doors.shared
    assert se.meta.doors.value == 5
    assert not se_l.meta.doors.shared
    assert not se.meta.power_ps.shared
    assert car.meta.trims.value == [{"power_ps": 150, "doors": 5}, {"power_ps": 180, "doors": 3}]
    assert car.strict() == CarModel(
        model="Kestrova", trims=[Trim(power_ps=150, doors=5), Trim(power_ps=180, doors=3)]
    )
    assert result.meta.active_schemas == ["CarModel"]

    # The parent's and the children's categorise Choices share a request; table cells are
    # asked only about the nested model's fields.
    categorised = {
        str(call.state): sorted(call.questions)
        for call in fake.calls
        if "CarModel.trims" in call.questions
    }
    assert categorised["{'statement': 'Kestrova'}"] == ["CarModel", "CarModel.trims"]
    assert categorised["{'statement': 'Power · SE: 150PS', 'section': 'Kestrova'}"] == [
        "CarModel.trims"
    ]
    [child_choice] = [
        q
        for call in fake.calls
        for key, q in call.questions.items()
        if key == "CarModel.trims" and call.state == {"statement": "Kestrova"}
    ]
    assert child_choice == Choice(
        instructions="Which detail does this statement state?",
        options={
            "power_ps": "Power (PS)",
            "doors": "Number of doors",
            "none": "None of these details",
        },
    )


async def test_children_are_categorised_only_for_the_nested_fields_their_section_passed() -> None:
    # The intro passes the gate for doors but not power, so its statement is offered doors
    # alone; the table passes for both.
    fake = car_jev().noul("contain the power (PS)?", p=0.1, state="Every Kestrova")
    result, fake = await extract_car(MODEL_PAGE, fake)

    [gate] = [
        call.questions
        for call in fake.calls
        if "CarModel.trims.doors" in call.questions and "Every Kestrova" in str(call.state)
    ]
    assert {k: q.instructions for k, q in gate.items()} == {
        "CarModel.model": "Does this section contain the model name?",
        "CarModel.trims": (
            "Does this section contain the power (PS) or number of doors of the trims?"
        ),
        "CarModel.trims.power_ps": "Does this section contain the power (PS)?",
        "CarModel.trims.doors": "Does this section contain the number of doors?",
    }
    offered = {
        str(call.state): list(q.options)
        for call in fake.calls
        for key, q in call.questions.items()
        if key == "CarModel.trims" and isinstance(q, Choice)
    }
    intro = "{'statement': 'Every Kestrova has 5 doors.', 'section': 'Kestrova'}"
    assert offered[intro] == ["doors", "none"]
    cell = "{'statement': 'Power · SE: 150PS', 'section': 'Kestrova'}"
    assert offered[cell] == ["power_ps", "doors", "none"]
    se, _ = result.one(CarModel).children["trims"]
    assert se.record.model_dump() == {"power_ps": 150, "doors": 5}


async def test_child_records_are_partial_until_complete() -> None:
    page = MODEL_PAGE.replace("<p>Every Kestrova has 5 doors.</p>", "")
    result, _ = await extract_car(page, car_jev())
    car = result.one(CarModel)
    se, se_l = car.children["trims"]
    assert se.record.model_dump() == {"power_ps": 150, "doors": None}
    assert not se.complete
    assert se_l.complete
    assert not car.complete
    assert car.to_dict()["children"]["trims"][1] == se_l.to_dict()
    assert se_l.to_dict()["children"] == {}


async def test_child_thresholds_are_keyed_by_the_nested_schema_name() -> None:
    fake = car_jev().choice("Which of these", first_option, confidence=0.5, state="Power")
    result, _ = await extract_car(MODEL_PAGE, fake, thresholds={"CarModel.trims.power_ps": 0.8})
    car = result.one(CarModel)
    se, _ = car.children["trims"]
    assert se.meta.power_ps.filtered
    assert car.record.trims is not None
    assert car.record.trims[0].power_ps is None
    assert car.meta.trims.value[0] == {"doors": 5}


async def test_a_parent_with_only_children_is_still_a_record() -> None:
    page = "<table><tr><th></th><th>SE</th></tr><tr><th>Power</th><td>150PS</td></tr></table>"
    result, _ = await extract_car(page, car_jev())
    car = result.one(CarModel)
    assert car.record.model_dump() == {"model": None, "trims": [{"power_ps": 150, "doors": None}]}


class Variant(BaseModel):
    """One variant of a car."""

    power_ps: int = Field(description="Power", unit="PS")
    variants: list["Variant"] = Field(default_factory=list, description="Variants")


class Range(BaseModel):
    """A car range."""

    name: str = Field(description="Range name")
    trims: list["RangeTrim"] = Field(description="Trims")


class RangeTrim(BaseModel):
    """One trim of a range."""

    power_ps: int = Field(description="Power", unit="PS")
    range: Range | None = Field(default=None, description="Range")


Range.model_rebuild()

SELF_NESTED_PAGE = (
    "<table><tr><th></th><th>SE</th><th>SE L</th></tr>"
    "<tr><th>Power</th><td>150PS</td><td>180PS</td></tr></table>"
)


@pytest.mark.parametrize("model", [Variant, Range], ids=["self-nested", "mutually-nested"])
async def test_children_of_recursive_models_fill_the_parents_record(
    model: type[BaseModel],
) -> None:
    fake = (
        FakeJev()
        .noul("Does this", p=0.95)
        .choice("Which detail", pick("power_ps"), state="Power")
        .choice("Which of these", first_option, confidence=0.9)
    )
    pipeline = default_pipeline().replace("entities", EntityStage(resolver=ParentChild()))
    async with Extractor([model], jev=fake.client(), pipeline=pipeline) as ex:
        result = await ex.extract(
            Document.from_bytes(SELF_NESTED_PAGE.encode(), content_type="text/html")
        )
    parent = result.one(model)
    field = "variants" if model is Variant else "trims"
    kids = [{"power_ps": 150}, {"power_ps": 180}]
    assert parent.meta[field].error is None
    assert parent.meta[field].value == kids
    assert parent.record.model_dump(exclude_none=True) == {field: kids}
    assert [type(k) for k in getattr(parent.record, field)] == [
        type(c.record) for c in parent.children[field]
    ]


async def test_a_field_holding_one_model_gets_its_first_child() -> None:
    fake = (
        FakeJev()
        .noul("Does this", p=0.95)
        .choice("Which detail", pick("size_cc"), state="Engine")
        .choice("Which of these", first_option, confidence=0.9)
    )
    page = (
        "<table><tr><th></th><th>SE</th><th>SE L</th></tr>"
        "<tr><th>Engine</th><td>1498cc</td><td>1968cc</td></tr></table>"
    )
    pipeline = default_pipeline().replace("entities", EntityStage(resolver=ParentChild()))
    async with Extractor([OneEngine], jev=fake.client(), pipeline=pipeline) as ex:
        result = await ex.extract(Document.from_bytes(page.encode(), content_type="text/html"))
    car = result.one(OneEngine)
    assert car.record.model_dump() == {"model": None, "engine": {"size_cc": 1498}}
    assert car.meta.engine.value == {"size_cc": 1498}
    assert [e.kind for e in result.meta.events] == ["extra_children"]


# --- values found for the whole document (embedded data) ------------------------------


def test_match_label_prefers_an_exact_name_then_the_longest_label_inside_one() -> None:
    labels = ["SE", "SE L", "Sport"]
    assert match_label(["SE"], labels) == "SE"
    assert match_label(["se-l"], labels) == "SE L"  # case and punctuation are ignored
    assert match_label(["Kestrova SE L 1.5"], labels) == "SE L"
    assert match_label(["Kestrova", "SE"], labels) == "SE"  # any of an item's names
    # A name inside a label isn't enough: "Kestrova" is inside every heading here.
    assert match_label(["Kestrova"], ["Kestrova SE", "Kestrova SE L"]) is None
    assert match_label(["SE"], ["Kestrova SE", "Kestrova SE L"]) is None
    assert match_label(["SE and Sport"], labels) is None  # a tie names nobody
    assert match_label(["Delivery"], labels) is None
    assert match_label([], labels) is None


def structured(value: object, sid: str) -> FieldMeta:
    return FieldMeta(value=value, method="structured", source=Source(statement_id=sid))


def item(path: str, names: tuple[str, ...], sids: set[str], **values: FieldMeta) -> StructuredItem:
    return StructuredItem(
        path=path, names=names, statement_ids=frozenset(sids), fields={"Car": values}
    )


class Car(BaseModel):
    model: str = Field(description="Model name")
    price: int = Field(description="Price")
    doors: int = Field(description="Doors")


def car_run(*labels: str) -> SchemaRun:
    run = SchemaRun(SchemaSpec.from_model(Car))
    run.scopes = [EntityScope(label=label) for label in labels]
    return run


def test_document_values_go_to_the_entity_an_item_names_and_the_rest_are_shared() -> None:
    run = car_run("SE", "SE L", "Sport")
    run.fields[SINGLE_ENTITY_LABEL] = {
        "model": structured("Kestrova", "s0"),
        "price": structured(26995, "s2"),  # the first offer's, which names SE L
        "doors": FieldMeta(method="structured", error="no value"),
    }
    run.structured_items = [
        item("offers[0]", ("Kestrova SE L",), {"s1", "s2"}, price=structured(26995, "s2")),
        item("offers[1]", ("Delivery",), {"s3", "s4"}, price=structured(750, "s4")),
        item("offers[2]", ("SE",), {"s5", "s6"}, price=structured(24995, "s6")),
    ]
    assert place_document_values(run) == (2, 4)
    assert SINGLE_ENTITY_LABEL not in run.fields
    se, se_l, sport = (run.fields[label] for label in ("SE", "SE L", "Sport"))
    assert (se["price"].value, se["price"].shared) == (24995, False)
    assert (se_l["price"].value, se_l["price"].shared) == (26995, False)
    # Sport gets no named offer: the price from the one naming no entity, shared.
    assert (sport["price"].value, sport["price"].shared) == (750, True)
    assert all(f["model"].value == "Kestrova" and f["model"].shared for f in (se, se_l, sport))
    assert all(not f["doors"].found for f in (se, se_l, sport))  # the error goes along


def test_a_value_from_outside_every_item_is_shared_before_an_unmatched_items() -> None:
    run = car_run("SE", "SE L")
    run.fields[SINGLE_ENTITY_LABEL] = {"price": structured(24995, "s1")}
    run.structured_items = [
        item("offers[0]", ("SE",), {"s1"}, price=structured(24995, "s1")),
        item("offers[1]", ("Delivery",), {"s2"}, price=structured(750, "s2")),
    ]
    run.structured_rest = {"price": structured(19995, "s9")}
    place_document_values(run)
    assert run.fields["SE"]["price"].value == 24995
    assert (run.fields["SE L"]["price"].value, run.fields["SE L"]["price"].shared) == (
        19995,
        True,
    )


def test_a_value_only_matched_items_give_isnt_shared() -> None:
    run = car_run("SE", "SE L", "Sport")
    run.fields[SINGLE_ENTITY_LABEL] = {"price": structured(24995, "s1")}
    run.structured_items = [
        item("trims[0]", ("SE",), {"s1"}, price=structured(24995, "s1")),
        # Inside a matched item, so not loose, even though it names no entity.
        item("trims[0].extras[0]", ("Delivery",), {"s1"}, price=structured(24995, "s1")),
        item("trims[1]", ("SE L",), {"s2"}),  # names SE L but gives no price
    ]
    assert place_document_values(run) == (1, 0)
    assert "price" not in run.fields.get("SE L", {})
    assert "Sport" not in run.fields


def test_with_one_entity_the_documents_values_are_its_own() -> None:
    run = car_run("listing")
    run.fields[SINGLE_ENTITY_LABEL] = {"model": structured("Kestrova", "s0")}
    assert place_document_values(run) == (1, 0)
    assert run.fields == {"listing": {"model": structured("Kestrova", "s0")}}


def test_nothing_moves_when_the_document_is_a_scope_or_found_nothing() -> None:
    run = car_run(SINGLE_ENTITY_LABEL)
    run.fields[SINGLE_ENTITY_LABEL] = {"model": structured("Kestrova", "s0")}
    assert place_document_values(run) is None
    assert list(run.fields) == [SINGLE_ENTITY_LABEL]
    assert place_document_values(car_run("SE", "SE L")) is None
    empty = car_run()
    empty.fields[SINGLE_ENTITY_LABEL] = {"model": structured("Kestrova", "s0")}
    assert place_document_values(empty) is None


async def test_the_entity_stage_places_document_values_and_reports_it() -> None:
    c = ctx(Listing, with_parsed=False)
    c.schemas["Listing"].fields[SINGLE_ENTITY_LABEL] = {"price": structured(18495, "s0")}
    await EntityStage(resolver=SingleEntity(label="listing")).run(c)
    assert c.schemas["Listing"].fields["listing"]["price"].value == 18495
    [event] = c.events
    assert (event.kind, event.data) == ("document_values_placed", {"matched": 1, "shared": 0})
    assert event.message == "Listing: 1 value(s) matched to an entity, 0 shared by every entity"


class TrimSpec(BaseModel):
    """A car's technical specification."""

    model: str = Field(description="Model name")
    price: int = Field(description="Price", unit="GBP")
    power_ps: int = Field(description="Power", unit="PS")


JSON_LD_TABLE_PAGE = """<html><head><script type="application/ld+json">
{"@context": "https://schema.org", "@type": "Car", "model": "Kestrova",
 "offers": [{"@type": "Offer", "name": "Kestrova SE L", "price": 26995},
            {"@type": "Offer", "name": "Delivery", "price": 750}]}
</script></head><body>
<h1>Kestrova</h1>
<table>
<thead><tr><th></th><th>SE</th><th>SE L</th></tr></thead>
<tbody>
<tr><th>Power</th><td>150PS</td><td>180PS</td></tr>
<tr><th>Price</th><td>£24,995</td><td>£26,495</td></tr>
</tbody>
</table></body></html>"""


def json_ld_table_jev() -> FakeJev:
    def pick(name: str) -> Callable[[Choice], str]:
        return lambda q: name if name in q.options else "none"

    return (
        FakeJev()
        .noul("Does this document", p=0.95)
        .noul("Does this section", p=0.9)
        .noul("name a separate", p=0.9)
        .noul("name a separate", p=0.1, state="Power")
        .choice("key path", "none")
        .choice('key path "model"', pick("model"))
        .choice('key path "offers[].price"', pick("price"))
        .choice("Which detail does this statement", "none")
        .choice("Which detail does this statement", "power_ps", state="Power")
        .choice("Which detail does this statement", "price", state="Price")
        .choice("Which of these", first_option, confidence=0.9)
    )


async def extract_json_ld_table(
    mode: StructuredMode, fake: FakeJev | None = None
) -> ExtractionResult:
    fake = fake or json_ld_table_jev()
    pipeline = (
        default_pipeline()
        .replace("structured", StructuredStage(mode=mode))
        .replace("entities", EntityStage(resolver=MultiEntity()))
    )
    doc = Document.from_bytes(JSON_LD_TABLE_PAGE.encode(), content_type="text/html")
    async with Extractor([TrimSpec], jev=fake.client(), pipeline=pipeline) as ex:
        return await ex.extract(doc)


async def test_json_ld_on_a_comparison_table_page_fills_the_trims_not_a_document_record() -> None:
    fake = json_ld_table_jev()
    result = await extract_json_ld_table("fill_gaps", fake)
    se, se_l = result.records
    assert (se.entity, se_l.entity) == ("SE", "SE L")  # no "document" record
    assert se.record.model_dump() == {"model": "Kestrova", "price": 24995, "power_ps": 150}
    assert se_l.record.model_dump() == {"model": "Kestrova", "price": 26995, "power_ps": 180}
    # The model is the page's, shared; SE L's offer names it, so its price is its own.
    assert se.meta.model.shared
    assert se_l.meta.model.shared
    assert se.meta.model.method == "structured"
    assert (se_l.meta.price.method, se_l.meta.price.shared) == ("structured", False)
    # SE's shared price (the delivery offer's) lost to its own table cell.
    assert (se.meta.price.method, se.meta.price.shared) == ("generator", False)
    assert se.meta.price.conflicts == []
    # Only SE's price was still wanted from the table.
    prices = [c.state for c in fake.calls if "Which of these is the price" in str(c.questions)]
    assert prices == [{"statement": "Price · SE: £24,995", "section": "Kestrova"}]
    [placed] = [e for e in result.meta.events if e.kind == "document_values_placed"]
    assert placed.message == ("TrimSpec: 1 value(s) matched to an entity, 3 shared by every entity")


async def test_in_merge_mode_a_trims_own_value_beats_the_shared_one_as_a_conflict() -> None:
    result = await extract_json_ld_table("merge")
    se, se_l = result.records
    assert (se.meta.price.value, se.meta.price.method) == (24995, "generator")
    [conflict] = se.meta.price.conflicts
    assert (conflict.value, conflict.method) == (750, "structured")
    # SE L's own embedded price is certain, so it beats the table's.
    assert (se_l.meta.price.value, se_l.meta.price.method) == (26995, "structured")
    [conflict] = se_l.meta.price.conflicts
    assert (conflict.value, conflict.method) == (26495, "generator")


def test_an_object_holding_matched_objects_isnt_matched_itself() -> None:
    run = car_run("Sport", "Sport Plus", "GT")
    run.fields[SINGLE_ENTITY_LABEL] = {
        "model": structured("Kestrova", "s1"),
        "price": structured(30000, "s3"),
    }
    run.structured_items = [
        # "Kestrova Sport Tourer" names Sport, but holds the offers that name each trim.
        item(
            "vehicles[0]",
            ("Kestrova Sport Tourer",),
            {"s0", "s1", "s2", "s3", "s4", "s5"},
            model=structured("Kestrova", "s1"),
            price=structured(30000, "s3"),
        ),
        item("vehicles[0].offers[0]", ("Sport Plus",), {"s2", "s3"}, price=structured(30000, "s3")),
        item("vehicles[0].offers[1]", ("Sport",), {"s4", "s5"}, price=structured(25000, "s5")),
    ]
    place_document_values(run)
    assert run.fields["Sport"]["price"].value == 25000
    assert run.fields["Sport Plus"]["price"].value == 30000
    assert "price" not in run.fields["GT"]
    assert all(run.fields[label]["model"].shared for label in ("Sport", "Sport Plus", "GT"))

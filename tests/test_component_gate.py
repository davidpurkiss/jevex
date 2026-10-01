from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

import pytest
from pydantic import BaseModel

from jevex import (
    BoilerplateCleaner,
    Component,
    ComponentGateStage,
    Context,
    Document,
    DomLocation,
    EntityStage,
    Field,
    NoulComponentGate,
    Questions,
    SchemaSpec,
    Statement,
    gate_units,
)
from jevex.entities import EntityScope
from jevex.extractor import default_pipeline
from jevex.interfaces import ComponentGate, ParsedDocument
from jevex.jev import (
    ChoiceAnswer,
    JevClient,
    JevResponse,
    JSONContent,
    NoulAnswer,
    Question,
    ScoreAnswer,
    UnexpectedAnswerError,
)
from jevex.layout import MAX_SECTION_CHARS, TableCell
from jevex.layout_html import HtmlLayoutParser
from jevex.testing import FakeJev


class Car(BaseModel):
    """A car's specification."""

    price: Decimal = Field(description="Price", unit="GBP")
    power_kw: float = Field(description="Engine power", unit="kW", group="performance")
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s", group="performance")


class Book(BaseModel):
    """A book."""

    title: str = Field(
        description="Title", questions=Questions(component_gate="Is this the book's title?")
    )


def comp(
    type_: str,
    text: str = "",
    cid: str = "c",
    *children: Component,
    trail: list[str] | None = None,
) -> Component:
    return Component.model_validate(
        {
            "id": cid,
            "type": type_,
            "text": text,
            "children": list(children),
            "heading_trail": trail or [],
            "location": DomLocation(dom_path=f"/{cid}"),
        }
    )


def page() -> Component:
    return comp(
        "section",
        "",
        "root",
        comp("heading", "Delmaro Kestrova", "h1"),
        comp("paragraph", "A roomy family hatchback.", "p1"),
        comp(
            "section",
            "",
            "s-price",
            comp("heading", "Pricing", "h2", trail=["Delmaro Kestrova"]),
            comp("paragraph", "From £24,995 on the road.", "p2", trail=["Pricing"]),
        ),
        comp(
            "section",
            "",
            "s-perf",
            comp("heading", "Performance", "h3", trail=["Delmaro Kestrova"]),
            comp(
                "list",
                "",
                "l1",
                comp("list_item", "Power: 110 kW", "li1", trail=["Performance"]),
                comp("list_item", "0-62 mph: 9.1 s", "li2", trail=["Performance"]),
                trail=["Performance"],
            ),
        ),
        comp("table", "Trim | Price\nSE | £24,995", "t1"),
        comp("paragraph", "Book a test drive today.", "p3"),
    )


# --- units ---------------------------------------------------------------------------


def test_units_are_each_containers_blocks_plus_tables() -> None:
    units = gate_units(page())
    assert [(u.id, u.component_ids) for u in units] == [
        ("root#0", ("h1", "p1")),
        ("s-price#0", ("h2", "p2")),
        ("s-perf#0", ("h3", "l1", "li1", "li2")),
        ("t1", ("t1",)),
        ("root#1", ("p3",)),
    ]
    perf = units[2]
    assert perf.text == "Performance\nPower: 110 kW\n0-62 mph: 9.1 s"
    assert perf.state() == {"content": perf.text, "section": "Delmaro Kestrova"}
    assert units[0].state() == {"content": "Delmaro Kestrova\nA roomy family hatchback."}


def test_a_unit_states_a_capped_heading_trail() -> None:
    trail = ["Kestrova", "word " * 1000, "Performance"]
    [unit] = gate_units(comp("section", "", "r", comp("paragraph", "9.1 s", "p", trail=trail)))
    section = unit.state()["section"]
    assert section.startswith("Kestrova › word word")
    assert section.endswith("… › Performance")
    assert len(section) <= MAX_SECTION_CHARS


def test_long_runs_are_chunked() -> None:
    paragraphs = [comp("paragraph", "x" * 40, f"p{i}") for i in range(5)]
    units = gate_units(comp("section", "", "root", *paragraphs), max_chars=100)
    assert [u.component_ids for u in units] == [("p0", "p1"), ("p2", "p3"), ("p4",)]


def test_an_oversized_paragraph_is_split_not_cut() -> None:
    sentences = [f"Sentence {i} is here." for i in range(30)] + ["It costs £24,995."]
    text = " ".join(sentences)
    units = gate_units(comp("section", "", "r", comp("paragraph", text, "p")), max_chars=100)
    assert len(units) > 1
    assert all(len(u.text) <= 100 for u in units)
    assert all(u.component_ids == ("p",) for u in units)
    assert "It costs £24,995." in units[-1].text
    # Nothing is lost: every sentence is whole in some unit.
    assert all(any(sent in u.text for u in units) for sent in sentences)


def test_a_sentence_longer_than_the_cap_becomes_overlapping_windows() -> None:
    text = "x" * 150 + " price 24995 " + "y" * 150
    units = gate_units(comp("section", "", "r", comp("paragraph", text, "p")), max_chars=100)
    assert all(len(u.text) <= 100 for u in units)
    assert any("24995" in u.text for u in units)


def table(cid: str, rows: int) -> Component:
    grid = [("Trim", "Price")] + [(f"Trim {r}", f"£{20000 + r}") for r in range(1, rows + 1)]
    cells = [
        {"row": r, "col": c, "text": text, "header": r == 0}
        for r, row in enumerate(grid)
        for c, text in enumerate(row)
    ]
    return Component.model_validate(
        {
            "id": cid,
            "type": "table",
            "text": "\n".join(" | ".join(row) for row in grid),
            "cells": cells,
            "location": DomLocation(dom_path=f"/{cid}"),
        }
    )


def test_an_oversized_table_is_split_by_rows_with_its_header_repeated() -> None:
    units = gate_units(comp("section", "", "r", table("t", 40)), max_chars=120)
    assert len(units) > 1
    assert [u.id for u in units][:2] == ["t:0", "t:1"]
    for u in units:
        assert len(u.text) <= 120
        assert u.text.startswith("Trim | Price\n")
        assert u.component_ids == ("t",)
    assert "Trim 40 | £20040" in units[-1].text


def test_an_oversized_tables_blank_row_is_a_body_row_without_gaps() -> None:
    t = table("t", 40)
    blank = [
        TableCell(row=41, col=0, text="Towing", header=True),
        TableCell(row=41, col=1, text=""),
    ]
    t = t.model_copy(update={"cells": [*t.cells, *blank]})
    units = gate_units(comp("section", "", "r", t), max_chars=120)
    lines = [line for u in units for line in u.text.split("\n")]
    assert all(u.text.startswith("Trim | Price\nTrim ") for u in units)
    assert lines.count("Towing") == 1
    assert units[-1].text.endswith("Trim 40 | £20040\nTowing")
    assert not any(line.startswith(" |") or line.endswith("| ") for line in lines)


def test_an_oversized_comparison_table_without_headers_repeats_its_inferred_header_row() -> None:
    grid = [("Spec", "SE", "GT")] + [(f"Spec {r}", f"{r}0 PS", f"{r}5 PS") for r in range(1, 30)]
    t = Component.model_validate(
        {
            "id": "t",
            "type": "table",
            "text": "\n".join(" | ".join(row) for row in grid),
            "cells": [
                {"row": r, "col": c, "text": text}
                for r, row in enumerate(grid)
                for c, text in enumerate(row)
            ],
            "location": DomLocation(dom_path="/t"),
        }
    )
    units = gate_units(comp("section", "", "r", t), max_chars=120)
    assert len(units) > 1
    assert all(u.text.startswith("Spec | SE | GT\nSpec ") for u in units)


def test_an_oversized_list_is_split_by_items() -> None:
    items = [comp("list_item", f"Feature number {i}", f"li{i}") for i in range(20)]
    units = gate_units(comp("section", "", "r", comp("list", "", "l", *items)), max_chars=100)
    assert len(units) > 1
    assert all(len(u.text) <= 100 for u in units)
    assert all(u.component_ids == ("l", *(f"li{i}" for i in range(20))) for u in units)
    lines = [line for u in units for line in u.text.split("\n")]
    assert lines == [f"Feature number {i}" for i in range(20)]


def test_a_heading_before_a_long_paragraph_leads_its_first_piece() -> None:
    root = comp(
        "section", "", "r", comp("heading", "H" * 50, "h"), comp("paragraph", "p " * 990, "p")
    )
    units = gate_units(root)
    assert all(u.component_ids == ("h", "p") for u in units)
    assert units[0].text.startswith("H" * 50 + "\n")
    assert all(len(u.text) <= 2000 for u in units)


def test_many_leading_headings_get_their_own_unit_and_no_unit_is_oversized() -> None:
    headings = [comp("heading", f"Heading number {i} " * 2, f"h{i}") for i in range(40)]
    root = comp("section", "", "r", *headings, table("t", 300))
    units = gate_units(root)
    assert all(len(u.text) <= 2000 for u in units)
    assert units[0].component_ids == tuple(f"h{i}" for i in range(40))
    assert all(u.component_ids == ("t",) for u in units[1:])


def test_headings_join_the_table_or_section_they_introduce() -> None:
    root = comp(
        "section",
        "",
        "root",
        comp("heading", "Delmaro Kestrova", "h1"),
        comp("section", "", "s", comp("paragraph", "A roomy hatchback.", "p")),
        comp("heading", "Specifications", "h2"),
        table("t", 2),
        comp("heading", "Trailing", "h3"),
    )
    units = gate_units(root)
    assert [(u.id, u.component_ids) for u in units] == [
        ("s#0", ("h1", "p")),
        ("t", ("h2", "t")),
        ("root#end", ("h3",)),
    ]
    assert units[0].text == "Delmaro Kestrova\nA roomy hatchback."
    assert units[1].text.startswith("Specifications\nTrim | Price")


def test_empty_blocks_make_no_unit_and_a_bare_root_is_one_unit() -> None:
    assert gate_units(comp("section", "", "r", comp("paragraph", "  ", "p"))) == []
    [unit] = gate_units(comp("paragraph", "Just text.", "p"))
    assert unit.component_ids == ("p",)


def scanned_page(alt: str = "") -> Component:
    """An image as the image stage leaves it: OCR paragraphs and headed sections below it."""
    return comp(
        "image",
        alt,
        "img",
        comp("paragraph", "Delmaro Kestrova brochure", "img-t0", trail=["Brochure"]),
        comp(
            "section",
            "",
            "img-t1",
            comp("heading", "Performance", "img-t2", trail=["Brochure"]),
            comp("paragraph", "0-62 mph: 9.1 s", "img-t3", trail=["Brochure", "Performance"]),
            comp("paragraph", "Power: 110 kW", "img-t4", trail=["Brochure", "Performance"]),
            trail=["Brochure"],
        ),
        comp(
            "section",
            "",
            "img-t5",
            comp("heading", "Price", "img-t6", trail=["Brochure"]),
            comp("paragraph", "From £24,995 on the road.", "img-t7", trail=["Brochure", "Price"]),
            trail=["Brochure"],
        ),
        trail=["Brochure"],
    )


def test_an_image_with_text_read_from_it_is_cut_like_a_section() -> None:
    root = comp(
        "section",
        "",
        "root",
        comp("paragraph", "Download the brochure.", "p1"),
        scanned_page(alt="Brochure page 1"),
        comp("paragraph", "Book a test drive today.", "p2"),
    )
    units = gate_units(root)
    assert [(u.id, u.component_ids) for u in units] == [
        ("root#0", ("p1",)),
        ("img#0", ("img", "img-t0")),
        ("img-t1#0", ("img-t2", "img-t3", "img-t4")),
        ("img-t5#0", ("img-t6", "img-t7")),
        ("root#1", ("p2",)),
    ]
    assert units[1].text == "Brochure page 1\nDelmaro Kestrova brochure"
    assert units[2].state() == {
        "content": "Performance\n0-62 mph: 9.1 s\nPower: 110 kW",
        "section": "Brochure",
    }


def test_an_images_ocr_headings_reach_the_section_of_units_below_them() -> None:
    units = gate_units(comp("section", "", "root", scanned_page()), max_chars=30)
    perf = [u for u in units if "img-t3" in u.component_ids or "img-t4" in u.component_ids]
    assert [(u.component_ids, u.state()) for u in perf] == [
        (("img-t2", "img-t3"), {"content": "Performance\n0-62 mph: 9.1 s", "section": "Brochure"}),
        (("img-t4",), {"content": "Power: 110 kW", "section": "Brochure › Performance"}),
    ]
    assert all(len(u.text) <= 30 for u in units)


def test_a_heading_before_a_read_image_opens_its_first_unit() -> None:
    root = comp("section", "", "root", comp("heading", "Specifications", "h"), scanned_page())
    first = gate_units(root)[0]
    assert (first.id, first.component_ids) == ("img#0", ("h", "img-t0"))
    assert first.text == "Specifications\nDelmaro Kestrova brochure"


def test_an_image_alone_at_the_root_is_cut_like_a_section() -> None:
    units = gate_units(scanned_page())
    assert [u.component_ids for u in units] == [
        ("img-t0",),
        ("img-t2", "img-t3", "img-t4"),
        ("img-t6", "img-t7"),
    ]


def test_an_image_without_text_read_from_it_is_one_block_as_before() -> None:
    root = comp(
        "section",
        "",
        "root",
        comp("paragraph", "The Kestrova.", "p1"),
        comp("image", "Side view", "img1"),
        comp("image", "Front view", "img2", comp("caption", "Figure 2: the grille", "cap")),
        comp("image", "", "img3"),
        comp("paragraph", "Book a test drive.", "p2"),
    )
    [unit] = gate_units(root)
    assert unit.component_ids == ("p1", "img1", "img2", "cap", "img3", "p2")
    assert unit.text == (
        "The Kestrova.\nSide view\nFront view\nFigure 2: the grille\nBook a test drive."
    )


def test_a_pdf_figure_with_text_found_in_it_is_cut_like_a_section() -> None:
    figure = comp(
        "image",
        "",
        "f",
        comp("caption", "Figure 3: prices", "cap"),
        comp("paragraph", "Prices exclude VAT.", "fn"),
    )
    root = comp(
        "section",
        "",
        "root",
        comp("paragraph", "The Kestrova.", "p1"),
        figure,
        comp("paragraph", "Book a test drive.", "p2"),
    )
    assert [(u.id, u.component_ids) for u in gate_units(root)] == [
        ("root#0", ("p1",)),
        ("f#0", ("cap", "fn")),
        ("root#1", ("p2",)),
    ]


# --- the gate ------------------------------------------------------------------------


def parsed(root: Component) -> ParsedDocument:
    return ParsedDocument(document=Document.from_bytes(b"<p/>"), root=root)


def test_noul_component_gate_is_a_component_gate() -> None:
    assert isinstance(NoulComponentGate(), ComponentGate)


async def test_groups_pass_the_units_jev_says_contain_them() -> None:
    fake = (
        FakeJev()
        .noul("contain the price (GBP)?", p=0.9, state="24,995")
        .noul("engine power (kW) or 0-62", p=0.8, state="Power: 110 kW")
    )
    result = await NoulComponentGate().gate(
        parsed(page()), [SchemaSpec.from_model(Car)], fake.client()
    )
    groups = result["Car"]
    # Passing units bring their ancestors, in reading order.
    assert groups["price"] == ["root", "s-price", "h2", "p2", "t1"]
    assert groups["performance"] == ["root", "s-perf", "h3", "l1", "li1", "li2"]


async def test_a_value_past_max_chars_still_passes_its_component() -> None:
    text = " ".join(f"Sentence {i} is filler." for i in range(200)) + " It costs £24,995."
    root = comp("section", "", "root", comp("paragraph", text, "p"))
    assert len(text) > 2000
    fake = FakeJev().noul("contain the price (GBP)?", p=0.9, state="24,995")
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    assert result["Car"]["price"] == ["root", "p"]


async def test_a_scanned_page_passes_only_the_ocr_text_jev_says_contains_the_field() -> None:
    fake = FakeJev().noul("contain the price (GBP)?", p=0.9, state="24,995")
    root = comp("section", "", "root", scanned_page(alt="Brochure page 1"))
    result = await NoulComponentGate().gate(
        parsed(root), [SchemaSpec.from_model(Car)], fake.client()
    )
    # The image passes as the price section's ancestor; the rest of the page doesn't.
    assert result["Car"]["price"] == ["root", "img", "img-t5", "img-t6", "img-t7"]
    assert result["Car"]["performance"] == []


async def test_non_noul_answers_are_an_error() -> None:
    class ChoiceBackend:
        async def system_one(
            self, state: JSONContent, questions: Mapping[str, Question]
        ) -> JevResponse:
            answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {
                k: ChoiceAnswer(choice="x", confidence=1.0, probabilities={"x": 1.0})
                for k in questions
            }
            return JevResponse(answers=answers, input_tokens=1, model="fake")

    with pytest.raises(UnexpectedAnswerError, match="expected a Noul answer"):
        await NoulComponentGate().gate(
            parsed(page()), [SchemaSpec.from_model(Car)], JevClient(ChoiceBackend())
        )


async def test_all_schemas_questions_about_a_unit_go_in_one_request() -> None:
    fake = FakeJev()
    specs = [SchemaSpec.from_model(Car), SchemaSpec.from_model(Book)]
    await NoulComponentGate().gate(parsed(page()), specs, fake.client())
    assert len(fake.calls) == len(gate_units(page()))
    assert set(fake.calls[0].questions) == {"Car.price", "Car.performance", "Book.title"}
    questions = fake.calls[0].questions
    assert questions["Car.price"].instructions == "Does this section contain the price (GBP)?"
    assert questions["Car.performance"].instructions == (
        "Does this section contain the engine power (kW) or 0-62 mph time (s)?"
    )
    assert questions["Book.title"].instructions == "Is this the book's title?"


async def test_threshold_is_inclusive_and_validated() -> None:
    fake = FakeJev(default_p=0.3)
    result = await NoulComponentGate(threshold=0.3).gate(
        parsed(page()), [SchemaSpec.from_model(Book)], fake.client()
    )
    assert "p3" in result["Book"]["title"]
    strict = await NoulComponentGate(threshold=0.31).gate(
        parsed(page()), [SchemaSpec.from_model(Book)], fake.client()
    )
    assert strict["Book"]["title"] == []
    with pytest.raises(ValueError, match="threshold"):
        NoulComponentGate(threshold=1.5)
    with pytest.raises(ValueError, match="max_chars"):
        NoulComponentGate(max_chars=0)


# --- the stage and what it does downstream -------------------------------------------


def context(fake: FakeJev, root: Component | None) -> Context:
    ctx = Context.create(
        Document.from_bytes(b"<p/>"),
        [SchemaSpec.from_model(Car), SchemaSpec.from_model(Book)],
        fake.client(),
    )
    if root is not None:
        ctx.parsed = parsed(root)
    return ctx


async def test_stage_sets_component_ids_and_scopes_keep_only_passing_components() -> None:
    fake = FakeJev().noul("price", p=0.9, state="24,995")
    ctx = context(fake, page())
    await ComponentGateStage().run(ctx)
    await EntityStage().run(ctx)

    car = ctx.schemas["Car"]
    assert car.component_ids is not None
    assert car.component_ids["performance"] == []
    [scope] = car.scopes
    assert scope.component_ids == ["root", "s-price", "h2", "p2", "t1"]
    assert [f.name for f in car.relevant_fields("p2")] == ["price"]
    assert car.relevant_fields("li1") == []

    book = ctx.schemas["Book"]
    assert book.scopes[0].component_ids == []
    assert [e.kind for e in ctx.events] == ["no_relevant_components"]


async def test_the_resolver_sees_only_gated_components() -> None:
    seen: list[list[str]] = []

    class Recording:
        async def resolve(
            self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
        ) -> list[EntityScope]:
            ids = [c.id for c in parsed.root.walk()]
            seen.append(ids)
            return [EntityScope(label="doc", component_ids=ids)]

    fake = FakeJev().noul("price", p=0.9, state="24,995")
    ctx = context(fake, page())
    await ComponentGateStage().run(ctx)
    await EntityStage(resolver=Recording()).run(ctx)
    # Car passed the price section; Book passed nothing, so it sees only the root.
    assert sorted(seen) == [["root"], ["root", "s-price", "h2", "p2", "t1"]]


def test_restricted_to_prunes_the_tree_and_keeps_structured_statements() -> None:
    loc = DomLocation(dom_path="/")
    statements = {
        "p1.0": Statement(id="p1.0", text="a", kind="sentence", component_id="p1", location=loc),
        "p2.0": Statement(id="p2.0", text="b", kind="sentence", component_id="p2", location=loc),
        "ld.0": Statement(id="ld.0", text="c", kind="structured", component_id="ld", location=loc),
    }
    doc = ParsedDocument(document=Document.from_bytes(b"<p/>"), root=page(), statements=statements)
    view = doc.restricted_to(["s-price", "h2", "p2"])
    assert [c.id for c in view.root.walk()] == ["root", "s-price", "h2", "p2"]
    assert set(view.statements) == {"p2.0", "ld.0"}
    assert [c.id for c in doc.root.walk()][:3] == ["root", "h1", "p1"]  # the original is untouched


async def test_without_a_gate_everything_is_relevant() -> None:
    ctx = context(FakeJev(), page())
    await EntityStage().run(ctx)
    car = ctx.schemas["Car"]
    assert car.component_ids is None
    assert car.relevant_components() is None
    assert [f.name for f in car.relevant_fields("li1")] == ["price", "power_kw", "zero_to_62_s"]
    assert len(car.scopes[0].component_ids) == len(list(page().walk()))


async def test_stage_skips_without_a_parsed_document_or_active_schemas() -> None:
    fake = FakeJev()
    ctx = context(fake, None)
    await ComponentGateStage().run(ctx)
    assert ctx.schemas["Car"].component_ids is None
    ctx = context(fake, page())
    for run in ctx.schemas.values():
        run.deactivate()
    await ComponentGateStage().run(ctx)
    assert fake.calls == []


def test_component_gate_is_a_default_stage_between_layout_and_entities() -> None:
    names = [s.name for s in default_pipeline().stages]
    assert names.index("layout") < names.index("component_gate") < names.index("entities")


# --- nested models ---------------------------------------------------------------------


class Trim(BaseModel):
    """One trim of a car."""

    power_kw: float = Field(description="Engine power", unit="kW")
    price: Decimal = Field(description="Trim price", unit="GBP")


class CarModel(BaseModel):
    """A car model page."""

    name: str = Field(description="Model name")
    trims: list[Trim] = Field(description="Trims")


@dataclass
class TrimsIn:
    """A resolver giving the whole view to one parent and one child."""

    async def resolve(
        self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
    ) -> list[EntityScope]:
        ids = [c.id for c in parsed.root.walk()]
        return [
            EntityScope(label="doc", component_ids=ids),
            EntityScope(label="SE", component_ids=ids, parent="doc", field="trims"),
        ]


async def test_nested_models_are_gated_per_field_in_the_same_requests() -> None:
    fake = (
        FakeJev()
        .noul("engine power (kW)?", p=0.9, state="Power: 110 kW")
        .noul("trim price (GBP)?", p=0.9, state="SE | £24,995")
    )
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(CarModel)], fake.client()
    )
    ctx.parsed = parsed(page())
    await ComponentGateStage().run(ctx)

    assert len(fake.calls) == len(gate_units(page()))
    questions = {k: q.instructions for k, q in fake.calls[0].questions.items()}
    assert questions == {
        "CarModel.name": "Does this section contain the model name?",
        "CarModel.trims": (
            "Does this section contain the engine power (kW) or trim price (GBP) of the trims?"
        ),
        "CarModel.trims.power_kw": "Does this section contain the engine power (kW)?",
        "CarModel.trims.price": "Does this section contain the trim price (GBP)?",
    }
    run = ctx.schemas["CarModel"]
    assert run.child_component_ids == {
        "trims": {
            "power_kw": ["root", "s-perf", "h3", "l1", "li1", "li2"],
            "price": ["root", "t1"],
        }
    }
    # Jev said no to "the trims" everywhere, but what passed a nested field reaches the
    # trims group, so the resolver sees it.
    assert run.component_ids == {
        "name": [],
        "trims": ["root", "s-perf", "h3", "l1", "li1", "li2", "t1"],
    }

    await EntityStage(resolver=TrimsIn()).run(ctx)
    child = ctx.schemas["CarModel.trims"]
    assert child.component_ids == run.child_component_ids["trims"]
    assert [f.name for f in child.relevant_fields("li1")] == ["power_kw"]
    assert [f.name for f in child.relevant_fields("t1")] == ["price"]
    assert child.relevant_fields("p3") == []


async def test_a_nested_field_keeps_what_passed_its_own_question() -> None:
    fake = FakeJev().noul("of the trims?", p=0.9, state="Book a test drive")
    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(CarModel)], fake.client()
    )
    ctx.parsed = parsed(page())
    await ComponentGateStage().run(ctx)
    run = ctx.schemas["CarModel"]
    assert run.component_ids is not None
    assert "p3" in run.component_ids["trims"]
    assert run.child_component_ids == {"trims": {"power_kw": [], "price": []}}


async def test_a_gate_that_ignores_nested_models_leaves_the_child_run_ungated() -> None:
    asked: list[list[str]] = []

    class ParentsOnly:
        async def gate(
            self, parsed: ParsedDocument, schemas: list[SchemaSpec], jev: JevClient
        ) -> dict[str, dict[str, list[str]]]:
            asked.append([s.name for s in schemas])
            ids = [c.id for c in parsed.root.walk()]
            return {"CarModel": {"name": ids, "trims": ids}}

    ctx = Context.create(
        Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(CarModel)], FakeJev().client()
    )
    ctx.parsed = parsed(page())
    await ComponentGateStage(gate=ParentsOnly()).run(ctx)
    assert asked == [["CarModel", "CarModel.trims"]]
    assert ctx.schemas["CarModel"].child_component_ids == {}
    await EntityStage(resolver=TrimsIn()).run(ctx)
    child = ctx.schemas["CarModel.trims"]
    assert child.component_ids is None
    assert [f.name for f in child.relevant_fields("li1")] == ["power_kw", "price"]


async def test_a_nested_model_jevex_cannot_extract_is_not_gated() -> None:
    class Odd(BaseModel):
        tags: dict[str, str] = Field(description="Tags")

    class Reserved(BaseModel):
        none: str = Field(description="Nothing")

    class Page(BaseModel):
        name: str = Field(description="Model name")
        odd: Odd = Field(description="Odd bits")
        reserved: Reserved = Field(description="Reserved bits")

    fake = FakeJev()
    ctx = Context.create(Document.from_bytes(b"<p/>"), [SchemaSpec.from_model(Page)], fake.client())
    ctx.parsed = parsed(page())
    await ComponentGateStage().run(ctx)
    assert set(fake.calls[0].questions) == {"Page.name", "Page.odd", "Page.reserved"}
    assert fake.calls[0].questions["Page.odd"].instructions == (
        "Does this section contain the odd bits?"
    )
    assert ctx.schemas["Page"].child_component_ids == {}


# --- a real page ---------------------------------------------------------------------

SPEC_PAGE = b"""<!doctype html><html><head><title>Delmaro Kestrova</title></head><body>
<nav><a href="/">Home</a> <a href="/cars">Cars</a></nav>
<main>
  <h1>Delmaro Kestrova 1.5 SE</h1>
  <p>A roomy family hatchback with a frugal petrol engine.</p>
  <section><h2>Performance</h2>
    <dl><dt>Power</dt><dd>110 kW</dd><dt>0-62 mph</dt><dd>9.1 s</dd></dl>
  </section>
  <section><h2>Price</h2><p>On the road from &pound;24,995.</p></section>
  <aside><h2>Newsletter</h2><p>Sign up for our weekly deals.</p></aside>
</main>
<footer>&copy; 2026 Example Motors</footer>
</body></html>"""


async def test_a_real_page_gates_by_section() -> None:
    doc = BoilerplateCleaner().clean(Document.from_bytes(SPEC_PAGE, url="https://cars.test/k"))
    root = await HtmlLayoutParser().parse(doc)
    fake = (
        FakeJev()
        .noul("engine power", p=0.9, state="0-62 mph: 9.1 s")
        .noul("price", p=0.9, state="24,995")
    )
    result = await NoulComponentGate().gate(
        ParsedDocument(document=doc, root=root), [SchemaSpec.from_model(Car)], fake.client()
    )
    by_id = {c.id: c for c in root.walk()}

    def texts(ids: list[str]) -> set[str]:
        return {by_id[i].text for i in ids if by_id[i].text}

    assert "0-62 mph: 9.1 s" in texts(result["Car"]["performance"])
    assert "On the road from £24,995." in texts(result["Car"]["price"])
    newsletter = texts(result["Car"]["price"]) | texts(result["Car"]["performance"])
    assert "Sign up for our weekly deals." not in newsletter

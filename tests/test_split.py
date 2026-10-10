import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from pydantic import BaseModel

from jevex import (
    BBox,
    BoilerplateCleaner,
    CandidateStage,
    Component,
    Context,
    DefaultSplitter,
    Document,
    DomLocation,
    Extractor,
    Field,
    ImageLocation,
    SchemaSpec,
    Statement,
    StatementStage,
    cut_statement,
)
from jevex.extractor import default_pipeline
from jevex.generators import default_registry
from jevex.interfaces import LocaleAwareSplitter, ParsedDocument, StatementSplitter
from jevex.jev import Choice
from jevex.layout import TableCell
from jevex.layout_html import HtmlLayoutParser
from jevex.normalise import normalise
from jevex.pipeline import Pipeline
from jevex.split import (
    ABBREVIATIONS,
    LANGUAGE_ABBREVIATIONS,
    MAX_SEGMENT_CHARS,
    MAX_STATEMENT_CHARS,
    DuplicateStatementError,
    _chunks,  # pyright: ignore[reportPrivateUsage]
    is_key_value,
    sentences,
)
from jevex.statements import StatementKind
from jevex.tables import table_statements
from jevex.testing import FakeJev
from jevex.testsite import VehicleSpec, generate, render

LOC = DomLocation(dom_path="/html/body/p")


def comp(
    type_: str,
    text: str = "",
    *,
    cid: str = "c1",
    path: str = "/html/body/p",
    trail: list[str] | None = None,
    children: list[Component] | None = None,
) -> Component:
    return Component.model_validate(
        {
            "id": cid,
            "type": type_,
            "text": text,
            "location": DomLocation(dom_path=path),
            "heading_trail": trail or [],
            "children": children or [],
        }
    )


def split(component: Component) -> list[tuple[str, str]]:
    return [(s.text, s.kind) for s in DefaultSplitter().split(component)]


def test_default_splitter_is_a_statement_splitter() -> None:
    assert isinstance(DefaultSplitter(), StatementSplitter)


# --- paragraphs ----------------------------------------------------------------------


def test_paragraphs_split_into_sentences() -> None:
    text = (
        "The 1.5 TSI SE does 0-62 mph in 9.1 s. It costs £24,995 (approx.). "
        "Top speed is 130 mph! Is it good? Mr. Smith, e.g., says yes."
    )
    assert split(comp("paragraph", text)) == [
        ("The 1.5 TSI SE does 0-62 mph in 9.1 s.", "sentence"),
        ("It costs £24,995 (approx.).", "sentence"),
        ("Top speed is 130 mph!", "sentence"),
        ("Is it good?", "sentence"),
        ("Mr. Smith, e.g., says yes.", "sentence"),
    ]


def test_decimal_numbers_and_abbreviations_dont_end_sentences() -> None:
    assert split(comp("paragraph", "It has a 1.5 l engine and 3.5 kWh of battery.")) == [
        ("It has a 1.5 l engine and 3.5 kWh of battery.", "sentence")
    ]


def test_line_broken_label_value_lines_are_pairs() -> None:
    text = "Engine: 1.5 TSI\nPower: 150 PS\nA lively car. Very frugal."
    assert split(comp("paragraph", text)) == [
        ("Engine: 1.5 TSI", "key_value"),
        ("Power: 150 PS", "key_value"),
        ("A lively car.", "sentence"),
        ("Very frugal.", "sentence"),
    ]


def test_a_label_line_with_several_sentences_is_split_as_sentences() -> None:
    assert split(comp("paragraph", "Verdict: quick. Also cheap.")) == [
        ("Verdict: quick.", "sentence"),
        ("Also cheap.", "sentence"),
    ]


def test_whitespace_is_collapsed_and_empty_text_gives_nothing() -> None:
    assert split(comp("paragraph", "  Fast   car. \n\n ")) == [("Fast car.", "sentence")]
    assert split(comp("paragraph", "   ")) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Range: approx. 300 miles.", ["Range: approx. 300 miles."]),
        ("Price excl. VAT is £20,000.", ["Price excl. VAT is £20,000."]),
        ("It costs £24,995 excl. VAT.", ["It costs £24,995 excl. VAT."]),
        ("Max. speed is 155 mph.", ["Max. speed is 155 mph."]),
        ("Est. delivery: 3 weeks.", ["Est. delivery: 3 weeks."]),
        ("It has 4 cyl. and 6 gears.", ["It has 4 cyl. and 6 gears."]),
        ("Vol. 2, pp. 34-56.", ["Vol. 2, pp. 34-56."]),
        ("The VW ID.3 Pro is here.", ["The VW ID.3 Pro is here."]),
        ("The ID.4 GTX is quick.", ["The ID.4 GTX is quick."]),
        # A real sentence end after an abbreviation stays split.
        ("It takes 5 min. Then it charges.", ["It takes 5 min.", "Then it charges."]),
        # Inline enumerations still split.
        (
            "Features include: 1. heated seats 2. a sunroof.",
            ["Features include:", "1. heated seats", "2. a sunroof."],
        ),
    ],
)
def test_mid_sentence_abbreviations_and_model_names_dont_split(
    text: str, expected: list[str]
) -> None:
    assert sentences(text) == expected


def test_a_range_label_with_an_abbreviation_stays_a_pair() -> None:
    assert split(comp("paragraph", "Range: approx. 300 miles.")) == [
        ("Range: approx. 300 miles.", "key_value")
    ]


def test_long_lines_are_chunked_without_losing_text() -> None:
    text = " ".join(f"Sentence {i} is approx. {i} words long." for i in range(1000))
    said = sentences(text)
    assert len(said) == 1000
    assert " ".join(said) == text


def test_long_lines_without_sentence_boundaries_are_chunked_at_spaces() -> None:
    text = " ".join(["approx", "max", "a", "turbo", "petrol", "hatchback"] * 3000)
    chunks = _chunks(text)
    assert len(chunks) > 1
    assert all(len(c) <= 2 * MAX_SEGMENT_CHARS for c in chunks)
    assert " ".join(chunks) == text
    assert " ".join(sentences(text)) == text


def test_sentences_is_safe_across_threads() -> None:
    texts = [
        " ".join(f"Car {i} does 0-62 mph in {i}.1 s." for i in range(300)),
        " ".join(f"Book {i} costs £{i}.99 today." for i in range(300)),
    ]

    def rejoin(text: str) -> str:
        return " ".join(sentences(text))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(rejoin, texts * 10))
    assert results == texts * 10


def test_unknown_languages_fall_back_to_english() -> None:
    assert sentences("One. Two.", language="xx") == ["One.", "Two."]
    assert sentences("Das ist gut. Das auch.", language="de") == ["Das ist gut.", "Das auch."]


# --- list items, headings, captions, images ------------------------------------------


def test_list_items_are_one_statement_each() -> None:
    assert split(comp("list_item", "Heated seats and a\nreversing camera. Also DAB.")) == [
        ("Heated seats and a reversing camera. Also DAB.", "list_item")
    ]


def test_label_value_list_items_are_pairs() -> None:
    assert split(comp("list_item", "Mileage: 12,000 miles")) == [
        ("Mileage: 12,000 miles", "key_value")
    ]


def test_definition_list_items_are_pairs_only_with_their_term() -> None:
    # The layout parser renders ``<dt>Fuel</dt><dd>Petrol</dd>`` as "Fuel: Petrol" at the dd.
    item = comp("list_item", "Fuel: Petrol", path="/html/body/dl/dd[2]")
    assert split(item) == [("Fuel: Petrol", "key_value")]
    # A term without a value, and a value without a term, aren't pairs.
    assert split(comp("list_item", "Sunroof", path="/html/body/dl/dt")) == [
        ("Sunroof", "list_item")
    ]
    assert split(comp("list_item", "Petrol", path="/html/body/dl/dd")) == [("Petrol", "list_item")]
    # A dd is trusted as a pair even when its label has no letter.
    assert split(comp("list_item", "0-62: 9.1 s", path="/html/body/dl/dd")) == [
        ("0-62: 9.1 s", "key_value")
    ]


async def layout_items(html: bytes) -> list[list[tuple[str, str]]]:
    root = await HtmlLayoutParser().parse(Document.from_bytes(html, content_type="text/html"))
    return [split(c) for c in root.walk() if c.type == "list_item"]


@pytest.mark.parametrize(
    "html",
    [
        b"<ul><li>Engine: 1.5 TSI<br>Power: 150 PS</li></ul>",
        b"<ul><li><p>Engine: 1.5 TSI</p><p>Power: 150 PS</p></li></ul>",
    ],
)
async def test_a_list_item_of_several_pairs_gives_one_pair_each(html: bytes) -> None:
    assert await layout_items(html) == [
        [("Engine: 1.5 TSI", "key_value"), ("Power: 150 PS", "key_value")]
    ]


async def test_a_list_item_without_pairs_stays_one_statement() -> None:
    assert await layout_items(b"<ul><li><strong>Engine</strong><br>1.5 TSI</li></ul>") == [
        [("Engine 1.5 TSI", "list_item")]
    ]
    assert split(comp("list_item", "Heated seats\nPower: 150 PS")) == [
        ("Heated seats", "list_item"),
        ("Power: 150 PS", "key_value"),
    ]


def test_headings_captions_and_alt_text() -> None:
    assert split(comp("heading", "A Light in the  Attic")) == [("A Light in the Attic", "sentence")]
    assert split(comp("caption", "Figure 1: the dashboard")) == [
        ("Figure 1: the dashboard", "caption")
    ]
    assert split(comp("image", "Four stars")) == [("Four stars", "alt_text")]


def test_text_read_from_an_image_is_ocr() -> None:
    in_image = ImageLocation(src="https://example.com/spec.png", bbox=BBox(x0=0, y0=0, x1=9, y1=9))
    paragraph = comp("paragraph", "Quiet and quick. Power: 150 PS").model_copy(
        update={"location": in_image}
    )
    heading = comp("heading", "Performance").model_copy(update={"location": in_image})
    assert split(paragraph) == [("Quiet and quick.", "ocr"), ("Power: 150 PS", "ocr")]
    assert split(heading) == [("Performance", "ocr")]
    pair = comp("paragraph", "Power: 150 PS").model_copy(update={"location": in_image})
    assert [(s.kind, s.location) for s in DefaultSplitter().split(pair)] == [
        ("key_value", in_image)
    ]


@pytest.mark.parametrize("type_", ["section", "column", "list", "breakout", "table"])
def test_containers_and_tables_give_no_statements(type_: str) -> None:
    child = comp("paragraph", "Inside.", cid="c2")
    assert split(comp(type_, "a | b", children=[child])) == []


def test_statements_carry_the_component_context() -> None:
    component = comp("paragraph", "One. Two.", cid="c7", trail=["Specs", "Performance"])
    one, two = DefaultSplitter().split(component)
    assert (one.id, two.id) == ("c7.0", "c7.1")
    assert one.component_id == "c7"
    assert one.heading_trail == ["Specs", "Performance"]
    assert one.location == component.location


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Engine: 1.5 TSI", True),
        ("Price : £24,995", True),
        ("0-62 mph: 9.1 s", True),
        (f"Colour{chr(0xFF1A)}Red", True),  # fullwidth colon
        ("The show starts at 10:30", False),
        ("Aspect ratio 16:9", False),
        ("ISBN-13: 978-0-14-032872-1", True),
        ("Series 5: 2019", True),
        ("Model 3: 39,990", True),
        ("12:30", False),
        ("See https://example.com", False),
        ("No colon here", False),
        ("Engine:", False),
        (": 150 PS", False),
        ("This label is far too long to be a label for anything: 3", False),
    ],
)
def test_is_key_value(text: str, expected: bool) -> None:
    assert is_key_value(text) is expected


# --- cutting long statements ---------------------------------------------------------


def stmt(text: str, kind: StatementKind = "list_item") -> Statement:
    return Statement(id="c1.0", text=text, kind=kind, component_id="c1", location=LOC)


def test_a_statement_that_fits_is_returned_as_is() -> None:
    s = stmt("x" * MAX_STATEMENT_CHARS)
    assert cut_statement(s) == [s]


def test_long_statements_are_cut_at_sentence_ends() -> None:
    text = "The car is quick. " * 3 + "It has a very long and winding sentence with no end in sight"
    pieces = cut_statement(stmt(text), 60)
    assert [p.text for p in pieces] == [
        "The car is quick. The car is quick. The car is quick.",
        "It has a very long and winding sentence with no end in sight",
    ]
    assert [p.id for p in pieces] == ["c1.0:0", "c1.0:1"]
    assert {(p.kind, p.component_id, p.location) for p in pieces} == {("list_item", "c1", LOC)}


def test_a_sentence_end_too_early_in_a_piece_is_passed_over_for_whitespace() -> None:
    # The only sentence end is in the first half, so the cut falls at the last space.
    pieces = cut_statement(stmt("Hi. " + "aaaa " * 20), 40)
    assert [p.text for p in pieces] == [
        "Hi. aaaa aaaa aaaa aaaa aaaa aaaa aaaa",
        "aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa",
        "aaaa aaaa aaaa aaaa aaaa",
    ]


def test_text_without_whitespace_is_cut_mid_word() -> None:
    text = "x" * 250
    pieces = cut_statement(stmt(text), 100)
    assert [len(p.text) for p in pieces] == [100, 100, 50]
    assert "".join(p.text for p in pieces) == text


def test_every_piece_of_a_cut_pair_keeps_its_label() -> None:
    pieces = cut_statement(stmt("Notes: " + "lorem ipsum " * 20, "key_value"), 60)
    assert len(pieces) == 5
    assert all(re.match(r"Notes: (lorem|ipsum)", p.text) and len(p.text) <= 60 for p in pieces)
    assert all(p.kind == "key_value" for p in pieces)


def test_every_piece_of_a_cut_table_cell_keeps_its_headers() -> None:
    table = Component(
        id="t1",
        type="table",
        text="(rows)",
        cells=[
            TableCell(row=0, col=0, text="", header=True),
            TableCell(row=0, col=1, text="SE", header=True),
            TableCell(row=1, col=0, text="Notes", header=True),
            TableCell(row=1, col=1, text="lorem ipsum " * 20),
        ],
        location=DomLocation(dom_path="/html/body/table"),
    )
    [cell] = [s for s in table_statements(table) if s.kind == "table_cell"]
    pieces = cut_statement(cell, 60)
    assert len(pieces) == 5
    assert all(re.match(r"Notes · SE: (lorem|ipsum)", p.text) and len(p.text) <= 60 for p in pieces)
    assert all(p.table == cell.table for p in pieces)
    assert pieces[0].id == "t1.r1c1:0"


def test_a_label_longer_than_half_a_piece_isnt_repeated() -> None:
    label = "Label with many words in it"  # 27 characters, over half of 50
    pieces = cut_statement(stmt(f"{label}: " + "value " * 20, "key_value"), 50)
    assert pieces[0].text.startswith(label)
    assert not any(p.text.startswith(label) for p in pieces[1:])
    assert all(len(p.text) <= 50 for p in pieces)


def test_a_custom_splitters_prefix_that_doesnt_match_the_headers_isnt_repeated() -> None:
    [cell] = table_statements(
        Component(
            id="t1",
            type="table",
            text="(rows)",
            cells=[
                TableCell(row=0, col=0, text="Notes", header=True),
                TableCell(row=0, col=1, text="x"),
            ],
            location=DomLocation(dom_path="/html/body/table"),
        )
    )
    reworded = cell.model_copy(update={"text": "The notes say " + "word " * 30})
    pieces = cut_statement(reworded, 50)
    assert pieces[0].text.startswith("The notes say")
    assert not any(p.text.startswith("Notes") for p in pieces)


def test_cut_statement_needs_a_positive_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars must be positive, got 0"):
        cut_statement(stmt("x"), 0)


# --- the stage -----------------------------------------------------------------------


def context(root: Component, existing: list[Statement] | None = None) -> Context:
    ctx = Context.create(
        Document.from_bytes(b"<p/>", url="https://example.com"),
        [SchemaSpec.from_model(VehicleSpec)],
        FakeJev().client(),
    )
    ctx.parsed = ParsedDocument(
        document=ctx.document, root=root, statements={s.id: s for s in existing or []}
    )
    return ctx


async def test_stage_splits_every_component_in_reading_order() -> None:
    root = comp(
        "section",
        cid="root",
        children=[
            comp("heading", "Specs", cid="h"),
            comp("paragraph", "Quick. Cheap.", cid="p"),
            comp(
                "list",
                cid="l",
                children=[
                    comp("list_item", "Heated seats", cid="l1"),
                    comp(
                        "list_item",
                        "Extras",
                        cid="l2",
                        children=[
                            comp("list", cid="l3", children=[comp("list_item", "DAB", cid="l4")])
                        ],
                    ),
                ],
            ),
        ],
    )
    structured = Statement(
        id="ld.0", text="name: Delmaro", kind="structured", component_id="ld", location=LOC
    )
    ctx = context(root, [structured])
    await StatementStage().run(ctx)
    assert ctx.parsed is not None
    assert list(ctx.parsed.statements) == ["ld.0", "h.0", "p.0", "p.1", "l1.0", "l2.0", "l4.0"]


async def test_stage_keeps_statements_already_on_a_component_in_reading_order() -> None:
    root = comp(
        "section",
        cid="root",
        children=[
            comp("paragraph", "First.", cid="p1"),
            comp("paragraph", "A red hatchback.", cid="v"),
            comp("paragraph", "Last.", cid="p2"),
        ],
    )
    vision = Statement(
        id="v.0", text="A red hatchback", kind="vision", component_id="v", location=LOC
    )
    structured = Statement(
        id="ld.0", text="name: Delmaro", kind="structured", component_id="ld", location=LOC
    )
    ctx = context(root, [vision, structured])
    await StatementStage().run(ctx)
    assert ctx.parsed is not None
    assert [(sid, s.kind) for sid, s in ctx.parsed.statements.items()] == [
        ("ld.0", "structured"),
        ("p1.0", "sentence"),
        ("v.0", "vision"),
        ("p2.0", "sentence"),
    ]


async def test_stage_reads_the_tables_the_gate_found_headers_in_with_them() -> None:
    def kv(cid: str) -> Component:
        cells = [
            TableCell(row=0, col=0, text="Engine"),
            TableCell(row=0, col=1, text="1.5 TSI"),
            TableCell(row=1, col=0, text="Power"),
            TableCell(row=1, col=1, text="150 PS"),
        ]
        return comp("table", "Engine | 1.5 TSI\nPower | 150 PS", cid=cid).model_copy(
            update={"cells": cells}
        )

    ctx = context(comp("section", cid="root", children=[kv("yes"), kv("no")]))
    ctx.headed_tables = frozenset({"yes"})
    await StatementStage().run(ctx)
    assert ctx.parsed is not None
    assert [(sid, s.text) for sid, s in ctx.parsed.statements.items()] == [
        ("yes.r0c1", "Engine: 1.5 TSI"),
        ("yes.r1c1", "Power: 150 PS"),
        ("no.r0", "Engine | 1.5 TSI"),
        ("no.r1", "Power | 150 PS"),
    ]


def trim_section(name: str, cid: str) -> Component:
    return comp(
        "section",
        cid=cid,
        children=[
            comp("heading", name, cid=f"{cid}h"),
            comp("paragraph", f"Power: {len(name)}00 PS", cid=f"{cid}p"),
        ],
    )


async def test_stage_gives_the_names_of_a_run_of_sections_to_each_name() -> None:
    root = comp(
        "section",
        cid="root",
        children=[
            comp("heading", "Kestrova trims", cid="title"),
            trim_section("SE", "a"),
            trim_section("Sport", "b"),
            trim_section("GT", "c"),
        ],
    )
    ctx = context(root)
    await StatementStage().run(ctx)
    assert ctx.parsed is not None
    assert {sid: s.sibling_labels for sid, s in ctx.parsed.statements.items()} == {
        "title.0": None,
        "ah.0": "SE, Sport, GT",
        "ap.0": None,
        "bh.0": "SE, Sport, GT",
        "bp.0": None,
        "ch.0": "SE, Sport, GT",
        "cp.0": None,
    }


async def test_stage_gives_the_names_to_every_piece_of_a_cut_name_from_any_splitter() -> None:
    class Headings:
        def split(self, component: Component) -> list[Statement]:
            if component.type != "heading":
                return []
            return [
                Statement(
                    id=f"{component.id}.0",
                    text=component.text,
                    kind="sentence",
                    component_id=component.id,
                    location=LOC,
                )
            ]

    long = "Sport " * 30
    root = comp("section", cid="root", children=[trim_section("SE", "a"), trim_section(long, "b")])
    ctx = context(root)
    await StatementStage(Headings(), max_chars=100).run(ctx)
    assert ctx.parsed is not None
    # Each name is shortened as an entity label is.
    names = "SE, " + "Sport " * 12 + "Sport…"
    assert {sid: s.sibling_labels for sid, s in ctx.parsed.statements.items()} == {
        "ah.0": names,
        "bh.0:0": names,
        "bh.0:1": names,
    }


async def test_stage_refuses_duplicate_statement_ids() -> None:
    clash = Statement(id="p.0", text="x", kind="structured", component_id="ld", location=LOC)
    ctx = context(
        comp("section", cid="root", children=[comp("paragraph", "Hi.", cid="p")]), [clash]
    )
    with pytest.raises(DuplicateStatementError, match=r"'p\.0'"):
        await StatementStage().run(ctx)


async def test_stage_does_nothing_without_a_parsed_document() -> None:
    ctx = context(comp("section"))
    ctx.parsed = None
    await StatementStage().run(ctx)
    assert ctx.parsed is None


async def test_stage_cuts_long_statements_from_any_splitter() -> None:
    class OneStatement:
        def split(self, component: Component) -> list[Statement]:
            return [
                Statement(
                    id=f"{component.id}.0",
                    text=component.text,
                    kind="list_item",
                    component_id=component.id,
                    location=LOC,
                )
            ]

    long = "One two three. " * 20
    ctx = context(comp("list_item", long.strip(), cid="li"))
    await StatementStage(OneStatement(), max_chars=100).run(ctx)
    assert ctx.parsed is not None
    pieces = list(ctx.parsed.statements.values())
    assert [s.id for s in pieces] == ["li.0:0", "li.0:1", "li.0:2", "li.0:3"]
    assert all(len(s.text) <= 100 for s in pieces)
    assert " ".join(s.text for s in pieces) == long.strip()
    [event] = ctx.events
    assert (event.stage, event.kind, event.data) == (
        "statements",
        "statements_cut",
        {"pieces": {"li.0": 4}},
    )
    assert event.message == "1 statement(s) over 100 characters were cut into 4 pieces"


async def test_stage_leaves_short_statements_alone_without_an_event() -> None:
    ctx = context(comp("paragraph", "Quick. Cheap.", cid="p"))
    await StatementStage().run(ctx)
    assert ctx.parsed is not None
    assert list(ctx.parsed.statements) == ["p.0", "p.1"]
    assert ctx.events == []


async def test_stage_refuses_a_piece_whose_id_is_taken() -> None:
    clash = Statement(id="li.0:1", text="x", kind="structured", component_id="ld", location=LOC)
    ctx = context(comp("list_item", "word " * 50, cid="li"), [clash])
    with pytest.raises(DuplicateStatementError, match=r"'li\.0:1'"):
        await StatementStage(max_chars=100).run(ctx)


def test_stage_needs_a_positive_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars must be positive, got 0"):
        StatementStage(max_chars=0)


def test_statements_is_a_default_stage_after_layout() -> None:
    names = [s.name for s in default_pipeline().stages]
    assert names.index("layout") < names.index("statements") < names.index("candidates")


# --- the test site, end to end -------------------------------------------------------


@pytest.fixture(scope="module")
def site_statements() -> list[tuple[str, str, list[Statement], Component]]:
    import asyncio

    async def parse_all() -> list[tuple[str, str, list[Statement], Component]]:
        out: list[tuple[str, str, list[Statement], Component]] = []
        for page in render(generate(7)):
            if page.content_type != "text/html":
                continue
            doc = BoilerplateCleaner().clean(
                Document.from_bytes(page.html.encode(), url=f"https://site.test/{page.path}")
            )
            root = await HtmlLayoutParser().parse(doc)
            statements = [s for c in root.walk() for s in DefaultSplitter().split(c)]
            out.append((page.path, page.family, statements, root))
        return out

    return asyncio.run(parse_all())


def test_site_statement_ids_are_unique(
    site_statements: list[tuple[str, str, list[Statement], Component]],
) -> None:
    for path, _, statements, _ in site_statements:
        ids = [s.id for s in statements]
        assert len(ids) == len(set(ids)), path


def test_site_kv_and_listing_pages_give_label_value_pairs(
    site_statements: list[tuple[str, str, list[Statement], Component]],
) -> None:
    for path, family, statements, _ in site_statements:
        if family not in ("kv", "listing", "grid"):
            continue
        pairs = [s for s in statements if s.kind == "key_value"]
        assert pairs, path
        assert all(":" in s.text for s in pairs), path


def test_site_prose_pages_give_one_statement_per_sentence(
    site_statements: list[tuple[str, str, list[Statement], Component]],
) -> None:
    for path, family, statements, _ in site_statements:
        if family != "prose":
            continue
        prose = [s for s in statements if s.kind == "sentence"]
        assert len(prose) >= 3, path
        for s in prose:
            # One sentence each: no sentence-ending punctuation followed by a capital.
            assert not re.search(r"[.!?] [A-Z][a-z]", s.text), (path, s.text)


def test_site_text_is_covered_by_statements(
    site_statements: list[tuple[str, str, list[Statement], Component]],
) -> None:
    """Every word of every non-table component reaches one of its own statements."""
    for path, _, statements, root in site_statements:
        said: dict[str, Counter[str]] = {}
        for s in statements:
            said.setdefault(s.component_id, Counter()).update(s.text.split())
        for c in root.walk():
            if c.type in ("table", "section", "column", "list", "breakout"):
                continue
            own = said.get(c.id, Counter())
            for word in c.text.split():
                assert own[word] > 0, (path, c.id, word)


# --- oversized statements, end to end ------------------------------------------------


class Car(BaseModel):
    """A car."""

    price: Decimal = Field(description="Price", unit="GBP")


FILLER = "word " * 40_000  # 200k characters without a sentence end


def pick_first_candidate(q: Choice) -> str:
    return next(o for o in q.options if o != "none")


@pytest.mark.parametrize(
    "body",
    [
        f"<table><tr><th></th><th>SE</th></tr><tr><th>Notes</th><td>{FILLER}"
        "Price £24,995.</td></tr></table>",
        f"<ul><li>{FILLER}Price £24,995.</li></ul>",
    ],
    ids=["table cell", "list item"],
)
async def test_a_200k_character_statement_is_extracted_without_raising(body: str) -> None:
    html = f"<html><body><main><h1>Kestrova</h1>{body}</main></body></html>"
    fake = (
        FakeJev(default_p=1.0)
        .choice("Which detail", "price", state="24,995")
        .choice("Which of these", pick_first_candidate, state="24,995")
    )
    async with Extractor([Car], jev=fake.client()) as ex:
        result = await ex.extract(Document.from_bytes(html.encode(), url="https://cars.test/"))

    assert result.one(Car).strict() == Car(price=Decimal("24995"))
    [cut] = [e for e in result.meta.events if e.kind == "statements_cut"]
    [pieces] = cut.data["pieces"].values()
    assert pieces >= len(FILLER) // MAX_STATEMENT_CHARS
    stated = [c.state for c in fake.calls if isinstance(c.state, dict) and "statement" in c.state]
    assert stated
    assert max(len(str(state["statement"])) for state in stated) <= MAX_STATEMENT_CHARS


# --- locales (#56) ---------------------------------------------------------------------


class Offre(BaseModel):
    prix: Decimal = Field(description="Price", unit="EUR")
    poids_kg: float = Field(description="Kerb weight", unit="kg")


async def test_no_break_spaces_survive_splitting_so_french_thousands_read_whole() -> None:
    html = "<p>Poids à vide 1&nbsp;234,5 kg. Prix 18&#8239;495 €.</p>".encode()
    root = await HtmlLayoutParser().parse(Document.from_bytes(html, content_type="text/html"))
    [para] = [c for c in root.walk() if c.type == "paragraph"]
    statements = DefaultSplitter().split(para)
    assert [s.text for s in statements] == [
        "Poids à vide 1\u00a0234,5 kg.",
        "Prix 18\u202f495 €.",
    ]
    spec = SchemaSpec.from_model(Offre)
    weight, price = statements
    found = {
        c.raw: normalise(c.raw, c.normalise, spec.field("poids_kg"))
        for c in default_registry().generate(
            weight, spec.field("poids_kg"), schema="Offre", locale="fr-FR"
        )
    }
    assert found["1\u00a0234,5 kg"] == 1234.5
    assert "234,5 kg" not in found
    amounts = [
        (c.raw, normalise(c.raw, c.normalise, spec.field("prix")))
        for c in default_registry().generate(
            price, spec.field("prix"), schema="Offre", locale="fr-FR"
        )
        if c.generator_id == "money"
    ]
    assert amounts == [("18\u202f495 €", Decimal(18495))]


def test_ascii_whitespace_still_collapses_around_a_no_break_space() -> None:
    assert split(comp("paragraph", "Prix\t 18\u00a0495 €  \f TTC.")) == [
        ("Prix 18\u00a0495 € TTC.", "sentence")
    ]


async def test_text_values_read_no_break_spaces_as_plain_spaces() -> None:
    html = b"<h1>A&nbsp;Light in the&nbsp;Attic</h1>"
    root = await HtmlLayoutParser().parse(Document.from_bytes(html, content_type="text/html"))
    [heading] = [c for c in root.walk() if c.type == "heading"]
    [statement] = DefaultSplitter().split(heading)
    assert statement.text == "A\u00a0Light in the\u00a0Attic"

    class Book(BaseModel):
        title: str = Field(description="Title")

    title = SchemaSpec.from_model(Book).field("title")
    [whole] = [
        c
        for c in default_registry().generate(statement, title, schema="Book")
        if c.generator_id == "whole_statement"
    ]
    assert normalise(whole.raw, whole.normalise, title) == "A Light in the Attic"


# --- the page's language (#125) --------------------------------------------------------

GERMAN = "Lieferung z. B. am 3. Mai möglich. Preis auf Anfrage."
GERMAN_SENTENCES = ["Lieferung z. B. am 3. Mai möglich.", "Preis auf Anfrage."]
GERMAN_UNDER_ENGLISH = ["Lieferung z.", "B. am 3.", "Mai möglich.", "Preis auf Anfrage."]


def test_default_splitter_is_locale_aware() -> None:
    assert isinstance(DefaultSplitter(), LocaleAwareSplitter)


def test_split_in_uses_the_locales_language() -> None:
    para = comp("paragraph", GERMAN)
    assert [s.text for s in DefaultSplitter().split(para)] == GERMAN_UNDER_ENGLISH
    assert [s.text for s in DefaultSplitter().split_in(para, "de-AT")] == GERMAN_SENTENCES
    assert [s.text for s in DefaultSplitter().split_in(para, "DE")] == GERMAN_SENTENCES


@pytest.mark.parametrize("locale", [None, ""])
def test_split_in_without_a_locale_keeps_the_splitters_language(locale: str | None) -> None:
    para = comp("paragraph", GERMAN)
    assert [s.text for s in DefaultSplitter("de").split_in(para, locale)] == GERMAN_SENTENCES
    assert [s.text for s in DefaultSplitter().split_in(para, locale)] == GERMAN_UNDER_ENGLISH


def test_split_in_an_unsupported_language_falls_back_to_english() -> None:
    para = comp("paragraph", GERMAN)
    assert [s.text for s in DefaultSplitter("de").split_in(para, "xx-XX")] == GERMAN_UNDER_ENGLISH


PRICE = "Der Wagen kostet ca. 25.000 € inkl. MwSt. und hat 150 PS."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (PRICE, [PRICE]),
        (
            "Preis 25.000 € zzgl. MwSt. und Überführung.",
            ["Preis 25.000 € zzgl. MwSt. und Überführung."],
        ),
        # German capitalises nouns, so a capitalised word can continue the sentence.
        (
            "Preis zzgl. Überführung und inkl. Garantie.",
            ["Preis zzgl. Überführung und inkl. Garantie."],
        ),
        ("Verbrauch ca. 5 l, max. Leistung 150 PS.", ["Verbrauch ca. 5 l, max. Leistung 150 PS."]),
        ("Preis inkl. Versand.", ["Preis inkl. Versand."]),
        ("Ein Paket evtl. Ende Mai.", ["Ein Paket evtl. Ende Mai."]),
        # An article or pronoun after one still starts a new sentence.
        ("Das kostet 5 € max. Der Rest ist frei.", ["Das kostet 5 € max.", "Der Rest ist frei."]),
        # So does one after a noun abbreviation, which can end a sentence.
        (
            "Der Preis beträgt 300 € zzgl. USt. Lieferung frei.",
            ["Der Preis beträgt 300 € zzgl. USt.", "Lieferung frei."],
        ),
        # English abbreviations keep the English rule on a German page.
        ("Fahrzeit 5 min. Danach Pause.", ["Fahrzeit 5 min.", "Danach Pause."]),
        # A word that isn't an abbreviation still ends the sentence.
        (
            "Ausstattung inkl. Navi. Danach kam der Test.",
            ["Ausstattung inkl. Navi.", "Danach kam der Test."],
        ),
    ],
)
def test_german_mid_sentence_abbreviations_dont_split(text: str, expected: list[str]) -> None:
    assert sentences(text, language="de") == expected


def test_language_abbreviations_add_to_the_english_ones() -> None:
    assert {"inkl", "zzgl", "ggf", "bzw", "evtl", "max", "nr", "mwst"} <= LANGUAGE_ABBREVIATIONS[
        "de"
    ]
    assert "inkl" not in ABBREVIATIONS
    # An English abbreviation is still repaired on a German page.
    assert sentences("Preis 300 € excl. VAT ab Werk.", language="de") == [
        "Preis 300 € excl. VAT ab Werk."
    ]


@pytest.mark.parametrize("language", ["en", "xx", "fr"])
def test_other_languages_dont_use_the_german_abbreviations(language: str) -> None:
    assert sentences(PRICE, language=language) == [
        "Der Wagen kostet ca. 25.000 € inkl.",
        "MwSt.",
        "und hat 150 PS.",
    ]
    assert sentences("It costs £300 incl. Delivery is free.", language=language) == [
        "It costs £300 incl.",
        "Delivery is free.",
    ]


def test_split_in_keeps_german_abbreviations_in_the_sentence() -> None:
    para = comp("paragraph", f"{PRICE} Lieferung zzgl. Überführung.")
    assert [s.text for s in DefaultSplitter().split_in(para, "de-DE")] == [
        PRICE,
        "Lieferung zzgl. Überführung.",
    ]


async def page_context(
    html: str, *, content_language: str | None = None, pipeline: Pipeline | None = None
) -> Context:
    document = Document.from_bytes(
        html.encode(), content_type="text/html", content_language=content_language
    )
    ctx = Context.create(document, [SchemaSpec.from_model(VehicleSpec)], FakeJev().client())
    ctx.pipeline = pipeline
    ctx.parsed = ParsedDocument(document=document, root=await HtmlLayoutParser().parse(document))
    return ctx


def texts(ctx: Context) -> list[str]:
    assert ctx.parsed is not None
    return [s.text for s in ctx.parsed.statements.values()]


async def test_a_german_page_is_split_under_its_own_language() -> None:
    ctx = await page_context(f'<html lang="de-DE"><body><p>{GERMAN}</p></body></html>')
    await StatementStage().run(ctx)
    assert texts(ctx) == GERMAN_SENTENCES


async def test_the_content_language_header_picks_the_language() -> None:
    ctx = await page_context(f"<p>{GERMAN}</p>", content_language="de-CH, en")
    await StatementStage().run(ctx)
    assert texts(ctx) == GERMAN_SENTENCES


async def test_a_page_without_a_language_is_split_by_the_stages_locale() -> None:
    ctx = await page_context(f"<p>{GERMAN}</p>")
    await StatementStage(locale="de-DE").run(ctx)
    assert texts(ctx) == GERMAN_SENTENCES


async def test_the_pages_language_wins_over_the_stages_locale() -> None:
    ctx = await page_context(f'<html lang="en-GB"><body><p>{GERMAN}</p></body></html>')
    await StatementStage(locale="de-DE").run(ctx)
    assert texts(ctx) == GERMAN_UNDER_ENGLISH


async def test_without_a_stage_locale_the_candidate_stages_is_used() -> None:
    pipeline = Pipeline([StatementStage(), CandidateStage(locale="de-DE")])
    ctx = await page_context(f"<p>{GERMAN}</p>", pipeline=pipeline)
    await StatementStage().run(ctx)
    assert texts(ctx) == GERMAN_SENTENCES

    ctx = await page_context(f"<p>{GERMAN}</p>", pipeline=pipeline)
    await StatementStage(locale="en-GB").run(ctx)
    assert texts(ctx) == GERMAN_UNDER_ENGLISH


async def test_a_page_without_any_language_is_split_as_english() -> None:
    ctx = await page_context(f"<p>{GERMAN}</p>", pipeline=Pipeline([CandidateStage()]))
    await StatementStage().run(ctx)
    assert texts(ctx) == GERMAN_UNDER_ENGLISH


async def test_a_page_in_a_language_pysbd_lacks_is_split_as_english() -> None:
    ctx = await page_context(f'<html lang="xx"><body><p>{GERMAN}</p></body></html>')
    await StatementStage(DefaultSplitter("de")).run(ctx)
    assert texts(ctx) == GERMAN_UNDER_ENGLISH


async def test_a_locale_aware_splitter_gets_the_documents_tag() -> None:
    seen: list[str | None] = []

    class Recording:
        def split(self, component: Component) -> list[Statement]:
            raise AssertionError("split_in should be called instead")

        def split_in(self, component: Component, locale: str | None) -> list[Statement]:
            seen.append(locale)
            return []

    ctx = await page_context('<html lang="de-AT"><body><p>Hallo.</p></body></html>')
    await StatementStage(Recording()).run(ctx)
    assert set(seen) == {"de-AT"}

    seen.clear()
    ctx = await page_context("<p>Hallo.</p>")
    await StatementStage(Recording()).run(ctx)
    assert set(seen) == {None}


async def test_the_default_pipeline_splits_a_cleaned_german_page_by_its_language() -> None:
    html = f'<html lang="de"><body><h1>Delmaro</h1><p>{GERMAN}</p></body></html>'
    document = Document.from_bytes(html.encode(), content_type="text/html")
    ctx = Context.create(document, [SchemaSpec.from_model(VehicleSpec)], FakeJev().client())
    wanted = {"clean", "layout", "statements"}
    await Pipeline([s for s in default_pipeline() if s.name in wanted]).run(ctx)
    assert texts(ctx) == ["Delmaro", *GERMAN_SENTENCES]

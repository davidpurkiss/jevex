import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from jevex import (
    BoilerplateCleaner,
    Component,
    Context,
    DefaultSplitter,
    Document,
    DomLocation,
    SchemaSpec,
    Statement,
    StatementStage,
)
from jevex.extractor import default_pipeline
from jevex.interfaces import ParsedDocument, StatementSplitter
from jevex.layout_html import HtmlLayoutParser
from jevex.split import DuplicateStatementError, is_key_value, sentences
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

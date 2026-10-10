import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from jevex.baselines import (
    BaselineInput,
    BaselineOutput,
    BaselineRunError,
    BaselineSetup,
    LLMBaseline,
    ResultRow,
    baseline_instructions,
    charge_usage,
    found_records,
    lenient_records,
    load_prompt,
    pinned_llm,
    prepare_input,
    read_inputs,
    read_results,
    records_model,
    render_text,
    run_baseline,
    schema_specs,
    schemas_text,
    score_results,
    summarise_results,
    write_inputs,
)
from jevex.benchmarks import PinnedModel, book_values
from jevex.clean import html_text_of
from jevex.document import Document
from jevex.examples.books import Book, books_pipeline
from jevex.extractor import default_pipeline
from jevex.layout import Component, DomLocation, LayoutStage
from jevex.llm import (
    LLMBudgetExceededError,
    LLMError,
    LLMImage,
    LLMResponse,
    LLMUsage,
)
from jevex.locales import document_locale
from jevex.schema import Field
from jevex.testing import FakeLLM
from jevex.testsite.schemas import Listing, VehicleSpec

ROOT = Path(__file__).parent.parent
PROMPT = ROOT / "benchmarks" / "baselines" / "prompt-v1.md"
BOOKS = Path(__file__).parent / "fixtures" / "books"
PAGES = ("a-light-in-the-attic_1000", "sapiens-a-brief-history-of-humankind_996")
BOOK_SPECS = schema_specs([Book])

HAIKU = PinnedModel(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    input_usd_per_mtok=1.0,
    output_usd_per_mtok=5.0,
    price_date=date(2026, 10, 1),
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two books.toscrape.com pages, labelled from their markup like the books corpus."""
    root = tmp_path / "corpus"
    (root / "pages").mkdir(parents=True)
    pages: list[dict[str, Any]] = []
    for name in PAGES:
        shutil.copy(BOOKS / f"{name}.html", root / "pages" / f"{name}.html")
        values = book_values(html_text_of((BOOKS / f"{name}.html").read_bytes()))
        pages.append(
            {
                "path": f"pages/{name}.html",
                "schema": "Book",
                "records": [{"entity": "document", "values": values}],
            }
        )
    (root / "truth.json").write_text(json.dumps({"pages": pages}))
    return root


def truth(corpus: Path) -> dict[str, dict[str, Any]]:
    pages = json.loads((corpus / "truth.json").read_text())["pages"]
    return {p["path"]: p["records"][0]["values"] for p in pages}


def answer_from_page(prompt: str, schema: type[BaseModel]) -> dict[str, Any]:
    """Answers each book page correctly, as an LLM that read it perfectly would."""
    for name in PAGES:
        values = book_values(html_text_of((BOOKS / f"{name}.html").read_bytes()))
        if f"# {values['title']}\n" in prompt:
            return {"Book": [values | {"price": float(values["price"])}]}
    return {"Book": []}


# --- input -----------------------------------------------------------------------------


def _component(id: str, type: Any, text: str = "", children: list[Component] | None = None):
    return Component(
        id=id, type=type, text=text, children=children or [], location=DomLocation(dom_path=id)
    )


def test_render_text_writes_one_block_per_component_in_reading_order() -> None:
    root = _component(
        "root",
        "section",
        children=[
            _component("h", "heading", "Spec"),
            _component("p", "paragraph", "  A small car.  "),
            _component(
                "l",
                "list",
                children=[_component("i1", "list_item", "Five seats"), _component("e", "list")],
            ),
            _component("t", "table", "Power | 85 kW\nSeats | 5"),
            _component("img", "image", "Badge", [_component("o", "paragraph", "SE L")]),
            _component("blank", "image"),
        ],
    )
    assert render_text(root) == (
        "# Spec\n\nA small car.\n\n- Five seats\n\nPower | 85 kW\nSeats | 5\n\n"
        "[Image: Badge]\n\nSE L"
    )


async def test_prepare_input_cleans_and_lays_out_with_the_given_pipeline() -> None:
    document = Document.from_path(BOOKS / "tipping-the-velvet_999.html")
    with_site_cleaner = await prepare_input(document, books_pipeline())
    assert "# Tipping the Velvet" in with_site_cleaner.text
    assert "Rating: One out of five stars" in with_site_cleaner.text
    assert b"Rating: One out of five stars" in with_site_cleaner.document.content
    assert "Price (incl. tax) | £53.74" in with_site_cleaner.text
    # jevex's default pipeline has no star-rating cleaner, so the rating isn't text.
    plain = await prepare_input(document)
    assert "Rating:" not in plain.text
    assert "Tipping the Velvet" in plain.text


async def test_prepare_input_needs_a_layout_parser() -> None:
    document = Document.from_bytes(b"%PDF-1.4\n", content_type="application/pdf")
    no_parsers = default_pipeline().replace("layout", LayoutStage(parsers=[]))
    with pytest.raises(BaselineRunError, match="no layout parser supports application/pdf"):
        await prepare_input(document, no_parsers)


@dataclass
class AsksJev:
    name: str = "clean"

    async def run(self, ctx: Any) -> None:
        from jevex.jev import Noul

        await ctx.jev.ask({"content": "x"}, {"q": Noul(instructions="Is this a page?")})


async def test_prepare_input_never_asks_jev() -> None:
    document = Document.from_path(BOOKS / "tipping-the-velvet_999.html")
    with pytest.raises(Exception, match="asked Jev a question"):
        await prepare_input(document, default_pipeline().replace("clean", AsksJev()))


# --- instructions and output -----------------------------------------------------------


def test_the_committed_prompt_writes_out_the_schemas() -> None:
    text = baseline_instructions(load_prompt(PROMPT), BOOK_SPECS)
    assert text.startswith("Extract structured records from the document.")
    assert "{schemas}" not in text
    assert schemas_text(BOOK_SPECS) in text


def test_schemas_text_lists_each_field_with_its_unit_and_type() -> None:
    assert schemas_text(BOOK_SPECS) == (
        "## Book\n"
        "A book for sale in an online bookshop.\n"
        "- title: Title of the book. Text.\n"
        "- price: Price, in GBP. A number.\n"
        "- in_stock: The book is in stock. True or false.\n"
        "- stock_count: Number of copies available. A number.\n"
        '- rating: Star rating out of five. One of "One", "Two", "Three", "Four", "Five".'
    )


class Engine(BaseModel):
    size_cc: int | None = Field(description="Engine displacement", unit="cc")


class Car(BaseModel):
    """A car."""

    registered: date = Field(description="Date of first registration")
    extras: list[str] = Field(description="Optional extras")
    engine: Engine = Field(description="The engine")


def test_schemas_text_covers_dates_lists_and_nested_models() -> None:
    assert schemas_text(schema_specs([Car])) == (
        "## Car\n"
        "A car.\n"
        "- registered: Date of first registration. A date, as YYYY-MM-DD.\n"
        "- extras: Optional extras. A list, each item text.\n"
        "- engine: The engine. An object."
    )


def test_load_prompt_rejects_a_template_without_one_schemas_placeholder(tmp_path: Path) -> None:
    for text in ("No placeholder", "{schemas} and {schemas}"):
        path = tmp_path / "prompt.md"
        path.write_text(text)
        with pytest.raises(BaselineRunError, match="exactly once"):
            load_prompt(path)
    with pytest.raises(BaselineRunError, match="can't read prompt"):
        load_prompt(tmp_path / "missing.md")


def test_records_model_has_an_optional_list_per_schema() -> None:
    model = records_model(schema_specs([VehicleSpec, Listing]))
    assert model is records_model(schema_specs([VehicleSpec, Listing]))
    empty = model.model_validate({})
    assert found_records(empty) == {"VehicleSpec": [], "Listing": []}
    parsed = model.model_validate(
        {"VehicleSpec": [{"make": "Ford", "fuel_type": "ev", "power_kw": 85, "automatic": True}]}
    )
    assert found_records(parsed)["VehicleSpec"] == [
        {
            "entity": "1",
            "values": {"make": "Ford", "fuel_type": "ev", "power_kw": 85.0, "automatic": True},
        }
    ]
    with pytest.raises(ValidationError):
        model.model_validate({"VehicleSpec": [{"fuel_type": "steam"}]})
    schema = model.model_json_schema()
    record = schema["$defs"]["VehicleSpecRecord"]
    assert record["properties"]["power_kw"]["description"] == "Maximum power output, in kW"
    assert "required" not in record


def test_records_model_takes_dates_as_text_and_nested_models_as_partials() -> None:
    model = records_model(schema_specs([Car]))
    parsed = model.model_validate(
        {"Car": [{"registered": "2021-03-01", "extras": ["Tow bar"], "engine": {}}]}
    )
    assert found_records(parsed) == {
        "Car": [
            {
                "entity": "1",
                "values": {
                    "registered": "2021-03-01",
                    "extras": ["Tow bar"],
                    "engine": {"size_cc": None},
                },
            }
        ]
    }


def test_found_records_drops_records_with_no_values() -> None:
    model = records_model(BOOK_SPECS)
    output = model.model_validate({"Book": [{}, {"title": "Dune"}, {"rating": "Two"}]})
    assert found_records(output) == {
        "Book": [
            {"entity": "1", "values": {"title": "Dune"}},
            {"entity": "2", "values": {"rating": "Two"}},
        ]
    }


# --- the LLM-only system ---------------------------------------------------------------


async def test_llm_baseline_sends_the_instructions_then_the_document() -> None:
    llm = FakeLLM(answer_from_page, model="claude-haiku-4-5-20251001", price=(1.0, 5.0))
    system = LLMBaseline(llm, BOOK_SPECS, "Extract books.")
    source = BaselineInput(Document.from_bytes(b"<p>x</p>"), "# A Light in the Attic\n\n£51.77")
    output = await system.extract(source)
    assert llm.calls[0].prompt == (
        "Extract books.\n\n<document>\n# A Light in the Attic\n\n£51.77\n</document>"
    )
    assert llm.calls[0].schema is records_model(BOOK_SPECS)
    assert output.records["Book"][0]["values"]["title"] == "A Light in the Attic"
    assert output.calls == 1
    assert output.model == "claude-haiku-4-5-20251001"
    assert output.cost == pytest.approx(
        (output.input_tokens * 1.0 + output.output_tokens * 5.0) / 1_000_000
    )
    assert output.cost > 0


@dataclass
class UnpricedLLM:
    calls: int = 0

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        self.calls += 1
        return LLMResponse(schema.model_validate({}), LLMUsage(10, 5, None), "mystery-model")


async def test_llm_baseline_refuses_a_model_without_a_price() -> None:
    system = LLMBaseline(UnpricedLLM(), BOOK_SPECS, "Extract books.")
    source = BaselineInput(Document.from_bytes(b"<p>x</p>"), "x")
    with pytest.raises(BaselineRunError, match="no price for mystery-model"):
        await system.extract(source)


def test_pinned_llm_builds_the_providers_adapter_at_the_pinned_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jevex.llm.anthropic import AnthropicLLM
    from jevex.llm.gemini import GeminiLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    claude = pinned_llm(HAIKU)
    assert isinstance(claude, AnthropicLLM)
    assert claude.model == "claude-haiku-4-5-20251001"
    assert claude.fallbacks is False
    assert claude.prices["claude-haiku-4-5-20251001"].output == 5.0
    flash = HAIKU.model_copy(update={"provider": "gemini", "model": "gemini-3.8-flash"})
    gemini = pinned_llm(flash)
    assert isinstance(gemini, GeminiLLM)
    assert gemini.prices["gemini-3.8-flash"].input == 1.0
    with pytest.raises(ValueError, match="is Jev, not an LLM"):
        pinned_llm(HAIKU.model_copy(update={"provider": "jev"}))


def test_charge_records_a_tools_usage_at_the_pinned_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm_spent: Callable[[], float]
) -> None:
    ledger = tmp_path / "ledger"
    monkeypatch.setenv("JEVEX_SPEND_LEDGER", str(ledger))
    assert charge_usage(HAIKU, 1_000_000, 100_000) == pytest.approx(1.5)
    assert llm_spent() == pytest.approx(1.5)
    assert ledger.read_text() == "llm 1.500000000\n"


# --- running ---------------------------------------------------------------------------


@dataclass
class Scripted:
    """A system that answers per page heading (its book's title), or raises what it's
    given. Other titles appear on each page, under "Products you recently viewed"."""

    outcomes: dict[str, object] = field(default_factory=dict[str, object])
    name: str = "scripted"
    seen: list[str] = field(default_factory=list[str])

    async def extract(self, source: BaselineInput) -> BaselineOutput:
        for title, outcome in self.outcomes.items():
            if f"# {title}" in source.text:
                self.seen.append(title)
                if isinstance(outcome, BaseException):
                    raise outcome
                assert isinstance(outcome, BaselineOutput)
                return outcome
        raise AssertionError("unscripted document")


def _output(title: str, cost: float = 0.01) -> BaselineOutput:
    records = {"Book": [{"entity": "1", "values": {"title": title}}]}
    return BaselineOutput(records, calls=1, input_tokens=100, output_tokens=20, cost=cost)


async def test_run_baseline_writes_a_row_per_document(corpus: Path, tmp_path: Path) -> None:
    llm = FakeLLM(answer_from_page, model="claude-haiku-4-5-20251001", price=(1.0, 5.0))
    system = LLMBaseline(llm, BOOK_SPECS, baseline_instructions(load_prompt(PROMPT), BOOK_SPECS))
    out = tmp_path / "results.jsonl"
    rows = await run_baseline(system, corpus, out, pipeline=books_pipeline(), concurrency=2)
    assert [r.path for r in rows] == [f"pages/{n}.html" for n in PAGES]
    assert sorted(r.path for r in read_results(out)) == sorted(r.path for r in rows)
    assert all(r.error is None and r.calls == 1 and r.cost > 0 for r in rows)
    assert all(r.model == "claude-haiku-4-5-20251001" for r in rows)
    # The site cleaner ran: the rating reached the model as text.
    assert all("out of five stars" in c.prompt for c in llm.calls)
    report = score_results(corpus, read_results(out), BOOK_SPECS)
    assert report.overall().accuracy == 1.0
    assert report.summary()["llm_calls_per_document"] == 1
    assert report.summary()["cost_per_document"] == pytest.approx(sum(r.cost for r in rows) / 2)
    assert report.summary()["resolution_mix"] == {"llm": 10}
    assert summarise_results(rows).startswith("2 documents, 0 failed, $")


async def test_run_baseline_records_a_failed_document_and_carries_on(
    corpus: Path, tmp_path: Path
) -> None:
    system = Scripted(
        {"A Light in the Attic": LLMError("bad output"), "Sapiens": _output("Sapiens")}
    )
    rows = await run_baseline(system, corpus, tmp_path / "out.jsonl", concurrency=1)
    failed, ok = rows
    assert failed.error == "LLMError: bad output"
    assert failed.records == {}
    assert failed.cost == 0.0
    assert ok.error is None
    assert ok.cost == 0.01
    report = score_results(corpus, rows, BOOK_SPECS)
    assert report.documents[0].error == "LLMError: bad output"
    assert report.documents[0].fields["Book.title"].missing == 1
    assert [d.path for d in report.failed] == [str(corpus / "pages" / f"{PAGES[0]}.html")]


async def test_run_baseline_stops_at_the_spend_cap_keeping_finished_rows(
    corpus: Path, tmp_path: Path
) -> None:
    system = Scripted(
        {
            "A Light in the Attic": _output("A Light in the Attic"),
            "Sapiens": LLMBudgetExceededError("cap"),
        }
    )
    out = tmp_path / "out.jsonl"
    with pytest.raises(LLMBudgetExceededError):
        await run_baseline(system, corpus, out, concurrency=1)
    assert [r.path for r in read_results(out)] == [f"pages/{PAGES[0]}.html"]


async def test_run_baseline_checks_the_cap_before_each_document(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "0")
    system = Scripted({"A Light in the Attic": _output("x"), "Sapiens": _output("y")})
    with pytest.raises(LLMBudgetExceededError):
        await run_baseline(system, corpus, tmp_path / "out.jsonl")
    assert system.seen == []


async def test_run_baseline_wont_overwrite_results(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    out.write_text("")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        await run_baseline(Scripted(), corpus, out)


# --- scoring ---------------------------------------------------------------------------


def _row(path: str, values: dict[str, Any], **kwargs: Any) -> ResultRow:
    return ResultRow(
        path=path,
        records={"Book": [{"entity": "1", "values": values}]},
        seconds=1.0,
        **kwargs,
    )


def test_score_results_scores_like_jevex_eval(corpus: Path) -> None:
    expected = truth(corpus)
    first, second = (f"pages/{n}.html" for n in PAGES)
    wrong = {"rating": "One"}
    rows = [
        _row(first, expected[first] | {"price": 51.77}, calls=1, cost=0.002),
        # One wrong value and one missing.
        _row(second, {k: v for k, v in expected[second].items() if k != "title"} | wrong),
    ]
    report = score_results(corpus, rows, BOOK_SPECS)
    scores = report.field_scores()
    assert scores["Book.price"].correct == 2
    assert scores["Book.title"].missing == 1
    assert scores["Book.rating"].wrong == 1
    assert report.overall().correct == 8
    assert report.summary()["cost_per_document"] == pytest.approx(0.001)
    assert report.summary()["jev_cost_per_document"] == 0


def test_score_results_counts_a_document_without_a_row_as_missing(corpus: Path) -> None:
    first = f"pages/{PAGES[0]}.html"
    report = score_results(corpus, [_row(first, truth(corpus)[first])], BOOK_SPECS)
    missing = report.documents[1]
    assert missing.error == "no result"
    assert missing.fields["Book.title"].missing == 1


def test_score_results_counts_another_schemas_records_as_spurious(corpus: Path) -> None:
    first = f"pages/{PAGES[0]}.html"
    row = ResultRow(
        path=first,
        records={"Listing": [{"entity": "1", "values": {"make": "Ford"}}]},
        seconds=1.0,
    )
    report = score_results(corpus, [row], schema_specs([Book, Listing]))
    assert report.documents[0].fields["Listing.make"].spurious == 1


def test_score_results_rejects_rows_the_corpus_doesnt_match(corpus: Path) -> None:
    first = f"pages/{PAGES[0]}.html"
    with pytest.raises(BaselineRunError, match="two results for"):
        score_results(corpus, [_row(first, {}), _row(first, {})], BOOK_SPECS)
    with pytest.raises(BaselineRunError, match="doesn't list"):
        score_results(corpus, [_row("pages/other.html", {})], BOOK_SPECS)
    with pytest.raises(ValueError, match="schemas the extractor doesn't have"):
        score_results(corpus, [], schema_specs([Listing]))


def test_read_results_names_a_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(_row("a.html", {}).model_dump_json() + '\n\n{"path": 1}\n')
    with pytest.raises(BaselineRunError, match="line 3 isn't a result"):
        read_results(path)
    with pytest.raises(BaselineRunError, match="can't read results"):
        read_results(tmp_path / "missing.jsonl")


def test_baseline_setup_gives_its_records_model() -> None:
    setup = BaselineSetup(BOOK_SPECS, "Extract books.", HAIKU)
    assert setup.records_model is records_model(BOOK_SPECS)


def test_found_values_keep_json_types_for_scoring() -> None:
    output = records_model(BOOK_SPECS).model_validate({"Book": [{"price": Decimal("51.77")}]})
    assert found_records(output)["Book"][0]["values"] == {"price": 51.77}


def test_lenient_records_keeps_the_values_that_fit_their_fields() -> None:
    raw = {
        "Book": [
            {"title": "Dune", "price": "51.77", "rating": "NA", "stock_count": "many"},
            {"price": "N/A", "in_stock": None},
            "not a record",
            {"title": "Emma", "isbn": "123", "in_stock": "true"},
        ],
        "Magazine": [{"title": "Wired"}],
    }
    assert lenient_records(raw, BOOK_SPECS) == {
        "Book": [
            {"entity": "1", "values": {"title": "Dune", "price": 51.77}},
            {"entity": "2", "values": {"title": "Emma", "in_stock": True}},
        ]
    }
    assert lenient_records({"Book": "nothing"}, BOOK_SPECS) == {"Book": []}


# --- prepared inputs -------------------------------------------------------------------


async def test_prepared_inputs_give_every_system_the_same_input(
    corpus: Path, tmp_path: Path
) -> None:
    inputs_file = tmp_path / "inputs.jsonl"
    rows = await write_inputs(corpus, inputs_file, pipeline=books_pipeline())
    assert [r.path for r in rows] == [f"pages/{n}.html" for n in PAGES]
    inputs = read_inputs(inputs_file)
    first = inputs[f"pages/{PAGES[0]}.html"]
    assert first.text.startswith("- Home")
    assert "Rating: Three out of five stars" in first.text
    assert first.document.content_type == "text/html"
    with pytest.raises(ValueError, match="refusing to overwrite an inputs file"):
        await write_inputs(corpus, inputs_file)

    llm = FakeLLM(answer_from_page, model="claude-haiku-4-5-20251001", price=(1.0, 5.0))
    system = LLMBaseline(llm, BOOK_SPECS, "Extract books.")
    out = tmp_path / "results.jsonl"
    await run_baseline(system, corpus, out, inputs=inputs)
    assert sorted(c.prompt for c in llm.calls) == sorted(
        f"Extract books.\n\n<document>\n{i.text}\n</document>" for i in inputs.values()
    )
    assert score_results(corpus, read_results(out), BOOK_SPECS).overall().accuracy == 1.0


async def test_prepared_inputs_keep_the_manifests_locale(corpus: Path, tmp_path: Path) -> None:
    truth = json.loads((corpus / "truth.json").read_text())
    truth["pages"][0]["locale"] = "de_de"
    (corpus / "truth.json").write_text(json.dumps(truth))
    await write_inputs(corpus, tmp_path / "inputs.jsonl", pipeline=books_pipeline())
    inputs = read_inputs(tmp_path / "inputs.jsonl")
    assert document_locale(inputs[f"pages/{PAGES[0]}.html"].document) == "de-DE"
    assert inputs[f"pages/{PAGES[1]}.html"].document.locale is None


async def test_run_baseline_needs_inputs_for_every_document(corpus: Path, tmp_path: Path) -> None:
    source = BaselineInput(Document.from_bytes(b"<p>x</p>"), "x")
    out = tmp_path / "out.jsonl"
    with pytest.raises(BaselineRunError, match="the inputs have no pages/sapiens"):
        await run_baseline(Scripted(), corpus, out, inputs={f"pages/{PAGES[0]}.html": source})
    assert not out.exists()


@dataclass
class BrokenLayout:
    """Fails on the first book's page, as a parser can on a malformed document."""

    name: str = "layout"

    async def run(self, ctx: Any) -> None:
        if b"A Light in the Attic</h1>" in ctx.document.content:
            raise ValueError("can't lay this out")
        await LayoutStage().run(ctx)


async def test_run_baseline_records_a_document_it_cant_prepare(
    corpus: Path, tmp_path: Path
) -> None:
    system = Scripted({"Sapiens": _output("Sapiens")})
    pipeline = default_pipeline().replace("layout", BrokenLayout())
    rows = await run_baseline(
        system, corpus, tmp_path / "out.jsonl", pipeline=pipeline, concurrency=1
    )
    assert rows[0].error == "ValueError: can't lay this out"
    assert rows[1].error is None


async def test_run_baseline_stops_when_no_parser_reads_a_document(
    corpus: Path, tmp_path: Path
) -> None:
    pipeline = default_pipeline().replace("layout", LayoutStage(parsers=[]))
    with pytest.raises(BaselineRunError, match="no layout parser supports text/html"):
        await run_baseline(Scripted(), corpus, tmp_path / "out.jsonl", pipeline=pipeline)


def test_read_inputs_names_a_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "inputs.jsonl"
    path.write_text('{"path": "a.html"}\n')
    with pytest.raises(BaselineRunError, match="line 1 isn't an input"):
        read_inputs(path)


# --- the tool scripts ------------------------------------------------------------------

SCRIPTS = ROOT / "benchmarks" / "baselines"
TOOLS = {"crawl4ai_baseline": "Crawl4AIBaseline", "scrapegraphai_baseline": "ScrapeGraphAIBaseline"}


def _script(name: str) -> Any:
    """A tool script as a module. They import their tool only when extracting."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_bench_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", TOOLS)
def test_tool_scripts_pin_their_tool_and_lock_their_environment(name: str) -> None:
    text = (SCRIPTS / f"{name}.py").read_text()
    metadata = text[text.index("# /// script") : text.index("# ///\n", 12)]
    pins = [line for line in metadata.splitlines() if "==" in line]
    assert pins, "the tool's version must be pinned exactly"
    assert '# jevex = { path = "../..", editable = true }' in metadata
    assert (SCRIPTS / f"{name}.py.lock").is_file()


@pytest.mark.parametrize("name", TOOLS)
def test_tool_scripts_give_the_tool_html(name: str) -> None:
    module = _script(name)
    html = Document.from_bytes(b"<html><body><h1>Dune</h1></body></html>")
    assert module.page_html(BaselineInput(html, "# Dune")) == (
        "<html><body><h1>Dune</h1></body></html>"
    )
    pdf = Document.from_bytes(b"%PDF-1.4\n", content_type="application/pdf")
    assert module.page_html(BaselineInput(pdf, "Price | <£5>")) == (
        "<html><body><pre>Price | &lt;£5&gt;</pre></body></html>"
    )


@pytest.mark.parametrize("name", TOOLS)
def test_tool_scripts_refuse_a_provider_they_dont_run(name: str) -> None:
    module = _script(name)
    other = HAIKU.model_copy(update={"provider": "litellm"})
    system: Any = getattr(module, TOOLS[name])
    with pytest.raises(ValueError, match="doesn't run litellm models"):
        system(BaselineSetup(BOOK_SPECS, "Extract books.", other))
    assert system(BaselineSetup(BOOK_SPECS, "Extract books.", HAIKU)).name in name


@pytest.mark.live
async def test_live_llm_baseline_costs_the_tokens_the_api_reports(corpus: Path) -> None:
    import os

    from jevex.benchmarks import BenchmarkConfig

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    pinned = BenchmarkConfig.load(ROOT / "benchmarks" / "config.yaml").models.baseline_fast
    llm = pinned_llm(pinned)
    system = LLMBaseline(llm, BOOK_SPECS, baseline_instructions(load_prompt(PROMPT), BOOK_SPECS))
    page = f"pages/{PAGES[0]}.html"
    source = await prepare_input(Document.from_path(corpus / page), books_pipeline())
    try:
        output = await system.extract(source)
    finally:
        close = getattr(llm, "aclose", None)  # the LLM protocol has no aclose
        if close is not None:
            await close()
    assert output.model == pinned.model
    assert output.input_tokens > 0
    assert output.output_tokens > 0
    assert output.cost == pytest.approx(pinned.cost(output.input_tokens, output.output_tokens))
    row = ResultRow(path=page, records=output.records, seconds=1.0, cost=output.cost)
    report = score_results(corpus, [row], BOOK_SPECS)
    assert report.documents[0].fields["Book.title"].correct == 1


# The tools themselves aren't installed here: these stubs stand in for the parts of their
# APIs the scripts use, to test what the scripts do with what the tools return.


@dataclass
class _TokenUsage:
    prompt_tokens: int
    completion_tokens: int


class _Crawl4AI:
    """``crawl4ai``: the crawl returns ``blocks`` and records ``usages`` on the strategy."""

    blocks: list[Any] = []  # noqa: RUF012
    usages: list[_TokenUsage] = []  # noqa: RUF012
    fail: Exception | None = None

    class LLMConfig:
        def __init__(self, provider: str, api_token: str) -> None:
            self.provider, self.api_token = provider, api_token

    class LLMExtractionStrategy:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.usages: list[_TokenUsage] = []

    class CrawlerRunConfig:
        def __init__(self, extraction_strategy: Any, cache_mode: Any) -> None:
            self.strategy = extraction_strategy

    class CacheMode:
        BYPASS = "bypass"

    class AsyncHTTPCrawlerStrategy:
        pass

    class AsyncWebCrawler:
        def __init__(self, crawler_strategy: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: object) -> None:
            pass

        async def arun(self, url: str, config: Any) -> Any:
            assert url.startswith("raw:<html>")
            config.strategy.usages.extend(_Crawl4AI.usages)
            if _Crawl4AI.fail is not None:
                raise _Crawl4AI.fail
            content = json.dumps(_Crawl4AI.blocks)
            return type("Result", (), {"success": True, "extracted_content": content})()


@pytest.fixture
def crawl4ai(monkeypatch: pytest.MonkeyPatch) -> Any:
    import sys
    import types

    module = types.ModuleType("crawl4ai")
    for name in (
        "LLMConfig",
        "LLMExtractionStrategy",
        "CrawlerRunConfig",
        "CacheMode",
        "AsyncWebCrawler",
    ):
        setattr(module, name, getattr(_Crawl4AI, name))
    strategies = types.ModuleType("crawl4ai.async_crawler_strategy")
    setattr(strategies, "AsyncHTTPCrawlerStrategy", _Crawl4AI.AsyncHTTPCrawlerStrategy)  # noqa: B010
    monkeypatch.setitem(sys.modules, "crawl4ai", module)
    monkeypatch.setitem(sys.modules, "crawl4ai.async_crawler_strategy", strategies)
    monkeypatch.setattr(_Crawl4AI, "fail", None)
    return _script("crawl4ai_baseline")


HTML_INPUT = BaselineInput(Document.from_bytes(b"<html><h1>Dune</h1></html>"), "# Dune")


async def test_crawl4ai_merges_blocks_and_charges_every_call(
    crawl4ai: Any, monkeypatch: pytest.MonkeyPatch, llm_spent: Callable[[], float]
) -> None:
    monkeypatch.setattr(
        _Crawl4AI,
        "blocks",
        [
            {"Book": [{"title": "Dune", "price": "9.99", "rating": "NA"}], "error": False},
            {"Book": [{"title": "Emma"}]},
            {"index": 2, "error": True, "tags": ["error"], "content": "unparsed: <score>"},
        ],
    )
    monkeypatch.setattr(_Crawl4AI, "usages", [_TokenUsage(1000, 100), _TokenUsage(500, 50)])
    system = crawl4ai.Crawl4AIBaseline(BaselineSetup(BOOK_SPECS, "Extract books.", HAIKU))
    output = await system.extract(HTML_INPUT)
    assert output.records == {
        "Book": [
            {"entity": "1", "values": {"title": "Dune", "price": 9.99}},
            {"entity": "2", "values": {"title": "Emma"}},
        ]
    }
    assert (output.calls, output.input_tokens, output.output_tokens) == (2, 1500, 150)
    assert output.cost == pytest.approx(HAIKU.cost(1500, 150))
    assert llm_spent() == pytest.approx(HAIKU.cost(1500, 150))


async def test_crawl4ai_fails_a_document_only_when_every_chunk_failed(
    crawl4ai: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = {"index": 0, "error": True, "tags": ["error"], "content": "rate limited"}
    monkeypatch.setattr(_Crawl4AI, "blocks", [error])
    monkeypatch.setattr(_Crawl4AI, "usages", [])
    system = crawl4ai.Crawl4AIBaseline(BaselineSetup(BOOK_SPECS, "Extract books.", HAIKU))
    with pytest.raises(crawl4ai.ToolError, match="rate limited"):
        await system.extract(HTML_INPUT)


async def test_crawl4ai_charges_calls_made_before_the_crawl_failed(
    crawl4ai: Any, monkeypatch: pytest.MonkeyPatch, llm_spent: Callable[[], float]
) -> None:
    monkeypatch.setattr(_Crawl4AI, "usages", [_TokenUsage(1000, 100)])
    monkeypatch.setattr(_Crawl4AI, "fail", RuntimeError("connection reset"))
    system = crawl4ai.Crawl4AIBaseline(BaselineSetup(BOOK_SPECS, "Extract books.", HAIKU))
    with pytest.raises(RuntimeError, match="connection reset"):
        await system.extract(HTML_INPUT)
    assert llm_spent() == pytest.approx(HAIKU.cost(1000, 100))


class _ScrapeGraph:
    """``scrapegraphai`` and the LangChain parts the script uses: the graph calls the
    model's callbacks once per ``responses`` entry, then returns ``answer``."""

    answer: Any = None
    responses: list[dict[str, int]] = []  # noqa: RUF012
    seen: list[dict[str, Any]] = []  # noqa: RUF012

    class BaseCallbackHandler:
        pass

    class ChatAnthropic:
        def __init__(self, model: str, max_tokens: int, callbacks: list[Any]) -> None:
            self.model, self.max_tokens, self.callbacks = model, max_tokens, callbacks

    class SmartScraperGraph:
        def __init__(self, prompt: str, source: str, config: dict[str, Any], schema: Any):
            _ScrapeGraph.seen.append(
                {"prompt": prompt, "source": source, "config": config, "schema": schema}
            )
            self.model = config["llm"]["model_instance"]

        def run(self) -> Any:
            for usage in _ScrapeGraph.responses:
                message = type("Message", (), {"usage_metadata": usage})()
                generation = type("Generation", (), {"message": message})()
                response = type("Result", (), {"generations": [[generation]]})()
                for callback in self.model.callbacks:
                    callback.on_llm_end(response)
            return _ScrapeGraph.answer


@pytest.fixture
def scrapegraphai(monkeypatch: pytest.MonkeyPatch) -> Any:
    import sys
    import types

    modules = {
        "scrapegraphai": {},
        "scrapegraphai.graphs": {"SmartScraperGraph": _ScrapeGraph.SmartScraperGraph},
        "langchain_core": {},
        "langchain_core.callbacks": {"BaseCallbackHandler": _ScrapeGraph.BaseCallbackHandler},
        "langchain_anthropic": {"ChatAnthropic": _ScrapeGraph.ChatAnthropic},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(_ScrapeGraph, "seen", [])
    return _script("scrapegraphai_baseline")


async def test_scrapegraphai_counts_and_charges_each_response(
    scrapegraphai: Any, monkeypatch: pytest.MonkeyPatch, llm_spent: Callable[[], float]
) -> None:
    monkeypatch.setattr(_ScrapeGraph, "answer", {"Book": [{"title": "Dune", "rating": "NA"}]})
    monkeypatch.setattr(
        _ScrapeGraph, "responses", [{"input_tokens": 1000, "output_tokens": 100}, {}]
    )
    system = scrapegraphai.ScrapeGraphAIBaseline(BaselineSetup(BOOK_SPECS, "Extract.", HAIKU))
    output = await system.extract(HTML_INPUT)
    assert output.records == {"Book": [{"entity": "1", "values": {"title": "Dune"}}]}
    assert (output.calls, output.input_tokens, output.output_tokens) == (2, 1000, 100)
    assert output.cost == pytest.approx(HAIKU.cost(1000, 100))
    assert llm_spent() == pytest.approx(HAIKU.cost(1000, 100))
    (call,) = _ScrapeGraph.seen
    assert call["prompt"] == "Extract."
    assert call["source"] == "<html><h1>Dune</h1></html>"
    assert call["schema"] is records_model(BOOK_SPECS)
    assert call["config"]["llm"]["model_tokens"] == 200_000
    assert call["config"]["llm"]["model_instance"].model == "claude-haiku-4-5-20251001"


async def test_scrapegraphai_fails_a_document_it_returns_an_error_for(
    scrapegraphai: Any, monkeypatch: pytest.MonkeyPatch, llm_spent: Callable[[], float]
) -> None:
    monkeypatch.setattr(_ScrapeGraph, "answer", {"error": "timed out", "raw_response": ""})
    monkeypatch.setattr(_ScrapeGraph, "responses", [{"input_tokens": 10, "output_tokens": 0}])
    system = scrapegraphai.ScrapeGraphAIBaseline(BaselineSetup(BOOK_SPECS, "Extract.", HAIKU))
    with pytest.raises(scrapegraphai.ToolError, match="timed out"):
        await system.extract(HTML_INPUT)
    assert llm_spent() == pytest.approx(HAIKU.cost(10, 0))
    monkeypatch.setattr(_ScrapeGraph, "answer", ["not", "an", "object"])
    with pytest.raises(scrapegraphai.ToolError, match="expected an object, got list"):
        await system.extract(HTML_INPUT)

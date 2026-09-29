from collections.abc import Mapping

import pytest
from pydantic import BaseModel

from jevex import (
    Context,
    Document,
    DocumentGateStage,
    DocumentText,
    Extractor,
    Field,
    HtmlTextReader,
    NoulDocumentGate,
    SchemaConfig,
    SchemaSpec,
    TextReader,
)
from jevex.extractor import default_pipeline
from jevex.gate import html_text
from jevex.interfaces import DocumentGate, GateDecision
from jevex.jev import (
    ChoiceAnswer,
    JevClient,
    JevResponse,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    ScoreAnswer,
    StateTooLargeError,
)
from jevex.testing import FakeJev

CAR_Q = "Does this document describe a car's technical specification?"
BOOK_Q = "Does this document list books for sale?"


class Car(BaseModel):
    """A car's technical specification.

    Filled from manufacturer spec pages and brochures.
    """

    model: str = Field(description="Model name")


class Book(BaseModel):
    """A book."""

    __jevex__ = SchemaConfig(document_question=BOOK_Q)

    title: str = Field(description="Title")


class Brochure(BaseModel):
    """A car brochure page."""

    __jevex__ = SchemaConfig(gate_unit="page")

    model: str = Field(description="Model name")


def specs(*models: type[BaseModel]) -> list[SchemaSpec]:
    return [SchemaSpec.from_model(m) for m in models]


def html(body: str, head: str = "") -> Document:
    return Document.from_bytes(
        f"<!DOCTYPE html><html><head>{head}</head><body>{body}</body></html>".encode(),
        content_type="text/html",
    )


PDF = Document.from_bytes(b"%PDF-1.7 fake", content_type="application/pdf")


class PagesReader:
    """Reads fixed pages for any document, as a PDF reader would."""

    def __init__(self, *pages: str) -> None:
        self.pages = pages

    def read(self, document: Document) -> DocumentText | None:
        return DocumentText("\n\n".join(self.pages), pages=self.pages)


# --- Questions and batching ------------------------------------------------------------


async def test_asks_one_noul_per_schema_in_one_request() -> None:
    fake = FakeJev(strict=True).noul(CAR_Q, p=0.9).noul(BOOK_Q, p=0.1)
    doc = html("<h1>Golf</h1><p>0-62 mph in 9.1 s</p>", head="<title>Golf specs</title>")
    decisions = await NoulDocumentGate().gate(doc, specs(Car, Book), fake.client())

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call.state == "Golf specs\n\nGolf\n0-62 mph in 9.1 s"
    assert call.questions == {"Car": Noul(instructions=CAR_Q), "Book": Noul(instructions=BOOK_Q)}
    assert decisions == {
        "Car": GateDecision(p=0.9, passed=True),
        "Book": GateDecision(p=0.1, passed=False),
    }


async def test_question_comes_from_docstring_or_document_question() -> None:
    fake = FakeJev()
    await NoulDocumentGate().gate(html("<p>text</p>"), specs(Car, Book), fake.client())
    assert [q.instructions for q in fake.questions] == [CAR_Q, BOOK_Q]


@pytest.mark.parametrize(("p", "passed"), [(0.69, False), (0.7, True), (0.71, True)])
async def test_threshold_is_inclusive(p: float, passed: bool) -> None:
    fake = FakeJev().noul(None, p=p)
    decisions = await NoulDocumentGate(threshold=0.7).gate(
        html("<p>text</p>"), specs(Car), fake.client()
    )
    assert decisions["Car"].passed is passed


async def test_long_text_is_cut_to_max_chars() -> None:
    fake = FakeJev()
    await NoulDocumentGate(max_chars=10).gate(html(f"<p>{'x' * 50}</p>"), specs(Car), fake.client())
    assert fake.calls[0].state == "x" * 10


async def test_oversized_state_raises_rather_than_being_dropped() -> None:
    jev = JevClient(FakeJev(), state_token_budget=100)
    with pytest.raises(StateTooLargeError):
        await NoulDocumentGate().gate(html(f"<p>{'word ' * 400}</p>"), specs(Car), jev)


async def test_non_noul_answer_raises() -> None:
    class ChoiceBackend:
        async def system_one(
            self, state: JSONContent, questions: Mapping[str, Question]
        ) -> JevResponse:
            answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {
                k: ChoiceAnswer(choice="a", confidence=1.0, probabilities={"a": 1.0})
                for k in questions
            }
            return JevResponse(answers=answers)

    with pytest.raises(TypeError, match="expected a Noul answer for 'Car'"):
        await NoulDocumentGate().gate(html("<p>x</p>"), specs(Car), JevClient(ChoiceBackend()))


def test_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="threshold"):
        NoulDocumentGate(threshold=1.5)
    with pytest.raises(ValueError, match="max_chars"):
        NoulDocumentGate(max_chars=0)


# --- Unreadable documents --------------------------------------------------------------


async def test_pdf_without_a_reader_gets_no_decision() -> None:
    fake = FakeJev()
    assert await NoulDocumentGate().gate(PDF, specs(Car, Brochure), fake.client()) == {}
    assert fake.calls == []


async def test_document_without_text_gets_no_decision() -> None:
    fake = FakeJev()
    doc = html('<img src="car.jpg"><script type="application/ld+json">{"a": 1}</script>')
    assert await NoulDocumentGate().gate(doc, specs(Car), fake.client()) == {}
    assert fake.calls == []


# --- Per-page gating -------------------------------------------------------------------


async def test_page_unit_asks_each_page_and_keeps_page_scores() -> None:
    fake = (
        FakeJev(strict=True)
        .noul("brochure", p=0.1)
        .noul("brochure", p=0.8, state="Technical data")
        .noul("brochure", p=0.3, state="Prices")
    )
    gate = NoulDocumentGate(reader=PagesReader("Welcome", "Technical data", "  ", "Prices"))
    decisions = await gate.gate(PDF, specs(Brochure), fake.client())

    # The blank third page isn't asked, and every page's questions go in one request.
    assert [call.state for call in fake.calls] == ["Welcome", "Technical data", "Prices"]
    assert all(
        call.questions
        == {"Brochure": Noul(instructions="Does this document describe a car brochure page?")}
        for call in fake.calls
    )
    assert decisions == {
        "Brochure": GateDecision(p=0.8, passed=True, pages={1: 0.1, 2: 0.8, 4: 0.3})
    }


async def test_page_unit_fails_when_no_page_passes() -> None:
    fake = FakeJev().noul(None, p=0.2)
    gate = NoulDocumentGate(reader=PagesReader("one", "two"))
    decisions = await gate.gate(PDF, specs(Brochure), fake.client())
    assert decisions["Brochure"] == GateDecision(p=0.2, passed=False, pages={1: 0.2, 2: 0.2})


async def test_mixed_units_ask_the_whole_text_and_each_page() -> None:
    fake = FakeJev().noul(None, p=0.9)
    gate = NoulDocumentGate(reader=PagesReader("one", "two"))
    decisions = await gate.gate(PDF, specs(Car, Brochure), fake.client())

    asked = {call.state: set(call.questions) for call in fake.calls}
    assert asked == {"one\n\ntwo": {"Car"}, "one": {"Brochure"}, "two": {"Brochure"}}
    assert decisions["Car"].pages == {}
    assert decisions["Brochure"].pages == {1: 0.9, 2: 0.9}


async def test_page_unit_on_unpaged_document_gates_the_whole_document() -> None:
    fake = FakeJev().noul(None, p=0.9)
    decisions = await NoulDocumentGate().gate(html("<p>Golf</p>"), specs(Brochure), fake.client())
    assert decisions == {"Brochure": GateDecision(p=0.9, passed=True)}
    assert len(fake.calls) == 1


async def test_page_unit_with_only_blank_pages_gets_no_decision() -> None:
    fake = FakeJev()
    gate = NoulDocumentGate(reader=PagesReader("", " \n"))
    assert await gate.gate(PDF, specs(Brochure), fake.client()) == {}
    assert fake.calls == []


# --- Stage ------------------------------------------------------------------------------


def context(doc: Document, fake: FakeJev, *models: type[BaseModel]) -> Context:
    return Context.create(doc, specs(*models), fake.client())


async def test_stage_deactivates_ruled_out_schemas_and_records_decisions() -> None:
    fake = FakeJev().noul(CAR_Q, p=0.9).noul(BOOK_Q, p=0.2)
    ctx = context(html("<p>Golf</p>"), fake, Car, Book)
    await DocumentGateStage().run(ctx)

    assert [run.name for run in ctx.active] == ["Car"]
    assert ctx.schemas["Car"].gate == GateDecision(p=0.9, passed=True)
    assert ctx.schemas["Book"].gate == GateDecision(p=0.2, passed=False)
    assert [(e.stage, e.kind, e.message, e.data) for e in ctx.events] == [
        ("document_gate", "gated_out", "Book ruled out (p=0.20)", {"schema": "Book", "p": 0.2})
    ]


async def test_stage_keeps_schemas_the_gate_could_not_decide() -> None:
    fake = FakeJev()
    ctx = context(PDF, fake, Car)
    await DocumentGateStage().run(ctx)

    assert [run.name for run in ctx.active] == ["Car"]
    assert ctx.schemas["Car"].gate is None
    assert [(e.kind, e.data) for e in ctx.events] == [("gate_skipped", {"schema": "Car"})]
    assert ctx.events[0].message == (
        "no gate decision for Car (no readable application/pdf text?), so it stays active"
    )


async def test_stage_only_gates_active_schemas() -> None:
    fake = FakeJev().noul(None, p=0.9)
    ctx = context(html("<p>Golf</p>"), fake, Car, Book)
    ctx.schemas["Book"].deactivate()
    await DocumentGateStage().run(ctx)
    assert set(fake.calls[0].questions) == {"Car"}
    assert ctx.schemas["Book"].gate is None


async def test_stage_runs_a_custom_gate() -> None:
    class KeepBooks:
        async def gate(
            self, document: Document, schemas: list[SchemaSpec], jev: JevClient
        ) -> dict[str, GateDecision]:
            return {s.name: GateDecision(p=1.0, passed=s.name == "Book") for s in schemas}

    ctx = context(html("<p>x</p>"), FakeJev(), Car, Book)
    await DocumentGateStage(KeepBooks()).run(ctx)
    assert [run.name for run in ctx.active] == ["Book"]


def test_defaults_satisfy_the_protocols() -> None:
    assert isinstance(NoulDocumentGate(), DocumentGate)
    assert isinstance(HtmlTextReader(), TextReader)


# --- Extractor --------------------------------------------------------------------------


def test_default_pipeline_gates_after_cleaning() -> None:
    assert default_pipeline().names[:2] == ["clean", "document_gate"]


async def test_extractor_reports_gates_in_document_meta() -> None:
    fake = FakeJev().noul(CAR_Q, p=0.95).noul(BOOK_Q, p=0.05)
    async with Extractor([Car, Book], jev=fake.client()) as ex:
        result = await ex.extract(html("<nav>Books</nav><p>Golf</p>"))

    assert fake.calls[0].state == "Golf"  # cleaned before gating
    assert result.meta.gates == {
        "Car": GateDecision(p=0.95, passed=True),
        "Book": GateDecision(p=0.05, passed=False),
    }
    assert result.meta.active_schemas == ["Car"]
    assert "document_gate" in result.meta.timings


async def test_extractor_skips_later_stages_when_every_schema_is_gated_out() -> None:
    ran: list[str] = []

    class Later:
        name = "later"

        async def run(self, ctx: Context) -> None:
            ran.append(self.name)

    pipeline = default_pipeline().append(Later())
    async with Extractor([Car], jev=FakeJev().client(), pipeline=pipeline) as ex:
        result = await ex.extract(html("<p>A recipe for scones</p>"))
    assert ran == []
    assert result.meta.active_schemas == []
    assert result.meta.gates["Car"].passed is False
    assert result.meta.stopped


# --- HTML text --------------------------------------------------------------------------


def test_html_text_puts_title_first_and_one_block_per_line() -> None:
    markup = (
        "<html><head><title> Golf  specs </title><meta charset='utf-8'></head><body>"
        "<h1>Golf</h1><div>Engine: <b>1.5</b> litre</div><ul><li>Petrol</li><li>SE L</li></ul>"
        "<table><tr><th>Trim</th><td>SE</td></tr></table>line<br>break</body></html>"
    )
    assert html_text(markup) == (
        "Golf specs\n\nGolf\nEngine: 1.5 litre\nPetrol\nSE L\nTrim SE\nline\nbreak"
    )


def test_html_text_skips_non_rendered_content() -> None:
    markup = (
        "<p>Kept</p><script>var x = 1;</script><style>p {}</style>"
        '<script type="application/ld+json">{"name": "Golf"}</script>'
        "<noscript>Enable JS</noscript><template><p>Later</p></template>"
        "<svg><title>icon</title><text>svg text</text></svg><p>Also kept</p>"
    )
    assert html_text(markup) == "Kept\nAlso kept"


def test_html_text_decodes_entities_and_collapses_whitespace() -> None:
    assert html_text("<p>Fish &amp; chips&nbsp;&pound;9\n\t  now</p>") == "Fish & chips £9 now"


def test_html_text_reads_body_after_an_unclosed_head() -> None:
    assert html_text("<html><head><title>T</title><body><p>Body</p>") == "T\n\nBody"


def test_html_text_of_empty_page_is_empty() -> None:
    assert html_text("<html><body>  </body></html>") == ""


def test_html_reader_uses_the_page_charset() -> None:
    doc = Document.from_bytes(
        '<meta charset="windows-1252"><p>Price £9</p>'.encode("cp1252"), content_type="text/html"
    )
    assert HtmlTextReader().read(doc) == DocumentText("Price £9")
    assert HtmlTextReader().read(PDF) is None

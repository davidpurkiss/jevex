import io
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from jevex import (
    Context,
    DefaultTextReader,
    Document,
    DocumentGateStage,
    DocumentText,
    Extractor,
    Field,
    HtmlTextReader,
    NoulDocumentGate,
    PdfTextReader,
    SchemaConfig,
    SchemaSpec,
    TextReader,
    UnreadablePdfError,
    _pdfium,
)
from jevex.extractor import default_pipeline
from jevex.gate import html_text
from jevex.interfaces import DocumentGate, GateDecision
from jevex.jev import (
    ChoiceAnswer,
    JevClient,
    JevResponse,
    JevTokenLimitError,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    ScoreAnswer,
    StateTooLargeError,
    UnexpectedAnswerError,
    estimate_tokens,
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
SPEC_PDF = Path(__file__).parent / "fixtures" / "pdf" / "spec.pdf"
SPEC_PAGE_1 = (
    "Skoda Octavia Estate\n"
    "The Octavia Estate combines a large boot with efficient engines. Prices start at "
    "£27,500 on\nthe road.\n"
    "Engines\n"
    "Two petrol engines and one diesel are available.\n"
    "1.5 TSI 150PS manual\n"
    "1.5 TSI 150PS DSG\n"
    "2.0 TDI 115PS manual\n"
    "Performance\n"
    "1.5 TSI SE 2.0 TDI SE L\n"
    "0-62 mph (s) 8.5 10.4\n"
    "Top speed (mph) 139 128\n"
    "CO2 (g/km) 131 118\n"
    "Figures are for the manufacturer's test cycle."
)
SPEC_PAGE_2 = (
    "Dimensions\n"
    "Exterior\n"
    "Length is 4,698 mm and width is 1,829 mm.\n"
    "Boot\n"
    "The boot holds 640 litres with the seats up."
)


def spec_pdf() -> Document:
    pytest.importorskip("pypdfium2")
    return Document.from_path(SPEC_PDF)


def pdf_with_a_blank_page() -> Document:
    """The spec sheet with a page without a text layer (as a scan has) between its pages."""
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(SPEC_PDF.read_bytes())
    pdf.new_page(595, 842, index=1)
    out = io.BytesIO()
    pdf.save(out)
    return Document.from_bytes(out.getvalue())


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


async def test_text_is_cut_to_what_fits_in_one_request() -> None:
    fake = FakeJev()
    jev = JevClient(fake, state_token_budget=100)
    await NoulDocumentGate().gate(html(f"<p>{'| 123 ' * 400}</p>"), specs(Car), jev)
    (call,) = fake.calls
    assert isinstance(call.state, str)
    assert call.state.startswith("| 123")
    assert estimate_tokens(call.state) < 100


class RejectsLongStates(FakeJev):
    """Rejects states longer than ``max_chars``, as Jev does when its count is higher than
    the client's estimate."""

    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.rejected = 0

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        if len(str(state)) > self.max_chars:
            self.rejected += 1
            raise JevTokenLimitError("max_tokens_exceeded")
        return await super().system_one(state, questions)


async def test_text_jev_rejects_is_halved_until_it_fits() -> None:
    fake = RejectsLongStates(max_chars=1000)
    await NoulDocumentGate().gate(html(f"<p>{'word ' * 1000}</p>"), specs(Car), fake.client())
    (call,) = fake.calls
    assert fake.rejected == 3  # 4999 characters, then 2499, then 1249
    assert call.state == ("word " * 1000).strip()[:624]


async def test_a_question_too_big_for_any_state_raises() -> None:
    jev = JevClient(FakeJev(), state_token_budget=5)
    with pytest.raises(StateTooLargeError, match="a question alone"):
        await NoulDocumentGate().gate(html("<p>A car.</p>"), specs(Car), jev)


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

    with pytest.raises(UnexpectedAnswerError, match="expected a Noul answer for 'Car'"):
        await NoulDocumentGate().gate(html("<p>x</p>"), specs(Car), JevClient(ChoiceBackend()))


def test_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="threshold"):
        NoulDocumentGate(threshold=1.5)
    with pytest.raises(ValueError, match="max_chars"):
        NoulDocumentGate(max_chars=0)


# --- Unreadable documents --------------------------------------------------------------


async def test_pdf_gets_no_decision_without_the_pdf_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_pdfium, "installed", lambda: False)
    fake = FakeJev()
    assert await NoulDocumentGate().gate(PDF, specs(Car, Brochure), fake.client()) == {}
    assert fake.calls == []


async def test_other_content_types_get_no_decision() -> None:
    fake = FakeJev()
    image = Document.from_bytes(b"\x89PNG\r\n\x1a\n")
    assert await NoulDocumentGate().gate(image, specs(Car), fake.client()) == {}
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
        "Brochure": GateDecision(
            p=0.8, passed=True, pages={1: 0.1, 2: 0.8, 4: 0.3}, passed_pages=[2]
        )
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
    assert decisions["Brochure"].passed_pages == [1, 2]


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
    ctx = context(html('<img src="car.jpg">'), fake, Car)
    await DocumentGateStage().run(ctx)

    assert [run.name for run in ctx.active] == ["Car"]
    assert ctx.schemas["Car"].gate is None
    assert [(e.kind, e.data) for e in ctx.events] == [("gate_skipped", {"schema": "Car"})]
    assert ctx.events[0].message == (
        "no gate decision for Car (no readable text/html text?), so it stays active"
    )


async def test_stage_suggests_the_pdf_extra_when_a_pdf_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_pdfium, "installed", lambda: False)
    ctx = context(PDF, FakeJev(), Car)
    await DocumentGateStage().run(ctx)

    assert [run.name for run in ctx.active] == ["Car"]
    assert [e.message for e in ctx.events] == [
        "no gate decision for Car (install jevex[pdf] to gate PDFs), so it stays active"
    ]


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
    assert isinstance(NoulDocumentGate().reader, DefaultTextReader)
    for reader in (DefaultTextReader(), HtmlTextReader(), PdfTextReader()):
        assert isinstance(reader, TextReader)


def test_default_reader_reads_pdfs_only_with_the_pdf_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pypdfium2")
    assert [type(r) for r in DefaultTextReader().readers] == [HtmlTextReader, PdfTextReader]
    monkeypatch.setattr(_pdfium, "installed", lambda: False)
    assert [type(r) for r in DefaultTextReader().readers] == [HtmlTextReader]


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


# --- PDF text ---------------------------------------------------------------------------


def test_pdf_reader_reads_one_string_per_page() -> None:
    text = PdfTextReader().read(spec_pdf())
    assert text == DocumentText(f"{SPEC_PAGE_1}\n\n{SPEC_PAGE_2}", pages=(SPEC_PAGE_1, SPEC_PAGE_2))


def test_default_reader_reads_both_html_and_pdfs() -> None:
    reader = DefaultTextReader()
    assert reader.read(spec_pdf()) == PdfTextReader().read(spec_pdf())
    assert reader.read(html("<p>Golf</p>")) == DocumentText("Golf")


def test_pdf_reader_reads_a_page_without_a_text_layer_as_empty() -> None:
    text = PdfTextReader().read(pdf_with_a_blank_page())
    assert text == DocumentText(
        f"{SPEC_PAGE_1}\n\n{SPEC_PAGE_2}", pages=(SPEC_PAGE_1, "", SPEC_PAGE_2)
    )


def test_pdf_reader_skips_other_documents() -> None:
    assert PdfTextReader().read(html("<p>Golf</p>")) is None


def test_pdf_reader_raises_on_a_pdf_it_cannot_open() -> None:
    pytest.importorskip("pypdfium2")
    with pytest.raises(UnreadablePdfError, match="pdfium couldn't open the PDF"):
        PdfTextReader().read(PDF)


async def test_default_gate_asks_about_each_pdf_page() -> None:
    brochure_q = "Does this document describe a car brochure page?"
    fake = (
        FakeJev(strict=True)
        .noul(CAR_Q, p=0.9, state=f"{SPEC_PAGE_1}\n\n{SPEC_PAGE_2}")
        .noul(brochure_q, p=0.8, state=SPEC_PAGE_1)
        .noul(brochure_q, p=0.2, state=SPEC_PAGE_2)
    )
    decisions = await NoulDocumentGate().gate(
        pdf_with_a_blank_page(), specs(Car, Brochure), fake.client()
    )

    # The blank second page isn't asked, so it's missing from the page scores.
    assert len(fake.calls) == 3
    assert decisions == {
        "Car": GateDecision(p=0.9, passed=True),
        "Brochure": GateDecision(p=0.8, passed=True, pages={1: 0.8, 3: 0.2}, passed_pages=[1]),
    }


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


async def test_a_page_without_a_working_charset_reads_as_windows_1252() -> None:
    # Latin-1 bytes, no <meta charset>: not valid UTF-8, so decode_html keeps them as lone
    # surrogates; the gate must read them as windows-1252 (the WHATWG fallback).
    page = Document.from_bytes(b"<html><body><p>caf\xe9 \xa320</p></body></html>")
    assert HtmlTextReader().read(page) == DocumentText("café £20")
    fake = FakeJev().noul(None, p=0.9)
    await NoulDocumentGate().gate(page, [SchemaSpec.from_model(Car)], fake.client())
    [call] = fake.calls
    assert call.state == "café £20"
    assert isinstance(call.state, str)
    call.state.encode("utf-8")  # raises on lone surrogates

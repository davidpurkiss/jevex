"""Stage 3, document gate: ask whether a document is worth extracting each schema from.

The default :class:`NoulDocumentGate` asks one Noul per registered schema ("Does this
document describe {schema description}?", or the schema's ``document_question``) about the
document's text. Every schema's question goes into one request. A schema whose answer
falls below the threshold is deactivated, so no later stage spends Jev calls on it.

Schemas with ``gate_unit="page"`` are asked once per page instead, so a 40-page brochure
only goes on to process its spec pages. The per-page probabilities are kept on the
:class:`~jevex.interfaces.GateDecision` for later stages to narrow to.

The gate runs before layout, so it reads text through a :class:`TextReader`. The default
reads HTML with the standard library. No PDF text reader ships yet (the PDF parser is
chosen in #24), so a PDF is only gated when a reader is passed in; otherwise its schemas
stay active, with a ``gate_skipped`` event.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from jevex.clean import decode_html
from jevex.interfaces import GateDecision
from jevex.jev import NoulAnswer

if TYPE_CHECKING:
    from jevex.document import Document
    from jevex.interfaces import DocumentGate
    from jevex.jev import JevClient, Noul
    from jevex.pipeline import Context
    from jevex.schema import SchemaSpec

DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_CHARS = 60_000
"""About 15k tokens: enough to tell what a document is about, well inside Jev's 32k state
limit. Longer text is cut to its leading part."""


@dataclass(frozen=True)
class DocumentText:
    """A document's plain text before layout. ``pages`` is set only for paged documents."""

    text: str
    pages: tuple[str, ...] | None = None


@runtime_checkable
class TextReader(Protocol):
    """Reads a document's text for the gate. Returns ``None`` for content it can't read."""

    def read(self, document: Document) -> DocumentText | None: ...


class HtmlTextReader:
    """The default :class:`TextReader`: the visible text of HTML documents.

    The ``<title>`` comes first, then the body text with one line per block element.
    Scripts, styles and other non-rendered elements are skipped. Other content types
    return ``None``.
    """

    def read(self, document: Document) -> DocumentText | None:
        if not document.is_html:
            return None
        return DocumentText(html_text(decode_html(document.content)[0]))


class NoulDocumentGate:
    """The default :class:`~jevex.interfaces.DocumentGate`: one Noul per schema.

    A schema passes when the probability is at least ``threshold``. Per page, it passes
    when any page does, and its ``p`` is the highest page's. Pages with no text aren't
    asked and are left out of ``GateDecision.pages``. A schema whose document (or every
    page) has no readable text gets no decision, so it is neither passed nor ruled out.
    Text longer than ``max_chars`` (per document or page) is cut to its leading part.
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        reader: TextReader | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        if max_chars < 1:
            raise ValueError(f"max_chars must be positive, got {max_chars}")
        self.threshold = threshold
        self.reader: TextReader = reader if reader is not None else HtmlTextReader()
        self.max_chars = max_chars

    async def gate(
        self, document: Document, schemas: list[SchemaSpec], jev: JevClient
    ) -> dict[str, GateDecision]:
        text = self.reader.read(document)
        if text is None:
            return {}
        by_page = [s for s in schemas if s.config.gate_unit == "page" and text.pages is not None]
        whole = [s for s in schemas if s not in by_page]
        pages = {i: page for i, page in enumerate(text.pages or (), start=1) if page.strip()}
        whole_p, page_ps = await asyncio.gather(
            self._ask(text.text, whole, jev),
            asyncio.gather(*(self._ask(page, by_page, jev) for page in pages.values())),
        )
        decisions = {
            name: GateDecision(p=p, passed=p >= self.threshold) for name, p in whole_p.items()
        }
        for spec in by_page:
            ps = {i: answers[spec.name] for i, answers in zip(pages, page_ps, strict=True)}
            if ps:
                best = max(ps.values())
                decisions[spec.name] = GateDecision(p=best, passed=best >= self.threshold, pages=ps)
        return decisions

    async def _ask(self, state: str, schemas: list[SchemaSpec], jev: JevClient) -> dict[str, float]:
        """Every schema's gate question about one state, in one request."""
        state = state.strip()[: self.max_chars]
        if not schemas or not state:
            return {}
        questions: dict[str, Noul] = {s.name: s.document_gate_question() for s in schemas}
        answers = await jev.ask(state, questions)
        out: dict[str, float] = {}
        for name, answer in answers.items():
            if not isinstance(answer, NoulAnswer):
                raise TypeError(f"expected a Noul answer for {name!r}, got {answer.type}")
            out[name] = answer.p
        return out


@dataclass
class DocumentGateStage:
    """Runs a :class:`~jevex.interfaces.DocumentGate` and deactivates schemas it rules out.

    Each decision is stored on its :class:`~jevex.pipeline.SchemaRun` and so reported in
    ``DocumentMeta.gates``. Ruled-out schemas get a ``gated_out`` event, and schemas the
    gate gave no decision for stay active with a ``gate_skipped`` event.
    """

    gate: DocumentGate = field(default_factory=NoulDocumentGate)
    name: str = "document_gate"

    async def run(self, ctx: Context) -> None:
        runs = ctx.active
        decisions = await self.gate.gate(ctx.document, [run.spec for run in runs], ctx.jev)
        for run in runs:
            decision = decisions.get(run.name)
            if decision is None:
                ctx.event(
                    self.name,
                    "gate_skipped",
                    f"no gate decision for {run.name} (no readable {ctx.document.content_type} "
                    "text?), so it stays active",
                    schema=run.name,
                )
                continue
            run.gate = decision
            if not decision.passed:
                run.deactivate()
                ctx.event(
                    self.name,
                    "gated_out",
                    f"{run.name} ruled out (p={decision.p:.2f})",
                    schema=run.name,
                    p=decision.p,
                )


# --- HTML text ---------------------------------------------------------------------------

# Elements whose content is never rendered as text. ``head`` isn't listed: its only text is
# the title, and pages often leave it unclosed.
_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "math", "iframe", "object"}
)
# Elements that start a new line of text.
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tr",
        "ul",
    }
)
_CELL_TAGS = frozenset({"td", "th"})
_SPACES = re.compile(r"[^\S\n]+")  # any whitespace but a newline


def html_text(markup: str) -> str:
    """The visible text of an HTML page: title first, then one line per block."""
    parser = _TextParser()
    parser.feed(markup)
    parser.close()
    lines = [_SPACES.sub(" ", line).strip() for line in "".join(parser.out).split("\n")]
    body = "\n".join(line for line in lines if line)
    title = _SPACES.sub(" ", "".join(parser.title)).strip()
    return f"{title}\n\n{body}".strip() if title else body


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.title: list[str] = []
        self._skip: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title" and not self._skip:  # not an SVG <title>
            self._in_title = True
        elif tag in _SKIP_TAGS:
            self._skip.append(tag)
        elif tag in _BLOCK_TAGS:
            self.out.append("\n")
        elif tag in _CELL_TAGS:
            self.out.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in self._skip:
            # Close the matching element and anything left open inside it.
            while self._skip.pop() != tag:
                pass
        elif tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            # Newlines in the source are just whitespace; only block elements break lines.
            self.out.append(data.replace("\n", " "))

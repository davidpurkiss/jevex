"""Protocols for every pluggable part of the pipeline.

Each protocol is the narrow, typed thing an implementer writes (a cleaner, a layout
parser, a candidate generator...). The pipeline wraps implementations in
:class:`jevex.pipeline.Stage` adapters, so replacing a default means passing a
different implementation, and adding behaviour means adding a stage.

Stage order and default implementations (spec: *Pipeline architecture*):

====  =====================  ============================  =====
 #    Protocol               Default                        Issue
====  =====================  ============================  =====
 1    Fetcher                SimpleFetcher                  #22
 2    Cleaner                boilerplate stripper           #11
 3    DocumentGate           Noul per schema                #12
 4    StructuredExtractor    JSON-LD, microdata, app state  #29, #30
 5    LayoutParser           HTML segmenter; Docling (PDF)  #13, #24
 6    ImageProcessor         OCR                            #26
 7    ComponentGate          Noul per component × group     #14
 8    StatementSplitter      by component type              #15
 9    EntityResolver         Single-, Multi-, ParentChild   #21, #27, #28
 10   StatementClassifier    Choice per statement           #16
 11   CandidateGenerator     built-ins + learned            #17
 12   CandidateSelector      Choice over candidates         #18
 13   Normaliser             declarative built-ins          #19
 14   LLMExtractor           opt-in LLM + Jev verification  #33
 15   Learner                GeneratorLearner (async)       #38
====  =====================  ============================  =====
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from jevex.document import Document
from jevex.layout import Component
from jevex.statements import Candidate, Statement

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from jevex.budgets import DocumentBudget
    from jevex.categorise import ToClassify
    from jevex.entities import EntityScope
    from jevex.images import ImageData, ImageReading
    from jevex.jev import Answer, ChoiceAnswer, JevClient, Question
    from jevex.keypaths import StructuredResult
    from jevex.schema import FieldSpec, SchemaSpec
    from jevex.store import Store, VerifiedExample


class ParsedDocument(BaseModel):
    """A document after layout: its component tree and the statements split from it."""

    document: Document
    root: Component
    statements: dict[str, Statement] = Field(default_factory=dict[str, Statement])

    def restricted_to(self, component_ids: Iterable[str]) -> ParsedDocument:
        """A copy holding only ``component_ids`` (and their statements).

        The root is always kept, so the tree stays a tree; a kept component's parent
        should be kept too (the component gate passes ancestors), or it's dropped with
        the parent. Statements of components not in the tree (structured data) are kept.
        """
        keep = set(component_ids)
        in_tree = {c.id for c in self.root.walk()}

        def prune(component: Component) -> Component:
            children = [prune(c) for c in component.children if c.id in keep]
            return component.model_copy(update={"children": children})

        statements = {
            sid: s
            for sid, s in self.statements.items()
            if s.component_id in keep or s.component_id not in in_tree
        }
        return ParsedDocument(document=self.document, root=prune(self.root), statements=statements)

    def statements_in(self, component_ids: list[str]) -> list[Statement]:
        """Statements belonging to the given components, in document order."""
        wanted = set(component_ids)
        return [s for s in self.statements.values() if s.component_id in wanted]

    def scope_statements(self, scope: EntityScope) -> list[Statement]:
        """An entity's statements, in document order: those of its components, plus its
        ``statement_ids`` (a table column's cells) and ``shared_statement_ids``."""
        components = set(scope.component_ids)
        ids = {*scope.statement_ids, *scope.shared_statement_ids}
        return [s for s in self.statements.values() if s.component_id in components or s.id in ids]


class GateDecision(BaseModel):
    """A document-gate answer for one schema.

    For per-page gating, ``pages`` maps each asked page (1-based) to its probability and
    ``passed_pages`` lists the pages that passed, so later stages can skip the rest. Pages
    missing from ``pages`` weren't asked (no text layer), so they aren't ruled out.
    """

    model_config = ConfigDict(frozen=True)

    p: float
    passed: bool
    pages: dict[int, float] = Field(default_factory=dict[int, float])
    passed_pages: list[int] = Field(default_factory=list[int])


class Scope(BaseModel):
    """Where a candidate generator applies. Empty means "everywhere"."""

    model_config = ConfigDict(frozen=True)

    kinds: frozenset[str] = frozenset()
    fields: frozenset[str] = frozenset()
    schemas: frozenset[str] = frozenset()
    locale: str | None = None
    sources: frozenset[str] = frozenset()


class Selection(BaseModel):
    """The selector's pick for one statement and field. ``candidate`` is None for "none".

    For ``list[...]`` fields, ``accepted`` holds every candidate the statement states (in
    text order); ``candidate`` is the most confident of them. ``alternatives`` maps the
    other raw spans that were weighed to their probabilities.
    """

    model_config = ConfigDict(frozen=True)

    candidate: Candidate | None
    confidence: float
    alternatives: dict[str, float] = Field(default_factory=dict[str, float])
    accepted: list[Candidate] = Field(default_factory=list[Candidate])


class LLMAnswer(BaseModel):
    """A fallback answer: a value plus the verbatim evidence it came from."""

    model_config = ConfigDict(frozen=True)

    value: Any
    evidence: str


@runtime_checkable
class Fetcher(Protocol):
    async def fetch(self, url: str) -> Document: ...


@runtime_checkable
class Cleaner(Protocol):
    def clean(self, document: Document) -> Document: ...


@runtime_checkable
class DocumentGate(Protocol):
    async def gate(
        self, document: Document, schemas: list[SchemaSpec], jev: JevClient
    ) -> dict[str, GateDecision]: ...


@runtime_checkable
class StructuredExtractor(Protocol):
    """Reads embedded data: ``structured`` statements plus the field values they give."""

    async def extract(
        self,
        document: Document,
        schemas: list[SchemaSpec],
        jev: JevClient,
        *,
        store: Store | None = None,
    ) -> StructuredResult:
        """``store`` is the extractor's learned state (``None`` without one)."""
        ...


@runtime_checkable
class LayoutParser(Protocol):
    def supports(self, document: Document) -> bool: ...

    async def parse(self, document: Document) -> Component: ...


@runtime_checkable
class PagedLayoutParser(LayoutParser, Protocol):
    """A :class:`LayoutParser` that can leave pages out of a paged document (a PDF).

    The layout stage calls :meth:`parse_pages` instead of ``parse`` when the document gate
    ruled pages out for every active schema, so a long brochure only lays out the pages
    that passed.
    """

    async def parse_pages(self, document: Document, skip: frozenset[int]) -> Component:
        """Lay out ``document`` without its 1-based pages in ``skip``. Components keep the
        document's own page numbers."""
        ...


@runtime_checkable
class ImageProcessor(Protocol):
    """Reads one image: OCR, or a vision model (spec: *Image stage*).

    ``data`` holds the image's bytes, loaded by the image stage. The reading's ``text`` is
    text seen in the image (OCR lines), which the stage parses into child components of
    ``image``; its ``statements`` are statements about the image (a vision model's), which
    become ``vision`` statements as they are. Raise
    :class:`~jevex.images.UnreadableImageError` for bytes that aren't an image you can read.
    """

    async def process(self, image: Component, data: ImageData) -> ImageReading: ...


@runtime_checkable
class ComponentGate(Protocol):
    """The ids of components relevant to each field group, per schema.

    Returns ``{schema name: {group: [component ids]}}``. Taking every schema at once lets
    an implementation ask all their questions about one component in one request.
    ``schemas`` includes each nested model's spec (named ``"<Parent>.<field>"``); leaving
    one out of the result leaves that child run ungated.
    """

    async def gate(
        self, parsed: ParsedDocument, schemas: list[SchemaSpec], jev: JevClient
    ) -> dict[str, dict[str, list[str]]]: ...


@runtime_checkable
class EntityResolver(Protocol):
    async def resolve(
        self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
    ) -> list[EntityScope]: ...


@runtime_checkable
class StatementSplitter(Protocol):
    def split(self, component: Component) -> list[Statement]: ...


@runtime_checkable
class LocaleAwareSplitter(StatementSplitter, Protocol):
    """A splitter whose sentence rules depend on the document's language.

    ``StatementStage`` calls ``split_in`` when a splitter has it, ahead of ``split``.
    ``locale`` is the document's BCP 47 tag (else the stage's), ``None`` when unknown: the
    splitter then keeps its own language. :class:`~jevex.DefaultSplitter` is locale-aware:
    "z. B. am 3. Mai" is one sentence on a ``de`` page and three on an ``en`` one.
    """

    def split_in(self, component: Component, locale: str | None) -> list[Statement]: ...


@runtime_checkable
class StatementClassifier(Protocol):
    """Categorises statements as one of each schema's fields, or "none".

    Each item is a statement plus the schemas (and, after the component gate, the field
    names) to categorise it for. Returns ``{schema name: {statement id: answer}}``; an
    item may be left out (e.g. no field was allowed). Taking every schema at once lets an
    implementation ask all their questions about a statement in one request.
    """

    async def classify(
        self, items: Sequence[ToClassify], jev: JevClient
    ) -> dict[str, dict[str, ChoiceAnswer]]: ...


@runtime_checkable
class CandidateGenerator(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def scope(self) -> Scope: ...

    def generate(self, statement: Statement) -> list[Candidate]: ...


@runtime_checkable
class FieldAwareGenerator(CandidateGenerator, Protocol):
    """A generator whose normaliser chain depends on the field it's generating for.

    ``GeneratorRegistry.generate`` calls ``generate_for`` instead of ``generate`` when a
    generator has it. ``key_value`` needs this: the same span "12 March 2024" wants
    ``parse_date`` for a date field but only ``strip`` for a string field.
    """

    def generate_for(self, statement: Statement, field: FieldSpec) -> list[Candidate]: ...


@runtime_checkable
class LocaleAwareGenerator(CandidateGenerator, Protocol):
    """A generator whose patterns and chains depend on the document's locale.

    ``GeneratorRegistry.generate`` calls ``generate_in`` when a generator has it, ahead of
    ``generate_for``. ``locale`` is the document's BCP 47 tag, ``None`` when unknown
    (read it as en-GB, :func:`jevex.locales.locale_conventions`). The built-in number,
    money, range, date and key-value generators are locale-aware: "1.234,5" is one number
    on a ``de-DE`` page and none on an ``en-GB`` one.
    """

    def generate_in(
        self, statement: Statement, field: FieldSpec, locale: str | None
    ) -> list[Candidate]: ...


@runtime_checkable
class CandidateSelector(Protocol):
    """Chooses among candidate spans, in two steps so the stage can batch every question
    about a statement into one Jev request: ``questions`` says what to ask, ``selection``
    reads the answers (keyed as ``questions`` returned them)."""

    def questions(
        self, statement: Statement, field: FieldSpec, candidates: list[Candidate]
    ) -> dict[str, Question]: ...

    def selection(
        self, field: FieldSpec, candidates: list[Candidate], answers: Mapping[str, Answer]
    ) -> Selection: ...


@runtime_checkable
class Normaliser(Protocol):
    @property
    def name(self) -> str: ...

    def apply(self, value: Any, **args: Any) -> Any: ...


@runtime_checkable
class LLMExtractor(Protocol):
    """Asks an LLM for one field's value in one statement (the fallback stage's plugin).

    Returns the value and its verbatim evidence, or ``None`` when the statement doesn't
    state the field or ``budget`` refused the call. Make LLM calls through
    ``budget.call_llm`` so the document's and run's budgets apply; raise
    :class:`~jevex.llm.LLMError` for a failed call. The stage checks the evidence, validates
    the value against the field and verifies it with Jev.
    """

    async def extract(
        self, statement: Statement, field: FieldSpec, budget: DocumentBudget
    ) -> LLMAnswer | None: ...


@runtime_checkable
class Learner(Protocol):
    """Learns from verified examples (the learn stage's plugin).

    ``submit`` must return without waiting for the learning itself, so the document
    carries on; it may skip examples it doesn't want (e.g. below its learn threshold).
    """

    async def submit(self, example: VerifiedExample) -> None: ...

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
 4    StructuredExtractor    JSON-LD, microdata, app state  #29
 5    LayoutParser           HTML segmenter; Docling (PDF)  #13, #24
 6    ImageProcessor         OCR                            #26
 7    ComponentGate          Noul per component × group     #14
 8    EntityResolver         SingleEntity                   #21
 9    StatementSplitter      by component type              #15
 10   StatementClassifier    Choice per statement           #16
 11   CandidateGenerator     built-ins + learned            #17
 12   CandidateSelector      Choice over candidates         #18
 13   Normaliser             declarative built-ins          #19
 14   LLMExtractor           opt-in LLM + Jev verification  #33
 15   Learner                async synthesise/test/swap     #38
====  =====================  ============================  =====
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from jevex.document import Document
from jevex.layout import Component
from jevex.statements import Candidate, Statement

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jevex.entities import EntityScope
    from jevex.jev import Answer, ChoiceAnswer, JevClient, Question
    from jevex.schema import FieldSpec, SchemaSpec


class ParsedDocument(BaseModel):
    """A document after layout: its component tree and the statements split from it."""

    document: Document
    root: Component
    statements: dict[str, Statement] = Field(default_factory=dict[str, Statement])

    def statements_in(self, component_ids: list[str]) -> list[Statement]:
        """Statements belonging to the given components, in document order."""
        wanted = set(component_ids)
        return [s for s in self.statements.values() if s.component_id in wanted]


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
    verified_p: float | None = None


class VerifiedExample(BaseModel):
    """An answer that passed verification, queued for learning."""

    model_config = ConfigDict(frozen=True)

    schema_name: str
    field: str
    statement: Statement
    value: Any
    evidence: str
    heading_trail: list[str] = Field(default_factory=list[str])


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
    """Reads embedded data and returns it as ``structured`` statements."""

    async def extract(
        self, document: Document, schemas: list[SchemaSpec], jev: JevClient
    ) -> list[Statement]: ...


@runtime_checkable
class LayoutParser(Protocol):
    def supports(self, document: Document) -> bool: ...

    async def parse(self, document: Document) -> Component: ...


@runtime_checkable
class ImageProcessor(Protocol):
    """Turns an image component into child components (OCR lines, vision statements)."""

    async def process(self, image: Component, document: Document) -> list[Component]: ...


@runtime_checkable
class ComponentGate(Protocol):
    """Returns the ids of components relevant to each field group."""

    async def gate(
        self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
    ) -> dict[str, list[str]]: ...


@runtime_checkable
class EntityResolver(Protocol):
    async def resolve(
        self, parsed: ParsedDocument, schema: SchemaSpec, jev: JevClient
    ) -> list[EntityScope]: ...


@runtime_checkable
class StatementSplitter(Protocol):
    def split(self, component: Component) -> list[Statement]: ...


@runtime_checkable
class StatementClassifier(Protocol):
    """Categorises each statement as one of the schema's fields, or "none"."""

    async def classify(
        self, statements: list[Statement], schema: SchemaSpec, jev: JevClient
    ) -> dict[str, ChoiceAnswer]: ...


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
    async def extract(self, statement: Statement, field: FieldSpec) -> LLMAnswer | None: ...


@runtime_checkable
class Learner(Protocol):
    async def submit(self, example: VerifiedExample) -> None: ...

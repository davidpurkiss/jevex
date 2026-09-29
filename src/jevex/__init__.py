"""jevex: extract typed records from web pages and PDFs using Jev."""

from jevex.categorise import CategoriseStage, JevStatementClassifier, ToClassify
from jevex.clean import BoilerplateCleaner, CleanStage
from jevex.component_gate import ComponentGateStage, GateUnit, NoulComponentGate, gate_units
from jevex.document import Document
from jevex.entities import EntityScope
from jevex.eval import EvalReport, Tolerance, evaluate, load_corpus
from jevex.extractor import ExtractionResult, Extractor
from jevex.fetch import FetchError, RobotsDisallowedError, SimpleFetcher
from jevex.gate import (
    DocumentGateStage,
    DocumentText,
    HtmlTextReader,
    NoulDocumentGate,
    TextReader,
)
from jevex.generators import (
    GeneratorRegistry,
    GeneratorSpec,
    InvalidGeneratorError,
    MatchSpec,
    Provenance,
    RegexGenerator,
    SpecScope,
    default_registry,
    generator_spec_json_schema,
)
from jevex.layout import (
    BBox,
    Component,
    ComponentType,
    DomLocation,
    ImageLocation,
    LayoutStage,
    Location,
    PageLocation,
    TableCell,
    UnsupportedDocumentError,
)
from jevex.layout_html import HtmlLayoutParser
from jevex.normalise import (
    BUILTIN_NORMALISERS,
    FunctionNormaliser,
    NormaliseError,
    NormaliserRegistry,
    NormaliseStage,
    normalise,
    run_chain,
)
from jevex.pipeline import Context, Pipeline, Stage
from jevex.resolve import EntityStage, SingleEntity
from jevex.results import Extracted, FieldMeta, Source
from jevex.schema import Field, FieldSpec, Questions, SchemaConfig, SchemaSpec
from jevex.select import CandidateStage, JevCandidateSelector, SelectStage
from jevex.split import DefaultSplitter, DuplicateStatementError, StatementStage
from jevex.statements import Candidate, NormaliserStep, Span, Statement, StatementKind
from jevex.store import (
    GeneratorRecord,
    GeneratorStats,
    KeyMapping,
    SpendEntry,
    SQLiteStore,
    Store,
    StoreError,
    VerifiedExample,
    open_store,
)
from jevex.structured import (
    EmbeddedData,
    EmbeddedDataReader,
    SkippedBlob,
    StructuredBlob,
    StructuredSource,
)

__version__ = "0.0.1"

__all__ = [
    "BUILTIN_NORMALISERS",
    "BBox",
    "BoilerplateCleaner",
    "Candidate",
    "CandidateStage",
    "CategoriseStage",
    "CleanStage",
    "Component",
    "ComponentGateStage",
    "ComponentType",
    "Context",
    "DefaultSplitter",
    "Document",
    "DocumentGateStage",
    "DocumentText",
    "DomLocation",
    "DuplicateStatementError",
    "EmbeddedData",
    "EmbeddedDataReader",
    "EntityScope",
    "EntityStage",
    "EvalReport",
    "Extracted",
    "ExtractionResult",
    "Extractor",
    "FetchError",
    "Field",
    "FieldMeta",
    "FieldSpec",
    "FunctionNormaliser",
    "GateUnit",
    "GeneratorRecord",
    "GeneratorRegistry",
    "GeneratorSpec",
    "GeneratorStats",
    "HtmlLayoutParser",
    "HtmlTextReader",
    "ImageLocation",
    "InvalidGeneratorError",
    "JevCandidateSelector",
    "JevStatementClassifier",
    "KeyMapping",
    "LayoutStage",
    "Location",
    "MatchSpec",
    "NormaliseError",
    "NormaliseStage",
    "NormaliserRegistry",
    "NormaliserStep",
    "NoulComponentGate",
    "NoulDocumentGate",
    "PageLocation",
    "Pipeline",
    "Provenance",
    "Questions",
    "RegexGenerator",
    "RobotsDisallowedError",
    "SQLiteStore",
    "SchemaConfig",
    "SchemaSpec",
    "SelectStage",
    "SimpleFetcher",
    "SingleEntity",
    "SkippedBlob",
    "Source",
    "Span",
    "SpecScope",
    "SpendEntry",
    "Stage",
    "Statement",
    "StatementKind",
    "StatementStage",
    "Store",
    "StoreError",
    "StructuredBlob",
    "StructuredSource",
    "TableCell",
    "TextReader",
    "ToClassify",
    "Tolerance",
    "UnsupportedDocumentError",
    "VerifiedExample",
    "__version__",
    "default_registry",
    "evaluate",
    "gate_units",
    "generator_spec_json_schema",
    "load_corpus",
    "normalise",
    "open_store",
    "run_chain",
]

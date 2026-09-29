"""jevex: extract typed records from web pages and PDFs using Jev."""

from jevex.clean import BoilerplateCleaner, CleanStage
from jevex.document import Document
from jevex.entities import EntityScope
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
    InvalidGeneratorError,
    RegexGenerator,
    default_registry,
)
from jevex.layout import (
    BBox,
    Component,
    ComponentType,
    DomLocation,
    ImageLocation,
    Location,
    PageLocation,
)
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
    "CleanStage",
    "Component",
    "ComponentType",
    "Context",
    "Document",
    "DocumentGateStage",
    "DocumentText",
    "DomLocation",
    "EmbeddedData",
    "EmbeddedDataReader",
    "EntityScope",
    "EntityStage",
    "Extracted",
    "ExtractionResult",
    "Extractor",
    "FetchError",
    "Field",
    "FieldMeta",
    "FieldSpec",
    "FunctionNormaliser",
    "GeneratorRecord",
    "GeneratorRegistry",
    "GeneratorStats",
    "HtmlTextReader",
    "ImageLocation",
    "InvalidGeneratorError",
    "JevCandidateSelector",
    "KeyMapping",
    "Location",
    "NormaliseError",
    "NormaliseStage",
    "NormaliserRegistry",
    "NormaliserStep",
    "NoulDocumentGate",
    "PageLocation",
    "Pipeline",
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
    "SpendEntry",
    "Stage",
    "Statement",
    "StatementKind",
    "Store",
    "StoreError",
    "StructuredBlob",
    "StructuredSource",
    "TextReader",
    "VerifiedExample",
    "__version__",
    "default_registry",
    "normalise",
    "open_store",
    "run_chain",
]

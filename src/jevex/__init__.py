"""jevex: extract typed records from web pages and PDFs using Jev."""

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
from jevex.statements import Candidate, NormaliserStep, Span, Statement, StatementKind
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
    "ComponentGateStage",
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
    "GeneratorRegistry",
    "HtmlLayoutParser",
    "HtmlTextReader",
    "ImageLocation",
    "InvalidGeneratorError",
    "JevCandidateSelector",
    "LayoutStage",
    "Location",
    "NormaliseError",
    "NormaliseStage",
    "NormaliserRegistry",
    "NormaliserStep",
    "NoulComponentGate",
    "NoulDocumentGate",
    "PageLocation",
    "Pipeline",
    "Questions",
    "RegexGenerator",
    "RobotsDisallowedError",
    "SchemaConfig",
    "SchemaSpec",
    "SelectStage",
    "SimpleFetcher",
    "SingleEntity",
    "SkippedBlob",
    "Source",
    "Span",
    "Stage",
    "Statement",
    "StatementKind",
    "StructuredBlob",
    "StructuredSource",
    "TableCell",
    "TextReader",
    "Tolerance",
    "UnsupportedDocumentError",
    "__version__",
    "default_registry",
    "evaluate",
    "gate_units",
    "load_corpus",
    "normalise",
    "run_chain",
]

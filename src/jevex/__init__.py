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
    "Source",
    "Span",
    "Stage",
    "Statement",
    "StatementKind",
    "TableCell",
    "TextReader",
    "UnsupportedDocumentError",
    "__version__",
    "default_registry",
    "normalise",
    "run_chain",
]

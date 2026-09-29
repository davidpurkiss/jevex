"""jevex: extract typed records from web pages and PDFs using Jev."""

from jevex.document import Document
from jevex.entities import EntityScope
from jevex.extractor import ExtractionResult, Extractor
from jevex.layout import (
    BBox,
    Component,
    ComponentType,
    DomLocation,
    ImageLocation,
    Location,
    PageLocation,
)
from jevex.pipeline import Context, Pipeline, Stage
from jevex.schema import Field, FieldSpec, Questions, SchemaConfig, SchemaSpec
from jevex.statements import Candidate, NormaliserStep, Span, Statement, StatementKind

__version__ = "0.0.1"

__all__ = [
    "BBox",
    "Candidate",
    "Component",
    "ComponentType",
    "Context",
    "Document",
    "DomLocation",
    "EntityScope",
    "ExtractionResult",
    "Extractor",
    "Field",
    "FieldSpec",
    "ImageLocation",
    "Location",
    "NormaliserStep",
    "PageLocation",
    "Pipeline",
    "Questions",
    "SchemaConfig",
    "SchemaSpec",
    "Span",
    "Stage",
    "Statement",
    "StatementKind",
    "__version__",
]

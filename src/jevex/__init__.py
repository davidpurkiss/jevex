"""jevex: extract typed records from web pages and PDFs using Jev."""

from jevex.document import Document
from jevex.entities import EntityScope
from jevex.layout import (
    BBox,
    Component,
    ComponentType,
    DomLocation,
    ImageLocation,
    Location,
    PageLocation,
)
from jevex.statements import Candidate, NormaliserStep, Span, Statement, StatementKind

__version__ = "0.0.1"

__all__ = [
    "BBox",
    "Candidate",
    "Component",
    "ComponentType",
    "Document",
    "DomLocation",
    "EntityScope",
    "ImageLocation",
    "Location",
    "NormaliserStep",
    "PageLocation",
    "Span",
    "Statement",
    "StatementKind",
    "__version__",
]

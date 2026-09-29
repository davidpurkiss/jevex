"""The generic component tree every layout parser produces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterator

ComponentType = Literal[
    "section",
    "column",
    "heading",
    "paragraph",
    "list",
    "list_item",
    "table",
    "image",
    "breakout",
    "caption",
]


class BBox(BaseModel):
    """A rectangle in page coordinates: (x0, y0) top-left, (x1, y1) bottom-right."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox corners must be ordered (x0 <= x1, y0 <= y1)")
        return self


class DomLocation(BaseModel):
    """Where a component sits in an HTML document."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["dom"] = "dom"
    dom_path: str


class PageLocation(BaseModel):
    """Where a component sits on a PDF page (1-based page number)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["page"] = "page"
    page: int = Field(ge=1)
    bbox: BBox | None = None


class ImageLocation(BaseModel):
    """Text found inside an image: the image's URL or page, plus the OCR line's bbox."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["image"] = "image"
    src: str | None = None
    page: int | None = Field(default=None, ge=1)
    bbox: BBox | None = None


Location = Annotated[DomLocation | PageLocation | ImageLocation, Field(discriminator="kind")]


class Component(BaseModel):
    """A node in the layout tree. Stages after layout only ever see components."""

    id: str
    type: ComponentType
    text: str = ""
    children: list[Component] = Field(default_factory=list["Component"])
    heading_trail: list[str] = Field(default_factory=list[str])
    location: Location

    def walk(self) -> Iterator[Component]:
        """Yield this component and all descendants, depth first, in reading order."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, component_id: str) -> Component | None:
        return next((c for c in self.walk() if c.id == component_id), None)

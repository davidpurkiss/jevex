"""Schemas the CLI tests load by file path."""

from pydantic import BaseModel

from jevex import Field


class Book(BaseModel):
    """A book for sale."""

    title: str = Field(description="Book title")


NotAModel = 42


class Outer(BaseModel):
    """Holds a nested model for dotted-attribute loading."""

    class Inner(BaseModel):
        name: str = Field(description="Name")

    inner: Inner | None = None

"""Entity scopes: which part of a document belongs to which record."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EntityScope(BaseModel):
    """A label plus the components and statements that belong to one record.

    Downstream stages run once per scope. ``shared_statement_ids`` hold statements
    assigned to "all of them", which apply to every entity. ``parent`` names the
    parent scope's label when a ``ParentChild`` resolver built this scope.
    """

    label: str
    component_ids: list[str] = Field(default_factory=list[str])
    statement_ids: list[str] = Field(default_factory=list[str])
    shared_statement_ids: list[str] = Field(default_factory=list[str])
    parent: str | None = None

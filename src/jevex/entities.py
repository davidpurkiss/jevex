"""Entity scopes: which part of a document belongs to which record."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EntityScope(BaseModel):
    """A label plus the components and statements that belong to one record.

    Downstream stages run once per scope, over its statements: those of
    ``component_ids``, plus ``statement_ids`` (a table column's cells, a sentence assigned
    to this entity) and ``shared_statement_ids``, statements assigned to "all of them",
    which apply to every entity (see
    :meth:`ParsedDocument.scope_statements <jevex.interfaces.ParsedDocument.scope_statements>`).
    ``parent`` names the parent scope's label when a ``ParentChild`` resolver built this
    scope.
    """

    label: str
    component_ids: list[str] = Field(default_factory=list[str])
    statement_ids: list[str] = Field(default_factory=list[str])
    shared_statement_ids: list[str] = Field(default_factory=list[str])
    parent: str | None = None

"""Store records and the ``Store`` protocol; see :mod:`jevex.store`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class StoreError(Exception):
    """A store couldn't be opened, or holds data this jevex can't read."""


class GeneratorRecord(BaseModel):
    """A learned generator: an opaque spec plus the fields the store filters on.

    ``field`` is ``"Schema.field"``. ``enabled`` is cleared by pruning (#40) or by a
    layer disabling it (#41); disabled generators are kept, not deleted.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    field: str
    spec: dict[str, Any]
    scope: dict[str, str] = Field(default_factory=dict[str, str])
    enabled: bool = True
    created_at: datetime = Field(default_factory=utcnow)


class KeyMapping(BaseModel):
    """A structured-data mapping: in documents with ``fingerprint``, ``path`` is ``field``.

    ``normalisers`` is the chain to apply, in :class:`~jevex.statements.NormaliserStep`'s
    compact form. One mapping per (fingerprint, path); putting another replaces it.
    """

    model_config = ConfigDict(frozen=True)

    fingerprint: str
    path: str
    field: str
    normalisers: list[Any] = Field(default_factory=list[Any])
    created_at: datetime = Field(default_factory=utcnow)


class VerifiedExample(BaseModel):
    """A value that passed Jev verification (or came from a human reviewer).

    ``evidence`` is the ``(start, end)`` character span of the value in ``statement``.
    ``context`` holds whatever the learner needs to replay it (heading trail, component
    type, locale...).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    field: str
    statement: str
    value: Any
    evidence: tuple[int, int] | None = None
    context: dict[str, Any] = Field(default_factory=dict[str, Any])
    source: Literal["llm", "human"] = "llm"
    probability: float | None = None
    created_at: datetime = Field(default_factory=utcnow)


class GeneratorStats(BaseModel):
    """Counts for one generator (spec: *Housekeeping*).

    ``documents``: scoped documents it ran on. ``hits``: it produced a candidate.
    ``wins``: its candidate was chosen and correct.
    """

    model_config = ConfigDict(frozen=True)

    generator_id: str
    documents: int = 0
    hits: int = 0
    wins: int = 0

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.documents if self.documents else None

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.hits if self.hits else None


class SpendEntry(BaseModel):
    """One charge in the spend ledger. ``kind`` is ``"jev"`` or ``"llm"``."""

    model_config = ConfigDict(frozen=True)

    amount_usd: float
    kind: Literal["jev", "llm"]
    run_id: str | None = None
    note: str | None = None
    at: datetime = Field(default_factory=utcnow)


@runtime_checkable
class Store(Protocol):
    """Everything jevex learns, shared by every process using the same backend.

    All methods are async (a store does I/O). Writes are atomic: a concurrent reader
    sees a write entirely or not at all.
    """

    # Generators
    async def put_generator(self, generator: GeneratorRecord) -> None:
        """Insert or replace by ``id``."""
        ...

    async def get_generator(self, generator_id: str) -> GeneratorRecord | None: ...

    async def generators(
        self, field: str | None = None, *, include_disabled: bool = False
    ) -> list[GeneratorRecord]:
        """Generators, oldest first, optionally for one ``"Schema.field"``."""
        ...

    async def set_generator_enabled(self, generator_id: str, enabled: bool) -> None:
        """Enable or disable a generator. Raises ``KeyError`` if there's no such id."""
        ...

    # Key mappings
    async def put_key_mapping(self, mapping: KeyMapping) -> None: ...

    async def key_mappings(self, fingerprint: str) -> list[KeyMapping]: ...

    # Verified examples
    async def add_example(self, example: VerifiedExample) -> None:
        """Insert or replace by ``id``."""
        ...

    async def examples(self, field: str, *, limit: int | None = None) -> list[VerifiedExample]:
        """Examples for one field, newest first."""
        ...

    # Generator stats
    async def record_generator_stats(
        self, generator_id: str, *, documents: int = 0, hits: int = 0, wins: int = 0
    ) -> None:
        """Add to a generator's counts (atomically, so concurrent workers don't race)."""
        ...

    async def generator_stats(self, generator_id: str) -> GeneratorStats:
        """Counts so far; all zero for a generator never recorded."""
        ...

    # Spend ledger
    async def record_spend(self, entry: SpendEntry) -> None: ...

    async def spend(
        self,
        *,
        since: datetime | None = None,
        kind: Literal["jev", "llm"] | None = None,
        run_id: str | None = None,
    ) -> float:
        """Total USD in the ledger matching every filter given."""
        ...

    async def try_spend(
        self, entry: SpendEntry, *, cap_usd: float, since: datetime | None = None
    ) -> bool:
        """Record ``entry`` only if it keeps spend since ``since`` within ``cap_usd``.

        The check and the write are one transaction, so workers sharing a cap can't
        overshoot it together. Returns whether the entry was recorded. The cap covers
        every kind and run; filter with :meth:`spend` for reporting.
        """
        ...

    async def aclose(self) -> None: ...

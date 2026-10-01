"""Store records and the ``Store`` protocol; see :mod:`jevex.store`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class StoreError(Exception):
    """A store couldn't be opened, or holds data this jevex can't read."""


class GeneratorRecord(BaseModel):
    """A learned generator: an opaque spec plus the fields the store filters on.

    ``field`` is ``"Schema.field"``. ``enabled`` reflects the store's disable list (see
    :meth:`Store.set_generator_enabled`): putting a record with ``enabled=False`` adds its
    id to that list, and ``enabled=True`` removes it. Disabled generators are kept, not
    deleted.
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

    ``schema`` is the schema the mapping was learned for, so extractors with different
    schemas can share a store. ``field`` is ``None`` when Jev answered "none" for the
    path: recording that keeps a later hit a pure lookup, with no question asked again
    (spec: *Structured-data stage*). ``normalisers`` is the chain to apply, in
    :class:`~jevex.statements.NormaliserStep`'s compact form. One mapping per
    (fingerprint, schema, path); putting another replaces it.

    ``unsure`` marks a "none" stored because Jev stayed unsure about the path (see
    :meth:`Store.count_unsure_key_paths`) rather than because it answered "none". The
    mapper treats it as "none"; a review or re-learn may re-ask it and put a confident
    mapping in its place.
    """

    # ``schema`` would shadow a BaseModel attribute, so the field is ``schema_name``;
    # construct it with ``schema=`` (or ``schema_name=``).
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    fingerprint: str
    schema_name: str = Field(alias="schema")
    path: str
    field: str | None
    normalisers: list[Any] = Field(default_factory=list[Any])
    unsure: bool = False
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _unsure_is_none(self) -> KeyMapping:
        if self.unsure and self.field is not None:
            raise ValueError("an unsure key mapping maps to no field (field=None)")
        return self


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


# One charge above this is a units bug; keeping it low means the ledger's int64
# nano-dollar sum can't overflow short of ~9,000 such charges.
MAX_ENTRY_USD = 1_000_000.0


SpendKind = Literal["jev", "llm", "llm_call"]
"""``jev`` and ``llm`` entries are spend; ``llm_call`` entries (amount 0) mark LLM calls for
rate limits, so a call and its cost are separate entries."""


class SpendEntry(BaseModel):
    """One charge in the spend ledger (see :data:`SpendKind`).

    Amounts are non-negative and finite: the ledger records charges, not refunds, so a
    shared cap can only fill up.
    """

    model_config = ConfigDict(frozen=True)

    amount_usd: float = Field(ge=0, le=MAX_ENTRY_USD, allow_inf_nan=False)
    kind: SpendKind
    run_id: str | None = None
    note: str | None = None
    at: datetime = Field(default_factory=utcnow)


@runtime_checkable
class Store(Protocol):
    """Everything jevex learns, shared by every process using the same backend.

    All methods are async (a store does I/O). Writes are atomic: a concurrent reader
    sees a write entirely or not at all. Cancelling a task that awaits a write doesn't
    undo it: the write may still commit (for :meth:`try_spend` that means a charge that
    is counted, never one that overshoots the cap).
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
        """Enable or disable a generator by id.

        Works for any id, stored here or not: the store's layer can disable a generator
        from a lower layer (a project or community pack) without copying its spec
        (spec: *Layering*, #41). Pruning (#40) uses the same list.
        """
        ...

    async def disabled_generator_ids(self) -> set[str]:
        """Every id this store disables, whether or not its spec is stored here."""
        ...

    # Key mappings
    async def put_key_mapping(self, mapping: KeyMapping) -> None:
        """Insert or replace by (fingerprint, schema, path), clearing that path's unsure
        count."""
        ...

    async def key_mappings(
        self, fingerprint: str, *, schema: str | None = None
    ) -> list[KeyMapping]:
        """Mappings for one fingerprint, by path, optionally for one schema."""
        ...

    async def count_unsure_key_paths(
        self, fingerprint: str, schema: str, paths: list[str]
    ) -> dict[str, int]:
        """Count one more unsure Jev answer for each of ``paths``; return each new total.

        Atomic across processes, so pages from one template mapped at the same time each
        add theirs. The mapper stores a path as an ``unsure`` "none" once its total
        reaches its limit, so a template never pays to re-ask a path forever.
        """
        ...

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
        kind: SpendKind | None = None,
        run_id: str | None = None,
    ) -> float:
        """Total USD in the ledger matching every filter given."""
        ...

    async def try_spend(
        self,
        entry: SpendEntry,
        *,
        cap_usd: float | None = None,
        max_count: int | None = None,
        since: datetime | None = None,
        kind: SpendKind | None = None,
    ) -> bool:
        """Record ``entry`` only if it keeps the ledger within the given limits.

        Both limits count the entries at or after ``since`` whose kind is ``kind`` (every
        kind when ``None``), plus ``entry`` itself: ``cap_usd`` caps their total and
        ``max_count`` their number. So #34 can give Jev and the LLM their own caps
        (``kind=``), a spend cap per period (``since=``) and a rate limit
        (``max_count`` over the last minute). The check and the write are one
        transaction, so workers sharing a limit can't overshoot it together. Returns
        whether the entry was recorded. With ``kind`` set, ``entry`` must be of that
        kind (``ValueError`` otherwise).
        """
        ...

    async def aclose(self) -> None: ...

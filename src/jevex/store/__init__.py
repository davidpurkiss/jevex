"""Learned state: the ``Store`` protocol and its records (spec: *Learned state*).

A store holds everything jevex learns, so that it survives the process and is shared by
every worker on the same database:

=====================  ==========================================  =================
Record                 Purpose                                      Used by
=====================  ==========================================  =================
``GeneratorRecord``    learned candidate generators                 #37, #38, #41
``KeyMapping``         structured-data key path → field (or none)   #30
unsure counts          unsure answers per key path, until "none"    #137
``VerifiedExample``    regression tests and eval corpus             #38, #42
``GeneratorStats``     hit and win counts, for pruning              #40
spend ledger           run-level budgets shared across workers      #34
``DocumentStat``       each document's cost, methods and events     #51 (stats UI)
=====================  ==========================================  =================

Generator specs are stored as JSON objects. Their format and validation are #37, and the
store doesn't interpret them, so a spec written by a newer jevex round-trips unchanged.

``open_store("sqlite:///jevex.db")`` returns the default :class:`SQLiteStore`;
``open_store("postgresql://host/db")`` returns a
:class:`~jevex.store.postgres.PostgresStore` (the ``postgres`` extra), for several hosts.
Other backends implement :class:`Store`.
"""

from __future__ import annotations

from pathlib import Path

from jevex.store.base import (
    MAX_STAT_VALUE_CHARS,
    DocumentEvent,
    DocumentStat,
    GeneratorRecord,
    GeneratorStats,
    KeyMapping,
    SpendEntry,
    SpendKind,
    Store,
    StoreError,
    ValueStat,
    VerifiedExample,
    example_id,
)
from jevex.store.sqlite import SQLiteStore


def open_store(url: str | Path) -> Store:
    """Open a store from a URL: ``sqlite:///relative.db``, ``sqlite:////abs/path.db``,
    ``sqlite://:memory:`` (or ``sqlite:///:memory:``), or a bare path (SQLite); or
    ``postgresql://`` (or ``postgres://``) for a :class:`~jevex.store.postgres.PostgresStore`
    with its tables in the ``jevex`` schema (construct one directly to choose another).
    """
    if isinstance(url, Path):
        return SQLiteStore(url)
    if url.startswith(("postgres://", "postgresql://")):
        try:
            from jevex.store.postgres import PostgresStore
        except ImportError as exc:
            raise StoreError(
                f"the Postgres store needs the postgres extra: pip install 'jevex[postgres]' "
                f"({exc})"
            ) from exc
        return PostgresStore(url)
    if url in ("sqlite://", "sqlite://:memory:", "sqlite:///:memory:", ":memory:"):
        return SQLiteStore(":memory:")
    if url.startswith("sqlite:///"):
        return SQLiteStore(Path(url.removeprefix("sqlite:///")))
    if "://" in url:
        raise StoreError(
            f"unsupported store URL: {url!r} (use sqlite:///path.db or postgresql://...)"
        )
    return SQLiteStore(Path(url))


__all__ = [
    "MAX_STAT_VALUE_CHARS",
    "DocumentEvent",
    "DocumentStat",
    "GeneratorRecord",
    "GeneratorStats",
    "KeyMapping",
    "SQLiteStore",
    "SpendEntry",
    "SpendKind",
    "Store",
    "StoreError",
    "ValueStat",
    "VerifiedExample",
    "example_id",
    "open_store",
]

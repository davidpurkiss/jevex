"""Learned state: the ``Store`` protocol and its records (spec: *Learned state*).

A store holds everything jevex learns, so that it survives the process and is shared by
every worker on the same database:

=====================  ==========================================  =================
Record                 Purpose                                      Used by
=====================  ==========================================  =================
``GeneratorRecord``    learned candidate generators                 #37, #38, #41
``KeyMapping``         structured-data key path → field             #30
``VerifiedExample``    regression tests and eval corpus             #38, #42
``GeneratorStats``     hit and win counts, for pruning              #40
spend ledger           run-level budgets shared across workers      #34
=====================  ==========================================  =================

Generator specs are stored as JSON objects. Their format and validation are #37, and the
store doesn't interpret them, so a spec written by a newer jevex round-trips unchanged.

``open_store("sqlite:///jevex.db")`` returns the default :class:`SQLiteStore`. Other
backends implement :class:`Store` (Postgres is #36).
"""

from __future__ import annotations

from pathlib import Path

from jevex.store.base import (
    GeneratorRecord,
    GeneratorStats,
    KeyMapping,
    SpendEntry,
    Store,
    StoreError,
    VerifiedExample,
)
from jevex.store.sqlite import SQLiteStore


def open_store(url: str | Path) -> Store:
    """Open a store from a URL: ``sqlite:///relative.db``, ``sqlite:////abs/path.db``,
    ``sqlite://:memory:``, or a bare path (SQLite).

    ``postgres://`` URLs are #36.
    """
    if isinstance(url, Path):
        return SQLiteStore(url)
    if url.startswith(("postgres://", "postgresql://")):
        raise StoreError("the Postgres store isn't available yet (#36)")
    if url in ("sqlite://", "sqlite://:memory:", ":memory:"):
        return SQLiteStore(":memory:")
    if url.startswith("sqlite:///"):
        return SQLiteStore(Path(url.removeprefix("sqlite:///")))
    if "://" in url:
        raise StoreError(f"unsupported store URL: {url!r} (use sqlite:///path.db)")
    return SQLiteStore(Path(url))


__all__ = [
    "GeneratorRecord",
    "GeneratorStats",
    "KeyMapping",
    "SQLiteStore",
    "SpendEntry",
    "Store",
    "StoreError",
    "VerifiedExample",
    "open_store",
]

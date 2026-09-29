"""Structured concurrency for the pipeline's fan-outs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable


async def gather[T](aws: Iterable[Awaitable[T]]) -> list[T]:
    """Like ``asyncio.gather``, but the first failure cancels the rest before it propagates.

    With plain ``gather``, a stage whose branch raises (a Jev request cap, a backend error)
    returns while its sibling branches keep running: their requests finish after the
    document's result was built, and they write into a context nobody reads. Here every
    branch has settled by the time the call returns or raises. The original exception is
    raised, not an ``ExceptionGroup``; if several branches fail together, the first one
    in branch order wins.
    """
    tasks: list[asyncio.Task[T]] = []
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_await(aw)) for aw in aws]
    except BaseExceptionGroup as grouped:
        for task in tasks:
            error = task.exception() if task.done() and not task.cancelled() else None
            if error is not None:
                raise error from None
        raise grouped.exceptions[0] from None
    return [task.result() for task in tasks]


async def _await[T](aw: Awaitable[T]) -> T:
    return await aw

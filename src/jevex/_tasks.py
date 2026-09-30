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
    raised, not an ``ExceptionGroup``; if several branches fail, the first to fail wins.
    """
    tasks: list[asyncio.Task[T]] = []
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_await(aw)) for aw in aws]
    except BaseExceptionGroup as grouped:
        # TaskGroup collects errors as tasks finish, so the first is the first failure in
        # time: the root cause, not an error a sibling raised while being cancelled.
        raise grouped.exceptions[0] from None
    return [task.result() for task in tasks]


async def _await[T](aw: Awaitable[T]) -> T:
    return await aw

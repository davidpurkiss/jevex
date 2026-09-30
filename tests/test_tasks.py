import asyncio

import pytest

from jevex._tasks import gather


async def test_results_keep_the_order_of_the_awaitables() -> None:
    async def after(delay: float, value: int) -> int:
        await asyncio.sleep(delay)
        return value

    assert await gather([after(0.02, 1), after(0.0, 2), after(0.01, 3)]) == [1, 2, 3]
    assert await gather([]) == []


async def test_the_first_failure_cancels_the_rest_and_keeps_its_type() -> None:
    finished: list[str] = []

    async def fail() -> None:
        await asyncio.sleep(0.01)
        raise KeyError("boom")

    async def slow() -> None:
        await asyncio.sleep(0.2)
        finished.append("slow")

    with pytest.raises(KeyError, match="boom"):
        await gather([fail(), slow()])
    await asyncio.sleep(0.3)
    assert finished == []


async def test_the_root_cause_wins_over_an_error_raised_while_cancelling() -> None:
    async def bad_cleanup() -> None:
        try:
            await asyncio.sleep(1)
        finally:
            raise RuntimeError("cleanup error")

    async def fail() -> None:
        await asyncio.sleep(0.01)
        raise ValueError("original")

    with pytest.raises(ValueError, match="original"):
        await gather([bad_cleanup(), fail()])

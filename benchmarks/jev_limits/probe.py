"""Measure Jev's practical limits against the live API (#6).

Live and billed: run only with a key, under a spend cap::

    (set -a; . .env; set +a; JEVEX_JEV_MAX_COST_USD=0.50 \\
        uv run python benchmarks/jev_limits/probe.py)

Writes ``benchmarks/jev_limits/results-<date>.json``. The script keeps its own tally from
the API's reported input tokens and stops before ``SELF_CAP_USD`` (the lower of $0.30 and
``JEVEX_JEV_MAX_COST_USD``, per pass). ``--edges`` re-runs only the limit search and
``--pipeline`` only the end-to-end pricing, adding to the same day's file. Requests go straight
through ``typesafe_sdk`` (retries off) so errors and token counts are the API's own.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    TypeSafeAPIError,
    TypeSafeError,
    TypeSafeRateLimitError,
)

from jevex.jev import PRICE_PER_MILLION_INPUT_TOKENS, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

SELF_CAP_USD = min(0.30, float(os.environ.get("JEVEX_JEV_MAX_COST_USD") or 0.30))
"""Per pass: the lower of $0.30 and ``JEVEX_JEV_MAX_COST_USD`` (the SDK calls here bypass
``JevClient``, so the env cap is applied by this script)."""
OUT = Path(__file__).parent / f"results-{datetime.now(UTC).date().isoformat()}.json"
WORDS = (  # noqa: SIM905 - readable as prose
    "engine power torque gearbox manual automatic hybrid petrol diesel electric range battery "
    "charge seats boot litres kilowatts emissions mileage warranty trim alloy wheels the a of "
    "and with in on for to from by is are offers delivers includes standard optional"
).split()

spent_tokens = 0
log: dict[str, Any] = {"started": datetime.now(UTC).isoformat()}


def cost(tokens: int) -> float:
    return tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000


def prose(chars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    n = 0
    while n < chars:
        sentence = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 18))).capitalize()
        sentence += f" {rng.randint(10, 999)} {rng.choice(['kW', 'PS', 'mph', 'g/km', 'l'])}."
        out.append(sentence)
        n += len(sentence) + 1
    return " ".join(out)[:chars]


def table(rows: int) -> str:
    rng = random.Random(1)
    lines = ["Specification | SE | SE L | GT | R-Line"]
    for i in range(rows):
        lines.append(
            f"Spec {i} ({rng.choice(['kW', 'mph', 's', 'g/km'])}) | "
            + " | ".join(str(rng.randint(1, 999)) for _ in range(4))
        )
    return "\n".join(lines)


def blob(keys: int) -> dict[str, Any]:
    rng = random.Random(2)
    return {
        "vehicle": {
            f"field_{i}": {"value": rng.randint(1, 9999), "unit": rng.choice(["kW", "mph"])}
            for i in range(keys)
        }
    }


async def ask(client: AsyncTypeSafeClient, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    """One request; returns timing, usage and answers or the API's error."""
    global spent_tokens
    if cost(spent_tokens) >= SELF_CAP_USD:
        raise SystemExit(f"self cap reached: ${cost(spent_tokens):.4f}")
    t0 = time.perf_counter()
    try:
        r = await client.system_one(state, questions)  # pyright: ignore[reportUnknownMemberType]
    except TypeSafeRateLimitError as exc:
        return {"error": "rate_limited", "status": 429, "detail": str(exc)[:300]}
    except TypeSafeAPIError as exc:
        return {"error": type(exc).__name__, "detail": str(exc)[:500]}
    except TypeSafeError as exc:
        return {"error": type(exc).__name__, "detail": str(exc)[:500]}
    seconds = time.perf_counter() - t0
    spent_tokens += r.usage.input_tokens or 0
    return {
        "seconds": round(seconds, 3),
        "input_tokens": r.usage.input_tokens,
        "output_tokens": getattr(r.usage, "output_tokens", None),
        "model": r.model,
        "nouls": {k: v.noul for k, v in r.nouls.items()},
        "choices": {k: [v.choice, v.confidence] for k, v in r.choices.items()},
    }


def noul(i: int) -> Noul:
    return Noul(instructions=f"Does the text state a value for spec {i}?")


async def calibration(c: AsyncTypeSafeClient) -> dict[str, Any]:
    """Real tokens vs jevex's estimate, and how a request's state is billed."""
    out: dict[str, Any] = {"states": {}, "billing": {}}
    samples = {"prose": prose(4000), "table": table(80), "json": blob(60)}
    for name, state in samples.items():
        r = await ask(c, state, {"q": noul(0)})
        out["states"][name] = {
            "chars": len(json.dumps(state)) if not isinstance(state, str) else len(state),
            "estimate_tokens": estimate_tokens(state),
            "input_tokens": r.get("input_tokens"),
            "question_estimate": estimate_tokens(noul(0).model_dump()),
        }
    for n in (1, 10, 50):
        r = await ask(c, samples["prose"], {f"q{i}": noul(i) for i in range(n)})
        out["billing"][n] = r.get("input_tokens")
    return out


async def state_limit(c: AsyncTypeSafeClient, chars_per_token: float) -> list[dict[str, Any]]:
    """Where the API rejects a large state (documented: 32k for state + longest question)."""
    out: list[dict[str, Any]] = []
    for target in (28_000, 31_000, 33_000, 40_000):
        text = prose(int(target * chars_per_token), seed=target)
        r = await ask(c, text, {"q": noul(0)})
        out.append(
            {
                "target_tokens": target,
                **{k: r.get(k) for k in ("input_tokens", "error", "detail", "seconds")},
            }
        )
    return out


async def request_limit(c: AsyncTypeSafeClient) -> list[dict[str, Any]]:
    """Many questions on a small state: is there a question cap, and the 64k request limit?"""
    out: list[dict[str, Any]] = []
    long_q = "Does the text state the value for " + "this particular specification " * 40
    for n in (200, 1000):
        qs = {f"q{i}": Noul(instructions=f"{long_q} {i}?") for i in range(n)}
        est = sum(estimate_tokens(q.model_dump()) for q in qs.values())
        r = await ask(c, "Spec 3 is 120 kW.", qs)
        out.append(
            {
                "questions": n,
                "kind": "long",
                "estimate_tokens": est,
                **{k: r.get(k) for k in ("input_tokens", "error", "detail", "seconds")},
            }
        )
    for n in (1000, 3000):
        qs = {f"q{i}": Noul(instructions=f"Is {i} stated?") for i in range(n)}
        r = await ask(c, "Spec 3 is 120 kW.", qs)
        out.append(
            {
                "questions": n,
                "kind": "tiny",
                **{k: r.get(k) for k in ("input_tokens", "error", "detail", "seconds")},
            }
        )
    return out


async def latency(c: AsyncTypeSafeClient) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chars in (1_000, 10_000, 60_000):
        for nq in (1, 10, 100):
            times: list[float] = []
            for rep in range(3):
                r = await ask(c, prose(chars, seed=rep), {f"q{i}": noul(i) for i in range(nq)})
                if "seconds" in r:
                    times.append(r["seconds"])
            out.append(
                {
                    "state_chars": chars,
                    "questions": nq,
                    "seconds": times,
                    "median": statistics.median(times) if times else None,
                }
            )
    return out


LABELLED = [
    ("The 1.5 TSI SE produces 110 kW.", "Does the text state the engine power?", True),
    ("The 1.5 TSI SE produces 110 kW.", "Does the text state the top speed?", False),
    ("0-62 mph takes 9.1 seconds.", "Does the text state the acceleration time?", True),
    ("0-62 mph takes 9.1 seconds.", "Does the text state the price?", False),
    ("On the road from £24,995.", "Does the text state the price?", True),
    ("On the road from £24,995.", "Does the text state the fuel type?", False),
    ("Available as a plug-in hybrid.", "Does the text state the fuel type?", True),
    ("Available as a plug-in hybrid.", "Does the text state the number of seats?", False),
    ("Seats seven adults.", "Does the text state the number of seats?", True),
    ("Seats seven adults.", "Does the text state the CO2 emissions?", False),
]


async def batch_accuracy(c: AsyncTypeSafeClient) -> dict[str, Any]:
    """Does burying a question among distractors change its answer? And repeatability."""
    out: dict[str, Any] = {}
    for distractors in (0, 100, 400):
        right = 0
        ps: list[float] = []
        for text, q, truth in LABELLED:
            qs: dict[str, Any] = {"target": Noul(instructions=q)}
            qs |= {
                f"d{j}": Noul(instructions=f"Does the text mention item {j}?")
                for j in range(distractors)
            }
            r = await ask(c, text, qs)
            p = r.get("nouls", {}).get("target")
            if p is not None:
                ps.append(round(p, 4))
                right += (p >= 0.5) == truth
        out[str(distractors)] = {"correct": right, "of": len(LABELLED), "p": ps}
    reps = [
        await ask(c, LABELLED[0][0], {"q": Noul(instructions=LABELLED[0][1])}) for _ in range(3)
    ]
    out["repeat_p"] = [r.get("nouls", {}).get("q") for r in reps]
    choice = Choice(
        instructions="Which fuel?",
        criteria={"petrol": None, "diesel": None, "plug-in hybrid": None},
    )
    out["choice_repeat"] = [
        (await ask(c, "Available as a plug-in hybrid.", {"c": choice})).get("choices")
        for _ in range(3)
    ]
    return out


async def rate(c: AsyncTypeSafeClient, n: int = 300, concurrency: int = 50) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async def one(i: int) -> None:
        async with sem:
            results.append(
                await ask(
                    c, f"Item {i} costs £{i}.", {"q": Noul(instructions="Is a price stated?")}
                )
            )

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - t0
    errors: dict[str, int] = {}
    for r in results:
        if "error" in r:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    ok = [r["seconds"] for r in results if "seconds" in r]
    return {
        "requests": n,
        "concurrency": concurrency,
        "wall_seconds": round(wall, 2),
        "requests_per_minute": round(n / wall * 60),
        "errors": errors,
        "median_seconds": statistics.median(ok) if ok else None,
    }


async def main() -> None:
    c = AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0, timeout=120.0))
    try:
        cal: dict[str, Any] = await calibration(c)
        log["calibration"] = cal
        # Long prose tokenises looser than the 4k sample, so calibrate on a long one.
        long = prose(100_000, seed=7)
        r = await ask(c, long, {"q": noul(0)})
        cpt = len(long) / r["input_tokens"] if r.get("input_tokens") else 4.0
        log["chars_per_token_prose"] = round(cpt, 3)
        log["state_limit"] = await state_limit(c, cpt)
        log["edges"] = await edges(c, cpt)
        log["request_limit"] = await request_limit(c)
        log["latency"] = await latency(c)
        log["batch_accuracy"] = await batch_accuracy(c)
        log["rate"] = await rate(c)
        log["pipeline"] = await pipeline(c)
    finally:
        log["input_tokens_total"] = spent_tokens
        log["cost_usd"] = round(cost(spent_tokens), 5)
        log["finished"] = datetime.now(UTC).isoformat()
        OUT.write_text(json.dumps(log, indent=2, default=str) + "\n")
        print(f"wrote {OUT} (${cost(spent_tokens):.4f})")
        await c.aclose()


async def edges(c: AsyncTypeSafeClient, chars_per_token: float = 4.4) -> dict[str, Any]:
    """Second pass: find the real state limit (above 32k) and the request limit (near 64k)."""
    out: dict[str, Any] = {"state": [], "request": []}
    # Long prose runs at ~4.4 chars per real token (calibrated in ``main``).
    for target in (32_500, 34_000, 38_000, 48_000):
        r = await ask(c, prose(int(target * chars_per_token), seed=target), {"q": noul(0)})
        out["state"].append(
            {"target_tokens": target} | {k: r.get(k) for k in ("input_tokens", "error", "detail")}
        )
    # Tiny Nouls cost ~14 real tokens each on a tiny state.
    for n in (4_000, 4_400, 4_800, 6_000):
        qs = {f"q{i}": Noul(instructions=f"Is {i} stated?") for i in range(n)}
        r = await ask(c, "Spec 3 is 120 kW.", qs)
        out["request"].append(
            {"questions": n} | {k: r.get(k) for k in ("input_tokens", "error", "detail")}
        )
    return out


async def _add_to_results(key: str, run: Callable[[AsyncTypeSafeClient], Awaitable[Any]]) -> None:
    """Run one extra pass and add its section to today's file, even if the cap stops it."""
    c = AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0, timeout=120.0))
    data: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    try:
        data[key] = await run(c)
    except SystemExit as stop:  # the self cap
        data[key] = {"stopped": str(stop)}
    finally:
        data[f"{key}_cost_usd"] = round(cost(spent_tokens), 5)
        OUT.write_text(json.dumps(data, indent=2, default=str) + "\n")
        print(f"{key}: ${cost(spent_tokens):.4f}")
        await c.aclose()


PIPELINE_QUESTION = (
    "Does this document give technical specifications for one or more vehicle variants?"
)
"""The page-level document-gate wording used for the pricing runs (#242)."""


async def pipeline(c: AsyncTypeSafeClient) -> dict[str, Any]:
    """The default pipeline on four test-site pages (seed 42), metered by ``JevClient``.

    Unlike the other passes, its spend goes through ``JevClient`` (which applies the env cap)
    with the SDK's default retries; ``SELF_CAP_USD`` is checked only between documents.

    First the default document gate on a 4-trim page (#242: it says no), then every page
    with :data:`PIPELINE_QUESTION` as the gate wording and otherwise default settings.
    """
    global spent_tokens
    from jevex import Document, Extractor, SchemaConfig, SchemaSpec
    from jevex.jev import JevClient, TypeSafeBackend
    from jevex.testsite import VehicleSpec, generate, render

    class PageSpec(VehicleSpec):
        """A manufacturer's technical specification for one vehicle variant."""

        __jevex__ = SchemaConfig(document_question=PIPELINE_QUESTION)

    PageSpec.__name__ = "VehicleSpec"
    pages = render(generate(42))
    picks = {f: next(p for p in pages if p.family == f) for f in ("pdf", "table", "kv", "prose")}
    del c  # this pass keeps the SDK's default retries, so it makes its own client
    out: dict[str, Any] = {
        "default_gate_question": SchemaSpec.from_model(VehicleSpec)
        .document_gate_question()
        .instructions,
        "page_gate_question": PIPELINE_QUESTION,
        "pages": {},
    }

    def doc(p: Any) -> Any:
        return Document.from_bytes(
            p.content, url=f"https://site.test/{p.path}", content_type=p.content_type
        )

    backend = TypeSafeBackend()
    try:
        async with Extractor([VehicleSpec], jev=JevClient(backend)) as ex:
            r = await ex.extract(doc(picks["table"]))
            out["default_gate_p_on_table_page"] = r.meta.gates["VehicleSpec"].p
            spent_tokens += r.meta.jev.input_tokens
        async with Extractor([PageSpec], jev=JevClient(backend)) as ex:
            for name, p in picks.items():
                t0 = time.perf_counter()
                r = await ex.extract(doc(p))
                m = r.meta.jev
                spent_tokens += m.input_tokens
                out["pages"][name] = {
                    "path": p.path,
                    "entities": len(p.records),
                    "gate_p": r.meta.gates["VehicleSpec"].p,
                    "requests": m.requests,
                    "questions": m.questions,
                    "input_tokens": m.input_tokens,
                    "cost_usd": round(m.cost, 5),
                    "seconds": round(time.perf_counter() - t0, 2),
                }
    finally:
        await backend.aclose()
    return out


if __name__ == "__main__":
    import sys

    if "--edges" in sys.argv:
        asyncio.run(_add_to_results("edges", edges))
    elif "--pipeline" in sys.argv:
        asyncio.run(_add_to_results("pipeline", pipeline))
    else:
        asyncio.run(main())

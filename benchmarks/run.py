"""Run the benchmarks: every system over every corpus (docs/benchmarks.md › Protocol).

A live run makes real, billed calls to Jev and every pinned LLM, so it needs all extras,
the providers' keys, and hard spend caps adding up to no more than the config's budget,
with a ledger every process it starts adds to::

    uv sync --all-extras
    JEVEX_BENCH_BOOKS=/data/books JEVEX_BENCH_SPEC_SHEETS=/data/spec-sheets \\
    JEVEX_SPEND_LEDGER=/tmp/bench.ledger JEVEX_JEV_MAX_COST_USD=2 JEVEX_LLM_MAX_COST_USD=23 \\
        uv run benchmarks/run.py --all

``--dry-run`` checks every step for free: fake Jev and LLM answers, and the tools' scripts
with ``--fake``. Results go to ``benchmarks/results/<date>/`` (``--out``); build the page
from them with ``benchmarks/report.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jevex.baselines import pinned_llm
from jevex.bench import SYSTEMS, BenchRunError, run_benchmarks
from jevex.benchmarks import BenchmarkConfigError
from jevex.testing import FakeJev, FakeLLM

if TYPE_CHECKING:
    from jevex.benchmarks import PinnedModel
    from jevex.llm import LLM

HERE = Path(__file__).parent
SPEND_ENV = ("JEVEX_SPEND_LEDGER", "JEVEX_JEV_MAX_COST_USD", "JEVEX_LLM_MAX_COST_USD")


def fake_llm(pinned: PinnedModel) -> LLM:
    """Answers every call with an empty object, at no cost."""
    return FakeLLM(lambda _prompt, _schema: {}, model=pinned.model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    chosen = parser.add_mutually_exclusive_group(required=True)
    chosen.add_argument("--all", action="store_true", help="Every system on every corpus")
    chosen.add_argument("--system", action="append", choices=SYSTEMS, help="Run this system")
    parser.add_argument("--corpus", action="append", help="Only this corpus (repeatable)")
    parser.add_argument("--config", default=str(HERE / "config.yaml"), help="The pinned config")
    parser.add_argument("--out", help="A new results directory (default results/<date>)")
    parser.add_argument("--work", help="Unpublished files (default <out>.work)")
    parser.add_argument("--dry-run", action="store_true", help="Fake answers, no spend")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(message)s", stream=sys.stderr)
    logging.getLogger("jevex.bench").setLevel(logging.INFO)
    if args.dry_run:
        # Fake Jev answers are still metered; they mustn't count against a real ledger.
        for name in SPEND_ENV:
            os.environ.pop(name, None)
    out = args.out or str(HERE / "results" / f"{datetime.now(UTC):%Y-%m-%d}")
    try:
        manifest = asyncio.run(
            run_benchmarks(
                args.config,
                out,
                systems=SYSTEMS if args.all else args.system,
                corpora=args.corpus,
                work=args.work,
                jev=FakeJev(default_p=1.0).client() if args.dry_run else None,
                llm=fake_llm if args.dry_run else pinned_llm,
                tool_args=["--fake"] if args.dry_run else [],
                dry_run=args.dry_run,
            )
        )
    except (BenchRunError, BenchmarkConfigError) as exc:
        print(f"run.py: error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}: Jev ${manifest.jev_spend:.4f}, LLM ${manifest.llm_spend:.4f}")
    return 0 if manifest.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

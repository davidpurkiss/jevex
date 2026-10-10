"""Build the benchmark results page from a results directory (docs/benchmarks.md › Protocol).

    uv run benchmarks/report.py benchmarks/results/2026-10-10

writes ``docs/benchmarks-results.md`` (``--out``) and its charts beside it, in
``docs/benchmark-results/``. It reads only the results directory, never the corpora.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jevex.bench_report import write_results_page

ROOT = Path(__file__).parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="A results directory from benchmarks/run.py")
    parser.add_argument(
        "--out", default=str(ROOT / "docs" / "benchmarks-results.md"), help="The page to write"
    )
    args = parser.parse_args(argv)
    try:
        written = write_results_page(args.results, args.out)
    except (ValueError, OSError) as exc:
        print(f"report.py: error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

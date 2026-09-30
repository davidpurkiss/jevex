"""``scripts/agent-loop.sh --budget``: the caps the runner would give the next run."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "agent-loop.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def budget(agent_dir: Path, **env: str) -> list[str]:
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("JEVEX_")}
    run_env |= {"JEVEX_AGENT_DIR": str(agent_dir), **env}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--budget"], env=run_env, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def this_week(agent_dir: Path) -> Path:
    year, week, _ = datetime.now(UTC).isocalendar()
    path = agent_dir / "spend" / f"{year}-W{week:02d}"
    path.mkdir(parents=True)
    return path


def test_fresh_week_gets_the_per_run_caps(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent dir"  # paths with spaces must work
    assert budget(agent_dir)[1] == "next run caps: Jev $0.5000, LLM $2.0000"
    assert not agent_dir.exists()  # --budget only reads


def test_week_spend_lowers_the_caps(tmp_path: Path) -> None:
    week = this_week(tmp_path)
    (week / "run-1.ledger").write_text("jev 0.3\nllm 1.5\n")
    (week / "run-2.ledger").write_text("jev 1.4\nllm 9.9\n")  # LLM went over (one call can)
    old = tmp_path / "spend" / "2020-W01"
    old.mkdir()
    (old / "run-0.ledger").write_text("jev 5\n")  # earlier weeks don't count
    lines = budget(tmp_path)
    assert lines[0].endswith(": Jev $1.7000 of $2, LLM $11.4000 of $10")
    assert lines[1] == "next run caps: Jev $0.3000, LLM $0.0000"


def test_caps_come_from_the_environment(tmp_path: Path) -> None:
    (this_week(tmp_path) / "run-1.ledger").write_text("jev 0.9\n")
    lines = budget(
        tmp_path,
        JEVEX_JEV_RUN_USD="5",
        JEVEX_JEV_WEEK_USD="1",
        JEVEX_LLM_RUN_USD="0.25",
        JEVEX_LLM_WEEK_USD="3",
    )
    assert lines[1] == "next run caps: Jev $0.1000, LLM $0.2500"

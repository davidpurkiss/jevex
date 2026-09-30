import os
import subprocess
import sys
from pathlib import Path

import pytest

from jevex._spend import LEDGER_ENV, ledger_add, ledger_path, ledger_total


class LedgerError(Exception):
    pass


def test_ledger_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(LEDGER_ENV, raising=False)
    assert ledger_path() is None
    monkeypatch.setenv(LEDGER_ENV, "  ")
    assert ledger_path() is None
    monkeypatch.setenv(LEDGER_ENV, str(tmp_path / "run.ledger"))
    assert ledger_path() == tmp_path / "run.ledger"


def test_totals_per_kind(tmp_path: Path) -> None:
    ledger = tmp_path / "run.ledger"
    ledger.write_text("jev 0.25\nllm 1.5\n\njev 0.125\n")
    assert ledger_total(ledger, "jev", LedgerError) == pytest.approx(0.375)
    assert ledger_total(ledger, "llm", LedgerError) == pytest.approx(1.5)


def test_missing_ledger_is_created_empty(tmp_path: Path) -> None:
    ledger = tmp_path / "new.ledger"
    assert ledger_total(ledger, "jev", LedgerError) == 0.0
    assert ledger.read_text() == ""


def test_unwritable_ledger_raises_the_callers_error(tmp_path: Path) -> None:
    ledger = tmp_path / "missing-dir" / "run.ledger"
    with pytest.raises(LedgerError, match="can't use"):
        ledger_total(ledger, "jev", LedgerError)
    with pytest.raises(LedgerError, match="can't write"):
        ledger_add(ledger, "jev", 0.1, LedgerError)


SPEND_ENV = {"JEVEX_SPEND_LEDGER", "JEVEX_JEV_MAX_COST_USD", "JEVEX_LLM_MAX_COST_USD"}


def test_offline_tests_see_no_live_spend_settings() -> None:
    # conftest drops them, so an agent run whose week is spent can still run the suite
    assert not SPEND_ENV & set(os.environ)


def test_offline_tests_drop_spend_settings_they_inherit(tmp_path: Path) -> None:
    ledger = tmp_path / "run.ledger"
    env = {k: v for k, v in os.environ.items() if k not in {"JEVEX_LIVE", "JEVEX_RECORD"}}
    env |= {
        "JEVEX_SPEND_LEDGER": str(ledger),
        "JEVEX_JEV_MAX_COST_USD": "0",
        "JEVEX_LLM_MAX_COST_USD": "0",
    }
    here = Path(__file__)
    test = f"{here}::test_offline_tests_see_no_live_spend_settings"
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
        cwd=here.parent.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert not ledger.exists()


def test_add_appends_lines_shell_tools_can_sum(tmp_path: Path) -> None:
    ledger = tmp_path / "run.ledger"
    ledger_add(ledger, "jev", 0.0000042, LedgerError)
    ledger_add(ledger, "llm", 0.5, LedgerError)
    ledger_add(ledger, "jev", 0.001, LedgerError)
    assert ledger.read_text() == "jev 0.000004200\nllm 0.500000000\njev 0.001000000\n"
    assert ledger_total(ledger, "jev", LedgerError) == pytest.approx(0.0010042)


@pytest.mark.parametrize("line", ["jev", "jev lots", "jev 1 2", "llm nan", "jev inf", "llm -1"])
def test_bad_line_raises_the_callers_error(tmp_path: Path, line: str) -> None:
    ledger = tmp_path / "run.ledger"
    ledger.write_text(f"jev 0.1\n{line}\n")
    with pytest.raises(LedgerError, match="bad line 2"):
        ledger_total(ledger, "llm", LedgerError)

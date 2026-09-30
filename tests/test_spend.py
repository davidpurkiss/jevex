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


def test_missing_ledger_is_empty(tmp_path: Path) -> None:
    assert ledger_total(tmp_path / "none.ledger", "jev", LedgerError) == 0.0


def test_add_appends_lines_shell_tools_can_sum(tmp_path: Path) -> None:
    ledger = tmp_path / "run.ledger"
    ledger_add(ledger, "jev", 0.0000042)
    ledger_add(ledger, "llm", 0.5)
    ledger_add(ledger, "jev", 0.001)
    assert ledger.read_text() == "jev 0.000004200\nllm 0.500000000\njev 0.001000000\n"
    assert ledger_total(ledger, "jev", LedgerError) == pytest.approx(0.0010042)


@pytest.mark.parametrize("line", ["jev", "jev lots", "jev 1 2"])
def test_bad_line_raises_the_callers_error(tmp_path: Path, line: str) -> None:
    ledger = tmp_path / "run.ledger"
    ledger.write_text(f"jev 0.1\n{line}\n")
    with pytest.raises(LedgerError, match="bad line 2"):
        ledger_total(ledger, "llm", LedgerError)

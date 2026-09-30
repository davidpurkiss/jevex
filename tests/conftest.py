"""Network is blocked in tests (pytest-socket). Tests that call real APIs are marked
``live`` and only run with ``--live`` or ``JEVEX_LIVE=1``. ``JEVEX_RECORD=1`` re-records
Jev cassettes, so it also allows the network."""

import os

import pytest

# LiteLLM fetches its price map from GitHub at import time unless told to use the bundled
# copy; set it before any test module imports litellm.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(autouse=True)
def _offline_spend_stays_off_the_ledger(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent runs set ``JEVEX_SPEND_LEDGER`` to count real spend. Fake and replayed calls
    must not be charged to it, so only live tests and cassette recording keep it."""
    if "live" in request.keywords or os.environ.get("JEVEX_RECORD") == "1":
        return
    monkeypatch.delenv("JEVEX_SPEND_LEDGER", raising=False)


@pytest.fixture
def typesafe_api_key() -> str:
    """The real Jev key for ``live`` tests; skips the test when it isn't set."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        pytest.skip("TYPESAFE_API_KEY not set")
    return key


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="run tests that call real APIs")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: calls real external APIs (needs --live)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live = bool(config.getoption("--live")) or os.environ.get("JEVEX_LIVE") == "1"
    record = os.environ.get("JEVEX_RECORD") == "1"
    for item in items:
        if "live" in item.keywords and not live:
            item.add_marker(pytest.mark.skip(reason="live test; run with --live"))
        elif live or record:
            item.add_marker(pytest.mark.enable_socket)

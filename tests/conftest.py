"""Network is blocked in tests (pytest-socket). Tests that call real APIs are marked
``live`` and only run with ``--live`` or ``JEVEX_LIVE=1``. ``JEVEX_RECORD=1`` re-records
Jev cassettes, so it also allows the network."""

import os

import pytest


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

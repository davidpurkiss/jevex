"""Network is blocked in tests (pytest-socket). Tests that call real APIs are marked
``live`` and only run with ``--live`` or ``JEVEX_LIVE=1``. ``JEVEX_RECORD=1`` re-records
Jev cassettes, so it also allows the network."""

import importlib
import importlib.util
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

# LiteLLM fetches its price map from GitHub at import time unless told to use the bundled
# copy; set it before any test module imports litellm.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _disable_onnxruntime_telemetry() -> None:
    # onnxruntime's telemetry thread (macOS builds) can race the interpreter's shutdown and
    # abort it ("recursive_mutex lock failed"), failing a run whose tests all passed.
    if importlib.util.find_spec("onnxruntime") is not None:
        importlib.import_module("onnxruntime").disable_telemetry_events()


_disable_onnxruntime_telemetry()


SPEND_ENV = ("JEVEX_SPEND_LEDGER", "JEVEX_JEV_MAX_COST_USD", "JEVEX_LLM_MAX_COST_USD")


@pytest.fixture(autouse=True)
def _offline_tests_ignore_live_spend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent runs set a spend ledger and caps for real calls. Fake and replayed calls must
    neither be charged to the ledger nor fail because the week's budget is spent, so only
    live tests and cassette recording keep them. Tests that need a cap set their own."""
    if "live" in request.keywords or os.environ.get("JEVEX_RECORD") == "1":
        return
    for name in SPEND_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def typesafe_api_key() -> str:
    """The real Jev key for ``live`` tests; skips the test when it isn't set."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        pytest.skip("TYPESAFE_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """A Postgres server for the store tests: ``JEVEX_TEST_POSTGRES_URL`` if set, else a
    bundled one (``pixeltable-pgserver``, a dev dependency) on a unix socket, which the
    network block allows. Skips when neither is available.

    The URL must be a throwaway database: tests create and drop schemas there, including
    ``jevex``, the default schema's learned state."""
    url = os.environ.get("JEVEX_TEST_POSTGRES_URL")
    if url:
        yield url
        return
    pgserver = pytest.importorskip("pixeltable_pgserver")
    # A short path: the server's unix socket lives here, and socket paths are capped at
    # ~100 characters.
    data = Path(tempfile.mkdtemp(prefix="jxpg"))
    server = pgserver.get_server(data, cleanup_mode="stop")
    try:
        yield server.get_uri()
    finally:
        server.cleanup()
        shutil.rmtree(data, ignore_errors=True)


@pytest.fixture
def pg_schema(postgres_url: str) -> Iterator[str]:
    """A fresh Postgres schema name for one test's store, dropped afterwards."""
    psycopg = pytest.importorskip("psycopg")
    name = f"test_{uuid.uuid4().hex[:12]}"
    yield name
    with psycopg.connect(postgres_url, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")


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

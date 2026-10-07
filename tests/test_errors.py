import pytest

from jevex.errors import (
    MAX_MESSAGE_CHARS,
    ExtractionError,
    PartError,
    PartErrors,
    status_of,
)


def test_failures_are_merged_per_stage_kind_part_and_type() -> None:
    errors = PartErrors()
    errors.add("candidates", "generator", "g1", RuntimeError("first"))
    errors.add("candidates", "generator", "g1", RuntimeError("second"))
    errors.add("candidates", "generator", "g1", KeyError("k"))
    errors.add("candidates", "generator", "g2", RuntimeError("other"))
    assert errors.errors == [
        PartError(
            stage="candidates",
            kind="generator",
            part="g1",
            type="RuntimeError",
            message="first",
            count=2,
        ),
        PartError(stage="candidates", kind="generator", part="g1", type="KeyError", message="'k'"),
        PartError(
            stage="candidates", kind="generator", part="g2", type="RuntimeError", message="other"
        ),
    ]
    assert len(errors) == 3
    assert errors.status == "partial"
    assert errors.cause is None


def test_a_fatal_error_fails_the_document_and_keeps_its_exception() -> None:
    errors = PartErrors()
    assert errors.status == "ok"
    boom = ValueError("bad")
    errors.add("layout", "stage", None, boom, fatal=True)
    errors.add("layout", "stage", None, ValueError("again"), fatal=True)
    assert errors.status == "failed"
    assert errors.cause is boom


def test_status_of() -> None:
    soft = PartError(stage="s", kind="generator", type="E", message="m")
    hard = soft.model_copy(update={"fatal": True})
    assert status_of([]) == "ok"
    assert status_of([soft]) == "partial"
    assert status_of([soft, hard]) == "failed"


class Silent(Exception):
    """Raised without a message."""


def test_long_and_empty_messages() -> None:
    errors = PartErrors()
    errors.add("s", "normaliser", "n", RuntimeError("x" * 2000))
    errors.add("s", "normaliser", "n", Silent())
    long, empty = errors.errors
    assert len(long.message) == MAX_MESSAGE_CHARS
    assert long.message.endswith("…")
    assert empty.message == "Silent"


def test_describe() -> None:
    error = PartError(
        stage="candidates", kind="generator", part="g1", type="RuntimeError", message="m", count=3
    )
    assert error.describe() == "candidates generator g1: RuntimeError: m (x3)"
    once = PartError(stage="layout", kind="document", type="PdfLayoutError", message="broken")
    assert once.describe() == "layout document: PdfLayoutError: broken"


def test_extraction_error_lists_the_errors() -> None:
    errors = [
        PartError(stage="select", kind="jev", type="JevBackendError", message="down", fatal=True)
    ]
    exc = ExtractionError("failed", errors, "https://a.test/x")
    assert str(exc) == ("extraction failed for https://a.test/x: select jev: JevBackendError: down")
    assert (exc.status, exc.errors, exc.url) == ("failed", errors, "https://a.test/x")
    with pytest.raises(ExtractionError, match=r"^extraction partial: "):
        raise ExtractionError("partial", errors)

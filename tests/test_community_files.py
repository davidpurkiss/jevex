"""GitHub only reports a broken issue form by silently dropping it from the chooser, so the
forms are checked here. PyYAML comes in with the dev tools (pre-commit depends on it)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"
INPUT_TYPES = {"input", "textarea", "dropdown", "checkboxes"}
_Mapping = TypeAdapter(dict[str, Any])
_Body = TypeAdapter(list[dict[str, Any]])


def load(name: str) -> dict[str, Any]:
    return _Mapping.validate_python(yaml.safe_load((TEMPLATES / name).read_text()))


def form_problems(form: dict[str, Any]) -> list[str]:
    """The subset of GitHub's issue-form rules that a hand edit is likely to break."""
    problems = [f"missing {key!r}" for key in ("name", "description", "body") if key not in form]
    body = _Body.validate_python(form.get("body") or [])
    if not body:
        return [*problems, "body must be a non-empty list"]
    ids: set[str] = set()
    for i, element in enumerate(body):
        kind = element.get("type")
        if kind == "markdown":
            continue
        if kind not in INPUT_TYPES:
            problems.append(f"body[{i}]: unknown type {kind!r}")
            continue
        if not element.get("attributes", {}).get("label"):
            problems.append(f"body[{i}]: missing label")
        if kind == "dropdown" and not element["attributes"].get("options"):
            problems.append(f"body[{i}]: dropdown without options")
        element_id = element.get("id")
        if element_id in ids:
            problems.append(f"body[{i}]: duplicate id {element_id!r}")
        if element_id:
            ids.add(element_id)
    if not any(e.get("validations", {}).get("required") is True for e in body):
        problems.append("no required field")
    return problems


@pytest.mark.parametrize("name", ["bug.yml", "feature.yml"])
def test_issue_forms_are_valid(name: str) -> None:
    assert form_problems(load(name)) == []


def test_issue_forms_apply_existing_labels() -> None:
    labels = {name: load(name)["labels"] for name in ("bug.yml", "feature.yml")}
    assert labels == {"bug.yml": ["bug"], "feature.yml": ["enhancement"]}


def test_form_problems_catches_broken_forms() -> None:
    form: dict[str, Any] = {
        "name": "Broken",
        "body": [
            {"type": "input", "id": "a", "attributes": {"label": "A"}},
            {"type": "input", "id": "a", "attributes": {}},
            {"type": "dropdown", "id": "b", "attributes": {"label": "B"}},
            {"type": "radio", "id": "c"},
        ],
    }
    assert form_problems(form) == [
        "missing 'description'",
        "body[1]: missing label",
        "body[1]: duplicate id 'a'",
        "body[2]: dropdown without options",
        "body[3]: unknown type 'radio'",
        "no required field",
    ]
    assert form_problems({"name": "x", "description": "y", "body": []}) == [
        "body must be a non-empty list"
    ]


def test_issue_chooser_config_keeps_blank_issues() -> None:
    # Backlog tasks and design questions fit neither form.
    config = load("config.yml")
    assert config["blank_issues_enabled"] is True
    assert all({"name", "url", "about"} <= link.keys() for link in config["contact_links"])


def test_pr_template_starts_with_closes_line() -> None:
    template = (ROOT / ".github" / "pull_request_template.md").read_text()
    assert template.startswith("Closes #")
    assert "## Departures from the spec" in template


@pytest.mark.parametrize(
    ("doc", "link"),
    [
        ("README.md", "(CONTRIBUTING.md)"),
        ("README.md", "(CODE_OF_CONDUCT.md)"),
        ("CONTRIBUTING.md", "(CODE_OF_CONDUCT.md)"),
        ("CONTRIBUTING.md", "(.github/pull_request_template.md)"),
    ],
)
def test_community_files_are_linked(doc: str, link: str) -> None:
    assert link in (ROOT / doc).read_text()

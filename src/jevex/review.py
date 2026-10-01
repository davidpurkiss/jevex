"""Review sink: uncertain fields go to people, and their answers come back as examples
(spec: *Results API › Review sink*; *Learning loop › Housekeeping*).

Opt-in through ``Extractor(review_sink=...)``. After each document, the extractor sends the
sink every found value whose confidence is below the review threshold
(``review_threshold``, or per field ``review_thresholds``), as :class:`ReviewItem`\\ s in
one call. Child records' values (``ParentChild``) are sent too. A value without a
confidence (a structured-data lookup) isn't: there's no question whose answer was unsure.

A person's answer goes back through ``Extractor.feedback(item, value)``. It becomes a human
:class:`~jevex.store.VerifiedExample` on the item's statement, stored and learned from like
an LLM answer Jev verified (a human's always passes the learn threshold).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from jevex.results import FieldMeta, threshold_for
from jevex.store import VerifiedExample, example_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from jevex.results import Extracted
    from jevex.statements import Statement

REVIEW_THRESHOLD = 0.8
"""Confidence below which a value is sent for review. Provisional: the spec's open questions
set the defaults from eval runs."""


class ReviewItem(BaseModel):
    """One uncertain value, for a person to confirm or correct.

    ``field`` is ``"Schema.field"`` (``"Parent.nested_field.field"`` in a child record), as
    in :class:`~jevex.store.VerifiedExample`. ``meta`` is everything known about the value,
    including its source statement and the alternatives. ``threshold`` is the review
    threshold it fell below. ``context`` is what the learner replays the statement with
    (heading trail and statement kind). ``id`` is the same for the same value from the same
    statement (id and text) of the same entity and URL, so a sink can drop repeats.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    field: str
    entity: str
    url: str | None
    meta: FieldMeta
    threshold: float
    context: dict[str, Any] = Field(default_factory=dict[str, Any])

    def example(self, value: Any, *, evidence: tuple[int, int] | None = None) -> VerifiedExample:
        """A person's answer as a verified example on the item's statement.

        ``evidence`` is the ``(start, end)`` span of the value in the statement. Confirming
        the extracted value without one reuses the value's own span (not a list field's:
        its span is one item's). ``value`` is taken as given; ``Extractor.feedback``
        normalises it first. Raises ``ValueError``
        when the value has no source statement, ``value`` is ``None``, or ``evidence`` isn't
        within the statement.
        """
        source = self.meta.source
        statement = source.statement if source is not None else None
        if source is None or statement is None:
            raise ValueError(f"{self.field} has no source statement to learn from")
        if value is None:
            raise ValueError(f"feedback for {self.field} needs a value, got None")
        if (
            evidence is None
            and source.span is not None
            and not isinstance(self.meta.value, list)
            and value == self.meta.value
        ):
            evidence = (source.span.start, source.span.end)
        if evidence is not None:
            start, end = evidence
            if not 0 <= start < end <= len(statement):
                raise ValueError(
                    f"evidence {evidence} isn't a span of the statement ({len(statement)} chars)"
                )
        return VerifiedExample(
            id=example_id(self.field, statement, value),
            field=self.field,
            statement=statement,
            value=value,
            evidence=evidence,
            context=dict(self.context),
            source="human",
        )


@runtime_checkable
class ReviewSink(Protocol):
    """Receives a document's uncertain values (``Extractor(review_sink=...)``).

    ``send`` is called once per document that has any, after its result is built. It
    should hand the items on (a queue, a database, a review UI) rather than wait for
    answers; an exception it raises propagates from ``extract``.
    """

    async def send(self, items: Sequence[ReviewItem]) -> None: ...


class ReviewQueue:
    """A :class:`ReviewSink` that keeps every item in memory, in arrival order (for
    notebooks, tests, and review loops in the same process)."""

    def __init__(self) -> None:
        self.items: list[ReviewItem] = []

    async def send(self, items: Sequence[ReviewItem]) -> None:
        self.items.extend(items)


def review_items(
    records: Iterable[Extracted[BaseModel]],
    *,
    threshold: float = REVIEW_THRESHOLD,
    thresholds: Mapping[str, float] | None = None,
    statements: Mapping[str, Statement] | None = None,
    url: str | None = None,
) -> list[ReviewItem]:
    """The records' (and their children's) found values with a confidence below the
    threshold: ``thresholds`` per field (keys as for the extractor's ``thresholds``), else
    ``threshold``. ``statements`` (by id) give each item its statement's ``context``."""
    thresholds = thresholds or {}
    statements = statements or {}
    out: list[ReviewItem] = []
    for record in records:
        for name, meta in record.meta.items():
            if not meta.found or meta.confidence is None:
                continue
            limit = threshold_for(thresholds, threshold, record.schema_name, name)
            if meta.confidence >= limit:
                continue
            field = f"{record.schema_name}.{name}"
            source = meta.source
            statement = (
                statements.get(source.statement_id)
                if source is not None and source.statement_id is not None
                else None
            )
            out.append(
                ReviewItem(
                    id=_item_id(url, field, record.entity, meta),
                    field=field,
                    entity=record.entity,
                    url=url,
                    meta=meta,
                    threshold=limit,
                    context={"heading_trail": statement.heading_trail, "kind": statement.kind}
                    if statement is not None
                    else {},
                )
            )
        for kids in record.children.values():
            out += review_items(
                kids, threshold=threshold, thresholds=thresholds, statements=statements, url=url
            )
    return out


def _item_id(url: str | None, field: str, entity: str, meta: FieldMeta) -> str:
    source = meta.source
    statement_id, statement = (source.statement_id, source.statement) if source else (None, None)
    key = f"{url}\0{field}\0{entity}\0{statement_id}\0{statement}\0{meta.value!r}"
    return f"rv-{hashlib.sha256(key.encode()).hexdigest()[:12]}"

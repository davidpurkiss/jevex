"""The generator registry: which generators apply to a field, and running them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jevex.document import normalise_source
from jevex.interfaces import FieldAwareGenerator, LocaleAwareGenerator

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from jevex.interfaces import CandidateGenerator, Scope
    from jevex.schema import FieldSpec
    from jevex.statements import Candidate, Statement


def scope_matches(
    scope: Scope,
    field: FieldSpec,
    *,
    schema: str,
    locale: str | None = None,
    source: str | None = None,
) -> bool:
    """Whether a generator with ``scope`` applies. Empty scope parts match anything.

    ``fields`` entries may be bare (``zero_to_62_s``) or qualified
    (``VehicleSpec.zero_to_62_s``). A scope locale of ``en`` matches ``en-GB``; locales
    compare case-insensitively with ``_`` treated as ``-``. ``source`` is the document's
    :attr:`~jevex.Document.source`; scope sources compare after
    :func:`~jevex.document.normalise_source`. A locale- or source-scoped
    generator does **not** run when the document's locale or source is unknown: a
    decimal-comma generator mustn't guess.
    """
    if scope.kinds and field.kind not in scope.kinds:
        return False
    if scope.fields and not ({field.name, f"{schema}.{field.name}"} & scope.fields):
        return False
    if scope.schemas and schema not in scope.schemas:
        return False
    if scope.locale:
        want, have = _locale(scope.locale), _locale(locale or "")
        if not have or not (have == want or have.startswith(f"{want}-")):
            return False
    if not scope.sources:
        return True
    return source is not None and normalise_source(source) in {
        normalise_source(s) for s in scope.sources
    }


def _locale(tag: str) -> str:
    return tag.replace("_", "-").lower()


class GeneratorRegistry:
    """An immutable, ordered set of generators with unique ids.

    ``with_generator`` and ``without`` return new registries, so documents already running
    keep the snapshot they started with (the learner's hot-swap relies on this, #38).
    Earlier generators win when two propose the same span.
    """

    def __init__(self, generators: Iterable[CandidateGenerator] = ()) -> None:
        self._generators = tuple(generators)
        ids = [g.id for g in self._generators]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate generator ids: {duplicates}")

    def __iter__(self) -> Iterator[CandidateGenerator]:
        return iter(self._generators)

    def __len__(self) -> int:
        return len(self._generators)

    def __contains__(self, generator_id: object) -> bool:
        return any(g.id == generator_id for g in self._generators)

    @property
    def ids(self) -> list[str]:
        """Generator ids in registry (tie-winning) order."""
        return [g.id for g in self._generators]

    def get(self, generator_id: str) -> CandidateGenerator:
        """The generator with this id; ``KeyError`` if there's none."""
        for g in self._generators:
            if g.id == generator_id:
                return g
        raise KeyError(generator_id)

    def with_generator(self, generator: CandidateGenerator) -> GeneratorRegistry:
        """Add ``generator``, replacing one with the same id in place."""
        if generator.id in self:
            return GeneratorRegistry(
                generator if g.id == generator.id else g for g in self._generators
            )
        return GeneratorRegistry([*self._generators, generator])

    def extended(self, generators: Iterable[CandidateGenerator]) -> GeneratorRegistry:
        """This registry followed by ``generators``, leaving out ids it already has (so its
        own generators keep winning span ties and ids)."""
        added = [g for g in generators if g.id not in self]
        return GeneratorRegistry([*self._generators, *added]) if added else self

    def without(self, generator_id: str) -> GeneratorRegistry:
        """A registry without this generator; ``KeyError`` if it isn't registered."""
        self.get(generator_id)
        return GeneratorRegistry(g for g in self._generators if g.id != generator_id)

    def for_field(
        self,
        field: FieldSpec,
        *,
        schema: str,
        locale: str | None = None,
        source: str | None = None,
    ) -> list[CandidateGenerator]:
        """Generators whose scope matches this field and document, in registry order."""
        return [
            g
            for g in self._generators
            if scope_matches(g.scope, field, schema=schema, locale=locale, source=source)
        ]

    def generate(
        self,
        statement: Statement,
        field: FieldSpec,
        *,
        schema: str,
        locale: str | None = None,
        source: str | None = None,
    ) -> list[Candidate]:
        """Candidates from every applicable generator, one per distinct span, in text order.

        A :class:`~jevex.interfaces.LocaleAwareGenerator` is given the field and ``locale``;
        a :class:`~jevex.interfaces.FieldAwareGenerator`, the field.
        """
        seen: set[tuple[int, int]] = set()
        out: list[Candidate] = []
        for generator in self.for_field(field, schema=schema, locale=locale, source=source):
            if isinstance(generator, LocaleAwareGenerator):
                candidates = generator.generate_in(statement, field, locale)
            elif isinstance(generator, FieldAwareGenerator):
                candidates = generator.generate_for(statement, field)
            else:
                candidates = generator.generate(statement)
            for candidate in candidates:
                key = (candidate.span.start, candidate.span.end)
                if key not in seen:
                    seen.add(key)
                    out.append(candidate)
        return sorted(out, key=lambda c: (c.span.start, c.span.end))

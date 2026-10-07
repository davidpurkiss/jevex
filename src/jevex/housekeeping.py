"""Generator housekeeping: hit and win rates, pruning, dedup (spec: *Learning loop ›
Housekeeping*).

The learn stage hands each finished document to a :class:`Housekeeper`, which adds to
the store's :class:`~jevex.store.GeneratorStats` for every generator the candidate stage
ran there (:func:`generator_use`):

- ``documents``: the document was in the generator's scope, and the candidate stage ran it
  on at least one statement.
- ``hits``: it gave at least one candidate there.
- ``wins``: one of its candidates became a field value that stood: it normalised, fitted
  the field, and no LLM fallback replaced it. Without ground truth at run time, a value
  standing is what "correct" means.

So ``hit_rate`` is hits per document and ``win_rate`` wins per hit. A learned generator
(one in the document's :class:`~jevex.learn.GeneratorSnapshot`: the store's or a pack's,
:mod:`jevex.packs`) with no wins after
``prune_after`` documents is pruned: added to the store's disable list (kept, not deleted)
and dropped from the snapshot later documents take. Built-in and stage generators are
counted but never pruned.

A generator that raises is skipped for that statement (:mod:`jevex.errors`), and each
failure is added to its ``failures``. A learned generator with :data:`QUARANTINE_AFTER`
failures is **quarantined**: disabled like a pruned one (kept for review) and reported
with a ``generator_quarantined`` event. Other generators and the LLM fallback still cover
its field.

:meth:`Housekeeper.dedupe` finds stored generators that are duplicates: same field and
scope, and the same candidates (spans and normaliser chain) on every stored example of
the field. It keeps the one with the most wins (the oldest on a tie) and disables the
rest. It runs no Jev or LLM calls, so it's safe to run whenever (``Extractor.dedupe_generators``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from jevex._tasks import gather
from jevex.learn import LearnedGenerators, example_statement
from jevex.logs import get_logger

if TYPE_CHECKING:
    from jevex.generators import GeneratorSpec
    from jevex.pipeline import Context
    from jevex.statements import Candidate
    from jevex.store import Store

log = get_logger(__name__)

PRUNE_AFTER = 50
"""Scoped documents a learned generator gets to win once before it's disabled.
Provisional: the spec's open questions set the defaults from eval runs (#49)."""

QUARANTINE_AFTER = 3
"""Failures (statements it raised on, over every document) after which a learned
generator is disabled (#230)."""


@dataclass(frozen=True)
class GeneratorUse:
    """Which generators one document ran, got candidates from, and took values from."""

    ran: frozenset[str]
    hits: frozenset[str]
    wins: frozenset[str]


def generator_use(ctx: Context) -> GeneratorUse:
    """What each generator did on this document (see the module docstring).

    ``ran`` comes from ``ctx.generators_ran`` (the candidate stage's), plus any generator
    whose candidates are on the context, in case a custom stage made them.
    """
    hits: set[str] = set()
    wins: set[str] = set()
    for run in ctx.schemas.values():
        for candidates in run.candidates.values():
            hits.update(c.generator_id for c in candidates)
        for scope, metas in run.fields.items():
            for name, meta in metas.items():
                if meta.method not in ("generator", "vision") or not meta.found:
                    continue
                used = run.value_generators.get((scope, name))
                if used is not None:
                    # A list holds the picks of every statement, not only the best one's.
                    wins.update(used)
                elif meta.generator_id is not None:
                    wins.add(meta.generator_id)
    return GeneratorUse(
        ran=frozenset(ctx.generators_ran | hits),
        hits=frozenset(hits),
        wins=frozenset(wins & hits),
    )


class DuplicateGenerator(BaseModel):
    """A generator :meth:`Housekeeper.dedupe` disabled, and the one it duplicated."""

    model_config = ConfigDict(frozen=True)

    id: str
    kept: str
    field: str
    examples: int
    """How many stored examples the two were compared on."""


@dataclass
class Housekeeper:
    """Keeps the learned generators healthy: stats, pruning, dedup.

    ``store`` holds the stats and the disable list. ``generators`` is the extractor's
    :class:`~jevex.learn.LearnedGenerators`: a generator disabled here is also dropped from
    its snapshot, so later documents in this process stop running it. ``prune_after=None``
    turns pruning off.
    """

    store: Store
    generators: LearnedGenerators | None = None
    prune_after: int | None = PRUNE_AFTER
    pruned: list[str] = field(default_factory=list[str])
    """Ids this housekeeper pruned, in order."""
    quarantined: list[str] = field(default_factory=list[str])
    """Ids this housekeeper quarantined (disabled for failing), in order."""

    def __post_init__(self) -> None:
        if self.prune_after is not None and self.prune_after < 1:
            raise ValueError(f"prune_after must be at least 1, got {self.prune_after}")

    async def record(self, ctx: Context) -> list[str]:
        """Add one document's counts to the store, quarantine, then prune; return the ids
        pruned.

        Each pruned generator gets a ``generator_pruned`` event on ``ctx``, and each
        quarantined one a ``generator_quarantined`` event.
        """
        use = generator_use(ctx)
        failures = _failures(ctx)
        await gather(
            self.store.record_generator_stats(
                gid,
                documents=1,
                hits=int(gid in use.hits),
                wins=int(gid in use.wins),
                failures=failures.get(gid, 0),
            )
            for gid in sorted(use.ran | set(failures))
        )
        if ctx.generators is None:
            return []
        learned = set(ctx.generators.registry.ids)
        await self._quarantine(ctx, sorted(set(failures) & learned))
        if self.prune_after is None:
            return []
        learned -= set(self.quarantined)
        suspects = sorted((use.ran & learned) - use.wins)
        stats = await gather(self.store.generator_stats(gid) for gid in suspects)
        pruned: list[str] = []
        for s in stats:
            # Another document may have pruned it while this one read the stats.
            if s.generator_id in self.pruned:
                continue
            if s.wins == 0 and s.documents >= self.prune_after:
                pruned.append(s.generator_id)
                ctx.event(
                    "learn",
                    "generator_pruned",
                    f"generator {s.generator_id} won nothing in {s.documents} documents",
                    generator_id=s.generator_id,
                    documents=s.documents,
                    hits=s.hits,
                )
                log.info(
                    "pruned generator %s: no wins in %d documents",
                    s.generator_id,
                    s.documents,
                    extra={"part": s.generator_id},
                )
        self.pruned.extend(pruned)  # before awaiting, so no other document claims them
        await self._disable(pruned)
        return pruned

    async def _quarantine(self, ctx: Context, failed: list[str]) -> None:
        """Disable the learned generators in ``failed`` that reached
        :data:`QUARANTINE_AFTER` failures."""
        stats = await gather(self.store.generator_stats(gid) for gid in failed)
        quarantined: list[str] = []
        for s in stats:
            if s.generator_id in self.quarantined or s.failures < QUARANTINE_AFTER:
                continue
            quarantined.append(s.generator_id)
            ctx.event(
                "learn",
                "generator_quarantined",
                f"generator {s.generator_id} failed {s.failures} times and was disabled",
                generator_id=s.generator_id,
                failures=s.failures,
            )
            log.warning(
                "quarantined generator %s: it failed %d times",
                s.generator_id,
                s.failures,
                extra={"part": s.generator_id},
            )
        self.quarantined.extend(quarantined)  # before awaiting, as for pruning
        await self._disable(quarantined)

    async def dedupe(self) -> list[DuplicateGenerator]:
        """Disable every enabled stored generator that duplicates another; return them.

        Two generators are duplicates when they have the same field and scope, and give
        the same candidates on every stored example of that field. A pair that finds
        nothing on any example isn't: there's no evidence they agree. Raises
        :class:`~jevex.store.StoreError` for a stored spec that doesn't validate.
        """
        by_field: dict[str, list[GeneratorSpec]] = {}
        for spec in await LearnedGenerators(self.store).stored():
            by_field.setdefault(spec.field, []).append(spec)
        found: list[DuplicateGenerator] = []
        for name, specs in by_field.items():
            if len(specs) > 1:
                found.extend(await self._duplicates(name, specs))
        await self._disable([d.id for d in found])
        for d in found:
            log.info("disabled generator %s: a duplicate of %s", d.id, d.kept, extra={"part": d.id})
        return found

    async def _duplicates(self, name: str, specs: list[GeneratorSpec]) -> list[DuplicateGenerator]:
        statements = [example_statement(e) for e in await self.store.examples(name)]
        groups: dict[tuple[str, tuple[frozenset[_Key], ...]], list[GeneratorSpec]] = {}
        for spec in specs:
            generator = spec.to_generator()
            found = tuple(_candidate_set(generator.generate(s)) for s in statements)
            if any(found):
                groups.setdefault((spec.scope.model_dump_json(), found), []).append(spec)
        out: list[DuplicateGenerator] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            stats = await gather(self.store.generator_stats(s.id) for s in group)
            wins = {s.generator_id: s.wins for s in stats}
            # max() keeps the first of equals, and specs come oldest first.
            kept = max(group, key=lambda s: wins[s.id])
            out.extend(
                DuplicateGenerator(id=s.id, kept=kept.id, field=name, examples=len(statements))
                for s in group
                if s is not kept
            )
        return out

    async def _disable(self, generator_ids: list[str]) -> None:
        for gid in generator_ids:
            await self.store.set_generator_enabled(gid, False)
        if self.generators is not None and generator_ids:
            self.generators.withdraw(generator_ids)


def _failures(ctx: Context) -> dict[str, int]:
    """Failures per generator id on this document (the candidate stage's part errors)."""
    out: dict[str, int] = {}
    for error in ctx.errors.errors:
        if error.kind == "generator" and error.part is not None:
            out[error.part] = out.get(error.part, 0) + error.count
    return out


type _Key = tuple[int, int, tuple[str, ...]]


def _candidate_set(candidates: list[Candidate]) -> frozenset[_Key]:
    return frozenset(
        (c.span.start, c.span.end, tuple(step.model_dump_json() for step in c.normalise))
        for c in candidates
    )

"""Candidates and selection (spec: *Value extraction*, stages 11 and 12).

After categorising, each statement is assigned to a field (or none). Then:

- :class:`CandidateStage` runs the generator registry on statements whose field needs
  candidates (numbers, dates, strings) and stores them in ``SchemaRun.candidates``.
- :class:`SelectStage` asks Jev about every categorised statement, **one request per
  statement** covering every schema and scope that assigned it:

  - candidate fields: the :class:`~jevex.interfaces.CandidateSelector`'s questions (by
    default a Choice over the candidate spans plus "none", split across several Choices
    above 254 candidates; for ``list[...]`` fields one Noul per candidate). Picks go to
    ``SchemaRun.selections`` for the normalise stage.
  - enum fields: a Choice over the options plus "not stated" (``list[...]``: one Noul per
    option).
  - bool fields: a Noul.
  - fields with a unit: for each bare-number candidate (no ``unit`` step) in a statement
    that names another unit of the field's dimension ("Power (kW) · SE: 110" for a field
    in PS), a Choice asking which unit it's in, the field's own unit first. The answer
    becomes the candidate's ``{unit: {from: ...}}`` step, and its confidence caps the
    selection's. A statement naming no other unit asks nothing more.

  Enum and bool answers are already values, so they're recorded as
  :class:`~jevex.results.FieldMeta` (``method="jev"``, or ``"vision"`` when the statement
  came from a vision model). That happens once every answer is
  in, combining statements in document order, so the result never depends on which Jev
  reply arrived first. Fields another route already filled (e.g. structured data) are
  left alone and not asked about, except in the structured stage's ``merge`` mode, where
  :meth:`~jevex.pipeline.SchemaRun.offer_field` settles disagreements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jevex._tasks import gather
from jevex.generators import GeneratorRegistry, default_registry
from jevex.generators.units import mentioned
from jevex.interfaces import Selection
from jevex.jev import MAX_CHOICE_OPTIONS, ChoiceAnswer, JSONContent, NoulAnswer
from jevex.layout import section_text
from jevex.locales import locale_conventions, localise_steps
from jevex.normalise import canonical_unit, dimension
from jevex.pipeline import ValuePick, vision_values
from jevex.results import Alternative, FieldMeta, Source
from jevex.schema import NONE_OPTION, NOT_STATED_OPTION
from jevex.statements import NormaliserStep

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jevex.entities import EntityScope
    from jevex.interfaces import CandidateGenerator, CandidateSelector
    from jevex.jev import Answer, Choice, Question
    from jevex.locales import LocaleConventions
    from jevex.pipeline import Context, SchemaRun
    from jevex.schema import FieldSpec
    from jevex.statements import Candidate, Statement

ACCEPT_AT = 0.5
"""Noul probability at or above which a bool is True or a list member is accepted."""


ALSO_CATEGORY_P = 0.3
"""A statement whose top category is a field also goes to any other field with at least
this probability: "In stock (22 available)" states both ``in_stock`` and
``stock_count``. Such a second route can make a bool True, never False. Tuned in #49."""


def field_statements(
    ctx: Context, run: SchemaRun, scope: EntityScope, *, include_found: bool = False
) -> list[tuple[Statement, FieldSpec]]:
    """(statement, field) pairs in scope that the classifier assigned to a field.

    A statement pairs with its top category and, when that is a field, with every other
    field whose probability is at least :data:`ALSO_CATEGORY_P`, top first. Fields another
    route already found for the scope (structured data, in ``fill_gaps`` mode) are left
    out, so nothing is asked about them; in ``merge`` mode, or with ``include_found`` (the
    LLM fallback second-guesses low-confidence values), every field stays.
    """
    if ctx.parsed is None:
        return []
    out: list[tuple[Statement, FieldSpec]] = []
    names = {f.name for f in run.spec.fields}
    for statement in ctx.parsed.scope_statements(scope):
        answer = run.categories.get(statement.id)
        if answer is None or answer.choice not in names:
            # A "none" answer routes nowhere, even if a field came close: a bool field
            # would record False from a statement that isn't about it.
            continue
        also = sorted(
            (p, name)
            for name, p in answer.probabilities.items()
            if name != answer.choice and name in names and p >= ALSO_CATEGORY_P
        )
        chosen = [answer.choice, *(name for _, name in reversed(also))]
        out.extend(
            (statement, run.spec.field(name))
            for name in chosen
            if include_found or run.needs(scope.label, name)
        )
    return out


def statement_state(statement: Statement) -> JSONContent:
    """What Jev sees for one statement: its text plus the headings above it, capped by
    :func:`~jevex.layout.section_text`, and a table header's corner cell
    (``table_corner``, shortened as a heading is) and the labels on its axis
    (``table_headers``, capped by :func:`~jevex.tables.axis_text`)."""
    state: dict[str, Any] = {"statement": statement.text}
    if (ref := statement.table) is not None:
        if ref.corner:
            state["table_corner"] = section_text([ref.corner])
        if ref.axis:
            state["table_headers"] = ref.axis
    if section := section_text(statement.heading_trail):
        state["section"] = section
    return state


def unique_spans(candidates: list[Candidate]) -> dict[str, Candidate]:
    """Candidates by raw span (first wins), without a literal "none" (a reserved option)."""
    by_raw: dict[str, Candidate] = {}
    for cand in candidates:
        by_raw.setdefault(cand.raw, cand)
    by_raw.pop(NONE_OPTION, None)
    return by_raw


# --- the unit of a bare number ---------------------------------------------------------

_NUMBER_STEPS = frozenset({"parse_number", "parse_range"})


def _bare(candidate: Candidate) -> bool:
    """A number or range whose chain doesn't say what unit it's in."""
    names = {step.name for step in candidate.normalise}
    return "unit" not in names and bool(names & _NUMBER_STEPS)


def _unit_questions(
    statement: Statement, field: FieldSpec, candidates: list[Candidate]
) -> dict[str, Choice]:
    """By raw span, a Choice asking which unit each bare-number candidate is in.

    Asked only when the statement spells out a unit of the field's dimension other than
    the field's own (:func:`~jevex.generators.units.mentioned`): otherwise the number is
    read in the field's unit, as :func:`~jevex.normalise.normalise` does. Options are the
    field's unit, then the others in the order the statement names them.
    """
    if field.kind != "number" or field.unit is None or (wanted := dimension(field.unit)) is None:
        return {}
    own = canonical_unit(field.unit)
    others = [u for u in mentioned(statement.text) if u != own and dimension(u) == wanted]
    if not others:
        return {}
    spans = dict.fromkeys(c.raw for c in candidates if _bare(c))
    return {raw: field.unit_question(raw, [own, *others]) for raw in spans}


def _in_units(
    candidates: list[Candidate], units: Mapping[str, ChoiceAnswer], conventions: LocaleConventions
) -> list[Candidate]:
    """The candidates, each bare one whose span has an answer in ``units`` with that unit
    as its chain's last step (localised, so mpg is read in the locale's gallons)."""
    out: list[Candidate] = []
    for cand in candidates:
        unit = units.get(cand.raw)
        if unit is not None and _bare(cand):
            step = NormaliserStep(name="unit", args={"from": unit.choice})
            chain = [*cand.normalise, *localise_steps([step], conventions)]
            cand = cand.model_copy(update={"normalise": chain})
        out.append(cand)
    return out


# --- the default candidate selector ----------------------------------------------------


@dataclass(frozen=True)
class JevCandidateSelector:
    """Choice over candidate spans plus "none"; for list fields, one Noul per span."""

    accept_at: float = ACCEPT_AT

    def questions(
        self, statement: Statement, field: FieldSpec, candidates: list[Candidate]
    ) -> dict[str, Question]:
        spans = list(unique_spans(candidates))
        if not spans:
            return {}
        if field.many:
            return {f"member{i}": field.member_question(raw) for i, raw in enumerate(spans)}
        size = MAX_CHOICE_OPTIONS - 1  # room for "none"
        return {
            f"choice{i}": field.select_question(spans[start : start + size])
            for i, start in enumerate(range(0, len(spans), size))
        }

    def selection(
        self, field: FieldSpec, candidates: list[Candidate], answers: Mapping[str, Answer]
    ) -> Selection:
        by_raw = unique_spans(candidates)
        spans = list(by_raw)
        if field.many:
            probs: dict[str, float] = {}
            for i, raw in enumerate(spans):
                answer = answers.get(f"member{i}")
                if isinstance(answer, NoulAnswer):
                    probs[raw] = answer.p
            accepted = [by_raw[raw] for raw in spans if probs.get(raw, 0.0) >= self.accept_at]
            best = max(accepted, key=lambda c: probs[c.raw], default=None)
            confidence = (
                probs[best.raw] if best else max((1 - p for p in probs.values()), default=0.0)
            )
            return Selection(
                candidate=best,
                confidence=confidence,
                alternatives={raw: p for raw, p in probs.items() if p < self.accept_at},
                accepted=accepted,
            )
        choices = [a for key, a in answers.items() if key.startswith("choice")]
        picks = [a for a in choices if isinstance(a, ChoiceAnswer) and a.choice in by_raw]
        if not picks:
            nones = [a.confidence for a in choices if isinstance(a, ChoiceAnswer)]
            return Selection(candidate=None, confidence=min(nones, default=0.0))
        best_pick = max(picks, key=lambda a: a.confidence)
        alternatives: dict[str, float] = {}
        for a in choices:
            if isinstance(a, ChoiceAnswer):
                for raw, p in a.probabilities.items():
                    if raw not in (best_pick.choice, NONE_OPTION):
                        alternatives[raw] = max(alternatives.get(raw, 0.0), p)
        return Selection(
            candidate=by_raw[best_pick.choice],
            confidence=best_pick.confidence,
            alternatives=alternatives,
            accepted=[by_raw[best_pick.choice]],
        )


# --- stages ----------------------------------------------------------------------------


@dataclass
class CandidateStage:
    """Generates candidate spans for every categorised statement whose field needs them.

    ``registry`` holds the stage's own generators; the document's learned ones
    (``ctx.generators``) run after them. Generators are scoped by the document's locale
    (:attr:`Context.locale <jevex.pipeline.Context.locale>`, else the stage's ``locale``)
    and its :attr:`~jevex.Document.source`; locale-aware ones read numbers and dates by
    that locale too. Every generator it runs is added to ``ctx.generators_ran``. A
    generator that raises is skipped for that statement and recorded with
    :meth:`Context.part_failed <jevex.pipeline.Context.part_failed>` (the housekeeper
    quarantines a learned one that keeps failing).
    """

    registry: GeneratorRegistry = field(default_factory=default_registry)
    locale: str | None = None
    name: str = "candidates"

    async def run(self, ctx: Context) -> None:
        registry = self.registry
        if ctx.generators is not None:
            registry = ctx.generators.on(registry)
        source = ctx.document.source
        locale = ctx.locale or self.locale

        def failed(generator: CandidateGenerator, exc: Exception) -> None:
            ctx.part_failed(self.name, "generator", generator.id, exc)

        for run in ctx.active:
            counted: set[str] = set()
            for scope in run.scopes:
                for statement, spec in field_statements(ctx, run, scope):
                    key = (statement.id, spec.name)
                    if spec.needs_candidates and key not in run.candidates:
                        run.candidates[key] = registry.generate(
                            statement,
                            spec,
                            schema=run.name,
                            locale=locale,
                            source=source,
                            on_error=failed,
                        )
                        if spec.name not in counted:
                            counted.add(spec.name)
                            ctx.generators_ran.update(
                                g.id
                                for g in registry.for_field(
                                    spec, schema=run.name, locale=locale, source=source
                                )
                            )


def candidate_locale(ctx: Context) -> str | None:
    """The locale the pipeline's candidate stage is configured with, if any: what a
    document without a locale of its own is read in."""
    if ctx.pipeline is None:
        return None
    stage = next((s for s in ctx.pipeline if s.name == "candidates"), None)
    return stage.locale if isinstance(stage, CandidateStage) else None


@dataclass
class _Ask:
    """One (schema, field) about one statement: its questions and who wants the answer."""

    run: SchemaRun
    spec: FieldSpec
    questions: dict[str, Question]
    units: dict[str, Choice]
    """:func:`_unit_questions`, by raw span; keyed apart from the selector's questions."""
    scopes: list[str] = field(default_factory=list[str])

    @property
    def prefix(self) -> str:
        return f"{self.run.name}.{self.spec.name}/"

    @property
    def unit_prefix(self) -> str:
        return f"{self.run.name}.{self.spec.name}#unit"


@dataclass
class _Outcome:
    """A direct (enum/bool) answer for one statement, before statements are combined."""

    order: int
    statement: Statement
    values: list[Any]
    confidence: float
    weighed: dict[str, float]


type _Plans = dict[str, tuple[Statement, dict[tuple[str, str], _Ask]]]


@dataclass
class SelectStage:
    """Asks Jev to pick values: one batched request per statement."""

    selector: CandidateSelector = field(default_factory=JevCandidateSelector)
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        plans = self._plan(ctx)
        conventions = locale_conventions(ctx.locale or candidate_locale(ctx))
        replies = await gather(
            ctx.jev.ask(statement_state(statement), _merged(asks))
            for statement, asks in plans.values()
        )
        order = {sid: i for i, sid in enumerate(ctx.parsed.statements)} if ctx.parsed else {}
        outcomes: dict[tuple[str, str, str], list[_Outcome]] = {}
        for (sid, (statement, asks)), answers in zip(plans.items(), replies, strict=True):
            for ask in asks.values():
                mine = {
                    key.removeprefix(ask.prefix): answer
                    for key, answer in answers.items()
                    if key.startswith(ask.prefix)
                }
                units = {
                    raw: answer
                    for i, raw in enumerate(ask.units)
                    if isinstance(answer := answers.get(f"{ask.unit_prefix}{i}"), ChoiceAnswer)
                }
                self._read(
                    ask, statement, mine, units, conventions, order.get(sid, len(order)), outcomes
                )
        for (schema, scope, field_name), found in outcomes.items():
            run = ctx.schemas[schema]
            _record_direct(ctx, run, scope, run.spec.field(field_name), found)

    def _plan(self, ctx: Context) -> _Plans:
        """Per statement id: the statement and one :class:`_Ask` per (schema, field)."""
        plans: _Plans = {}
        for run in ctx.active:
            for scope in run.scopes:
                for statement, spec in field_statements(ctx, run, scope):
                    asks = plans.setdefault(statement.id, (statement, {}))[1]
                    key = (run.name, spec.name)
                    if key not in asks:
                        questions = self._questions(run, statement, spec)
                        if not questions:
                            continue
                        candidates = run.candidates.get((statement.id, spec.name), [])
                        units = _unit_questions(statement, spec, candidates)
                        asks[key] = _Ask(run=run, spec=spec, questions=questions, units=units)
                    asks[key].scopes.append(scope.label)
        return {sid: plan for sid, plan in plans.items() if plan[1]}

    def _questions(
        self, run: SchemaRun, statement: Statement, spec: FieldSpec
    ) -> dict[str, Question]:
        if spec.kind == "enum":
            if spec.many:
                return {f"member{i}": spec.member_question(o) for i, o in enumerate(spec.options)}
            return {"enum": spec.enum_question()}
        if spec.kind == "bool":
            return {"bool": spec.bool_question()}
        if spec.needs_candidates:
            return self.selector.questions(
                statement, spec, run.candidates.get((statement.id, spec.name), [])
            )
        return {}

    def _read(
        self,
        ask: _Ask,
        statement: Statement,
        answers: dict[str, Answer],
        units: dict[str, ChoiceAnswer],
        conventions: LocaleConventions,
        order: int,
        outcomes: dict[tuple[str, str, str], list[_Outcome]],
    ) -> None:
        spec, run = ask.spec, ask.run
        if spec.needs_candidates:
            key = (statement.id, spec.name)
            candidates = run.candidates.get(key, [])
            if units:
                candidates = run.candidates[key] = _in_units(candidates, units, conventions)
            selection = self.selector.selection(spec, candidates, answers)
            picks = (
                (selection.accepted or [selection.candidate])
                if spec.many
                else [selection.candidate]
            )
            read = [units[c.raw].confidence for c in picks if c is not None and c.raw in units]
            if read:
                # The value is only as sure as the units it's read in.
                confidence = min(selection.confidence, *read)
                selection = selection.model_copy(update={"confidence": confidence})
            for scope in ask.scopes:
                run.selections[(scope, spec.name, statement.id)] = selection
            return
        category = run.categories.get(statement.id)
        secondary = category is not None and category.choice != spec.name
        outcome = _direct(spec, statement, answers, order, secondary=secondary)
        if outcome is not None:
            for scope in ask.scopes:
                outcomes.setdefault((run.name, scope, spec.name), []).append(outcome)


def _merged(asks: Mapping[tuple[str, str], _Ask]) -> dict[str, Question]:
    """All of a statement's questions in one request, namespaced per schema and field."""
    out: dict[str, Question] = {}
    for ask in asks.values():
        out.update((f"{ask.prefix}{key}", q) for key, q in ask.questions.items())
        out.update((f"{ask.unit_prefix}{i}", q) for i, q in enumerate(ask.units.values()))
    return out


def _direct(
    spec: FieldSpec,
    statement: Statement,
    answers: dict[str, Answer],
    order: int,
    *,
    secondary: bool = False,
) -> _Outcome | None:
    """Read one statement's enum or bool answer. ``None`` when it states nothing.

    ``secondary``: the field isn't the statement's top category (see
    :data:`ALSO_CATEGORY_P`). Such a statement can say a bool is True but never that it
    is False: a low p there means "not about this", which would otherwise outvote the
    statement that is about it.
    """
    if spec.kind == "bool":
        answer = answers["bool"]
        assert isinstance(answer, NoulAnswer)
        value = answer.p >= ACCEPT_AT
        if secondary and not value:
            return None
        # Confidence in the stated value: p for True, 1 - p for False.
        return _Outcome(order, statement, [value], answer.p if value else 1 - answer.p, {})
    if spec.many:
        probs: dict[str, float] = {}
        for i, option in enumerate(spec.options):
            answer = answers.get(f"member{i}")
            if isinstance(answer, NoulAnswer):
                probs[option] = answer.p
        accepted = [o for o in spec.options if probs.get(o, 0.0) >= ACCEPT_AT]
        if not accepted:
            return None
        # In the order the statement mentions them; options it doesn't spell out go last.
        text = statement.text.lower()
        accepted.sort(key=lambda o: i if (i := text.find(o.lower())) >= 0 else len(text))
        weighed = {o: p for o, p in probs.items() if o not in accepted}
        return _Outcome(order, statement, accepted, max(probs[o] for o in accepted), weighed)
    choice = answers["enum"]
    assert isinstance(choice, ChoiceAnswer)
    if choice.choice == NOT_STATED_OPTION:
        return None
    weighed = {
        o: p for o, p in choice.probabilities.items() if o not in (choice.choice, NOT_STATED_OPTION)
    }
    return _Outcome(order, statement, [choice.choice], choice.confidence, weighed)


def _record_direct(
    ctx: Context, run: SchemaRun, scope: str, spec: FieldSpec, found: list[_Outcome]
) -> None:
    """Combine one field's direct answers across statements, deterministically.

    Scalars: the most confident answer, ties to the earliest statement. Lists: every
    accepted value, in document order. The entity's own statements win over those it
    shares with every entity; a value only shared ones give is marked ``shared``. Skipped
    if another route already found the field, unless routes are merged.
    """
    if not run.needs(scope, spec.name):
        return
    shared = run.shared_statements(scope)
    own = [o for o in found if o.statement.id not in shared]
    if own and not spec.many:
        found = own
    ordered = sorted(found, key=lambda o: o.order)
    values: list[Any] = []
    if spec.many:
        for outcome in ordered:
            values.extend(v for v in outcome.values if v not in values)
        best = max(ordered, key=lambda o: o.confidence)
        value: Any = values
    else:
        best = max(ordered, key=lambda o: (o.confidence, -o.order))
        values = best.values
        value = best.values[0]
    weighed: dict[str, float] = {}
    for outcome in ordered:
        for option, p in outcome.weighed.items():
            if option not in values:
                weighed[option] = max(weighed.get(option, 0.0), p)
    run.offer_field(
        scope,
        spec.name,
        FieldMeta(
            value=value,
            confidence=best.confidence,
            method="vision" if best.statement.kind == "vision" else "jev",
            source=_direct_source(ctx, best.statement),
            alternatives=[
                Alternative(value=o, raw=o, p=p)
                for o, p in sorted(weighed.items(), key=lambda kv: -kv[1])
            ],
            shared=not own,
        ),
    )
    if spec.many:
        given = [(o.statement, v, None) for o in ordered for v in o.values]
    else:
        given = [(best.statement, value, None)]
    if checks := vision_values(given):
        run.vision_values[(scope, spec.name)] = checks
        if spec.many:
            run.value_picks[(scope, spec.name)] = [
                ValuePick(
                    items=tuple(o.values),
                    method="vision" if o.statement.kind == "vision" else "jev",
                    source=_direct_source(ctx, o.statement),
                    confidence=o.confidence,
                    shared=o.statement.id in shared,
                )
                for o in sorted(ordered, key=lambda o: -o.confidence)
                if o.values
            ]


def _direct_source(ctx: Context, statement: Statement) -> Source:
    return Source(
        url=ctx.document.url,
        component_id=statement.component_id,
        statement_id=statement.id,
        statement=statement.text,
        location=statement.location,
    )

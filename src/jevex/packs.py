"""Packs: learned state as reviewable YAML (spec: *Learned state › Packs* and *Layering*).

A pack is a directory::

    manifest.yaml                    name, version, schemas, locales, disables
    generators/<id>.yaml             one generator spec each (the spec's own format)
    key_mappings/<fingerprint>.yaml  one structured-data fingerprint's key mappings
    examples/<Schema.field>.yaml     verified examples (optional)

It is shared as a git directory, or as a PyPI package (``jevex-pack-automotive-uk``)
registered under the ``jevex.packs`` entry point (:data:`ENTRY_POINT_GROUP`). The entry
point names the package that holds the pack's files, a path, or a function returning one::

    [project.entry-points."jevex.packs"]
    automotive-uk = "jevex_pack_automotive_uk"

The store is the local layer. An extractor resolves generators through the store, then
its project packs, then the community packs installed (:func:`layered_generators`): the
first layer with an id wins, and a layer can disable a lower layer's generator by id
without editing it (the store's disable list, a pack's ``manifest.disables``).

Key mappings and examples in a pack reach the extractor when the pack is imported into a
store (:func:`import_pack`, ``jevex pack import``); only generators are layered.
:func:`export_pack` writes a store's state as a pack, and :func:`diff_packs` compares two
packs (or a pack and a store) for review.
"""

from __future__ import annotations

import hashlib
import os
import re
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from jevex.generators import GeneratorSpec, InvalidGeneratorError
from jevex.generators.spec import load_yaml
from jevex.store import GeneratorRecord, KeyMapping, StoreError, VerifiedExample

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence
    from importlib.metadata import EntryPoint

    from jevex.store import Store

ENTRY_POINT_GROUP = "jevex.packs"
"""The entry point group installed packs register under."""
PACK_MANIFEST = "manifest.yaml"
PACK_GENERATORS = "generators"
"""The directory in a pack (and a pack diff) holding one ``<id>.yaml`` per generator."""
PACK_KEY_MAPPINGS = "key_mappings"
PACK_EXAMPLES = "examples"

_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
_SAFE_FILE = re.compile(_NAME)


class PackError(Exception):
    """A pack couldn't be found or read, or doesn't hold together."""


class PackManifest(BaseModel):
    """A pack's ``manifest.yaml``.

    ``schemas``: the schemas the pack is for; when given, everything in the pack must
    belong to one of them (empty: no restriction). ``locales``: the locales it was learned
    on, for people choosing a pack. ``disables``: ids of generators in lower layers this
    pack turns off (never its own).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=_NAME)
    version: str = Field(min_length=1, max_length=64)
    description: str | None = None
    schemas: list[str] = Field(default_factory=list[str])
    locales: list[str] = Field(default_factory=list[str])
    disables: list[str] = Field(default_factory=list[str])

    @field_validator("version", mode="before")
    @classmethod
    def _quoted(cls, value: object) -> object:
        # YAML reads `version: 1.10` as the number 1.1: say so rather than guess.
        if isinstance(value, int | float) and not isinstance(value, bool):
            raise ValueError(f"must be a string: quote it, as version: '{value}'")
        return value


class Pack(BaseModel):
    """A pack in memory: its manifest and contents (``path``: where it was loaded from)."""

    model_config = ConfigDict(frozen=True)

    manifest: PackManifest
    generators: list[GeneratorSpec] = Field(default_factory=list[GeneratorSpec])
    key_mappings: list[KeyMapping] = Field(default_factory=list[KeyMapping])
    examples: list[VerifiedExample] = Field(default_factory=list[VerifiedExample])
    path: Path | None = None

    @property
    def name(self) -> str:
        return self.manifest.name

    @model_validator(mode="after")
    def _consistent(self) -> Pack:
        _unique("generator id", [g.id for g in self.generators])
        _unique(
            "key mapping",
            [f"{m.fingerprint} {m.schema_name} {m.path}" for m in self.key_mappings],
        )
        _unique("example id", [e.id for e in self.examples])
        own = sorted({g.id for g in self.generators} & set(self.manifest.disables))
        if own:
            raise ValueError(f"the manifest disables the pack's own generators: {own}")
        if self.manifest.schemas:
            allowed = set(self.manifest.schemas)
            used = (
                [g.schema_name for g in self.generators]
                + [m.schema_name for m in self.key_mappings]
                + [_schema_of(e.field) for e in self.examples]
            )
            others = sorted(set(used) - allowed)
            if others:
                raise ValueError(f"holds state for schemas the manifest doesn't list: {others}")
        return self

    @classmethod
    def load(cls, directory: Path) -> Pack:
        """Read a pack directory. Raises :class:`PackError`, naming the file, for anything
        missing or invalid."""
        if not directory.is_dir():
            raise PackError(f"no such pack directory: {directory}")
        manifest_path = directory / PACK_MANIFEST
        if not manifest_path.is_file():
            raise PackError(f"{directory} has no {PACK_MANIFEST}")
        manifest = _read(manifest_path, PackManifest)
        try:
            generators = pack_generators(directory)
        except InvalidGeneratorError as exc:
            raise PackError(str(exc)) from None
        mappings: list[KeyMapping] = []
        for path in _files(directory / PACK_KEY_MAPPINGS):
            file = _read(path, _MappingFile)
            mappings += [
                KeyMapping(fingerprint=file.fingerprint, **m.model_dump()) for m in file.mappings
            ]
        examples: list[VerifiedExample] = []
        for path in _files(directory / PACK_EXAMPLES):
            file = _read(path, _ExampleFile)
            examples += [VerifiedExample(field=file.field, **e.model_dump()) for e in file.examples]
        try:
            return cls(
                manifest=manifest,
                generators=generators,
                key_mappings=mappings,
                examples=examples,
                path=directory,
            )
        except ValidationError as exc:
            raise PackError(f"{directory}: {_errors(exc)}") from None

    def write(self, directory: Path) -> list[Path]:
        """Write the pack to ``directory`` and return the files written.

        ``directory`` must not exist or be empty, so no file from an earlier version
        lingers (:class:`FileExistsError` otherwise). Files and their entries are sorted,
        so exporting the same state twice writes the same bytes.
        """
        _check_empty(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest.model_dump(mode="json", exclude_none=True)
        manifest["disables"] = sorted(set(self.manifest.disables))
        written = [_write(directory / PACK_MANIFEST, manifest)]
        for spec in sorted(self.generators, key=lambda g: g.id):
            path = directory / PACK_GENERATORS / f"{spec.id}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(spec.to_yaml(), encoding="utf-8")
            written.append(path)
        by_fingerprint: dict[str, list[KeyMapping]] = {}
        for m in self.key_mappings:
            by_fingerprint.setdefault(m.fingerprint, []).append(m)
        for fingerprint in sorted(by_fingerprint):
            ordered = sorted(by_fingerprint[fingerprint], key=lambda m: (m.schema_name, m.path))
            data = {
                "fingerprint": fingerprint,
                "mappings": [_MappingEntry.of(m).dump() for m in ordered],
            }
            written.append(_write(directory / PACK_KEY_MAPPINGS / _file_name(fingerprint), data))
        by_field: dict[str, list[VerifiedExample]] = {}
        for e in self.examples:
            by_field.setdefault(e.field, []).append(e)
        for name in sorted(by_field):
            ordered = sorted(by_field[name], key=lambda e: e.id)
            data = {"field": name, "examples": [_ExampleEntry.of(e).dump() for e in ordered]}
            written.append(_write(directory / PACK_EXAMPLES / _file_name(name), data))
        return written


# --- file formats ------------------------------------------------------------------------


class _MappingEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_name: str = Field(alias="schema")
    path: str
    field: str | None
    normalisers: list[Any] = Field(default_factory=list[Any])
    unsure: bool = False

    @classmethod
    def of(cls, m: KeyMapping) -> _MappingEntry:
        return cls(
            schema=m.schema_name,
            path=m.path,
            field=m.field,
            normalisers=m.normalisers,
            unsure=m.unsure,
        )

    def dump(self) -> dict[str, Any]:
        # ``field`` has no default, so a "none" answer is always written out.
        return self.model_dump(mode="json", by_alias=True, exclude_defaults=True)


class _MappingFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint: str = Field(min_length=1)
    mappings: list[_MappingEntry]


class _ExampleEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    statement: str
    value: Any
    evidence: tuple[int, int] | None = None
    context: dict[str, Any] = Field(default_factory=dict[str, Any])
    source: Literal["llm", "human"] = "llm"
    probability: float | None = None

    @classmethod
    def of(cls, e: VerifiedExample) -> _ExampleEntry:
        return cls.model_validate(e.model_dump(exclude={"field", "created_at"}))

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_defaults=True)


class _ExampleFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    examples: list[_ExampleEntry]


def _read[M: BaseModel](path: Path, model: type[M]) -> M:
    try:
        data = load_yaml(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PackError(f"{path} isn't valid YAML: {exc}") from None
    except RecursionError:
        raise PackError(f"{path} is nested too deeply") from None
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise PackError(f"{path}: {_errors(exc)}") from None


def _write(path: Path, data: object) -> Path:
    """Write ``data`` as YAML, as :meth:`GeneratorSpec.to_yaml` does (Unicode where that
    round-trips, escaped otherwise)."""
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    if load_yaml(text) != data:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False, width=100)
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.yaml"))


def _file_name(key: str) -> str:
    """``<key>.yaml``, or a hash of it when ``key`` isn't safe as a file name."""
    if _SAFE_FILE.match(key):
        return f"{key}.yaml"
    return f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.yaml"


def _check_empty(directory: Path) -> None:
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"{directory} already exists and isn't an empty directory")


def _errors(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(parts)


def _unique(what: str, keys: list[str]) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for k in keys:
        (dupes if k in seen else seen).add(k)
    if dupes:
        raise ValueError(f"duplicate {what}: {sorted(dupes)}")


def _schema_of(field: str) -> str:
    return field.split(".", 1)[0]


def pack_generators(directory: Path) -> list[GeneratorSpec]:
    """The generators in a pack directory: every ``generators/*.yaml``, by file name.

    Needs no manifest, so it reads pack diffs (``jevex learn``'s output) too. A directory
    without ``generators`` has none. Raises :class:`FileNotFoundError` when ``directory``
    doesn't exist, and :class:`InvalidGeneratorError` (naming the file) for a spec that
    doesn't validate or an id two files share.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"no such pack directory: {directory}")
    specs: dict[str, GeneratorSpec] = {}
    for path in _files(directory / PACK_GENERATORS):
        try:
            spec = GeneratorSpec.from_yaml(path.read_text(encoding="utf-8"))
        except InvalidGeneratorError as exc:
            raise InvalidGeneratorError(f"{path}: {exc}") from None
        if spec.id in specs:
            raise InvalidGeneratorError(f"{path}: another file already has id {spec.id!r}")
        specs[spec.id] = spec
    return list(specs.values())


# --- installed packs ---------------------------------------------------------------------


def community_packs(names: Sequence[str] | None = None) -> list[Pack]:
    """Installed packs (``jevex.packs`` entry points): every one by entry point name, or
    the ``names`` given, in that order. Raises :class:`PackError` for a name that isn't
    installed or a pack that doesn't load."""
    installed = {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}
    if names is None:
        names = sorted(installed)
    packs: list[Pack] = []
    for name in names:
        if name not in installed:
            known = ", ".join(sorted(installed)) or "none"
            raise PackError(f"no installed pack named {name!r} (installed: {known})")
        packs.append(Pack.load(_entry_point_directory(installed[name])))
    return packs


def _entry_point_directory(ep: EntryPoint) -> Path:
    try:
        target: object = ep.load()
        if callable(target) and not isinstance(target, ModuleType):
            target = target()
    except Exception as exc:  # a third-party package: whatever it raises, name the pack
        raise PackError(f"installed pack {ep.name!r} ({ep.value}) failed to load: {exc}") from exc
    if isinstance(target, ModuleType):
        paths: list[str] = list(getattr(target, "__path__", []))
        if not paths:
            raise PackError(
                f"installed pack {ep.name!r} names module {target.__name__!r}, which isn't a "
                "package directory"
            )
        return Path(paths[0])
    if isinstance(target, str):
        return Path(target)
    if isinstance(target, os.PathLike):
        return Path(os.fspath(cast("os.PathLike[str]", target)))
    raise PackError(
        f"installed pack {ep.name!r} ({ep.value}) is a {type(target).__name__}, not a "
        "package, a path or a function returning a path"
    )


def load_pack(source: Pack | str | Path) -> Pack:
    """A pack from a directory, or an installed pack's name (a ``str`` that isn't an
    existing directory and has no path separator). A :class:`Pack` is returned as is."""
    if isinstance(source, Pack):
        return source
    path = Path(source)
    if isinstance(source, Path) or path.is_dir() or "/" in source or os.sep in source:
        return Pack.load(path)
    return community_packs([source])[0]


# --- layering ----------------------------------------------------------------------------


def layered_generators(
    local: Iterable[GeneratorSpec], disabled: Collection[str], packs: Sequence[Pack]
) -> list[GeneratorSpec]:
    """The generators in use, through the layers in order: ``local`` (the store's enabled
    generators), then each of ``packs`` (project packs, then community packs).

    The first layer with an id wins: a lower layer's generator with the same id is
    dropped. ``disabled`` (the store's disable list) turns off every pack's generators with
    those ids, and each pack's ``manifest.disables`` turns off those of the packs after it.
    """
    out = list(local)
    seen = {spec.id for spec in out}
    off = set(disabled)
    for pack in packs:
        for spec in pack.generators:
            if spec.id not in seen and spec.id not in off:
                out.append(spec)
            seen.add(spec.id)
        off.update(pack.manifest.disables)
    return out


# --- stores ------------------------------------------------------------------------------


def generator_record(spec: GeneratorSpec) -> GeneratorRecord:
    """The store record for ``spec`` (enabled, with its locale as the store scope)."""
    scope = {"locale": spec.scope.locale} if spec.scope.locale else {}
    return GeneratorRecord(id=spec.id, field=spec.field, spec=spec.to_data(), scope=scope)


async def stored_generators(store: Store) -> list[GeneratorSpec]:
    """The store's enabled generator specs, oldest first.

    Raises :class:`~jevex.store.StoreError` for a stored spec that doesn't validate.
    """
    specs: list[GeneratorSpec] = []
    for record in await store.generators():
        try:
            specs.append(GeneratorSpec.parse(record.spec))
        except InvalidGeneratorError as exc:
            raise StoreError(f"stored generator {record.id!r} is invalid: {exc}") from exc
    return specs


async def export_pack(
    store: Store,
    name: str,
    version: str,
    *,
    schemas: Sequence[str] = (),
    locales: Sequence[str] = (),
    description: str | None = None,
    examples: bool = False,
) -> Pack:
    """The store's learned state as a pack: its enabled generators, key mappings, its
    disable list as ``disables``, and with ``examples=True`` its verified examples.

    With ``schemas``, only those schemas' state; the manifest lists them, or else the
    schemas exported. ``locales`` default to those the generators are scoped to. Raises
    :class:`~jevex.store.StoreError` for a stored generator that doesn't validate.
    """
    wanted = set(schemas)

    def keep(schema: str) -> bool:
        return not wanted or schema in wanted

    generators = [g for g in await stored_generators(store) if keep(g.schema_name)]
    mappings = [m for m in await store.key_mappings() if keep(m.schema_name)]
    found = [e for e in await store.examples() if keep(_schema_of(e.field))] if examples else []
    used = {g.schema_name for g in generators} | {m.schema_name for m in mappings}
    used |= {_schema_of(e.field) for e in found}
    manifest = PackManifest(
        name=name,
        version=version,
        description=description,
        schemas=list(schemas) or sorted(used),
        locales=list(locales) or sorted({g.scope.locale for g in generators if g.scope.locale}),
        disables=sorted(await store.disabled_generator_ids()),
    )
    return Pack(manifest=manifest, generators=generators, key_mappings=mappings, examples=found)


async def import_pack(pack: Pack, store: Store, *, examples: bool = True) -> None:
    """Copy ``pack`` into the store, the local layer, so it overrides every pack.

    Generators replace any with the same id and are enabled (even one the store had
    disabled); key mappings replace those on the same fingerprint, schema and path; the
    manifest's ``disables`` join the store's disable list. With ``examples``, the pack's
    verified examples are stored too (replacing by id).
    """
    for spec in pack.generators:
        await store.put_generator(generator_record(spec))
    for mapping in pack.key_mappings:
        await store.put_key_mapping(mapping)
    if examples:
        for example in pack.examples:
            await store.add_example(example)
    for generator_id in pack.manifest.disables:
        await store.set_generator_enabled(generator_id, False)


# --- diffs -------------------------------------------------------------------------------


class Changes[T](BaseModel):
    """What differs in one kind of entry: ``changed`` holds the new versions."""

    model_config = ConfigDict(frozen=True)

    added: list[T] = []  # pydantic copies mutable defaults
    removed: list[T] = []
    changed: list[T] = []

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


class PackChanges(BaseModel):
    """What going from one pack to another changes (:func:`diff_packs`).

    ``manifest``: each changed manifest key (other than ``disables``) with its old and
    new value. Entries count as changed when anything but their creation time differs.
    """

    model_config = ConfigDict(frozen=True)

    manifest: dict[str, tuple[Any, Any]] = Field(default_factory=dict[str, tuple[Any, Any]])
    generators: Changes[GeneratorSpec] = Field(default_factory=Changes[GeneratorSpec])
    key_mappings: Changes[KeyMapping] = Field(default_factory=Changes[KeyMapping])
    examples: Changes[VerifiedExample] = Field(default_factory=Changes[VerifiedExample])
    disables: Changes[str] = Field(default_factory=Changes[str])

    @property
    def empty(self) -> bool:
        return not self.manifest and all(
            c.empty for c in (self.generators, self.key_mappings, self.examples, self.disables)
        )


def diff_packs(
    old: Pack, new: Pack, *, manifest: bool = True, examples: bool = True
) -> PackChanges:
    """What changes from ``old`` to ``new``. ``manifest=False`` skips the manifest's own
    keys (when one side is a store's export, say); ``examples=False`` skips examples.
    """
    meta: dict[str, tuple[Any, Any]] = {}
    if manifest:
        before = old.manifest.model_dump(exclude={"disables"})
        after = new.manifest.model_dump(exclude={"disables"})
        meta = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    return PackChanges(
        manifest=meta,
        generators=_changes(old.generators, new.generators, lambda g: g.id, _spec_data),
        key_mappings=_changes(
            old.key_mappings,
            new.key_mappings,
            lambda m: f"{m.fingerprint}\n{m.schema_name}\n{m.path}",
            lambda m: m.model_dump(mode="json", exclude={"created_at"}),
        ),
        examples=(
            _changes(
                old.examples,
                new.examples,
                lambda e: e.id,
                lambda e: e.model_dump(mode="json", exclude={"created_at"}),
            )
            if examples
            else Changes[VerifiedExample]()
        ),
        disables=Changes[str](
            added=sorted(set(new.manifest.disables) - set(old.manifest.disables)),
            removed=sorted(set(old.manifest.disables) - set(new.manifest.disables)),
        ),
    )


def _spec_data(spec: GeneratorSpec) -> object:
    return spec.to_data()


def _changes[T](
    old: Sequence[T], new: Sequence[T], key: Callable[[T], str], content: Callable[[T], object]
) -> Changes[T]:
    before = {key(x): x for x in old}
    after = {key(x): x for x in new}
    both = sorted(after.keys() & before.keys())
    return Changes[T](
        added=[after[k] for k in sorted(after.keys() - before.keys())],
        removed=[before[k] for k in sorted(before.keys() - after.keys())],
        changed=[after[k] for k in both if content(before[k]) != content(after[k])],
    )

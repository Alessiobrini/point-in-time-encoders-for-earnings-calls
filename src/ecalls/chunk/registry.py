"""The embedding registry: resolve a date to the model that may legitimately embed it.

Every embedding set this project produces is registered here with its training cutoff, and callers
resolve a model **by date or by name, never by handing in a path**. That is the whole point: a
backtest at date `t` must use a model that saw no text after `t`, and the easiest way to get that
wrong is to load a directory because it happens to be there.

Three refusals are built in, because each corresponds to a way the point-in-time discipline fails
quietly rather than loudly:

- `for_date` raises when no registered vintage has a cutoff at or before the date, instead of falling
  back to the earliest one.
- `for_date` never returns an **atemporal** model. A model whose training cutoff is unknown cannot
  support a predictive claim, and several strong finance embedding models are in that position because
  they are fine-tuned on synthetic data (Fin-E5) or do not report a calendar range (BAM, FinBERT2).
  Those are registered as `atemporal` so they stay available for retrieval and teaching, and barred
  from prediction.
- Resolving a model whose embeddings are not on this machine raises with the path it looked for,
  rather than returning an entry that cannot be loaded.

The registry file is tracked in git and the embeddings are not, so the contract travels with the code
and the data stays local.
"""
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ecalls.config import paths as P

REGISTRY_PATH = P.PROJECT_ROOT / "registry" / "embeddings.json"

VINTAGE = "vintage"        # has a known training cutoff, usable for predictive work
ATEMPORAL = "atemporal"    # cutoff unknown or not applicable, retrieval only


def as_date(value) -> dt.date:
    """A plain date from a date, a datetime, a string, or anything pandas-like.

    `for_date` compares against `dt.date`, and a `datetime` raises `TypeError` on that comparison
    rather than coercing. Parquet stores a call date as a datetime, so every caller reading one out
    of an artifact hit this. Normalising here rather than at each call site, because the point of
    this module is that there is one place where a date becomes a model.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    for attr in ("date", "to_pydatetime"):      # pandas Timestamp, numpy datetime64 via pandas
        f = getattr(value, attr, None)
        if callable(f):
            got = f()
            return got.date() if isinstance(got, dt.datetime) else got
    raise TypeError(
        f"cannot read a calendar date from {value!r} of type {type(value).__name__}. Pass a date, "
        "a datetime, or an ISO string.")


@dataclass
class Entry:
    model_id: str
    kind: str                       # VINTAGE or ATEMPORAL
    source: str                     # e.g. "huggingface:manelalab/chrono-bert-v1-20151231"
    dim: int
    training_cutoff: str | None = None    # ISO date; required for VINTAGE
    window_start: str | None = None
    pooling: str = "mean"
    normalized: bool = True
    quantization: str | None = None
    chunker_version: str | None = None
    notes: str = ""
    registered_at: str = ""
    git_sha: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in (VINTAGE, ATEMPORAL):
            raise ValueError(f"kind must be {VINTAGE!r} or {ATEMPORAL!r}, got {self.kind!r}")
        if self.kind == VINTAGE and not self.training_cutoff:
            raise ValueError(
                f"{self.model_id}: a vintage entry needs a training_cutoff. Register it as "
                f"{ATEMPORAL!r} if the cutoff is unknown, which bars it from predictive use.")
        if self.training_cutoff:
            dt.date.fromisoformat(self.training_cutoff)   # validate early, not at query time

    @property
    def cutoff(self) -> dt.date | None:
        return dt.date.fromisoformat(self.training_cutoff) if self.training_cutoff else None

    @property
    def embeddings_dir(self) -> Path:
        return P.EMBEDDINGS_DIR / self.model_id

    def require_on_disk(self) -> Path:
        d = self.embeddings_dir
        if not d.exists():
            raise FileNotFoundError(
                f"{self.model_id} is registered but its embeddings are not on this machine. "
                f"Expected {d}. Build them, or copy the directory from wherever the corpus lives.")
        return d


class Registry:
    def __init__(self, entries: list[Entry]):
        seen = set()
        for e in entries:
            if e.model_id in seen:
                raise ValueError(f"duplicate model_id in registry: {e.model_id}")
            seen.add(e.model_id)
        self._entries = list(entries)

    # --- reading -------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> list[Entry]:
        return list(self._entries)

    def vintages(self) -> list[Entry]:
        """Vintage entries, oldest cutoff first."""
        return sorted((e for e in self._entries if e.kind == VINTAGE), key=lambda e: e.cutoff)

    def by_name(self, model_id: str) -> Entry:
        """Any entry, including atemporal ones. Use for retrieval, exploration and teaching."""
        for e in self._entries:
            if e.model_id == model_id:
                return e
        raise KeyError(f"{model_id!r} is not registered. Known: "
                       f"{sorted(x.model_id for x in self._entries)}")

    def for_date(self, date, family: str | None = None) -> Entry:
        """The vintage that may legitimately embed text as of `date`: the largest cutoff at or
        before it. Raises rather than falling back, and never returns an atemporal model.

        `family` is a `model_id` prefix and is required once two vintages share the winning cutoff.
        That became the normal case on 2026-10-01: this project's own fifteen retrievers have cutoffs
        2010 to 2024, exactly matching fifteen ChronoBERT checkpoints. Before then every cutoff had
        one model and the resolver could return the last eligible entry, which with ties means
        whichever sorted last.

        Phase 5's whole design is our vintage against ChronoBERT's checkpoint of the same year, so a
        resolver that silently picks one of them would decide the comparison by sort order. Refusing
        is consistent with the other three refusals above: each exists because the point-in-time
        discipline otherwise fails quietly instead of loudly.
        """
        date = as_date(date)
        vs = self.vintages()
        if family:
            vs = [e for e in vs if e.model_id.startswith(family)]
            if not vs:
                raise LookupError(
                    f"no vintage's model_id starts with {family!r}. Registered families: "
                    f"{sorted({e.model_id.rsplit('-', 1)[0] for e in self.vintages()})}")
        if not vs:
            raise LookupError(
                "no vintage models are registered, only atemporal ones, so no date can be resolved "
                "for predictive use. Atemporal models are reachable with by_name for retrieval.")
        eligible = [e for e in vs if e.cutoff <= date]
        if not eligible:
            raise LookupError(
                f"no registered vintage has a cutoff at or before {date}. The earliest is "
                f"{vs[0].model_id} at {vs[0].cutoff}. Falling back to it would embed {date} with a "
                "model trained on later text, which is the leakage this resolver exists to prevent.")
        best = eligible[-1].cutoff
        tied = [e for e in eligible if e.cutoff == best]
        if len(tied) > 1:
            raise LookupError(
                f"{len(tied)} vintages share the cutoff {best}, which is the newest at or before "
                f"{date}: {sorted(e.model_id for e in tied)}. Pass family= to choose, for example "
                f"family={tied[0].model_id.rsplit('-', 1)[0]!r}. Returning one of them by sort order "
                "would let the registry decide a comparison that is supposed to be between them.")
        return tied[0]

    def require_point_in_time(self, model_id: str, first_use) -> Entry:
        """The entry for `model_id`, if it may embed text used from `first_use` on, else raise.

        The benchmarks and backtests pick their point-in-time vintage by name, through a rule such as
        "vintage t-1 for test year t", rather than through `for_date`. This is the check that the name
        the rule produced is a vintage whose cutoff falls before the first day it is used. The leaky and
        same-year arms are not point in time by design and do not call it.
        """
        e = self.by_name(model_id)
        first_use = as_date(first_use)
        if e.kind != VINTAGE:
            raise LookupError(f"{model_id} is {e.kind}, so it has no cutoff and cannot be point in time.")
        if not e.cutoff < first_use:
            raise LookupError(
                f"{model_id} has cutoff {e.cutoff}, not before {first_use}, the first day it would "
                "embed text as the vintage available at the time.")
        return e

    def families(self) -> list[str]:
        """The registered vintage families, which are `model_id` prefixes up to the dated suffix."""
        return sorted({e.model_id.rsplit("-", 1)[0] for e in self.vintages()})

    def path_for_date(self, date, family: str | None = None) -> Path:
        return self.for_date(date, family).require_on_disk()

    def path_by_name(self, model_id: str) -> Path:
        return self.by_name(model_id).require_on_disk()

    # --- writing -------------------------------------------------------------------------------
    def register(self, entry: Entry, replace: bool = False) -> "Registry":
        if any(e.model_id == entry.model_id for e in self._entries):
            if not replace:
                raise ValueError(f"{entry.model_id} is already registered; pass replace=True")
            self._entries = [e for e in self._entries if e.model_id != entry.model_id]
        self._entries.append(entry)
        return self

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in sorted(self._entries, key=lambda e: e.model_id)],
                          indent=2)

    def save(self, path: Path | None = None) -> Path:
        path = path or REGISTRY_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.to_json() + "\n")
        tmp.replace(path)
        return path


_CACHED: dict = {}


def require_point_in_time(model_id: str, first_use) -> Entry:
    """`Registry.require_point_in_time` on the registry file, read once per process."""
    if "r" not in _CACHED:
        _CACHED["r"] = load()
    return _CACHED["r"].require_point_in_time(model_id, first_use)


def load(path: Path | None = None) -> Registry:
    path = path or REGISTRY_PATH
    if not path.exists():
        return Registry([])
    return Registry([Entry(**d) for d in json.loads(path.read_text())])

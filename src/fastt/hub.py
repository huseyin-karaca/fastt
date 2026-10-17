"""The Hub layer: one dataset repository for the data, one for the run records, either of them a local folder.

`FasttHub` reads and writes configs (the audio bases, the experts' pooled features and WER counters) in the data
repository, and files under `results/` and `studies/` in the records repository (`fastt.core.RECORDS_REPO`). A
repository written `local:/path` is a folder with the same layout, which is how a run works without a Hub account.
`set_read_only(True)` turns every write into `HubReadOnly`; the caches that mirror to the Hub check `read_only()`
and keep their files local instead.
"""

__all__ = ['FasttHub', 'HubReadOnly', 'set_read_only', 'read_only', 'is_local']

import hashlib
import json
import re
import shutil
from pathlib import Path

import datasets
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

# Read this repo's feature types from its own dataset card, not from the Hub
# dataset-viewer's server-side re-export. The viewer runs `datasets` 4.x, which
# serialises a fixed-size list column as `{"_type": "List"}` — a type name
# `datasets<4.0` has never heard of, and one failure there raises for EVERY
# config in the repo. See the note above; this is the whole fix.
datasets.config.USE_PARQUET_EXPORT = False

import fastt.core as core
from fastt.core import active_dataset


class HubReadOnly(PermissionError):
    """A write was attempted while the Hub layer is read-only (`set_read_only(True)`)."""


_READ_ONLY = False


def set_read_only(flag=True):
    """Refuse every write to the Hub for the rest of the session (the reproduction notebooks do)."""
    global _READ_ONLY
    _READ_ONLY = bool(flag)


def read_only():
    return _READ_ONLY


def is_local(repo):
    return str(repo).startswith("local:")


def _root(repo):
    return Path(str(repo)[len("local:"):]).expanduser()


def _check_writable(what):
    if _READ_ONLY:
        raise HubReadOnly(f"read-only: not writing {what}")


class FasttHub:
    """Read and write the configs of one Hugging Face dataset repo.

    Parameters
    ----------
    repo_id : the dataset repo. Defaults to the active spec's repo, which is
        the global `REPO_ID` unless the spec overrides it.
    spec : the `DatasetSpec` whose artefact naming this instance uses. Defaults
        to the active dataset (§1.5).
    stream : read columns over HTTP range requests instead of downloading the
        parquet shards first. `False` — download — is the default because it is
        the faster of the two for a whole-store read; see the note above for
        when to flip it.

    Reading is anonymous for a public repo; writing needs `login_hf()` first.
    """

    def __init__(self, repo_id=None, spec=None, stream=False, records=None):
        self.spec = active_dataset(spec)
        self.repo_id = self.spec.hub_repo(repo_id)
        # The records go to RECORDS_REPO when the data come from the default repository; a repository named
        # explicitly (a test double, a spec that overrides it) holds its own records.
        self.records_id = records or (core.RECORDS_REPO if self.repo_id == core.REPO_ID else self.repo_id)
        self.api = HfApi()
        self.stream = bool(stream)
        self._configs = self._files = None      # cached; invalidated by push()
        self._filesystem = None                 # created lazily by read_columns()
        self._revision = None                   # resolved lazily by revision()

    def __repr__(self):
        mode = "stream" if self.stream else "download"
        rec = "" if self.records_id == self.repo_id else f", records={self.records_id!r}"
        return f"FasttHub({self.repo_id!r}{rec}, dataset={self.spec.name!r}, {mode})"

    def repo_for(self, path):
        """The repository a repo-relative path lives in: the records repository for `results/` and `studies/`."""
        return self.records_id if str(path).startswith(core.RECORD_ROOTS) else self.repo_id

    @staticmethod
    def _listing(repo, api):
        if is_local(repo):
            root = _root(repo)
            return sorted(str(f.relative_to(root)) for f in root.rglob("*")
                          if f.is_file() and not f.name.startswith(".")) if root.is_dir() else []
        return api.list_repo_files(repo, repo_type="dataset")

    def revision(self):
        """The repo's current commit sha, or `None` if the Hub is unreachable.

        This is the cache key everything downstream hangs off: re-pushing a
        config changes the sha, so a stale local cache is invalidated instead of
        being served silently. `None` means "could not verify", and callers are
        expected to treat that as "do not write a new cache entry" rather than
        as an error — an offline session should still be able to read a cache it
        already has.
        """
        if self._revision is None:
            if is_local(self.repo_id):
                root = _root(self.repo_id)
                stamp = [(f, (root / f).stat().st_size, (root / f).stat().st_mtime_ns)
                         for f in self._listing(self.repo_id, self.api)]
                self._revision = "local-" + hashlib.sha1(json.dumps(stamp).encode()).hexdigest()[:12]
                return self._revision
            try:
                self._revision = self.api.dataset_info(self.repo_id).sha
            except Exception:                              # noqa: BLE001 — offline
                return None
        return self._revision

    # ------------------------------------------------------------ discovery --

    def files(self, refresh=False):
        """Every file path in the repo. Cached after the first call."""
        if refresh or self._files is None:
            data = self._listing(self.repo_id, self.api)
            if self.records_id == self.repo_id:
                self._files = data
            else:
                self._files = ([f for f in data if not f.startswith(core.RECORD_ROOTS)]
                               + [f for f in self._listing(self.records_id, self.api)
                                  if f.startswith(core.RECORD_ROOTS)])
        return self._files

    def configs(self, refresh=False):
        """Every config name in the repo, sorted. Cached after the first call.

        Derived from the repo's directory layout — each config's parquet files
        live under `<config>/<split>-*.parquet` — and deliberately NOT from
        `datasets.get_dataset_config_names`. That function parses the repo's
        stored feature metadata, so a single config pushed by a newer `datasets`
        (which writes feature types that `datasets<4.0` does not know, such as
        `List`) makes it raise for the WHOLE repo, including configs that would
        otherwise read fine. Directory names are just strings.
        """
        if refresh or self._configs is None:
            names = set()
            for f in self.files(refresh):
                head, _, tail = f.partition("/")
                if tail.endswith(".parquet"):
                    # `push_to_hub` stores the unnamed default config under `data/`.
                    names.add("default" if head == "data" else head)
            self._configs = sorted(names)
        return self._configs

    def model_names(self):
        """Short names of every model with a config for THIS dataset.

        This is what makes §5 self-updating: the leaderboard iterates over
        whatever is actually on the Hub instead of a hardcoded list that goes
        stale the moment a model is added.

        Filtering on the spec's prefix is also what keeps two corpora in one
        repo apart. `model_whisper_large_v3` and `voxpopuli_model_whisper_large_v3`
        both exist there, and only one of them belongs to a LibriSpeech table.
        """
        prefix = self.spec.model_prefix
        return [c[len(prefix):] for c in self.configs() if c.startswith(prefix)]

    def has(self, config_name):
        """True if `config_name` already exists in the repo."""
        return config_name in self.configs()

    # ---------------------------------------------------------------- read --

    def split_names(self, config_name):
        """The splits `config_name` actually has files for, from the layout alone.

        Like `configs()`, this reads directory entries rather than the repo's
        feature metadata, so it answers even for a config whose card entry is
        stale — which is exactly the case `load(splits=...)` has to survive.
        """
        pattern = re.compile(rf"^{re.escape(config_name)}/(.+)-\d+-of-\d+\.parquet$")
        return sorted({m.group(1) for m in map(pattern.match, self.files()) if m})

    def load(self, config_name, splits=None):
        """Load one config as a `DatasetDict`.

        Falls back to reading the parquet files directly when the config's own
        metadata is unreadable — see the note above this class. A config that
        does not exist at all raises immediately instead, so a typo does not
        turn into a confusing parquet-glob failure.

        Naming a strict subset of the config's splits is taken as permission to
        **restrict the download** rather than fetch everything and discard the
        rest — a large corpus's `base` can be ~13 GB, ~10 GB of it `train`, so
        `splits=["train"]` fetching the other two would cost 3 GB for nothing.
        It is an optimisation and it is treated as one: if the raw-parquet route
        is unavailable, every split is loaded and the subset taken afterwards,
        exactly as before. The return value is the same either way.
        """
        if config_name not in self.configs():
            raise KeyError(f"no config {config_name!r} in {self.repo_id}; "
                           f"have {self.configs()}")
        wanted = list(splits) if splits else None
        ds = None

        if is_local(self.repo_id):
            root = _root(self.repo_id)
            ds = load_dataset("parquet", data_files={
                s: str(root / config_name / f"{s}-*.parquet")
                for s in (wanted or self.split_names(config_name))})
            return DatasetDict({s: ds[s] for s in wanted}) if wanted else ds

        if wanted is not None and set(wanted) < set(self.split_names(config_name)):
            # Feature types come from the footers `push_to_hub` wrote, so
            # fixed-size lists survive; `ensure_audio_feature` repairs the audio
            # column in the case where they do not.
            try:
                ds = load_dataset("parquet", data_files={
                    s: f"hf://datasets/{self.repo_id}/{config_name}/{s}-*.parquet"
                    for s in wanted})
            except Exception as e:                         # noqa: BLE001
                print(f"  note: reading {config_name}/{'+'.join(wanted)} directly "
                      f"failed ({type(e).__name__}: {e}); loading every split "
                      "instead. Same rows, more download.")

        if ds is None:
            try:
                ds = load_dataset(self.repo_id, config_name)
            except Exception as e:                         # noqa: BLE001
                print(f"  note: load_dataset({config_name!r}) failed ({type(e).__name__}); "
                      "falling back to the raw parquet files.")
                ds = load_dataset("parquet", data_files={
                    s: f"hf://datasets/{self.repo_id}/{config_name}/{s}*.parquet"
                    for s in (wanted or self.spec.splits)
                })
        return DatasetDict({s: ds[s] for s in wanted}) if wanted else ds

    def parquet_paths(self, config_name, split):
        """The repo-relative parquet shards backing one config/split, sorted."""
        prefix = f"{config_name}/{split}-"
        paths = sorted(f for f in self.files()
                       if f.startswith(prefix) and f.endswith(".parquet"))
        if not paths:
            raise FileNotFoundError(
                f"no parquet for {config_name}/{split} in {self.repo_id}")
        return paths

    def column_names(self, config_name, split, stream=None):
        """The column names of one config/split, read from the parquet footer.

        Never parses the repo's feature metadata — unlike
        `datasets.get_dataset_config_info` — so it works regardless of which
        `datasets` version wrote the config. Streaming makes it one small range
        request; downloading fetches the first shard, which `read_columns` is
        about to want anyway.
        """
        path = self.parquet_paths(config_name, split)[0]
        if self.streaming(stream):
            with self._fs().open(f"datasets/{self.repo_id}/{path}") as f:
                return [field.name for field in pq.ParquetFile(f).schema_arrow]
        return [f.name for f in pq.ParquetFile(self.local_path(path)).schema_arrow]

    def read_columns(self, config_name, split, columns, stream=None):
        """Read named columns of one config/split as a `pyarrow.Table`.

        `stream=None` follows the instance's own policy; pass a bool to override
        it for this call. Either way parquet stores each column separately, so
        only the named columns are ever materialised in memory — the difference
        is whether the bytes come off local disk or off the wire.
        """
        paths, fs = self._shards(config_name, split, stream)
        return pq.read_table(paths, columns=list(columns), filesystem=fs)

    def streaming(self, stream=None):
        """This call's read policy: the explicit argument, else the instance's. A local folder never streams."""
        if is_local(self.repo_id):
            return False
        return self.stream if stream is None else bool(stream)

    def _shards(self, config_name, split, stream=None):
        """`(paths, filesystem)` for `pq.read_table` — local files, or Hub URIs."""
        paths = self.parquet_paths(config_name, split)
        if self.streaming(stream):
            return [f"datasets/{self.repo_id}/{p}" for p in paths], self._fs()
        return [self.local_path(p) for p in paths], None

    def local_path(self, path):
        """Download one repo file and return its local path. Cached on disk."""
        repo = self.repo_for(path)
        if is_local(repo):
            local = _root(repo) / path
            if not local.exists():
                raise FileNotFoundError(f"no {path} in {repo}")
            return str(local)
        return hf_hub_download(repo, filename=path, repo_type="dataset")

    def prefetch(self, configs, splits=None, verbose=True):
        """Download every parquet shard of these configs in ONE parallel pass.

        This is the whole reason `stream=False` is faster rather than merely
        different. Called shard by shard, `hf_hub_download` serialises hundreds
        of transfers and pays a round trip for each; `snapshot_download` runs
        them concurrently and skips whatever the cache already holds. Reading is
        then local and the `stream=True` path's per-column round trips vanish.

        Returns the local snapshot directory, or `None` if the Hub is
        unreachable — an offline session with a warm cache still reads fine, so
        a failure here is a slow path, not an error.
        """
        from huggingface_hub import snapshot_download

        if is_local(self.repo_id):
            return str(_root(self.repo_id))
        splits = tuple(splits or self.spec.splits)
        patterns = [f"{c}/{s}-*.parquet" for c in configs for s in splits]
        if verbose:
            print(f"  prefetching {len(configs)} config(s) x {len(splits)} split(s) "
                  f"from {self.repo_id} ...")
        try:
            path = snapshot_download(self.repo_id, repo_type="dataset",
                                     allow_patterns=patterns)
        except Exception as e:                             # noqa: BLE001 — offline
            print(f"  note: prefetch failed ({type(e).__name__}: {e}); "
                  "falling back to per-shard reads.")
            return None
        if verbose:
            # Counted against the patterns asked for, not every parquet in the
            # snapshot — the cache accumulates other configs and rglob would
            # report those too, which reads as a much larger download.
            root = Path(path)
            n = sum(len(list(root.glob(pat))) for pat in patterns)
            print(f"  {n} shard(s) local under {path}")
        return path

    def _fs(self):
        """One cached `HfFileSystem`; each instance holds its own HTTP session."""
        if self._filesystem is None:
            self._filesystem = HfFileSystem()
        return self._filesystem

    def load_base(self, splits=None):
        """Load this dataset's `base` config — audio, references and `nsamples`.

        `load_base()` in §2.2 is the entry point to prefer; this is the raw read
        it wraps, without the audio-feature repair.
        """
        return self.load(self.spec.base_config, splits)

    def load_model(self, name, splits=None):
        """Load this dataset's model config: WER columns + pooled embeddings."""
        return self.load(self.spec.model_config(name), splits)

    def load_probe(self, name):
        """Load this dataset's probe config as a pandas DataFrame."""
        return self.load(self.spec.probe_config(name), splits=["train"])["train"].to_pandas()

    def load_results(self, name):
        """Load this dataset's `results/…json` — the machine-readable run summary.

        Cheaper than `load_model` when all you want is the headline numbers:
        it is a few KB, not a few hundred MB.
        """
        path = self.local_path(self.spec.results_path(name))
        with open(path) as f:
            return json.load(f)

    # --------------------------------------------------------------- write --

    def push(self, ds, config_name, message=None):
        """Push a `Dataset` / `DatasetDict` as `config_name`, overwriting it."""
        _check_writable(f"config {config_name}")
        if is_local(self.repo_id):
            folder = _root(self.repo_id) / config_name
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob("*.parquet"):
                old.unlink()
            parts = ds.items() if isinstance(ds, DatasetDict) else [("train", ds)]
            for split, d in parts:
                d.to_parquet(str(folder / f"{split}-00000-of-00001.parquet"))
            self._configs = self._files = self._revision = None
            print(f"wrote config   {config_name} -> {folder}")
            return
        ds.push_to_hub(self.repo_id, config_name=config_name,
                       commit_message=message or f"Update config {config_name}")
        self._configs = self._files = None      # a new config may now exist
        print(f"pushed config  {config_name}")

    def push_dataframe(self, df, config_name, message=None):
        """Push a pandas DataFrame as a single-split config (used for probe tables)."""
        self.push(Dataset.from_pandas(df, preserve_index=False), config_name, message)

    def copy_splits(self, src_config, dst_config, splits, message=None):
        """Copy whole splits between two configs of this repo, **without moving bytes**.

        The Hub can duplicate an LFS file from one path to another server-side,
        so this is a metadata commit however large the split is. That is the
        whole point: a wider spec's `validation` and `test` can be the *same
        rows* as a narrower one's — same source globs, same filter, same sort — so
        re-deriving them would mean 2.7 GB down and 2.7 GB up to produce files
        that already exist, and for a `model_*` config it would mean re-running
        the GPU.

        `push_to_hub` and this method are complementary and the ORDER matters.
        `push_to_hub` rewrites the config's card entry to name exactly the
        splits it was given, so pushing `train` alone un-advertises anything
        copied earlier. Push first, copy second.

        Both card blocks are updated, as in `delete_config`: `configs` (the
        data-file globs `load_dataset` follows) and `dataset_info` (the schema
        and row counts), the latter taken verbatim from the source config —
        which is only correct because the rows are identical.
        """
        _check_writable(f"{dst_config}")
        from huggingface_hub import (CommitOperationCopy, CommitOperationDelete,
                                     DatasetCard)

        splits = list(splits)
        if src_config == dst_config:
            raise ValueError(f"src and dst are both {src_config!r}")
        if src_config not in self.configs():
            raise KeyError(f"no config {src_config!r} in {self.repo_id}")

        copies, moved = [], {}
        for split in splits:
            paths = self.parquet_paths(src_config, split)
            for src in paths:
                copies.append(CommitOperationCopy(
                    src_path_in_repo=src,
                    path_in_repo=f"{dst_config}/{src.split('/', 1)[1]}"))
            moved[split] = len(paths)

        # Stale shards of the same split in the destination must go — a different
        # `-of-N` count means different filenames, so overwriting is not enough.
        # But a path that is ALSO being written must never be queued for
        # deletion: within one commit the delete wins, so a re-run would remove
        # exactly the files it just copied. (`push_to_hub` guards the same case with
        # `not in repo_files_to_add`.) Copies are
        # ordered first for the same reason `datasets` orders additions first.
        written = {op.path_in_repo for op in copies}
        deletions = [CommitOperationDelete(path_in_repo=f)
                     for f in self.files()
                     if f.endswith(".parquet") and f not in written
                     and f.startswith(tuple(f"{dst_config}/{s}-" for s in splits))]

        self.api.create_commit(
            repo_id=self.repo_id, repo_type="dataset", operations=copies + deletions,
            commit_message=message or f"Copy {'+'.join(splits)} from "
                                      f"{src_config} to {dst_config}")
        ops = copies + deletions

        # --- the card, so `load_dataset` can see what was just copied --------
        card = DatasetCard.load(self.repo_id, repo_type="dataset")
        cfgs = card.data.get("configs") or []
        infos = card.data.get("dataset_info") or []

        entry = next((c for c in cfgs if c.get("config_name") == dst_config), None)
        if entry is None:
            entry = {"config_name": dst_config, "data_files": []}
            cfgs.append(entry)
        files = [f for f in entry.get("data_files", []) if f.get("split") not in splits]
        files += [{"split": s, "path": f"{dst_config}/{s}-*"} for s in splits]
        entry["data_files"] = files

        src_info = next((c for c in infos if c.get("config_name") == src_config), None)
        dst_info = next((c for c in infos if c.get("config_name") == dst_config), None)
        if src_info and dst_info:
            carried = [d for d in (src_info.get("splits") or []) if d.get("name") in splits]
            kept = [d for d in (dst_info.get("splits") or []) if d.get("name") not in splits]
            dst_info["splits"] = kept + carried
            dst_info["dataset_size"] = sum(d.get("num_bytes") or 0
                                           for d in dst_info["splits"])
        elif src_info:
            print(f"  note: {dst_config} has no dataset_info entry yet — copying "
                  f"{src_config}'s row counts is not possible; the card will carry "
                  "data-file patterns only.")

        card.data["configs"], card.data["dataset_info"] = cfgs, infos
        card.push_to_hub(self.repo_id, repo_type="dataset",
                         commit_message=f"Advertise {'+'.join(splits)} on {dst_config}")
        self._configs = self._files = None
        print(f"copied         {moved} from {src_config} -> {dst_config} "
              f"({len(ops)} file op(s), no upload)")
        return moved

    def delete_config(self, config_name, dry_run=True, message=None):
        """Delete a config: its parquet folder AND its two dataset-card entries.

        Deleting only the folder leaves the card advertising a config whose files
        are gone, which reads worse than the config did — `load_dataset` then
        fails on a missing path rather than on a missing config. So both YAML
        blocks are rewritten too: `dataset_info` (the schema) and `configs` (the
        data-file globs).

        `dry_run=True` by default. It prints exactly what would go and changes
        nothing; pass `dry_run=False` to actually delete. **There is no undo** —
        the files are removed from the repo head, and the bytes stay only in the
        git history.
        """
        if not dry_run:
            _check_writable(f"delete {config_name}")
        from huggingface_hub import DatasetCard

        if config_name not in self.configs():
            raise KeyError(f"no config {config_name!r} in {self.repo_id}")

        # `push_to_hub` stores the unnamed default config under `data/`.
        folder = "data" if config_name == "default" else config_name
        doomed = [f for f in self.files() if f.startswith(f"{folder}/")]
        card = DatasetCard.load(self.repo_id, repo_type="dataset")
        info = [c for c in (card.data.get("dataset_info") or [])
                if c.get("config_name") != config_name]
        cfgs = [c for c in (card.data.get("configs") or [])
                if c.get("config_name") != config_name]

        print(f"{'would delete' if dry_run else 'deleting'} config {config_name!r}: "
              f"{len(doomed)} file(s) under {folder}/, "
              f"{len(card.data.get('dataset_info') or []) - len(info)} card entr(y/ies)")
        if dry_run:
            for f in doomed:
                print(f"    {f}")
            print("  dry_run=True — nothing removed. Pass dry_run=False to delete.")
            return doomed

        self.api.delete_folder(path_in_repo=folder, repo_id=self.repo_id,
                               repo_type="dataset",
                               commit_message=message or f"Remove config {config_name}")
        card.data["dataset_info"], card.data["configs"] = info, cfgs
        card.push_to_hub(self.repo_id, repo_type="dataset",
                         commit_message=f"Drop {config_name} from the dataset card")
        self._configs = self._files = None
        print(f"deleted config {config_name}")
        return doomed

    def push_json(self, payload, path_in_repo, message=None):
        """Upload a dict as a JSON file inside the repo (used for `results/*.json`)."""
        _check_writable(path_in_repo)
        local = path_in_repo.replace("/", "_")
        with open(local, "w") as f:
            json.dump(payload, f, indent=2)
        self.push_file(local, path_in_repo, message=message, verbose=False)
        print(f"pushed file    {path_in_repo}")

    # -------------------------------------------------------- opaque files --

    def push_file(self, local_path, path_in_repo, message=None, verbose=True):
        """Upload one local file verbatim. Returns `path_in_repo`.

        `push_json` serialises a dict; this takes whatever is already on disk,
        which is what an Optuna SQLite study needs — it is a binary the tuner
        owns and this layer must not interpret.
        """
        _check_writable(path_in_repo)
        repo = self.repo_for(path_in_repo)
        if is_local(repo):
            dest = _root(repo) / path_in_repo
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_path, dest)
        else:
            self.api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                                 repo_id=repo, repo_type="dataset",
                                 commit_message=message or f"Update {path_in_repo}")
        self._files = None
        if verbose:
            size = Path(local_path).stat().st_size
            print(f"pushed file    {path_in_repo}  ({size / 1e6:.1f} MB)")
        return path_in_repo

    def push_files(self, files, message=None, verbose=True):
        """Upload several local files in ONE commit. `files` is `{path_in_repo: local_path}`.

        Thirty per-arm frames written after one seed are one logical write;
        thirty commits for them is what gets a repo rate-limited, and a run
        that dies between them leaves the mirror half-updated. One commit is
        atomic and cheap.
        """
        from huggingface_hub import CommitOperationAdd

        if not files:
            return []
        _check_writable(f"{len(files)} file(s)")
        by_repo = {}
        for path, local in files.items():
            by_repo.setdefault(self.repo_for(path), {})[path] = local
        for repo, group in by_repo.items():
            if is_local(repo):
                for path, local in group.items():
                    dest = _root(repo) / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, dest)
                continue
            ops = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local))
                   for path, local in group.items()]
            self.api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                                   commit_message=message or f"Update {len(ops)} file(s)")
        ops = list(files)
        self._files = None
        if verbose:
            size = sum(Path(v).stat().st_size for v in files.values())
            print(f"pushed {len(ops)} file(s) in one commit  ({size / 1e6:.1f} MB)")
        return list(files)

    def pull_file(self, path_in_repo, local_path=None, verbose=True):
        """Download one file from the repo, or return `None` if it is not there.

        `None` rather than an exception because the caller — a resumable study —
        cannot tell "first run" from "lost the database" any other way, and the
        first run is the common case.
        """
        repo = self.repo_for(path_in_repo)
        try:
            cached = self.local_path(path_in_repo)
        except Exception as e:                                 # noqa: BLE001
            if verbose:
                print(f"  no {path_in_repo} in {repo} ({type(e).__name__})")
            return None
        if local_path is None:
            return cached
        # The cache entry is a symlink into a blob store and is REPLACED on the
        # next download, so an SQLite file opened from it would be swapped under
        # the connection. Copy it out before handing it to anything that writes.
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, local_path)
        if verbose:
            print(f"pulled file    {path_in_repo} -> {local_path} "
                  f"({local_path.stat().st_size / 1e6:.1f} MB)")
        return str(local_path)

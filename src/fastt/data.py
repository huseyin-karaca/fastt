__all__ = ['ensure_audio_feature', 'load_base', 'load_raw', 'build_base', 'extend_model_config', 'build_mfcc_frames']

from datasets import Audio, DatasetDict, load_dataset

from fastt.core import SAMPLING_RATE, active_dataset, require_audio_datasets
from fastt.hub import FasttHub


def ensure_audio_feature(ds, fallback_sr=SAMPLING_RATE):
    """Guarantee a decoding `Audio` column. Returns `(ds, sampling_rate)`.

    If a dataset was pushed while its audio column had lost the `Audio` feature
    type, `load_dataset` hands back a plain `{"bytes":…, "path":…}` struct:
    decoding never happens, so `example["audio"]` has no `"array"` key. Casting
    the column restores the type without re-encoding anything.

    Sampling rate is resolved in this order: the feature's declared rate, then
    the rate of the first decoded example, then `fallback_sr`.
    """
    feats = ds.features if hasattr(ds, "features") else next(iter(ds.values())).features
    audio = feats.get("audio")

    if not isinstance(audio, Audio) or not audio.decode:
        print(f"  repairing 'audio' (found {type(audio).__name__}) "
              f"-> Audio(sampling_rate={fallback_sr})")
        return ds.cast_column("audio", Audio(sampling_rate=fallback_sr)), fallback_sr

    if audio.sampling_rate is not None:
        return ds, audio.sampling_rate

    probe = ds if hasattr(ds, "features") else next(iter(ds.values()))
    return ds, probe[0]["audio"]["sampling_rate"]


def load_base(spec=None, repo_id=None, splits=None, sampling_rate=None):
    """Load the active dataset's `base` config, ready for any extractor in §4.

    This is the single entry point every model subsection uses, so the audio
    repair and the rate check happen once, here, instead of being re-derived in
    each model cell. Nothing about it is corpus-specific: `use_dataset(...)`
    then `load_base()` is the entire switch.
    """
    spec = active_dataset(spec)
    sampling_rate = sampling_rate or spec.sampling_rate
    ds = FasttHub(repo_id, spec=spec).load(spec.base_config, splits)
    ds, sr = ensure_audio_feature(ds, fallback_sr=sampling_rate)
    if sr != sampling_rate:
        raise ValueError(
            f"{spec.base_config} decodes at {sr} Hz but {sampling_rate} Hz is "
            f"expected. Cast first: "
            f"ds.cast_column('audio', Audio(sampling_rate={sampling_rate}))"
        )
    missing = [s for s in (splits or spec.splits) if s not in ds]
    if missing:
        raise KeyError(f"{spec.base_config} is missing split(s) {missing}; "
                       f"it has {list(ds)}. Rebuild it with build_base() (§2.3).")
    n = sum(d.num_rows for d in ds.values())
    print(f"{spec.name}/{spec.base_config}: {len(ds)} splits, {n:,} utterances, {sr} Hz")
    return ds

import os
from io import BytesIO

import soundfile as sf


def load_raw(spec=None, splits=None, **load_kwargs):
    """Load one corpus's evaluation splits straight from its source repo.

    Only the splits named by the spec are requested — via `data_files` globs
    when the spec has them, which is what keeps a training set nobody here has
    any use for from being downloaded.

    `splits` narrows that further to a subset of `spec.splits`, which is how a
    corpus whose *train* split has grown gets rebuilt without re-downloading a
    validation and test set that have not changed.

    Returns a `DatasetDict` keyed by the requested split names, still carrying
    the source repo's own column names; `build_base` canonicalises them.
    """
    require_audio_datasets()
    spec = active_dataset(spec)
    splits = tuple(splits or spec.splits)
    unknown = [s for s in splits if s not in spec.splits]
    if unknown:
        raise KeyError(f"{spec.name}: {unknown} not in splits={spec.splits}")
    kwargs = {"verification_mode": "no_checks", **load_kwargs}
    if spec.data_files:
        return load_dataset(path=spec.source,
                            data_files={s: spec.data_files[s] for s in splits},
                            **kwargs)

    # No globs: name the config and ask for one split at a time. This path
    # requires `spec.splits` to be the source's own split names — prefer
    # `data_files` when they are not, or when the source has a training split
    # that must not be downloaded.
    return DatasetDict({
        s: load_dataset(spec.source, spec.source_config, split=s, **kwargs)
        for s in splits
    })


def _scorable(spec):
    """A row predicate: True when the reference can actually be scored.

    Empty references make WER undefined (`nref == 0`), and a blocklisted one
    marks a segment the corpus itself says not to score. LibriSpeech has
    neither; VoxPopuli and AMI have both.
    """
    blocked = {b.strip().lower() for b in spec.text_blocklist}
    text_col = spec.source_column("text")

    def keep(row):
        text = (row[text_col] or "").strip()
        if spec.drop_empty_text and not text:
            return False
        return text.lower() not in blocked

    return keep


def build_base(spec=None, repo_id=None, target_sr=None, push=False, num_proc=None,
               splits=None):
    """Build one corpus's `base` config: canonical columns, `nsamples`, sorted.

    Parameters
    ----------
    spec : the corpus to build. Defaults to the active dataset (§1.5).
    push : nothing is uploaded unless this is True.
    splits : build only these of `spec.splits`. Every step here is per-split —
        the filter, the `nsamples` map and the sort all are — so a split built
        alone is byte-identical to the same split built alongside the others.
        That is what makes it safe to build one split now and copy the rest
        across from a corpus that shares them (`FasttHub.copy_splits`).

        **Pushing a subset rewrites the config's card entry to name only those
        splits.** The files of the others survive, but they stop being
        advertised, so push the subset first and re-advertise the rest after.

    Returns the built `DatasetDict`.
    """
    require_audio_datasets()
    spec = active_dataset(spec)
    target_sr = target_sr or spec.sampling_rate
    splits = tuple(splits or spec.splits)
    print(f"building {spec.base_config} from {spec.source}"
          f"{'/' + spec.source_config if spec.source_config else ''}"
          f"{'' if set(splits) == set(spec.splits) else '  splits=' + str(list(splits))}")
    ds = load_raw(spec, splits=splits)

    # decode=False keeps the column as {"bytes":…, "path":…}: no decoder objects,
    # no re-encoding on write, and the map below stays cheap.
    ds = ds.cast_column("audio", Audio(decode=False))

    # Unscorable rows go BEFORE the length map, so nothing is spent on them.
    if spec.drop_empty_text or spec.text_blocklist:
        before = {s: d.num_rows for s, d in ds.items()}
        ds = ds.filter(_scorable(spec))
        dropped = {s: before[s] - d.num_rows for s, d in ds.items() if before[s] > d.num_rows}
        if dropped:
            print(f"  dropped unscorable rows: {dropped}")

    # A corpus whose own identifier is only unique within a recording gets `id`
    # composed here, before the rename loop, so the loop's "already occupied"
    # check sees the composed column. This is a `map` rather than a rename
    # because there is nothing to rename: the column does not exist yet.
    if spec.id_columns:
        have = next(iter(ds.values())).column_names
        gone = [c for c in spec.id_columns if c not in have]
        if gone:
            raise KeyError(f"{spec.name}: id_columns {gone} are not in {spec.source}, "
                           f"which has {sorted(have)}")
        if "id" in have:
            raise KeyError(f"{spec.name}: {spec.source} already has an 'id' column; "
                           "composing one would overwrite it. Use columns={'id': ...}.")
        sep = spec.id_separator
        ds = ds.map(lambda b: {"id": [sep.join(str(v) for v in row)
                                      for row in zip(*(b[c] for c in spec.id_columns))]},
                    batched=True, batch_size=1000)

    # Canonicalise: whatever the corpus calls its id and its reference, `base`
    # calls them `id` and `text`.
    columns = set(next(iter(ds.values())).column_names)
    for canonical, source in spec.columns.items():
        if source == canonical:
            continue
        if source not in columns:
            raise KeyError(f"{spec.name}: columns[{canonical!r}] = {source!r}, which is "
                           f"not in {spec.source}; it has {sorted(columns)}")
        if canonical in columns:
            # Renaming onto an occupied name is an Arrow error several frames
            # away from the spec that caused it. Say so here instead.
            raise KeyError(
                f"{spec.name}: cannot rename {source!r} -> {canonical!r}, "
                f"{spec.source} already has a {canonical!r} column. Carry the "
                f"original under a different name via `extra_columns`, or drop it.")
        ds = ds.rename_column(source, canonical)

    def add_nsamples(batch):
        n = []
        for a in batch["audio"]:
            # Either bytes are embedded (Hub parquet) or a path is set (script loaders).
            src = BytesIO(a["bytes"]) if a.get("bytes") is not None else a["path"]
            info = sf.info(src)                        # header only, no PCM decoded
            frames = info.frames
            # nsamples must describe the audio AS CONSUMED downstream, so scale to
            # the resampled length when the file's native rate differs.
            if info.samplerate != target_sr:
                frames = round(frames * target_sr / info.samplerate)
            n.append(frames)
        return {"nsamples": n}

    ds = ds.map(add_nsamples, batched=True, batch_size=512,
                num_proc=num_proc or os.cpu_count()).sort("nsamples")

    # Zero-length audio goes here, AFTER the sort, and via `select` rather than
    # `filter`. Both details are load-bearing, and getting either wrong hangs the
    # build on a large corpus rather than failing:
    #
    #   * `Dataset.sort` reads its key with `query_table(..., indices=self._indices)`.
    #     With no indices mapping that is a zero-copy slice of the memory-mapped
    #     table; with one it is a `fast_gather` of EVERY column — so a `filter`
    #     placed just before the sort drags all ~10 GB of audio bytes into RAM.
    #     (The text filter above escapes this only because the `map` between them
    #     writes a fresh Arrow file and clears the mapping.)
    #   * Sorting first also makes the survivors contiguous — the sort is
    #     ascending and the predicate is `n > 0`, so the dropped rows are a
    #     prefix. `select` detects that and slices the table instead of building
    #     an indices mapping, which keeps `push_to_hub` on its fast path.
    #
    # Dropping after sorting is otherwise identical to dropping before: removing
    # rows from a sorted sequence leaves it sorted.
    if spec.drop_empty_audio:
        before = {s_: d.num_rows for s_, d in ds.items()}
        ds = DatasetDict({
            s_: d.select([i for i, n in enumerate(d["nsamples"]) if n > 0])
            for s_, d in ds.items()
        })
        gone = {s_: before[s_] - d.num_rows for s_, d in ds.items()
                if before[s_] > d.num_rows}
        if gone:
            print(f"  dropped rows with zero-length audio: {gone}")

    have = set(next(iter(ds.values())).column_names)
    missing = [c for c in spec.base_column_dtypes() if c not in have and c != "nsamples"]
    if missing:
        raise KeyError(
            f"{spec.name}: column(s) {missing} are not in {spec.source}, which has "
            f"{sorted(have)}. Fix `columns` / `extra_columns` on the spec.")
    base = ds.select_columns(["id", "audio", "text"]
                             + [c for c in spec.extra_columns] + ["nsamples"])
    # Metadata only — the stored bytes are untouched — but this is what makes the
    # pushed config load as decodable audio.
    base = base.cast_column("audio", Audio(sampling_rate=target_sr))

    n = sum(d.num_rows for d in base.values())
    print(f"  {len(base)} splits, {n:,} utterances, "
          f"{sum(sum(d['nsamples']) for d in base.values()) / target_sr / 3600:.1f} h")

    if push:
        FasttHub(repo_id, spec=spec).push(
            base, spec.base_config,
            f"Add {spec.name} base metadata and audio (sorted by nsamples)")
    else:
        print("push=False — nothing uploaded.")
    return base

def extend_model_config(ex, ds_base, src_spec, cfg=None, repo_id=None, verbose=True):
    """Extend one model's config from a smaller corpus onto a larger, nested one.

    Parameters
    ----------
    ex : the extractor, already bound to the LARGE corpus (build it after
        `use_dataset(<large corpus>)`). Its `.spec` decides the output splits
        and schema; `ex.name` picks the config to read on the other side.
    ds_base : the large corpus's `base`, restricted to the splits that actually
        grew — in practice `load_base(splits=["train"])`. Any split of
        `ex.spec.splits` NOT in here is carried over from `src_spec` untouched,
        which is the point: no audio is decoded and no GPU is used for it.
    src_spec : the smaller corpus whose `model_<name>` config is being extended.
    cfg : the `ExtractConfig` for the rows that do have to be extracted.

    Returns a `DatasetDict` with the source config's exact columns, ready for
    `ex.probe(...)` and `ex.push(...)`. Nothing is uploaded here.

    Rows are returned in `base` order — the same duration sort every other
    config in the repo follows. Alignment downstream is by `(split, id)`, so
    this is tidiness rather than correctness, but a config that is *almost* in
    the canonical order is the kind of thing that goes unnoticed until a
    row-order assumption is added somewhere.
    """
    from datasets import DatasetDict, concatenate_datasets

    src_spec = active_dataset(src_spec)
    src_config = src_spec.model_config(ex.name)
    old = FasttHub(repo_id, spec=src_spec).load(src_config)
    cols = old[list(old)[0]].column_names
    if verbose:
        print(f"{src_config}: " + ", ".join(f"{s} {d.num_rows:,}" for s, d in old.items()))

    out = {}
    for split in ex.spec.splits:
        if split not in ds_base:
            if split not in old:
                raise KeyError(
                    f"{split!r} is neither in ds_base nor in {src_config} "
                    f"(which has {list(old)}) — nothing to build it from.")
            out[split] = old[split].select_columns(cols)
            if verbose:
                print(f"  {split:<12} {out[split].num_rows:>7,} carried over unchanged")
            continue

        target = ds_base[split]
        target_ids = target["id"]
        done = set(old[split]["id"]) if split in old else set()
        stale = done - set(target_ids)
        if stale:
            raise ValueError(
                f"{src_config}/{split} has {len(stale)} id(s) that {ex.spec.name}'s "
                f"base does not (e.g. {sorted(stale)[:3]}). The two corpora are not "
                "nested, so splicing them would mix two different row sets.")

        todo = (target.filter(lambda b: [i not in done for i in b["id"]],
                              batched=True, batch_size=2048)
                if done else target)
        if verbose:
            print(f"  {split:<12} {target.num_rows:>7,} rows: "
                  f"{len(done):,} reused, {todo.num_rows:,} to extract")

        parts = [old[split].select_columns(cols)] if done else []
        if todo.num_rows:
            fresh = ex.extract(DatasetDict({split: todo}), cfg)[split]
            parts.append(fresh.select_columns(cols))
        merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

        position = {i: k for k, i in enumerate(target_ids)}
        merged_ids = merged["id"]
        out[split] = merged.select(sorted(range(len(merged_ids)),
                                          key=lambda r: position[merged_ids[r]]))

    return DatasetDict({s: out[s] for s in ex.spec.splits})

def build_mfcc_frames(ds, spec=None, repo_id=None, push=False):
    """Extract frame-level 13-d MFCCs and (optionally) push them as `<prefix>mfcc`.

    Parameters
    ----------
    ds : the `base` config, with a decoding `Audio` column.
    spec : which dataset `ds` came from; defaults to the active one. It supplies
        the carried-through column types, which differ per corpus.
    push : nothing is uploaded unless this is True.

    Stored as float16: MFCCs have a small dynamic range, so the precision loss
    is far below the noise floor of anything downstream.
    """
    import torch
    import torchaudio
    from datasets import Features, Sequence, Value

    spec = active_dataset(spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HOP, NFFT, N_MFCC = 160, 400, 13        # 10 ms hop, 25 ms window at 16 kHz
    mfcc_tf = torchaudio.transforms.MFCC(
        sample_rate=spec.sampling_rate,
        n_mfcc=N_MFCC,
        melkwargs=dict(n_fft=NFFT, hop_length=HOP, n_mels=40),   # center=True by default
    ).to(device)

    features = Features({
        **{c: Value(dt) for c, dt in spec.base_column_dtypes().items()},
        "mfcc": Sequence(Sequence(Value("float16"))),
        "mfcc_n_frames": Value("int32"),
    })

    def _map(batch):
        arrays = [a["array"] for a in batch["audio"]]
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(a, dtype=torch.float32) for a in arrays], batch_first=True
        ).to(device)
        out = mfcc_tf(padded).transpose(1, 2)             # (B, T, 13)

        mfccs, n_frames = [], []
        for i, a in enumerate(arrays):
            # With center=True the output has L // HOP + 1 frames; clamp for rounding.
            frames = min(len(a) // HOP + 1, out.shape[1])
            mfccs.append(out[i, :frames].cpu().numpy().astype("float16"))
            n_frames.append(frames)
        batch["mfcc"] = mfccs
        batch["mfcc_n_frames"] = n_frames
        return batch

    ds_mfcc = ds.map(_map, batched=True, batch_size=32, remove_columns=["audio"],
                     features=features, writer_batch_size=1000)

    if push:
        FasttHub(repo_id, spec=spec).push(
            ds_mfcc.select_columns(["id", "mfcc", "mfcc_n_frames"]),
            spec.mfcc_config, f"Add frame-level MFCC features for {spec.name}")
    else:
        print("push=False — nothing uploaded.")
    return ds_mfcc

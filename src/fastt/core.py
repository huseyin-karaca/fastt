"""The corpora (`DatasetSpec` and the registry), the Hub repositories, and the environment helpers."""

__all__ = ['PUBLIC_REPO', 'REPO_ID', 'RECORDS_REPO', 'RECORD_ROOTS', 'use_hub', 'SAMPLING_RATE', 'DATASETS', 'LIBRISPEECH', 'VOXPOPULI_TRAIN_SHARDS', 'VOXPOPULI', 'AMI_TRAIN_SHARDS',
           'AMI_SDM', 'PEOPLES_SPEECH_SHARDS', 'PEOPLES_SPEECH_ROWS_PER_SHARD', 'PEOPLES_SPEECH',
           'EARNINGS22_SHARDS', 'EARNINGS22_ROWS_PER_SHARD', 'EARNINGS22', 'DatasetSpec', 'register_dataset',
           'use_dataset', 'active_dataset', 'login_hf', 'require_audio_datasets', 'empty_cache', 'gpu_report',
           'describe_env']

import gc
import os
import subprocess
from dataclasses import dataclass, field

# Two Hub dataset repositories. Paths under `RECORD_ROOTS` (the run records: results JSON, the cross-validation
# frames and the search databases) live in RECORDS_REPO; everything else (audio bases, the experts' configs, probe
# tables) in REPO_ID. Either may be a local folder, written `local:/path`. FASTT_HUB and FASTT_RECORDS override them.
PUBLIC_REPO = "huseyin-karaca/fastt"
REPO_ID = os.environ.get("FASTT_HUB", PUBLIC_REPO)
RECORDS_REPO = os.environ.get("FASTT_RECORDS", PUBLIC_REPO)
RECORD_ROOTS = ("results/", "studies/")
SAMPLING_RATE = 16000                     # every checkpoint here expects 16 kHz mono


@dataclass(frozen=True)
class DatasetSpec:
    """Everything corpus-specific about one evaluation set, in one object.

    Parameters
    ----------
    name : short id, e.g. `"librispeech"`. Also the `results/` subdirectory for
        any prefixed corpus.
    source : the Hub repo the raw audio comes from.
    splits : the split names as they appear in `base` and in every derived
        config. `fit_splits` and `eval_splits` must partition a subset of these
        and must not overlap — probes are fitted on the first and scored on the
        second, and nothing is ever selected on the second.
    source_config : the source repo's config name (`"en"`, `"ihm"`). Used only
        when `data_files` is None.
    data_files : `{our split: glob within the source repo}`. Preferred over
        `source_config`, because it is what keeps a `load_dataset` call from
        pulling a training set nobody here has any use for — every model in
        this notebook is frozen, so there is nothing to train on. LibriSpeech
        train is ~1000 h; VoxPopuli `en` train is ~97 GB.
    columns : `{canonical name: source column}` for `id` and `text`. Anything
        not listed is assumed to already carry its canonical name.
    id_columns : source columns to **join into** `id`, for a corpus whose own
        identifier is only unique within a recording. Earnings-22's
        `segment_id` counts from zero in every call, so `("file_id",
        "segment_id")` is what makes a row identifiable at all. `columns` can
        rename a column but not build one, and `(split, id)` is the key every
        section after §2 aligns on — a duplicate there is not a warning, it is
        two different utterances silently becoming one.
    id_separator : what `id_columns` are joined with. Must not occur inside any
        of them, or the composition stops being reversible.
    extra_columns : `{column: dtype string}` — corpus metadata carried through
        into `base` and into every extracted dataset. Dtypes are strings rather
        than `datasets.Value` objects so this module stays importable without
        `datasets`, which matters in the ESPnet kernel.
    drop_empty_text : drop rows whose reference is empty after stripping. WER
        is undefined without reference words, and both VoxPopuli and AMI ship
        rows like that.
    drop_empty_audio : drop rows whose audio decodes to zero samples. The mirror
        case of `drop_empty_text` and dropped for the same reason — a segment
        with no audio has no transcription any model could produce, so its WER
        is 100% deletions by construction rather than by measurement. It is also
        the one row shape that CRASHES rather than degrades: the wav2vec2/
        data2vec conv stack computes a negative output length for it, and
        because `base` is duration-sorted these rows land together in the very
        first batch, where they set the padded batch width to zero and take the
        whole batch down with `Kernel size can't be greater than actual input
        size`. AMI's full sdm train split has six (all in meeting `TS3007c`);
        no other corpus here has any.
    text_blocklist : reference strings that mark an unscorable segment, matched
        case-insensitively after stripping.
    config_prefix : namespaces every artefact this corpus writes. Empty for
        LibriSpeech, for backwards compatibility with what is already pushed.
    repo_id : the Hub repo for artefacts. `None` means the global `REPO_ID`,
        which is the usual case — corpora are separated by `config_prefix`, not
        by repo. Set it to give one corpus a repo of its own.
    """

    name: str
    source: str
    splits: tuple
    fit_splits: tuple
    eval_splits: tuple
    source_config: str = None
    data_files: dict = None
    columns: dict = field(default_factory=dict)
    id_columns: tuple = ()
    id_separator: str = "-"
    extra_columns: dict = field(default_factory=dict)
    drop_empty_text: bool = True
    drop_empty_audio: bool = True
    text_blocklist: tuple = ()
    language: str = "en"
    config_prefix: str = ""
    repo_id: str = None
    sampling_rate: int = SAMPLING_RATE
    note: str = ""

    def __post_init__(self):
        overlap = set(self.fit_splits) & set(self.eval_splits)
        if overlap:
            raise ValueError(f"{self.name}: {sorted(overlap)} is both fit and eval")
        unknown = (set(self.fit_splits) | set(self.eval_splits)) - set(self.splits)
        if unknown:
            raise ValueError(f"{self.name}: {sorted(unknown)} not in splits={self.splits}")
        # Two ways to say where `id` comes from is one too many: whichever ran
        # second in `build_base` would win, and which that is depends on the
        # order of two loops nobody reads together.
        if self.id_columns and "id" in self.columns:
            raise ValueError(f"{self.name}: set either columns['id'] or id_columns, "
                             f"not both (got {self.columns['id']!r} and "
                             f"{tuple(self.id_columns)})")
        if self.id_columns and len(self.id_columns) < 2:
            raise ValueError(f"{self.name}: id_columns={tuple(self.id_columns)} joins "
                             "one column; use columns={'id': ...} to rename instead")

    # ------------------------------------------------------ artefact names --

    def hub_repo(self, repo_id=None):
        """The repo to read and write. Explicit argument wins, then the spec."""
        return repo_id or self.repo_id or REPO_ID

    @property
    def base_config(self):
        return f"{self.config_prefix}base"

    @property
    def mfcc_config(self):
        return f"{self.config_prefix}mfcc"

    @property
    def model_prefix(self):
        """What every per-model config name starts with — how §5 finds them."""
        return f"{self.config_prefix}model_"

    def model_config(self, name):
        return f"{self.config_prefix}model_{name}"

    def probe_config(self, name):
        return f"{self.config_prefix}probe_{name}"

    def results_path(self, name):
        """`results/<name>.json`, one directory down for any prefixed corpus."""
        return f"results/{self.name}/{name}.json" if self.config_prefix else f"results/{name}.json"

    # ------------------------------------------------------------ columns --

    def source_column(self, canonical):
        """The source repo's name for a canonical column."""
        return self.columns.get(canonical, canonical)

    def base_column_dtypes(self):
        """`{column: dtype string}` for every non-audio column of `base`.

        The order is the order the columns are stored in, and `id` first is
        deliberate: it is the join key every later section aligns on.
        """
        return {"id": "string", "text": "string",
                **self.extra_columns, "nsamples": "int64"}


DATASETS = {}                              # name -> DatasetSpec
_ACTIVE = None                             # the spec every default resolves to

def register_dataset(spec, activate=False):
    """Add a spec to the registry (and optionally make it the active one)."""
    DATASETS[spec.name] = spec
    if activate or _ACTIVE is None:
        use_dataset(spec, verbose=False)
    return spec


def use_dataset(spec, verbose=True):
    """Make `spec` (a name or a `DatasetSpec`) the active dataset. Returns it.

    Every `splits=None` / `repo_id=None` / `spec=None` default in the notebook
    resolves through `active_dataset()` at call time, so this one call switches
    the corpus for §2 through §9 at once. It changes no state on the Hub and
    nothing that is already loaded: re-run `ds = load_base()` and rebuild any
    `EnsembleStore` after switching.
    """
    global _ACTIVE
    if isinstance(spec, str):
        if spec not in DATASETS:
            raise KeyError(f"no dataset {spec!r}; have {sorted(DATASETS)}")
        spec = DATASETS[spec]
    _ACTIVE = spec
    if verbose:
        print(f"active dataset : {spec.name}  ({spec.source}"
              f"{', ' + spec.source_config if spec.source_config else ''})")
        print(f"  splits       : fit {list(spec.fit_splits)} -> eval {list(spec.eval_splits)}")
        print(f"  artefacts    : {spec.base_config}, {spec.model_config('<model>')}, "
              f"{spec.results_path('<model>')} in {spec.hub_repo()}")
    return spec


def active_dataset(spec=None):
    """The active `DatasetSpec`, or `spec` itself when one is passed.

    Written to be called as `spec = active_dataset(spec)` at the top of any
    function that takes an optional `spec` — which is why it accepts a name or
    a spec as well as `None`.
    """
    if spec is not None:
        return DATASETS[spec] if isinstance(spec, str) else spec
    if _ACTIVE is None:
        raise RuntimeError("no active dataset; call use_dataset('librispeech')")
    return _ACTIVE


LIBRISPEECH = register_dataset(DatasetSpec(
    name="librispeech",
    source="openslr/librispeech_asr",
    # Four dev/test splits only. Without `data_files` this pulls the ~1000 h
    # training set as well.
    data_files={
        "validation_clean": "clean/validation/*.parquet",
        "validation_other": "other/validation/*.parquet",
        "test_clean": "clean/test/*.parquet",
        "test_other": "other/test/*.parquet",
    },
    splits=("validation_clean", "validation_other", "test_clean", "test_other"),
    fit_splits=("validation_clean", "validation_other"),
    eval_splits=("test_clean", "test_other"),
    extra_columns={"file": "string", "speaker_id": "int64", "chapter_id": "int64"},
    config_prefix="",                      # see the note above — do not change
    note="Read audiobook speech. References are UPPERCASE, no punctuation, so "
         "the CTC recipes' raw WER nearly matches their normalised WER.",
), activate=True)


# VoxPopuli's own sharding of its English training set. Hardcoded for the same
# reason `AMI_TRAIN_SHARDS` is: a glob cannot ask for "the first N shards", and
# the filenames carry the total, so a count borrowed from another config
# silently matches nothing.
VOXPOPULI_TRAIN_SHARDS = 30


def _voxpopuli(n_train_shards=1, name="voxpopuli"):
    """The VoxPopuli-en spec, with `n_train_shards` of its 30 training shards.

    `validation` and `test` are one parquet each and are always taken whole;
    only `train` is subsampled. One shard is ~3.2 GB and about 6k utterances,
    which is the same order as AMI's 4-shard control and enough to give the
    probe a fit set that is not just `validation`.

    Shards are whole files, so this is a contiguous slice of the corpus rather
    than a random sample — a size control, not a sampling scheme, exactly as in
    `_ami`. `name` overrides the registry name and with it the `config_prefix`,
    which is how a second shard count gets artefacts of its own instead of
    quietly overwriting the first one's.
    """
    total = VOXPOPULI_TRAIN_SHARDS
    if not 0 <= n_train_shards <= total:
        raise ValueError(f"voxpopuli/en has {total} train shards, "
                         f"asked for {n_train_shards}")
    train = [f"en/train-{i:05d}-of-{total:05d}.parquet"
             for i in range(n_train_shards)]
    splits = (("train",) if train else ()) + ("validation", "test")
    return DatasetSpec(
        name=name,
        source="facebook/voxpopuli",
        source_config="en",
        data_files={**({"train": train} if train else {}),
                    "validation": "en/validation-*.parquet",
                    "test": "en/test-*.parquet"},
        splits=splits,
        # Same shape as AMI: fit the probe and the router on train+validation,
        # score on test. With `n_train_shards=0` this degenerates to the old
        # validation-only spec, which is what every pre-2026-08 result used.
        fit_splits=(("train", "validation") if train else ("validation",)),
        eval_splits=("test",),
        # `normalized_text` is the scorable reference: lowercased, numbers spelled
        # out, no punctuation. `raw_text` keeps the casing and punctuation and is
        # carried along so a punctuation-aware rescoring stays possible later.
        columns={"id": "audio_id", "text": "normalized_text"},
        extra_columns={"raw_text": "string", "speaker_id": "string",
                       "gender": "string", "is_gold_transcript": "bool"},
        config_prefix=f"{name}_",
        note="European Parliament plenary speeches: long utterances, many non-native "
             "accents, and some rows whose `normalized_text` is empty — those are "
             "dropped when `base` is built. `is_gold_transcript=False` marks a "
             "machine-aligned transcript; keep it in mind before reading a WER gap "
             f"as an acoustic effect. Train subsampled to {n_train_shards}/{total} "
             "shard(s).",
    )


VOXPOPULI = register_dataset(_voxpopuli())


# AMI's own sharding of its training set, per microphone condition. Hardcoded
# because there is no way to ask for "the first N shards" with a glob, and the
# two conditions are NOT sharded the same way — a count borrowed from one
# silently produces filenames that match nothing in the other.
AMI_TRAIN_SHARDS = {"ihm": 42, "sdm": 27}


def _ami(mic, n_train_shards=4, name=None):
    """The AMI spec for one microphone condition — `"ihm"` or `"sdm"`.

    Same meetings, same references, same segmentation; only the recording
    changes. Keeping them as two registered specs rather than one parameter
    means their artefacts cannot be confused, which matters because a WER
    difference between them is the entire point of running both.

    `name` overrides the registry name, and with it the `config_prefix` — which
    is how a second shard count gets artefacts of its own instead of quietly
    overwriting the first one's: the same corpus at two training-set sizes
    must write to two sets of configs.

    `n_train_shards` takes a **subsample** of AMI's training set — 4 of 27 sdm
    shards is about 15,900 utterances, against 107k for the whole thing. This
    is the one corpus here with a training split worth having: it is the only
    one large enough that fitting a probe on `validation` alone leaves the
    router short of rows, and unlike LibriSpeech train it is the same kind of
    speech as the eval splits. Shards are whole files, so the subsample is a
    contiguous slice of the corpus rather than a random one; it is a size
    control, not a sampling scheme.
    """
    total = AMI_TRAIN_SHARDS[mic]
    if not 0 <= n_train_shards <= total:
        raise ValueError(f"ami/{mic} has {total} train shards, asked for {n_train_shards}")
    train = [f"{mic}/train-{i:05d}-of-{total:05d}.parquet"
             for i in range(n_train_shards)]
    name = name or (f"ami_{mic}" if mic != "ihm" else "ami")
    return DatasetSpec(
        name=name,
        source="edinburghcstr/ami",
        source_config=mic,
        data_files={"train": train,
                    "validation": f"{mic}/validation-*.parquet",
                    "test": f"{mic}/test-*.parquet"},
        splits=("train", "validation", "test"),
        # Probes and routers fit on train+validation and are scored on test —
        # ~29k rows against 12.6k. LibriSpeech's `fit_splits` is a pair too, so
        # the protocol keeps the same shape across the two corpora even though
        # the split names do not.
        fit_splits=("train", "validation"),
        eval_splits=("test",),
        columns={"id": "audio_id"},
        extra_columns={"meeting_id": "string", "speaker_id": "string",
                       "microphone_id": "string",
                       "begin_time": "float32", "end_time": "float32"},
        text_blocklist=("ignore_time_segment_in_scoring",),
        config_prefix=f"{name}_",
        note="Spontaneous meeting speech, pre-segmented into short turns — the "
             "hardest of the three and the one where WER spreads widest across "
             "models, which is what §6 wants. Segments are short (a couple of "
             "seconds), so Whisper's fixed 30 s window pads heavily: expect it "
             "to be slower per second of audio here than on LibriSpeech. "
             f"Microphone condition: {mic}; train subsampled to "
             f"{n_train_shards}/{total} shards.",
    )


AMI_SDM = register_dataset(_ami("sdm"))        # single distant microphone


# People's Speech ships every split as whole parquet shards, so the shard-slicing
# size control `_voxpopuli` and `_ami` apply to *train* applies here to all
# three. The totals below are the ones in the repo's own filenames
# (`dirty/train-00000-of-03140.parquet`); the `n_shards.json` at the repo root
# disagrees with them — it predates a re-shard — and a count taken from it
# produces filenames that match nothing at all.
PEOPLES_SPEECH_SHARDS = {"train": 3140, "validation": 6, "test": 11}

# Rows per shard: 5,476,898/3140, 18,622/6 and 34,898/11. Only used to print an
# estimate before anything is downloaded, so approximate is fine.
PEOPLES_SPEECH_ROWS_PER_SHARD = {"train": 1744, "validation": 3104, "test": 3172}


def _peoples_speech(n_train_shards=1, n_validation_shards=2, n_test_shards=2,
                    name="peoples_speech"):
    """The People's Speech `dirty` spec, sliced to `n_*_shards` of each split.

    The noisy, heterogeneous end of the corpus set: archive.org audio — council
    meetings, lectures, radio, conference talks — recorded on whatever hardware
    the room had. Against LibriSpeech's read audiobooks, and against AMI-SDM's
    single consistently-bad microphone, the distinguishing property is not that
    it is harder but that it is *varied within the split*. That is what an
    ensemble is for: the oracle gap is bounded by how differently the members
    fail, and members fail differently when the conditions differ.

    `dirty` rather than `clean` is deliberate. The two differ in how
    aggressively the source dropped utterances whose forced alignment disagreed
    with the transcript, so `clean` is easier for the models under test by
    construction — a selection effect on the reference, not an acoustic one.

    **Every split is subsampled here**, unlike the other corpora where only
    `train` is: validation and test are 18,622 and 34,898 utterances whole, and
    eleven models over 53k utterances is several GPU-days for a corpus whose job
    is to be compared with three others of 10-40k. The slice is a contiguous
    prefix of whole shards — a size control, not a sampling scheme, so it needs
    no seed to reproduce. A wider slice is a **second spec**
    (`_peoples_speech(..., name="peoples_speech_full")`), never an edit to this
    one: an edit would silently change every artefact the spec already wrote.
    """
    want = {"train": n_train_shards, "validation": n_validation_shards,
            "test": n_test_shards}
    if want["test"] < 1:
        raise ValueError("peoples_speech: eval_splits is ('test',), so "
                         "n_test_shards must be at least 1")
    files = {}
    for split, k in want.items():
        total = PEOPLES_SPEECH_SHARDS[split]
        if not 0 <= k <= total:
            raise ValueError(f"peoples_speech/dirty {split} has {total} shard(s), "
                             f"asked for {k}")
        files[split] = [f"dirty/{split}-{i:05d}-of-{total:05d}.parquet"
                        for i in range(k)]
    splits = tuple(s for s in ("train", "validation", "test") if files[s])
    rows = sum(want[s] * PEOPLES_SPEECH_ROWS_PER_SHARD[s] for s in splits)
    return DatasetSpec(
        name=name,
        source="MLCommons/peoples_speech",
        source_config="dirty",
        data_files={s: files[s] for s in splits},
        splits=splits,
        fit_splits=tuple(s for s in ("train", "validation") if s in splits),
        eval_splits=("test",),
        # No `columns` mapping: `id` and `text` already carry their canonical
        # names, and `text` is already the scorable form — lowercase, no
        # punctuation, numbers spelled out. `duration_ms` is the corpus's own
        # length column, carried through because `build_base` measures
        # `nsamples` independently and disagreement between the two is the
        # cheapest available check on a resampling bug.
        extra_columns={"duration_ms": "int32"},
        config_prefix=f"{name}_",
        note="Archive.org speech: council meetings, lectures, radio, talks, on "
             "whatever hardware the room had. Noisy, and — unlike every other "
             "corpus here — varied WITHIN the split, so a per-utterance router "
             "has genuinely different acoustic conditions to route between. "
             "Transcripts are crowd- and machine-sourced and `dirty` keeps the "
             "poorly-aligned ones, so a share of the WER is reference error "
             "rather than model error: read the GAP between two models, not an "
             "absolute number. Sliced to "
             f"{n_train_shards}/{PEOPLES_SPEECH_SHARDS['train']} train, "
             f"{n_validation_shards}/{PEOPLES_SPEECH_SHARDS['validation']} validation "
             f"and {n_test_shards}/{PEOPLES_SPEECH_SHARDS['test']} test shard(s) — "
             f"about {rows:,} utterances.",
    )


# ~1.7k train + ~6.2k validation + ~6.3k test. That puts it between VoxPopuli
# (9.7k) and AMI-SDM (41.6k), which is what keeps a pooled mixture of the four
# from being one corpus with three garnishes.
PEOPLES_SPEECH = register_dataset(_peoples_speech())


# Earnings-22 ships as ONE split of 38 shards, 1,511 rows each. The filenames
# carry a content hash (`chunked/test-00000-of-00038-6f4f182bdbb6f186.parquet`),
# so unlike VoxPopuli's and AMI's they cannot be formatted from an index — the
# index is globbed and the hash left to match.
EARNINGS22_SHARDS = 38
EARNINGS22_ROWS_PER_SHARD = 1511          # 57,391 / 38, and flat across shards


def _earnings22(n_fit_shards=4, gap_shards=1, n_test_shards=4, name="earnings22"):
    """The Earnings-22 spec: `validation` and `test` cut from its one split.

    119 hours of real earnings calls from 125 globally-listed companies, in
    seven language regions. The corpus exists to measure **accented** English,
    and it is the only one here whose difficulty is a property of the speakers
    rather than of the room: the audio is clean conference-call audio, the
    references are professional transcripts, and models still do badly on it.
    That makes it the complement of `peoples_speech` — varied speakers under a
    constant channel, against varied channels under ordinary speakers.

    ### It has no fit split, so one is cut here

    The source publishes everything as `test`, because it is a benchmark and
    nothing is meant to be fitted on it. This project fits probes and routers,
    so it needs two disjoint sets, and they are taken as **shard ranges**:
    `validation` is the first `n_fit_shards`, `test` the next `n_test_shards`.
    Nothing is fitted on anything scored, which is the property that matters.

    ### `gap_shards` is not decoration

    A call's segments are consecutive, and a call is about a third of a shard,
    so **one call straddles every adjacent pair of shards** — measured, not
    assumed: shards 0 and 1 share `4482249`, 1 and 2 share `4420696`, 2 and 3
    share `4468678`, while non-adjacent shards share nothing. Cutting
    `validation` and `test` at a shard boundary would therefore put one call on
    both sides, and a call is one set of speakers in one acoustic session — a
    router could learn the speaker rather than the utterance and score itself on
    the result. `gap_shards=1` drops the shard between the two ranges and
    absorbs the straddle. §5 of `16_earnings22` re-checks the disjointness on
    the built corpus rather than trusting this docstring.

    ### The id has to be composed

    `segment_id` restarts at zero in every call — it is unique *within* a file
    and nowhere else — so `id_columns=("file_id", "segment_id")` builds the key
    that §3 onward aligns on. `(file_id, segment_id)` is unique; `segment_id`
    alone is not, and using it would silently merge utterances from different
    calls.
    """
    total = EARNINGS22_SHARDS
    if n_fit_shards < 1 or n_test_shards < 1:
        raise ValueError("earnings22: both splits need at least one shard")
    if gap_shards < 0:
        raise ValueError(f"earnings22: gap_shards={gap_shards} must be >= 0")
    used = n_fit_shards + gap_shards + n_test_shards
    if used > total:
        raise ValueError(f"earnings22 has {total} shards, asked for {n_fit_shards} "
                         f"+ {gap_shards} gap + {n_test_shards} = {used}")

    def files(start, count):
        # The trailing `-*` matches the per-file content hash.
        return [f"chunked/test-{i:05d}-of-{total:05d}-*.parquet"
                for i in range(start, start + count)]

    test_start = n_fit_shards + gap_shards
    rows = (n_fit_shards + n_test_shards) * EARNINGS22_ROWS_PER_SHARD
    return DatasetSpec(
        name=name,
        source="distil-whisper/earnings22",
        source_config="chunked",
        data_files={"validation": files(0, n_fit_shards),
                    "test": files(test_start, n_test_shards)},
        splits=("validation", "test"),
        # The same shape as LibriSpeech: fit on `validation`, score on `test`,
        # and there is no training set to carry because there is no training set.
        fit_splits=("validation",),
        eval_splits=("test",),
        columns={"text": "transcription"},
        id_columns=("file_id", "segment_id"),
        # `file_id` is the call, and therefore the speaker group — it is what
        # `16_earnings22` groups by to check the split is call-disjoint and to
        # join the accent metadata from the `full` config. The timestamps place
        # a segment inside its call, which is the only ordering the corpus has.
        extra_columns={"file_id": "string", "start_ts": "float32",
                       "end_ts": "float32"},
        config_prefix=f"{name}_",
        note="Real earnings calls from 125 globally-listed companies, seven "
             "language regions. Clean channel, professional references, hard "
             "ACCENTS — the complement of People's Speech, which varies the "
             "channel and not the speakers. References are cased and "
             "punctuated and carry inline `<inaudible>` markers, which the "
             "Whisper normaliser strips: read `wer_norm`, never `wer_raw`, "
             "which here compares punctuated references against unpunctuated "
             f"CTC output. Cut into {n_fit_shards} fit + {n_test_shards} eval "
             f"shards of {total} with a {gap_shards}-shard gap between them, so "
             f"no call appears on both sides — about {rows:,} utterances.",
    )


# ~6.0k fit + ~6.0k eval, the same shape as `peoples_speech`, so the two noisy
# corpora are comparable with each other as well as with the quiet ones. Shard 4
# is the gap and is never downloaded.
EARNINGS22 = register_dataset(_earnings22())

def use_hub(data=None, records=None):
    """Point the data repository (`REPO_ID`) and/or the records repository (`RECORDS_REPO`) somewhere else, for
    the rest of the session: a dataset repo id, or `local:/path` for a folder. Returns `(REPO_ID, RECORDS_REPO)`."""
    global REPO_ID, RECORDS_REPO
    if data:
        REPO_ID = data
    if records:
        RECORDS_REPO = records
    return REPO_ID, RECORDS_REPO


def login_hf(token=None):
    """Authenticate against the Hugging Face Hub.

    Tries, in order: an explicit `token`, the `HF_TOKEN` environment variable,
    and the Colab secret named `HF_TOKEN`. Writing to the dataset repo needs a
    token with write scope; reading public configs needs none, so a failure
    here is only fatal if you intend to push.
    """
    from huggingface_hub import login

    if token is None:
        token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:                                  # noqa: BLE001 — not on Colab
            pass
    if token is None:
        raise RuntimeError(
            "No token found. Pass login_hf(token=...), set HF_TOKEN, or add an "
            "HF_TOKEN secret in Colab (key icon in the left sidebar)."
        )
    login(token=token)
    print("Logged in to the Hugging Face Hub.")


def require_audio_datasets():
    """Fail early, and legibly, if `datasets` is too new to decode audio here.

    `datasets` 4.0 replaced the audio decoder with `torchcodec`. Every path in
    this project that touches `example["audio"]["array"]` — extraction and
    corpus building — breaks on it, and the way it breaks is unhelpful: a
    `ModuleNotFoundError: No module named 'datasets.features._torchcodec'`
    raised fifteen frames deep inside `Dataset.map`, part-way through the
    progress bar, naming a private module nobody has heard of.

    Colab now ships 4.x by default, so the failure mode is "opened the notebook,
    skipped the install cell" rather than anything exotic. Called at the top of
    the audio paths, never at import: §5-§9 read finished parquet and are
    perfectly happy on 4.x.
    """
    import datasets

    major = int(datasets.__version__.split(".")[0])
    if major >= 4:
        raise RuntimeError(
            f"datasets {datasets.__version__} cannot decode audio for this "
            f"project (4.x moved to torchcodec). Run the Group A/B/C install "
            f"cell in this kernel — it pins `datasets<4.0` — then restart the "
            f"runtime:\n"
            f'    !uv pip install -q "datasets<4.0" jiwer librosa soundfile '
            f'"transformers>=4.57" scipy pandas torchaudio')


def empty_cache(report=True):
    """Drop unreferenced tensors and return the freed GPU memory to the driver.

    `gc.collect()` first, because a tensor still held by a reference cycle is
    not freed by `empty_cache()` alone — which is the usual reason "I deleted
    the model but VRAM is still full". Delete the big names yourself
    (`del model, ds_feat`) before calling this.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    if report:
        print(gpu_report())


def gpu_report():
    """One line of `nvidia-smi` memory use, or a note if there is no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "nvidia-smi returned nothing"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "no NVIDIA GPU visible"


def describe_env():
    """Print the versions that actually decide whether a model cell will run.

    Worth running right after a kernel restart: it is the fastest way to tell
    which install group the current kernel is carrying.
    """
    import platform
    print(f"python       {platform.python_version()}")
    for mod in ("torch", "transformers", "datasets", "numpy", "espnet", "nemo"):
        try:
            m = __import__(mod)
            print(f"{mod:<12} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{mod:<12} —")
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"
        print(f"device       {dev}")
    except ImportError:
        pass
    print(f"gpu memory   {gpu_report()}")

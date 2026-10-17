__all__ = ['FRAME_COLUMN', 'SCORE_COLUMNS', 'base_columns', 'ExtractConfig', 'RunResult', 'Encoded', 'ASRProbeExtractor']

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, Features, Sequence, Value

from fastt.core import SAMPLING_RATE, active_dataset, require_audio_datasets
from fastt.hub import FasttHub
from fastt.pooling import ALL_POOLINGS, pool_hidden
from fastt.probing import run_probe
from fastt.scoring import WerScorer, corpus_wer_table


def base_columns(spec=None):
    """The `base` columns carried straight through into the extracted dataset.

    Read from the active `DatasetSpec` rather than hardcoded, because the
    metadata differs per corpus and the types differ with it — LibriSpeech's
    `speaker_id` is an `int64` and AMI's is a `string`, so a fixed schema here
    would fail on one corpus or the other. `id` is always present, which is what
    lets every model config be joined back to the audio.
    """
    return {c: Value(dt) for c, dt in active_dataset(spec).base_column_dtypes().items()}

# Valid encoder frames, padding excluded. Its own dict because the MFCC baseline
# has this but none of the score columns.
FRAME_COLUMN = {"n_frames": Value("int32")}

SCORE_COLUMNS = {
    "transcription": Value("string"),        # exactly as emitted by the model
    "transcription_norm": Value("string"),   # Whisper-normalised
    "text_norm": Value("string"),            # reference, Whisper-normalised
    "sub": Value("int32"), "dele": Value("int32"), "ins": Value("int32"),
    "nref": Value("int32"), "wer": Value("float32"),
    "sub_raw": Value("int32"), "dele_raw": Value("int32"), "ins_raw": Value("int32"),
    "nref_raw": Value("int32"), "wer_raw": Value("float32"),
}


@dataclass
class ExtractConfig:
    """Everything tunable about one extraction run.

    layers : which entries of the encoder's `hidden_states` tuple to pool.
        Negative indices count from the end, so the default `(-1,)` is the final
        encoder layer. They are resolved to ABSOLUTE indices before use, and the
        absolute index is what appears in the column name — `pool_l32_mean` is
        unambiguous in a way that `pool_l-1_mean` is not.

        Extra layers multiply storage and Arrow write time but cost almost no
        GPU time, since `output_hidden_states=True` already materialises every
        layer. Mid-stack layers (roughly 50-75% of depth) usually carry more
        linearly separable phonetic content than the final layer, which sits
        under the decoder and is specialised for it. Only the transformers-based
        models support this; see `final_layer_only`.

    batch_size : utterances per forward pass. `base` is sorted by `nsamples`, so
        batches are length-homogeneous and padding is minimal — except for
        Whisper, which pads everything to 30 s regardless.
    """
    layers: tuple = (-1,)
    poolings: tuple = ALL_POOLINGS
    batch_size: int = 16
    writer_batch_size: int = 500


@dataclass
class RunResult:
    """What `ASRProbeExtractor.run()` hands back.

    A named container rather than a tuple because the corpus WER table is easy
    to want later — recomputing it from `features` costs a full column scan.
    """
    features: DatasetDict          # the extracted dataset: WER columns + pooled columns
    probe: pd.DataFrame            # one row per (layer, pooling, target, eval split)
    corpus: pd.DataFrame           # corpus WER per split, normalised and raw
    payload: dict = None           # the results/<name>.json body, if pushed


def _take(obj, idx):
    """`obj` restricted to batch rows `idx`, whatever `obj` is made of.

    Tensors and arrays are sliced on their first axis, lists by position, and
    dicts, tuples and `ModelOutput`s are rebuilt field by field; anything else
    (a scalar, a string, `None`) is a per-batch constant and passes through.
    This is what lets `Encoded.take` hand one member's decoder only the
    utterances routed to it, without a per-model slicing rule.
    """
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj[idx]
    if isinstance(obj, np.ndarray):
        return obj[np.asarray(idx)]
    if isinstance(obj, dict):                       # ModelOutput is an OrderedDict
        return type(obj)(**{k: _take(v, idx) for k, v in obj.items()})
    if isinstance(obj, tuple):
        return type(obj)(_take(v, idx) for v in obj)
    if isinstance(obj, list):
        return [obj[i] for i in idx]
    return obj


@dataclass
class Encoded:
    """One batch after the encoder and before the decoder.

    `hidden` and `lengths` are what extraction pools; `state` is whatever this
    model's decoder needs to finish the job — logits for a CTC head, the mel
    features and encoder output for Whisper, the padded encoder output and
    its lengths for ESPnet. A `Router` keeps every member's `Encoded`, asks the
    selector, and decodes only the chosen one.
    """
    hidden: dict                    # {absolute layer index -> (B, T, D) tensor}
    lengths: torch.Tensor           # (B,) valid encoder frames per utterance
    state: object = None            # decoder input; model-specific, opaque here

    def __len__(self):
        return int(self.lengths.shape[0])

    def take(self, idx):
        """The same batch restricted to rows `idx` (a list of positions)."""
        idx = [int(i) for i in idx]
        return Encoded({a: h[idx] for a, h in self.hidden.items()},
                       self.lengths[idx], _take(self.state, idx))


class ASRProbeExtractor(ABC):
    """Extract pooled embeddings + WER, probe them, push the result.

    Subclasses supply `_load`, `hidden_size`, `num_encoder_layers` and the
    two halves of the forward pass, `_encode_batch` and `_decode_batch`;
    everything downstream is inherited and identical across models, which is
    the entire point of the class. The public `encode` / `decode` pair is the
    same split for callers with raw audio in hand (`fastt.deploy.Router`).

    Usage
    -----
        ex = Wav2Vec2BaseExtractor()
        ex.smoke_test(ds)                       # shapes and one transcript first
        res = ex.run(ds, push=False)            # extract -> WER -> probe
        ex.push(res)                            # ... then upload
    """

    name: str = None                    # short id, used in config and file names
    model_id: str = None                # Hub checkpoint id
    sampling_rate: int = SAMPLING_RATE
    frame_rate_hz: float = None         # encoder frames per second; may be measured
    final_layer_only: bool = False      # see the table above
    emits_transcription: bool = True
    grad_mode: str = "inference"        # "inference" | "no_grad"
    max_input_seconds: float = None     # set to enable the preflight truncation check
    min_input_samples: int = 0          # zero-pad clips shorter than this; see `_pad_short`

    def __init__(self, model_id=None, device=None, dtype=torch.bfloat16,
                 scorer=None, repo_id=None, spec=None):
        self.model_id = model_id or self.model_id
        # Bound once, at construction: an extractor belongs to the corpus that
        # was active when it was made, so switching datasets mid-run cannot
        # silently retarget a push that is already in flight.
        self.spec = active_dataset(spec)
        self.repo_id = self.spec.hub_repo(repo_id)
        self._warned_pad = False        # `_pad_short` reports once per run, not per batch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # bf16 weights rather than fp32-plus-autocast: same numerics for inference,
        # half the memory, noticeably faster on tensor cores. Hidden states are
        # promoted back to float32 inside pool_hidden. Models with an in-graph STFT
        # override this to float32 in their own __init__.
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.scorer = scorer or (WerScorer() if self.emits_transcription else None)
        self._gen_kwargs = {}
        self._reuse_encoder = True
        self._loaded = False
        self._clock = None              # armed by `latency.time_extractor`; see `_mark_encoded`

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, model_id={self.model_id!r})"

    # ---------------------------------------------------------------- hooks --

    @abstractmethod
    def _load(self):
        """Populate `self.model` and anything else the subclass needs."""

    @property
    @abstractmethod
    def hidden_size(self):
        """Encoder width D."""

    @property
    @abstractmethod
    def num_encoder_layers(self):
        """Number of encoder blocks; `hidden_states` has this many + 1 entries."""

    @abstractmethod
    def _encode_batch(self, arrays, layer_indices):
        """The encoder half: one batch of waveforms -> `Encoded`.

        `hidden` is `{absolute layer index -> (B, T, D) tensor}`, still on
        device; `lengths` the valid encoder frame count per utterance, padding
        excluded; `state` whatever `_decode_batch` needs. Called inside the
        model's grad context; models that autocast do so in here.
        """

    def _decode_batch(self, encoded):
        """The decoder half: `Encoded` -> transcriptions as emitted (no
        normalisation), or `None` when `emits_transcription` is False."""
        return None

    def _encode_and_decode(self, arrays, layer_indices):
        """Run the model on one batch of waveforms: encode, mark, decode.

        Returns `(hidden, lengths, hyps)` — see `_encode_batch` and
        `_decode_batch`. Extraction calls this, so the two halves are the one
        forward pass they always were; `_mark_encoded` between them is the
        timing hook `latency.time_extractor` reads.
        """
        with self._grad_ctx():
            enc = self._encode_batch(arrays, layer_indices)
            self._mark_encoded()
            hyps = self._decode_batch(enc) if self.emits_transcription else None
        return enc.hidden, enc.lengths, hyps

    # ------------------------------------------------------------ inference --

    def encode(self, arrays, layers=(-1,)):
        """Raw waveforms -> `Encoded`. Loads the model on first use.

        `arrays` are 1-d float arrays at `self.sampling_rate`. The clips are
        padded to this model's input floor exactly as extraction pads them.
        """
        self.load()
        arrays = self._pad_short([np.asarray(a, dtype=np.float32) for a in arrays])
        with self._grad_ctx():
            return self._encode_batch(arrays, self._resolve_layers(layers))

    def decode(self, encoded):
        """`Encoded` (or a `take` of one) -> one transcript per utterance."""
        if not self.emits_transcription:
            return None
        with self._grad_ctx():
            return self._decode_batch(encoded)

    def features(self, encoded, pooling="mean", layer=-1):
        """The pooled vector the ensemble store holds for this model: `(B, D)` float32.

        The same `pool_hidden` call extraction makes, on the same layer, so a
        selector fitted on stored features can be applied to a live utterance.
        """
        a = self._resolve_layers((layer,))[0]
        if a not in encoded.hidden:
            raise KeyError(f"layer {a} was not encoded; have {sorted(encoded.hidden)}")
        with self._grad_ctx():                  # the hidden states are inference tensors
            M = pool_hidden(encoded.hidden[a], encoded.lengths, (pooling,))[pooling]
        return M.cpu().numpy().astype(np.float32)

    def transcribe(self, arrays):
        """Raw waveforms -> transcripts, the whole model. Convenience for one call."""
        return self.decode(self.encode(arrays))

    # ------------------------------------------------------------ internals --

    def load(self):
        """Load the checkpoint once; a no-op on later calls. Returns `self`."""
        if not self._loaded:
            print(f"Loading {self.model_id} on {self.device} ({self.dtype})")
            t0 = time.perf_counter()
            self._load()
            self._loaded = True
            print(f"  ready in {time.perf_counter() - t0:.1f}s — "
                  f"{self.num_encoder_layers} encoder layers, d={self.hidden_size}")
        return self

    def _grad_ctx(self):
        """The no-grad context this model tolerates.

        `inference_mode` is strictly faster, but tensors created under it carry a
        flag that some NeMo and ESPnet code paths reject downstream. Those models
        set `grad_mode = "no_grad"`.
        """
        return torch.inference_mode() if self.grad_mode == "inference" else torch.no_grad()

    def _mark_encoded(self):
        """Timing hook. `_encode_and_decode` calls this once, between the
        encoder half and the decoder half.

        A no-op unless `latency.time_extractor` has armed `self._clock`, so it
        costs nothing in extraction. When armed it synchronises the device and
        records the moment the encoder finished, which is what splits one
        model's wall-clock into the encoder half every router must pay for
        every member and the decoder half only the chosen member pays.
        """
        if self._clock is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._clock["encoded"] = time.perf_counter()

    def _resolve_layers(self, layers):
        """Map possibly-negative layer indices onto `hidden_states` positions.

        `hidden_states` is a tuple of `num_encoder_layers + 1` tensors: index 0
        is the convolutional / subsampling output, 1..L are the encoder blocks.
        """
        n = self.num_encoder_layers + 1
        out = []
        for idx in layers:
            a = idx if idx >= 0 else n + idx
            if not 0 <= a < n:
                raise IndexError(f"layer {idx} out of range for {n} hidden states")
            out.append(a)
        out = sorted(set(out))

        if self.final_layer_only and out != [self.num_encoder_layers]:
            raise NotImplementedError(
                f"{type(self).__name__} exposes the final encoder layer only "
                f"(layers=(-1,) -> [{self.num_encoder_layers}]); got {out}. This "
                "model's encoder call returns its output, not the whole stack; "
                "intermediate layers would need forward hooks on the encoder blocks."
            )
        return out

    @staticmethod
    def _token_ids(out):
        """Pull the `(B, L)` id tensor out of whatever `generate` returned.

        Most heads return the ids directly, but some wrap them: Parakeet's TDT
        head returns `ParakeetRNNTGenerateOutput(sequences=..., durations=...)`.
        That is a `ModelOutput`, i.e. an `OrderedDict`, so passing it on to
        `batch_decode` iterates its **keys** and dies far away from the cause
        with `'str' object cannot be interpreted as an integer`. Unwrap here,
        once, instead of in every model.
        """
        seq = getattr(out, "sequences", None)
        if seq is None and isinstance(out, dict):
            seq = out.get("sequences")
        if seq is None and isinstance(out, (tuple, list)):
            seq = out[0]
        return out if seq is None else seq

    def _generate(self, gen_inputs, encoder_outputs):
        """Decode, reusing the encoder pass that already produced hidden states.

        Whisper accepts `encoder_outputs` in `generate`, so the encoder normally
        runs once per batch rather than twice. Support is version- and
        model-dependent, so the first failure disables reuse for the rest of the
        run and says so; a model that is known not to support it sets
        `self._reuse_encoder = False` in `_load`. Results are identical either
        way; only speed moves.
        """
        if self._reuse_encoder:
            try:
                return self._token_ids(
                    self.model.generate(**gen_inputs, encoder_outputs=encoder_outputs,
                                        **self._gen_kwargs))
            except (TypeError, ValueError, AttributeError) as e:
                self._reuse_encoder = False
                print(f"  note: encoder_outputs reuse rejected ({type(e).__name__}: {e}). "
                      "Falling back to a second encoder pass per batch — same "
                      "results, roughly 1.3-1.7x slower.")
        return self._token_ids(self.model.generate(**gen_inputs, **self._gen_kwargs))

    def _build_features(self, cfg, layer_abs):
        """The explicit Arrow schema for the extracted dataset."""
        feats = base_columns(self.spec)
        feats.update(FRAME_COLUMN)
        if self.emits_transcription:
            feats.update(SCORE_COLUMNS)
        for a in layer_abs:
            for p in cfg.poolings:
                # Fixed-length Sequence -> Arrow fixed_size_list: the width is stored
                # once in the schema, not once per row, and it survives push/pull.
                feats[f"pool_l{a}_{p}"] = Sequence(Value("float32"), length=self.hidden_size)
        return Features(feats)

    def _pad_short(self, arrays):
        """Zero-pad any clip below `min_input_samples`. Identity when that is 0.

        Some front ends have a hard floor on input length and raise rather than
        degrade. NeMo's is the strict one: `normalize_batch(..., "per_feature")`
        refuses a mel spectrogram of exactly one frame, because the per-feature
        standard deviation over a single frame is `nan`. Its frame count is
        `floor(n_samples / hop_length)`, so at a 160-sample hop **anything from
        10 ms up to 20 ms produces exactly one frame and raises** — and the
        check is `torch.any(seq_len == 1)`, so one such utterance kills the
        whole batch.

        LibriSpeech has no clip that short. AMI does: it is segmented into
        conversational turns, and `AMI_ES2011c_sdm_FEE041_0104556_0104558` is
        319 samples (19.9 ms) of someone saying "B". Since `base` is sorted by
        length, a clip like that lands in the very first batch and the run dies
        at 0%.

        Padding rather than dropping, for two reasons. Dropping changes the
        corpus, which would desynchronise every model config already extracted
        against this `base` — §6 aligns them row by row on `(split, id)`.
        And it costs nothing: batches are padded to their longest member
        anyway, so the model already sees trailing zeros here. The only thing
        that changes is the declared valid length, and a couple of extra silent
        frames in the pooled mean of a 20 ms utterance is not a measurable
        effect on anything.
        """
        if not self.min_input_samples:
            return arrays
        out, n_padded = [], 0
        for a in arrays:
            if len(a) < self.min_input_samples:
                a = np.pad(np.asarray(a), (0, self.min_input_samples - len(a)))
                n_padded += 1
            out.append(a)
        if n_padded and not self._warned_pad:
            self._warned_pad = True
            ms = 1000 * self.min_input_samples / self.sampling_rate
            print(f"  note: zero-padding clips shorter than {self.min_input_samples} "
                  f"samples ({ms:.0f} ms) up to that length — {n_padded} in this "
                  "batch. Reported once per run.")
        return out

    @staticmethod
    def pooled_columns(ds):
        """Every `pool_l<layer>_<pooling>` column present in a dataset."""
        first = ds[list(ds)[0]] if isinstance(ds, DatasetDict) else ds
        return [c for c in first.column_names if c.startswith("pool_l")]

    # ------------------------------------------------------------- pipeline --

    def extract(self, ds, cfg=None):
        """Run the model over `ds`; return it with pooled columns and WER.

        `ds` must carry the columns of the `base` config — use `load_base()`.
        """
        cfg = cfg or ExtractConfig()
        self.load()
        layer_abs = self._resolve_layers(cfg.layers)
        features = self._build_features(cfg, layer_abs)

        n_cols = len(layer_abs) * len(cfg.poolings)
        print(f"Layers -> {layer_abs} (of 0..{self.num_encoder_layers})   "
              f"poolings -> {list(cfg.poolings)}")
        print(f"{n_cols} pooled columns x {self.hidden_size} float32 = "
              f"{n_cols * self.hidden_size * 4 / 1024:.0f} KB per utterance")

        def _map(batch):
            arrays = self._pad_short([a["array"] for a in batch["audio"]])
            hidden, lengths, hyps = self._encode_and_decode(arrays, layer_abs)

            for a, h in hidden.items():
                for p, M in pool_hidden(h, lengths, cfg.poolings).items():
                    # The one and only device->host transfer: (B, D), not (B, T, D).
                    batch[f"pool_l{a}_{p}"] = M.cpu().numpy()

            batch["n_frames"] = lengths.cpu().tolist()
            if self.emits_transcription:
                batch["transcription"] = hyps
                batch.update(self.scorer.score(batch["text"], hyps))
            return batch

        # Fails here with one clear line rather than fifteen frames deep
        # inside `Dataset.map`, part-way through the progress bar.
        require_audio_datasets()
        print(f"Extracting {self.name} features...")
        return ds.map(_map, batched=True, batch_size=cfg.batch_size,
                      remove_columns=["audio"], features=features,
                      writer_batch_size=cfg.writer_batch_size)

    def _prepare_probe(self, ds_feat, targets):
        """Hook between extraction and probing. Identity for every ASR model.

        It exists for feature-only extractors (the MFCC baseline), which have no
        WER of their own and must borrow the target column from another model's
        config before `probe()` can find it.
        """
        return ds_feat

    def probe(self, ds_feat, targets=("wer",), train_splits=None,
              eval_splits=None, **kw):
        """Probe every (layer, pooling) column, one fit each, identical splits.

        Each `run_probe` call fits the real probe, the shuffled-target control
        and the length-residual probe from a single decomposition, so seven
        poolings cost seven SVDs of an (n_train x D) matrix — seconds.
        """
        train_splits = tuple(train_splits or self.spec.fit_splits)
        eval_splits = tuple(eval_splits or self.spec.eval_splits)
        cols = self.pooled_columns(ds_feat)
        if not cols:
            raise ValueError("no pooled columns found; run extract() first")
        print(f"Probing {len(cols)} feature sets on {list(targets)} "
              f"(fit: {'+'.join(train_splits)}; score: {'+'.join(eval_splits)})")

        frames = []
        for col in cols:
            df = run_probe(ds_feat, targets=targets, feature_col=col,
                           train_splits=train_splits, eval_splits=eval_splits, **kw)
            layer, pooling = col[len("pool_l"):].split("_", 1)
            frames.append(df.assign(model=self.name, layer=int(layer), pooling=pooling))

        results = pd.concat(frames, ignore_index=True)
        lead = ["model", "layer", "pooling", "target", "split", "cv_r2", "r2",
                "selectivity", "r2_beyond_length", "r2_length_only",
                "pearson_r", "spearman_r"]
        return results[[c for c in lead if c in results]
                       + [c for c in results.columns if c not in lead]]

    def select_best(self, results, target="wer", criterion="cv_r2"):
        """Rank pooled columns and return the winner as a one-element list.

        The default criterion is `cv_r2` — leave-one-out R^2 on the FIT splits,
        computed without ever looking at the test splits. Ranking by test `r2`
        instead and reporting the maximum over seven poolings would inflate the
        reported figure by roughly the spread of the seven.

        `cv_r2` belongs to the fit, so it is constant across eval splits; the
        groupby below just deduplicates it.
        """
        sub = results[results["target"] == target]
        if sub.empty:
            raise ValueError(f"no probe rows for target {target!r}")
        ranked = sub.groupby("feature")[criterion].max().sort_values(ascending=False)
        best = ranked.index[0]
        print(f"Best feature set by {criterion} on the fit splits: {best} "
              f"({criterion}={ranked.iloc[0]:.4f})")
        if len(ranked) > 1:
            print(f"  runner-up: {ranked.index[1]} ({ranked.iloc[1]:.4f}); "
                  f"margin {ranked.iloc[0] - ranked.iloc[1]:.4f}")
        return [best]

    # ----------------------------------------------------------------- push --

    def push(self, result, select="all", target="wer", criterion="cv_r2", cfg=None):
        """Push the three artefacts for this model. Returns the JSON payload.

        select : `"all"` (default) pushes every pooled column that was extracted;
            `"best"` pushes only the highest-`criterion` one; a list pushes
            exactly those columns.

        `"all"` is the default because a uniform schema across models is worth
        more than the storage it costs — seven poolings of a 1024-d encoder over
        11k utterances is under 100 MB per split. `probe_<name>` and the JSON
        always describe EVERY pooling extracted, even when fewer are pushed as
        data: discarding the losing columns should not discard the evidence that
        they lost.
        """
        cfg = cfg or ExtractConfig()
        hub = FasttHub(self.repo_id, spec=self.spec)
        ds_feat, results, corpus = result.features, result.probe, result.corpus

        all_cols = self.pooled_columns(ds_feat)
        if select == "best":
            keep = self.select_best(results, target=target, criterion=criterion)
        elif select == "all":
            keep = all_cols
        else:
            keep = list(select)
            unknown = [c for c in keep if c not in all_cols]
            if unknown:
                raise ValueError(f"unknown pooled column(s) {unknown}; have {all_cols}")

        meta = ["id"] + (["transcription", "transcription_norm", "text_norm"]
                         if self.emits_transcription else [])
        meta += ["n_frames"]
        if self.emits_transcription:
            meta += ["sub", "dele", "ins", "nref", "wer",
                     "sub_raw", "dele_raw", "ins_raw", "nref_raw", "wer_raw"]
        push_cols = meta + keep

        mb = len(keep) * self.hidden_size * 4 * sum(d.num_rows for d in ds_feat.values()) / 1e6
        print(f"Pushing {len(keep)} pooled column(s) (~{mb:.0f} MB of features) "
              f"as config {self.spec.model_config(self.name)}")

        hub.push(ds_feat.select_columns(push_cols), self.spec.model_config(self.name),
                 f"{self.model_id} on {self.spec.name}: WER + pooled embeddings "
                 f"({', '.join(keep)})")
        hub.push_dataframe(results, self.spec.probe_config(self.name),
                           f"Linear-probe results for {self.model_id} on {self.spec.name}")

        payload = {
            "model": self.name,
            "model_id": self.model_id,
            # Which corpus these numbers describe. Without it a results file is
            # ambiguous the moment a second dataset exists in the repo.
            "dataset": {
                "name": self.spec.name,
                "source": self.spec.source,
                "source_config": self.spec.source_config,
                "splits": list(self.spec.splits),
                "fit_splits": list(self.spec.fit_splits),
                "eval_splits": list(self.spec.eval_splits),
            },
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "encoder": {
                "hidden_size": self.hidden_size,
                "num_encoder_layers": self.num_encoder_layers,
                "frame_rate_hz": self.frame_rate_hz,
            },
            "extraction": {
                "layers": list(cfg.layers),
                "poolings": list(cfg.poolings),
                "batch_size": cfg.batch_size,
                "text_normalizer": "whisper EnglishTextNormalizer",
            },
            "corpus_wer": corpus.to_dict(orient="records") if corpus is not None else None,
            "probe": {
                "selection_criterion": criterion,
                "selected_columns": self.select_best(results, target, criterion),
                "pushed_columns": push_cols,
                "rows": json.loads(results.to_json(orient="records")),
            },
        }
        hub.push_json(payload, self.spec.results_path(self.name),
                      f"Evaluation summary for {self.model_id} on {self.spec.name}")
        result.payload = payload
        return payload

    def run(self, ds, push=False, cfg=None, targets=("wer",), train_splits=None,
            eval_splits=None, push_select="all", criterion="cv_r2"):
        """Extract -> report WER -> probe -> optionally push. One call.

        `push=False` by default: nothing reaches the Hub until the numbers have
        been looked at. The usual loop is one run with `push=False`, then
        `ex.push(result)` once they look right.
        """
        cfg = cfg or ExtractConfig()
        t0 = time.perf_counter()

        ds_feat = self.extract(ds, cfg)
        ds_feat = self._prepare_probe(ds_feat, targets)

        corpus = corpus_wer_table(ds_feat) if self.emits_transcription else None
        if corpus is not None:
            print(f"\n--- Corpus WER ({self.model_id}) ---")
            print(corpus[["split", "n_utts", "wer_norm", "wer_raw"]].to_string(index=False))
            print("normalised = comparable to the model card; "
                  "raw = comparable to the CTC subsets\n")

        results = self.probe(ds_feat, targets=targets, train_splits=train_splits,
                             eval_splits=eval_splits)
        print(results.round(4).to_string(index=False))

        result = RunResult(features=ds_feat, probe=results, corpus=corpus)
        if push:
            self.push(result, select=push_select, target=targets[0], criterion=criterion, cfg=cfg)
        else:
            print("\npush=False — nothing uploaded. Call ex.push(result) when ready.")

        print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")
        return result

    # ---------------------------------------------------------- diagnostics --

    def smoke_test(self, ds, n=4):
        """Shapes, frame rate and one transcript. **Run this before a full extraction.**

        A wrong attribute path or a transposed tensor fails here in ten seconds
        instead of forty minutes in. Also measures and stores `frame_rate_hz`.

        The sample is the `n-1` **shortest** rows plus the **longest** one, and
        the rate is measured on that last one. `base` is sorted by duration, so
        the head is where the pathological clips live — it is also exactly what
        a real batch 0 looks like, which is what makes it worth smoke-testing.
        But a frame rate cannot be read off it: AMI's shortest clip is 320
        samples (20 ms), fewer than one conv receptive field, so it yields
        **zero** encoder frames. Reading the rate there gave
        `frame_rate_hz = 0.0`, stored it, and then divided by it — which is why
        the published `results/ami_sdm/*.json` carry 0.0 where LibriSpeech
        carries 49.83. Padding the short clips up to a model's input floor does
        not rescue the measurement either, since they then all sit at the floor.
        Including the longest row does, and it costs one extra utterance.
        """
        self.load()
        split = ds[list(ds)[0]]
        n = max(1, min(n, split.num_rows))
        rows = split.select(sorted({*range(n - 1), split.num_rows - 1}))
        arrays = self._pad_short([a["array"] for a in rows["audio"]])
        hidden, lengths, hyps = self._encode_and_decode(
            arrays, self._resolve_layers((-1,)))

        h = next(iter(hidden.values()))
        i = int(np.argmax(lengths.cpu().numpy()))
        seconds = len(arrays[i]) / self.sampling_rate
        if lengths[i] > 0 and seconds > 0:
            self.frame_rate_hz = round(float(lengths[i]) / seconds, 2)
        else:
            print(f"  WARNING: none of the {rows.num_rows} sampled clips produced "
                  f"an encoder frame (longest is {seconds * 1000:.0f} ms); leaving "
                  f"frame_rate_hz at {self.frame_rate_hz}.")

        print(f"hidden  {tuple(h.shape)}   (expect (B, T, {self.hidden_size}))")
        print(f"frames  {lengths.tolist()}  -> {self.frame_rate_hz} frames/s"
              + (f" ({1000 / self.frame_rate_hz:.0f} ms per frame)"
                 if self.frame_rate_hz else ""))
        if h.shape[-1] != self.hidden_size:
            print(f"  WARNING: last dim is {h.shape[-1]}, not {self.hidden_size} — "
                  "the tensor is probably (B, D, T) and needs a transpose.")
        if self.emits_transcription:
            print(f"ref : {rows[0]['text']}")
            print(f"hyp : {hyps[0]}")
        return hidden, lengths, hyps

    def preflight(self, ds):
        """Report anything about `ds` that will quietly distort this model's WER.

        Currently the 30 s input window: a feature extractor that truncates
        long clips inflates their WER, and it is better to know the count up
        front than to wonder about it afterwards.
        """
        if not self.max_input_seconds:
            print("no input-length limit for this model")
            return
        limit = self.max_input_seconds * self.sampling_rate
        for split, d in ds.items():
            over = sum(1 for n in d["nsamples"] if n > limit)
            if over:
                print(f"  WARNING {split}: {over} utterance(s) exceed "
                      f"{self.max_input_seconds:.0f}s (longest "
                      f"{max(d['nsamples']) / self.sampling_rate:.1f}s) and will be "
                      "truncated by the feature extractor, inflating their WER.")

    def benchmark(self, ds, n=16):
        """Time one batch and print the real-time factor.

        RTF is wall-clock divided by audio duration, so RTF 0.2 means the four
        splits (~21 h of audio) would take about 4.2 h. This answers "can I
        afford this decoder" on YOUR machine rather than from a table.
        """
        self.load()
        rows = ds[list(ds)[0]].select(range(n))
        arrays = [a["array"] for a in rows["audio"]]
        seconds = sum(len(a) for a in arrays) / self.sampling_rate
        arrays = self._pad_short(arrays)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._encode_and_decode(arrays, self._resolve_layers((-1,)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        print(f"{n} utterances, {seconds:.1f}s of audio: {dt:.2f}s  "
              f"RTF {dt / seconds:.4f}  -> ~{dt / seconds * 21.2:.2f} h for all four splits")
        return dt / seconds

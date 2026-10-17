__all__ = ['ESPNET_TRANSFORMER', 'ESPNET_CONFORMER', 'ESPNET_TRANSDUCER', 'ESPnetExtractor']

# %% ../../nbs/02_models.ipynb #96532a3b
import sys
import types
from itertools import groupby

import numpy as np
import pandas as pd
import torch

from fastt.extractor import ASRProbeExtractor, Encoded


# Full HF repo ids. The "-truncated-a63357" suffix on the conformer is correct,
# not a typo: HF repo names have a length limit, so the ESPnet exporter truncated
# the original recipe name and appended a hash.
ESPNET_TRANSFORMER = (
    "espnet/Shinji_Watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
)
ESPNET_CONFORMER = (
    "espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer6_n_fft512_hop_length2-truncated-a63357"
)
ESPNET_TRANSDUCER = (
    "espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp"
)


def _ensure_transducer_loss_stub():
    """Satisfy ESPnet's eager import of the RNN-T *training* loss.

    `ESPnetASRModel.__init__` constructs `criterion_transducer = RNNTLoss(...)`
    for any model carrying a joint network, so a transducer checkpoint cannot be
    built without `warprnnt_pytorch` importable — even for pure inference, where
    the loss is never evaluated. The two AED checkpoints have no joint network
    and never reach that branch, which is why only the RNN-T model fails.

    warp-transducer needs source compilation against the CUDA toolkit and does
    not build on current Colab, so rather than fight it, register a stub. It
    raises on call, so a silently-wrong loss is impossible; and it is an
    `nn.Module` with no parameters, so it registers cleanly as a submodule and
    contributes no keys for a strict `load_state_dict` to mismatch on.
    """
    try:
        import warprnnt_pytorch                             # noqa: F401
        return
    except ImportError:
        pass

    class RNNTLoss(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError(
                "warprnnt_pytorch is stubbed for inference only; this model cannot "
                "be trained. Install warp-transducer to train.")

    stub = types.ModuleType("warprnnt_pytorch")
    stub.RNNTLoss = RNNTLoss
    sys.modules["warprnnt_pytorch"] = stub
    print("  note: warprnnt_pytorch not found; stubbed (training loss, unused here).")


class ESPnetExtractor(ASRProbeExtractor):
    """ESPnet ASR encoder features + a decoder you choose by cost.

    One class covers all three checkpoints; they differ only by tag and by which
    decoders are available, both detected from the loaded model.

    Precision stays float32: the ESPnet frontend runs an STFT inside the model
    graph, and half precision there is a known source of trouble for marginal
    gain — the encoder is not the bottleneck in this pipeline anyway.
    """

    frame_rate_hz = None            # measured by smoke_test
    final_layer_only = True
    grad_mode = "no_grad"

    def __init__(self, name, model_id, decode="ctc_greedy", beam_size=10,
                 ctc_weight=0.3, lm_weight=0.0, maxlenratio=0.0, **kw):
        if decode not in {"ctc_greedy", "transducer_greedy", "beam"}:
            raise ValueError(f"unknown decode mode {decode!r}")
        self.name = name
        self.decode_mode = decode
        self._decode_mode_default = decode      # `set_beam_size(1)` returns here
        # Beam knobs, used only when decode="beam" (or by beam_wer_subset).
        # lm_weight=0.0 is not cosmetic: ESPnet's BeamSearch skips any scorer whose
        # weight is zero, so the RNN-LM is genuinely never run.
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight
        self.lm_weight = lm_weight
        self.maxlenratio = maxlenratio
        kw.setdefault("dtype", torch.float32)
        super().__init__(model_id=model_id, **kw)

    def _load(self):
        # Must precede the ESPnet import: the transducer checkpoint pulls in the
        # RNN-T training loss while the model object is being constructed.
        _ensure_transducer_loss_stub()
        from espnet2.bin.asr_inference import Speech2Text

        self.s2t = Speech2Text.from_pretrained(
            self.model_id,
            device=str(self.device),        # NOT optional: the default is "cpu"
            beam_size=self.beam_size, ctc_weight=self.ctc_weight,
            lm_weight=self.lm_weight, maxlenratio=self.maxlenratio,
            minlenratio=0.0, nbest=1,
        )
        self.asr_model = self.s2t.asr_model.eval()
        self.processor = None
        self._gen_kwargs = {}

        # Is the CTC head trained, or merely constructed? A transducer recipe with
        # ctc_weight=0.0 still builds a `ctc` module, and decoding through an
        # untrained head silently produces garbage rather than an error.
        train_conf = getattr(self.s2t.asr_train_args, "model_conf", {}) or {}
        trained_ctc_weight = float(train_conf.get("ctc_weight", 0.5))
        self._ctc_usable = (getattr(self.asr_model, "ctc", None) is not None
                            and trained_ctc_weight > 0.0)
        self._is_transducer = getattr(self.s2t, "beam_search_transducer", None) is not None
        print(f"  ctc head usable: {self._ctc_usable} "
              f"(training ctc_weight={trained_ctc_weight}) | "
              f"transducer: {self._is_transducer}")

        if self.decode_mode == "ctc_greedy" and not self._ctc_usable:
            raise RuntimeError(
                f"{self.model_id} has no usable CTC head (training ctc_weight="
                f"{trained_ctc_weight}). Use decode='transducer_greedy' for the "
                "RNN-T checkpoint, or decode='beam'.")
        if self.decode_mode == "transducer_greedy":
            if not self._is_transducer:
                raise RuntimeError(f"{self.model_id} is not a transducer model.")
            self.s2t.beam_search_transducer.beam_size = 1        # -> greedy_search

        self.min_input_samples = self._min_samples()

    def _min_samples(self):
        """The input floor this checkpoint's subsampling imposes, in samples.

        ESPnet's encoders refuse a batch outright rather than degrading:
        `check_short_utt` raises `TooShortUttError: has 4 frames and is too short
        for subsampling (it needs more than 11 frames)`. The check is on the
        **padded batch width**, not on the individual utterance — so what matters
        is the longest clip in the batch, and `base` is sorted by duration, which
        puts the shortest clips together in batch 0. On a large AMI slice the
        sixteen shortest are 320-480 samples, so batch 0 pads to 480 -> 4 frames
        and the run dies before it starts. This is a different floor from the one
        `drop_empty_audio` handles: that one is about clips with NO audio, and
        wav2vec2/data2vec need only 400 samples, which those clips clear.

        Both numbers are read off the loaded model rather than assumed, because
        they differ per recipe — the conformer's name (`n_fft512_hop_length...`)
        advertises its own STFT geometry. With the frontend's `center=True` the
        frame count is `n // hop + 1`, so `hop * (limit + 1)` clears the floor
        with two frames to spare.
        """
        hop = 128
        try:
            hop = int(self.asr_model.frontend.stft.hop_length)
        except (AttributeError, TypeError):
            print(f"  note: could not read the frontend hop length; assuming {hop}.")

        limit = 15                       # Conv2dSubsampling8, the strictest variant
        try:
            from espnet.nets.pytorch_backend.nets_utils import check_short_utt
            short, needed = check_short_utt(self.asr_model.encoder.embed, 0)
            if short:
                limit = int(needed)
        except Exception as e:           # noqa: BLE001 — internal API, may move
            print(f"  note: could not query the subsampling floor "
                  f"({type(e).__name__}); assuming {limit} frames.")

        n = hop * (limit + 1)
        print(f"  min input: {n} samples ({1000 * n / self.sampling_rate:.0f} ms) "
              f"— {limit} frames at hop {hop}; shorter clips are zero-padded")
        return n

    @property
    def hidden_size(self):
        return int(self.asr_model.encoder.output_size())

    @property
    def num_encoder_layers(self):
        return len(self.asr_model.encoder.encoders)

    # -------------------------------------------------------------- encode --

    def _encode(self, arrays):
        """Batched frontend + encoder. Returns ((B, T, D), (B,) valid lengths).

        This is the part that scales: unlike `Speech2Text.__call__`, `encode`
        accepts a padded batch and returns true per-utterance frame counts, so no
        length arithmetic is needed and no padding reaches `pool_hidden`.
        """
        speech = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(a, dtype=torch.float32) for a in arrays], batch_first=True
        ).to(self.device)
        lengths = torch.tensor([len(a) for a in arrays], device=self.device)

        enc, enc_lens = self.asr_model.encode(speech, lengths)
        if isinstance(enc, tuple):          # interctc models return (final, intermediates)
            enc = enc[0]
        return enc, enc_lens.long()

    # -------------------------------------------------------------- decode --

    def _ids_to_text(self, token_ids):
        return self.s2t.tokenizer.tokens2text(self.s2t.converter.ids2tokens(token_ids))

    def _decode_ctc_greedy(self, enc, enc_lens):
        """Batched greedy CTC off the joint head: collapse repeats, drop blanks.

        The entire cost is one linear projection plus an argmax, both already on
        the GPU — which is why this path runs at encoder speed. ESPnet reserves
        token id 0 for <blank>.
        """
        best = self.asr_model.ctc.argmax(enc)                   # (B, T)
        hyps = []
        for i, L in enumerate(enc_lens.tolist()):
            ids = [k for k, _ in groupby(best[i, :L].tolist()) if k != 0]
            hyps.append(self._ids_to_text(ids))
        return hyps

    def _decode_transducer_greedy(self, enc, enc_lens):
        """One-hypothesis transducer search, one utterance at a time.

        ESPnet's `BeamSearchTransducer` takes a single (T, D) encoder output, so
        this loops. `_load` set its `beam_size` to 1 after construction, which
        keeps `default_beam_search` but with a single live hypothesis — the
        search algorithm itself is bound at construction, so this is not the
        cheaper `greedy_search` path. Roughly an order of magnitude cheaper than
        the default beam all the same, and the slowest thing here.
        """
        hyps = []
        for i, L in enumerate(enc_lens.tolist()):
            nbest = self.s2t.beam_search_transducer(enc[i, :L])
            hyps.append(self._ids_to_text(nbest[0].yseq[1:]))    # strip blank/SOS
        return hyps

    def _decode_beam(self, enc, enc_lens):
        """Full beam search from an encoder output we already have. Correct, and slow.

        Joint CTC/attention for the attention checkpoints, the transducer beam
        for the RNN-T. `Speech2Text._decode_single_sample` is the step
        `Speech2Text.__call__` runs after its own encoder pass, so decoding this
        way pays the search once and the encoder never twice — which is what
        lets `latency.time_extractor` clock the two halves separately.
        """
        return [self.s2t._decode_single_sample(enc[i, :L])[0][0]
                for i, L in enumerate(enc_lens.tolist())]

    def set_beam_size(self, k):
        """Re-point the loaded search(es) at width `k` without reloading.

        `k <= 1` restores the mode this extractor was constructed with (the one
        the stored configs were decoded with); anything larger switches to
        `decode="beam"` at that width. `Speech2Text` fixes the width at
        construction and `BeamSearch` derives its CTC pre-beam from it, so both
        are rewritten in place. Returns the mode now in force.
        """
        k = int(k)
        bs = getattr(self.s2t, "beam_search", None)
        bst = getattr(self.s2t, "beam_search_transducer", None)
        if k <= 1:
            self.decode_mode = self._decode_mode_default
            if bst is not None:
                bst.beam_size = 1 if self.decode_mode == "transducer_greedy" else self.beam_size
            return self.decode_mode
        self.beam_size = k
        self.decode_mode = "beam"
        if bs is not None:
            bs.beam_size = k
            bs.pre_beam_size = int(1.5 * k)          # BeamSearch's default pre_beam_ratio
            bs.do_pre_beam = (bs.pre_beam_score_key is not None
                              and bs.pre_beam_size < bs.n_vocab and len(bs.part_scorers) > 0)
        if bst is not None:
            bst.beam_size = k
        return self.decode_mode

    def _encode_batch(self, arrays, layer_indices):
        enc, enc_lens = self._encode(arrays)
        return Encoded({layer_indices[0]: enc}, enc_lens, state=(enc, enc_lens))

    def _decode_batch(self, encoded):
        enc, enc_lens = encoded.state
        if self.decode_mode == "ctc_greedy":
            return self._decode_ctc_greedy(enc, enc_lens)
        if self.decode_mode == "transducer_greedy":
            return self._decode_transducer_greedy(enc, enc_lens)
        return self._decode_beam(enc, enc_lens)

    # ---------------------------------------------------------- diagnostics --

    def benchmark_decoding(self, ds, n=16, modes=("ctc_greedy", "beam")):
        """Time each decode mode on `n` utterances and print the real-time factor.

        Answers "how much is the beam search actually costing me" on YOUR machine
        rather than from a table. Anything above RTF ~0.01 is not viable for the
        full corpus.
        """
        self.load()
        saved = self.decode_mode
        try:
            for mode in modes:
                if mode == "ctc_greedy" and not self._ctc_usable:
                    print(f"  {mode:<20} unavailable (no trained CTC head)")
                    continue
                if mode == "transducer_greedy" and not self._is_transducer:
                    print(f"  {mode:<20} unavailable (not a transducer)")
                    continue
                self.decode_mode = mode
                print(f"  {mode:<20}", end=" ")
                self.benchmark(ds, n=n)
        finally:
            self.decode_mode = saved

    def beam_wer_subset(self, ds, n_per_split=300, seed=0):
        """Corpus WER from full beam search on a random sample of each split.

        Reports the published-recipe number alongside the greedy number that
        labels the full corpus, without paying for beam search on 11k utterances.
        Sampling is uniform, so the estimate is unbiased; at n=300 its standard
        error is roughly a tenth of a WER point — enough to confirm the
        checkpoint is behaving.
        """
        self.load()
        saved = self.decode_mode
        self.decode_mode = "beam"
        rows = []
        try:
            for split, d in ds.items():
                idx = np.random.default_rng(seed).choice(
                    d.num_rows, min(n_per_split, d.num_rows), replace=False)
                sub = d.select(sorted(idx.tolist()))
                arrays = [a["array"] for a in sub["audio"]]
                hyps = []
                with self._grad_ctx():
                    for j in range(0, len(arrays), 8):       # bounded padding
                        hyps += self._decode_beam(*self._encode(arrays[j:j + 8]))
                cols = self.scorer.score(sub["text"], hyps)
                err = sum(cols["sub"]) + sum(cols["dele"]) + sum(cols["ins"])
                rows.append({"split": split, "n": len(hyps),
                             "wer_beam": err / max(sum(cols["nref"]), 1),
                             "beam_size": self.beam_size, "ctc_weight": self.ctc_weight,
                             "lm_weight": self.lm_weight})
        finally:
            self.decode_mode = saved
        return pd.DataFrame(rows)

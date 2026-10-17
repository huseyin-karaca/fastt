__all__ = ['NEMO_MIN_SAMPLES', 'ParakeetNeMoExtractor', 'CanaryQwenExtractor']

# %% ../../nbs/02_models.ipynb #c7c440e4
import numpy as np
import torch

from fastt.extractor import ASRProbeExtractor, Encoded


# 0.1 s at 16 kHz. NeMo's `normalize_batch(..., "per_feature")` raises on a
# one-frame mel spectrogram, and its frame count is `floor(n_samples / 160)`, so
# every clip between 10 ms and 20 ms trips it — and takes its whole batch with
# it. 1600 samples is 10 mel frames, which also survives FastConformer's 8x
# subsampling with frames to spare. See `ASRProbeExtractor._pad_short`.
NEMO_MIN_SAMPLES = 1600


class ParakeetNeMoExtractor(ASRProbeExtractor):
    """nvidia/parakeet-tdt-0.6b-v2 through NeMo, batched."""

    name = "parakeet_tdt_0_6b_v2"
    min_input_samples = NEMO_MIN_SAMPLES
    model_id = "nvidia/parakeet-tdt-0.6b-v2"
    frame_rate_hz = 12.5
    final_layer_only = True
    grad_mode = "no_grad"
    use_cuda_graph_decoder = False        # see `_set_cuda_graph_decoder`

    def __init__(self, **kw):
        # NeMo's mel frontend runs an STFT inside the graph; keep it in float32.
        kw.setdefault("dtype", torch.float32)
        super().__init__(**kw)
        self._direct_decode = True

    def _load(self):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_id).to(self.device).eval()
        self.processor = None
        self._gen_kwargs = {}

        # Dither injects noise into the mel frontend. NeMo gates it on
        # self.training, so eval() already disables it, but zeroing it explicitly
        # makes the embeddings bit-reproducible across runs rather than merely
        # reproducible in principle.
        try:
            self.model.preprocessor.featurizer.dither = 0.0
        except AttributeError:
            print("  note: could not zero preprocessor dither; embeddings may vary "
                  "slightly between runs.")

        self._set_cuda_graph_decoder(self.use_cuda_graph_decoder)

    def _set_cuda_graph_decoder(self, enabled):
        """Turn NeMo's CUDA-graph transducer decoder on or off. Default: **off**.

        The label-looping TDT decoder captures a CUDA graph whose static buffers
        are sized `(batch_size, max_time, encoder_dim)` from the first batch it
        sees, and re-captures whenever a later batch is *wider or longer* — see
        `need_reinit` in NeMo's `transducer_decoding/tdt_label_looping.py`. On
        several NeMo / torch / driver combinations that mid-run re-capture faults
        with `CUDA error: an illegal memory access was encountered`, surfacing at
        `batched_hyps.scores.cpu()` inside `batched_hyps_to_hypotheses`.

        That explains the shape of the failure exactly: at `batch_size=1` the
        buffers are allocated once and essentially never grow, so it survives a
        whole split, while a larger batch dies part-way through — at whichever
        batch first holds an utterance longer than anything seen before it.
        LibriSpeech is not length-sorted, so "part-way through" can be anywhere.

        Disabling graphs keeps the same label-looping algorithm and produces the
        same transcripts, on the plain PyTorch path. Decoding gets perhaps
        1.2-1.5x slower; the encoder pass, which dominates, is untouched.

        Once the fault does fire it is not recoverable — the CUDA context is
        dead and the runtime has to be restarted.
        """
        decoder = getattr(getattr(self.model, "decoding", None), "decoding", None)
        try:
            from nemo.collections.common.parts.optional_cuda_graphs import (
                WithOptionalCudaGraphs)
            switch = (WithOptionalCudaGraphs.enable_cuda_graphs_recursive if enabled
                      else WithOptionalCudaGraphs.disable_cuda_graphs_recursive)
            switch(self.model, attribute_path="decoding.decoding")
        except ImportError:                       # older NeMo: poke the decoder itself
            method = "maybe_enable_cuda_graphs" if enabled else "disable_cuda_graphs"
            getattr(decoder, method, lambda: None)()

        # `cuda_graphs_mode` is None when graphs are off, so this reports what
        # actually happened rather than what we asked for.
        mode = getattr(getattr(decoder, "decoding_computer", None),
                       "cuda_graphs_mode", None)
        print(f"  CUDA-graph decoder: {mode.value if mode else 'off'}"
              + ("" if enabled else " (avoids the mid-run graph re-capture fault)"))

    @property
    def hidden_size(self):
        enc = self.model.encoder
        for attr in ("d_model", "_feat_out", "output_dim"):
            if hasattr(enc, attr):
                return int(getattr(enc, attr))
        return 1024                                       # config.json: d_model 1024

    @property
    def num_encoder_layers(self):
        return len(self.model.encoder.layers)

    def set_beam_size(self, k):
        """Switch NeMo's decoding strategy between greedy and a `k`-wide beam.

        Rebuilds `model.decoding` through `change_decoding_strategy`, which is
        what `_decode_rnnt` then calls — so the timing notebook can widen the
        TDT beam without reloading. Returns the strategy now in force. NeMo's
        TDT beam support is version-dependent; a rejected strategy raises and
        `latency.set_decoding` reports the model as greedy-only.
        """
        import copy
        from omegaconf import open_dict

        k = int(k)
        cfg = copy.deepcopy(self.model.cfg.decoding)
        with open_dict(cfg):
            if k <= 1:
                cfg.strategy = "greedy_batch"
            else:
                cfg.strategy = "beam"
                cfg.beam.beam_size = k
                cfg.beam.return_best_hypothesis = True
        self.model.change_decoding_strategy(cfg, verbose=False)
        if k <= 1:
            self._set_cuda_graph_decoder(self.use_cuda_graph_decoder)
        self._direct_decode = True
        return str(cfg.strategy)

    def _decode_rnnt(self, enc_bdt, enc_lens):
        """TDT greedy decoding straight off an encoder output we already have.

        NeMo's return signature for `rnnt_decoder_predictions_tensor` changed
        across versions — older releases return `(best, all_hyps)`, newer ones a
        bare list of `Hypothesis`. Both are normalised to a list of strings.
        Returns `None` if the direct call is rejected, and the caller falls back
        to `model.transcribe` (one extra encoder pass, identical results).
        """
        if self._direct_decode:
            try:
                out = self.model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc_bdt, encoded_lengths=enc_lens,
                    return_hypotheses=False)
                if isinstance(out, tuple):
                    out = out[0]
                return [h.text if hasattr(h, "text") else str(h) for h in out]
            except (TypeError, AttributeError) as e:
                self._direct_decode = False
                print(f"  note: direct RNNT decoding rejected ({type(e).__name__}: {e}). "
                      "Falling back to model.transcribe — same results, one extra "
                      "encoder pass per batch.")
        return None

    def _encode_batch(self, arrays, layer_indices):
        sig = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(a, dtype=torch.float32) for a in arrays], batch_first=True
        ).to(self.device)
        sig_len = torch.tensor([len(a) for a in arrays], device=self.device)

        feats, feat_len = self.model.preprocessor(input_signal=sig, length=sig_len)
        enc, enc_len = self.model.encoder(audio_signal=feats, length=feat_len)
        # `enc` is (B, D, T). Clamping guarantees no length can point past the
        # last frame, which would index out of bounds inside the decoder.
        # The lower clamp matters for the same reason `min_input_samples`
        # does: a zero-length encoder output makes `pool_hidden` divide by
        # zero and quietly fills the pooled column with NaN.
        enc_len = enc_len.long().clamp(min=1, max=enc.shape[-1])
        # The waveforms ride along for the `transcribe` fallback only.
        return Encoded({layer_indices[0]: enc.transpose(1, 2)}, enc_len,
                       state={"enc": enc, "enc_len": enc_len, "arrays": list(arrays)})

    def _decode_batch(self, encoded):
        st = encoded.state
        hyps = self._decode_rnnt(st["enc"], st["enc_len"])
        if hyps is None:
            out = self.model.transcribe(
                audio=[np.asarray(a, dtype=np.float32) for a in st["arrays"]],
                batch_size=len(st["arrays"]), verbose=False)
            hyps = [h.text if hasattr(h, "text") else str(h) for h in out]
        return hyps

# %% ../../nbs/02_models.ipynb #078e4205
import os
import tempfile

import soundfile as sf


class CanaryQwenExtractor(ASRProbeExtractor):
    """nvidia/canary-qwen-2.5b FastConformer encoder features + SALM transcription."""

    name = "canary_qwen_2_5b"
    min_input_samples = NEMO_MIN_SAMPLES
    model_id = "nvidia/canary-qwen-2.5b"
    frame_rate_hz = 12.5
    final_layer_only = True
    grad_mode = "no_grad"
    prompt = "Transcribe the following:"

    def _load(self):
        from nemo.collections.speechlm2.models import SALM

        self.model = SALM.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = None
        self._gen_kwargs = {}
        self._perception = self.model.perception
        self._scratch = tempfile.mkdtemp(prefix="canary_")

    @property
    def hidden_size(self):
        enc = self._perception.encoder
        for attr in ("d_model", "_feat_out", "output_dim"):
            if hasattr(enc, attr):
                return int(getattr(enc, attr))
        return 1024                                    # documented for this model

    @property
    def num_encoder_layers(self):
        return len(self._perception.encoder.layers)

    def _encode(self, arrays):
        """Preprocessor -> FastConformer encoder. Returns ((B, T, D), lengths)."""
        sig = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(a, dtype=torch.float32) for a in arrays], batch_first=True
        ).to(self.device)
        sig_len = torch.tensor([len(a) for a in arrays], device=self.device)

        feats, feat_len = self._perception.preprocessor(input_signal=sig, length=sig_len)
        enc, enc_len = self._perception.encoder(audio_signal=feats, length=feat_len)
        # NeMo conformers emit channels-first (B, D, T); pool_hidden wants (B, T, D).
        if enc.shape[1] == self.hidden_size and enc.shape[1] != enc.shape[2]:
            enc = enc.transpose(1, 2)
        return enc, enc_len.long().clamp(min=1, max=enc.shape[1])

    def _transcribe(self, arrays):
        """Write the batch to disk, prompt SALM, then delete the files."""
        paths = []
        for i, a in enumerate(arrays):
            p = os.path.join(self._scratch, f"{i}.wav")
            sf.write(p, np.asarray(a, dtype=np.float32), self.sampling_rate)
            paths.append(p)
        try:
            prompts = [[{
                "role": "user",
                "content": f"{self.prompt} {self.model.audio_locator_tag}",
                "audio": [p],
            }] for p in paths]
            ids = self.model.generate(prompts=prompts, max_new_tokens=256)
            return [self.model.tokenizer.ids_to_text(row.cpu()) for row in ids]
        finally:
            for p in paths:
                os.unlink(p)

    def _encode_batch(self, arrays, layer_indices):
        hidden, lengths = self._encode(arrays)
        # SALM re-reads the audio from disk to transcribe; the decoder half is
        # the whole `generate`, so the waveforms are the decoder state.
        return Encoded({layer_indices[0]: hidden}, lengths, state=list(arrays))

    def _decode_batch(self, encoded):
        return self._transcribe(encoded.state)

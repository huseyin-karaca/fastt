__all__ = ['ParakeetExtractor']

# %% ../../nbs/02_models.ipynb #1b033d22
from transformers import AutoModelForTDT, AutoProcessor

from fastt.extractor import ASRProbeExtractor, Encoded


class ParakeetExtractor(ASRProbeExtractor):
    """nvidia/parakeet-tdt-0.6b-v3 FastConformer features + TDT transcription.

    Valid frame counts come from the encoder's own downsampled attention mask
    rather than from subsampling arithmetic, which avoids an off-by-one at the
    convolution boundary. Passing `attention_mask` is not optional for batched
    inference here — the HF docs are explicit that omitting it is subtly wrong.

    Two things about this head are easy to trip over:

    * `generate` returns `ParakeetRNNTGenerateOutput`, not a bare id tensor.
      `_token_ids` in the base class unwraps it.
    * It prints `Using the model-agnostic default max_length (=N)` on every
      batch. That warning is harmless here: Parakeet overrides
      `_prepare_generated_length` and derives `max_length` from the encoder
      length (`max_symbols_per_step * encoder_frames`), but the warning is
      emitted from the *caller*, which never sees that the override ran.
      Nothing is truncated.
    """

    name = "parakeet_tdt_0_6b_v3"
    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    frame_rate_hz = 12.5

    def _load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForTDT.from_pretrained(
            self.model_id, dtype=self.dtype
        ).to(self.device).eval()
        self._encoder = getattr(self.model, "encoder", None) or self.model.get_encoder()
        self._gen_kwargs = {}

        # Parakeet's `_prepare_model_inputs` calls the encoder itself and
        # overwrites whatever `encoder_outputs` we hand in, so reuse buys
        # nothing here — and the stray kwarg gets forwarded down into
        # `get_audio_features(**kwargs)`. Ask for the second pass honestly.
        self._reuse_encoder = False

    @property
    def _enc_cfg(self):
        return self.model.config.encoder_config

    @property
    def hidden_size(self):
        return self._enc_cfg.hidden_size

    @property
    def num_encoder_layers(self):
        return self._enc_cfg.num_hidden_layers

    def _encode_batch(self, arrays, layer_indices):
        inputs = self.processor(
            audio=arrays, sampling_rate=self.sampling_rate, return_tensors="pt",
            padding="longest", return_attention_mask=True,
        )
        feats = inputs["input_features"].to(self.device, self.dtype, non_blocking=True)
        mask = inputs["attention_mask"].to(self.device, non_blocking=True)
        enc = self._encoder(input_features=feats, attention_mask=mask,
                            output_hidden_states=True)

        enc_mask = getattr(enc, "attention_mask", None)
        if enc_mask is not None:
            lengths = enc_mask.sum(-1).long()
        else:                                       # fall back to the subsampling ratio
            f = self._enc_cfg.subsampling_factor
            lengths = ((mask.sum(-1) - 1) // f + 1).long()
        return Encoded({a: enc.hidden_states[a] for a in layer_indices}, lengths,
                       state={"input_features": feats, "attention_mask": mask,
                              "encoder_outputs": enc})

    def _decode_batch(self, encoded):
        # `generate` runs the encoder again (`_reuse_encoder = False`), so this
        # model's decoder clock includes a second encoder pass.
        st = encoded.state
        ids = self._generate({"input_features": st["input_features"],
                              "attention_mask": st["attention_mask"]}, st["encoder_outputs"])
        # `ids` is already a (B, L) tensor: `_generate` unwraps the ModelOutput.
        return self.processor.batch_decode(ids, skip_special_tokens=True)

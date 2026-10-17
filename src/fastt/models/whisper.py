__all__ = ['WhisperExtractor']

# %% ../../nbs/02_models.ipynb #29a5e9e7
import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from fastt.extractor import ASRProbeExtractor, Encoded


class WhisperExtractor(ASRProbeExtractor):
    """openai/whisper-large-v3 encoder features + greedy transcription."""

    name = "whisper_large_v3"
    model_id = "openai/whisper-large-v3"
    frame_rate_hz = 50.0
    samples_per_frame = 320          # 160-sample mel hop x 2 (conv stride)
    max_input_seconds = 30

    def _load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, dtype=self.dtype
        ).to(self.device).eval()
        self._gen_kwargs = dict(
            language="en",
            task="transcribe",
            num_beams=1,             # greedy: the model card's setting, and several
                                     # times faster than beam search
            max_new_tokens=200,
            return_timestamps=False,
        )

    @property
    def hidden_size(self):
        return self.model.config.d_model

    @property
    def num_encoder_layers(self):
        return self.model.config.encoder_layers

    def _encode_batch(self, arrays, layer_indices):
        inputs = self.processor(arrays, sampling_rate=self.sampling_rate,
                                return_tensors="pt")
        feats = inputs.input_features.to(self.device, self.dtype, non_blocking=True)
        enc = self.model.get_encoder()(feats, output_hidden_states=True)

        # The encoder emits a fixed 1500 frames; recover the real count from the
        # audio itself so the padding never reaches the pooling.
        max_frames = enc.last_hidden_state.shape[1]
        lengths = torch.tensor(
            [max(1, min(max_frames, int(np.ceil(len(a) / self.samples_per_frame))))
             for a in arrays], dtype=torch.long)
        return Encoded({a: enc.hidden_states[a] for a in layer_indices}, lengths,
                       state={"input_features": feats, "encoder_outputs": enc})

    def _decode_batch(self, encoded):
        st = encoded.state
        ids = self._generate({"input_features": st["input_features"]}, st["encoder_outputs"])
        return self.processor.batch_decode(ids, skip_special_tokens=True)

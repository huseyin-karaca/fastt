__all__ = ['MFCCExtractor']

# %% ../../nbs/02_models.ipynb #3ef0c865
import pandas as pd
import torch

from fastt.extractor import ASRProbeExtractor, Encoded
from fastt.hub import FasttHub


class MFCCExtractor(ASRProbeExtractor):
    """13-d MFCC features, pooled and probed exactly like a learned encoder.

    Parameters
    ----------
    target_model : which model's config supplies the WER targets, e.g.
        `"wav2vec2"`. Its `model_<name>` config must already be on the Hub.
    n_mfcc : number of cepstral coefficients; this is `hidden_size`.

    Layer indexing: there is only one representation, so `num_encoder_layers`
    is 1 and the pooled columns come out as `pool_l1_<pooling>`.
    """

    name = "mfcc_pooled"
    model_id = "torchaudio MFCC (13, 25ms/10ms)"
    frame_rate_hz = 100.0                 # 10 ms hop
    final_layer_only = True
    emits_transcription = False
    grad_mode = "no_grad"

    def __init__(self, target_model="wav2vec2", n_mfcc=13, hop=160, n_fft=400,
                 n_mels=40, **kw):
        kw.setdefault("dtype", torch.float32)
        self.target_model = target_model
        self.n_mfcc, self.hop, self.n_fft, self.n_mels = n_mfcc, hop, n_fft, n_mels
        super().__init__(**kw)

    def _load(self):
        import torchaudio

        self.model = torchaudio.transforms.MFCC(
            sample_rate=self.sampling_rate, n_mfcc=self.n_mfcc,
            melkwargs=dict(n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mels),
        ).to(self.device)
        self.processor = None

    @property
    def hidden_size(self):
        return self.n_mfcc

    @property
    def num_encoder_layers(self):
        return 1

    def _encode_batch(self, arrays, layer_indices):
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(a, dtype=torch.float32) for a in arrays], batch_first=True
        ).to(self.device)
        out = self.model(padded).transpose(1, 2)              # (B, T, n_mfcc)

        # With center=True (the torchaudio default) a signal of length L yields
        # L // hop + 1 frames; clamp for rounding at the batch maximum.
        lengths = torch.tensor(
            [min(len(a) // self.hop + 1, out.shape[1]) for a in arrays], dtype=torch.long)
        return Encoded({layer_indices[0]: out}, lengths)     # nothing to decode

    def attach_targets(self, ds_feat, targets):
        """Join WER columns from `self.target_model`'s config onto `ds_feat` by `id`.

        Joining on `id` rather than trusting row order: both datasets derive from
        the same sorted `base`, so the orders do agree today, but a silent
        misalignment here would produce a plausible-looking and completely wrong
        probe result.
        """
        hub = FasttHub(self.repo_id, spec=self.spec)
        src = hub.load_model(self.target_model)
        print(f"borrowing {list(targets)} from {self.spec.model_config(self.target_model)}")

        out = {}
        for split, d in ds_feat.items():
            ref = pd.DataFrame({"id": src[split]["id"],
                                **{t: src[split][t] for t in targets}}).set_index("id")
            ids = d["id"]
            missing = set(ids) - set(ref.index)
            if missing:
                raise KeyError(
                    f"{split}: {len(missing)} ids missing from "
                    f"{self.spec.model_config(self.target_model)} "
                    f"(e.g. {sorted(missing)[:3]})")
            aligned = ref.loc[ids]
            for t in targets:
                d = d.add_column(t, aligned[t].tolist())
            out[split] = d
        return type(ds_feat)(out)

    def _prepare_probe(self, ds_feat, targets):
        """Hook called by `run()` between extraction and probing."""
        return self.attach_targets(ds_feat, targets)

__all__ = ['HFCtcExtractor', 'ctc_prefix_beam_search', 'Wav2Vec2BaseExtractor', 'Wav2Vec2LargeExtractor', 'Data2VecExtractor', 'Wav2Vec2Large960Lv60Extractor']

# %% ../../nbs/02_models.ipynb #f9dc053c
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor

from fastt.extractor import ASRProbeExtractor, Encoded


class HFCtcExtractor(ASRProbeExtractor):
    """Any CTC speech checkpoint that `transformers` exposes as `AutoModelForCTC`.

    Frame counts come from the model's own `_get_feat_extract_output_lengths`,
    which walks the convolutional stack — exact, and identical across wav2vec2
    base/large and data2vec because all three share the conv geometry
    (kernels [10,3,3,3,3,2,2], strides [5,2,2,2,2,2,2]). That is also why every
    subset in the repo lines up at 50 frames/s.

    Unlike the NeMo/ESPnet models, `output_hidden_states=True` gives the whole
    stack, so `ExtractConfig(layers=(0, 6, 12))` works here.
    """

    frame_rate_hz = 50.0

    def __init__(self, **kw):
        # float32 weights + bf16 autocast in the forward; see the section note.
        kw.setdefault("dtype", torch.float32)
        super().__init__(**kw)

    def _load(self):
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device).eval()

        expected = self.processor.feature_extractor.sampling_rate
        if expected != self.sampling_rate:
            raise ValueError(f"{self.model_id} expects {expected} Hz, not {self.sampling_rate}")

        self._pass_mask = bool(self.processor.feature_extractor.return_attention_mask)
        print(f"  checkpoint return_attention_mask={self._pass_mask} -> "
              f"{'passing' if self._pass_mask else 'NOT passing'} attention_mask to the model")

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def num_encoder_layers(self):
        return self.model.config.num_hidden_layers

    def _encode_batch(self, arrays, layer_indices):
        lens = torch.tensor([len(a) for a in arrays])
        out_lens = self.model._get_feat_extract_output_lengths(lens).long()

        inputs = self.processor(
            arrays, return_tensors="pt", padding="longest",
            sampling_rate=self.sampling_rate, return_attention_mask=True,
        )
        input_values = inputs.input_values.to(self.device, non_blocking=True)

        model_kwargs = {}
        if self._pass_mask:
            model_kwargs["attention_mask"] = inputs.attention_mask.to(
                self.device, non_blocking=True)

        # bf16 autocast on CUDA only; silently disabled on CPU.
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(self.device.type == "cuda")):
            out = self.model(input_values, output_hidden_states=True, **model_kwargs)

        # The CTC head is part of the encoder pass: the logits ARE the decoder's
        # input, so the decoder half is only the search over them.
        return Encoded({a: out.hidden_states[a] for a in layer_indices}, out_lens,
                       state=out.logits)

    def _decode_batch(self, encoded):
        return self._decode_logits(encoded.state, encoded.lengths)

    # CTC beam width. None — the default, and what every stored config was
    # decoded with — is the greedy argmax. `latency.set_decoding` sets it to
    # price what a CTC model pays once it is decoded with anything better.
    ctc_beam_width = None

    def _decode_logits(self, logits, out_lens):
        """Greedy argmax, or a prefix beam search when `ctc_beam_width` is set.

        Slicing to :L drops padded frames from the decode; pool_hidden does the
        same for the embeddings via `out_lens`.
        """
        if not self.ctc_beam_width:
            pred_ids = logits.float().argmax(dim=-1).cpu()      # greedy CTC path
            return [self.processor.decode(pred_ids[i, :L])
                    for i, L in enumerate(out_lens.tolist())]
        logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        blank = self.model.config.pad_token_id
        # `group_tokens=False`: the search already collapsed repeats and dropped
        # blanks, and a genuine double letter must survive the string decode.
        return [self.processor.tokenizer.decode(
                    ctc_prefix_beam_search(logp[i, :L], int(self.ctc_beam_width), blank=blank),
                    group_tokens=False, skip_special_tokens=True)
                for i, L in enumerate(out_lens.tolist())]


def ctc_prefix_beam_search(logp, beam_width, blank=0, token_min_logp=-5.0):
    """CTC prefix beam search over one utterance's log-posteriors, no language model.

    `logp` is `(T, V)`; returns the best label sequence as a list of ids, repeats
    collapsed and blanks removed. Hannun et al.'s algorithm: each live prefix
    carries the probability of ending in a blank and of ending in its last
    label, so extending by a repeated label and merging through a blank stay
    distinct. Pure Python on the CPU, frame by frame — the cost a CTC model
    pays the moment its decoder is anything but an argmax, and deliberately
    not a compiled one: `pyctcdecode` pins `numpy<2` and cannot be installed
    beside the rest of group A.

    `token_min_logp` skips tokens far below the frame's mass (the frame's
    argmax is always kept), which is `pyctcdecode`'s default pruning.
    """
    import math

    neg = -math.inf

    def lse(*xs):
        m = max(xs)
        return neg if m == neg else m + math.log(sum(math.exp(x - m) for x in xs))

    beams = {(): (0.0, neg)}                       # prefix -> (log p_blank, log p_label)
    for row in logp:
        top = int(row.argmax())
        cands = [c for c in range(len(row)) if row[c] >= token_min_logp or c == top]
        nxt = {}
        for prefix, (pb, pl) in beams.items():
            total = lse(pb, pl)
            for c in cands:
                p = float(row[c])
                if c == blank:
                    b, l = nxt.get(prefix, (neg, neg))
                    nxt[prefix] = (lse(b, total + p), l)
                    continue
                ext = prefix + (c,)
                b, l = nxt.get(ext, (neg, neg))
                if prefix and c == prefix[-1]:
                    nxt[ext] = (b, lse(l, pb + p))          # only through a blank
                    b0, l0 = nxt.get(prefix, (neg, neg))
                    nxt[prefix] = (b0, lse(l0, pl + p))    # the repeat collapses
                else:
                    nxt[ext] = (b, lse(l, total + p))
        beams = dict(sorted(nxt.items(), key=lambda kv: lse(*kv[1]), reverse=True)[:beam_width])
    return list(max(beams.items(), key=lambda kv: lse(*kv[1]))[0])


class Wav2Vec2BaseExtractor(HFCtcExtractor):
    """facebook/wav2vec2-base-960h — 12 layers, d=768. Card: 3.4 / 8.6 WER."""
    name = "wav2vec2"
    model_id = "facebook/wav2vec2-base-960h"


class Wav2Vec2LargeExtractor(HFCtcExtractor):
    """facebook/wav2vec2-large-960h — 24 layers, d=1024. Card: 2.8 / 6.3 WER.

    The sibling checkpoint `facebook/wav2vec2-large-960h-lv60-self` reports
    1.9 / 3.9 and was trained WITH an attention mask. Swapping to it needs no
    code change — `model_id="facebook/wav2vec2-large-960h-lv60-self"` — because
    the mask policy is read from the checkpoint.
    """
    name = "wav2vec2_large"
    model_id = "facebook/wav2vec2-large-960h"


class Data2VecExtractor(HFCtcExtractor):
    """facebook/data2vec-audio-base-960h — 12 layers, d=768. Card: 2.5 / 6.3 WER.

    Ships a `Wav2Vec2Processor` and a layer-norm feature encoder, so it *does*
    want the attention mask at the model input; without it the padded region
    leaks into valid frames through self-attention.
    """
    name = "data2vec"
    model_id = "facebook/data2vec-audio-base-960h"


class Wav2Vec2Large960Lv60Extractor(HFCtcExtractor):
    """facebook/wav2vec2-large-960h-lv60 — same size as §4.1.2, different pretraining.

    24 layers and d=1024, exactly like `wav2vec2-large-960h`, so the two are
    architecturally interchangeable and differ only in what they were
    pretrained on: LV-60k (Libri-Light, ~60k h of unlabelled read speech)
    rather than the 960 h of LibriSpeech itself. That makes the pair a
    controlled comparison of pretraining data at fixed capacity — and on
    out-of-domain speech like AMI it is the more interesting of the two, since
    the wider pretraining set is exactly what is supposed to help there.
    """

    name = "wav2vec2_large_lv60"
    model_id = "facebook/wav2vec2-large-960h-lv60"

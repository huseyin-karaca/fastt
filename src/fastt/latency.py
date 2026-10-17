__all__ = ['EXTRACTORS', 'time_extractor', 'set_decoding', 'install_group', 'load_extractor', 'timing_summary', 'decoding_table', 'ensemble_cost', 'timing_table', 'routing_latency']

import time

import numpy as np
import pandas as pd

from fastt.core import SAMPLING_RATE


def time_extractor(extractor, ds, n=32, batch_size=1, warmup=3, split=None,
                   verbose=True, indices=None, seed=0, decoding="greedy", score=True):
    """Per-utterance encoder and decoder wall-clock for one model. Returns a frame.

    Parameters
    ----------
    extractor : an `ASRProbeExtractor` subclass instance, already constructed.
    ds : a `base` dataset or `DatasetDict`; `split` picks one.
    n : utterances to time, after `warmup` untimed ones. The first calls into a
        freshly loaded model pay for CUDA context and kernel autotuning and are
        not representative of anything.
    batch_size : keep at 1 for online latency. Raise it only to quote throughput,
        and label it as such.
    decoding : a label for the decoder configuration in force (`set_decoding`
        returns it), stored on every row so one model's greedy and beam clocks
        can sit in the same frame.
    score : also count word errors on the timed clips (`err`, `nref`), so a
        wider beam can be shown to buy accuracy and not only cost time.

    Notes
    -----
    Timing is split by calling the extractor's own `_encode_and_decode` and
    measuring around a `torch.cuda.synchronize()`, so the numbers are real device
    time and not the moment the kernel was queued. Without the synchronise every
    figure here would be an underestimate, and the encoder would look free.
    """
    import torch

    if hasattr(ds, "keys"):
        ds = ds[split or list(ds.keys())[0]]
    extractor.load()
    layers = extractor._resolve_layers((-1,))
    score = score and extractor.emits_transcription and "text" in ds.column_names

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # A RANDOM sample of clips. `base` is duration-sorted, so the first `n`
    # rows are the shortest clips in the corpus and would flatter every model;
    # the sample is seeded so two models are timed on the same clips.
    total = min(warmup + n, len(ds))
    if indices is None:
        indices = np.random.default_rng(seed).choice(len(ds), total, replace=False)
    indices = [int(i) for i in indices][:total]

    rows = []
    for k, i in enumerate(indices):
        batch = ds[i: i + batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        extractor._clock = {}                   # arms `_mark_encoded`
        sync(); t0 = time.perf_counter()
        hidden, lengths, hyps = extractor._encode_and_decode(arrays, layers)
        sync(); t1 = time.perf_counter()
        marked = extractor._clock.get("encoded")
        extractor._clock = None
        if k < warmup:
            continue
        b = max(len(arrays), 1)
        err = nref = np.nan
        if score:
            cols = extractor.scorer.score(batch["text"], hyps)
            err = float(sum(cols["sub"]) + sum(cols["dele"]) + sum(cols["ins"])) / b
            nref = float(sum(cols["nref"])) / b
        rows.append({
            "model": extractor.name,
            "decoding": decoding,
            "row": i,
            "batch_size": batch_size,
            "n_frames": float(np.mean([int(x) for x in lengths.cpu().tolist()])),
            "audio_seconds": float(np.mean([len(a) for a in arrays])) / SAMPLING_RATE,
            # One call returns both, so this is the honest end-to-end figure.
            "total_seconds": (t1 - t0) / b,
            # The split, where the model marked it. `nan` means the subclass
            # never called `_mark_encoded` and only the total is trustworthy.
            "encoder_seconds": (marked - t0) / b if marked else np.nan,
            "decoder_seconds": (t1 - marked) / b if marked else np.nan,
            "err": err,
            "nref": nref,
        })
    out = pd.DataFrame(rows)
    if verbose and len(out):
        m = out["total_seconds"].mean()
        e = out["encoder_seconds"].mean()
        rtf = m / max(out["audio_seconds"].mean(), 1e-9)
        split = f"encoder {e * 1000:.1f} + decoder {(m - e) * 1000:.1f}" if np.isfinite(e) \
            else "no encoder/decoder mark"
        wer = (f", WER {out['err'].sum() / max(out['nref'].sum(), 1):.3f}"
               if score and out["nref"].sum() > 0 else "")
        print(f"{extractor.name:<24} {decoding:<8} {m * 1000:8.1f} ms/utt at batch={batch_size} "
              f"({split}; RTF {rtf:.3f}{wer}, {len(out)} clips after {warmup} warmup)")
    return out


def set_decoding(extractor, beam=1):
    """Point a loaded extractor's decoder at beam width `beam`; 1 is greedy.

    Returns the label the timing records carry (`"greedy"` or `"beam<k>"`), or
    `None` when this model has no beam to widen — the caller then times it once,
    greedy, and says so. Extraction never calls this: every stored config was
    decoded greedily, and `beam=1` puts each model back exactly there.

    What "beam" means per family, since the knob is not the same object twice:

    * HF CTC (wav2vec2, data2vec): `ctc_prefix_beam_search` of that width over
      the CTC posteriors, no language model. Greedy is the argmax.
    * Whisper: `num_beams` in `generate`.
    * ESPnet: `Speech2Text`'s joint CTC/attention beam (the RNN-T's transducer
      beam), re-pointed in place by `set_beam_size`.
    * NeMo Parakeet: `change_decoding_strategy` to `beam` at that width, when
      the installed NeMo accepts it for TDT.
    * HF Parakeet-TDT and Canary-Qwen expose no beam here: `None`.
    """
    beam = int(beam)
    label = "greedy" if beam <= 1 else f"beam{beam}"
    if hasattr(extractor, "ctc_beam_width"):
        extractor.ctc_beam_width = None if beam <= 1 else beam
        return label
    if type(extractor).__name__ == "WhisperExtractor":
        extractor.load()
        extractor._gen_kwargs["num_beams"] = beam
        return label
    if hasattr(extractor, "set_beam_size"):
        extractor.load()
        try:
            extractor.set_beam_size(beam)
        except Exception as e:                       # NeMo: strategy rejected
            print(f"  {extractor.name}: beam {beam} not available "
                  f"({type(e).__name__}: {e}); greedy only")
            extractor.set_beam_size(1)
            return None
        return label
    return label if beam <= 1 else None


# Every extractor by the name the Hub knows it under, with the install group it
# needs. The three groups cannot share a kernel (§1.2–§1.4), which is why the
# timing notebook is sectioned by group and `load_extractor` is by name.
EXTRACTORS = {
    "wav2vec2":             ("A", "fastt.models.hf_ctc",   "Wav2Vec2BaseExtractor", {}),
    "wav2vec2_large":       ("A", "fastt.models.hf_ctc",   "Wav2Vec2LargeExtractor", {}),
    "wav2vec2_large_lv60":  ("A", "fastt.models.hf_ctc",   "Wav2Vec2Large960Lv60Extractor", {}),
    "data2vec":             ("A", "fastt.models.hf_ctc",   "Data2VecExtractor", {}),
    "whisper_large_v3":     ("A", "fastt.models.whisper",  "WhisperExtractor", {}),
    "parakeet_tdt_0_6b_v3": ("A", "fastt.models.parakeet", "ParakeetExtractor", {}),
    "mfcc_pooled":          ("A", "fastt.models.mfcc",     "MFCCExtractor", {}),
    "parakeet_tdt_0_6b_v2": ("B", "fastt.models.nemo",     "ParakeetNeMoExtractor", {}),
    "canary_qwen_2_5b":     ("B", "fastt.models.nemo",     "CanaryQwenExtractor", {}),
    # One ESPnet class serves three checkpoints; the constructor arguments are
    # the ones §4.4 uses, so the timed model is the extracted one.
    "espnet_transformer":   ("C", "fastt.models.espnet",   "ESPnetExtractor", {
        "name": "espnet_transformer", "decode": "ctc_greedy",
        "model_id": "espnet/Shinji_Watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"}),
    "espnet_conformer":     ("C", "fastt.models.espnet",   "ESPnetExtractor", {
        "name": "espnet_conformer", "decode": "ctc_greedy",
        "model_id": "espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer6_n_fft512_hop_length2-truncated-a63357"}),
    "espnet_transducer":    ("C", "fastt.models.espnet",   "ESPnetExtractor", {
        "name": "espnet_transducer", "decode": "transducer_greedy",
        "model_id": "espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp"}),
}


def install_group(name):
    """`"A"` (transformers), `"B"` (NeMo) or `"C"` (ESPnet) for a model name."""
    return EXTRACTORS[name][0]


def load_extractor(name, **kw):
    """Construct the extractor registered under `name`, importing only its module."""
    import importlib

    _group, module, cls, fixed = EXTRACTORS[name]
    return getattr(importlib.import_module(module), cls)(**{**fixed, **kw})


def timing_summary(frames, decoding=None):
    """Mean per-utterance seconds per model (and decoder) from `time_extractor` frames.

    One row per model and `decoding` label: `encoder_seconds`,
    `decoder_seconds`, `total_seconds`, `audio_seconds`, `rtf` (total over
    audio duration), `dec_over_enc` (decoder seconds per encoder second — the
    ratio the routing argument rests on), `wer` on the timed clips where they
    were scored, and `n_clips`. Frames without a `decoding` column are read as
    greedy. `decoding=` keeps one label only, which is what `timing_table`
    wants: one row per model.
    """
    df = pd.concat(list(frames), ignore_index=True) if not isinstance(frames, pd.DataFrame) \
        else frames.copy()
    if "decoding" not in df.columns:
        df["decoding"] = "greedy"
    if decoding is not None:
        df = df[df["decoding"] == decoding]
    g = df.groupby(["model", "decoding"], sort=False)
    out = g[["encoder_seconds", "decoder_seconds", "total_seconds", "audio_seconds"]].mean()
    out["rtf"] = out["total_seconds"] / out["audio_seconds"].clip(lower=1e-9)
    out["dec_over_enc"] = out["decoder_seconds"] / out["encoder_seconds"].clip(lower=1e-9)
    if "err" in df.columns:
        sums = g[["err", "nref"]].sum(min_count=1)
        out["wer"] = sums["err"] / sums["nref"].where(sums["nref"] > 0)
    out["n_clips"] = g.size()
    return out.reset_index()


def decoding_table(per_model, members=None):
    """Encoder vs decoder milliseconds per model, one column block per decoder.

    `per_model` is `timing_summary` output over every decoder label. The
    encoder is the same pass whatever the decoder, so it is reported once
    (averaged over the labels, which only differ by noise); each decoder label
    then gets `decode_ms`, `x_encoder` (decoder over encoder) and `wer`.
    """
    t = per_model.copy()
    if members is not None:
        t = t[t["model"].isin(members)]
    labels = list(dict.fromkeys(t["decoding"]))
    enc = t.groupby("model", sort=False)["encoder_seconds"].mean() * 1e3
    out = pd.DataFrame({"model": enc.index, "encode_ms": enc.values}).set_index("model")
    for lab in labels:
        sub = t[t["decoding"] == lab].set_index("model")
        out[f"{lab}_decode_ms"] = sub["decoder_seconds"] * 1e3
        out[f"{lab}_x_encoder"] = sub["decoder_seconds"] * 1e3 / out["encode_ms"]
        if "wer" in sub.columns:
            out[f"{lab}_wer"] = sub["wer"]
    return out.reset_index()


def ensemble_cost(per_model, members, decoding, shares=None, selector_seconds=0.0,
                  feature_only=()):
    """Per-utterance cost of choosing among `members` before vs after decoding.

    Parameters
    ----------
    per_model : `timing_summary` output.
    members : the base models in the ensemble.
    decoding : the decoder label to price every member at.
    shares : `{member: fraction}` — how often the selector picks each member.
        Uniform when None. A selector pays the decoder of the member it picked.
    selector_seconds : the selector's own forward pass per utterance.
    feature_only : encoders every selector runs but never decodes (MFCC).

    Returns one row per member plus three systems, in seconds per utterance:
    `encode`, `decode`, `selector`, `total`, `x_best_single` (total relative
    to the cheapest member) and `x_after` (total relative to deciding after
    decoding — the saving the encoder-side selector buys):

    * `decode all, then choose` — every encoder and every decoder, the cost of
      any method that needs the members' transcripts to decide (ROVER, a
      confidence vote, an oracle over hypotheses).
    * `choose from encoders (FASTT)` — every encoder, the selector, and the
      ONE decoder it chose.
    * `oracle-free floor` — every encoder and the single cheapest decoder: the
      best any encoder-side selector could do on cost, whatever it chooses.
    """
    t = per_model[per_model["decoding"] == decoding].set_index("model")
    missing = [m for m in list(members) + list(feature_only) if m not in t.index]
    if missing:
        raise KeyError(f"no {decoding!r} timings for {missing}")
    enc = {m: float(t.loc[m, "encoder_seconds"]) for m in list(members) + list(feature_only)}
    dec = {m: float(t.loc[m, "decoder_seconds"]) for m in members}
    if shares is None:
        shares = {m: 1.0 / len(members) for m in members}
    all_enc = sum(enc.values())
    rows = [{"system": m, "kind": "member", "encode": enc[m], "decode": dec[m], "selector": 0.0}
            for m in members]
    rows += [
        {"system": "decode all, then choose", "kind": "after decoding",
         "encode": all_enc, "decode": sum(dec.values()), "selector": float(selector_seconds)},
        {"system": "choose from encoders (FASTT)", "kind": "before decoding",
         "encode": all_enc, "decode": float(sum(shares[m] * dec[m] for m in members)),
         "selector": float(selector_seconds)},
        {"system": "cheapest decoder only", "kind": "floor",
         "encode": all_enc, "decode": min(dec.values()), "selector": float(selector_seconds)},
    ]
    out = pd.DataFrame(rows)
    out["decoding"] = decoding
    out["total"] = out["encode"] + out["selector"] + out["decode"]
    out["x_best_single"] = out["total"] / min(enc[m] + dec[m] for m in members)
    after = float(out.loc[out["kind"] == "after decoding", "total"].iloc[0])
    out["x_after"] = out["total"] / after
    return out


def timing_table(per_model, members, arms, feature_only=(), n_train=None):
    """Table 4: encoding, decoding, training, inference and end-to-end per system.

    Parameters
    ----------
    per_model : `timing_summary` output — measured per-utterance encoder and
        decoder seconds for every member and every feature-only model.
    members : the routed member names, in the order of the `share_*` columns.
    arms : a frame with one row per system and the columns
        `model` (registry key), `label`, `kind` (`"member"`, `"selector"`,
        `"fusion"` or `"reference"`), `fit_seconds` (selector training wall-clock
        per fold), `predict_us_per_utt` (selector inference per utterance) and
        `share_<member>` (the arm's selection distribution). `results_table`
        and `cost_table` between them provide every column.
    feature_only : models whose encoder every router pays for but which are
        never decoded (the MFCC block).
    n_train : utterances in one training fold, to price the one-off feature
        extraction of the training set (`encode_train_seconds`).

    Returns one row per system with, in seconds per utterance unless named
    otherwise: `encode` (every encoder the system needs), `decode` (the decoder
    it actually runs — for a selector, the members' decoders weighted by how
    often each is chosen; for fusion, every member's), `selector_infer`,
    `end_to_end` (= encode + selector_infer + decode), and `train_selector`
    (seconds, per fold) plus `encode_train` (seconds, the training fold's
    features). A base model pays one encoder and one decoder and trains
    nothing.
    """
    t = per_model
    if "decoding" in t.columns and t["decoding"].nunique() > 1:
        raise ValueError("timing_table wants one decoder per model — "
                         "pass timing_summary(frames, decoding=<label>)")
    t = t.set_index("model")
    enc = {m: float(t.loc[m, "encoder_seconds"]) for m in list(members) + list(feature_only)}
    dec = {m: float(t.loc[m, "decoder_seconds"]) for m in members}
    all_enc = sum(enc.values())
    rows = []
    for _, a in arms.iterrows():
        kind, name = a["kind"], a["model"]
        sel_us = float(a.get("predict_us_per_utt", 0.0) or 0.0)
        fit = float(a.get("fit_seconds", 0.0) or 0.0)
        if kind == "member":
            e, d, s, fit = enc[name], dec[name], 0.0, 0.0
        elif kind == "fusion":
            e, d, s = all_enc, sum(dec.values()), sel_us / 1e6
        elif kind == "selector":
            shares = np.array([float(a.get(f"share_{m}", np.nan)) for m in members])
            if not np.isfinite(shares).all():
                shares = np.full(len(members), 1.0 / len(members))
            e, d, s = all_enc, float(sum(sh * dec[m] for sh, m in zip(shares, members))), sel_us / 1e6
        else:                                         # reference: oracle / bsm / random
            shares = np.array([float(a.get(f"share_{m}", np.nan)) for m in members])
            if not np.isfinite(shares).all():
                shares = np.full(len(members), 1.0 / len(members))
            e, d, s, fit = all_enc, float(sum(sh * dec[m] for sh, m in zip(shares, members))), 0.0, 0.0
        rows.append({
            "model": name, "label": a.get("label", name), "kind": kind,
            "encode": e, "decode": d, "selector_infer": s,
            "end_to_end": e + s + d,
            "train_selector": fit,
            "encode_train": (all_enc * n_train if (n_train and kind != "member") else 0.0),
        })
    out = pd.DataFrame(rows)
    out["vs_fastest_member"] = out["end_to_end"] / min(enc[m] + dec[m] for m in members)
    return out


def routing_latency(per_model, members, selector_ms=0.0, decoder_share=None):
    """Compose measured per-model timings into the cost of routing against the cost of decoding every member.

    Parameters
    ----------
    per_model : a frame with `model` and `total_seconds`, e.g. several
        `time_extractor` results concatenated.
    members : the routed member set.
    selector_ms : measured selector inference time per utterance. It is
        microseconds for a tree or a linear probe — include it anyway, because
        leaving it out would flatter the selector.
    decoder_share : optional `{model: fraction}` giving the share of that model's
        time spent decoding. Where it is known, routing pays every member's
        *encoder* plus the chosen member's decoder rather than every member's
        full pass, which is a materially smaller number. Where it is not known,
        the full pass is charged and the result is an upper bound — labelled as
        such in the returned `basis` column.
    """
    t = per_model.groupby("model")["total_seconds"].mean()
    missing = set(members) - set(t.index)
    if missing:
        raise KeyError(f"no timings for {sorted(missing)}")

    share = decoder_share or {}
    enc = {m: t[m] * (1 - share.get(m, 0.0)) for m in members}
    dec = {m: t[m] * share.get(m, 1.0) for m in members}
    basis = "encoder+decoder split" if decoder_share else "upper bound (full pass per member)"

    rows = [{"system": m, "seconds": t[m], "basis": "single model"} for m in members]
    rows.append({
        "system": "routed(" + "+".join(members) + ")",
        # Every encoder, plus the selector, plus the decoder of whichever member
        # is chosen — averaged over members, since the choice varies per utterance.
        "seconds": sum(enc.values()) + selector_ms / 1000
                   + float(np.mean(list(dec.values()))),
        "basis": basis,
    })
    out = pd.DataFrame(rows)
    best = t[list(members)].min()
    out["vs_fastest_member"] = out["seconds"] / best
    return out

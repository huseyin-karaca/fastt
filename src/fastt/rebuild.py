"""Rebuilding the experts' features from the audio, as `notebooks/extract` runs it (level 3).

Every expert of a corpus's trio decodes the corpus in full: its transcripts, their word-error counters against the
reference (Whisper-normalised), and the mean of its final-layer encoder states per clip — the columns
`fastt.ensemble.EnsembleStore` reads. They are written to a local folder (`fastt.experiment.rebuild_dir()`) as the
`model_<expert>` configurations of the published dataset are laid out, and `main_<corpus>` at level 3 reads them from
there. The audio is the published `base` configuration of each corpus (16 kHz, the clips the paper uses).

The experts come from three toolkits that do not install into one environment: **A** 🤗 Transformers (wav2vec2,
data2vec, Whisper), **B** NeMo (Parakeet-TDT, Canary-Qwen), **C** ESPnet. Run the notebook once per group the trios
need, each in a fresh runtime; an expert whose features are already in the folder is skipped.
"""

__all__ = ['GROUPS', 'INSTALL', 'rebuild_plan', 'rebuild', 'compare_with_published']

import time
from pathlib import Path

import numpy as np
import pandas as pd

from fastt.configs import MAIN
from fastt.latency import install_group

GROUPS = {"A": "🤗 Transformers", "B": "NeMo", "C": "ESPnet"}
# What each group installs beside `fastt[paper]`. `datasets<4.0` is load-bearing: 4.x decodes audio differently.
INSTALL = {
    "A": ["datasets<4.0", "jiwer", "librosa", "soundfile", "transformers>=4.57", "torchaudio", "sentencepiece"],
    "B": ["datasets<4.0", "jiwer", "librosa", "soundfile",
          "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"],
    "C": ["datasets<4.0", "jiwer", "librosa", "soundfile", "transformers", "espnet", "espnet_model_zoo",
          "sentencepiece"],
}
META = ["id", "transcription", "transcription_norm", "text_norm", "n_frames", "sub", "dele", "ins", "nref", "wer",
        "sub_raw", "dele_raw", "ins_raw", "nref_raw", "wer_raw"]


def _folder_has(folder, config):
    return any((Path(folder) / config).glob("*.parquet"))


def rebuild_plan(corpora=tuple(MAIN), group=None, folder=None):
    """One row per (corpus, expert): its install group, and whether it is done, to do in this runtime, or needs
    another group's runtime."""
    from fastt.core import use_dataset
    from fastt.experiment import LABELS, rebuild_dir
    folder = Path(folder or rebuild_dir())
    rows = []
    for c in corpora:
        spec = use_dataset(c, verbose=False)
        for m in MAIN[c].members:
            g = install_group(m)
            done = _folder_has(folder, spec.model_config(m))
            rows.append({"corpus": c, "expert": m, "label": LABELS.get(m, m), "group": g,
                         "status": "done" if done else ("to do" if group in (None, g) else f"needs group {g}")})
    plan = pd.DataFrame(rows)
    todo = plan[plan["status"] == "to do"]
    print(f"{len(todo)} of {len(plan)} expert pass(es) to run here"
          + (f" (group {group}: {GROUPS[group]})" if group else "")
          + f"; {int((plan['status'] == 'done').sum())} already in {folder}")
    return plan


def rebuild(plan, folder=None, batch_size=16, smoke=True):
    """Every `to do` row of `plan`: the corpus's audio loaded once (from the published dataset), then per expert a
    smoke test (shapes, one transcript), the full pass, and its `model_<expert>` configuration written to `folder`
    (default `rebuild_dir()`). Returns a log, one row per pass and split, with the rebuilt corpus WER."""
    import torch

    from fastt import core
    from fastt.core import use_dataset
    from fastt.data import load_base
    from fastt.experiment import rebuild_dir
    from fastt.extractor import ExtractConfig
    from fastt.hub import FasttHub
    from fastt.latency import load_extractor
    from fastt.scoring import corpus_wer_table
    folder = Path(folder or rebuild_dir())
    folder.mkdir(parents=True, exist_ok=True)
    audio_repo = core.PUBLIC_REPO if core.REPO_ID.startswith("local:") else core.REPO_ID
    if not torch.cuda.is_available():
        print("note: no CUDA GPU — the pass runs on the CPU and takes many hours")
    cfg = ExtractConfig(layers=(-1,), poolings=("mean",), batch_size=batch_size)
    log = []
    for c, rows in plan[plan["status"] == "to do"].groupby("corpus", sort=False):
        spec = use_dataset(c, verbose=False)
        ds = load_base(repo_id=audio_repo)
        for m in rows["expert"]:
            t0 = time.perf_counter()
            ex = load_extractor(m)
            if smoke:
                ex.smoke_test(ds)
            feats = ex.extract(ds, cfg)
            pooled = [n for n in ex.pooled_columns(feats) if n.endswith("_mean")]
            FasttHub(f"local:{folder}", spec=spec).push(feats.select_columns([n for n in META if n in feats[list(feats)[0]].column_names]
                                                          + pooled), spec.model_config(m))
            corpus = corpus_wer_table(feats)
            for _, r in corpus.iterrows():
                log.append({"corpus": c, "expert": m, "split": r["split"], "n_utts": int(r["n_utts"]),
                            "wer_norm": float(r["wer_norm"]), "minutes": (time.perf_counter() - t0) / 60})
            del ex, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(log)


def compare_with_published(log, folder=None):
    """Each rebuilt expert against the published one on the same clips: the share of identical normalised
    transcripts and the corpus WER of both, per split. The same checkpoint, decoder and normaliser give the same
    transcripts up to the GPU's numerics."""
    from fastt import core
    from fastt.core import PUBLIC_REPO, use_dataset
    from fastt.experiment import rebuild_dir
    from fastt.hub import FasttHub
    folder = Path(folder or rebuild_dir())
    out = []
    for (c, m), _ in log.groupby(["corpus", "expert"], sort=False):
        spec = use_dataset(c, verbose=False)
        mine = FasttHub(f"local:{folder}", spec=spec)
        theirs = FasttHub(PUBLIC_REPO if core.REPO_ID.startswith("local:") else core.REPO_ID, spec=spec)
        cols = ["id", "transcription_norm", "sub", "dele", "ins", "nref"]
        for split in spec.splits:
            a = mine.read_columns(spec.model_config(m), split, cols).to_pandas().set_index("id")
            b = theirs.read_columns(spec.model_config(m), split, cols).to_pandas().set_index("id")
            b = b.loc[a.index]
            err = lambda d: (d["sub"] + d["dele"] + d["ins"]).sum() / d["nref"].sum()     # noqa: E731
            out.append({"corpus": c, "expert": m, "split": split, "clips": len(a),
                        "identical transcripts": float(np.mean(a["transcription_norm"] == b["transcription_norm"])),
                        "WER rebuilt": float(err(a)), "WER published": float(err(b))})
    df = pd.DataFrame(out)
    df["Δ WER"] = df["WER rebuilt"] - df["WER published"]
    return df

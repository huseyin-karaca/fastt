"""Table 4, the inference cost, as `notebooks/timing` runs it: where the time goes (encoding or decoding) and what
choosing an expert before decoding saves.

    t = Timing("voxpopuli", level=1)
    t.clocks()        # level 1: the per-expert clocks from the Hub; level 2: measured on this machine's GPU
    t.compose()       # the per-expert table, the ensemble cost per beam width, Table 4 for every arm
    t.decoding(); t.ensemble(); t.savings(); t.table4()
    t.save()          # timing_<corpus>.json

**What is measured.** Each expert's encoder and decoder, separately, per clip at batch size 1, on the same 32 random
clips (after 3 untimed warm-up clips), at greedy decoding and at beam widths 5 and 20. Everything else is composed
from those clocks and the main run's record: an ensemble that decodes every expert and then chooses pays every encoder
and every decoder; FASTT pays every encoder, the selector, and the decoder of the expert it picked, weighted by how
often it picks each (its shares in Table 3). The selectors' own fit and inference times come from the main run's 5x2
folds.

**Levels.** 1 reads the clocks the paper measured (`results/<corpus>/timing/<expert>.json`) and recomposes every
table from them, exactly. 2 measures the clocks again: it needs a CUDA GPU, the audio (`base`), and the experts'
checkpoints and toolkits (`pip install "fastt[extract]"`); absolute numbers then depend on the GPU, the ratios much
less.
"""

__all__ = ['Timing', 'BEAM_SIZES', 'N_CLIPS', 'WARMUP', 'BATCH_SIZE', 'CLIP_SEED', 'FASTT_ARM']

import json

import numpy as np
import pandas as pd

from fastt.configs import MAIN
from fastt.core import use_dataset
from fastt.experiment import LABELS
from fastt.hub import FasttHub
from fastt.latency import EXTRACTORS, decoding_table, ensemble_cost, timing_summary, timing_table
from fastt.pretty import Table
from fastt.records import load_record

BEAM_SIZES = (1, 5, 20)          # 1 = greedy, what every stored configuration decodes with
N_CLIPS = 32                     # timed clips per expert and beam width
WARMUP = 3                       # untimed clips first: a fresh model pays for CUDA context and autotuning
BATCH_SIZE = 1                   # online latency: one clip arriving on its own
CLIP_SEED = 0                    # the same random clips for every expert and every beam width
FASTT_ARM = "fastt_boosted"      # whose shares and selector clock the ensemble cost uses


class Timing:
    """The inference cost on one corpus (VoxPopuli in the paper) at one reproduction level."""

    def __init__(self, corpus="voxpopuli", level=1):
        if level not in (1, 2):
            raise ValueError("level is 1 (the paper's clocks) or 2 (measured here)")
        self.corpus, self.level = corpus, level
        self.spec = use_dataset(corpus, verbose=False)
        self.hub = FasttHub(spec=self.spec)
        self.main = load_record(f"main_{corpus}", corpus)
        self.members = tuple(self.main["members"])
        self.feature_only = tuple(self.main["feature_only"])
        assert self.members == tuple(MAIN[corpus].members), "the main record is not the paper's trio"
        t3 = {r["model"]: r for r in self.main["tables"]["table3"]}
        self.t3, self.cost = t3, {r["model"]: r for r in self.main["tables"]["cost"]}
        self.shares = {m: float(t3[FASTT_ARM][f"share_{m}"]) for m in self.members}
        self.selector_seconds = float(self.cost[FASTT_ARM]["infer_us_per_utt"] or 0.0) / 1e6
        self.to_time = [m for m in self.members + self.feature_only if m in EXTRACTORS]
        self.records = {}
        print(f"{MAIN[corpus].title}: level {level} — clocks "
              + ("read from the Hub" if level == 1 else "measured on this machine") + "\n"
              f"experts  {', '.join(LABELS.get(m, m) for m in self.to_time)}\n"
              f"{LABELS[FASTT_ARM]} picks " + ", ".join(f"{LABELS.get(m, m)} {self.shares[m]:.0%}" for m in self.members)
              + f"; selector {self.selector_seconds * 1e6:.1f} µs per clip")

    # ------------------------------------------------------------------ clocks --

    def clocks(self, models=None):
        """Every expert's clocks: read (level 1) or measured (level 2), one record per expert."""
        for name in models or self.to_time:
            if self.level == 1:
                self.records[name] = json.loads(open(self.hub.local_path(
                    self.spec.results_path(f"timing/{name}"))).read())
            else:
                self.records[name] = self._measure(name)
        return Table(self._summary().round(4), title="Per-clip seconds, batch size 1",
                     caption="mean encoder and decoder wall clock per expert and decoder setting; dec_over_enc = "
                             "decoder seconds per encoder second; WER on the timed clips")

    def _measure(self, name):
        """One expert, every beam width, on the same clips (level 2: a CUDA GPU and the expert's toolkit)."""
        import torch

        from fastt.data import load_base
        from fastt.latency import load_extractor, set_decoding, time_extractor
        if not torch.cuda.is_available():
            raise RuntimeError("level 2 times the experts on a CUDA GPU, and this machine has none")
        if getattr(self, "_ds", None) is None:
            self._ds = load_base()
        ex = load_extractor(name)
        frames = []
        for beam in BEAM_SIZES:
            label = set_decoding(ex, beam)
            if label is None:
                print(f"  {name}: no beam to widen — timed greedy only")
                break
            frames.append(time_extractor(ex, self._ds, n=N_CLIPS, batch_size=BATCH_SIZE, warmup=WARMUP,
                                         seed=CLIP_SEED, decoding=label))
        frame = pd.concat(frames, ignore_index=True)
        rec = {"model": name, "dataset": self.corpus, "batch_size": BATCH_SIZE,
               "beam_sizes": list(BEAM_SIZES), "n_clips": N_CLIPS, "clip_seed": CLIP_SEED,
               "device": str(ex.device), "dtype": str(ex.dtype), "rows": json.loads(frame.to_json(orient="records"))}
        del ex
        torch.cuda.empty_cache()
        return rec

    def _summary(self, decoding=None):
        missing = [m for m in self.to_time if m not in self.records]
        if missing:
            raise RuntimeError(f"no clocks yet for {missing}: run `clocks()` first")
        return timing_summary([pd.DataFrame(self.records[m]["rows"]) for m in self.to_time], decoding=decoding)

    # ------------------------------------------------------------------ compose --

    def compose(self):
        """Every table from the clocks and the main record."""
        self.per_model = self._summary()
        self.decoding_frame = decoding_table(self.per_model, members=self.members)
        labels = [lab for lab in dict.fromkeys(self.per_model["decoding"])
                  if all(((self.per_model["model"] == m) & (self.per_model["decoding"] == lab)).any()
                         for m in self.members)]
        self.costs = pd.concat([ensemble_cost(self.per_model, self.members, lab, shares=self.shares,
                                              selector_seconds=self.selector_seconds,
                                              feature_only=self.feature_only) for lab in labels], ignore_index=True)
        after = self.costs[self.costs["kind"] == "after decoding"].set_index("decoding")
        before = self.costs[self.costs["kind"] == "before decoding"].set_index("decoding")
        self.saving = pd.DataFrame({
            "decode_share_after": (after["decode"] / after["total"]).round(3),
            "total_after_ms": (after["total"] * 1e3).round(1),
            "total_before_ms": (before["total"] * 1e3).round(1),
            "saved_ms": ((after["total"] - before["total"]) * 1e3).round(1),
            "saved_fraction": (1 - before["total"] / after["total"]).round(3),
        }).reset_index()

        def kind_of(model, family):
            if family == "individual" and model != "bsm":
                return "member"
            if family == "oracle" or model in ("bsm", "random"):
                return "reference"
            return "fusion" if self.t3.get(model, {}).get("n_fused", 0) > 0 else "selector"

        share_cols = [f"share_{m}" for m in self.members]
        arms = pd.DataFrame([
            {"model": m, "label": self.main["labels"].get(m, m), "kind": kind_of(m, fam),
             "fit_seconds": self.cost.get(m, {}).get("fit_s_mean", 0.0),
             "predict_us_per_utt": self.cost.get(m, {}).get("infer_us_per_utt", 0.0),
             **{c: self.t3.get(m, {}).get(c, np.nan) for c in share_cols}}
            for m, fam in (tuple(r) for r in self.main["rows"])])
        self.table4_frame = timing_table(self._summary(decoding="greedy"), self.members, arms,
                                         feature_only=self.feature_only, n_train=self.main["n_train_per_fold"])
        return self

    # ------------------------------------------------------------------ tables --

    def decoding(self):
        """Per expert: the encoder once, then per beam width the decoder's milliseconds, its multiple of the encoder,
        and the WER on the timed clips."""
        t = self.decoding_frame.copy()
        t.insert(0, "expert", t.pop("model").map(lambda m: LABELS.get(m, m)))
        for c in t.columns:
            if c.endswith("_ms"):
                t[c] = t[c].round(1)
            elif c.endswith("_x_encoder"):
                t[c] = t[c].round(2)
            elif c.endswith("_wer"):
                t[c] = t[c].round(3)
        return Table(t, title=f"Decoding against encoding — {MAIN[self.corpus].title}",
                     caption=f"ms per clip (batch {BATCH_SIZE}, {N_CLIPS} random clips); <beam>_x_encoder = that "
                             "decoder's time as a multiple of the encoder pass")

    def ensemble(self):
        """Per beam width: each expert alone, decoding every expert and then choosing, choosing from the encoders and
        decoding one (FASTT), and the floor of always decoding the cheapest expert."""
        c = self.costs
        t = pd.DataFrame({"decoding": c["decoding"], "system": c["system"].map(lambda s: LABELS.get(s, s)),
                          "encode_ms": (c["encode"] * 1e3).round(1), "decode_ms": (c["decode"] * 1e3).round(1),
                          "selector_ms": (c["selector"] * 1e3).round(3), "total_ms": (c["total"] * 1e3).round(1),
                          "x_best_single": c["x_best_single"].round(2), "x_after_decoding": c["x_after"].round(2)})
        return Table(t, title=f"Ensemble cost per clip — {MAIN[self.corpus].title}", group="decoding",
                     caption="x_after_decoding: total relative to decoding every expert at the same beam width",
                     highlight=lambda r: str(r["system"]).startswith("choose from encoders"))

    def savings(self):
        """Per beam width: decoding's share of the decode-every-expert pipeline, and what choosing first saves."""
        for _, r in self.saving.iterrows():
            print(f"{r['decoding']:<8} decoding is {r['decode_share_after']:.0%} of decode-then-choose; choosing "
                  f"from the encoders saves {r['saved_ms']:.1f} ms per clip ({r['saved_fraction']:.0%})")
        return Table(self.saving, title="Choosing before decoding — the saving per beam width",
                     caption="saved_fraction = 1 − before / after")

    def table4(self):
        """Every arm of Table 3 at greedy decoding: encode, decode, selector inference, end to end, and the selector's
        training time on one 5x2 fold."""
        t = self.table4_frame
        shown = pd.DataFrame({"system": t["label"], "encode_ms": (t["encode"] * 1e3).round(1),
                              "decode_ms": (t["decode"] * 1e3).round(1),
                              "selector_us": (t["selector_infer"] * 1e6).round(1),
                              "end_to_end_ms": (t["end_to_end"] * 1e3).round(1),
                              "train_selector_s": t["train_selector"].round(2),
                              "x_fastest_member": t["vs_fastest_member"].round(2)})
        return Table(shown, title=f"Table 4 — cost per clip, greedy decoding, {MAIN[self.corpus].title}",
                     caption=f"selector training seconds per 5x2 fold ({self.main['n_train_per_fold']:,} clips)",
                     best={"end_to_end_ms": "min", "train_selector_s": "min"},
                     highlight=lambda r: str(r["system"]).startswith("FASTT"))

    # ------------------------------------------------------------------ record --

    def record(self):
        rec = lambda df: json.loads(df.to_json(orient="records"))                       # noqa: E731
        return {"dataset": self.corpus, "members": list(self.members), "feature_only": list(self.feature_only),
                "beam_sizes": list(BEAM_SIZES), "n_clips": N_CLIPS, "batch_size": BATCH_SIZE,
                "shares": self.shares, "selector_seconds": self.selector_seconds, "level": self.level,
                "per_model": rec(self.per_model), "decoding": rec(self.decoding_frame),
                "ensemble_cost": rec(self.costs), "savings": rec(self.saving), "table4": rec(self.table4_frame)}

    def save(self, path=None):
        """Write the record (`timing_<corpus>.json` in the working directory) and, at level 2, every expert's clocks
        (`timing_<corpus>_<expert>.json`)."""
        path = path or f"timing_{self.corpus}.json"
        with open(path, "w") as fh:
            json.dump(self.record(), fh, indent=2)
        if self.level == 2:
            for name, r in self.records.items():
                with open(f"timing_{self.corpus}_{name}.json", "w") as fh:
                    json.dump(r, fh, indent=2)
        print(f"wrote {path}")
        return path

    def compare(self, reference=None):
        """Each composed table against the record the paper used (`timing_<corpus>.json` on the Hub): identical at
        level 1; at level 2, the ratio of every clock to the paper's."""
        ref = reference or json.loads(open(self.hub.local_path(
            self.spec.results_path(f"timing_{self.corpus}"))).read())
        mine = self.record()
        out = []
        for key in ("per_model", "decoding", "ensemble_cost", "savings", "table4"):
            a, b = pd.DataFrame(mine[key]), pd.DataFrame(ref[key])
            num = [c for c in b.select_dtypes("number").columns if c in a]
            diff = float((a[num] - b[num]).abs().max().max()) if num else 0.0
            out.append({"table": key, "rows": len(b), "identical": mine[key] == ref[key], "max_abs_diff": diff})
        return Table(pd.DataFrame(out), title=f"Against the paper's record — {MAIN[self.corpus].title}")


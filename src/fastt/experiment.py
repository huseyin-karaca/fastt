"""The main experiment on one corpus, as `notebooks/main_<corpus>` runs it: the hyperparameter search on a hold-out,
5x2 cross-validation on the remaining clips, and the paper's tables.

    exp = MainExperiment(MAIN["ami_sdm"], level=1)
    exp.spaces()                  # Table 2
    exp.load()                    # the trio's pooled features, WER counters and transcripts
    exp.search()                  # every arm's hyperparameters, on the hold-out
    exp.dataset(); exp.selected() # Table 1, and what the search selected
    exp.cross_validate()          # every arm on the same ten folds
    exp.results(); exp.tests()    # Table 3 and Table 6
    exp.save()                    # main_<corpus>.json

**Levels.** 1 reads every search and every fold from the Hub and fits nothing (a missing one stops the run); 2 refits
every fold from the published features (`retune=True`: the search as well); 3 is level 2 on features you rebuilt from
the audio: `notebooks/extract` writes them to a local folder (`rebuild_dir()`), which level 3 reads unless
`FASTT_HUB` names another place. Nothing is ever written to the Hub.
"""

__all__ = ['MainExperiment', 'LABELS', 'MODEL_LABELS', 'ARM_LABELS', 'table_rows', 'rebuild_dir']

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import fastt.rover  # noqa: F401  (registers the fusion arms)
from fastt import core
from fastt.configs import ARMS, BASELINE_ARMS, SELECTED, SELECTION_ARMS, SPACES
from fastt.core import PUBLIC_REPO, active_dataset, use_dataset, use_hub
from fastt.crossval import (ACC_TOLERANCES, METRICS, RepetitionSplit, cost_table, format_results, results_table,
                            selection_distribution, set_achievable_margin)
from fastt.eda import dataset_table, feature_dims
from fastt.ensemble import EnsembleStore, best_single_model
from fastt.estimators import set_determinism
from fastt.hub import set_read_only
from fastt.pretty import Table
from fastt.rover import check_counters
from fastt.runlog import RunLog
from fastt.significance import SIGNIFICANCE_COLUMNS, better_arm, format_significance, significance_table
from fastt.tuning import OURS, check_spaces, completed, hpt, space_source

MODEL_LABELS = {
    "wav2vec2": "wav2vec2-base-960h", "wav2vec2_large": "wav2vec2-large-960h",
    "wav2vec2_large_lv60": "wav2vec2-large-lv60", "data2vec": "data2vec-audio-base",
    "whisper_large_v3": "Whisper-large-v3", "parakeet_tdt_0_6b_v2": "Parakeet-TDT-0.6b-v2",
    "parakeet_tdt_0_6b_v3": "Parakeet-TDT-0.6b-v3", "canary_qwen_2_5b": "Canary-Qwen-2.5b",
    "espnet_transformer": "ESPnet Transformer", "espnet_conformer": "ESPnet Conformer",
    "espnet_transducer": "ESPnet Conformer-RNN-T",
}
ARM_LABELS = {
    "bsm": "Best-Single-Model", "random": "Random-Ensemble",
    "rover_w": "ROVER (weighted)", "rover_conf": "ROVER (confidence-weighted)",
    "cn_mbr": "CN-MBR", "score_fusion": "Score-Fusion",
    "argmin_clf": "Hard-Selector (plain CE)", "probe": "Hard-Selector (cost-sensitive CE)",
    "qe_router": "QE-Router",
    "adastt": "Raw XGBoost (AdaSTT)", "raw_sdt": "Raw SDT",
    "kbest_xgboost": "SelectKBest XGBoost", "kbest_sdt": "SelectKBest SDT",
    "fastt_boosted": "FASTT-Boosted", "fastt_alternating": "FASTT-Alternating",
    # `oracle_tol{n}` switches expert only for a gain of more than n words; `oracle` is the plain per-utterance one.
    "oracle": "Oracle (no tolerance)", "oracle_tol1": "Oracle (1 word tolerance)",
    "oracle_tol2": "Oracle (2 word tolerance)",
    # computed and recorded, not shown in Table 3
    "worst": "Worst-Single-Model", "achievable_oracle": "Achievable-Oracle", "soft_probe": "Soft-Selector (mixture)",
}
LABELS = {**MODEL_LABELS, **ARM_LABELS}
METRIC_TITLES = {"mean_utt_wer": "mean utterance WER", "corpus_wer": "corpus WER"}


def table_rows(members):
    """Table 3's rows in order: `(key, family)`."""
    return ([(m, "individual") for m in members] + [("bsm", "individual")]
            + [(a, "external") for a in ("random", "rover_w", "rover_conf", "cn_mbr",
                                         "score_fusion", "argmin_clf", "probe", "qe_router")]
            + [(a, "classical") for a in ("adastt", "raw_sdt", "kbest_xgboost", "kbest_sdt")]
            + [(a, "fastt") for a in ("fastt_boosted", "fastt_alternating")]
            + [(a, "oracle") for a in ("oracle", "oracle_tol1", "oracle_tol2")])


def rebuild_dir():
    """Where `notebooks/extract` writes the features it rebuilds from the audio, and level 3 reads them."""
    return Path(os.environ.get("FASTT_CACHE", Path.home() / ".cache" / "fastt")) / "rebuilt"


def installed_commit():
    """The commit the installed package came from (a `pip install git+...`), or `None`."""
    import importlib.metadata as md
    for dist in md.packages_distributions().get("fastt", ["fastt"]):
        try:
            return json.loads(md.distribution(dist).read_text("direct_url.json"))["vcs_info"]["commit_id"][:8]
        except Exception:                                      # noqa: BLE001
            continue
    return None


class MainExperiment:
    """One corpus's main experiment at one reproduction level. See the module docstring for the order of calls."""

    def __init__(self, cfg, level=1, retune=False, workdir=None):
        if level not in (1, 2, 3):
            raise ValueError("level is 1, 2 or 3")
        if level == 3 and core.REPO_ID == PUBLIC_REPO:
            folder = rebuild_dir()
            if not folder.is_dir() or not any(folder.iterdir()):
                raise RuntimeError(f"level 3 reads the features `notebooks/extract` rebuilt, and {folder} holds none: "
                                   "run `extract` first (same machine), or set FASTT_HUB to where they are")
            use_hub(data=f"local:{folder}")
        self.cfg, self.level, self.retune = cfg, level, bool(retune)
        use_dataset(cfg.dataset, verbose=False)
        self.spec = active_dataset()
        set_achievable_margin(cfg.achievable_margin)
        set_determinism(cfg.model_seed)
        set_read_only(True)
        check_spaces({a: SPACES[a] for a in ARMS})
        self.refit_search, self.refit_folds = level >= 2 and self.retune, level >= 2
        self.workdir = Path(workdir or Path(os.environ.get("FASTT_CACHE", Path.home() / ".cache" / "fastt"))
                            / "runs" / f"level{level}{'_retune' if self.refit_search else ''}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.members = tuple(cfg.members)
        self.rows = table_rows(self.members)
        self.tables = {}
        print(f"{cfg.title}: level {level} — search {'refitted' if self.refit_search else 'read from the Hub'}, "
              f"folds {'refitted' if self.refit_folds else 'read from the Hub'}\n"
              + (f"features from {core.REPO_ID}\n" if level == 3 else "")
              + f"experts  {', '.join(LABELS.get(m, m) for m in self.members)}\n"
              f"arms     {', '.join(LABELS.get(a, a) for a in ARMS)}")

    # ------------------------------------------------------------------ Table 2 --

    def spaces(self):
        """Table 2: the free parameters of every arm and their choices (`fastt.configs.SPACES`)."""
        rows = []
        for arm in ARMS:
            free = [(n, p) for n, p in SPACES[arm].items() if p.frozen is None]
            for n, p in free:
                rows.append({"arm": f"{LABELS.get(arm, arm)}  ({len(free)} free of {len(SPACES[arm])})",
                             "parameter": n, "kind": p.kind + (" (log)" if p.log else ""),
                             "choices / range": ("{" + ", ".join(str(c) for c in p.choices) + "}"
                                                 if p.kind == "cat" else f"[{p.low:g}, {p.high:g}]")})
        return Table(pd.DataFrame(rows), title="Table 2 — hyperparameter spaces",
                     caption=f"{self.cfg.search_budget} random draws per arm; parameters not listed are fixed",
                     group="arm", mono=("choices / range",), wrap=("choices / range",))

    # --------------------------------------------------------------------- data --

    def load(self):
        """The trio's pooled encoder features, WER counters and transcripts. The stored WER counters are recomputed
        from the stored transcripts on a sample of clips."""
        self.store = EnsembleStore(models=self.members, pooling=self.cfg.pooling).load()
        self.store.load_texts(self.members)
        counters = check_counters(self.store, self.members, n=400, seed=self.cfg.model_seed)
        assert float(counters["match"].mean()) == 1.0, "the stored WER counters do not reproduce"
        self.bsm_fit = self.members[best_single_model(self.store, self.members)]
        return self.store

    # ------------------------------------------------------------------- search --

    def _search(self, arms, spaces=None):
        c = self.cfg
        spaces = spaces or {a: SPACES[a] for a in arms}
        return hpt(self.store, self.members, models=arms, name=None, spaces=spaces,
                   exact_budget=True, rounds=c.search_rounds, trials_final=c.search_budget, pruner="none",
                   sync=not self.refit_search, versions=c.arm_versions, sampler=c.sampler,
                   objectives=("corpus_wer",), holdout_frac=c.holdout_frac, inner_val_frac=c.inner_val_frac,
                   partition_seed=c.partition_seed, study_seed=c.study_seed, model_seed=c.model_seed,
                   nthread=c.nthread, dirname=str(self.workdir), strict=not self.refit_search,
                   verbose=self.refit_search)

    def _pin(self, run, pinned=None):
        """Use the reported configuration of every arm. It must be one of the best trials of the stored search."""
        pinned = pinned or SELECTED[self.cfg.dataset]
        for arm, r in list(run.results.items()):
            want = pinned[arm]
            if r.params == want:
                continue
            study = optuna.load_study(study_name=f"{run.names[arm]}_final", storage=run.studies[arm].url())
            done = completed(study)
            best = min(t.values[0] for t in done)
            match = [t for t in done if t.values[0] == best and all(want.get(k) == v for k, v in t.params.items())]
            if not match:
                raise AssertionError(f"{arm}: the reported configuration is not among the best trials of {study.study_name}")
            run.results[arm] = replace(r, params=dict(want), trial_number=match[0].number)
        return run

    def search(self):
        """Every arm's hyperparameters, chosen on a random 25 % hold-out: `search_budget` random draws from its
        space, each fitted on 70 % of the hold-out and scored on the rest. The two families are two searches on one
        partition. Levels 1 and 2 read the stored searches and use the reported configurations."""
        self.run = self._search(SELECTION_ARMS)
        self.baseline_run = self._search(BASELINE_ARMS)
        for part in ("tune", "fit", "val", "eval"):
            assert np.array_equal(self.run.tuner.rows[part], self.baseline_run.tuner.rows[part]), part
        if not self.refit_search:
            self._pin(self.run)
            self._pin(self.baseline_run)
        self.results_all = {**self.run.results, **self.baseline_run.results}
        bad = [(a, r.n_trials) for a, r in self.results_all.items() if r.n_trials != self.cfg.search_budget]
        assert not bad, f"budget not exact: {bad}"
        self.run.tuner.report()
        print(f"  {len(self.results_all)} arms x {self.cfg.search_budget} trials"
              + (", searched on this machine" if self.refit_search else ", read from the Hub")
              + f"; {len(self.run.eval_rows):,} evaluation clips the search never saw")
        if self.refit_search:
            moved = [a for a, r in self.results_all.items() if r.params != SELECTED[self.cfg.dataset][a]]
            if moved:
                print(f"  selected differently from the paper (equal hold-out WER can tie): {', '.join(moved)}")
        return self.run

    def dataset(self):
        """Table 1: the corpus, the hold-out and the folds."""
        dims = feature_dims(self.store, self.members)
        self.n_features = int(dims["n_features"].sum())
        self.tables["table1"] = t = dataset_table(self.store, self.run.tuner.rows, self.n_features,
                                                  self.cfg.seconds_per_frame)
        return Table(t, title=f"Table 1 — {self.cfg.title}",
                     caption="train / test sizes are those of one 5x2 fold; the hold-out is the search partition")

    def selected(self):
        """What the search selected per arm, with its WER on the hold-out's scoring rows."""
        self.tables["selected"] = t = pd.DataFrame([
            {"arm": a, "label": LABELS.get(a, a), "trials": r.n_trials, "holdout_corpus_wer": r.value,
             "params": json.dumps(r.params, sort_keys=True, default=str)}
            for a, r in self.results_all.items()])
        return Table(t.drop(columns="arm").rename(columns={"holdout_corpus_wer": "hold-out WER",
                                                           "params": "selected configuration"}),
                     title="Selected configurations",
                     caption=f"corpus WER on the search's scoring rows after {self.cfg.search_budget} trials per arm "
                             "— a hold-out number, not a test number",
                     best={"hold-out WER": "min"}, mono=("selected configuration",), wrap=("selected configuration",))

    # --------------------------------------------------------------- 5x2 folds --

    def cross_validate(self):
        """Each repetition shuffles the evaluation clips with its seed and halves them; every arm is fitted on one half
        and scored on the other, then the other way round. Returns `(fold_scores, fold_picks)`."""
        c, run = self.cfg, self.run
        log = RunLog(f"main_{c.dataset}", dirname=self.workdir, spec=self.spec, sync=not self.refit_folds,
                     refresh=self.refit_folds, strict=not self.refit_folds)
        key = {"partition": run.partition(), "n_eval": int(len(run.eval_rows)), "random_seed": c.cv_random_seed,
               "member_arms": True, "tolerances": list(ACC_TOLERANCES), "achievable_margin": c.achievable_margin,
               "metrics": sorted(METRICS)}
        arms = {**run.arms, **self.baseline_run.arms}
        arm_keys = {**run.arm_keys(), **self.baseline_run.arm_keys()}

        def one_repetition(seed, estimators):
            return run.evaluate(estimators=estimators, scheme=RepetitionSplit(seed), random_seed=c.cv_random_seed,
                                member_arms=True, tolerances=ACC_TOLERANCES, fold_table=self.refit_folds)

        self.fold_scores, self.fold_picks = log.per_arm("cv5x2", c.cv_seeds, arms, one_repetition, key=key,
                                                        arm_keys=arm_keys)
        self.n_folds = int(self.fold_scores.drop_duplicates(["repeat", "fold"]).shape[0])
        self.bsm = self.fold_scores["bsm"].mode().iloc[0]
        print(f"{len(c.cv_seeds)} repetitions x 2 folds x {self.fold_scores['model'].nunique()} arms"
              + ("" if self.refit_folds else ", read from the Hub"))
        return self.fold_scores, self.fold_picks

    # ------------------------------------------------------------------- tables --

    def results(self):
        """Table 3: every arm, mean ± sd over the ten folds."""
        order = [k for k, _ in self.rows]
        t3 = results_table(self.fold_scores, self.fold_picks, self.members, order=order, labels=LABELS)
        t3.insert(2, "family", t3["model"].map(dict(self.rows)).fillna("hidden"))
        self.tables["table3"] = t3
        shown = t3[t3["model"].isin(order)].copy()
        shown.loc[shown["model"] == "bsm", "label"] = f"Best-Single-Model ({LABELS.get(self.bsm, self.bsm)})"
        fmt = format_results(shown)
        fmt.insert(1, "family", shown["family"].values)
        self.tables["table3_formatted"] = fmt
        return Table(fmt, title=f"Table 3 — main results, {self.cfg.title}",
                     caption=f"mean ± sd over {self.n_folds} folds; GC / uGC / Acc in %; "
                             f"dist = share of clips routed to {' / '.join(LABELS.get(m, m) for m in self.members)}; "
                             "best non-oracle value per column in bold",
                     group="family", best="auto", best_exclude=lambda r: r["family"] == "oracle",
                     highlight=lambda r: r["family"] == "fastt", mono=("dist",))

    def tests(self):
        """Table 6: every arm against the corpus's FASTT anchor (the better FASTT variant on that metric), with
        Dietterich's 5x2cv t and Alpaydin's combined 5x2cv F on the identical folds."""
        comparators = [k for k, _ in self.rows]
        self.table6, frames = {}, []
        for metric, title in METRIC_TITLES.items():
            anchor, means = better_arm(self.fold_scores, OURS, metric)
            t = significance_table(self.fold_scores, metric, anchor, comparators, alpha=self.cfg.alpha)
            self.table6[metric] = t
            f = format_significance(t, labels=LABELS)
            f.insert(0, "metric", f"{title} — anchor {LABELS[anchor]}")
            frames.append(f)
        alpha = self.cfg.alpha
        return Table(pd.concat(frames, ignore_index=True),
                     title=f"Table 6 — every arm against the FASTT anchor, {self.cfg.title}",
                     caption=f"on the same {self.n_folds} folds; positive effect = the anchor is better, in the "
                             f"metric's own units; p < {alpha:g} in bold. The F-test is the one the paper reports.",
                     group="metric", cell_style=_p_style(alpha))

    def routing(self):
        """How each selector distributes the clips over the trio: shares, entropy (bits), switch rate."""
        beh = selection_distribution(self.fold_picks, list(self.members), fold_scores=self.fold_scores)
        self.tables["behaviour"] = beh
        return Table(beh, title=f"Selection behaviour — {self.cfg.title}")

    # ------------------------------------------------------------------- record --

    def record(self):
        """Everything above in one JSON-able dict: the configuration, the selected hyperparameters, the search spaces
        as source, every table."""
        c, run = self.cfg, self.run
        rec = lambda df: json.loads(df.to_json(orient="records"))                         # noqa: E731
        return {
            "notebook": f"main_{c.dataset}", "fastt_commit": installed_commit(), "level": self.level,
            "retune": self.retune, "config": asdict(c), "members": list(self.members),
            "feature_only": list(self.store.feature_only),
            "corpus": {"name": self.spec.name, "source": self.spec.source, "fit_splits": list(self.spec.fit_splits),
                       "eval_splits": list(self.spec.eval_splits)},
            "n_features": self.n_features, "n_eval_rows": int(len(run.eval_rows)),
            "n_train_per_fold": int(len(run.eval_rows) // 2), "bsm": self.bsm_fit, "bsm_per_fold": self.bsm,
            "study": run.name, "baseline_study": self.baseline_run.name,
            "studies": {**run.study_paths(), **self.baseline_run.study_paths()},
            "versions": dict(c.arm_versions),
            "search_spaces": {a: space_source(SPACES[a], varname=a) for a in ARMS},
            "params": {a: self.results_all[a].params for a in ARMS},
            "arms": {"selection": list(SELECTION_ARMS), "baseline": list(BASELINE_ARMS)},
            "rows": self.rows, "labels": LABELS,
            "tables": {
                "table1": rec(self.tables["table1"]), "table3": rec(self.tables["table3"]),
                "table3_formatted": rec(self.tables["table3_formatted"]),
                "table6": {m: {"anchor": t.attrs["anchor"], "test": list(SIGNIFICANCE_COLUMNS), "rows": rec(t)}
                           for m, t in self.table6.items()},
                "selected": rec(self.tables["selected"]),
                "cost": rec(cost_table(self.fold_scores, results=self.results_all)),
            },
        }

    def save(self, path=None):
        """Write the record (`main_<corpus>.json` in the working directory)."""
        path = path or f"main_{self.cfg.dataset}.json"
        with open(path, "w") as fh:
            json.dump(self.record(), fh, indent=2, default=str)
        print(f"wrote {path}")
        return path


def _p_style(alpha):
    def style(col, value, row):
        if not col.startswith("p_"):
            return None
        try:
            return "font-weight:600" if float(value) < alpha else "opacity:.55"
        except ValueError:
            return None
    return style


def main(argv=None):
    """`python -m fastt.experiment <corpus> [--level N] [--retune]`: a main notebook as a script, the tables printed
    as text and the record written to the working directory."""
    import argparse

    from fastt.configs import MAIN
    ap = argparse.ArgumentParser(description=main.__doc__.split(":")[0])
    ap.add_argument("corpus", choices=sorted(MAIN))
    ap.add_argument("--level", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--retune", action="store_true")
    a = ap.parse_args(argv)
    exp = MainExperiment(MAIN[a.corpus], level=a.level, retune=a.retune)
    print(exp.spaces())
    exp.load()
    exp.search()
    for table in (exp.dataset(), exp.selected()):
        print(table)
    exp.cross_validate()
    for table in (exp.results(), exp.tests()):
        print(table)
    exp.save()


if __name__ == "__main__":
    main()

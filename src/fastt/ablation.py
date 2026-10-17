"""Table 5: the two ablations, as `notebooks/ablation` runs them.

**5a — the feature transformation.** For each corpus and FASTT variant, the best hold-out corpus WER reached by each
transformation (identity, diagonal, linear, low-rank, nonlinear) in the extended hyperparameter exploration on the
main trio: every configuration drawn from a discrete grid, scored with three inner folds on the hold-out the main
search uses. The trials are read from `results/<corpus>/exploration_<corpus>.json`; nothing is fitted.

**5b — the training objective.** On AMI-SDM, the same architecture, features, budget, partition and folds trained once
with cross-entropy on the per-clip best expert and once with the expected-WER objective, and compared with the paired
5x2cv tests. `LossAblation` reads (level 1) or refits (level 2) the three cross-entropy controls; the expected-WER rows
are the main run's own rows.
"""

__all__ = ['TRANSFORMS', 'TRANSFORM_LABELS', 'ABLATION_CORPORA', 'exploration_trials', 'transform_cells',
           'transform_table', 'transform_latex', 'LossAblation', 'LADDER', 'loss_latex']

import json

import numpy as np
import pandas as pd

from fastt.configs import MAIN, SELECTED, SELECTED_CE, SPACES
from fastt.crossval import ACC_TOLERANCES, METRICS, RepetitionSplit, results_table
from fastt.experiment import LABELS, MainExperiment
from fastt.pretty import Table
from fastt.records import load_record
from fastt.runlog import RunLog
from fastt.significance import significance_table
from fastt.tuning import CE_ARMS, CE_PARENT

TRANSFORMS = ("identity", "diagonal", "linear", "low_rank", "nonlinear")
TRANSFORM_LABELS = {"identity": "Raw", "diagonal": "Diag.", "linear": "Linear", "low_rank": "Low-rank",
                    "nonlinear": "Nonlin."}
ABLATION_CORPORA = ("voxpopuli", "ami_sdm", "earnings22", "peoples_speech")
CORPUS_TITLES = {"voxpopuli": "VoxPopuli", "ami_sdm": "AMI (SDM)", "earnings22": "Earnings-22",
                 "peoples_speech": "People's Speech"}
VARIANTS = ("fastt_boosted", "fastt_alternating")            # Table 5a's row order


# ---------------------------------------------------------------- Table 5a --

def exploration_trials(corpus):
    """The exploration's trials of both FASTT variants on the corpus's main trio, one row per trial."""
    rec = load_record(f"exploration_{corpus}", corpus)
    rows = [{"corpus": corpus, "arm": t["arm"], "trial": t["trial"],
             "transform": t["params"]["transform_type"], "corpus_wer": t["corpus_wer"]} for t in rec["trials"]]
    return pd.DataFrame(rows)


def transform_cells(corpora=ABLATION_CORPORA):
    """One row per (corpus, variant, transform): the best hold-out WER, the mean, how many trials, and whether it is the
    transform the main search selected."""
    rows = []
    for corpus in corpora:
        trials = exploration_trials(corpus)
        for arm in VARIANTS:
            chosen = SELECTED[corpus][arm]["transform_type"]
            for tf in TRANSFORMS:
                cell = trials[(trials["arm"] == arm) & (trials["transform"] == tf)]["corpus_wer"]
                rows.append({"corpus": corpus, "arm": arm, "transform": tf, "n_trials": int(len(cell)),
                             "best_wer": float(cell.min()) if len(cell) else np.nan,
                             "mean_wer": float(cell.mean()) if len(cell) else np.nan, "selected": tf == chosen})
    return pd.DataFrame(rows)


def transform_table(cells):
    """Table 5a as displayed: per corpus and variant, the best hold-out WER per transform; `*` marks the transform the
    main search selected, bold the best in the row."""
    out = []
    for (corpus, arm), g in cells.groupby(["corpus", "arm"], sort=False):
        g = g.set_index("transform")
        row = {"dataset": CORPUS_TITLES.get(corpus, corpus), "variant": LABELS.get(arm, arm)}
        for tf in TRANSFORMS:
            v = g.loc[tf]
            row[TRANSFORM_LABELS[tf]] = ("—" if not v["n_trials"] else f"{v['best_wer']:.4f}") + ("*" if v["selected"] else "")
        row["trials"] = int(g["n_trials"].sum())
        out.append(row)

    def best_in_row(col, value, row):
        if col not in TRANSFORM_LABELS.values() or value == "—":
            return None
        vals = [float(str(row[c]).rstrip("*")) for c in TRANSFORM_LABELS.values() if row[c] != "—"]
        return "font-weight:700" if float(str(value).rstrip("*")) == min(vals) else None

    return Table(pd.DataFrame(out), title="Table 5a — feature transformation: best hold-out corpus WER per transform",
                 caption="the extended exploration on each corpus's main trio; * = the transform the main search "
                         "selected (budget of 10); bold = best in the row", group="dataset", cell_style=best_in_row)


def transform_latex(cells):
    """Table 5a as the manuscript's `table*` (bold: best in the row; underlined: the main search's choice)."""
    lines = [r"\begin{table*}[t]", r"    \centering",
             r"    \caption{Ablation of the feature transformation: best validation corpus WER per transform in the "
             r"hyperparameter exploration (same members, features and partition as the main search; 15--40 "
             r"configurations per cell). Bold: best transform in the row. Underlined: the transform the main search "
             r"selected for Table~\ref{tab:main_merged} under its budget of 10 configurations. The two differ where the "
             r"budget landed on a transform the richer search ranks lower.}",
             r"    \label{tab:ablation_transform} \footnotesize \setlength{\tabcolsep}{3pt}",
             r"    \begin{adjustbox}{max width=\linewidth}", r"    \begin{tabular}{lccccc}", r"        \toprule",
             "        " + " & ".join([r"\textbf{Method}"] + [rf"\textbf{{{TRANSFORM_LABELS[t]}}}" for t in TRANSFORMS])
             + r" \\"]
    for i, corpus in enumerate(dict.fromkeys(cells["corpus"])):
        lines.append(("        \\midrule\n        " if i == 0 else "        \\midrule ")
                     + rf"\multicolumn{{6}}{{l}}{{\textit{{{CORPUS_TITLES.get(corpus, corpus)}}}}} \\")
        for arm in VARIANTS:
            g = cells[(cells["corpus"] == corpus) & (cells["arm"] == arm)].set_index("transform")
            best = g["best_wer"].min()
            out = []
            for tf in TRANSFORMS:
                v, s = g.loc[tf, "best_wer"], f"{g.loc[tf, 'best_wer']:.4f}"
                if v == best:
                    s = rf"\textbf{{{s}}}"
                if g.loc[tf, "selected"]:
                    s = rf"\underline{{{s}}}"
                out.append(s)
            lines.append("        " + " & ".join([LABELS.get(arm, arm)] + out) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"    \end{adjustbox}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- Table 5b --

# The ladder, rung by rung: (arm, architecture, loss). Read row against row, not against a winner.
LADDER = [
    ("argmin_clf", "linear", "plain CE"),
    ("probe", "linear", "cost-sensitive CE"),
    ("qe_router", "ridge regression", "per-expert WER, argmin"),
    ("adastt_ce", "XGBoost", "plain CE"),
    ("adastt", "XGBoost", "expected WER"),
    ("fastt_boosted_ce", "FASTT-Boosted", "plain CE"),
    ("fastt_boosted", "FASTT-Boosted", "expected WER"),
    ("fastt_alternating_ce", "FASTT-Alternating", "plain CE (both swaps)"),
    ("fastt_alternating", "FASTT-Alternating", "expected WER"),
]
CE_LABELS = {"adastt_ce": "Raw XGBoost (AdaSTT), CE", "fastt_boosted_ce": "FASTT-Boosted, CE",
             "fastt_alternating_ce": "FASTT-Alternating, CE"}
ARCH = {"adastt": "XGBoost", "fastt_boosted": "FASTT-Boosted", "fastt_alternating": "FASTT-Alternating"}


def loss_latex(df):
    """Table 5b as the manuscript's `table`, from `LossAblation.tests()`'s frame (or the record's `table5b`)."""
    lines = [r"\begin{table}[t]", r"    \centering",
             r"    \caption{Ablation of the training objective on AMI (SDM): the same architecture, features, "
             r"search budget, partition and 5$\times$2 folds, trained with cross-entropy (CE) or the expected-WER "
             r"objective (E[WER]) and evaluated on corpus WER; mean over the 10 folds, better in bold. $\Delta$: CE "
             r"minus E[WER] in WER points; $p$ is Alpaydın's combined 5$\times$2 cv $F$ test. Every difference is "
             r"significant at $p<0.05$.}",
             r"    \label{tab:ablation_loss_ami_sdm} \footnotesize \setlength{\tabcolsep}{3pt}",
             r"    \begin{adjustbox}{max width=\linewidth}", r"    \begin{tabular}{lrrrr}", r"        \toprule",
             r"        \textbf{Architecture} & \textbf{CE} & \textbf{E[WER]} & \textbf{$\Delta$} & \textbf{p (F)} \\",
             r"        \midrule"]
    for _, r in df.iterrows():
        ce, ew = f"{r['CE']:.4f}", f"{r['E[WER]']:.4f}"
        ce, ew = (rf"\textbf{{{ce}}}", ew) if r["CE"] < r["E[WER]"] else (ce, rf"\textbf{{{ew}}}")
        p = "$<$0.001" if r["p_alpaydin"] < 1e-3 else f"{r['p_alpaydin']:.3f}"
        lines.append(f"        {r['architecture']} & {ce} & {ew} & {r['delta_points']:+.2f} & {p} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"    \end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


class LossAblation(MainExperiment):
    """The loss ablation on one corpus (AMI-SDM in the paper). Each cross-entropy control is its parent with one flag
    flipped — trained on the argmin label with no WER-derived weights — searched over the parent's space with the main
    run's budget, sampler and partition, and scored on the main run's five 5x2 repetitions.

        abl = LossAblation(MAIN["ami_sdm"], level=1)
        abl.load(); abl.search(); abl.cross_validate()
        abl.ladder(); abl.tests(); abl.latex()
    """

    EXISTING = tuple(a for a, _, _ in LADDER if a not in CE_ARMS)

    def __init__(self, cfg=MAIN["ami_sdm"], level=1, retune=False, workdir=None):
        super().__init__(cfg, level=level, retune=retune, workdir=workdir)

    def search(self):
        """The expected-WER rungs read from the main run's searches; the CE controls from their own."""
        self.run = self._search(self.EXISTING)
        self.ce_run = self._search(CE_ARMS, spaces={ce: SPACES[CE_PARENT[ce]] for ce in CE_ARMS})
        assert np.array_equal(self.run.eval_rows, self.ce_run.eval_rows), "the CE search cut different rows"
        if not self.refit_search:
            self._pin(self.run)
            self._pin(self.ce_run, SELECTED_CE[self.cfg.dataset])
        self.results_all = {**self.run.results, **self.ce_run.results}
        print(f"  {len(self.results_all)} arms x {self.cfg.search_budget} trials"
              + (", searched on this machine" if self.refit_search else ", read from the Hub"))
        return self.ce_run

    def cross_validate(self):
        """Every rung on the main run's folds: the existing rungs and the references are the main run's own cached rows,
        the CE controls are fitted (level 2) or read (level 1) under the same cache."""
        c = self.cfg
        log = RunLog(f"main_{c.dataset}", dirname=self.workdir, spec=self.spec, sync=not self.refit_folds,
                     refresh=self.refit_folds, strict=not self.refit_folds)
        key = {"partition": self.ce_run.partition(), "n_eval": int(len(self.ce_run.eval_rows)),
               "random_seed": c.cv_random_seed, "member_arms": True, "tolerances": list(ACC_TOLERANCES),
               "achievable_margin": c.achievable_margin, "metrics": sorted(METRICS)}

        def one_repetition(seed, estimators):
            return self.ce_run.evaluate(estimators=estimators, scheme=RepetitionSplit(seed),
                                        random_seed=c.cv_random_seed, member_arms=True, tolerances=ACC_TOLERANCES,
                                        fold_table=self.refit_folds)

        self.fold_scores, self.fold_picks = log.per_arm(
            "cv5x2", c.cv_seeds, {**self.run.arms, **self.ce_run.arms}, one_repetition, key=key,
            arm_keys={**self.run.arm_keys(), **self.ce_run.arm_keys()})
        self.n_folds = int(self.fold_scores.drop_duplicates(["repeat", "fold"]).shape[0])
        return self.fold_scores, self.fold_picks

    def ladder(self):
        """Every rung: corpus WER, mean utterance WER, gap closed, mean ± sd over the folds."""
        order = [a for a, _, _ in LADDER]
        labels = {**LABELS, **CE_LABELS}
        tab = results_table(self.fold_scores, self.fold_picks, self.members, order=order, labels=labels)
        tab = tab[tab["model"].isin(order)].set_index("model").reindex(order).reset_index()
        self.tables["ladder"] = tab
        pm = lambda m, s, d=4, k=1.0: f"{k * m:.{d}f} ± {k * s:.{d}f}"                    # noqa: E731
        shown = pd.DataFrame({
            "arm": [labels.get(a, a) for a in tab["model"]],
            "architecture": [a for _, a, _ in LADDER], "loss": [l for _, _, l in LADDER],
            "corpus WER": [pm(m, s) for m, s in zip(tab["corpus_wer"], tab["corpus_wer_sd"])],
            "mean utt. WER": [pm(m, s) for m, s in zip(tab["mean_utt_wer"], tab["mean_utt_wer_sd"])],
            "GC@0 (%)": [pm(m, s, 1, 100) for m, s in zip(tab["gap_closed_tol0"], tab["gap_closed_tol0_sd"])]})
        return Table(shown, title=f"Loss ablation ladder — {self.cfg.title}",
                     caption=f"test folds, mean ± sd over {self.n_folds}",
                     best={"corpus WER": "min", "mean utt. WER": "min", "GC@0 (%)": "max"},
                     highlight=lambda r: r["loss"].startswith("expected"))

    def tests(self):
        """Table 5b: per architecture, the expected-WER arm against its CE twin on the identical folds."""
        rows = []
        for ce in CE_ARMS:
            parent = CE_PARENT[ce]
            t = significance_table(self.fold_scores, "corpus_wer", parent, [ce], alpha=self.cfg.alpha).iloc[0]
            means = self.fold_scores[self.fold_scores["metric"] == "corpus_wer"].groupby("model")["value"].mean()
            rows.append({"architecture": ARCH[parent], "CE": means[ce], "E[WER]": means[parent],
                         "delta_points": 100 * (means[ce] - means[parent]), "F_alpaydin": t["F_alpaydin"],
                         "p_alpaydin": t["p_alpaydin"], "t_dietterich": t["t_dietterich"],
                         "p_dietterich": t["p_dietterich"]})
        self.table5b = df = pd.DataFrame(rows)
        shown = df.assign(**{"CE": df["CE"].map("{:.4f}".format), "E[WER]": df["E[WER]"].map("{:.4f}".format),
                             "Δ (points)": df["delta_points"].map("{:+.2f}".format),
                             "p (F)": df["p_alpaydin"].map(lambda p: "<0.001" if p < 1e-3 else f"{p:.3f}")})
        return Table(shown[["architecture", "CE", "E[WER]", "Δ (points)", "p (F)"]],
                     title=f"Table 5b — training objective, {self.cfg.title}",
                     caption=f"corpus WER, mean over the {self.n_folds} folds; Δ = CE minus E[WER] in WER points; "
                             "p from Alpaydin's combined 5x2cv F-test", best={"CE": "min", "E[WER]": "min"})

    def latex(self):
        """Table 5b as the manuscript's `table`."""
        return loss_latex(self.table5b)

    def record(self):
        rec = lambda df: json.loads(df.to_json(orient="records", double_precision=15))  # noqa: E731
        return {"notebook": "ablation", "dataset": self.cfg.dataset, "level": self.level, "members": list(self.members),
                "ladder": LADDER, "loss_ladder": rec(self.tables["ladder"]), "table5b": rec(self.table5b),
                "ce_params": {a: self.ce_run.results[a].params for a in CE_ARMS},
                "ce_studies": self.ce_run.study_paths(), "n_folds": self.n_folds}

    def save(self, path=None):
        """Write the record (`ablation_<corpus>.json` in the working directory)."""
        return super().save(path or f"ablation_{self.cfg.dataset}.json")

"""The manuscript's tables, from the records the notebooks wrote (`notebooks/manuscript_tables`).

Each method displays one table and writes it as the manuscript inputs it — a complete float with its caption and label —
to `<out>/<name>.tex`. Every number comes from a record: `main_<corpus>.json` (Tables 1, 2, 3 and 6),
`timing_voxpopuli.json` (Table 4), `exploration_<corpus>.json` (Table 5a) and `ablation_ami_sdm.json` (Table 5b).
Run against the published records, every file is identical to the one in the repository's `tables/`, which is what
the paper compiles; `git diff tables/` after a run is the check.
"""

__all__ = ['ManuscriptTables', 'default_table_dir', 'PAPER_TABLES', 'FAMILIES']

from pathlib import Path

import pandas as pd

from fastt.ablation import loss_latex, transform_cells, transform_latex, transform_table
from fastt.configs import MAIN, SPACES
from fastt.experiment import LABELS
from fastt.pretty import Table
from fastt.records import load_record

# The files the manuscript inputs, in its order.
PAPER_TABLES = ("table1", "table2", "table3_merged", "table6_WER", "table4c_voxpopuli", "table4b_voxpopuli",
                "table4a_voxpopuli", "table5a", "table5b_ami_sdm")
CORPUS_TITLES = {"librispeech": "LibriSpeech", "voxpopuli": "VoxPopuli", "ami_sdm": "AMI (SDM)",
                 "earnings22": "Earnings-22", "peoples_speech": "People's Speech"}
# Table 3's row groups; the oracles are shown but never compete for bold.
FAMILIES = {
    "Reference": ["bsm"],
    "External baselines": ["random", "rover_w", "rover_conf", "cn_mbr", "score_fusion", "argmin_clf", "probe",
                           "qe_router"],
    "Classical feature selection": ["adastt", "raw_sdt", "kbest_xgboost", "kbest_sdt"],
    "FASTT": ["fastt_boosted", "fastt_alternating"],
    "Oracles": ["oracle", "oracle_tol1", "oracle_tol2"],
}
NOT_COMPETING = {"oracle", "oracle_tol1", "oracle_tol2"}
MERGED_METRICS = {"WER": ("corpus_wer", "min", "wer"), "GC": ("gap_closed_tol0", "max", "pct"),
                  "Acc": ("acc_tol0", "max", "pct")}
ALPHA = 0.05
TABLE1_DROP = {"Hours (est.)", "Reference words"}     # in the record, not in the manuscript
BEAMS = {"greedy": "greedy", "beam5": "beam 5", "beam20": "beam 20"}
TABLE_SIZE = r"\footnotesize"
TABLE_COLSEP = "3pt"

# Table 2 collapses the near-identical grids of each base learner into one representative arm.
SDT_ARM, XGB_ARM, KBEST_ARM = "fastt_boosted", "fastt_alternating", "kbest_xgboost"
SPACE_BASELINES = ["rover_w", "rover_conf", "cn_mbr", "score_fusion", "argmin_clf", "probe", "qe_router"]
PARAM_PRETTY = {
    "transform_type": "Feature transform", "lambda1": r"Transform $\ell_1$ ($\lambda_1$)",
    "lambda2": r"Transform $\ell_2$ ($\lambda_2$)", "out_ratio": "Bottleneck output ratio",
    "bottleneck_ratio": "Bottleneck ratio", "num_rounds": "Boosting rounds", "num_trees": "Trees per round",
    "depth": "Tree depth", "lmbda": r"SDT $\ell_1$ ($\lambda$)", "lmbda2": r"SDT $\ell_2$ ($\lambda_2$)",
    "adam_lr": "Learning rate", "adam_weight_decay": "Weight decay", "batch_size": "Batch size",
    "patience": "Early-stopping patience", "t_weight_decay": "Transform weight decay",
    "n_estimators": "Num. estimators", "learning_rate": "Learning rate", "max_depth": "Max depth",
    "subsample": "Subsample", "colsample": "Column subsample", "min_child_weight": "Min. child weight",
    "reg_lambda": r"$\ell_2$ regularization", "num_iterations": "Alternating iterations",
    "transform_lr": "Transform learning rate", "transform_steps": "Transform steps",
    "surrogate_lr": "Surrogate learning rate", "surrogate_steps": "Surrogate steps", "weight_decay": "Weight decay",
    "k": "Num. features ($k$)", "alpha": r"Smoothing ($\alpha$)", "conf_source": "Confidence source",
    "weighting": "Vote weighting", "weight_by_conf": "Weight by confidence", "conf_temp": "Confidence temperature",
    "null_penalty": "Null-arc penalty", "decode": "Decoding source", "candidates": "Candidate hypotheses",
    "prior": "Prior", "temperature": "Temperature", "null_prior": "Null prior", "use_conf": "Use confidence",
    "mode": "Fusion mode", "w_conf": "Confidence weight", "w_consensus": "Consensus weight",
    "w_length": "Length weight", "target": "Regression target", "alpha_lo": r"Lowest $\alpha$",
    "alpha_decades": r"$\alpha$ decades", "n_alphas": r"Num. $\alpha$ values",
    "standardize": "Standardize features", "objective_weight": "Objective weighting",
}


def default_table_dir():
    """`tables/` at the repository root when run from its `notebooks/`, else `tables/` here."""
    here = Path.cwd()
    if here.name == "notebooks" and (here.parent / "pyproject.toml").exists():
        return here.parent / "tables"
    return here / "tables"


# ------------------------------------------------- one table shape, two renderers --
# A table is a list of rows, each a kind (`group` header, `rule`, `plain` or indented `sub` row), a label and its cells; a cell is text
# and two flags, bold and underline, plus a superscript mark. The same object becomes a displayed frame and LaTeX.

def _cell(text, bold=False, under=False, mark=""):
    return {"text": text, "bold": bold, "under": under, "mark": mark}


def _row(kind, label, cells=()):
    return {"kind": kind, "label": label, "cells": list(cells)}


def _fmt(v, kind="wer"):
    if v is None:
        return "—"
    if kind == "pct":
        return f"{100 * v:.1f}"
    if kind == "ms":
        return f"{1000 * v:.1f}"
    if kind == "int":
        return f"{int(round(v)):,}"
    return f"{v:.4f}"


def _best_flags(values, direction, digits):
    """True where a value ties the best at the printed precision. `None` never wins."""
    xs = [None if v is None else round(v, digits) for v in values]
    live = [x for x in xs if x is not None]
    if not live:
        return [False] * len(xs)
    top = min(live) if direction == "min" else max(live)
    return [x is not None and x == top for x in xs]


def _tex(s):
    """Escape for LaTeX outside math; `$...$` spans pass through untouched."""
    out = []
    for i, seg in enumerate(str(s).split("$")):
        if i % 2 == 0:
            for a, b in (("±", r"{\scriptsize$\pm$}"), ("—", "---"), ("%", r"\%"), ("&", r"\&"), ("_", r"\_"),
                         ("×", r"$\times$"), ("Δ", r"$\Delta$"), ("µ", r"$\mu$"), ("<", r"$<$")):
                seg = seg.replace(a, b)
            out.append(seg)
        else:
            out.append("$" + seg + "$")
    return "".join(out)


def _tex_cell(c):
    s = _tex(c["text"])
    if c["bold"]:
        s = r"\textbf{" + s + "}"
    if c["under"]:
        s = r"\underline{" + s + "}"
    if c["mark"]:
        s += "$^{" + c["mark"] + "}$"
    return s


def _caption(s):
    return s.replace("%", r"\%").replace("&", r"\&")


def _latex(table, caption, label, colspec=None, env="table*", first="Method"):
    """A complete booktabs float; `adjustbox` shrinks only a table wider than the line."""
    ncol = len(table["columns"]) + 1
    colspec = colspec or "l" + "c" * (ncol - 1)
    head = " & ".join([c if c.startswith("$") else r"\textbf{" + _tex(c) + "}"
                       for c in [first] + table["columns"]]) + r" \\"
    if table.get("groups"):
        spans, rules, at = [], [], 2
        for title, n in table["groups"]:
            spans.append(r"\multicolumn{%d}{c}{\textbf{%s}}" % (n, _tex(title)))
            rules.append(r"\cmidrule(lr){%d-%d}" % (at, at + n - 1))
            at += n
        head = " & ".join([""] + spans) + r" \\ " + " ".join(rules) + "\n    " + head
    body = []
    for i, r in enumerate(table["rows"]):
        if r["kind"] == "group":
            body.append(("" if i == 0 else r"\midrule ") + r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, _tex(r["label"])))
        elif r["kind"] == "rule":
            if i:
                body.append(r"\midrule")
        else:
            lab = (r"\quad " if r["kind"] == "sub" else "") + _tex(r["label"])
            body.append(" & ".join([lab] + [_tex_cell(c) for c in r["cells"]]) + r" \\")
    return "\n".join([
        r"\begin{%s}[t]" % env, r"    \centering", r"    \caption{%s}" % _caption(caption),
        r"    \label{%s} %s \setlength{\tabcolsep}{%s}" % (label, TABLE_SIZE, TABLE_COLSEP),
        r"    \begin{adjustbox}{max width=\linewidth}", r"    \begin{tabular}{%s}" % colspec, r"        \toprule",
        "        " + head, r"        \midrule", "\n".join("        " + b for b in body), r"        \bottomrule",
        r"    \end{tabular}", r"    \end{adjustbox}", r"\end{%s}" % env])


def _latex_two_up(table, caption, label, first="Parameter"):
    """A two-column list (Table 2) as two side-by-side halves of a `table*`, split at the group boundary nearest the
    middle so no arm's rows are separated."""
    col = table["columns"][0]
    items = []
    for r in table["rows"]:
        if r["kind"] == "group":
            items.append((bool(items), "group", r["label"], ""))
        else:
            items.append((False, "row", r["label"], r["cells"][0]["text"]))
    groups_at = [i for i, it in enumerate(items) if it[1] == "group"]
    cut = min(groups_at[1:], key=lambda i: abs(i - len(items) / 2))
    left, right = items[:cut], items[cut:]
    right[0] = (False,) + right[0][1:]

    def side(it):
        if it is None:
            return " & "
        _, kind, lab, text = it
        return r"\multicolumn{2}{l}{\textit{%s}}" % _tex(lab) if kind == "group" else _tex(lab) + " & " + _tex(text)

    body = []
    for i in range(max(len(left), len(right))):
        lt = left[i] if i < len(left) else None
        rt = right[i] if i < len(right) else None
        rules = (r"\cmidrule(lr){1-2}" if lt and lt[0] else "") + (r"\cmidrule(lr){3-4}" if rt and rt[0] else "")
        if rules:
            body.append("        " + rules)
        body.append("        " + side(lt) + " & " + side(rt) + r" \\")
    head = " & ".join([r"\textbf{%s}" % _tex(first), r"\textbf{%s}" % _tex(col)] * 2) + r" \\"
    return "\n".join([
        r"\begin{table*}[t]", r"    \centering", r"    \caption{%s}" % _caption(caption),
        r"    \label{%s} %s \setlength{\tabcolsep}{%s}" % (label, TABLE_SIZE, TABLE_COLSEP),
        r"    \begin{adjustbox}{max width=\linewidth}", r"    \begin{tabular}{ll@{\hspace{2em}}ll}",
        r"        \toprule", "        " + head, r"        \midrule", "\n".join(body), r"        \bottomrule",
        r"    \end{tabular}", r"    \end{adjustbox}", r"\end{table*}"])


def _shown(table, title, caption, first=""):
    """The same table as a displayed frame: group rows become group headers, bold and underline survive."""
    names = table["columns"]
    if table.get("groups"):
        flat = [g for g, n in table["groups"] for _ in range(n)]
        names = [f"{g} · {c}" if g else c for g, c in zip(flat, names)]
    group, records, styles = None, [], {}
    for r in table["rows"]:
        if r["kind"] in ("group", "rule"):
            group = r["label"]
            continue
        i = len(records)
        records.append({"_group": group or "", first or " ": ("\u2003" if r["kind"] == "sub" else "") + r["label"],
                        **{n: c["text"] + (" †" if c["mark"] else "") for n, c in zip(names, r["cells"])}})
        for n, c in zip(names, r["cells"]):
            css = ("font-weight:700;" if c["bold"] else "") + ("text-decoration:underline;" if c["under"] else "")
            if css:
                styles[(i, n)] = css
    df = pd.DataFrame(records)
    grouped = df["_group"].ne("").any()
    if not grouped:
        df = df.drop(columns="_group")
    return Table(df, title=title, caption=caption, group="_group" if grouped else None,
                 cell_style=lambda col, value, row: styles.get((row.name, col)))


# ------------------------------------------------------------------- the tables --

class ManuscriptTables:
    """Every table of the manuscript. `out` is where the `.tex` files go (default: `tables/` at the repository root);
    `corpora` are the main records read, in the column order of the tables."""

    def __init__(self, out=None, corpora=tuple(MAIN), timing="voxpopuli", loss="ami_sdm"):
        self.out = Path(out) if out is not None else default_table_dir()
        self.out.mkdir(parents=True, exist_ok=True)
        self.corpora, self.timing_corpus, self.loss_corpus = tuple(corpora), timing, loss
        self.main = {c: load_record(f"main_{c}", c) for c in self.corpora}
        self.t3 = {c: {r["model"]: r for r in self.main[c]["tables"]["table3"]} for c in self.corpora}
        self.titles = {c: CORPUS_TITLES.get(c, c) for c in self.corpora}
        self.written = {}
        print(f"read {len(self.main)} main records; the LaTeX goes to {self.out.name}/")

    def _write(self, name, text):
        (self.out / f"{name}.tex").write_text(text + "\n")
        self.written[name] = text
        return f"→ {self.out.name}/{name}.tex"

    def _bsm(self, c):
        return self.main[c].get("bsm_per_fold") or self.main[c]["bsm"]

    # ---------------------------------------------------------------- Table 1 --

    def datasets(self):
        """Table 1: clips, features, the hyperparameter hold-out and the 5x2 evaluation rows per corpus."""
        first = self.main[self.corpora[0]]["tables"]["table1"]
        rows = []
        for stat in [r["statistic"] for r in first if r["statistic"] not in TABLE1_DROP]:
            cells = [_cell(_fmt(next(r for r in self.main[c]["tables"]["table1"] if r["statistic"] == stat)[c], "int"))
                     for c in self.corpora]
            rows.append(_row("plain", stat, cells))
        t = {"columns": list(self.titles.values()), "rows": rows}
        where = self._write("table1", _latex(
            t, "Dataset statistics. Clips are those with non-empty audio and reference; the hyperparameter "
               "hold-out (25%) is disjoint from the 5$\\times$2 cv evaluation rows.",
            "tab:datasets", colspec="l" + "c" * len(self.corpora), first="Statistic"))
        return _shown(t, "Table 1 — datasets", where, first="statistic")

    def experts(self):
        """The expert trio per corpus with each expert's test WER (mean over the ten folds); bold: the best single
        model. Shown, not written: the manuscript names the trios in its text."""
        rows = []
        for c in self.corpora:
            cells = [_cell(f"{LABELS.get(m, m)} ({self.t3[c][m]['corpus_wer']:.4f})", bold=(m == self._bsm(c)))
                     for m in self.main[c]["members"]]
            rows.append(_row("plain", self.titles[c], cells))
        t = {"columns": ["expert 1", "expert 2", "expert 3"], "rows": rows}
        return _shown(t, "The experts", "test corpus WER in brackets; bold: the best single model", first="corpus")

    # ---------------------------------------------------------------- Table 2 --

    @staticmethod
    def _choices(p):
        if p.frozen is not None:
            return f"fixed: {p.frozen}"
        ch = list(p.choices)
        try:
            ch = sorted(ch)
        except TypeError:
            pass
        if len(ch) <= 1:
            return f"fixed: {ch[0]}"
        if len(ch) > 5 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in ch):
            return "{" + f"{ch[0]:g}, {ch[1]:g}, \\ldots, {ch[-1]:g} ({len(ch)} values)" + "}"
        return "{" + ", ".join(str(c) for c in ch) + "}"

    def _swept(self, arm, strip=""):
        out = []
        for name, p in SPACES[arm].items():
            if name == "transform_type" or p.frozen is not None or len(p.choices) <= 1:
                continue
            label = name[len(strip):] if strip and name.startswith(strip) else name
            out.append((PARAM_PRETTY.get(label, label.replace("_", " ").capitalize()), self._choices(p)))
        return out

    def spaces(self):
        """Table 2: the free hyperparameters and their grids (the same on every corpus). The FASTT, raw and
        SelectKBest variants of a base learner share one grid, shown once; the feature transform is Table 5a's."""
        rows = [_row("group", "SDT")] + [_row("plain", n, [_cell(c)]) for n, c in self._swept(SDT_ARM)]
        rows += [_row("group", "XGBoost")] + [_row("plain", n, [_cell(c)]) for n, c in self._swept(XGB_ARM, "sel_")]
        rows += [_row("group", "SelectKBest"),
                 _row("plain", PARAM_PRETTY["k"], [_cell(self._choices(SPACES[KBEST_ARM]["k"]))])]
        for arm in SPACE_BASELINES:
            rows += [_row("group", LABELS[arm])] + [_row("plain", n, [_cell(c)]) for n, c in self._swept(arm)]
        t = {"columns": ["Search space"], "rows": rows}
        budget = self.main[self.corpora[0]]["config"]["search_budget"]
        where = self._write("table2", _latex_two_up(
            t, f"Hyperparameter search spaces. Every arm draws {budget} configurations with the same sampler and seed "
               "on the corpus's hold-out partition; fixed parameters (a single-choice grid) are omitted. The "
               "SDT and XGBoost groups collapse the near-identical grids of the FASTT, raw and SelectKBest variants "
               "of each base learner into one representative set; $k$ is the SelectKBest feature count.",
            "tab:spaces"))
        return _shown(t, "Table 2 — hyperparameter spaces", where, first="parameter")

    # ---------------------------------------------------------------- Table 3 --

    def _anchor_beats(self, c, arm, metric="corpus_wer"):
        """Whether corpus `c`'s FASTT anchor is significantly better than `arm` (Alpaydın's F, p < 0.05)."""
        t6 = self.main[c]["tables"]["table6"][metric]
        if arm == t6["anchor"]:
            return False
        r = next((r for r in t6["rows"] if r["comparator"] == arm), None)
        return r is not None and r["p_alpaydin"] is not None and r["p_alpaydin"] < ALPHA and r["effect"] > 0

    def main_results(self):
        """Table 3: test corpus WER, the share of the oracle gap closed (GC, %) and the selection accuracy (Acc, %),
        mean over the ten folds. Bold: best per corpus and metric among the competing arms; †: the corpus's FASTT
        anchor is significantly better on WER."""
        digits = {"wer": 4, "pct": 3}
        competing = [a for fam in FAMILIES.values() for a in fam if a not in NOT_COMPETING]
        best = {}
        for c in self.corpora:
            for col, (key, direction, kind) in MERGED_METRICS.items():
                flags = _best_flags([self.t3[c][a][key] for a in competing], direction, digits[kind])
                best.update({(c, col, a): f for a, f in zip(competing, flags)})
        rows = []
        for family, arms in FAMILIES.items():
            rows.append(_row("group", family))
            for arm in arms:
                cells = []
                for c in self.corpora:
                    for col, (key, _, kind) in MERGED_METRICS.items():
                        mark = r"\dagger" if (col == "WER" and arm not in NOT_COMPETING
                                              and self._anchor_beats(c, arm)) else ""
                        cells.append(_cell(_fmt(self.t3[c][arm][key], kind), bold=best.get((c, col, arm), False),
                                           mark=mark))
                rows.append(_row("plain", LABELS[arm], cells))
        t = {"columns": list(MERGED_METRICS) * len(self.corpora),
             "groups": [(self.titles[c], len(MERGED_METRICS)) for c in self.corpora], "rows": rows}
        where = self._write("table3_merged", _latex(
            t, "Test corpus WER, share of the oracle gap closed (GC, %), and selection accuracy (Acc, %), "
               "5$\\times$2 cv mean over the 10 folds. Bold: best per dataset and metric among the competing arms "
               "(oracles excluded). $^{\\dagger}$: the corpus's FASTT anchor (the better FASTT arm) is significantly "
               "better at $p<0.05$ (paired 5$\\times$2 cv $F$ test). Fusion arms produce a new hypothesis and have "
               "no accuracy.",
            "tab:main_merged", colspec="l" + "r" * (len(MERGED_METRICS) * len(self.corpora))))
        return _shown(t, "Table 3 — main results", where + "; bold: best per corpus and metric; † the FASTT anchor is "
                      "significantly better on WER", first="method")

    # ---------------------------------------------------------------- Table 6 --

    def significance(self, metric="corpus_wer"):
        """Table 6: Alpaydın's combined 5x2cv F statistic and its p-value, every arm against the corpus's FASTT anchor
        (the better FASTT arm on corpus WER); the anchor's own cells are `-`."""
        col = {"corpus_wer": "WER", "mean_utt_wer": "uWER"}[metric]
        rows = []
        for family, arms in FAMILIES.items():
            rows.append(_row("rule", family))
            for arm in arms:
                cells = []
                for c in self.corpora:
                    t6 = self.main[c]["tables"]["table6"][metric]
                    r = next((r for r in t6["rows"] if r["comparator"] == arm), None)
                    if arm == t6["anchor"]:
                        cells += [_cell("-"), _cell("-")]
                    elif r is None or r["F_alpaydin"] is None:
                        cells += [_cell("—"), _cell("—")]
                    else:
                        p = r["p_alpaydin"]
                        cells += [_cell(f"{r['F_alpaydin']:.2f}"), _cell("<0.001" if p < 0.001 else f"{p:.3f}")]
                rows.append(_row("plain", LABELS[arm], cells))
        t = {"columns": ["$F$", "$p$"] * len(self.corpora), "groups": [(self.titles[c], 2) for c in self.corpora],
             "rows": rows}
        where = self._write(f"table6_{col}", _latex(
            t, f"Paired 5$\\times$2 cv test against the corpus's FASTT anchor on {col}: the Alpaydın combined $F$ "
               "statistic and its p-value.",
            f"tab:sig_{col.lower()}", colspec="l" + "r" * (2 * len(self.corpora)), first="Comparator"))
        anchors = ", ".join(f"{self.titles[c]}: {LABELS[self.main[c]['tables']['table6'][metric]['anchor']]}"
                            for c in self.corpora)
        return _shown(t, f"Table 6 — {col}, every arm against the FASTT anchor", f"{where}; anchors — {anchors}",
                      first="comparator")

    # ---------------------------------------------------------------- Table 4 --

    def cost(self):
        """Table 4 (VoxPopuli, seconds per clip at batch size 1 on one GPU): 4c the selector overhead of every arm at
        greedy decoding, 4b the cost of decoding every expert against choosing first, 4a the per-expert clocks."""
        d = self.timing_corpus
        rec = load_record(f"timing_{d}", d)
        where = f"{CORPUS_TITLES.get(d, d)}, {rec['n_clips']} clips, batch size {rec['batch_size']}"

        by = {r["model"]: r for r in rec["table4"]}
        rows = [_row("group", "Members")]
        for m in rec["members"]:
            r = by[m]
            rows.append(_row("plain", LABELS[m], [_cell(_fmt(r["encode"], "ms")), _cell(_fmt(r["decode"], "ms")),
                                                  _cell("—"), _cell(_fmt(r["end_to_end"], "ms")), _cell("—")]))
        for family, arms in FAMILIES.items():
            rows.append(_row("group", family))
            for arm in arms:
                r = by.get(arm)
                if r is None:
                    continue
                rows.append(_row("plain", LABELS[arm], [
                    _cell(_fmt(r["encode"], "ms")), _cell(_fmt(r["decode"], "ms")),
                    _cell(f"{1e6 * r['selector_infer']:.1f}" if r["selector_infer"] else "—"),
                    _cell(_fmt(r["end_to_end"], "ms")),
                    _cell("—" if r["train_selector"] == 0 else f"{r['train_selector']:.2f}")]))
        t4c = {"columns": ["encode (ms)", "decode (ms)", "selector (µs)", "end-to-end (ms)", "train selector (s)"],
               "rows": rows}
        w4c = self._write(f"table4c_{d}", _latex(
            t4c, f"Selector overhead ({where}, greedy decoding): time per utterance to encode all members, decode the "
                 "chosen member(s), and run the selector; end-to-end is their sum. Train: fitting the selector on one "
                 "5$\\times$2 fold. ROVER decodes every member before voting. CN-MBR and Score-Fusion instead pick a "
                 "single member's decode from confidence/score signals computed before decoding, like a selector, "
                 "which is why their decode cost is far below ROVER's rather than the sum of all three.",
            f"tab:overhead_{d}", colspec="l" + "r" * len(t4c["columns"])))

        names = {"decode all, then choose": "Decode all, then choose (fusion)",
                 "choose from encoders (FASTT)": "Choose from encoders, decode one (FASTT)",
                 "cheapest decoder only": "Cheapest decoder only (floor)"}
        saved = {s["decoding"]: s for s in rec["savings"]}
        rows = []
        for beam, title in BEAMS.items():
            rows.append(_row("group", f"Decoding: {title}"))
            for r in [r for r in rec["ensemble_cost"] if r["decoding"] == beam]:
                ours = r["kind"] == "before decoding"
                rows.append(_row("sub" if r["kind"] == "member" else "plain",
                                 LABELS.get(r["system"], names.get(r["system"], r["system"])), [
                    _cell(_fmt(r["encode"], "ms")), _cell(_fmt(r["decode"], "ms")),
                    _cell(f"{1e6 * r['selector']:.1f}" if r["selector"] else "—"),
                    _cell(_fmt(r["total"], "ms"), bold=ours), _cell(f"{r['x_after']:.2f}"),
                    _cell(f"{100 * saved[beam]['saved_fraction']:.0f}%" if ours else "—")]))
        t4b = {"columns": ["encode (ms)", "decode (ms)", "selector (µs)", "total (ms)", "× decode-all", "saved"],
               "rows": rows}
        w4b = self._write(f"table4b_{d}", _latex(
            t4b, f"Cost per utterance of the two ensemble strategies ({where}). Every strategy encodes all members "
                 "once; a fusion system then decodes every member, FASTT decodes only the selected one. `saved' is the "
                 "fraction of the decode-all time FASTT avoids; the floor always decodes the cheapest member.",
            f"tab:systems_{d}", colspec="l" + "r" * len(t4b["columns"]), first="System"))

        rows = []
        for r in rec["decoding"]:
            cells = [_cell(f"{r['encode_ms']:.1f}")] + [_cell(f"{r[f'{k}_decode_ms']:.1f}") for k in BEAMS]
            cells += [_cell(f"{r[f'{k}_wer']:.4f}") for k in BEAMS]
            rows.append(_row("plain", LABELS[r["model"]], cells))
        t4a = {"columns": ["encode (ms)"] + list(BEAMS.values()) * 2,
               "groups": [("", 1), ("decode (ms)", len(BEAMS)), ("WER", len(BEAMS))], "rows": rows}
        w4a = self._write(f"table4a_{d}", _latex(
            t4a, f"Per-member encoder and decoder time per utterance ({where}) at three decoding settings, with the "
                 "WER on the timed clips. The encoder is run once; the decoder's share grows with the beam.",
            f"tab:clocks_{d}", colspec="l" + "r" * len(t4a["columns"]), first="Member"))
        return (_shown(t4c, f"Table 4c — selector overhead per clip, greedy ({where})", w4c, first="method"),
                _shown(t4b, f"Table 4b — decode every expert, or choose first ({where})", w4b, first="system"),
                _shown(t4a, f"Table 4a — per-expert clocks ({where})", w4a, first="expert"))

    # ---------------------------------------------------------------- Table 5 --

    def ablation(self):
        """Table 5a, the feature transformation (best hold-out WER per transform in the exploration on each corpus's
        main trio), and Table 5b, the training objective on AMI-SDM (cross-entropy against expected WER)."""
        cells = transform_cells()
        w5a = self._write("table5a", transform_latex(cells))
        a = transform_table(cells)
        a.caption += f"; {w5a}"
        rec = load_record(f"ablation_{self.loss_corpus}", self.loss_corpus)
        df = pd.DataFrame(rec["table5b"])
        w5b = self._write(f"table5b_{self.loss_corpus}", loss_latex(df))
        shown = pd.DataFrame({"architecture": df["architecture"], "CE": df["CE"].map("{:.4f}".format),
                              "E[WER]": df["E[WER]"].map("{:.4f}".format),
                              "Δ (points)": df["delta_points"].map("{:+.2f}".format),
                              "p (F)": df["p_alpaydin"].map(lambda p: "<0.001" if p < 1e-3 else f"{p:.3f}")})
        b = Table(shown, title=f"Table 5b — training objective, {CORPUS_TITLES.get(self.loss_corpus)}",
                  caption=f"corpus WER, mean over the {rec['n_folds']} folds; Δ = CE minus E[WER] in WER points; "
                          f"p from Alpaydın's combined 5x2cv F test; {w5b}", best={"CE": "min", "E[WER]": "min"})
        return a, b

    # ------------------------------------------------------------------ all --

    def all(self):
        """Every table, in the manuscript's order of appearance; returns the files written."""
        from fastt.pretty import show
        show(self.datasets(), self.spaces(), self.main_results(), self.significance(), *self.cost(), *self.ablation())
        return [self.out / f"{n}.tex" for n in PAPER_TABLES]

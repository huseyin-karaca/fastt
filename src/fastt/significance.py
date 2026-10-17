"""Paired significance tests over 5x2 cross-validation folds.

Every arm is scored on the same ten folds (five repetitions, each halving the evaluation rows both ways), so two
arms are compared fold by fold: Dietterich's 5x2cv paired t-test and Alpaydin's combined 5x2cv F-test, which uses
all ten differences and is the one the paper reports.
"""

__all__ = ['SIGNIFICANCE_COLUMNS', 'TestResult', 'fold_differences', 'BaseSignificanceTest', 'FiveByTwoCVTest',
           'FiveByTwoCVFTest', 'summarise', 'better_arm', 'repetition_means', 'significance_table',
           'format_significance', 'print_significance']

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from fastt.pretty import HEX, show_table


@dataclass
class TestResult:
    """One comparison of two models under one metric."""

    test: str
    model_a: str
    model_b: str
    metric: str
    statistic: float
    p_value: float
    dof: str
    effect: float                       # mean signed difference, metric's own units
    better: str
    detail: dict = field(default_factory=dict)

    def as_row(self):
        d = {k: v for k, v in self.__dict__.items() if k != "detail"}
        return {**d, **self.detail}


def fold_differences(fold_scores, model_a, model_b, metric="corpus_wer", members=None):
    """`(n_repeats, 2)` signed differences, positive = `model_a` is better.

    Sign correction is driven by the `direction` column written by `run_cv`, so
    a lower-is-better metric and a higher-is-better one both come out with
    "positive favours A" — which is what lets one statistic serve every metric.
    """
    df = fold_scores[fold_scores["metric"] == metric]
    if members is not None:
        df = df[df["members"] == members]
    elif df["members"].nunique() > 1:
        raise ValueError(f"fold_scores covers {df['members'].nunique()} groups; "
                         "pass members=... to choose one")

    df = df[df["model"].isin([model_a, model_b])]
    missing = {model_a, model_b} - set(df["model"])
    if missing:
        raise KeyError(f"no folds for {sorted(missing)} under metric {metric!r}")

    wide = df.pivot_table(index=["repeat", "fold"], columns="model", values="value")
    direction = df["direction"].iloc[0]
    # "lower is better" -> A wins when it is smaller, so the difference is B - A.
    diff = (wide[model_b] - wide[model_a] if direction == "lower"
            else wide[model_a] - wide[model_b])

    D = diff.unstack("fold").to_numpy(dtype=np.float64)
    if D.ndim != 2 or D.shape[1] != 2:
        raise ValueError(f"expected 2 folds per repetition, got shape {D.shape}")
    if not np.isfinite(D).all():
        raise ValueError(f"metric {metric!r} produced non-finite folds for "
                         f"{model_a} vs {model_b}; it may be undefined here")
    return D


def _repeat_variances(D):
    """`s_i^2` per repetition: squared deviations of the two folds about their mean."""
    return ((D - D.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)


class BaseSignificanceTest(ABC):
    """A paired test over two models' cached fold results.

    Subclasses declare what they consume — `"fold_scores"` for anything computed
    from per-fold aggregates, `"fold_picks"` for tests that need per-utterance
    decisions (McNemar) — and implement `run`.
    """

    name = "base"
    consumes = "fold_scores"

    @abstractmethod
    def run(self, data, model_a, model_b, metric="corpus_wer", members=None) -> TestResult:
        ...

    def _verdict(self, p, effect, model_a, model_b, alpha=0.05):
        if not np.isfinite(p) or p > alpha or effect == 0:
            return "—"
        return model_a if effect > 0 else model_b


class FiveByTwoCVTest(BaseSignificanceTest):
    """Dietterich's 5x2cv paired t-test. Two-sided, `t_R` with R repetitions."""

    name = "5x2cv_t"

    def run(self, data, model_a, model_b, metric="corpus_wer", members=None, alpha=0.05):
        D = fold_differences(data, model_a, model_b, metric, members)
        R = len(D)
        var = _repeat_variances(D).mean()

        if var <= 0:
            # Identical on every fold, or a degenerate metric. No test to run.
            t, p = float("nan"), float("nan")
        else:
            t = D[0, 0] / np.sqrt(var)
            p = float(2 * stats.t.sf(abs(t), df=R))

        effect = float(D.mean())
        return TestResult(
            test=self.name, model_a=model_a, model_b=model_b, metric=metric,
            statistic=float(t), p_value=p, dof=str(R), effect=effect,
            better=self._verdict(p, effect, model_a, model_b, alpha),
            detail={"numerator": float(D[0, 0]), "sd": float(np.sqrt(var)),
                    "n_repeats": R},
        )


class FiveByTwoCVFTest(BaseSignificanceTest):
    """Alpaydin's combined 5x2cv F-test. Uses all 2R differences; `F(2R, R)`."""

    name = "5x2cv_f"

    def run(self, data, model_a, model_b, metric="corpus_wer", members=None, alpha=0.05):
        D = fold_differences(data, model_a, model_b, metric, members)
        R = len(D)
        denom = 2 * _repeat_variances(D).sum()

        if denom <= 0:
            f, p = float("nan"), float("nan")
        else:
            f = (D ** 2).sum() / denom
            p = float(stats.f.sf(f, dfn=2 * R, dfd=R))

        # The F-test is two-sided by construction (it squares the differences),
        # so the direction of the effect has to come from the mean difference.
        effect = float(D.mean())
        return TestResult(
            test=self.name, model_a=model_a, model_b=model_b, metric=metric,
            statistic=float(f), p_value=p, dof=f"({2 * R}, {R})", effect=effect,
            better=self._verdict(p, effect, model_a, model_b, alpha),
            detail={"numerator": float((D ** 2).sum()), "sd": float(np.sqrt(
                _repeat_variances(D).mean())), "n_repeats": R},
        )


def summarise(fold_scores, metric="corpus_wer", members=None):
    """Mean and standard deviation of each model's metric across the folds."""
    df = fold_scores[fold_scores["metric"] == metric]
    if members is not None:
        df = df[df["members"] == members]
    out = (df.groupby("model")["value"].agg(["mean", "std", "min", "max", "count"])
           .rename(columns={"count": "n_folds"}))
    ascending = df["direction"].iloc[0] == "lower"
    return out.sort_values("mean", ascending=ascending)


def repetition_means(fold_scores, metric="corpus_wer", members=None, models=None):
    """Mean of each model's metric within each repetition: one row per repetition (labelled by its seed)."""
    df = fold_scores[fold_scores["metric"] == metric]
    if members is not None:
        df = df[df["members"] == members]
    if models is not None:
        df = df[df["model"].isin(list(models))]
    index = "repeat_seed" if "repeat_seed" in df.columns else "repeat"
    wide = df.pivot_table(index=index, columns="model", values="value")
    wide.attrs["direction"] = df["direction"].iloc[0]
    return wide


def better_arm(fold_scores, candidates, metric="corpus_wer", members=None):
    """Whichever of `candidates` has the better mean over every fold present. Returns `(winner, means)`.

    The paper tests every arm against the better of the two FASTT variants on each corpus (the corpus's FASTT
    anchor); this is that choice, made on the reported folds and returned with the means it was made on.
    """
    wide = repetition_means(fold_scores, metric, members, models=candidates)
    missing = set(candidates) - set(wide.columns)
    if missing:
        raise KeyError(f"no folds for {sorted(missing)} under metric {metric!r}")
    means = wide[list(candidates)].mean()
    winner = (means.idxmin() if wide.attrs["direction"] == "lower" else means.idxmax())
    return winner, means


# The columns of Table 6, in order.
SIGNIFICANCE_COLUMNS = ("t_dietterich", "p_dietterich", "F_alpaydin", "p_alpaydin")
TESTS = {"dietterich": FiveByTwoCVTest, "alpaydin": FiveByTwoCVFTest}


def significance_table(fold_scores, metric, anchor, comparators, alpha=0.05, members=None, tests=None):
    """Every comparator against one anchor, the tests side by side. One row per comparator.

    Positive `effect` means the anchor is better, in the metric's own units. `tests` maps a column key to a test
    instance (`{"dietterich": FiveByTwoCVTest(), "alpaydin": FiveByTwoCVFTest()}` by default); the F-test's
    statistic lands in `F_<key>`, every other in `t_<key>`, and each p-value in `p_<key>`.
    """
    tests = tests or {k: cls() for k, cls in TESTS.items()}
    rows = []
    for rival in comparators:
        if rival == anchor:
            continue
        row = {"comparator": rival}
        for key, test in tests.items():
            stat = f"F_{key}" if isinstance(test, FiveByTwoCVFTest) else f"t_{key}"
            try:
                r = test.run(fold_scores, anchor, rival, metric, members, alpha=alpha)
                row[stat], row[f"p_{key}"] = r.statistic, r.p_value
                row["effect"] = r.effect
                for k in getattr(test, "detail_columns", ()):
                    row[f"{key}_{k}"] = r.detail.get(k)
            except (KeyError, ValueError) as e:
                row[stat] = row[f"p_{key}"] = np.nan
                row.setdefault("effect", np.nan)
                row["note"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.attrs.update(anchor=anchor, metric=metric, alpha=alpha)
    first = ["comparator", "effect"] + [c for c in out.columns if c[:2] in ("t_", "F_", "p_")]
    return out[first + [c for c in out.columns if c not in first]]


def format_significance(table, labels=None, digits=4, columns=None):
    """A `significance_table` as manuscript strings: signed effect, statistics, p-values.

    `effect` is signed to `digits` places, the t and F statistics to three, the p-values to `digits`; an undefined
    cell is `—`.
    """
    columns = list(columns or [c for c in SIGNIFICANCE_COLUMNS if c in table.columns])
    show = table[["comparator", "effect", *columns]].copy()
    if labels:
        show["comparator"] = show["comparator"].map(lambda k: labels.get(k, k))

    def cell(c, v):
        if v is None or not np.isfinite(v):
            return "—"
        if c == "effect":
            return f"{v:+.{digits}f}"
        return f"{v:.{digits}f}" if c.startswith("p_") else f"{v:.3f}"

    for c in ["effect", *columns]:
        show[c] = [cell(c, v) for v in table[c]]
    return show


def _significance_cell_style(alpha):
    """Colour rule for one cell: sign for effects and t, significance for p."""
    def style(col, value, row):
        try:
            v = float(str(value).replace("—", "nan"))
        except ValueError:
            return None
        if not np.isfinite(v):
            return None
        if col == "effect" or col.startswith("t_"):
            return f"color:{HEX['green' if v > 0 else 'red']}" if v != 0 else None
        if col.startswith("p_"):
            eff = float(str(row["effect"]).replace("—", "nan"))
            if v < alpha:
                return (f"font-weight:600;color:{HEX['green' if eff > 0 else 'red']}")
            return "opacity:.55"
        return None
    return style


def print_significance(table, labels=None, digits=4, title=None, caption=None, columns=None, note=None,
                       header_style=None):
    """Show a `significance_table` with its anchor and metric.

    In a notebook this is a styled table — positive effects and t green, negative red; p-values below `alpha` in
    bold and the rest dimmed. Anywhere else it is the same text, plain. Returns the formatted frame.
    """
    a = table.attrs
    alpha = float(a.get("alpha", 0.05))
    head = (f"anchor: {a.get('anchor')}   metric: {a.get('metric')}   "
            f"(positive effect = anchor better; p < {alpha:g} emphasised)")
    caption = f"{caption}\n{head}" if caption else head
    show = format_significance(table, labels, digits, columns)
    show_table(show, title=title, caption=caption, note=note, cell_style=_significance_cell_style(alpha),
               header_style=header_style)
    return show

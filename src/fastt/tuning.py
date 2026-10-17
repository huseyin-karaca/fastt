__all__ = ['SELECTOR_PARAMS', 'LOOP_PARAMS', 'TREE_PARAMS', 'MODELS', 'OURS', 'COMPETITORS', 'COSTS', 'OBJECTIVES', 'SELECT_POLICIES', 'SAMPLERS', 'TOP_FRAC', 'EDGE', 'PRESSURE', 'SPAN_MIN', 'IMPORTANCE_FLOOR', 'MIN_EVIDENCE', 'MIN_FREEZE_EVIDENCE', 'EXPAND', 'MARGIN', 'DISPLAY', 'PARTITION_FIELDS', 'RIVALS', 'SELECTOR_BASELINES', 'CE_ARMS', 'CE_PARENT', 'Param', 'BuildContext', 'ModelSpec', 'random_partition', 'alternating_space', 'suggest_alternating', 'build_alternating', 'boosted_space', 'suggest_boosted', 'boosted_param_count', 'build_boosted', 'adastt_space', 'suggest_adastt', 'build_adastt', 'probe_space', 'suggest_probe', 'build_probe', 'soft_probe_space', 'suggest_soft_probe', 'build_soft_probe', 'pca_space', 'suggest_pca', 'build_pca', 'val_best_single_space', 'suggest_val_best_single', 'build_val_best_single', 'win_rate_space', 'suggest_win_rate', 'build_win_rate', 'suggest_kwargs', 'study_signs', 'check_objectives', 'completed', 'best_trial', 'trial_frame', 'pareto_front', 'rescore', 'top_params', 'enqueue_params', 'compare_metrics', 'select_trial', 'StudyStore', 'trial_metrics_line', 'Tuner', 'diagnose', 'refine', 'describe', 'edit_space', 'TuningResult', 'tune', 'tune_all', 'tuning_table', 'best_estimators', 'save_tuning', 'load_tuning', 'partition_identity', 'arm_identity', 'arm_study_name', 'cv_arm_key', 'HPTRun', 'load_hpt_json', 'hpt_records', 'adopt_study', 'hpt', 'hpt_identity', 'hpt_status', 'raw_sdt_space', 'kbest_xgboost_space', 'kbest_sdt_space', 'suggest_kbest_xgboost', 'suggest_kbest_sdt', 'build_kbest_xgboost', 'build_kbest_sdt', 'kbest_sdt_param_count', 'space_source', 'check_spaces', 'space_from_trials', 'argmin_clf_space', 'qe_router_space', 'suggest_argmin_clf', 'suggest_qe_router', 'build_argmin_clf', 'build_qe_router', 'build_adastt_ce', 'build_fastt_boosted_ce', 'build_fastt_alternating_ce']

import hashlib
import inspect
import json
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fastt.crossval import (CONTEXT_METRICS, METRICS, TABLE_LABELS,
                            TABLE_METRICS, FiveByTwoSplit, build_design,
                            fit_arm, format_shares, run_cv, score_arm,
                            selection_shares, wants_rows)
from fastt.estimators import (AdaSTTXGBoost, FASTTAlternating, FASTTBoosted,
                              PCAPipeline, SoftmaxProbeRouter, SoftProbeRouter,
                              ValBestSingleRouter, WinRateRouter,
                              set_determinism, wer_matrix)
from fastt.hub import read_only
from fastt.pretty import (HEX, STATES, in_notebook, metric_token, paint, show_table,
                          signed, state)

try:
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:                                    # pragma: no cover
    optuna = TrialState = None

# XGBoost's gradient-shape notice fires once per trial and drowns the log, and
# TPE's `multivariate`/`group` arguments raise an ExperimentalWarning per study.
# Both are known and neither is actionable.
warnings.filterwarnings("ignore", category=FutureWarning)
if optuna is not None:
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def _require_optuna():
    if optuna is None:
        raise ImportError("§7.5 needs Optuna — `pip install optuna`.")


@dataclass
class Param:
    """One searchable dimension, with the physical limits it may never leave.

    `hard_low` / `hard_high` bound **expansion** only; `low` / `high` are the
    current range and move from round to round. `frozen` is set by `refine` and
    takes a parameter out of the search without removing it from the record.
    """

    name: str
    kind: str                      # "float" | "int" | "cat"
    low: float = None
    high: float = None
    log: bool = False
    choices: tuple = ()
    hard_low: float = None
    hard_high: float = None
    frozen: object = None

    def suggest(self, trial):
        if self.frozen is not None:
            return self.frozen
        if self.kind == "cat":
            return trial.suggest_categorical(self.name, list(self.choices))
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)

    def position(self, v):
        """Where `v` sits in the range, as a 0..1 coordinate. Log-aware."""
        lo, hi = (np.log(self.low), np.log(self.high)) if self.log else (self.low, self.high)
        v = np.log(v) if self.log else v
        return 0.5 if hi <= lo else float((v - lo) / (hi - lo))

    def value_at(self, pos):
        """The inverse of `position`."""
        lo, hi = (np.log(self.low), np.log(self.high)) if self.log else (self.low, self.high)
        v = lo + pos * (hi - lo)
        v = np.exp(v) if self.log else v
        return int(round(v)) if self.kind == "int" else float(v)

    def clamped(self, low, high):
        """A copy with the range widened or narrowed, respecting the hard limits.

        **Both** endpoints are pushed inside `[hard_low, hard_high]`, not just
        the one being moved. `refine` only ever nudges a range that already sits
        inside the limits, so clamping one end was enough for it; `edit_space`
        lets a person type a range that is entirely outside them, and clamping
        one end there produced `[20, 21]` for a parameter whose `hard_high` is
        16 — a range above its own ceiling, which the search then sampled.
        """
        lo_lim, hi_lim = self.hard_low, self.hard_high
        if lo_lim is not None:
            low, high = max(low, lo_lim), max(high, lo_lim)
        if hi_lim is not None:
            low, high = min(low, hi_lim), min(high, hi_lim)
        if low > high:
            low, high = high, low
        if self.kind == "int":
            low, high = int(np.floor(low)), int(np.ceil(high))
            if high <= low:
                # Keep an int range non-degenerate where there is room to, and
                # take that room from whichever side is not against a limit.
                # When the limits pin it to one value, a one-value range is the
                # honest answer — Optuna accepts `suggest_int(x, n, n)`.
                if hi_lim is None or low + 1 <= hi_lim:
                    high = low + 1
                elif lo_lim is None or high - 1 >= lo_lim:
                    low = high - 1
        return replace(self, low=low, high=high)


@dataclass
class BuildContext:
    """What a builder needs that is not a hyperparameter.

    `errors` is the store's `(N, k)` error matrix, needed only by
    `SoftProbeRouter`, which scores a mixture rather than selecting a member.
    Passing it here rather than closing over a global is what lets the same
    registry be used from a module, a notebook cell and a test.

    `texts` and `members` are the fusion door (§9). A combination arm works on
    transcripts, which are addressed by store row and cannot be recovered from
    the standardised design matrix — so the frame `store.load_texts()` returns
    travels here, alongside the member order its columns are read in. Both stay
    `None`/`()` for every selection arm, which is why they are defaulted rather
    than required.

    `cache` is one dict shared by every arm, every trial and every fold of a
    run. The expensive parts of fusion — the confusion-network alignment, the
    scoring of a fused transcript, the fold's confidence model — depend on the
    row or on the fold, never on the trial, so computing them once and reading
    them back is the difference between a 40-fold pool that finishes and one
    that does not. It is deliberately a plain dict: the arms namespace their own
    keys, and a caller who wants a cold run passes a fresh one.
    """

    nthread: int = 4
    epochs_full: int = 200
    errors: object = None
    texts: object = None
    members: tuple = ()
    cache: dict = field(default_factory=dict)


@dataclass
class ModelSpec:
    """One tunable algorithm: its space, its sampler, its builder, its budget.

    Parameters
    ----------
    trials : scales this arm's share of the trial budget. A two-parameter space
        does not need — and TPE cannot use — the 120 trials a fifteen-parameter
        one gets, so the budget is per arm rather than global.
    discovery_cap : the main-loop cap applied during the cheap discovery rounds.
        `None` means the arm has no main loop worth capping.
    prunable : whether `build(...)` accepts a `callback=` that reports an
        intermediate score. Only the two FASTT arms have an iteration structure
        to prune on; pruning the others would mean pruning on nothing.
    param_count : `(params, in_features) -> int`, checked BEFORE anything is
        allocated. A trial that OOMs takes the whole study with it.
    """

    name: str
    space: object
    suggest: object
    build: object
    discovery_cap: int = None
    trials: float = 1.0
    prunable: bool = False
    param_count: object = None
    prune_warmup: int = None

    def warmup(self):
        """Steps a trial gets before the pruner may touch it.

        This has to be per arm, because the arms report on loops of wildly
        different length: `FASTTAlternating` reports once per alternating
        iteration and converges in ~12, while `FASTTBoosted` reports once per
        AdamW **epoch** and is given 200. A single global `n_warmup_steps` that
        is sane for the first is destructive for the second — at epoch 2 of 200
        a boosted trial has not started learning, and pruning on that signal
        throws away the search budget for nothing.

        `discovery_cap` already encodes exactly this judgement ("40 of 200
        epochs is enough to rank a boosted config"), so it is the default rather
        than a second constant to keep in sync.
        """
        if self.prune_warmup is not None:
            return int(self.prune_warmup)
        return int(self.discovery_cap) if self.discovery_cap else 1

    def n_free(self, space=None):
        """How many dimensions are still being searched."""
        return sum(1 for p in (space or self.space()).values() if p.frozen is None)


def random_partition(n, holdout_frac=0.25, inner_val_frac=0.30, seed=20260820):
    """Pool-and-recut: `{tune, eval, fit, val}` row indices over `n` pooled rows.

    A function of `seed` alone, so the same call in another session returns the
    same four arrays and a re-run tunes on exactly the same utterances.

    * `tune` / `eval` — the outer cut. `eval` goes to `run_cv(rows=...)` and is
      never seen by a trial.
    * `fit` / `val` — `tune` cut again, so a trial is scored on rows it did not
      fit on. Both are subsets of `tune`, indexed back into the pooled row
      numbering so they can be used against the store directly.

    Each array is sorted, which keeps `X[rows]` in the store's own row order and
    makes anything derived from it comparable across calls.
    """
    if not 0 < holdout_frac < 1:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")
    if not 0 < inner_val_frac < 1:
        raise ValueError(f"inner_val_frac must be in (0, 1), got {inner_val_frac}")

    perm = np.random.default_rng(seed).permutation(n)
    cut = int(np.clip(round(n * holdout_frac), 1, n - 1))
    tune_rows, eval_rows = np.sort(perm[:cut]), np.sort(perm[cut:])

    # A second generator, keyed off the same seed, so the inner cut moves with
    # the outer one instead of being an independent thing to remember.
    inner = np.random.default_rng(seed + 1).permutation(len(tune_rows))
    icut = int(np.clip(round(len(tune_rows) * (1 - inner_val_frac)), 1, len(tune_rows) - 1))
    return {"tune": tune_rows, "eval": eval_rows,
            "fit": tune_rows[np.sort(inner[:icut])],
            "val": tune_rows[np.sort(inner[icut:])]}

SELECTOR_PARAMS = ("sel_n_estimators", "sel_learning_rate", "sel_max_depth",
                   "sel_subsample", "sel_colsample", "sel_min_child_weight",
                   "sel_reg_lambda")
LOOP_PARAMS = ("num_iterations", "transform_lr", "transform_steps",
               "surrogate_lr", "surrogate_steps")
TREE_PARAMS = ("num_rounds", "num_trees", "depth", "lmbda", "lmbda2",
               "adam_lr", "adam_weight_decay", "batch_size", "patience")


def _selector_params(prefix="sel_"):
    """The AdaSTT/XGBoost dimensions, shared by three of the spaces below.

    `n_estimators` spans two orders of magnitude on purpose: §7.2.2 shows AdaSTT
    collapsing onto the best single model at small budgets, so the range has to
    reach far enough to escape that regime rather than merely bracket the
    shipped value.
    """
    return [
        Param(f"{prefix}n_estimators", "int", low=20, high=1500, log=True,
              hard_low=5, hard_high=8000),
        Param(f"{prefix}learning_rate", "float", low=0.005, high=0.5, log=True,
              hard_low=1e-4, hard_high=1.0),
        Param(f"{prefix}max_depth", "int", low=2, high=14, hard_low=1, hard_high=32),
        Param(f"{prefix}subsample", "float", low=0.4, high=1.0, hard_low=0.1, hard_high=1.0),
        Param(f"{prefix}colsample", "float", low=0.1, high=1.0, hard_low=0.02, hard_high=1.0),
        Param(f"{prefix}min_child_weight", "float", low=0.5, high=32, log=True,
              hard_low=0.01, hard_high=512),
        Param(f"{prefix}reg_lambda", "float", low=0.1, high=50, log=True,
              hard_low=1e-3, hard_high=1000),
    ]


def _build_selector(p, seed, ctx, prefix="sel_", objective="expected_wer"):
    """An `AdaSTTXGBoost` from the prefixed selector parameters."""
    return AdaSTTXGBoost(
        n_estimators=p[f"{prefix}n_estimators"],
        learning_rate=p[f"{prefix}learning_rate"],
        max_depth=p[f"{prefix}max_depth"], subsample=p[f"{prefix}subsample"],
        colsample_bytree=p[f"{prefix}colsample"],
        min_child_weight=p[f"{prefix}min_child_weight"],
        reg_lambda=p[f"{prefix}reg_lambda"], early_stopping_rounds=50,
        nthread=ctx.nthread, random_state=seed, objective=objective)


def alternating_space():
    """FASTT-Alternating. Deliberately too wide; §7.5.2 makes it the right size."""
    return {p.name: p for p in [
        Param("transform_type", "cat",
              choices=("identity", "diagonal", "linear", "low_rank", "nonlinear")),

        Param("num_iterations", "int", low=1, high=12, hard_low=1, hard_high=40),
        Param("transform_lr", "float", low=1e-5, high=1e-1, log=True,
              hard_low=1e-8, hard_high=1.0),
        Param("transform_steps", "int", low=10, high=400, log=True,
              hard_low=1, hard_high=4000),
        Param("surrogate_lr", "float", low=1e-4, high=1e-1, log=True,
              hard_low=1e-7, hard_high=1.0),
        Param("surrogate_steps", "int", low=20, high=400, log=True,
              hard_low=1, hard_high=4000),

        Param("weight_decay", "float", low=1e-6, high=1e-1, log=True,
              hard_low=1e-10, hard_high=10.0),
        Param("lambda1", "float", low=1e-5, high=1e-1, log=True,
              hard_low=1e-10, hard_high=10.0),
        Param("lambda2", "float", low=1e-5, high=1e-1, log=True,
              hard_low=1e-10, hard_high=10.0),
        Param("out_ratio", "float", low=0.05, high=1.0, log=True,
              hard_low=0.005, hard_high=2.0),
        Param("bottleneck_ratio", "float", low=0.02, high=1.0, log=True,
              hard_low=0.002, hard_high=2.0),
    ] + _selector_params()}


def suggest_alternating(trial, space, max_iter=None):
    """Define-by-run sampling. Only the parameters that apply are ever suggested."""
    p = {"transform_type": space["transform_type"].suggest(trial)}
    tt = p["transform_type"]

    if tt == "identity":
        # No transform parameters means the alternating loop refits the selector
        # on identical features every iteration. One pass is the whole algorithm.
        p["num_iterations"] = 1
    else:
        for k in LOOP_PARAMS:
            p[k] = space[k].suggest(trial)
        if max_iter is not None:
            p["num_iterations"] = min(p["num_iterations"], max_iter)
        if tt == "diagonal":
            p["lambda1"] = space["lambda1"].suggest(trial)
            p["lambda2"] = space["lambda2"].suggest(trial)
        else:
            p["weight_decay"] = space["weight_decay"].suggest(trial)
            p["out_ratio"] = space["out_ratio"].suggest(trial)
            if tt in ("low_rank", "nonlinear"):
                p["bottleneck_ratio"] = space["bottleneck_ratio"].suggest(trial)

    for k in SELECTOR_PARAMS:
        p[k] = space[k].suggest(trial)
    return p


def _transform_kwargs(p, in_features):
    """Absolute widths from the ratio parameters, floored so nothing degenerates."""
    tt, tkw = p["transform_type"], {}
    if tt in ("linear", "low_rank", "nonlinear"):
        tkw["out_features"] = max(8, int(round(in_features * p["out_ratio"])))
    if tt in ("low_rank", "nonlinear"):
        tkw["bottleneck_dim"] = max(4, int(round(in_features * p["bottleneck_ratio"])))
    return tkw

def build_alternating(p, in_features, seed=42, ctx=None, objective="expected_wer"):
    """A configured `FASTTAlternating` from one sampled parameter dict.

    `objective` reaches BOTH the selector and the alternation, which is what
    keeps the `fastt_alternating_ce` control from being half-swapped.
    """
    ctx = ctx or BuildContext()
    return FASTTAlternating(
        base_selector=_build_selector(p, seed, ctx, objective=objective),
        objective=objective,
        transform_type=p["transform_type"],
        transform_kwargs=_transform_kwargs(p, in_features),
        num_iterations=p["num_iterations"],
        transform_lr=p.get("transform_lr", 1e-3),
        transform_steps=p.get("transform_steps", 100),
        surrogate_lr=p.get("surrogate_lr", 1e-2),
        surrogate_steps=p.get("surrogate_steps", 200),
        transform_weight_decay=p.get("weight_decay", 1e-4),
        transform_lambda1=p.get("lambda1", 0.01),
        transform_lambda2=p.get("lambda2", 0.01),
        random_state=seed)


def boosted_space():
    """FASTT-Boosted. Same conditional structure, a different second half.

    Two ranges come from measurement rather than from the shipped config:
    `lmbda` / `lmbda2` reach down to 1e-6 because §7.2.4 shows the shipped
    0.1 / 0.01 putting the regularisers at ninety per cent of the loss, where the
    model stops routing entirely — bracketing those values would search only the
    degenerate region. And `depth` stops at 7 because every level doubles both
    the leaf count and the width of that level's split layer, so depth is the one
    parameter that turns a plausible trial into an out-of-memory crash.

    `epochs` is deliberately absent. It is fixed high and `patience` does the
    stopping; searching both is searching the same thing twice.
    """
    return {p.name: p for p in [
        Param("transform_type", "cat",
              choices=("identity", "diagonal", "linear", "low_rank", "nonlinear")),

        Param("num_rounds", "int", low=1, high=6, hard_low=1, hard_high=16),
        Param("num_trees", "int", low=2, high=24, log=True, hard_low=1, hard_high=128),
        Param("depth", "int", low=2, high=7, hard_low=1, hard_high=10),

        Param("lmbda", "float", low=1e-6, high=1.0, log=True,
              hard_low=1e-10, hard_high=10.0),
        Param("lmbda2", "float", low=1e-7, high=1e-1, log=True,
              hard_low=1e-12, hard_high=10.0),

        Param("adam_lr", "float", low=1e-4, high=3e-2, log=True,
              hard_low=1e-6, hard_high=1.0),
        Param("adam_weight_decay", "float", low=1e-7, high=1e-1, log=True,
              hard_low=1e-12, hard_high=10.0),
        Param("batch_size", "int", low=64, high=2048, log=True,
              hard_low=8, hard_high=16384),
        Param("patience", "int", low=5, high=40, hard_low=2, hard_high=200),

        Param("t_weight_decay", "float", low=1e-6, high=1e-1, log=True,
              hard_low=1e-10, hard_high=10.0),
        Param("lambda1", "float", low=1e-5, high=1e-1, log=True,
              hard_low=1e-10, hard_high=10.0),
        Param("lambda2", "float", low=1e-5, high=1e-1, log=True,
              hard_low=1e-10, hard_high=10.0),
        Param("out_ratio", "float", low=0.02, high=1.0, log=True,
              hard_low=0.002, hard_high=2.0),
        Param("bottleneck_ratio", "float", low=0.02, high=1.0, log=True,
              hard_low=0.002, hard_high=2.0),
    ]}


def suggest_boosted(trial, space, max_iter=None, epochs_full=200):
    """Define-by-run over `boosted_space`. `max_iter` caps epochs during discovery."""
    p = {"transform_type": space["transform_type"].suggest(trial)}
    tt = p["transform_type"]

    for k in TREE_PARAMS:
        p[k] = space[k].suggest(trial)

    if tt == "diagonal":
        p["lambda1"] = space["lambda1"].suggest(trial)
        p["lambda2"] = space["lambda2"].suggest(trial)
    elif tt != "identity":
        p["t_weight_decay"] = space["t_weight_decay"].suggest(trial)
        p["out_ratio"] = space["out_ratio"].suggest(trial)
        if tt in ("low_rank", "nonlinear"):
            p["bottleneck_ratio"] = space["bottleneck_ratio"].suggest(trial)

    p["epochs"] = epochs_full if max_iter is None else min(epochs_full, max_iter)
    return p


def boosted_param_count(p, in_features):
    """Projected parameter count, WITHOUT building the model.

    The split layer at level `l` is `Linear(width, num_trees * 2**l)`, so a round
    costs `width * num_trees * (2**depth - 1)` in splits alone. Checking this up
    front is the difference between skipping a trial and losing the study to an
    out-of-memory kill.
    """
    tt = p["transform_type"]
    width = in_features
    if tt in ("linear", "low_rank", "nonlinear"):
        width = max(8, int(round(in_features * p["out_ratio"])))
    splits = width * p["num_trees"] * (2 ** p["depth"] - 1)
    leaves = p["num_trees"] * 3 * (2 ** p["depth"])
    return int(p["num_rounds"] * (splits + leaves))


def build_boosted(p, in_features, seed=42, ctx=None, objective="expected_wer"):
    """A configured `FASTTBoosted` from one sampled parameter dict."""
    ctx = ctx or BuildContext()
    return FASTTBoosted(
        objective=objective,
        num_rounds=p["num_rounds"], transform_type=p["transform_type"],
        transform_kwargs=_transform_kwargs(p, in_features),
        num_trees=p["num_trees"], depth=p["depth"],
        lmbda=p["lmbda"], lmbda2=p["lmbda2"],
        transform_weight_decay=p.get("t_weight_decay", 1e-4),
        transform_lambda1=p.get("lambda1", 0.01),
        transform_lambda2=p.get("lambda2", 0.01),
        lr=p["adam_lr"], weight_decay=p["adam_weight_decay"],
        batch_size=p["batch_size"], epochs=p.get("epochs", ctx.epochs_full),
        patience=p["patience"], random_state=seed)

def adastt_space():
    """The XGBoost selector alone — FASTT-Alternating with the transform removed."""
    return {p.name: p for p in _selector_params()}


def suggest_adastt(trial, space, max_iter=None):
    return {k: space[k].suggest(trial) for k in SELECTOR_PARAMS}


def build_adastt(p, in_features, seed=42, ctx=None, objective="expected_wer"):
    return _build_selector(p, seed, ctx or BuildContext(), objective=objective)


def probe_space():
    """`SoftmaxProbeRouter`: the ridge penalty and what the objective weights by."""
    return {p.name: p for p in [
        Param("alpha", "float", low=1e-4, high=1e3, log=True,
              hard_low=1e-8, hard_high=1e6),
        Param("objective_weight", "cat", choices=("uniform", "length")),
    ]}


def suggest_probe(trial, space, max_iter=None):
    return {k: space[k].suggest(trial) for k in ("alpha", "objective_weight")}


def build_probe(p, in_features, seed=42, ctx=None):
    return SoftmaxProbeRouter(alpha=p["alpha"],
                              objective_weight=p["objective_weight"])


def soft_probe_space():
    """The same, plus the inverse temperature that decides how hard it commits."""
    space = probe_space()
    space["beta"] = Param("beta", "float", low=1e-2, high=1e2, log=True,
                          hard_low=1e-4, hard_high=1e4)
    return space


def suggest_soft_probe(trial, space, max_iter=None):
    return {k: space[k].suggest(trial) for k in ("alpha", "objective_weight", "beta")}


def build_soft_probe(p, in_features, seed=42, ctx=None):
    ctx = ctx or BuildContext()
    if ctx.errors is None:
        raise ValueError("soft_probe needs BuildContext(errors=store.err(members))")
    return SoftProbeRouter(errors=ctx.errors, beta=p["beta"], alpha=p["alpha"],
                           objective_weight=p["objective_weight"])


def pca_space():
    """The unsupervised control: projection width and whitening, plus its selector."""
    return {p.name: p for p in [
        Param("n_components", "int", low=8, high=1024, log=True,
              hard_low=2, hard_high=4096),
        Param("whiten", "cat", choices=(False, True)),
    ] + _selector_params()}


def suggest_pca(trial, space, max_iter=None):
    p = {k: space[k].suggest(trial) for k in ("n_components", "whiten")}
    p.update({k: space[k].suggest(trial) for k in SELECTOR_PARAMS})
    return p


def build_pca(p, in_features, seed=42, ctx=None):
    ctx = ctx or BuildContext()
    # `n_components` cannot exceed the width it is projecting, and the space is
    # shared across ensembles of different widths, so it is clipped rather than
    # bounded — a clipped trial is a valid trial, a bounded space is a different
    # search per member set.
    k = int(min(p["n_components"], in_features))
    return PCAPipeline(base=lambda: _build_selector(p, seed, ctx),
                       n_components=k, whiten=bool(p["whiten"]))


def val_best_single_space():
    """One dimension: how much of the fit data is spent choosing the member."""
    return {p.name: p for p in [
        Param("val_frac", "float", low=0.05, high=0.5, hard_low=0.01, hard_high=0.9),
    ]}


def suggest_val_best_single(trial, space, max_iter=None):
    return {"val_frac": space["val_frac"].suggest(trial)}


def build_val_best_single(p, in_features, seed=42, ctx=None):
    return ValBestSingleRouter(val_frac=p["val_frac"], random_state=seed)


def win_rate_space():
    """One dimension: whether a tied row counts towards a member's win rate."""
    return {p.name: p for p in [Param("sole", "cat", choices=(False, True))]}


def suggest_win_rate(trial, space, max_iter=None):
    return {"sole": space["sole"].suggest(trial)}


def build_win_rate(p, in_features, seed=42, ctx=None):
    return WinRateRouter(sole=bool(p["sole"]))


MODELS = {
    # --- ours -----------------------------------------------------------------
    # 6 of ~12 alternating iterations, and 40 of 200 AdamW epochs: enough to rank
    # configurations. The caps differ because the loops are not comparable — six
    # alternating iterations is a nearly-converged model, six epochs is one that
    # has not started learning.
    "fastt_alternating": ModelSpec("fastt_alternating", alternating_space,
                                   suggest_alternating, build_alternating,
                                   discovery_cap=6, trials=1.0, prunable=True),
    "fastt_boosted": ModelSpec("fastt_boosted", boosted_space, suggest_boosted,
                               build_boosted, discovery_cap=40, trials=1.0,
                               prunable=True, param_count=boosted_param_count),
    # --- competitors ----------------------------------------------------------
    "adastt": ModelSpec("adastt", adastt_space, suggest_adastt, build_adastt,
                        trials=0.6),
    "probe": ModelSpec("probe", probe_space, suggest_probe, build_probe, trials=0.2),
    "soft_probe": ModelSpec("soft_probe", soft_probe_space, suggest_soft_probe,
                            build_soft_probe, trials=0.3),
    "pca_adastt": ModelSpec("pca_adastt", pca_space, suggest_pca, build_pca,
                            trials=0.6),
    "val_best_single": ModelSpec("val_best_single", val_best_single_space,
                                 suggest_val_best_single, build_val_best_single,
                                 trials=0.1),
    "win_rate": ModelSpec("win_rate", win_rate_space, suggest_win_rate,
                          build_win_rate, trials=0.05),
}

# The order results are reported in: ours first, then the competitors, so a
# printed table reads as "the claim, then the field".
OURS = ("fastt_alternating", "fastt_boosted")


def suggest_kwargs(spec, ctx):
    """What the tuner passes to `spec.suggest` beyond the trial and the space.

    `epochs_full` goes to every suggester whose signature takes it — decided by
    the signature, not by the arm's name. The previous `spec.name !=
    "fastt_boosted"` test silently starved `raw_sdt`, `kbest_sdt` and any arm
    registered later (the `_ce` controls) of the context's epoch budget, so a
    control could be trained for a different number of epochs than the arm it
    is compared against: a confound that looks like a result.
    """
    accepts = inspect.signature(spec.suggest).parameters
    return {"epochs_full": ctx.epochs_full} if "epochs_full" in accepts else {}
COMPETITORS = ("adastt", "probe", "soft_probe", "pca_adastt",
               "val_best_single", "win_rate")

# The measured costs. Every one is minimised, and every one is a property of the
# run rather than of the routing, which is why they are listed here by hand
# instead of coming from `METRICS`.
COSTS = {
    "trial_seconds": ("minimize",
                      "wall-clock for the whole trial: every inner fold, fit and score"),
    "fit_seconds": ("minimize",
                    "wall-clock inside `fit`, summed over the inner folds"),
    "predict_seconds": ("minimize",
                        "wall-clock inside `predict`, summed over the inner folds"),
    "n_params": ("minimize",
                 "parameter count of the built model, counted before it is fitted"),
}

# What a trial may be scored on: every routing metric §7.3 defines, plus the
# costs. `(direction, what it measures)`.
#
# The metrics are taken from `crossval.METRICS` rather than restated, which is
# the whole point — the tuner and the evaluation then cannot drift apart, and a
# metric added there is immediately available as an objective here. `"lower"` /
# `"higher"` is translated to Optuna's vocabulary; **both directions are
# supported**, so `gap_closed_mean_utt` (higher is better) is as valid an
# objective as `corpus_wer` (lower is better).
#
# `CONTEXT_METRICS` are excluded: `tie_rate` and `decisive_rate` describe the
# scoring rows, not the arm, so every trial would score identically on them.
OBJECTIVES = {
    **{name: ("minimize" if direction == "lower" else "maximize",
              f"crossval.METRICS[{name!r}], on the tuning-score rows")
       for name, (_fn, direction) in METRICS.items()
       if name not in CONTEXT_METRICS},
    **COSTS,
}

# Compact labels for the per-trial progress line, which has to fit two of these
# plus the running best on one terminal row.
_OBJ_LABEL = {"corpus_wer": "wer", "mean_utt_wer": "mwer",
              "gap_closed": "gap", "gap_closed_mean_utt": "mgap",
              "gap_closed_abs": "gapabs", "routing_error": "rerr",
              "regret_per_word": "regret", "acc_lenient": "acc",
              "acc_strict": "sacc",
              "trial_seconds": "secs", "fit_seconds": "fit",
              "predict_seconds": "pred", "n_params": "params"}


def _sign(direction):
    """`+1` for a minimised objective, `-1` for a maximised one.

    Multiplying a value by this turns it into a **loss** — always lower-is-better
    — which is what lets every `min` / `argmin` below stay a plain `min` whatever
    direction the objective actually has.
    """
    return 1.0 if str(direction).upper().startswith("MIN") else -1.0


def study_signs(study):
    """One `_sign` per objective, read off the study's own directions."""
    return [_sign(d.name) for d in study.directions]


def check_objectives(objectives):
    """Validate an objective tuple and return it. Raises rather than warns."""
    objectives = tuple(objectives)
    if not objectives:
        raise ValueError("at least one objective is required")
    unknown = [o for o in objectives if o not in OBJECTIVES]
    if unknown:
        raise KeyError(f"unknown objective(s) {unknown}; have {sorted(OBJECTIVES)}")
    if len(set(objectives)) != len(objectives):
        raise ValueError(f"duplicate objective in {objectives}")
    return objectives


def _fmt_objective(name, v):
    """One objective value, in the units it is actually read in."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if name.endswith("seconds"):
        return f"{v:.1f}s"
    if name == "n_params":
        return f"{v:,.0f}"
    if name.startswith("gap_closed") or name.startswith("acc_"):
        return f"{v:+.1%}" if name.startswith("gap") else f"{v:.1%}"
    return f"{v:.4f}"


def completed(study):
    """Every COMPLETE trial that carries values.

    Multi-objective safe, which `t.value` is not: on a study with several
    directions Optuna raises rather than pick one for you. Everything in this
    section reads `t.values[i]` through this.
    """
    return [t for t in study.trials
            if t.state == TrialState.COMPLETE and t.values is not None]


def best_trial(study, index=0):
    """The completed trial that is BEST on objective `index`, in its direction.

    `study.best_trial` raises on a multi-objective study — there is no single
    best trial, that is the whole point of a front — so `refine`, `_progress`
    and `tune` all go through this instead. The direction is read off the study
    rather than assumed, so this is correct for `gap_closed_mean_utt` (higher is
    better) as well as `corpus_wer` (lower is better).
    """
    done = completed(study)
    if not done:
        raise ValueError("no completed trials")
    sign = study_signs(study)[index]
    return min(done, key=lambda t: sign * t.values[index])


def trial_frame(study, objectives=("corpus_wer",), extra_metrics=True):
    """Every completed trial as a row, **best first** on the primary objective.

    `extra_metrics` adds every metric the trial recorded but is not being
    optimised on — which is what makes it possible to see, after the fact,
    how a search that minimised `corpus_wer` would have ranked under
    `gap_closed_mean_utt` without re-fitting anything.
    """
    objectives = tuple(objectives)
    sign = _sign(OBJECTIVES[objectives[0]][0])
    rows = []
    for t in completed(study):
        row = {"trial": t.number}
        row.update({o: t.values[i] for i, o in enumerate(objectives)
                    if i < len(t.values)})
        if extra_metrics:
            row.update({k: v for k, v in t.user_attrs.items()
                        if k in OBJECTIVES and k not in row})
        row.update(t.params)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # A stable sort: tied trials stay in trial order, so the configuration a search selects among equal scores is
    # the same on every machine (the default quicksort breaks ties differently on different CPUs).
    return (df.assign(_k=sign * df[objectives[0]])
              .sort_values("_k", ignore_index=True, kind="stable").drop(columns="_k"))


def pareto_front(study, objectives=("corpus_wer",)):
    """The non-dominated trials, as a frame sorted by the primary objective.

    `study.best_trials` is the front on a multi-objective study and the (tied)
    best trials on a single-objective one, so this reads the same either way and
    a single-objective front is simply one row long.
    """
    objectives = tuple(objectives)
    front = {t.number for t in study.best_trials}
    df = trial_frame(study, objectives)
    if df.empty:
        return df
    # `trial_frame` already sorted best-first in the primary objective's own
    # direction, so this only has to filter.
    return df[df["trial"].isin(front)].reset_index(drop=True)


def rescore(study, tuner, spec, space, trials=None, top=None, key=None,
            verbose=True):
    """Re-fit stored trial parameters and score them under EVERY metric.

    This is the answer to "I tuned for a long time and now I want a different
    metric". A finished study stores each trial's **parameters**, so the search
    — the expensive, sequential, TPE-driven part — never has to be repeated. It
    stores only the metrics that were recorded at the time, though, so a study
    run before `Tuner.objective` recorded all of them has to have its candidates
    *re-fitted* once each to learn what they score under the new metric. That is
    embarrassingly parallel work with no sampler in the loop, and `top` cuts it
    to the only trials that could plausibly win.

    Parameters
    ----------
    study, spec, space : the finished study, its `ModelSpec`, and the space it
        actually ran on — `TuningResult.space`, **not** `spec.space()`, which is
        the pre-refinement box. Re-running `tune()` against the same store
        replays the discovery rounds from the database in seconds and hands it
        back.
    tuner : must be built with the SAME `store`, `members`, `partition_seed`,
        `holdout_frac` and `inner_folds` as the original run, or the scoring
        rows differ and the numbers are not comparable with the stored ones.
        `corpus_wer` is re-derived here and printed beside the stored value
        precisely so that mistake is visible rather than silent.
    trials : which trials to re-fit. Defaults to every completed one.
    top, key : keep only the best `top` trials by `key` (a metric name,
        defaulting to the study's primary objective) before re-fitting. The
        cheap, usual mode: the winner under a *related* metric is very unlikely
        to sit in the bottom half under the old one, and re-fitting 25 of 276
        costs a tenth of what the search did.

    Returns a frame with one row per re-fitted trial: `trial`, every metric,
    every cost, and the sampled parameters.
    """
    _require_optuna()
    done = list(trials) if trials is not None else completed(study)
    if not done:
        raise ValueError("no completed trials to re-score")
    if top is not None:
        key = key or _study_objective_names(study)[0]
        sign = _sign(OBJECTIVES[key][0]) if key in OBJECTIVES else 1.0

        def rank(t):
            v = (t.values[0] if key == _study_objective_names(study)[0]
                 else t.user_attrs.get(key))
            return sign * (v if v is not None else np.inf)

        done = sorted(done, key=rank)[:int(top)]

    # Full fidelity and no pruning: these are final numbers, not a search.
    objective = tuner.objective(spec, space, fidelity=1.0, max_iter=None,
                                pruner="none")
    rows, t0 = [], time.perf_counter()
    for i, t in enumerate(done, 1):
        ft = optuna.trial.FixedTrial(t.params)
        try:
            with warnings.catch_warnings():
                # FixedTrial warns for any stored value outside the space it is
                # handed. That is expected when `refine` narrowed a range after
                # the trial ran, and the stored value is still the right one.
                warnings.simplefilter("ignore")
                objective(ft)
        except optuna.TrialPruned as e:
            if verbose:
                print(f"  [{i}/{len(done)}] trial {t.number}: pruned ({e})")
            continue
        except Exception as e:                                 # noqa: BLE001
            if verbose:
                print(f"  [{i}/{len(done)}] trial {t.number}: FAILED "
                      f"{type(e).__name__}: {e}")
            continue
        row = {"trial": t.number, **{k: v for k, v in ft.user_attrs.items()
                                     if k in OBJECTIVES}}
        row["stored_value"] = t.values[0]
        rows.append({**row, **t.params})
        if verbose:
            print(f"  [{i}/{len(done)}] trial {t.number}: "
                  + "  ".join(f"{k} {_fmt_objective(k, row[k])}"
                              for k in list(OBJECTIVES) if k in row and
                              not k.endswith("seconds") and k != "n_params"),
                  flush=True)

    out = pd.DataFrame(rows)
    if verbose:
        print(f"\nre-fitted {len(out)} of {len(done)} trial(s) in "
              f"{(time.perf_counter() - t0) / 60:.1f} min")
        if len(out) and "corpus_wer" in out and "stored_value" in out:
            # The tuner-matches-the-original check. A non-trivial drift here
            # means this `tuner` is not the one that produced the study.
            d = (out["corpus_wer"] - out["stored_value"]).abs().max()
            print(f"  max |re-fitted corpus_wer - stored value| = {d:.2e}"
                  + ("   <- the tuner reproduces the original run" if d < 1e-9
                     else "   !! NOT the same scoring rows — check the Tuner's "
                          "store/members/seeds against the original"))
    return out


def _study_objective_names(study):
    """Best-effort objective names for a study: from its own user attrs, else
    positional placeholders. `tune` records them so `rescore` can be called with
    nothing but the study."""
    names = study.user_attrs.get("objectives")
    return tuple(names) if names else tuple(
        f"objective_{i}" for i in range(len(study.directions)))


def top_params(study, top=None, key=None, states=(TrialState.COMPLETE,)):
    """The best `top` parameter dicts from a finished study, best first.

    `key` is a metric name; it defaults to the study's own primary objective and
    may name any metric the trials recorded, so a study that *minimised*
    `corpus_wer` can be mined for the trials that scored best on
    `gap_closed_mean_utt` — provided the trials recorded it.
    """
    done = [t for t in study.trials if t.state in states and t.values]
    if not done:
        return []
    key = key or _study_objective_names(study)[0]
    primary = _study_objective_names(study)[0]
    sign = _sign(OBJECTIVES[key][0]) if key in OBJECTIVES else 1.0

    def rank(t):
        v = t.values[0] if key == primary else t.user_attrs.get(key)
        return sign * (v if v is not None else np.inf)

    done = sorted(done, key=rank)
    return [dict(t.params) for t in (done[:int(top)] if top else done)]


def enqueue_params(study, params, skip_if_exists=True, verbose=True):
    """Queue parameter sets to be evaluated BEFORE any fresh sampling. Returns n.

    This is how a previous run's work carries into a new one whose **objective
    has changed**. The old trial *values* are worthless under a new metric — they
    measure something else — but the old *parameters* are not: they are a map of
    where a competent search already looked, and re-measuring them under the new
    metric turns them into real observations that TPE can model. A warm-started
    search spends its first trials on configurations known to be interesting
    instead of on a quasi-random initial design.

    Nothing is assumed about the values being comparable, because nothing is
    carried over except the coordinates. Each queued trial is **run**, at full
    fidelity, and scored by the current objective like any other trial — so the
    resulting study is exactly as valid as one searched from scratch, and merely
    better-aimed.

    A queued value outside the current (possibly refined) range is used anyway,
    with a warning from Optuna. That is the wanted behaviour: it is real evidence
    about a real configuration, and `refine` narrowed the box on the strength of
    a *different* metric.
    """
    n = 0
    for p in params:
        if p:
            study.enqueue_trial(p, skip_if_exists=skip_if_exists)
            n += 1
    if verbose and n:
        print(f"  warm start: queued {n} parameter set(s); they run before any "
              f"fresh TPE draw")
    return n


def compare_metrics(frame, a, b, top=10):
    """How differently two metrics rank the same trials. Returns a frame.

    The question `rescore` exists to answer is not "what is the best trial under
    the new metric" but "does the new metric even disagree". Spearman near 1 and
    a shared winner means the objective switch was cosmetic; a low rank
    correlation means the old search was optimising the wrong thing and the
    winner really does move.
    """
    sa, sb = _sign(OBJECTIVES[a][0]), _sign(OBJECTIVES[b][0])
    d = frame[["trial", a, b]].dropna().copy()
    d["rank_" + a] = (sa * d[a]).rank()
    d["rank_" + b] = (sb * d[b]).rank()
    rho = d["rank_" + a].corr(d["rank_" + b], method="spearman")
    best_a = d.loc[(sa * d[a]).idxmin(), "trial"]
    best_b = d.loc[(sb * d[b]).idxmin(), "trial"]
    print(f"Spearman rank correlation {a} vs {b}: {rho:+.3f}")
    print(f"  best on {a}: trial {best_a}    best on {b}: trial {best_b}"
          + ("   <- the same trial wins both" if best_a == best_b
             else "   <- the winner MOVES"))
    return d.sort_values("rank_" + b).head(top).reset_index(drop=True)


SELECT_POLICIES = ("best_value", "best_within_cost", "fastest_within_value",
                   "knee", "cheapest")


def select_trial(study, objectives=("corpus_wer",), policy="best_within_cost",
                 tol=2.0, max_cost=None, anchor="median", verbose=True):
    """Pick ONE trial off the Pareto front. Returns `(trial, front_frame, reason)`.

    A two-objective search does not have a winner, it has a frontier, and
    something has to turn the frontier back into the single configuration
    `run_cv` will fit. That rule is a *decision*, not a result, so it is named
    and printed rather than buried.

    Policies, all evaluated over the front only — a dominated trial can never
    win any of them, since some front trial is at least as good on both axes:

    * `"best_within_cost"` (default) — **the lowest primary objective among the
      trials that cost no more than the budget.** This is the policy for "WER
      matters most, but I will not deploy something slow": it never trades away
      accuracy it did not have to, and it refuses the corner of the front where
      the last 0.1% of WER costs ten times the compute.

      The budget is `max_cost` when given. Otherwise it is `tol` times an
      **anchor** taken over every completed trial, and the anchor is
      `"median"` by default rather than `"min"` for a reason worth stating: the
      cheapest trial in a search is usually a degenerate one — two boosting
      rounds, a frozen transform — that finishes in a tenth of a second, and
      anchoring on it makes `tol=2` mean "twice as fast as anything useful"
      rather than "not slow". Against the median, `tol=2.0` reads as *at most
      twice what a typical trial cost*, which is both stable across corpora and
      the sentence you would actually defend. Use `anchor="min"` when you really
      do mean "within a small multiple of the fastest thing that works".
    * `"fastest_within_value"` — the mirror image: the cheapest trial whose
      primary objective is within `tol` **relative** of the best. Use it when
      the deadline is the binding constraint and WER has slack.
    * `"best_value"` — ignore cost entirely. What a single-objective study does,
      and what makes the front a diagnostic rather than a decision.
    * `"cheapest"` — the fast end of the front, whatever it scores.
    * `"knee"` — the elbow, by shortest normalised distance to the utopia point.
      No constant to choose, but the answer moves when the shape of the front
      does, which makes it hard to defend across corpora.

    `tol` means different things per policy on purpose: a **multiple** of the
    cheapest cost for `best_within_cost` (2.0 = "at most twice the cheapest"), a
    **relative slack** on the objective for `fastest_within_value` (0.01 = "1%
    worse WER is acceptable"). Both are printed with the choice.
    """
    objectives = check_objectives(objectives)
    front = pareto_front(study, objectives)
    if front.empty:
        raise ValueError("no completed trials to select from")
    done = completed(study)
    by_number = {t.number: t for t in done}
    primary = objectives[0]
    # "best" rather than "lowest": the primary objective may be maximised, and a
    # reason string that says the wrong direction is worse than no reason at all.
    _best_word = "lowest" if _sign(OBJECTIVES[primary][0]) > 0 else "highest"
    # Everything below reasons about LOSSES — value * sign, always
    # lower-is-better — so a maximised objective like `gap_closed_mean_utt`
    # needs no separate code path. Only the printed messages use raw values.
    signs = [_sign(OBJECTIVES[o][0]) for o in objectives]

    if len(objectives) == 1 or policy == "best_value":
        pick = front.iloc[0]
        reason = f"{policy}: single objective, {_best_word} {primary}"
        if len(objectives) > 1 and policy != "best_value":
            reason = (f"best_value: {policy!r} needs a second objective, "
                      f"and this study has one")
    else:
        cost = objectives[1]
        v_raw, c_raw = front[primary].to_numpy(float), front[cost].to_numpy(float)
        v, c = signs[0] * v_raw, signs[1] * c_raw          # losses
        if policy == "cheapest":
            i = int(np.argmin(c))
            reason = f"cheapest: best {cost}"
        elif policy == "knee":
            # Normalise each loss axis onto [0, 1] over the front, then take the
            # point closest to (0, 0). A degenerate axis (every trial identical)
            # is dropped rather than divided by zero.
            def unit(a):
                span = a.max() - a.min()
                return np.zeros_like(a) if span <= 0 else (a - a.min()) / span
            i = int(np.argmin(np.hypot(unit(v), unit(c))))
            reason = "knee: shortest normalised distance to the utopia point"
        elif policy == "fastest_within_value":
            # Slack is taken against |best|, not against `best`, so the rule
            # means the same thing whichever direction the objective has and
            # does not invert when the best value is negative — which
            # `gap_closed*` genuinely can be, for a router that loses to the BSM.
            best_raw = v_raw[np.argmin(v)]
            slack = tol * abs(best_raw)
            ok = np.flatnonzero(v <= v.min() + slack + 1e-12)
            i = int(ok[np.argmin(c[ok])])
            reason = (f"fastest_within_value: best {cost} among the {len(ok)} "
                      f"front trial(s) within {tol:.1%} of the best {primary} "
                      f"({best_raw:.6g}), i.e. slack {slack:.6g}")
        elif policy == "best_within_cost":
            if max_cost is not None:
                budget, how = float(max_cost), f"max_cost={float(max_cost):.6g}"
            else:
                if tol <= 0:
                    raise ValueError(f"tol must be positive, got {tol}")
                # Over EVERY completed trial, not just the front: "a typical
                # trial" is a fact about the search, and the front is by
                # construction its extremes. The cost objective is always
                # minimised and positive, so no sign handling is needed here.
                allc = np.array([t.values[1] for t in done], float)
                ref = float(np.median(allc)) if anchor == "median" else float(allc.min())
                if anchor not in ("median", "min"):
                    raise ValueError(f"anchor must be 'median' or 'min', got {anchor!r}")
                budget = ref * float(tol)
                how = (f"{tol:g}x the {anchor} completed-trial {cost} "
                       f"({ref:.6g}) = {budget:.6g}")
            ok = np.flatnonzero(c_raw <= budget * (1 + 1e-12))
            if not len(ok):
                # Reachable with an absolute `max_cost` nothing met, or a `tol`
                # below 1 against the `min` anchor. Say so loudly and fall back
                # to the cheapest rather than return something that silently
                # violates the budget.
                warnings.warn(f"no front trial costs <= {how}; the cheapest on the "
                              f"front is {c_raw.min():.6g}. Falling back to it.")
                i = int(np.argmin(c_raw))
                reason = f"best_within_cost: budget ({how}) unmet, took the cheapest"
            else:
                i = int(ok[np.argmin(v[ok])])
                reason = (f"best_within_cost: {_best_word} {primary} among the "
                          f"{len(ok)} of {len(front)} front trial(s) with "
                          f"{cost} <= {how}")
        else:
            raise ValueError(f"unknown policy {policy!r}; have {SELECT_POLICIES}")
        pick = front.iloc[i]

    trial = by_number[int(pick["trial"])]
    if verbose:
        vals = "   ".join(f"{o} {_fmt_objective(o, trial.values[i])}"
                          for i, o in enumerate(objectives))
        print(f"  selected trial {trial.number} from a front of {len(front)}: {vals}")
        print(f"    rule: {reason}")
    return trial, front, reason

import shutil


class StudyStore:
    """A resumable Optuna storage: SQLite on disk, mirrored to the Hub.

    Parameters
    ----------
    name : the file's stem. `<name>.db` locally, `studies/<name>.db` in the repo.
    dirname : local directory for the file. Defaults to the working directory.
    repo_id, spec : where the mirror lives; both default the way `FasttHub` does.
    sync : `False` keeps the SQLite file and skips the Hub entirely — the right
        setting on a machine whose disk persists.
    sync_every : push after this many finished trials. Every push is a commit,
        so this trades repo history against how much a teardown can cost you.
    """

    def __init__(self, name, dirname=None, repo_id=None, spec=None, sync=True,
                 sync_every=25):
        self.name = name
        self.path = Path(dirname or ".") / f"{name}.db"
        self.path_in_repo = f"studies/{name}.db"
        self.repo_id, self.spec, self.sync = repo_id, spec, bool(sync)
        self.sync_every = int(sync_every)
        self._pushed_at = 0

    def __repr__(self):
        n = f"{self.path.stat().st_size / 1e6:.1f} MB" if self.path.exists() else "absent"
        return (f"StudyStore({self.name!r}, {self.path} [{n}], "
                f"{'sync -> ' + self.path_in_repo if self.sync else 'local only'})")

    def hub(self):
        from fastt.hub import FasttHub                    # local: optional dep
        return FasttHub(self.repo_id, spec=self.spec)

    def url(self):
        """The `storage=` string Optuna wants."""
        return f"sqlite:///{self.path}"

    def trial_count(self, path=None):
        """Total trials across every study in a database file. 0 if unreadable."""
        path = Path(path or self.path)
        if not path.exists():
            return 0
        try:
            url = f"sqlite:///{path}"
            return sum(len(optuna.load_study(study_name=s.study_name, storage=url).trials)
                       for s in optuna.get_all_study_summaries(url, include_best_trial=False))
        except Exception:                                      # noqa: BLE001
            return 0

    def pull(self, verbose=True):
        """Reconcile the local database with the Hub's copy. Returns self.

        Called once per session, before any study is created. **The copy with
        more trials wins**, and that rule rather than "the Hub is authoritative"
        is deliberate: they are the same database with one writer, so more
        trials means strictly further along, and the case that actually happens
        is a Colab *kernel* restart where the disk survives. There the local
        file holds every trial and the Hub holds only as many as the last sync —
        overwriting local with Hub would silently throw away the trials since
        then, which is exactly what this class exists to prevent.

        A genuine *runtime* teardown leaves no local file at all, so the Hub
        copy wins by default and nothing special is needed for it.
        """
        if not self.sync:
            return self
        mine = self.trial_count()
        # Downloaded beside the target, not over it, so a smaller Hub copy never
        # destroys a larger local one even transiently.
        staged = self.path.with_suffix(".db.hub")
        got = self.hub().pull_file(self.path_in_repo, staged, verbose=verbose)
        if got is None:
            if verbose:
                print(f"  no copy on the Hub — "
                      + (f"continuing the local {self.path} ({mine} trials)"
                         if mine else f"starting a new database at {self.path}"))
            return self
        theirs = self.trial_count(staged)
        if theirs >= mine:
            shutil.move(str(staged), str(self.path))
            if verbose:
                print(f"  using the Hub copy: {theirs} trials (local had {mine})")
        else:
            staged.unlink(missing_ok=True)
            if verbose:
                print(f"  KEEPING the local database: {mine} trials, ahead of the "
                      f"Hub's {theirs}. It will be pushed at the next sync.")
        return self

    def push(self, message=None, verbose=True):
        """Upload the current database. A no-op when `sync=False`, read-only, or nothing exists."""
        if not self.sync or read_only() or not self.path.exists():
            return None
        try:
            return self.hub().push_file(self.path, self.path_in_repo,
                                        message=message or f"Update {self.path_in_repo}",
                                        verbose=verbose)
        except Exception as e:                                 # noqa: BLE001
            # The mirror is a safety net, not the record: the local database
            # holds every trial and the next push (or a re-run of the cell, or
            # the next `pull()`, which keeps the copy with more trials) sends
            # it. A Hub rate limit must not kill a search that is going well.
            print(f"  (could not mirror {self.path_in_repo} to the Hub: "
                  f"{str(e).splitlines()[0][:160]} — the local copy is kept)")
            return None

    def callback(self):
        """An Optuna callback that pushes every `sync_every` finished trials."""
        def cb(study, trial):
            if not self.sync or read_only():
                return
            n = len(study.trials)
            if n - self._pushed_at >= self.sync_every:
                self._pushed_at = n
                self.push(message=f"{self.name}: {n} trials", verbose=False)
        return cb

    def studies(self):
        """Every study name already in the database, with its trial counts."""
        if not self.path.exists():
            return pd.DataFrame(columns=["study", "trials", "complete", "pruned"])
        rows = []
        for s in optuna.get_all_study_summaries(self.url(), include_best_trial=False):
            trials = optuna.load_study(study_name=s.study_name,
                                       storage=self.url()).trials
            rows.append({"study": s.study_name, "trials": len(trials),
                         "complete": sum(t.state == TrialState.COMPLETE for t in trials),
                         "pruned": sum(t.state == TrialState.PRUNED for t in trials)})
        return pd.DataFrame(rows)

# Optuna samplers, by the name a notebook uses. `random` draws every trial
# independently from the space and so spends an exact budget with no adaptation
# — which is the point when the protocol fixes B in advance and a reader has
# to be able to see that no trial was informed by any other.
SAMPLERS = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                                   group=True),
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
}


def trial_metrics_line(attrs):
    """One trial's `TABLE_METRICS` and selection distribution, as one line.

    Reads the user attributes `Tuner.objective` records on every finished trial
    — the same `METRICS` functions on the scoring rows — so what the search
    prints per trial is the results table's own metric set under its own
    headers. Gap and accuracy figures are shown as percentages, WERs as
    decimals, matching `format_results`. Each value takes its family's colour
    (`metric_token`) where colour can be shown, and is bare text elsewhere.
    """
    parts = []
    for m in TABLE_METRICS:
        v = attrs.get(m)
        if v is None or not np.isfinite(v):
            continue
        lab = TABLE_LABELS.get(m, m)
        parts.append(metric_token(lab, f"{100 * v:.1f}%" if m.startswith(("gap_closed", "acc_"))
                                  else f"{v:.4f}"))
    dist = attrs.get("selection_dist")
    if dist:
        parts.append(metric_token("dist", dist))
    return "  ".join(parts)


class Tuner:
    """The data one study needs, plus the partition that keeps the test honest.

    Holds the pooled design matrix, the random re-split of §7.5, and the two
    reference points every trial is read against. One `Tuner` serves every arm,
    so all of them are searched on identical rows.

    Parameters
    ----------
    store, members : as everywhere else in §7.
    holdout_frac : share of the pooled utterances the tuner may see at all.
    inner_val_frac : share of THAT which a trial is scored on rather than fitted.
    partition_seed : fixes the cut. Independent of the CV seeds by construction —
        they index different randomisations and sharing a constant between them
        would couple the tuning partition to the fold structure.
    model_seed : every estimator fit, every trial.
    nthread : XGBoost reproduces only at a FIXED thread count. Changing this
        changes results; it is a determinism parameter, not a performance one.
    objectives : what a trial is scored on, in order. One name is the ordinary
        single-objective search; two make the study multi-objective and its
        answer a Pareto front (§7.5.1b). The FIRST is the primary everywhere —
        `diagnose` ranks by it and every selection policy measures against it.
    """

    def __init__(self, store, members, holdout_frac=0.25, inner_val_frac=0.30,
                 partition_seed=20260820, model_seed=42, study_seed=7, nthread=4,
                 max_params=60_000_000, epochs_full=200,
                 use_extra_features=True, use_duration=True, pruner="median",
                 inner_folds=1, objectives=("corpus_wer",), sampler="tpe"):
        self.store, self.members = store, tuple(members)
        self.model_seed, self.study_seed = model_seed, study_seed
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {sorted(SAMPLERS)}")
        self.sampler = sampler
        self.max_params = max_params
        self.objectives = check_objectives(objectives)
        self.multi = len(self.objectives) > 1
        # `trial.report` carries a single value, so Optuna refuses it on a
        # multi-objective study — pruning and several objectives cannot both be
        # had. Coerce rather than fail at trial 1, but say so: the round will
        # cost more than the single-objective one it replaces.
        if self.multi and pruner not in (None, "none"):
            warnings.warn(f"multi-objective study: pruner={pruner!r} is not "
                          f"available (trial.report takes one value), using 'none'")
            pruner = "none"
        self.pruner = pruner
        self.inner_folds = int(inner_folds)
        if self.inner_folds < 1:
            raise ValueError(f"inner_folds must be >= 1, got {inner_folds}")

        # Kept because an evaluation has to rebuild the SAME design matrix: a
        # configuration tuned with the MFCC columns in it is not a configuration
        # for a design without them, and `run_cv` defaults both to True whatever
        # was searched. `HPTRun.evaluate` forwards these two.
        self.use_extra_features = bool(use_extra_features)
        self.use_duration = bool(use_duration)
        self.X, self.source = build_design(store, self.members,
                                           use_extra_features, use_duration)
        self.E = store.err(self.members)
        self.nref = store.nref
        self.Y = wer_matrix(self.E, self.nref)
        # `store.texts` is `None` until `load_texts` is called, and that is the
        # honest state to pass on: a fusion builder raises a message naming the
        # call, rather than a selection run paying for transcripts it will not
        # read.
        self.ctx = BuildContext(nthread=nthread, epochs_full=epochs_full,
                                errors=self.E, texts=getattr(store, "texts", None),
                                members=self.members, cache={})

        self.rows = random_partition(len(self.X), holdout_frac, inner_val_frac,
                                     partition_seed)
        self.partition_seed = partition_seed

        # Standardised on the FIT rows only — see the note above.
        fit = self.rows["fit"]
        self.mu, self.sd = self.X[fit].mean(0), self.X[fit].std(0).clip(1e-8)

        # With `inner_folds > 1` every tuning row is scored, in turn, by a model
        # that did not fit it — so the trial's score is read over the whole tuning
        # hold-out rather than over one fixed 30% cut of it. The reference points
        # move with it, or `gap_closed` would be a ratio of two different row sets.
        self.score_rows = self.rows["val"] if self.inner_folds == 1 else self.rows["tune"]

        # `bsm` stays a choice made on the FIT rows in both cases: it is a
        # reference point, and picking it on the rows it is then measured over
        # would flatter it.
        bsm = int((self.E[fit].sum(0) / self.nref[fit].sum()).argmin())
        sr = self.score_rows
        self.bsm, self.bsm_model = bsm, self.members[bsm]
        self.bsm_wer = self.corpus_wer(sr, np.full(len(sr), bsm))
        self.oracle_wer = self.corpus_wer(sr, self.E[sr].argmin(1))

    def __repr__(self):
        return (f"Tuner({'+'.join(self.members)}, "
                f"tune {len(self.rows['tune']):,} / eval {len(self.rows['eval']):,}, "
                f"D={self.X.shape[1]}, inner_folds={self.inner_folds}, "
                f"objectives={'+'.join(self.objectives)})")

    def directions(self):
        """The `direction`s Optuna is handed, one per objective."""
        return [OBJECTIVES[o][0] for o in self.objectives]

    def inner_splits(self, fidelity=1.0):
        """The `(fit_rows, val_rows)` pairs one trial is fitted and scored over.

        `inner_folds == 1` returns the single fixed `fit`/`val` cut. Anything
        larger returns a k-fold partition of the whole tuning hold-out, so a
        trial's score averages over k splits instead of resting on one. That is
        the difference between ranking trials on a 3k-row sample and ranking them
        on the full hold-out, which matters when the spread between the top
        trials is smaller than one sample's sampling error.

        `fidelity` subsamples the FIT side only, in every fold. The scoring rows
        stay whole so trial values remain comparable across rounds.
        """
        rng = np.random.default_rng(self.partition_seed + 2)

        if self.inner_folds == 1:
            pairs = [(self.rows["fit"], self.rows["val"])]
        else:
            tune = self.rows["tune"]
            order = rng.permutation(len(tune))
            chunks = np.array_split(order, self.inner_folds)
            pairs = [(np.sort(tune[np.concatenate([c for j, c in enumerate(chunks)
                                                   if j != i])]),
                      np.sort(tune[np.sort(chunk)]))
                     for i, chunk in enumerate(chunks)]

        if fidelity >= 1.0:
            return pairs
        out = []
        for i, (f, v) in enumerate(pairs):
            sub = np.random.default_rng(self.model_seed + i).choice(
                len(f), max(1, int(len(f) * fidelity)), replace=False)
            out.append((f[np.sort(sub)], v))
        return out

    def scaled(self, rows):
        """The standardised design matrix on `rows`."""
        return (self.X[rows] - self.mu) / self.sd

    def corpus_wer(self, rows, choice):
        """Corpus WER on `rows` when routing to `choice` — sum(errors)/sum(words)."""
        return float(self.E[rows, np.asarray(choice, dtype=int)].sum()
                     / self.nref[rows].sum())

    def gap_closed(self, wer):
        """A trial's WER as a share of the BSM-to-oracle gap on the scoring rows."""
        gap = self.bsm_wer - self.oracle_wer
        return float("nan") if gap <= 0 else (self.bsm_wer - wer) / gap

    def report(self, verbose=True):
        """The hold-out's own briefing — sizes and the two reference points."""
        sizes = pd.DataFrame([{"partition": k, "n_utts": len(v),
                               "share": len(v) / len(self.X)}
                              for k, v in self.rows.items()])
        if verbose:
            show_table(sizes.round(4),
                       title=f"{len(self.X):,} pooled utterances x {self.X.shape[1]:,} features",
                       caption="the tuning partition — what a trial may fit, what it is "
                               "scored on, and what it never sees")
            print(f"  on the tuning-score rows: bsm({self.bsm_model}) "
                  f"{paint(f'{self.bsm_wer:.4f}', 'bold')}   "
                  f"oracle {paint(f'{self.oracle_wer:.4f}', 'bold')}   "
                  f"gap {self.bsm_wer - self.oracle_wer:.4f}")
            print(f"  the {len(self.rows['eval']):,} evaluation rows are never "
                  f"seen by a trial")
            print(f"  objectives   : " + ", ".join(
                f"{o} ({OBJECTIVES[o][0]}) — {OBJECTIVES[o][1]}"
                for o in self.objectives))
            if self.multi:
                print("                 multi-objective: the answer is a Pareto "
                      "front and pruning is off")
        return sizes

    # ------------------------------------------------------------ objective --

    def _progress(self, name, n_trials):
        """An Optuna callback that prints one line per finished trial.

        `study.optimize` is otherwise silent until it returns, which on a
        120-trial final search means twenty minutes of nothing. Optuna's own
        per-trial logging is at INFO and the notebooks set the level to WARNING
        to suppress its noise, so this prints directly instead.
        """
        def callback(study, trial):
            done = completed(study)
            pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
            secs = ((trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_complete and trial.datetime_start else float("nan"))

            # The running best on the PRIMARY objective, in ITS direction.
            # `study.best_value` raises on a multi-objective study, so it is
            # never read here.
            _s = _sign(OBJECTIVES[self.objectives[0]][0])
            best_val = (min((t.values[0] for t in done), key=lambda x: _s * x)
                        if done else None)
            best = _fmt_objective(self.objectives[0], best_val) if done else "--"

            # Column widths are computed on the bare text and colour is added
            # afterwards, so escape codes never disturb the alignment.
            if trial.state == TrialState.COMPLETE and trial.values:
                objs = [f"{_OBJ_LABEL.get(o, o)} {_fmt_objective(o, v)}"
                        for o, v in zip(self.objectives, trial.values)]
                gap = trial.user_attrs.get("gap_closed")
                gap_txt = (f"gap {gap:+.1%}" if gap is not None and not np.isnan(gap)
                           else "")
                plain = "  ".join(objs + ([gap_txt] if gap_txt else []))
                what = "  ".join([paint(objs[0], "bold"), *objs[1:]]
                                 + ([signed(gap_txt, gap)] if gap_txt else []))
                improved = best_val is not None and trial.values[0] == best_val
            else:
                steps = len(trial.intermediate_values)
                plain = "pruned" + (f" @ step {steps}" if steps else " (too large)")
                what, improved = state("pruned") + plain[len("pruned"):], False
            what += " " * max(0, 38 - len(plain))
            best_txt = paint(f"best {best} *", "green", "bold") if improved else f"best {best}  "

            n_front = f"  front {len(study.best_trials):2d}" if self.multi and done else ""
            print(f"  {paint(name, 'dim')} {paint(f'{trial.number + 1:4d}/{n_trials}', 'bold')}  "
                  f"{what} {best_txt}  pruned {pruned:3d}{n_front}  "
                  f"{paint(f'{secs:5.1f}s', 'dim')}", flush=True)
            # The results table's own metric set, on the trial's scoring rows —
            # the same definitions the final table uses, so a trial line can be
            # read against Table 3 directly. Only for a finished trial: a
            # pruned one recorded nothing.
            if trial.state == TrialState.COMPLETE and trial.values:
                print("      " + trial_metrics_line(trial.user_attrs), flush=True)

        return callback

    def objective(self, spec, space, fidelity=1.0, max_iter=None, pruner="median"):
        """An Optuna objective over this tuner's hold-out.

        Returns one value per entry of `self.objectives`, in that order — a bare
        float when there is one, which is what keeps a single-objective study
        byte-identical to the one §7.5 has always run. Every cost in
        `OBJECTIVES` is measured and recorded as a user attribute whether or not
        it is being optimised, so a single-objective study still leaves a record
        of what its trials cost.

        `fidelity` subsamples the FIT rows only. The scoring rows stay fixed, so
        trial values remain comparable between a low-fidelity discovery round and
        the full-fidelity final search. **Timings are not comparable that way**:
        a discovery trial is fast because `fidelity` and `discovery_cap` made it
        so. Read a front only within the round that produced it.

        `pruner` selects which axis a trial is pruned on, and the two are
        mutually exclusive because `trial.report` has a single step axis:

        * `"median"` — steps are **training iterations**. Only `prunable` arms
          report, and they report `expected_wer`, the quantity their loop
          descends.
        * `"wilcoxon"` — steps are **utterances**. Every arm reports, after the
          fit, one value per scoring row; `WilcoxonPruner` then pairs those
          values across trials. See the note in `study()` about what this does
          and does not buy.
        """
        seed = self.model_seed
        splits = self.inner_splits(fidelity)
        # Standardise per inner fold, on that fold's fit rows only — reusing one
        # scaler across folds would let every fold's scoring half contribute to
        # the statistics it is then scored under. `run_cv` refits per fold for
        # the same reason.
        prepared = [(self.scaled(f), self.Y[f], f, self.scaled(v), self.Y[v], v)
                    for f, v in splits]

        def objective(trial):
            p = spec.suggest(trial, space, max_iter=max_iter, **suggest_kwargs(spec, self.ctx))

            # Size guard: refuse the trial before allocating anything. A model
            # that OOMs takes the whole study with it, and the corners of these
            # spaces that do so are reachable in one unlucky draw.
            if spec.param_count is not None:
                n_params = spec.param_count(p, self.X.shape[1])
                trial.set_user_attr("n_params", n_params)
                if n_params > self.max_params:
                    raise optuna.TrialPruned(f"{n_params:,} params > max_params")

            errs, refs, Es, per_fold, choices = [], [], [], [], []
            # `time.perf_counter` rather than `trial.duration`: the trial clock
            # also covers TPE's own sampling and the standardisation above, which
            # are the tuner's overhead and not the configuration's cost.
            t_trial = time.perf_counter()
            t_fit = t_pred = 0.0
            for step, (Xf, Yf, fit_rows, Xv, Yv, val_rows) in enumerate(prepared):
                model = spec.build(p, Xf.shape[1], seed=seed, ctx=self.ctx)

                # The iteration-axis callback belongs to the median pruner on a
                # SINGLE inner fold. With several folds the step axis is the fold
                # — one trial cannot report on both — and under Wilcoxon it is
                # the utterance.
                epoch_prune = (spec.prunable and pruner == "median"
                               and len(prepared) == 1)
                if epoch_prune:
                    def report(iteration, m):
                        trial.report(m.expected_wer(Xv, Yv), iteration)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    fit_kw = {"callback": report}
                else:
                    fit_kw = {}

                # Routed through `fit_arm`/`score_arm` so a trial is fitted and
                # read EXACTLY the way `run_cv` will fit and read it — including
                # the `wants_rows` door `soft_probe` needs and the
                # `produces="errors"` path its output takes. A tuner that scored
                # arms its own way would be tuning for a protocol the evaluation
                # does not run.
                if fit_kw:
                    t0 = time.perf_counter()
                    model.fit(Xf, Yf, nref=self.nref[fit_rows], **fit_kw)
                    t1 = time.perf_counter()
                    out = np.asarray(model.predict(Xv))
                    t_fit += t1 - t0
                    t_pred += time.perf_counter() - t1
                    kind = getattr(model, "produces", "choice")
                else:
                    # `fit_arm` already separates the two clocks — fit is paid
                    # once per deployment, inference every utterance forever —
                    # so they are read from it rather than measured again here.
                    timings = {}
                    kind, out = fit_arm(model, Xf, Yf, self.nref[fit_rows],
                                        fit_rows, Xv, val_rows, timings=timings)
                    t_fit += timings.get("fit_seconds", 0.0)
                    t_pred += timings.get("predict_seconds", 0.0)
                picked, choice = score_arm(self.E[val_rows], self.nref[val_rows], kind,
                                           out, self.bsm)
                # `-1` marks a fused utterance, as `run_cv` records it, so the
                # trial's selection distribution is read the same way a fold's is.
                choices.append(np.full(len(picked), -1) if choice is None else choice)
                errs.append(picked)
                refs.append(self.nref[val_rows])
                # Kept so every metric in `METRICS` can be evaluated below —
                # `routing_error` and the `gap_closed*` family all need the
                # member error matrix, not just the picked counts.
                Es.append(self.E[val_rows])
                per_fold.append(float(picked.sum() / self.nref[val_rows].sum()))

                # Fold-axis pruning: the running POOLED WER, which is the value
                # the trial will eventually return, so the series converges on it.
                if len(prepared) > 1 and pruner == "median":
                    running = float(np.concatenate(errs).sum()
                                    / np.concatenate(refs).sum())
                    trial.report(running, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            picked = np.concatenate(errs)
            refs_all = np.concatenate(refs)
            E_all = np.concatenate(Es, axis=0)
            total_ref = refs_all.sum()
            # Pooled across folds, so the value is a corpus WER over every scoring
            # row rather than an average of per-fold corpus WERs — the same micro
            # average §7 reports.
            wer = float(picked.sum() / total_ref)
            if len(per_fold) > 1:
                trial.set_user_attr("fold_wers", per_fold)
                trial.set_user_attr("fold_wer_sd", float(np.std(per_fold, ddof=1)))

            if pruner == "wilcoxon":
                # Corpus WER is a MICRO average, sum(errors)/sum(words), but
                # WilcoxonPruner minimises the MEAN of the reported values. So
                # each utterance reports its error count scaled by n/sum(nref) —
                # a constant across trials — whose mean is corpus WER exactly.
                # Reporting `errors_i / nref_i` instead would silently switch the
                # objective to a macro average and tune for a different quantity
                # than §7 reports.
                scale = len(picked) / total_ref
                order = np.random.default_rng(seed + trial.number).permutation(len(picked))
                for i in order:
                    trial.report(float(picked[i]) * scale, int(i))
                    if trial.should_prune():
                        # The docs ask for an estimate rather than a raise, so
                        # the sampler still learns from the trial.
                        break

            trial.set_user_attr("n_fit_rows", len(prepared[0][2]))
            trial.set_user_attr("n_inner_folds", len(prepared))
            # The selection distribution over the scoring rows — the `xx-yy-zz`
            # cell of the results table — so a trial that "wins" by always
            # picking one member is visible as such in the search log.
            shares = selection_shares(np.concatenate(choices), self.E.shape[1])
            trial.set_user_attr("selection_dist", format_shares(shares))
            for i, s in enumerate(shares):
                trial.set_user_attr(f"share_{i}", float(s))

            # EVERY metric and EVERY cost, recorded whether or not it is being
            # optimised. Eleven extra floats per trial is nothing beside the fit
            # that produced them, and it is what makes the objective a choice
            # you can revisit: a study that minimised `corpus_wer` can be
            # re-read under `gap_closed_mean_utt` afterwards, and `rescore` only
            # has to re-fit anything at all for trials run before this existed.
            scores = {}
            for name, (fn, _d) in METRICS.items():
                if name in CONTEXT_METRICS:
                    continue
                try:
                    scores[name] = float(fn(E_all, refs_all, picked, self.bsm))
                except Exception:                              # noqa: BLE001
                    scores[name] = float("nan")
            scores.update({
                "trial_seconds": time.perf_counter() - t_trial,
                "fit_seconds": t_fit,
                "predict_seconds": t_pred,
                # `param_count` is optional on a ModelSpec; NaN keeps the key
                # present and unusable rather than absent and surprising.
                "n_params": float(trial.user_attrs.get("n_params", float("nan"))),
            })
            for k, v in scores.items():
                trial.set_user_attr(k, float(v))

            values = [float(scores[o]) for o in self.objectives]
            if any(np.isnan(v) for v in values):
                # Optuna records a NaN objective as FAILED and the message lands
                # three frames from the cause — usually an arm with no
                # `param_count` scored on `n_params`, or a fold with no gap.
                bad = [o for o, v in zip(self.objectives, values) if np.isnan(v)]
                raise optuna.TrialPruned(f"{spec.name} cannot report {bad}")
            return values[0] if len(values) == 1 else tuple(values)

        return objective

    def make_pruner(self, kind=None, spec=None):
        """The pruner named by `kind`, or a pruner instance passed straight through.

        `"median"` prunes on the training-iteration axis and is the default: it
        is where this objective's cost actually accrues, since the fit dominates
        and every iteration is a chance to stop paying for it.

        `"wilcoxon"` prunes on the utterance axis, pairing per-utterance errors
        against the best trial. It is the statistically stronger test — and for
        this objective it saves almost nothing, because no per-utterance value
        exists until the fit is already paid for. It earns its keep only when the
        expensive loop *is* the instance loop (a k-fold or per-fold objective,
        as in §7.8). Measure before switching; §7.5.5 has the numbers.
        """
        kind = self.pruner if kind is None else kind
        if not isinstance(kind, str):
            return kind
        if kind == "wilcoxon":
            return optuna.pruners.WilcoxonPruner(p_threshold=0.1, n_startup_steps=8)
        if kind == "median":
            # `n_warmup_steps` is the arm's own, not a global constant — see
            # `ModelSpec.warmup`. With inner folds the step axis is the fold, so
            # one completed fold is the right warmup.
            warm = 1 if self.inner_folds > 1 else (spec.warmup() if spec else 1)
            return optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=warm)
        if kind == "none":
            return optuna.pruners.NopPruner()
        raise ValueError(f"unknown pruner {kind!r}")

    def study(self, spec, space, n_trials, fidelity=1.0, max_iter=None,
              seed=None, name="study", verbose=True, pruner=None, store=None,
              warm_start=None):
        """One study over `space`, run up to `n_trials` **in total**. Returns it.

        The sampler is `self.sampler` — `"tpe"` by default, `"random"` for a
        fixed-budget discrete random search.

        `store` is a `StudyStore` (or a raw Optuna storage string). With one, the
        study is persisted under `name` and **resumed**: `n_trials` is a target,
        not an increment, so a study that already holds 90 of 120 trials runs the
        remaining 30 and one that holds 120 runs none and returns immediately.
        Re-running the cell after a disconnect is therefore the whole recovery
        procedure. Without a store the study is in-memory and `n_trials` is run
        from scratch, exactly as before.

        `warm_start` is a list of parameter dicts queued ahead of any fresh
        sampling (see `enqueue_params`); they count against `n_trials` rather
        than adding to it.

        The name is the identity of a resumed study — see the warning in
        §7.5.1c. Two different search spaces under one name silently pool their
        trials, and TPE will happily model the union.
        """
        _require_optuna()
        kind = self.pruner if pruner is None else pruner
        storage = store.url() if isinstance(store, StudyStore) else store
        sampler = SAMPLERS[self.sampler](
            seed if seed is not None else self.study_seed)

        # `directions` vs `direction`: Optuna rejects a single-element
        # `directions` on some paths and `direction` is what a single-objective
        # study is stored with, so keeping the two calls distinct is what lets an
        # old single-objective database still be resumed.
        kw = dict(study_name=name, sampler=sampler, storage=storage,
                  load_if_exists=storage is not None,
                  pruner=self.make_pruner(kind, spec))
        study = (optuna.create_study(directions=self.directions(), **kw) if self.multi
                 else optuna.create_study(direction=self.directions()[0], **kw))
        # Written on the study itself so a database found later is
        # self-describing: `rescore` and `select_trial` can then be handed the
        # study alone rather than the objective tuple that produced it.
        study.set_user_attr("objectives", list(self.objectives))

        # Warm start BEFORE the count below, so queued trials are part of the
        # budget rather than extra work on top of it.
        if warm_start:
            enqueue_params(study, warm_start, verbose=verbose)

        # Resume arithmetic. RUNNING trials are counted as done: they are the
        # remains of a killed process and Optuna will never finish them, but
        # neither will it re-run them, so treating them as outstanding would
        # make the target unreachable. WAITING trials are the opposite — queued
        # work that has not run — so they must NOT count, or a warm start would
        # silently shrink the search it was meant to accelerate.
        already = sum(1 for t in study.trials if t.state != TrialState.WAITING)
        todo = max(0, n_trials - already)
        if verbose and already:
            print(f"  {name}: resuming — {already} trial(s) already in "
                  f"{getattr(store, 'path', storage)}, {todo} to go")
        if todo:
            callbacks = ([self._progress(name, n_trials)] if verbose else [])
            if isinstance(store, StudyStore):
                callbacks.append(store.callback())
            study.optimize(
                self.objective(spec, space, fidelity, max_iter,
                               pruner=kind if isinstance(kind, str) else "median"),
                n_trials=todo, gc_after_trial=True, callbacks=callbacks or None)
            if isinstance(store, StudyStore):
                store.push(message=f"{name}: {len(study.trials)} trials",
                           verbose=verbose)

        if verbose:
            done = completed(study)
            pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
            _s = _sign(OBJECTIVES[self.objectives[0]][0])
            best = (_fmt_objective(self.objectives[0],
                                   min((t.values[0] for t in done), key=lambda x: _s * x))
                    if done else "--")
            front = (f", {len(study.best_trials)} on the Pareto front"
                     if self.multi and done else "")
            print(f"  {paint(name, 'dim')}: {len(done)} complete, {len(pruned)} pruned, "
                  f"best {self.objectives[0]} {paint(best, 'green', 'bold')}{front} "
                  f"(bsm {self.bsm_wer:.4f}, oracle {self.oracle_wer:.4f})")
        return study

# Diagnosis thresholds. Grouped rather than inlined because they are one policy —
# "how much evidence before we move a range" — and tuning them individually is a
# good way to talk yourself into a space.
TOP_FRAC = 0.25              # "good trials" = this share of completed trials
EDGE = 0.10                  # outer 10% of a range counts as the boundary
PRESSURE = 0.30              # ...and this share of good trials there -> expand
SPAN_MIN = 0.30              # good trials inside this share of the range -> contract
IMPORTANCE_FLOOR = 0.02      # below this share of explained variance -> freeze
MIN_EVIDENCE = 8             # never act on a parameter sampled fewer times
MIN_FREEZE_EVIDENCE = 25     # ...and never FREEZE on fewer, importance is noisy
EXPAND = 3.0                 # expansion factor per round
MARGIN = 0.15                # slack kept when contracting

DISPLAY = ["param", "n_used", "n_top", "importance", "span",
           "pressure_low", "pressure_high", "detail", "action"]


def _importances(study, index=0):
    """fANOVA importances for objective `index`, defensively.

    Conditional spaces can defeat it, and a multi-objective study *requires* the
    `target=` argument — Optuna will not guess which objective you meant. Both
    failure modes end in `{}` rather than an exception, because an importance
    estimate is evidence for freezing a parameter and not having one only means
    nothing gets frozen.
    """
    target = (lambda t: t.values[index]) if len(study.directions) > 1 else None
    for evaluator in (None, optuna.importance.MeanDecreaseImpurityImportanceEvaluator()):
        try:
            kw = {"target": target} if target is not None else {}
            if evaluator is not None:
                kw["evaluator"] = evaluator
            return optuna.importance.get_param_importances(study, **kw)
        except Exception:                                      # noqa: BLE001
            continue
    return {}


def diagnose(study, space, top_frac=TOP_FRAC, edge=EDGE, pressure=PRESSURE,
             span_min=SPAN_MIN, importance_floor=IMPORTANCE_FLOOR,
             min_evidence=MIN_EVIDENCE, min_freeze_evidence=MIN_FREEZE_EVIDENCE):
    """One row per free parameter: where the good trials are, and what to do."""
    # Ranked by the PRIMARY objective, which on a multi-objective study is a
    # deliberate simplification: refining the space around the trials that are
    # good on cost as well would drag the search towards the cheap corner, and
    # the space is meant to be where the accuracy is. The cost objective shapes
    # the front, not the box.
    done = completed(study)
    if not done:
        raise ValueError("no completed trials to diagnose")
    sign = study_signs(study)[0]
    top = sorted(done, key=lambda t: sign * t.values[0])[
        :max(3, int(round(len(done) * top_frac)))]
    imp = _importances(study)

    rows = []
    for name, prm in space.items():
        if prm.frozen is not None:
            continue
        used = sum(1 for t in done if name in t.params)
        vals = [t.params[name] for t in top if name in t.params]
        row = {"param": name, "kind": prm.kind, "n_used": used, "n_top": len(vals),
               "importance": float(imp.get(name, np.nan))}

        if prm.kind == "cat":
            share = (pd.Series(vals).value_counts(normalize=True) if vals
                     else pd.Series(dtype=float))
            row["detail"] = ", ".join(f"{c}:{share.get(c, 0.0):.0%}" for c in prm.choices)
            dead = [c for c in prm.choices if share.get(c, 0.0) == 0.0]
            # Only drop a category we gave a fair chance: it must have been
            # sampled, and still never have reached the top group.
            sampled = pd.Series([t.params[name] for t in done
                                 if name in t.params]).value_counts()
            dead = [c for c in dead if sampled.get(c, 0) >= max(3, min_evidence // 2)]
            row["action"] = ("drop:" + "|".join(str(c) for c in dead)
                             if dead and len(dead) < len(prm.choices) else "keep")
            rows.append(row)
            continue

        if used < min_evidence or not vals:
            row.update(detail=f"only {used} trials", action="keep")
            rows.append(row)
            continue

        pos = np.array([prm.position(v) for v in vals])
        q05, q50, q95 = np.quantile(pos, [0.05, 0.5, 0.95])
        lo_press, hi_press = float((pos < edge).mean()), float((pos > 1 - edge).mean())
        row.update(low=prm.low, high=prm.high, log=prm.log,
                   pressure_low=lo_press, pressure_high=hi_press,
                   span=float(q95 - q05), q05=float(q05), q95=float(q95),
                   median=prm.value_at(q50),
                   detail=f"[{prm.low:.4g}, {prm.high:.4g}] median {prm.value_at(q50):.4g}")

        # Precedence: escape the box first, then simplify, then tighten.
        if lo_press >= pressure:
            row["action"] = "expand_low"
        elif hi_press >= pressure:
            row["action"] = "expand_high"
        elif (used >= min_freeze_evidence and not np.isnan(row["importance"])
              and row["importance"] < importance_floor):
            # Freezing is irreversible within a run, and an fANOVA importance on
            # ~20 trials is mostly noise — so it needs more evidence than merely
            # narrowing a range does.
            row["action"] = "freeze"
        elif q95 - q05 < span_min:
            row["action"] = "contract"
        else:
            row["action"] = "keep"
        rows.append(row)

    cols = ["param", "kind", "n_used", "n_top", "importance", "span",
            "pressure_low", "pressure_high", "detail", "action", "q05", "q95", "median"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]]


def refine(space, diag, study, expand=EXPAND, margin=MARGIN):
    """Apply the diagnosis. Returns a NEW space; the old one is untouched."""
    # `study.best_trial` raises on a multi-objective study; `best_trial` takes
    # the best on the primary objective, which is what a frozen value should be.
    best = best_trial(study).params
    new = dict(space)

    for _, r in diag.iterrows():
        name, action, prm = r["param"], r["action"], space[r["param"]]

        if action.startswith("drop:"):
            dead = set(action.split(":", 1)[1].split("|"))
            keep = tuple(c for c in prm.choices if str(c) not in dead)
            new[name] = replace(prm, choices=keep)
            if len(keep) == 1:
                new[name] = replace(new[name], frozen=keep[0])

        elif action in ("expand_low", "expand_high"):
            width = ((np.log(prm.high) - np.log(prm.low)) if prm.log
                     else (prm.high - prm.low))
            if action == "expand_low":
                low = prm.low / expand if prm.log else prm.low - expand * width
                new[name] = prm.clamped(low, prm.high)
            else:
                high = prm.high * expand if prm.log else prm.high + expand * width
                new[name] = prm.clamped(prm.low, high)

        elif action == "contract":
            # Around the good trials' [q05, q95], with slack — NOT around the
            # midpoint of the old range, which is an arbitrary place to look.
            new[name] = prm.clamped(prm.value_at(max(0.0, r["q05"] - margin)),
                                    prm.value_at(min(1.0, r["q95"] + margin)))

        elif action == "freeze":
            # At the best trial's value where it has one, else the median of the
            # good trials — either way, a value the search actually liked.
            new[name] = replace(prm, frozen=best.get(name, r.get("median")))

    return new


def describe(space):
    """The current space, as a table.

    `hard_low` / `hard_high` are shown because they are the limits `refine` may
    never expand past, and a parameter that ends the search pressed against one
    is a finding about the model rather than about the tuner.
    """
    rows = []
    for n, p in space.items():
        row = {"param": n, "hard_low": p.hard_low, "hard_high": p.hard_high}
        if p.frozen is not None:
            row.update(state="frozen", value=p.frozen)
        elif p.kind == "cat":
            row.update(state="categorical",
                       value="|".join(str(c) for c in p.choices))
        else:
            row.update(state="log" if p.log else "linear",
                       value=f"[{p.low:.4g}, {p.high:.4g}]")
        rows.append(row)
    return pd.DataFrame(rows)[["param", "state", "value", "hard_low", "hard_high"]]


def edit_space(space, **edits):
    """A COPY of `space` with named parameters retargeted. The old one is untouched.

    `spec.space()` is the author's default, not a law, and editing it by hand is
    the difference between a search you can steer and one you can only re-run.
    This is the door: every form below says the same thing to `Param`, in the
    notation that reads best at the call site.

    ```python
    space = edit_space(MODELS["fastt_boosted"].space(),
                       num_rounds=(50, 400),           # a new range
                       adam_lr=(1e-4, 1e-2, True),     # ...and make it log-scaled
                       depth=3,                        # freeze it here
                       transform_type=["mlp", "linear"],   # narrow the choices
                       patience=None,                  # unfreeze
                       batch_size={"hard_high": 4096}) # one field, by name
    ```

    A range is clamped to the parameter's `hard_low` / `hard_high` unless the
    edit moves those too. Clamping **warns** rather than raising: a range typed
    a little wide should do the obvious thing, but a range typed entirely
    outside the limits — `num_rounds=(20, 250)` on a parameter whose ceiling is
    16 — is a mistake, and it is invisible in the resulting table unless
    something says so. Raise the ceiling in the same call if you meant it:
    `num_rounds={"hard_high": 400}` first, then the range.
    """
    unknown = [k for k in edits if k not in space]
    if unknown:
        raise KeyError(f"no such parameter(s) {unknown}; "
                       f"this space has {sorted(space)}")
    out = dict(space)
    for name, v in edits.items():
        prm = out[name]
        if isinstance(v, Param):
            out[name] = v
        elif isinstance(v, dict):
            out[name] = replace(prm, **v)
        elif isinstance(v, tuple) and len(v) in (2, 3):
            lo, hi = v[0], v[1]
            prm = replace(prm, log=bool(v[2])) if len(v) == 3 else prm
            # Through `clamped` so an edit cannot put a range outside the
            # physical limits, and so an int range stays integral.
            new = prm.clamped(lo, hi)
            if (new.low, new.high) != (lo, hi):
                warnings.warn(
                    f"edit_space: {name}=({lo:g}, {hi:g}) clamped to "
                    f"[{new.low:g}, {new.high:g}] by hard_low={prm.hard_low}, "
                    f"hard_high={prm.hard_high}", stacklevel=2)
            out[name] = replace(new, frozen=None)
        elif isinstance(v, list):
            out[name] = replace(prm, kind="cat", choices=tuple(v), frozen=None)
        elif v is None:
            out[name] = replace(prm, frozen=None)
        else:
            out[name] = replace(prm, frozen=v)
    return out

@dataclass
class TuningResult:
    """One tuned algorithm: what won, what it was searched over, and what it cost."""

    name: str
    params: dict
    value: float
    gap_closed: float
    space: dict
    study: object = None
    history: list = field(default_factory=list)
    n_trials: int = 0
    seconds: float = 0.0
    # Multi-objective only; all four are `None` / empty on a single-objective run.
    objectives: tuple = ("corpus_wer",)
    values: tuple = ()          # the selected trial's value on EVERY objective
    front: object = None        # the final search's Pareto front, as a frame
    selection: str = ""         # the rule that turned the front into `params`
    trial_number: int = None

    def factory(self, tuner, seed=None):
        """A zero-argument builder for `run_cv`, closing over the tuned parameters."""
        spec = MODELS[self.name]
        in_features = tuner.X.shape[1]
        s = tuner.model_seed if seed is None else seed
        return lambda: spec.build(self.params, in_features, seed=s, ctx=tuner.ctx)

    def as_row(self):
        row = {"model": self.name, "value": self.value,
               "gap_closed": self.gap_closed, "n_trials": self.n_trials,
               "seconds": self.seconds}
        # The cost the selected trial paid, under its own name, so a table of
        # several arms reads as "this WER, at this price".
        row.update({o: v for o, v in zip(self.objectives[1:], self.values[1:])})
        row["params"] = json.dumps(self.params, default=str, sort_keys=True)
        return row


def tune(spec, tuner, rounds=2, trials_discovery=60, trials_final=120,
         fidelity=0.5, verbose=True, space=None, store=None, tag=None,
         select="best_within_cost", select_tol=2.0, select_max_cost=None,
         select_anchor="median", warm_start=None, warm_start_top=None,
         warm_start_key=None, exact_budget=False):
    """Discovery rounds, then the final search. Returns a `TuningResult`.

    Each discovery round searches the current space at low fidelity, diagnoses
    it, and applies the diagnosis. A round where every `action` reads `keep`
    means the space has converged and further rounds are wasted budget — the
    printed table is the thing to watch.

    Parameters
    ----------
    space : the space to start from. `None` takes `spec.space()`, which is the
        author's default; pass an `edit_space(...)` result to search a different
        box without editing the package. `refine` moves it from round to round
        either way, so what is passed here is a starting point, not a promise.
    store : a `StudyStore`, which makes every round resumable — see
        `Tuner.study`. Re-running `tune` with the same `store` and `tag` picks up
        wherever the last one stopped, round by round: a finished discovery round
        is replayed from the database in seconds rather than re-searched.
    tag : prefix for the study names, and therefore the resume key. **Required
        with a store**, because the default names (`discovery1`, `final`) would
        collide across corpora and arms and silently pool their trials. The
        notebooks pass `f"{DATASET}_{spec.name}"`.
    select, select_tol, select_max_cost, select_anchor : how a multi-objective
        front is turned back into one configuration; see `select_trial`.
        Ignored, apart from a note in `selection`, when the study has a single
        objective.
    exact_budget : run **exactly** `trials_final` trials for this arm, ignoring
        `spec.trials` and the floor of 8. Off by default; see the comment in the
        body for why the default is the other way. Pair it with
        `sampler="random"` and `rounds=0` for a fixed-budget search in which no
        trial is informed by any other.
    warm_start : a previous `Study`, or a list of parameter dicts, to evaluate
        before any fresh TPE draw. **This is how a run whose objective changed
        re-uses the one before it.** The old values are not carried over — they
        measure a different quantity — only the coordinates, which are then
        re-measured under the current objective. See `enqueue_params`.
    warm_start_top, warm_start_key : when `warm_start` is a study, keep only its
        best `warm_start_top` trials, ranked by `warm_start_key` (default: that
        study's own primary objective). Queued trials are paid for at full
        fidelity, so this is the budget dial — 40 of 276 is the usual shape.

        The warm start goes to the **final** search only. The discovery rounds
        exist to move the box, and they do that from a fresh spread; seeding them
        with the previous run's optimum would refine the space around an answer
        chosen under the old metric before the new one has had a look.
    """
    _require_optuna()

    if isinstance(spec, str):
        spec = MODELS[spec]
    if store is not None and not tag:
        raise ValueError("tune(store=...) needs a tag: study names are the resume "
                         "key, and 'discovery1' is not unique across corpora or arms")
    prefix = f"{tag}_" if tag else ""
    if exact_budget:
        # B trials for this arm, full stop: no `spec.trials` share and no floor.
        # The default is deliberately neither — a two-parameter space cannot use
        # the budget a fifteen-parameter one needs, so `spec.trials` spends the
        # total where it does something and the floor of 8 keeps a cheap arm
        # from being searched at all. Both are the right defaults and both make
        # "B trials per arm" false: at trials_final=10 the floor alone gives
        # `probe` 8 and `fastt_alternating` 10. A protocol that fixes B in
        # advance and reports it needs the other behaviour, and
        # needs to ask for it explicitly rather than discover it in a table.
        n_disc = n_final = int(trials_final)
        if n_final < 1:
            raise ValueError(f"exact_budget needs trials_final >= 1, got {trials_final}")
    else:
        n_disc = max(8, int(round(trials_discovery * spec.trials)))
        n_final = max(8, int(round(trials_final * spec.trials)))

    t0 = time.perf_counter()
    space = spec.space() if space is None else dict(space)
    history, n_seen = [], 0
    # A one- or two-dimensional space has nothing for the diagnosis to find: a
    # range cannot be "over-wide" when TPE covers it in a dozen draws, and
    # freezing the only free parameter would end the search.
    n_rounds = rounds if spec.n_free(space) > 2 else 0
    if verbose:
        print(paint(f"\n### tuning {spec.name}", "bold")
              + f": {spec.n_free(space)} free parameter(s), "
              f"{n_rounds} discovery round(s) x {n_disc} + {n_final} final trials"
              + (f", resumable as {prefix}*" if store is not None else ""))

    for i in range(1, n_rounds + 1):
        study = tuner.study(spec, space, n_disc, fidelity=fidelity,
                            max_iter=spec.discovery_cap,
                            seed=tuner.study_seed + i, name=f"{prefix}discovery{i}",
                            verbose=verbose, store=store)
        n_seen += len(study.trials)
        try:
            diag = diagnose(study, space)
        except ValueError as e:
            # Every trial pruned. Refining on nothing would narrow the space
            # towards whatever the pruner happened to kill last.
            print(f"  round {i}: {e}; keeping the space as it is")
            break
        if verbose:
            print(diag[[c for c in DISPLAY if c in diag.columns]].round(3)
                  .to_string(index=False))
        history.append({"round": i, "diagnosis": diag,
                        "best": best_trial(study).values[0]})
        space = refine(space, diag, study)

    queued = warm_start
    if queued is not None and hasattr(queued, "trials"):        # a Study
        queued = top_params(queued, top=warm_start_top, key=warm_start_key)
        if verbose:
            print(f"  warm start: {len(queued)} parameter set(s) taken from the "
                  f"previous study" +
                  (f" (top {warm_start_top} by "
                   f"{warm_start_key or 'its own primary objective'})"
                   if warm_start_top else ""))

    final = tuner.study(spec, space, n_final, fidelity=1.0, max_iter=None,
                        seed=tuner.study_seed + 100, name=f"{prefix}final",
                        verbose=verbose, store=store, warm_start=queued)
    n_seen += len(final.trials)

    trial, front, reason = select_trial(final, tuner.objectives, policy=select,
                                        tol=select_tol, max_cost=select_max_cost,
                                        anchor=select_anchor, verbose=verbose)
    params = spec.suggest(optuna.trial.FixedTrial(trial.params), space,
                          **suggest_kwargs(spec, tuner.ctx))
    wer = float(trial.values[0])

    return TuningResult(name=spec.name, params=params, value=wer,
                        gap_closed=tuner.gap_closed(wer), space=space,
                        study=final, history=history, n_trials=n_seen,
                        seconds=time.perf_counter() - t0,
                        objectives=tuner.objectives,
                        values=tuple(float(v) for v in trial.values),
                        front=front, selection=reason, trial_number=trial.number)


def tune_all(tuner, names=None, rounds=2, trials_discovery=60, trials_final=120,
             fidelity=0.5, verbose=True, store=None, tag=None, spaces=None, **kw):
    """Tune every named algorithm on the same hold-out. Returns `{name: TuningResult}`.

    Prints the winning configuration for each as it finishes rather than at the
    end, because on a real budget this runs for hours and a session that dies
    two arms in should still have printed the two — and with a `store`, one that
    dies two arms in resumes at the third.

    `tag` is extended with each arm's name, so the arms get separate studies in
    one database and `tune_all(store=..., tag="voxpopuli")` is resumable as a
    whole. Both may instead be dicts keyed by arm — `{arm: StudyStore}` and
    `{arm: study prefix}` — which gives every arm a database of its own.

    `spaces` is `{arm: space}` — a starting space per arm, for the arms it names.
    `tune`'s `space=` is singular and forwarding one through `**kw` would hand
    the same space to every algorithm, which is not a thing that can be right.
    """
    names = list(names or (list(OURS) + list(COMPETITORS)))
    spaces = spaces or {}
    out = {}
    for name in names:
        # A dict hands each arm its own resumable database and study prefix —
        # `hpt` does this, one file per arm; a single store and tag keep the
        # family layout, every arm in one file under `<tag>_<arm>_*`.
        st = store.get(name) if isinstance(store, dict) else store
        tg = tag.get(name) if isinstance(tag, dict) else (f"{tag}_{name}" if tag else None)
        res = tune(MODELS[name], tuner, rounds, trials_discovery, trials_final,
                   fidelity, verbose, store=st, space=spaces.get(name), tag=tg, **kw)
        out[name] = res
        print(paint(f"\n=== best {name}", "bold", "green")
              + f"   corpus WER {paint(f'{res.value:.4f}', 'bold')}   "
              f"gap closed {signed(f'{res.gap_closed:+.1%}', res.gap_closed)}   "
              f"({res.n_trials} trials, {res.seconds / 60:.1f} min)")
        print(paint(json.dumps(res.params, indent=2, default=str, sort_keys=True), "dim"))
    return out


def tuning_table(results):
    """Every tuned arm side by side, best first."""
    return (pd.DataFrame([r.as_row() for r in results.values()])
            .sort_values("value", ignore_index=True))


def best_estimators(results, tuner, seed=None):
    """`{name: factory}` for `run_cv` — the tuned configurations, ready to run."""
    return {name: res.factory(tuner, seed) for name, res in results.items()}


def save_tuning(results, path, tuner=None, extra=None):
    """Persist every tuned configuration as one JSON file.

    A tuned configuration is only valid for the corpus and member set it was
    tuned on, so both are written beside it. Loading a file whose `members` do
    not match the store in front of you is a mistake worth catching, and
    `load_tuning` catches it.
    """
    payload = {
        "members": list(tuner.members) if tuner else None,
        "dataset": tuner.store.spec.name if tuner else None,
        "partition_seed": tuner.partition_seed if tuner else None,
        "bsm_wer": tuner.bsm_wer if tuner else None,
        "oracle_wer": tuner.oracle_wer if tuner else None,
        "objectives": list(next(iter(results.values())).objectives) if results else None,
        "results": {n: {"params": r.params, "value": r.value,
                        "gap_closed": r.gap_closed, "n_trials": r.n_trials,
                        # The whole point of a multi-objective run is that the
                        # answer is "this WER at this price", so the price is
                        # written down beside it and the rule that chose it too.
                        "values": dict(zip(r.objectives, r.values)),
                        "selection": r.selection, "trial": r.trial_number}
                    for n, r in results.items()},
        **(extra or {}),
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {path}  ({len(results)} arm(s))")
    return payload


def load_tuning(path, tuner=None):
    """Restore `{name: TuningResult}` from `save_tuning`. Checks the member set."""
    payload = json.loads(Path(path).read_text())
    if tuner is not None and payload.get("members") not in (None, list(tuner.members)):
        raise ValueError(f"{path} was tuned on {payload['members']}, not "
                         f"{list(tuner.members)} — a configuration does not transfer")
    return {n: TuningResult(name=n, params=r["params"], value=r["value"],
                            gap_closed=r["gap_closed"], space={},
                            n_trials=r.get("n_trials", 0))
            for n, r in payload["results"].items()}

import re


def _hpt_digest(payload):
    """Eight hex characters over everything that makes one tuning run distinct."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


# The fields of an `hpt_identity` that fix WHICH ROWS a call cut — the
# evaluation complement included. Two calls that agree on these put every arm
# on identical folds whatever else differs, and that is what lets one arm's
# cross-validated scores stay valid while another arm is searched again.
PARTITION_FIELDS = ("dataset", "members", "n_rows", "n_features", "holdout_frac",
                    "partition_seed", "use_extra_features", "use_duration")


def partition_identity(identity):
    """The rows: the part of an `hpt_identity` that determines the partition."""
    return {k: identity[k] for k in PARTITION_FIELDS if k in identity}


def arm_identity(identity, arm, version=None):
    """One arm's search identity: the family's, minus the other arms.

    `hpt_identity` describes a whole call — the arm list and every space passed.
    An arm's *own* search depends on none of that: the same rows, objective and
    sampler, and its own space. Digesting only those is what lets a changed
    space for one arm leave every other arm's finished study where it is.
    `version` is a free tag for the one change no digest can see — the arm's
    code — and bumping it starts that arm, and only that arm, over.
    """
    out = {k: v for k, v in identity.items() if k not in ("models", "spaces")}
    out.update(arm=arm, space=(identity.get("spaces") or {}).get(arm, []))
    if version is not None:
        out["version"] = str(version)
    return out


def arm_study_name(spec, arm, identity, version=None):
    """`hpt_<corpus>_<arm>_<8 hex>` — one arm's database stem and study prefix."""
    return f"hpt_{spec.name}_{arm}_{_hpt_digest(arm_identity(identity, arm, version))}"


def _arm_names(spec, models, identity, name=None, versions=None):
    """`{arm: study stem}`. Derived per arm, or hung off an explicit family `name`."""
    versions = versions or {}
    out = {}
    for arm in models:
        v = versions.get(arm)
        out[arm] = (arm_study_name(spec, arm, identity, v) if name is None
                    else f"{name}_{arm}" + (f"_{v}" if v is not None else ""))
    return out


def cv_arm_key(identity, arm, params, version=None):
    """What one arm's cross-validated scores depend on, the rows aside.

    The configuration it runs — `params`, JSON-normalised so a fresh search and
    a reloaded file digest alike — and its `version`. The rows are
    `partition_identity`, which the caller puts beside it (`RunLog.per_arm`).
    A re-search that selects the same configuration is therefore a cache hit,
    which is correct: given the folds, the fit is a function of these alone.
    """
    key = {"arm": arm,
           "params": json.loads(json.dumps(params, sort_keys=True, default=str))}
    if version is not None:
        key["version"] = str(version)
    return key


@dataclass
class HPTRun:
    """One tuning run: what won, where it is stored, and what it may be scored on.

    Attributes
    ----------
    name : the family name — the JSON's stem, and the study database's before
        arms had databases of their own (still the legacy lookup key).
    tuner : the `Tuner`, which owns the partition. `tuner.rows` is the whole cut.
    results : `{arm: TuningResult}`.
    studies : `{arm: StudyStore}`, or `None` if `hpt(studies=False)`.
    path : where `save_tuning` wrote, or `None`.
    identity : the dict `name`'s digest was taken over — what the digest means.
    names : `{arm: study stem}` — each arm's database and study prefix.
    versions : `{arm: tag}` as passed; part of the arm's identity and CV key.
    """

    name: str
    tuner: object
    results: dict
    studies: object = None
    path: str = None
    identity: dict = field(default_factory=dict)
    names: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)

    def __repr__(self):
        best = min(self.results.values(), key=lambda r: r.value, default=None)
        won = f"best {best.name} {best.value:.4f}" if best else "no arms"
        return (f"HPTRun({self.name!r}, {len(self.results)} arm(s), {won}, "
                f"eval on {len(self.eval_rows):,} rows)")

    # ----------------------------------------------------------- the pieces --

    @property
    def arms(self):
        """`{name: factory}` — the tuned configurations, ready for `run_cv`."""
        return best_estimators(self.results, self.tuner)

    @property
    def eval_rows(self):
        """The rows the tuner never saw. The only rows an evaluation may use."""
        return self.tuner.rows["eval"]

    def table(self):
        """Every tuned arm side by side, best first. See §7.5.4 on reading it."""
        return tuning_table(self.results)

    def partition(self):
        """`partition_identity` of this run — the rows every arm is scored on."""
        return partition_identity(self.identity)

    def arm_keys(self):
        """`{arm: cv_arm_key}` — what `RunLog.per_arm` caches each arm's folds under."""
        return {a: cv_arm_key(self.identity, a, r.params, self.versions.get(a))
                for a, r in self.results.items()}

    def study_paths(self):
        """`{arm: {"db": path in the repo, "study": study name}}` — where each arm's trials are."""
        return {a: {"db": self.studies[a].path_in_repo if self.studies else None,
                    "study": f"{self.names.get(a, f'{self.name}_{a}')}_final"}
                for a in self.results}

    # ------------------------------------------------------------- evaluate --

    def evaluate(self, estimators=None, scheme=None, seeds=None, verbose=True, **kw):
        """5x2cv over the held-out rows. Returns `(fold_scores, fold_picks)`.

        The tuned arms plus `run_cv`'s parameter-free references, on the
        complement of the tuning partition — which is the whole point of this
        method existing rather than the call being written out.

        `estimators` adds to or replaces the tuned arms; pass
        `{**run.arms, "probe": lambda: SoftmaxProbeRouter(alpha=1.0)}` to put an
        untuned comparator on the identical folds. `use_extra_features` and
        `use_duration` are taken from the `Tuner`, not from `run_cv`'s defaults,
        because a configuration tuned on one design matrix is not a configuration
        for a different one.
        """
        if scheme is None:
            scheme = FiveByTwoSplit(seeds=seeds) if seeds else FiveByTwoSplit()
        arms = dict(self.arms if estimators is None else estimators)
        rows = self.eval_rows
        if verbose:
            print(f"{scheme.__class__.__name__} over {len(rows):,} of "
                  f"{len(self.tuner.X):,} rows — the {len(self.tuner.rows['tune']):,} "
                  f"the tuner saw are excluded.\narms: {', '.join(arms)}\n")
        return run_cv(self.tuner.store, self.tuner.members, arms, scheme=scheme,
                      rows=rows, verbose=verbose,
                      use_extra_features=self.tuner.use_extra_features,
                      use_duration=self.tuner.use_duration, **kw)


def _tuning_json_path(spec, name):
    """Where `hpt` mirrors its `<name>.json` on the Hub: `results/<corpus>/tuning/`."""
    return spec.results_path(f"tuning/{name}")


def load_hpt_json(spec, name, dirname=None, sync=True):
    """The `<name>.json` an earlier `hpt` wrote: the local copy, else the Hub's. `None` if neither."""
    local = Path(dirname or ".") / f"{name}.json"
    if local.exists():
        return json.loads(local.read_text())
    if not sync:
        return None
    from fastt.hub import FasttHub                        # local: optional dep
    try:
        got = FasttHub(spec=spec).pull_file(_tuning_json_path(spec, name), verbose=False)
    except Exception:                                      # noqa: BLE001
        return None
    return json.loads(Path(got).read_text()) if got else None


def hpt_records(spec, dirname=None, sync=True):
    """Every `hpt` record readable from here: `{family name: payload}`.

    Local `*.json` files carrying an `hpt` identity, plus the corpus's `tuning/`
    directory on the Hub. This is the index the legacy lookups search: a payload
    names its family database (`study`) and carries the identity and the selected
    configuration of every arm it tuned — enough to decide, without opening a
    database, whether an old search is the same search as a new one.
    """
    out = {}
    for p in sorted(Path(dirname or ".").glob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except Exception:                                  # noqa: BLE001
            continue
        if isinstance(payload, dict) and "hpt" in payload and "results" in payload:
            out[payload.get("study", p.stem)] = payload
    if sync:
        from fastt.hub import FasttHub                     # local: optional dep
        prefix = _tuning_json_path(spec, "")[:-len(".json")]
        if prefix not in _HUB_RECORDS:
            # Listed and read once per session: `hpt` and `hpt_status` ask for
            # this once per arm, and the answer does not change underneath them.
            try:
                names = [f[len(prefix):-len(".json")] for f in FasttHub(spec=spec).files()
                         if f.startswith(prefix) and f.endswith(".json")]
            except Exception:                              # noqa: BLE001
                names = []
            _HUB_RECORDS[prefix] = {n: payload for n in sorted(names)
                                    if (payload := load_hpt_json(spec, n, dirname, sync))}
        for n, payload in _HUB_RECORDS[prefix].items():
            out.setdefault(n, payload)
    return out


_HUB_RECORDS = {}


def _study_names(db):
    """The studies in a `StudyStore`'s file; empty if there is no file."""
    if not db.path.exists():
        return []
    return [s.study_name for s in
            optuna.get_all_study_summaries(db.url(), include_best_trial=False)]


def _legacy_study(spec, arm, target, family, version=None, dirname=None,
                  sync=True):
    """`arm`'s search in a family-layout database, as `(StudyStore, study prefix)`; else `None`.

    Before per-arm databases (2026-09-12) one file held every arm of an `hpt`
    call, under a name digested over the whole call, and those files are still
    on the Hub. An arm whose own identity is unchanged should resume from them
    rather than search again. Two places to look: the family this very call
    would have been named (`family`), and every `hpt` record whose identity
    gives `arm` the per-arm name `target` — the same rows, sampler and space,
    whatever the other arms in that call were.
    """
    candidates = [family]
    for name, payload in hpt_records(spec, dirname, sync).items():
        if (arm in payload.get("results", {}) and name not in candidates
                and arm_study_name(spec, arm, payload.get("hpt") or {}, version) == target):
            candidates.append(name)
    for name in candidates:
        src = StudyStore(name, dirname=dirname, spec=spec, sync=sync)
        src.pull(verbose=False)
        prefix = f"{name}_{arm}_"
        if f"{prefix}final" in _study_names(src):
            return src, prefix
    return None


def adopt_study(dst, src, src_prefix, dst_prefix, verbose=True):
    """Copy `src`'s `<src_prefix>{final,discovery*}` studies into `dst` as `<dst_prefix>...`.

    Returns how many were copied. Trials, states and user attributes travel
    with the study, so the copy resumes exactly where the original stopped.
    """
    n = 0
    for study in _study_names(src):
        if not study.startswith(src_prefix):
            continue
        rest = study[len(src_prefix):]
        if rest != "final" and not re.fullmatch(r"discovery\d+", rest):
            continue
        optuna.copy_study(from_study_name=study, from_storage=src.url(),
                          to_study_name=dst_prefix + rest, to_storage=dst.url())
        n += 1
    if verbose and n:
        print(f"  adopted {n} stud{'y' if n == 1 else 'ies'} from {src.path.name} "
              f"-> {dst.path.name}")
    return n


def hpt(store, members, models=OURS, name=None, objectives=("corpus_wer",),
        sampler="tpe", exact_budget=False,
        rounds=2, trials_discovery=60, trials_final=120, fidelity=0.5,
        holdout_frac=0.25, inner_val_frac=0.30, inner_folds=1,
        partition_seed=20260820, model_seed=42, study_seed=7, nthread=4,
        max_params=60_000_000, epochs_full=200, use_extra_features=True,
        use_duration=True, pruner="median", spaces=None, studies=True,
        sync=True, sync_every=25, dirname=None, save=True, retune=True,
        versions=None, verbose=True, strict=False, **kw):
    """Tune `models` on `store` and return an `HPTRun`. The §7.5 driver, in one call.

    Composes `Tuner` -> `StudyStore` -> `tune_all` -> `save_tuning` and hands
    back the pieces rather than hiding them: `run.tuner`, `run.results`,
    `run.arms` and `run.studies` are the same objects §7.5.4 builds by hand.

    Parameters
    ----------
    store, members : as everywhere in §7. The store's row count is in the study
        name, so re-capping a store starts a new search instead of pooling
        trials from a different row set.
    models : which algorithms, as keys of `MODELS`. Defaults to `OURS`, i.e.
        FASTT-Alternating and FASTT-Boosted.
    name : the family name — the JSON's stem. Derived from a digest when `None`.
        **Each arm's study is named on its own** (`arm_study_name`): a digest of
        the rows, the objective, the sampler and that arm's space, so changing
        one arm's space re-searches that arm and resumes every other. With an
        explicit `name`, the arms are `<name>_<arm>`; change it to start every
        arm over.
    versions : `{arm: tag}`. An arm's code is the one change its digest cannot
        see; bump the arm's tag and that arm alone is searched — and scored —
        afresh. Written into the JSON beside the identity.
    objectives : one name for an ordinary search, two for a Pareto front
        (§7.5.1b). Any key of `OBJECTIVES`.
    retune : `False` reads `<name>.json` back instead of searching, and raises if
        it was tuned on a different member set. This is the cheap path for a
        notebook re-run: the same call, one constant flipped.
    studies : keep a resumable SQLite database per arm. `sync` mirrors each to
        the Hub, so a runtime teardown costs at most `sync_every` trials. An arm
        whose database is empty is first looked for in the family-layout
        databases of earlier runs (`_legacy_study`) and adopted from there. Turn
        `studies` off only for a throwaway search — without it an interrupted
        run is lost.
    save : write `<name>.json` with every winning configuration, the partition
        seed and the reference points, so a result can be quoted without the
        database.
    spaces : `{arm: space}`, to start an arm somewhere other than its default
        box. Included in the arm's digest, because replaying trials sampled from
        one box as evidence for another is the one way to get a corrupt result.

    strict : read, never search — an arm whose final study holds fewer than
        `trials_final` complete trials raises `fastt.runlog.CacheMiss`.

    Anything else is forwarded to `Tuner` or to `tune`.
    """
    _require_optuna()
    members = tuple(members)
    models = tuple(models)
    versions = {k: str(v) for k, v in (versions or {}).items()}

    tuner = Tuner(store, members, holdout_frac=holdout_frac,
                  inner_val_frac=inner_val_frac, partition_seed=partition_seed,
                  model_seed=model_seed, study_seed=study_seed, nthread=nthread,
                  max_params=max_params, epochs_full=epochs_full,
                  use_extra_features=use_extra_features, use_duration=use_duration,
                  pruner=pruner, inner_folds=inner_folds, objectives=objectives,
                  sampler=sampler)
    spec = tuner.store.spec

    identity = hpt_identity(tuner, members, models, sampler=sampler, spaces=spaces,
                            holdout_frac=holdout_frac, inner_val_frac=inner_val_frac,
                            inner_folds=inner_folds, partition_seed=partition_seed,
                            use_extra_features=use_extra_features,
                            use_duration=use_duration)
    if exact_budget:
        kw = {**kw, "exact_budget": True}
    names = _arm_names(spec, models, identity, name, versions)
    name = name or f"hpt_{spec.name}_{_hpt_digest(identity)}"
    path = Path(dirname or ".") / f"{name}.json"

    if verbose:
        print(f"{paint(name, 'bold')}\n  corpus   {identity['dataset']}   "
              f"{identity['n_rows']:,} rows x {identity['n_features']:,} features")
        print(f"  members  {'+'.join(members)}")
        print(f"  arms     {', '.join(models)}   objectives {'+'.join(tuner.objectives)}")
        print(f"  file     {path}")
        for arm in models:
            print(f"  study    {names[arm]}")
        print()
        tuner.report()

    reuse = not retune and path.exists()
    dbs, adopted = None, {}
    if studies and not reuse:
        dbs = {}
        for arm in models:
            db = StudyStore(names[arm], dirname=dirname, spec=spec,
                            sync=sync, sync_every=sync_every).pull(verbose=verbose)
            if db.trial_count() == 0:
                found = _legacy_study(spec, arm, names[arm], name, versions.get(arm),
                                      dirname=dirname, sync=sync)
                if found and adopt_study(db, found[0], found[1], f"{names[arm]}_",
                                         verbose=verbose):
                    # Mirrored below, all arms in one commit: the Hub allows
                    # 128 commits an hour and five corpora x fourteen arms of
                    # one-file pushes is how that limit was found.
                    adopted[arm] = db
            dbs[arm] = db
        if strict:
            from fastt.runlog import CacheMiss
            for arm, db in dbs.items():
                study = f"{names[arm]}_final"
                try:                                           # counted as `tune` counts its budget
                    n = sum(t.state != TrialState.WAITING for t in
                            optuna.load_study(study_name=study, storage=db.url()).trials)
                except Exception:                              # noqa: BLE001 — absent
                    n = 0
                if n < trials_final:
                    raise CacheMiss(f"{arm}: {study} holds {n} of {trials_final} trials (strict run)")

    if reuse:
        results = load_tuning(path, tuner)
        print(f"reused {path} — {len(results)} arm(s), nothing searched")
    else:
        results = tune_all(tuner, names=models, rounds=rounds,
                           trials_discovery=trials_discovery,
                           trials_final=trials_final, fidelity=fidelity,
                           verbose=verbose, store=dbs, tag=names, spaces=spaces, **kw)
        if save:
            # `identity` goes into the JSON so the digest in the filename can be
            # read back rather than merely matched; `studies` says where each
            # arm's trials are, which the ablation notebook reads.
            save_tuning(results, path, tuner, extra={
                "hpt": identity, "study": name, "versions": versions,
                "studies": {a: {"db": names[a], "study": f"{names[a]}_final"}
                            for a in models}})
            if studies and sync and not read_only():
                # One commit: the kilobyte summary beside the databases — what
                # won, readable from the Hub without Optuna; `hpt_status` and
                # the legacy lookups read it — plus every database adopted
                # above. Arms that were actually searched were pushed by
                # `Tuner.study` as each finished, which is the teardown safety
                # net and is left alone.
                files = {db.path_in_repo: str(db.path) for db in adopted.values()}
                files[_tuning_json_path(spec, name)] = str(path)
                _push_files(StudyStore(name, spec=spec).hub(), files,
                            f"{name}: tuned configurations"
                            + (f", {len(adopted)} adopted database(s)" if adopted else ""))

    return HPTRun(name=name, tuner=tuner, results=results, studies=dbs,
                  path=str(path) if save else None, identity=identity,
                  names=names, versions=versions)


def _push_files(hub, files, message):
    """`{path_in_repo: local}` in one commit, per file where the Hub layer cannot batch.

    Never raises: a mirror that fails leaves every local file where it is and
    says so, and the run continues. The `push_files` door is the batched one.
    """
    try:
        if len(files) > 1 and hasattr(hub, "push_files"):
            hub.push_files(files, message, verbose=False)
        else:
            for repo, local in files.items():
                hub.push_file(local, repo, message, verbose=False)
        print(f"  pushed {len(files)} file(s) to the Hub in one commit")
    except Exception as e:                                     # noqa: BLE001
        print(f"  (could not mirror {len(files)} file(s) to the Hub: "
              f"{str(e).splitlines()[0][:160]} — local copies are kept)")


def hpt_identity(tuner, members, models, sampler="tpe", spaces=None,
                 holdout_frac=0.25, inner_val_frac=0.30, inner_folds=1,
                 partition_seed=20260820, use_extra_features=True, use_duration=True):
    """Everything that makes one tuning call a different search from another.

    Budget is deliberately NOT in here: `trials_final` is a target, so raising
    it resumes rather than restarts. The sampler IS: a random search and a TPE
    search over the same box are not the same experiment, and resuming one into
    the other would pool adaptive trials with independent ones under a single
    name — the exact failure the digest exists to prevent. `spaces` is digested
    **as passed**, every entry, including entries for arms not in `models`.

    This is the *family* identity — the JSON's name. Each arm's study is named
    over `arm_identity`, which drops `models` and every other arm's space.
    """
    return {
        "dataset": tuner.store.spec.name,
        "members": list(members),
        "models": list(models),
        "objectives": list(tuner.objectives),
        "n_rows": int(tuner.X.shape[0]),
        "n_features": int(tuner.X.shape[1]),
        "holdout_frac": float(holdout_frac), "inner_val_frac": float(inner_val_frac),
        "inner_folds": int(inner_folds), "partition_seed": int(partition_seed),
        "use_extra_features": bool(use_extra_features),
        "use_duration": bool(use_duration),
        "sampler": str(sampler),
        "spaces": {k: sorted((n, repr(p)) for n, p in v.items())
                   for k, v in (spaces or {}).items()},
    }


def hpt_status(store, members, models, trials_final, spaces=None, name=None,
               sampler="tpe", objectives=("corpus_wer",), holdout_frac=0.25,
               inner_val_frac=0.30, inner_folds=1, partition_seed=20260820,
               use_extra_features=True, use_duration=True, dirname=None,
               sync=True, versions=None, verbose=True, **tuner_kw):
    """What a matching `hpt(...)` call would find already done. Returns a frame.

    The change-detection preview: builds the same per-arm names `hpt` would,
    pulls each arm's database (the local copy or the Hub's, whichever holds more
    trials) — or, failing that, finds the arm in a family-layout database of an
    earlier run — and reports per arm how many trials exist against the budget
    and, where the search finished, the configuration it selected. So the
    question *"under this space, this budget, this sampler and this seed we
    previously searched and found X; use it?"* is answered before anything
    runs, one arm at a time. A changed space changes that arm's digest and
    that arm reads `absent` — the others do not — which is the detection.

    `status` per arm is one of `complete` (the budget is met and the study will
    be reused as is), `partial` (some trials exist; `hpt` resumes from them) or
    `absent` (nothing found; `hpt` searches from trial 1). `source` names the
    database the trials were found in — the arm's own, or the legacy family
    `hpt` will adopt them from. Nothing here fits a model. Every `Tuner`-shaping
    argument must be passed exactly as it will be passed to `hpt`, or the
    identity differs and the preview is of a different search.
    """
    _require_optuna()
    members, models = tuple(members), tuple(models)
    versions = {k: str(v) for k, v in (versions or {}).items()}
    tuner = Tuner(store, members, holdout_frac=holdout_frac,
                  inner_val_frac=inner_val_frac, partition_seed=partition_seed,
                  inner_folds=inner_folds, objectives=objectives, sampler=sampler,
                  use_extra_features=use_extra_features, use_duration=use_duration,
                  **tuner_kw)
    spec = tuner.store.spec
    identity = hpt_identity(tuner, members, models, sampler=sampler, spaces=spaces,
                            holdout_frac=holdout_frac, inner_val_frac=inner_val_frac,
                            inner_folds=inner_folds, partition_seed=partition_seed,
                            use_extra_features=use_extra_features,
                            use_duration=use_duration)
    names = _arm_names(spec, models, identity, name, versions)
    name = name or f"hpt_{spec.name}_{_hpt_digest(identity)}"

    rows = []
    for arm in models:
        db = StudyStore(names[arm], dirname=dirname, spec=spec, sync=sync)
        db.pull(verbose=False)
        study_name, source = f"{names[arm]}_final", "own"
        if study_name in _study_names(db):
            study = optuna.load_study(study_name=study_name, storage=db.url())
        else:
            study = None
            found = _legacy_study(spec, arm, names[arm], name, versions.get(arm),
                                  dirname=dirname, sync=sync)
            if found:
                src, prefix = found
                study = optuna.load_study(study_name=f"{prefix}final", storage=src.url())
                source = src.name
        done = completed(study) if study is not None else []
        n_all = len(study.trials) if study is not None else 0
        row = {"arm": arm, "study": names[arm], "trials": n_all, "complete": len(done),
               "budget": int(trials_final), "best_value": np.nan, "best_params": "",
               "source": source if n_all else ""}
        if done:
            best = best_trial(study)
            row["best_value"] = float(best.values[0])
            row["best_params"] = json.dumps(best.params, sort_keys=True, default=str)
        row["status"] = ("absent" if n_all == 0 else
                         "complete" if n_all >= trials_final else "partial")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(name=name, identity=identity, names=names,
                     hub_json=_tuning_json_path(spec, name))

    if verbose:
        print(f"{name}  ({identity['sampler']} sampler, budget {trials_final} per arm, "
              f"partition seed {identity['partition_seed']}, "
              f"{identity['n_rows']:,} rows x {identity['n_features']:,} features)")
        n_c = int((out["status"] == "complete").sum())
        if n_c == len(out):
            print(f"  every arm was previously searched to its budget under exactly "
                  f"its space — the configurations below will be REUSED "
                  f"(bump the arm's version, or pass a different `name=`, to search again):")
        elif n_c or (out["status"] == "partial").any():
            print(f"  {n_c} of {len(out)} arm(s) finished, "
                  f"{int((out['status'] == 'partial').sum())} partial — "
                  f"`hpt` resumes each from where it stopped:")
        else:
            print("  nothing found under this identity — `hpt` will search from trial 1:")
        obj = OBJECTIVES[tuner.objectives[0]][0][:3]
        if in_notebook():
            view = pd.DataFrame({
                "arm": out["arm"], "status": out["status"],
                "trials": [f"{a}/{b}" for a, b in zip(out["trials"], out["budget"])],
                f"best ({obj})": out["best_value"],
                "selected configuration": out["best_params"].where(out["complete"] > 0, "—"),
                "source": out["source"].where(~out["source"].isin(["", "own"]), "own")})
            show_table(view, best={f"best ({obj})": obj},
                       cell_style=lambda c, v, r: (f"font-weight:600;color:{HEX[STATES[v]]}"
                                                   if c == "status" and v in STATES else None),
                       mono=("selected configuration",), wrap=("selected configuration",))
        else:
            for _, r in out.iterrows():
                found = (f"best {obj} {r['best_value']:.4f}  "
                         + paint(r["best_params"], "dim") if r["complete"] else "")
                legacy = f"  [from {r['source']}]" if r["source"] not in ("", "own") else ""
                status = state(f"{r['status']:9s}")
                print(f"    {r['arm']:18s} {status} {r['trials']:3d}/{r['budget']:<3d} "
                      f"{found}{legacy}")
    return out

from fastt.estimators import SelectKBestPipeline


def _identity_transform(space):
    """`space` with the feature transformation frozen off — the paper's Raw-* rows."""
    return {**space,
            "transform_type": replace(space["transform_type"], frozen="identity")}


def raw_sdt_space():
    """`fastt_boosted`'s space with no feature transformation. The paper's Raw-SDT."""
    return _identity_transform(boosted_space())


# The filter width. Log-scaled because the interesting question is an order of
# magnitude — is 50 columns enough, or 500, or all of them — not a linear
# increment, and the paper's own grid {50, 100, 200, 500} is geometric too.
def _kbest_param():
    return Param("k", "int", low=32, high=2048, log=True, hard_low=8, hard_high=16384)


def kbest_xgboost_space():
    """ANOVA-F filter width on top of `adastt`'s space. The paper's SelectKBest-XGBoost."""
    return {**adastt_space(), "k": _kbest_param()}


def kbest_sdt_space():
    """ANOVA-F filter width on top of the identity-transform booster. SelectKBest-SDT."""
    return {**raw_sdt_space(), "k": _kbest_param()}


def suggest_kbest_xgboost(trial, space, max_iter=None):
    p = suggest_adastt(trial, space, max_iter)
    p["k"] = space["k"].suggest(trial)
    return p


def suggest_kbest_sdt(trial, space, max_iter=None, epochs_full=200):
    p = suggest_boosted(trial, space, max_iter, epochs_full)
    p["k"] = space["k"].suggest(trial)
    return p


def build_kbest_xgboost(p, in_features, seed=42, ctx=None):
    """`SelectKBest` -> `AdaSTTXGBoost`, both fitted inside the fold."""
    ctx = ctx or BuildContext()
    # The base sees the FILTERED width, which is what its own `in_features`
    # means. Passing the unfiltered one would only matter for a transform, and
    # this arm has none — but getting it right keeps the two k-best arms
    # symmetric, and `kbest_sdt` does read it.
    kept = min(int(p["k"]), in_features)
    return SelectKBestPipeline(
        base=lambda: build_adastt(p, kept, seed=seed, ctx=ctx), k=p["k"])


def build_kbest_sdt(p, in_features, seed=42, ctx=None):
    """`SelectKBest` -> the identity-transform soft-tree booster."""
    ctx = ctx or BuildContext()
    kept = min(int(p["k"]), in_features)
    return SelectKBestPipeline(
        base=lambda: build_boosted(p, kept, seed=seed, ctx=ctx), k=p["k"])


def kbest_sdt_param_count(p, in_features):
    """Same count as the booster, over the filtered width."""
    return boosted_param_count(p, min(int(p["k"]), in_features))


# Registered into the same table everything else is drawn from, so `tune_all`,
# `hpt` and `best_estimators` reach them by name with no special case.
MODELS.update({
    "raw_sdt": ModelSpec("raw_sdt", raw_sdt_space, suggest_boosted, build_boosted,
                         discovery_cap=40, trials=0.8, prunable=True,
                         param_count=boosted_param_count),
    "kbest_xgboost": ModelSpec("kbest_xgboost", kbest_xgboost_space,
                               suggest_kbest_xgboost, build_kbest_xgboost,
                               trials=0.7),
    "kbest_sdt": ModelSpec("kbest_sdt", kbest_sdt_space, suggest_kbest_sdt,
                           build_kbest_sdt, discovery_cap=40, trials=0.8,
                           prunable=True, param_count=kbest_sdt_param_count),
})

# The paper's comparator set, in Table 3's order. `adastt` is Raw-XGBoost.
RIVALS = ("adastt", "raw_sdt", "kbest_xgboost", "kbest_sdt")

def _param_source(p):
    """One `Param(...)` constructor call, as source. Only non-default fields."""
    bits = [repr(p.name), repr(p.kind)]
    if p.kind == "cat":
        bits.append(f"choices={tuple(p.choices)!r}")
    else:
        bits.append(f"low={p.low!r}")
        bits.append(f"high={p.high!r}")
        if p.log:
            bits.append("log=True")
    for field_name in ("hard_low", "hard_high", "frozen"):
        v = getattr(p, field_name)
        if v is not None:
            bits.append(f"{field_name}={v!r}")
    return f"Param({', '.join(bits)})"


def space_source(space, varname="SPACE"):
    """The space as pasteable Python. Round-trips through `eval` with `Param` in scope.

    This is what keeps a search space *in* the notebook rather than behind an
    import. The output is deliberately verbose — every bound and every hard
    limit spelled out — because the point is that a reader can audit the box
    without running the code that produced it.
    """
    lines = [f"{varname} = {{p.name: p for p in ["]
    lines += [f"    {_param_source(p)}," for p in space.values()]
    lines.append("]}")
    return "\n".join(lines)


def check_spaces(spaces, strict=True):
    """Validate hand-written spaces against what each arm's `suggest_*` reads.

    Every `suggest_*` is `{k: space[k].suggest(trial) for k in <fixed list>}`,
    and that list is fixed in code — a space is a set of *ranges* for known
    parameters, never a way to add or drop one. Which makes two hand-editing
    mistakes possible, with very different symptoms:

    * a **missing** parameter raises `KeyError` on the first trial — loud, but
      only after the store has loaded and the partition has been cut, which on
      a real corpus is minutes of work before the error;
    * an **extra** parameter is silently ignored. Nothing reads it, no trial
      varies it, and the study looks entirely normal. This is the one worth a
      function: a space written out by hand and then edited is exactly where a
      renamed or invented parameter goes unnoticed.

    Raises `KeyError` naming both sets. `strict=False` downgrades the extra-key
    case to a warning, for the deliberate case of carrying a parameter that a
    future `suggest_*` will read.
    """
    problems = []
    for name, space in spaces.items():
        if name not in MODELS:
            problems.append(f"{name}: not a known arm; have {sorted(MODELS)}")
            continue
        want = set(MODELS[name].space())
        got = set(space)
        if want - got:
            problems.append(f"{name}: missing {sorted(want - got)}")
        if got - want:
            msg = f"{name}: unknown {sorted(got - want)} (nothing would read these)"
            problems.append(msg) if strict else warnings.warn(msg)
    if problems:
        raise KeyError("hand-written spaces do not match the arms they name:\n  "
                       + "\n  ".join(problems))
    return spaces


def space_from_trials(trials, space, top=10, key=None, pad=0.0,
                      freeze_constant=True, min_evidence=MIN_EVIDENCE,
                      verbose=True):
    """A refined space: the smallest box containing the best `top` trials' values.

    The starter refinement rule, and the one to replace. It is deliberately
    naive — it reads only *where the good trials landed*, not the shape of the
    response surface, and it cannot expand a range the optimum sits outside of
    (`diagnose`/`refine` is the machinery that can). What it is good at is the
    thing this stage needs: turning a wide exploratory search into a narrow box
    that a final search can cover densely.

    Parameters
    ----------
    trials : a finished Optuna `Study`, or a `trial_frame` of one.
    space : the space that was actually **searched** — supplies each parameter's
        kind, scale and hard limits, and frozen parameters stay frozen. That is
        `TuningResult.space`, **not** `MODELS[name].space()`: the discovery
        rounds move the box before the trials are drawn, so refining against the
        author's default can produce a range that excludes trials it was
        supposed to contain, or one wider than anything was sampled from.
    top : how many of the best trials define the box.
    key : the metric to rank by. Defaults to the study's primary objective.
    pad : widen each numeric range by this fraction of its own width, **in the
        parameter's own scale** — so a log parameter is padded multiplicatively.
        `0.0` gives the exact hull of the surviving trials.
    freeze_constant : freeze a parameter that every surviving trial agreed on.
        This is where most of the dimensionality reduction comes from, and it is
        the one decision here worth arguing with: agreement among ten trials of
        a TPE search is weak evidence when the parameter barely matters, and
        strong evidence when it does. `_importances` tells the two apart.
    min_evidence : **advisory only — it does not change the box.** A parameter
        acted on from fewer than this many observations is named in a warning
        and flagged `!` in the summary, and is then narrowed or frozen anyway.

        It matters because several spaces here are *conditional*:
        `suggest_alternating` samples `lambda1`/`lambda2` only when
        `transform_type == "diagonal"` and `bottleneck_ratio` only for
        `low_rank`/`nonlinear`, so with `top=10` a conditional parameter can be
        refined — or frozen — on two or three observations. `refine` refuses to
        act below `MIN_EVIDENCE` and to freeze below `MIN_FREEZE_EVIDENCE` for
        exactly this reason; this function deliberately does not, because the
        judgement belongs to whoever reads the warning. Raise `top` until the
        conditional branches have enough trials, or edit the emitted source.
    """
    frame = trials if isinstance(trials, pd.DataFrame) else trial_frame(
        trials, objectives=(key,) if key else _study_objective_names(trials))
    if key is not None and key in frame.columns:
        higher = OBJECTIVES.get(key, ("minimize",))[0] == "maximize"
        frame = frame.sort_values(key, ascending=not higher)
    best = frame.head(int(top))
    if best.empty:
        raise ValueError("no completed trials to refine from")

    out, notes = {}, []

    def note(name, n_used, saw, became, acted):
        notes.append({"param": name, "n": n_used, "saw": saw, "became": became,
                      "acted": acted})

    for name, p in space.items():
        if p.frozen is not None or name not in best.columns:
            out[name] = p
            continue
        values = best[name].dropna()
        if values.empty:                       # never sampled (a conditional branch)
            out[name] = p
            note(name, 0, "not sampled", "kept", acted=False)
            continue

        n_used = len(values)
        if p.kind == "cat":
            seen = [c for c in p.choices if c in set(values)]
            if len(seen) == 1 and freeze_constant:
                out[name] = replace(p, frozen=seen[0])
                note(name, n_used, f"all {seen[0]!r}", "frozen", acted=True)
            else:
                out[name] = replace(p, choices=tuple(seen) or p.choices)
                note(name, n_used, f"{len(seen)} of {len(p.choices)}",
                     "|".join(str(c) for c in seen),
                     acted=len(seen) < len(p.choices))
            continue

        lo, hi = float(values.min()), float(values.max())
        if lo == hi and freeze_constant:
            out[name] = replace(p, frozen=int(lo) if p.kind == "int" else lo)
            note(name, n_used, f"all {lo:.4g}", "frozen", acted=True)
            continue
        if pad:
            # Padded through `position`/`value_at`, so a log parameter widens by
            # a factor and a linear one by an increment. Padding in raw units
            # would make `pad` mean something different per parameter.
            a, b = p.position(lo), p.position(hi)
            width = (b - a) or 1e-9
            lo, hi = p.value_at(a - pad * width), p.value_at(b + pad * width)
        new = p.clamped(lo, hi)
        out[name] = replace(new, frozen=None)
        note(name, n_used, f"[{lo:.4g}, {hi:.4g}]",
             f"[{new.low:.4g}, {new.high:.4g}]",
             acted=(new.low, new.high) != (p.low, p.high))

    # Acted on from too little evidence. The box is NOT changed back — see
    # `min_evidence` — but nothing here is allowed to be silent about it.
    weak = [n for n in notes if n["acted"] and n["n"] < min_evidence]
    if weak:
        warnings.warn(
            f"space_from_trials narrowed or froze {len(weak)} parameter(s) from "
            f"fewer than {min_evidence} observations: "
            + ", ".join(f"{n['param']} ({n['n']} of {len(best)})" for n in weak)
            + ". These are usually CONDITIONAL parameters, sampled only on some "
            "branches of the space; raise `top` or edit the emitted source.",
            stacklevel=2)

    if verbose:
        free_before = sum(1 for p in space.values() if p.frozen is None)
        free_after = sum(1 for p in out.values() if p.frozen is None)
        print(f"refined from the best {len(best)} of {len(frame)} trial(s): "
              f"{free_before} free dimension(s) -> {free_after}")
        for n in notes:
            flag = "!" if (n["acted"] and n["n"] < min_evidence) else " "
            print(f" {flag}{n['param']:<22} {n['n']:>3}/{len(best)}  "
                  f"{n['saw']:<26} -> {n['became']}")
        if weak:
            print(f"\n  ! {len(weak)} parameter(s) above were refined from fewer "
                  f"than {min_evidence} observations. The box was changed anyway; "
                  f"check them before pasting.")
    return out

from fastt.estimators import ArgminClassifier, QERouter


def argmin_clf_space():
    """`ArgminClassifier`: the ridge penalty, and nothing else."""
    return {p.name: p for p in [
        Param("alpha", "float", low=1e-4, high=1e3, log=True,
              hard_low=1e-8, hard_high=1e6),
    ]}


def qe_router_space():
    """`QERouter`: what it regresses, and the ridge path it selects over."""
    return {p.name: p for p in [
        Param("target", "cat", choices=("wer", "errors")),
        Param("alpha_lo", "float", low=1e-4, high=1e2, log=True,
              hard_low=1e-10, hard_high=1e4),
        Param("alpha_decades", "int", low=2, high=12, hard_low=1, hard_high=20),
        Param("n_alphas", "int", low=8, high=40, hard_low=2, hard_high=200),
        Param("standardize", "cat", choices=(True, False)),
    ]}


def suggest_argmin_clf(trial, space, max_iter=None):
    return {"alpha": space["alpha"].suggest(trial)}


def suggest_qe_router(trial, space, max_iter=None):
    return {k: space[k].suggest(trial) for k in
            ("target", "alpha_lo", "alpha_decades", "n_alphas", "standardize")}


def build_argmin_clf(p, in_features, seed=42, ctx=None):
    return ArgminClassifier(alpha=p["alpha"])


def build_qe_router(p, in_features, seed=42, ctx=None):
    return QERouter(target=p["target"], alpha_lo=p["alpha_lo"],
                    alpha_decades=p["alpha_decades"], n_alphas=p["n_alphas"],
                    standardize=p["standardize"])


MODELS.update({
    "argmin_clf": ModelSpec("argmin_clf", argmin_clf_space, suggest_argmin_clf,
                            build_argmin_clf, trials=0.1),
    "qe_router": ModelSpec("qe_router", qe_router_space, suggest_qe_router,
                           build_qe_router, trials=0.3),
})

# The simple selectors, in ladder order: what each one is allowed to know about
# WER goes up from left to right. `adastt` closes the ladder and is already in
# `RIVALS`, so it is not repeated here.
SELECTOR_BASELINES = ("argmin_clf", "probe", "soft_probe", "qe_router")

def build_adastt_ce(p, in_features, seed=42, ctx=None):
    return build_adastt(p, in_features, seed=seed, ctx=ctx, objective="cross_entropy")


def build_fastt_boosted_ce(p, in_features, seed=42, ctx=None):
    return build_boosted(p, in_features, seed=seed, ctx=ctx, objective="cross_entropy")


def build_fastt_alternating_ce(p, in_features, seed=42, ctx=None):
    return build_alternating(p, in_features, seed=seed, ctx=ctx, objective="cross_entropy")


MODELS.update({
    "adastt_ce": ModelSpec("adastt_ce", adastt_space, suggest_adastt, build_adastt_ce,
                           trials=MODELS["adastt"].trials),
    "fastt_boosted_ce": ModelSpec("fastt_boosted_ce", boosted_space, suggest_boosted,
                                  build_fastt_boosted_ce, discovery_cap=40,
                                  trials=MODELS["fastt_boosted"].trials, prunable=False,
                                  param_count=boosted_param_count),
    "fastt_alternating_ce": ModelSpec("fastt_alternating_ce", alternating_space,
                                      suggest_alternating, build_fastt_alternating_ce,
                                      discovery_cap=6,
                                      trials=MODELS["fastt_alternating"].trials,
                                      prunable=False),
})

# The loss ladder, rung by rung: (control, parent). The controls are FASTT with
# the objective removed; read the table row against row, not against a winner.
CE_ARMS = ("adastt_ce", "fastt_boosted_ce", "fastt_alternating_ce")
CE_PARENT = {"adastt_ce": "adastt", "fastt_boosted_ce": "fastt_boosted",
             "fastt_alternating_ce": "fastt_alternating"}

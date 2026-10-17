__all__ = ['ACHIEVABLE_MARGIN', 'ACC_TOLERANCES', 'METRICS', 'CONTEXT_METRICS', 'TABLE_METRICS', 'TABLE_LABELS', 'BaseSplitScheme', 'FiveByTwoSplit', 'RepetitionSplit', 'KFoldSplit', 'HoldoutSplit', 'picked_errors', 'set_achievable_margin', 'achievable_choice', 'achievable_errors', 'selection_shares', 'format_shares', 'build_design', 'reference_choices', 'fold_rng', 'wants_rows', 'fit_arm', 'score_arm', 'run_cv', 'fold_arms', 'fold_report', 'cost_table', 'selection_distribution', 'results_table', 'format_results', 'to_latex_rows']

import hashlib
import time

import numpy as np
import pandas as pd

from fastt.eda import fold_brief
from fastt.estimators import realised_wer, wer_matrix
from fastt.pretty import paint, show_table


class BaseSplitScheme:
    """A resampling scheme. Subclasses yield `(repeat, fold, train_idx, eval_idx)`.

    Subclass this to add a grouped or stratified variant — the rest of §7 only
    needs `split()` and `n_folds`, so nothing downstream changes.
    """

    name = "base"

    def split(self, n):
        raise NotImplementedError

    def repeat_folds(self, n, seed):
        """The folds of ONE repetition, addressed by `seed` rather than position.

        `split()` walks a fixed list of repetitions; this walks one, chosen by
        its seed. That is the whole difference between §7.3 and §7.8: a
        repetition you can name is a repetition you can run alone, cache, and
        later include in or exclude from an analysis.

        Returns a list of `(fold, train_idx, eval_idx)`.
        """
        raise NotImplementedError

    @property
    def n_folds(self):
        raise NotImplementedError


class FiveByTwoSplit(BaseSplitScheme):
    """Dietterich's 5x2: one shuffle per repetition, each cut into complementary halves.

    Parameters
    ----------
    seeds : one seed **per repetition**, given explicitly. Repetition `i` is
        determined by `seeds[i]` and nothing else, so a run is reproducible from
        this tuple alone and a repetition can be re-run in isolation. Defaults to
        five seeds; pass your own to pin them in a paper.
    seed : legacy single-seed form, kept working. When `seeds` is None the
        repetitions derive from `(seed, i)` exactly as before, so existing runs
        reproduce bit-for-bit.
    n_repeats : only consulted in the legacy form; with `seeds` it is `len(seeds)`.

    Because the 5x2cv t-statistic puts a single fold difference in its numerator
    (§7.4), `seeds[0]` alone decides that numerator. Report the seeds alongside
    any p-value.
    """

    name = "5x2cv"

    def __init__(self, n_repeats=5, seed=0, seeds=None):
        if seeds is not None:
            seeds = tuple(int(s) for s in seeds)
            if len(seeds) < 2:
                raise ValueError("5x2cv needs at least 2 repetitions")
            if len(set(seeds)) != len(seeds):
                # Two identical seeds give two identical repetitions, which
                # silently deflates s_i^2 and inflates the statistic.
                raise ValueError(f"seeds must be distinct, got {seeds}")
            n_repeats = len(seeds)
        self.n_repeats, self.seed, self.seeds = n_repeats, seed, seeds

    @property
    def n_folds(self):
        return 2 * self.n_repeats

    def repeat_seed(self, repeat):
        """The seed governing one repetition — the whole of its randomness."""
        return self.seeds[repeat] if self.seeds is not None else (self.seed, repeat)

    def repeat_folds(self, n, seed):
        """One shuffle, cut into complementary halves, then both orderings of it."""
        # A generator per repetition, keyed only on that repetition's seed, so
        # adding or re-running one never perturbs the others.
        order = np.random.default_rng(seed).permutation(n)
        a, b = order[: n // 2], order[n // 2:]
        return [(0, a, b), (1, b, a)]

    def split(self, n):
        for repeat in range(self.n_repeats):
            for fold, tr, ev in self.repeat_folds(n, self.repeat_seed(repeat)):
                yield repeat, fold, tr, ev


class RepetitionSplit(FiveByTwoSplit):
    """ONE repetition of a 5x2, addressed by its seed. Two folds.

    `FiveByTwoSplit` refuses fewer than two repetitions because the tests need
    them — but a *cache* wants exactly one: `RunLog.per_seed` runs each
    repetition alone, stores it under its own key, and concatenates. The folds
    are byte-identical to the ones `FiveByTwoSplit(seeds=(..., seed, ...))`
    would produce for that seed, because both come from `repeat_folds(n, seed)`.
    """

    def __init__(self, seed):
        self.n_repeats, self.seed, self.seeds = 1, int(seed), (int(seed),)


class KFoldSplit(BaseSplitScheme):
    """Plain shuffled k-fold. The usual **inner** loop for nested CV (§7.8).

    Not a replacement for 5x2cv on the outside: k-fold's training sets overlap
    heavily, which is precisely the dependence Dietterich's design avoids. Inside
    an outer fold that does not matter, because nothing is being tested there —
    the inner loop only has to rank configurations.
    """

    name = "kfold"

    def __init__(self, k=3, seed=0):
        if int(k) < 2:
            raise ValueError(f"k-fold needs at least 2 folds, got {k}")
        self.k, self.seed = int(k), seed

    @property
    def n_folds(self):
        return self.k

    def repeat_folds(self, n, seed):
        order = np.random.default_rng(seed).permutation(n)
        parts = np.array_split(order, self.k)
        return [(i, np.concatenate([q for j, q in enumerate(parts) if j != i]), parts[i])
                for i in range(self.k)]

    def split(self, n):
        for fold, tr, ev in self.repeat_folds(n, self.seed):
            yield 0, fold, tr, ev


class HoldoutSplit(BaseSplitScheme):
    """A single shuffled train/validation cut — the cheapest inner loop there is.

    The default for `Arm.inner` because inner cost multiplies: with `n_trials`
    trials over `n_folds` inner folds and 10 outer folds, a 3-fold inner loop
    costs three times a holdout for a ranking that is usually the same. Move to
    `KFoldSplit` when the tuning signal is noisy enough that the extra fits buy
    a different answer, not merely a more comfortable one.
    """

    name = "holdout"

    def __init__(self, val_frac=0.25, seed=0):
        if not 0 < val_frac < 1:
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
        self.val_frac, self.seed = float(val_frac), seed

    @property
    def n_folds(self):
        return 1

    def repeat_folds(self, n, seed):
        order = np.random.default_rng(seed).permutation(n)
        cut = int(np.clip(round(n * self.val_frac), 1, n - 1))
        return [(0, order[cut:], order[:cut])]

    def split(self, n):
        for fold, tr, ev in self.repeat_folds(n, self.seed):
            yield 0, fold, tr, ev

def picked_errors(E, choice):
    """(n,) error counts incurred by routing each utterance to `choice`."""
    return E[np.arange(len(E)), choice]


# Every metric below takes `picked` — the error count actually incurred on each
# utterance — rather than a member index. For a router that is `picked_errors(E,
# choice)`; for a fusion arm (§9) it is the error count of the fused transcript,
# which belongs to no member at all. Same metrics, both kinds of system, and the
# oracle-bounded assumption lives in the metrics that actually make it rather
# than in the plumbing.
def _corpus_wer(E, nref, picked, bsm):
    return float(picked.sum() / nref.sum())


def _mean_utt_wer(E, nref, picked, bsm):
    return float((picked / np.maximum(nref, 1)).mean())


def _routing_error(E, nref, picked, bsm):
    """Tie-tolerant: matching any member that achieves the row minimum counts."""
    return float((picked > E.min(axis=1)).mean())


def _regret_per_word(E, nref, picked, bsm):
    """Words lost against the per-utterance oracle. **Negative** for a fusion arm
    that beats it, which is not a bug — ROVER can invent a transcript no member
    produced, so it is not bounded below by `E.min(axis=1)`."""
    return float((picked - E.min(axis=1)).sum() / nref.sum())


def _gap_closed(E, nref, picked, bsm):
    """Share of the BSM-to-oracle gap captured. `nan` where the fold has no gap."""
    total = nref.sum()
    bsm_wer, oracle_wer = E[:, bsm].sum() / total, E.min(axis=1).sum() / total
    gap = bsm_wer - oracle_wer
    if gap <= 0:                                  # nothing to win on this fold
        return float("nan")
    return float((bsm_wer - picked.sum() / total) / gap)


def _gap_closed_mean_utt(E, nref, picked, bsm):
    """`gap_closed`, but on the MEAN utterance WER rather than the corpus WER.

    Corpus WER is a micro average: a long utterance counts for as many words as
    it has, so the headline `gap_closed` is dominated by long turns. This is the
    macro reading — every utterance counts once — and on a corpus of very uneven
    turn lengths (AMI, where segments run from a word to a sentence) the two can
    disagree about whether a router helped at all. Report both; a router that
    wins on one and loses on the other is buying long utterances at the expense
    of short ones, or the reverse.

    `nan` where the fold has no macro gap, matching `gap_closed`.
    """
    utt = np.maximum(nref, 1)
    bsm_mw = float((E[:, bsm] / utt).mean())
    oracle_mw = float((E.min(axis=1) / utt).mean())
    gap = bsm_mw - oracle_mw
    if gap <= 0:
        return float("nan")
    return float((bsm_mw - (picked / utt).mean()) / gap)


def _gap_closed_abs(E, nref, picked, bsm):
    """The same quantity in WER points, not as a ratio.

    Always defined — unlike `gap_closed`, which divides by a gap that can be
    zero — and it is the number a deployment feels. A negative value means the
    arm did worse than simply always using the BSM.
    """
    total = nref.sum()
    return float(E[:, bsm].sum() / total - picked.sum() / total)


# How many words a switch must save before the achievable oracle takes it. The
# standard oracle switches for a single word, and a large share of its "gain" is
# exactly that: near-ties between two members that differ by one substitution.
# Nothing predicts those — they are not a property of the utterance, they are
# noise in the reference — so a headroom measured against them is a target no
# router can hit, and reporting a router as closing 8% of it says less than it
# appears to. This margin is the corpus-independent knob that makes the target
# reachable; 2 words is the default and `set_achievable_margin` changes it.
ACHIEVABLE_MARGIN = 2.0


def set_achievable_margin(words):
    """Set the global switch threshold, in reference words. Returns the old value.

    A module global rather than a parameter because `METRICS` entries take a
    fixed `(E, nref, picked, bsm)` signature — that signature is what lets every
    metric be computed for every arm on every fold without any caller knowing
    which metrics exist. Set it once, at the top of a run, and record it: two
    runs at different margins are not comparable and nothing downstream checks.
    """
    global ACHIEVABLE_MARGIN
    old, ACHIEVABLE_MARGIN = ACHIEVABLE_MARGIN, float(words)
    return old


def achievable_choice(E, bsm, margin=None):
    """The achievable oracle's per-utterance choice.

    Keeps the best single model unless another member saves **more than
    `margin` words** on that utterance, in which case it takes the best one.
    Returns member indices, so it passes through every metric exactly like a
    fitted arm's choices do.
    """
    margin = ACHIEVABLE_MARGIN if margin is None else margin
    best = E.argmin(axis=1)
    gain = E[:, bsm] - E[np.arange(len(E)), best]
    return np.where(gain > margin, best, bsm)


def achievable_errors(E, bsm, margin=None):
    """(N,) errors the achievable oracle would make."""
    return E[np.arange(len(E)), achievable_choice(E, bsm, margin)]


def _gap_closed_achievable(E, nref, picked, bsm):
    """Share of the BSM-to-**achievable**-oracle gap captured.

    The same ratio as `gap_closed` with a nearer denominator, so it reads as
    "how much of the reachable prize did this arm take". It can exceed 1: an arm
    is free to harvest the near-tie gains the achievable oracle declines, and a
    value above 1 says it did. `nan` where the margin leaves no gap at all,
    which on a corpus of near-identical members is the finding.
    """
    total = nref.sum()
    bsm_wer = E[:, bsm].sum() / total
    ach_wer = achievable_errors(E, bsm).sum() / total
    gap = bsm_wer - ach_wer
    if gap <= 0:
        return float("nan")
    return float((bsm_wer - picked.sum() / total) / gap)


def _gap_closed_achievable_mean_utt(E, nref, picked, bsm):
    """`gap_closed_achievable` on the MEAN utterance WER — the macro reading."""
    utt = np.maximum(nref, 1)
    bsm_mw = float((E[:, bsm] / utt).mean())
    ach_mw = float((achievable_errors(E, bsm) / utt).mean())
    gap = bsm_mw - ach_mw
    if gap <= 0:
        return float("nan")
    return float((bsm_mw - float((picked / utt).mean())) / gap)


def _acc_achievable(E, nref, picked, bsm):
    """Share of rows routed to within `ACHIEVABLE_MARGIN` words of the minimum.

    The accuracy that matches the achievable oracle: a pick that misses the best
    member by one word is not a routing failure in any sense a deployment cares
    about, and both other accuracies count it as one.

    It is the **loosest** of the three, not the middle one:
    `acc_achievable >= acc_lenient >= acc_strict`, because it counts every row
    lenient accuracy counts plus the near-misses. Report all three — lenient is
    dominated by rows where every member ties, strict is the hardest reading
    over the decisive rows only, and this one is the reading that tracks WER,
    since a near-miss costs almost nothing in words.
    """
    return float((picked <= E.min(axis=1) + ACHIEVABLE_MARGIN).mean())


def _best_mask(E):
    """`(is_best, n_best)` — which members achieve each row's minimum, and how many."""
    is_best = E == E.min(axis=1, keepdims=True)
    return is_best, is_best.sum(axis=1)


def _acc_lenient(E, nref, picked, bsm):
    """Share of rows routed to *a* member achieving the minimum. Ties count as hits.

    `<=` rather than `==` so a §9 fusion arm, which can beat every member, is not
    scored as a miss for doing better than the oracle.
    """
    return float((picked <= E.min(axis=1)).mean())


def _acc_strict(E, nref, picked, bsm):
    """Lenient accuracy over the **decisive** rows only — those with a unique best.

    This is the accuracy that can actually be wrong. Over the whole corpus a
    router is handed a large majority of free rows (§6.9), so lenient accuracy
    is dominated by them and two arms that differ substantially on the contested
    rows can look nearly identical. `nan` when no row is decisive.
    """
    _, n_best = _best_mask(E)
    decisive = n_best == 1
    if not decisive.any():
        return float("nan")
    return float((picked[decisive] <= E[decisive].min(axis=1)).mean())


def _tie_rate(E, nref, picked, bsm):
    """Share of rows where every member ties. A property of the fold, not the arm."""
    is_best, n_best = _best_mask(E)
    return float((n_best == E.shape[1]).mean())


def _decisive_rate(E, nref, picked, bsm):
    """Share of rows with a unique best member. A property of the fold, not the arm."""
    _, n_best = _best_mask(E)
    return float((n_best == 1).mean())


# Selection accuracy at an explicit word tolerance, as a family. `N = 0` is
# `_acc_lenient` by construction and is listed anyway: a table reporting
# accuracy at N in {0, 1, 2} should read its three columns off three metrics
# with one naming scheme, not off two unrelated names and a global.
#
# Unlike `_acc_achievable` these do NOT consult `ACHIEVABLE_MARGIN`. That is the
# point — the margin is a run-wide setting chosen once, and this family exists
# to vary the tolerance WITHIN one run and show the reader how fast a routing
# claim decays as near-ties are forgiven.
def _acc_at(n):
    def metric(E, nref, picked, bsm):
        return float((picked <= E.min(axis=1) + n).mean())
    metric.__name__ = f"_acc_tol{n}"
    metric.__doc__ = (f"Share of rows routed to within {n} word(s) of the "
                      f"minimum. Fixed tolerance, independent of "
                      f"`ACHIEVABLE_MARGIN`.")
    return metric


ACC_TOLERANCES = (0, 1, 2)


# Gap closed at an explicit word tolerance, as a family — the counterpart of
# `acc_tol{n}`. The denominator is the gap from the BSM to the oracle that only
# switches expert when it saves MORE than `n` words (`achievable_choice` at
# margin `n`), so `n = 0` is `gap_closed` exactly: at margin 0 the switching
# oracle takes every strictly-better member and its WER equals the plain
# oracle's. Unlike `gap_closed_achievable` these never read `ACHIEVABLE_MARGIN`.
def _gap_closed_at(n, macro=False):
    def metric(E, nref, picked, bsm):
        if macro:
            utt = np.maximum(nref, 1)
            ref = float((E[:, bsm] / utt).mean())
            target = float((achievable_errors(E, bsm, n) / utt).mean())
            got = float((picked / utt).mean())
        else:
            total = nref.sum()
            ref = E[:, bsm].sum() / total
            target = achievable_errors(E, bsm, n).sum() / total
            got = picked.sum() / total
        gap = ref - target
        if gap <= 0:
            return float("nan")
        return float((ref - got) / gap)
    kind = "mean_utt" if macro else "corpus"
    metric.__name__ = f"_gap_closed_{kind}_tol{n}"
    metric.__doc__ = (f"Share of the BSM-to-oracle gap closed on {kind} WER, where "
                      f"the oracle only switches for a gain of more than {n} word(s).")
    return metric


METRICS = {
    "corpus_wer": (_corpus_wer, "lower"),
    "mean_utt_wer": (_mean_utt_wer, "lower"),
    "routing_error": (_routing_error, "lower"),
    "regret_per_word": (_regret_per_word, "lower"),
    "gap_closed": (_gap_closed, "higher"),
    "gap_closed_abs": (_gap_closed_abs, "higher"),
    "gap_closed_mean_utt": (_gap_closed_mean_utt, "higher"),
    "gap_closed_achievable": (_gap_closed_achievable, "higher"),
    "gap_closed_achievable_mean_utt": (_gap_closed_achievable_mean_utt, "higher"),
    "acc_lenient": (_acc_lenient, "higher"),
    "acc_strict": (_acc_strict, "higher"),
    "acc_achievable": (_acc_achievable, "higher"),
    "tie_rate": (_tie_rate, "higher"),
    "decisive_rate": (_decisive_rate, "higher"),
    **{f"acc_tol{n}": (_acc_at(n), "higher") for n in ACC_TOLERANCES},
    **{f"gap_closed_tol{n}": (_gap_closed_at(n), "higher") for n in ACC_TOLERANCES},
    **{f"gap_closed_mean_utt_tol{n}": (_gap_closed_at(n, macro=True), "higher")
       for n in ACC_TOLERANCES},
}

# The metrics that describe the FOLD rather than the arm. Identical in every
# arm's row, so a paired test over them is degenerate by construction; `pairwise`
# would return `nan` rather than anything misleading, but naming them here lets
# a caller filter them out deliberately.
CONTEXT_METRICS = ("tie_rate", "decisive_rate")

# The manuscript's main-results metric set (Table 3), in column order. ONE
# definition, so the per-trial line the tuner prints, the per-fold table
# `run_cv` prints and the final table all report the same quantities under the
# same names — which is what makes the diagnostics comparable to the result.
TABLE_METRICS = (
    "corpus_wer", "mean_utt_wer",
    *[f"gap_closed_tol{n}" for n in ACC_TOLERANCES],
    *[f"gap_closed_mean_utt_tol{n}" for n in ACC_TOLERANCES],
    *[f"acc_tol{n}" for n in ACC_TOLERANCES],
)

# Short column headers for the same set, for terminal-width tables.
TABLE_LABELS = {
    "corpus_wer": "WER", "mean_utt_wer": "uWER",
    **{f"gap_closed_tol{n}": f"GC@{n}" for n in ACC_TOLERANCES},
    **{f"gap_closed_mean_utt_tol{n}": f"uGC@{n}" for n in ACC_TOLERANCES},
    **{f"acc_tol{n}": f"Acc@{n}" for n in ACC_TOLERANCES},
}


def selection_shares(choice, k):
    """`(k,)` share of utterances routed to each member. NaN vector for a fusion arm.

    `choice` is per-utterance member indices, or `-1` where the arm fused a
    transcript instead of picking a member (`fold_picks.choice`). A row with any
    fusion is reported as all-NaN rather than as the shares of the rows it did
    route, because a mixed policy's "distribution" would answer a different
    question from every other row's.
    """
    c = np.asarray(choice, dtype=int)
    if c.size == 0 or (c < 0).any():
        return np.full(k, np.nan)
    return np.bincount(c, minlength=k)[:k] / len(c)


def format_shares(shares, digits=2):
    """`xx-yy-zz`, the manuscript's selection-distribution cell. `—` when undefined."""
    shares = np.asarray(shares, dtype=np.float64)
    if shares.size == 0 or not np.isfinite(shares).all():
        return "—"
    return "-".join(f"{s:.{digits}f}" for s in shares)

def build_design(store, members, use_extra_features=True, use_duration=True):
    """`(X, source)` — the unstandardised design matrix over ALL pooled rows.

    Deliberately not standardised: in cross-validation the scaler has to be fitted
    per fold, on the training half only. `source` labels each column with the
    block it came from, for the composition analysis of §6.6.
    """
    blocks, source = [], []
    names = list(members) + (list(store.feature_only) if use_extra_features else [])
    for name in names:
        F = store.features[name]
        blocks.append(F)
        source += [name] * F.shape[1]
    if use_duration:
        blocks.append(np.log1p(store.n_frames)[:, None])
        source.append("duration")
    return np.concatenate(blocks, axis=1).astype(np.float64), np.array(source)


def reference_choices(E, bsm, rng=None, members=None, tolerances=()):
    """The parameter-free arms, as per-utterance choices.

    `random` is a **seeded realisation** rather than the expectation. It has to
    be a per-utterance choice to pass through the same metrics as everything
    else, and a realisation carries the fold-to-fold variance a real coin flip
    would — which is the point of putting it in a paired test at all. Its
    expectation is `ensemble_stats(...)["random_wer"]`; the two agree to within
    that variance, and the paired test needs the comparator, not the mean.

    `members` adds one constant arm PER MEMBER, keyed by the member's own name —
    the "individual models" rows of the results table, scored on exactly the
    folds every learned arm is scored on. `tolerances` adds `oracle_tol{n}` for
    each `n > 0`: the oracle that only switches for more than `n` words, which
    is the reference the `gap_closed_tol{n}` family measures against. Both are
    off by default so the classic four-plus-one set is unchanged.
    """
    arms = {
        "oracle": E.argmin(axis=1),
        # The reachable target. It sits between `bsm` and `oracle` by
        # construction, and the distance from `bsm` to HERE is the headroom
        # worth quoting — see `ACHIEVABLE_MARGIN`.
        "achievable_oracle": achievable_choice(E, bsm),
        "bsm": np.full(len(E), bsm),
        "worst": E.argmax(axis=1),
    }
    if rng is not None:
        arms["random"] = rng.integers(0, E.shape[1], len(E))
    for n in tolerances:
        if n > 0:
            arms[f"oracle_tol{n}"] = achievable_choice(E, bsm, margin=n)
    if members is not None:
        for i, m in enumerate(members):
            if m in arms:
                raise KeyError(f"member {m!r} collides with a reference arm name")
            arms[m] = np.full(len(E), i)
    return arms


def fold_rng(random_seed, repeat_key, fold):
    """The `random` arm's generator, keyed on the repetition's **seed**.

    Not on its position. Two runs that differ only in their fold seeds should get
    different coin flips — otherwise the random arm is frozen while everything it
    is being compared against moves, which is the one thing a reference arm must
    not do. Keying on the seed also makes `run_cv` and `NestedCV` agree on the
    same repetition, so a nested run and a flat one remain comparable.
    """
    digest = hashlib.sha1(f"{random_seed}|{repeat_key}|{fold}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def wants_rows(model):
    """True if this estimator asks for store row ids alongside `X` (§9 fusion)."""
    return bool(getattr(model, "wants_rows", False))


def fit_arm(model, X_tr, Y_tr, nref_tr, rows_tr, X_ev, rows_ev, timings=None):
    """Fit one estimator on a fold and read its evaluation-half output.

    Returns `(kind, out)`. `kind` is `"choice"` for a router — `out` is member
    indices — or `"errors"` for a fusion arm, where `out` is the error count of
    the transcript it produced and no member was selected at all.

    An estimator opts into `rows=` by setting `wants_rows = True`. That is how a
    fusion arm reaches the stored transcripts, which are indexed by store row and
    not derivable from the standardised design matrix it is handed.

    `timings`, when a dict is passed, receives `fit_seconds` and
    `predict_seconds` **separately**. They are not interchangeable costs: fit is
    paid once per deployment and inference is paid per utterance forever, so a
    selector that trains for an hour and routes in a millisecond is a very
    different proposition from the reverse. Filling a caller's dict keeps the
    return signature at two values, which `NestedCV` (§7.8) relies on.
    """
    kw = {"nref": nref_tr}
    if wants_rows(model):
        kw["rows"] = rows_tr

    t0 = time.perf_counter()
    model.fit(X_tr, Y_tr, **kw)
    t1 = time.perf_counter()
    out = model.predict(X_ev, rows=rows_ev) if wants_rows(model) else model.predict(X_ev)
    t2 = time.perf_counter()
    if timings is not None:
        timings["fit_seconds"] = t1 - t0
        # Per-utterance, so an arm evaluated on a half-fold is comparable with
        # one evaluated on a whole corpus. The total is recoverable as
        # `predict_us_per_utt * n_eval`.
        timings["predict_seconds"] = t2 - t1
        timings["predict_us_per_utt"] = (t2 - t1) * 1e6 / max(len(X_ev), 1)
    return getattr(model, "produces", "choice"), np.asarray(out)


def score_arm(E_ev, nref_ev, kind, out, bsm):
    """`(picked, choice)` for one arm's output — `choice` is None for a fusion arm."""
    if kind == "errors":
        return np.asarray(out, dtype=np.float64), None
    choice = np.asarray(out, dtype=int)
    return picked_errors(E_ev, choice), choice

def run_cv(store, members, estimators, scheme=None, use_extra_features=True,
           use_duration=True, references=True, rows=None, verbose=True,
           random_seed=20260821, transform=None, fold_eda=False,
           member_arms=False, tolerances=(), fold_table=False):
    """Fit and score every estimator on every fold. Returns `(fold_scores, fold_picks)`.

    Parameters
    ----------
    store : a loaded `EnsembleStore`.
    members : the ensemble group to route between, e.g.
        `("espnet_transducer", "espnet_transformer", "data2vec")`.
    estimators : `{name: callable}` — **factories**, not instances, so every fold
        gets an object with no state carried over from the last one.
    scheme : a `BaseSplitScheme`. Defaults to `FiveByTwoSplit()`.
    references : include the parameter-free `oracle` / `bsm` / `worst` / `random`
        arms. `random` is seeded from `random_seed` and the fold, so it is
        reproducible but genuinely varies across folds.
    rows : restrict cross-validation to these store rows. This is how a tuning
        holdout is honoured — tune on one partition, pass the complement here, and
        no utterance the tuner saw is ever scored. `fold_picks.row` still refers to
        the **store's** row numbering, not a position within the subset, so the
        cache joins back to `store.err(members)` unchanged.
    transform : optional `X -> X` fitted per fold on the training half. This is
        where a PCA control goes (§7.2.6): it must be refitted inside the fold
        like the scaler, or it leaks.
    fold_eda : print `fold_brief` — the split sizes and the base-model table for
        THIS fold — before each fold is fitted (§7.0). Off by default because a
        10-fold run prints ten of them; on, it is the record of what each fold
        actually contained.
    member_arms : add one constant reference arm per member, named after it —
        the individual-model rows of the results table. See `reference_choices`.
    tolerances : add `oracle_tol{n}` reference arms for each `n > 0` given —
        the oracle that switches only for a gain above `n` words.
    fold_table : after every fold, print the `TABLE_METRICS` of every arm as one
        table (plus its selection distribution) instead of the one-line WER
        summary. The same metric set the results table reports, so what is
        printed while the test runs is directly comparable to the final table.
    """
    scheme = scheme or FiveByTwoSplit()
    X_all, _ = build_design(store, members, use_extra_features, use_duration)
    E_all = store.err(members)
    nref_all = store.nref

    row_ids = np.arange(len(X_all)) if rows is None else np.asarray(rows, dtype=int)
    if rows is not None:
        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError("`rows` contains duplicates")
        X_all, E_all, nref_all = X_all[row_ids], E_all[row_ids], nref_all[row_ids]
    Y_all = wer_matrix(E_all, nref_all)
    tag = "+".join(members)

    n_ref_arms = (5 + sum(1 for n in tolerances if n > 0)
                  + (len(members) if member_arms else 0)) if references else 0
    if verbose:
        scope = "pooled" if rows is None else f"of {len(store.nref):,} pooled"
        print(f"{tag}\n  {len(X_all):,} {scope} utterances x {X_all.shape[1]:,} features, "
              f"{scheme.n_folds} folds ({scheme.name}), "
              f"{len(estimators)} estimator(s)"
              f"{f' + {n_ref_arms} references' if references else ''}")

    scores, picks = [], []
    for repeat, fold, tr, ev in scheme.split(len(X_all)):
        seed = (scheme.repeat_seed(repeat) if hasattr(scheme, "repeat_seed")
                else repeat)
        # Stamped on every row, so an analysis can address a repetition by the
        # only thing that reproduces it. `repeat` is kept beside it because
        # `fold_differences` pivots on (repeat, fold) and a tuple-valued seed —
        # the legacy `(seed, i)` form — is not a usable pivot key.
        if fold_eda:
            fold_brief(E_all, nref_all, members, tr, ev, seed=seed, fold=fold)
        rows_out, picks_out = fold_arms(
            estimators, X_all, Y_all, E_all, nref_all, row_ids, tr, ev,
            members=members, tag=tag, repeat=repeat, fold=fold,
            references=references, transform=transform,
            rng=fold_rng(random_seed, seed, fold),
            verbose=verbose, extra={"repeat_seed": _seed_label(seed)},
            member_arms=member_arms, tolerances=tolerances, fold_table=fold_table)
        scores += rows_out
        picks += picks_out

    return pd.DataFrame(scores), pd.concat(picks, ignore_index=True)


def _seed_label(seed):
    """A repetition's seed as a scalar where it is one, else a printable string.

    `FiveByTwoSplit(seeds=...)` gives an int per repetition, which is what
    §7.4.1 wants to group and select on. The legacy `(seed, i)` form gives a tuple, and
    a tuple in a DataFrame column is a hashable object that groups correctly but
    prints badly — so it is rendered once, here, rather than at every use site.
    """
    return seed if isinstance(seed, (int, np.integer)) else str(seed)


def fold_arms(estimators, X_all, Y_all, E_all, nref_all, row_ids, tr, ev,
              members, tag, repeat, fold, references=True, transform=None,
              rng=None, verbose=True, extra=None, member_arms=False,
              tolerances=(), fold_table=False):
    """Fit and score every arm on ONE fold. Returns `(score_rows, pick_frames)`.

    Factored out of `run_cv` because §7.8 needs exactly this and must do it
    identically — the standardisation, the `bsm` rule and the metric set are the
    parts that have to agree between a flat run and a nested one, or the two are
    not comparable and the nesting proves nothing.

    `extra` is merged into every score row; §7.8 uses it to stamp the repetition
    seed on results that `run_cv` keys by position.
    """
    assert not np.intersect1d(tr, ev).size, "train and eval folds overlap"

    # Scaler from the training half only — see the note above on leakage.
    mu, sd = X_all[tr].mean(0), X_all[tr].std(0).clip(1e-8)
    X_tr, X_ev = (X_all[tr] - mu) / sd, (X_all[ev] - mu) / sd
    if transform is not None:
        # Fitted inside the fold for the same reason the scaler is. A PCA basis
        # estimated on the evaluation half is leakage that looks like a result.
        fitted = transform().fit(X_tr)
        X_tr, X_ev = fitted.transform(X_tr), fitted.transform(X_ev)
    E_ev, nref_ev = E_all[ev], nref_all[ev]
    # The best single member, chosen on the training half by corpus WER.
    bsm = int((E_all[tr].sum(0) / nref_all[tr].sum()).argmin())

    arms = {}
    if references:
        # A reference arm is read straight off the error matrix: nothing is
        # fitted and nothing is inferred, so its costs are exactly zero rather
        # than merely small.
        free = {"fit_seconds": 0.0, "predict_seconds": 0.0, "predict_us_per_utt": 0.0}
        refs = reference_choices(E_ev, bsm, rng,
                                 members=members if member_arms else None,
                                 tolerances=tolerances)
        for name, choice in refs.items():
            arms[name] = (picked_errors(E_ev, choice), choice, dict(free), np.nan)

    for name, factory in estimators.items():
        timings = {}
        t0 = time.perf_counter()
        model = factory()
        kind, out = fit_arm(model, X_tr, Y_all[tr], nref_all[tr], row_ids[tr],
                            X_ev, row_ids[ev], timings=timings)
        # Construction is charged to `fit`: a factory that builds a 60M-parameter
        # network is doing real work, and hiding it flatters the arm.
        timings["fit_seconds"] = time.perf_counter() - t0 - timings["predict_seconds"]
        picked, choice = score_arm(E_ev, nref_ev, kind, out, bsm)
        # The soft objective the model actually descended, next to the hard WER
        # its argmax realises. The gap between the two is worth reporting; it is only
        # defined for estimators that carry a routing distribution.
        soft = (float(model.expected_wer(X_ev, Y_all[ev]))
                if hasattr(model, "expected_wer") else np.nan)
        arms[name] = (picked, choice, timings, soft)

    score_rows, pick_frames = [], []
    for name, (picked, choice, timings, soft) in arms.items():
        base = {"members": tag, "model": name, "repeat": repeat, "fold": fold,
                "n_train": len(tr), "n_eval": len(ev), "bsm": members[bsm],
                **timings, **(extra or {})}
        for metric, (fn, direction) in METRICS.items():
            score_rows.append({**base, "metric": metric,
                               "value": fn(E_ev, nref_ev, picked, bsm),
                               "direction": direction})
        if np.isfinite(soft):
            score_rows.append({**base, "metric": "soft_expected_wer",
                               "value": soft, "direction": "lower"})
        pick_frames.append(pd.DataFrame({
            "members": tag, "model": name, "repeat": repeat, "fold": fold,
            "row": row_ids[ev], "errors": picked,
            "choice": -1 if choice is None else choice, **(extra or {})}))

    if verbose:
        # Name the repetition by its SEED when there is one. A repetition is
        # reproducible from its seed and from nothing else, so `r3` names a
        # position in a list that a different `seeds=` tuple would renumber,
        # while `seed=44` names the thing itself — and that is what §7.4.1
        # selects over, logs, and puts in the paper.
        seed = (extra or {}).get("repeat_seed")
        label = f"r{repeat}" if seed is None else f"seed={seed}"
        if fold_table:
            # One table per fold, in the results table's own units and colours;
            # the oracle rows do not compete for "best", being 100% by construction.
            show_table(fold_report(score_rows, pick_frames, members),
                       title=f"[{label} f{fold}]  train {len(tr):,} / eval {len(ev):,}  "
                             f"bsm={members[bsm]}",
                       best="auto", best_exclude=lambda r: "oracle" in str(r["arm"]),
                       highlight=lambda r: str(r["arm"]).startswith("fastt"),
                       mono=("dist",))
        else:
            body = "  ".join(f"{n} {v[0].sum() / nref_ev.sum():.4f}"
                             for n, v in arms.items())
            print(f"  {label:<12} f{fold}  bsm={members[bsm]:<20} {body}")
    return score_rows, pick_frames


def fold_report(score_rows, pick_frames, members, digits=4):
    """One fold's `TABLE_METRICS` per arm, plus its selection distribution.

    What `run_cv(fold_table=True)` prints after each fold: the results table's
    own metric set, read off the very rows just scored, so a reader can follow
    the test as it runs in the units the final table will use.
    """
    df = pd.DataFrame(score_rows)
    wide = (df[df["metric"].isin(TABLE_METRICS)]
            .pivot_table(index="model", columns="metric", values="value", sort=False)
            .reindex(columns=list(TABLE_METRICS)))
    dist = {p["model"].iloc[0]: format_shares(selection_shares(p["choice"], len(members)))
            for p in pick_frames}
    wide.insert(0, "arm", wide.index)
    wide["dist"] = [dist.get(m, "—") for m in wide.index]
    # Gap and accuracy as percentages, WER as a decimal — `format_results`' units.
    for m in TABLE_METRICS:
        if m in wide.columns and m.startswith(("gap_closed", "acc_")):
            wide[m] = (100 * wide[m]).round(1)
        elif m in wide.columns:
            wide[m] = wide[m].round(digits)
    return wide.rename(columns=TABLE_LABELS).reset_index(drop=True)

def cost_table(fold_scores, results=None, per=1000):
    """Wall-clock per arm: search, training, inference. One row per model.

    Parameters
    ----------
    fold_scores : the frame `run_cv` returns. Timings are stamped on every score
        row of a fold, so they are de-duplicated to `(model, repeat, fold)`
        before anything is averaged — reading them straight off `fold_scores`
        would weight each fold by the number of metrics, which is a constant and
        therefore harmless for a mean and wrong for a sum.
    results : optional `{arm: TuningResult}` (i.e. `HPTRun.results`), which adds
        the search cost — the part paid once, before any of this ran, and the
        part a comparison of "our method vs a baseline" most often omits.
    per : inference is quoted per this many utterances as well as per utterance,
        because microseconds per utterance is the honest unit and nobody has an
        intuition for it.

    The three clocks are kept apart rather than summed. Search is paid once ever,
    training once per deployment, inference on every utterance forever, and an
    arm that is cheap on one can be ruinous on another.
    """
    cols = ["fit_seconds", "predict_seconds", "predict_us_per_utt"]
    have = [c for c in cols if c in fold_scores.columns]
    if not have:
        raise ValueError("no timing columns in fold_scores; it did not come "
                         f"from run_cv (looked for {cols})")
    folds = fold_scores.drop_duplicates(subset=["model", "repeat", "fold"])

    g = folds.groupby("model")
    out = pd.DataFrame({
        "n_folds": g.size(),
        "fit_s_mean": g["fit_seconds"].mean(),
        "fit_s_sd": g["fit_seconds"].std(),
        "fit_s_total": g["fit_seconds"].sum(),
    })
    if "predict_us_per_utt" in have:
        out["infer_us_per_utt"] = g["predict_us_per_utt"].mean()
        out[f"infer_s_per_{per}"] = out["infer_us_per_utt"] * per / 1e6
    if "predict_seconds" in have:
        out["predict_s_total"] = g["predict_seconds"].sum()
    out["n_eval"] = g["n_eval"].mean()

    if results is not None:
        search = pd.DataFrame([
            {"model": name, "search_s": float(getattr(r, "seconds", np.nan)),
             "trials": int(getattr(r, "n_trials", 0))}
            for name, r in results.items()]).set_index("model")
        search["search_s_per_trial"] = (search["search_s"]
                                        / search["trials"].replace(0, np.nan))
        out = out.join(search)
        # A reference arm has no search and no fit; NaN would read as "missing",
        # which is a different claim from "free".
        for c in ("search_s", "trials", "search_s_per_trial"):
            out[c] = out[c].fillna(0.0)

    return out.reset_index().sort_values("fit_s_total", ignore_index=True)


def selection_distribution(fold_picks, members, fold_scores=None):
    """How often each arm chose each member. One row per model.

    Returns a `share_<member>` column per member, plus:

    * `entropy` — Shannon entropy of the share vector in bits, and `entropy_max`
      beside it. An arm at zero entropy is a constant, whatever its WER says.
    * `share_top` / `top_member` — the most-used member and how dominant it is.
    * `share_sd_mean` — the mean, over members, of the standard deviation of
      that member's share **across folds**. A router with a stable policy and one
      that flips between folds can have identical mean shares; this separates
      them.
    * `switch_rate` — the share of utterances routed to something other than the
      fold's best single model, which is the number that says whether the arm is
      doing anything at all.

    Fusion arms are recorded by `run_cv` with `choice == -1`, because they build
    a transcript instead of selecting a member. They keep their row — with
    `n_fused` set and the share columns `NaN` — rather than being dropped: a
    table that silently omits the fusion baselines is the table this function
    exists to replace.
    """
    members = list(members)
    picks = fold_picks.copy()
    if "choice" not in picks.columns:
        raise ValueError("fold_picks has no `choice` column")

    bsm = None
    if fold_scores is not None and "bsm" in fold_scores.columns:
        # The fold's own best single model, which is chosen on the training half
        # and can differ from fold to fold — so `switch_rate` is measured against
        # what the arm was actually competing with on THAT fold.
        bsm = (fold_scores.drop_duplicates(subset=["model", "repeat", "fold"])
               [["model", "repeat", "fold", "bsm"]])
        picks = picks.merge(bsm, on=["model", "repeat", "fold"], how="left")
        picks["bsm_idx"] = picks["bsm"].map({m: i for i, m in enumerate(members)})

    rows = []
    for name, grp in picks.groupby("model", sort=False):
        n = len(grp)
        fused = int((grp["choice"] < 0).sum())
        row = {"model": name, "n_picks": n, "n_fused": fused}
        routed = grp[grp["choice"] >= 0]

        if len(routed) == 0:
            row.update({f"share_{m}": np.nan for m in members})
            row.update(entropy=np.nan, entropy_max=np.log2(len(members)),
                       share_top=np.nan, top_member="— fused —",
                       share_sd_mean=np.nan, switch_rate=np.nan)
            rows.append(row)
            continue

        counts = routed["choice"].value_counts(normalize=True)
        shares = np.array([counts.get(i, 0.0) for i in range(len(members))])
        for m, s in zip(members, shares):
            row[f"share_{m}"] = float(s)

        nz = shares[shares > 0]
        # `+ 0.0` normalises the sign: a single-member policy gives
        # -(1 * log2 1) = -0.0, which prints as "-0.000" and reads as a bug.
        row["entropy"] = float(-(nz * np.log2(nz)).sum()) + 0.0
        row["entropy_max"] = float(np.log2(len(members)))
        top = int(shares.argmax())
        row["share_top"], row["top_member"] = float(shares[top]), members[top]

        # Per fold, then the spread of each member's share across folds.
        per_fold = (routed.groupby(["repeat", "fold"])["choice"]
                    .value_counts(normalize=True).unstack(fill_value=0.0)
                    .reindex(columns=range(len(members)), fill_value=0.0))
        row["share_sd_mean"] = float(per_fold.std(axis=0).mean())

        row["switch_rate"] = (float((routed["choice"] != routed["bsm_idx"]).mean())
                              if "bsm_idx" in routed.columns else np.nan)
        rows.append(row)

    cols = (["model", "n_picks", "n_fused"] + [f"share_{m}" for m in members]
            + ["top_member", "share_top", "entropy", "entropy_max",
               "share_sd_mean", "switch_rate"])
    return pd.DataFrame(rows)[cols].sort_values("model", ignore_index=True)


def results_table(fold_scores, fold_picks, members, order=None, labels=None,
                  metrics=TABLE_METRICS):
    """The main-results table (Table 3), one row per arm, numeric.

    For every metric in `metrics` two columns: `<metric>` (mean over folds) and
    `<metric>_sd` (standard deviation over folds, `ddof=1`). Then one
    `share_<member>` per member — the arm's selection distribution over every
    evaluation utterance of every fold — and `n_fused`, the number of utterances
    on which the arm fused a transcript instead of choosing a member.

    **Undefined cells are NaN, not zero.** A fusion arm has no selection to be
    accurate about, so its `acc_tol*` and `share_*` are NaN whenever `n_fused`
    is positive; `format_results` renders those as `—`. Everything is kept
    numeric here so the frame can be joined, sorted and tested; the manuscript
    strings are produced by `format_results`.

    `order` fixes the row order (arms absent from `fold_scores` are skipped, and
    arms present but not in `order` are appended, so nothing is silently lost);
    `labels` maps registry keys to display names, applied to the `label` column
    only — `model` keeps the key.
    """
    members = list(members)
    have = list(dict.fromkeys(fold_scores["model"]))
    order = ([m for m in (order or []) if m in have]
             + [m for m in have if m not in (order or [])])

    df = fold_scores[fold_scores["metric"].isin(metrics)]
    mean = df.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
    sd = df.pivot_table(index="model", columns="metric", values="value", aggfunc="std")

    rows = []
    for m in order:
        row = {"model": m, "label": (labels or {}).get(m, m)}
        for metric in metrics:
            row[metric] = float(mean.loc[m, metric]) if metric in mean.columns else np.nan
            row[f"{metric}_sd"] = float(sd.loc[m, metric]) if metric in sd.columns else np.nan
        picks = fold_picks[fold_picks["model"] == m]
        fused = int((picks["choice"] < 0).sum()) if len(picks) else 0
        shares = selection_shares(picks["choice"], len(members)) if len(picks) else \
            np.full(len(members), np.nan)
        for name, s in zip(members, shares):
            row[f"share_{name}"] = float(s)
        row["n_fused"] = fused
        if fused > 0:
            # No member was selected on these rows, so "selection accuracy" and
            # the distribution do not exist for the arm — not merely unmeasured.
            for metric in metrics:
                if metric.startswith("acc_"):
                    row[metric] = row[f"{metric}_sd"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["members"] = members
    out.attrs["n_folds"] = int(fold_scores.drop_duplicates(["repeat", "fold"]).shape[0])
    return out


def format_results(table, metrics=TABLE_METRICS, digits=4, sd=True, pct=None):
    """`results_table` as manuscript strings: `mean ± sd`, `—` where undefined.

    `pct` names the metrics rendered as percentages (default: every `gap_closed*`
    and `acc_*` column), with the WERs left as plain decimals.
    """
    if pct is None:
        pct = tuple(m for m in metrics if m.startswith(("gap_closed", "acc_")))
    members = table.attrs.get("members") or [
        c[len("share_"):] for c in table.columns if c.startswith("share_")]

    def cell(v, s, as_pct):
        if not np.isfinite(v):
            return "—"
        if as_pct:
            main = f"{100 * v:.1f}"
            return f"{main} ± {100 * s:.1f}" if sd and np.isfinite(s) else main
        main = f"{v:.{digits}f}"
        return f"{main} ± {s:.{digits}f}" if sd and np.isfinite(s) else main

    out = pd.DataFrame({"arm": table["label"] if "label" in table.columns else table["model"]})
    for metric in metrics:
        out[TABLE_LABELS.get(metric, metric)] = [
            cell(v, s, metric in pct)
            for v, s in zip(table[metric], table[f"{metric}_sd"])]
    out["dist"] = [format_shares([r[f"share_{m}"] for m in members])
                   for _, r in table.iterrows()]
    return out


def to_latex_rows(formatted, sep=" & ", end=r" \\"):
    """`format_results` output as LaTeX table rows, one per line, no preamble.

    Deliberately no `tabular` environment: the manuscript's tables merge several
    corpora into one, and the merge is done by hand — this only spares the
    retyping of the numbers.
    """
    lines = []
    for _, r in formatted.iterrows():
        cells = [str(v).replace("±", r"$\pm$").replace("—", "---") for v in r.values]
        lines.append(sep.join(cells) + end)
    return "\n".join(lines)

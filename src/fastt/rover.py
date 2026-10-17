__all__ = ['WEIGHT_SCHEMES', 'FUSION_RIVALS', 'word_counters', 'build_network', 'vote', 'system_weights', 'RoverFusion', 'rover_arm', 'check_counters', 'rover_grid', 'edit_distance', 'agreement', 'FusionCache', 'fiscus_vote', 'ConfWeightedRover', 'CNMBRFusion', 'ScoreFusion', 'conf_rover_space', 'cn_mbr_space', 'score_fusion_space', 'suggest_conf_rover', 'suggest_cn_mbr', 'suggest_score_fusion', 'build_conf_rover', 'build_cn_mbr', 'build_score_fusion', 'rover_w_space']

import hashlib

import numpy as np
import pandas as pd

WEIGHT_SCHEMES = ("uniform", "inverse_wer", "log_odds")


def word_counters(ref, hyp):
    """`(sub, dele, ins, nref)` for one already-normalised pair, via `jiwer`.

    Deliberately not `WerScorer`: the strings on the Hub have already been through
    the Whisper normaliser (that is what `transcription_norm` and `text_norm`
    are), so re-normalising would be wasted work and, worse, would let this
    section disagree with the stored counters. Scoring the same strings the same
    way is what makes a fused transcript comparable to a routed one.
    """
    import jiwer

    if not ref.strip():
        return 0, 0, len(hyp.split()), 0
    w = jiwer.process_words(ref, hyp)
    return w.substitutions, w.deletions, w.insertions, (
        w.hits + w.substitutions + w.deletions)


def _levenshtein_path(a, b):
    """Backtrace of the word-level edit distance between two token lists.

    Returns pairs `(i, j)` with `None` for an unmatched side: `(i, None)` is a
    deletion from `a`, `(None, j)` an insertion from `b`. Substitutions and
    matches come back as `(i, j)`.
    """
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)

    path, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            path.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            path.append((i - 1, None)); i -= 1
        else:
            path.append((None, j - 1)); j -= 1
    return path[::-1]


def build_network(hyps):
    """Iterative ROVER alignment of `k` hypotheses into a word transition network.

    Returns a list of slots, each a length-`k` list of tokens or `None`. Slot `s`
    holds what each system contributed at that position; `None` means the system
    proposed nothing there, which is a vote for deletion and not an absence of a
    vote.

    The first hypothesis seeds the network and each subsequent one is aligned into
    it. That makes the result depend on the order of `hyps` — a known property of
    ROVER, not of this implementation. `RoverFusion` puts the best system first,
    which is the usual convention and the one that makes the seed network the
    most reliable.
    """
    k = len(hyps)
    tokens = [h.split() for h in hyps]
    slots = [[w] + [None] * (k - 1) for w in tokens[0]]

    for s in range(1, k):
        # Each slot is represented by the majority word among the systems that
        # have already filled it: that is the string the next hypothesis has to
        # align against.
        spine = [_majority([t for t in slot if t is not None]) for slot in slots]
        merged, si = [], 0
        for slot_i, tok_j in _levenshtein_path(spine, tokens[s]):
            if slot_i is None:                             # insertion: a new slot
                new = [None] * k
                new[s] = tokens[s][tok_j]
                merged.append(new)
            else:
                slot = slots[slot_i]
                slot[s] = None if tok_j is None else tokens[s][tok_j]
                merged.append(slot)
        slots = merged
    return slots


def _majority(words):
    """The most common string, ties broken by first appearance. `''` if empty."""
    if not words:
        return ""
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(words, key=lambda w: (counts[w], -words.index(w)))


def vote(slots, weights, null_penalty=1.0):
    """Vote one network into a transcript.

    `null_penalty` scales the weight a system's `None` contributes. At 1.0 a
    silent system argues for deletion exactly as strongly as a speaking system
    argues for its word; below 1.0 the fusion becomes reluctant to delete, which
    is usually right when the members have different insertion/deletion balances.
    It is the one parameter worth tuning, and `NestedCV` can tune it.
    """
    words = []
    for slot in slots:
        scores = {}
        for w, weight in zip(slot, weights):
            key = w if w is not None else None
            scores[key] = scores.get(key, 0.0) + (weight * (null_penalty
                                                            if w is None else 1.0))
        best = max(scores, key=lambda key: (scores[key], key is not None))
        if best is not None:
            words.append(best)
    return " ".join(words)


def system_weights(scheme, wer, floor=1e-3):
    """Per-system voting weights from their WER on the training half.

    `"uniform"` is plain majority voting, kept so the difference weighting
    makes can be shown rather than asserted. `"inverse_wer"` is the usual
    weighted ROVER. `"log_odds"` weights by `log((1 - wer) / wer)`, i.e. by the
    log-odds of being right, which is the weighting a naive-Bayes combination of
    independent voters implies — the sharper of the two, and the better choice
    when the members are far apart in quality.
    """
    wer = np.clip(np.asarray(wer, dtype=np.float64), floor, 1 - floor)
    if scheme == "uniform":
        w = np.ones_like(wer)
    elif scheme == "inverse_wer":
        w = 1.0 / wer
    elif scheme == "log_odds":
        w = np.log((1.0 - wer) / wer)
    else:
        raise ValueError(f"weight scheme must be one of {WEIGHT_SCHEMES}, got {scheme!r}")
    w = np.clip(w, 1e-6, None)
    return w / w.sum()

class RoverFusion:
    """Weighted ROVER as a §7.1 arm: fits weights on a fold, emits error counts.

    Follows the estimator contract with the two opt-ins §7.3 defines for fusion:

    * `wants_rows = True` — it works on stored transcripts, which are addressed
      by store row and cannot be recovered from the standardised design matrix.
    * `produces = "errors"` — it returns the error count of the transcript it
      built, because no member was selected. `run_cv` scores that through the
      same metrics as any router.

    Parameters
    ----------
    texts : the `(N, k+1)` frame from `store.load_texts()` — one column per
        member plus `reference` — indexed like the store.
    members : which columns to fuse, in any order; the best system on the
        training half is moved to the front before alignment.
    weighting : one of `WEIGHT_SCHEMES`. Fitted on the training half only.
    null_penalty : see `vote`.
    cache : shared `{row -> network}` dict. Alignment is the expensive part and
        it does not depend on the fold, so passing one cache across all folds of
        a run turns nine of ten folds into pure voting. `RoverFusion.shared_cache()`
        makes one.
    """

    wants_rows = True
    produces = "errors"

    def __init__(self, texts, members, weighting="inverse_wer", null_penalty=1.0,
                 cache=None):
        self.texts, self.members = texts, tuple(members)
        self.weighting, self.null_penalty = weighting, float(null_penalty)
        self.cache = {} if cache is None else cache
        self.weights_ = self.order_ = None

    @staticmethod
    def shared_cache():
        """A network cache to hand to every fold's instance of this arm."""
        return {}

    def fit(self, X, y, nref=None, rows=None):
        """Weights from the members' WER on these rows. No transcripts are touched."""
        if rows is None:
            raise ValueError("RoverFusion needs `rows`; run_cv passes them when "
                             "`wants_rows` is set")
        E = np.asarray(y, dtype=np.float64)          # per-utterance WER matrix
        n = np.ones(len(E)) if nref is None else np.asarray(nref, dtype=np.float64)
        # Corpus WER per member on the training half: sum(errors) / sum(words),
        # reconstructed from the WER matrix the contract hands over.
        wer = (E * n[:, None]).sum(0) / n.sum()
        self.weights_ = system_weights(self.weighting, wer)
        # Best system first: it seeds the network, and seeding from the weakest
        # hypothesis is a measurable part of why unweighted ROVER underperforms.
        self.order_ = np.argsort(wer)
        return self

    def _network(self, row):
        key = int(row)
        if key not in self.cache:
            hyps = [str(self.texts.iloc[key][m]) for m in self.members]
            self.cache[key] = build_network(hyps)
        return self.cache[key]

    def transcribe(self, rows):
        """The fused transcript for each store row, in order."""
        weights = self.weights_[self.order_]
        out = []
        for row in np.asarray(rows, dtype=int):
            slots = self._network(row)
            # Reorder each slot to match the weight vector rather than rebuilding
            # the network: the alignment is order-dependent but the *vote* only
            # needs system and weight to line up.
            ordered = [[slot[i] for i in self.order_] for slot in slots]
            out.append(vote(ordered, weights, self.null_penalty))
        return out

    def predict(self, X, rows=None):
        """(n,) error counts of the fused transcript against the reference."""
        if rows is None:
            raise ValueError("RoverFusion needs `rows` at predict time too")
        rows = np.asarray(rows, dtype=int)
        hyps = self.transcribe(rows)
        refs = [str(self.texts.iloc[r]["reference"]) for r in rows]
        return np.array([sum(word_counters(ref, hyp)[:3])
                         for ref, hyp in zip(refs, hyps)], dtype=np.float64)


def rover_arm(store, members, weighting="inverse_wer", null_penalty=1.0, cache=None):
    """A `run_cv`-ready factory for `RoverFusion`, sharing one alignment cache.

    ```python
    store.load_texts(MEMBERS)
    arms["rover_w"] = rover_arm(store, MEMBERS, "inverse_wer")
    arms["rover_u"] = rover_arm(store, MEMBERS, "uniform")   # the straw man
    ```
    """
    if store.texts is None:
        raise ValueError("call store.load_texts(members) first — fusion needs transcripts")
    shared = RoverFusion.shared_cache() if cache is None else cache
    return lambda: RoverFusion(store.texts, members, weighting, null_penalty, shared)


def check_counters(store, members, n=200, seed=0):
    """Sanity check: recomputed WER counters must match the ones on the Hub.

    Fusion is only comparable to selection if both are scored the same way. This
    re-scores `n` random utterances from the stored strings and compares against
    the stored `sub + dele + ins`. A mismatch means the normalisation used at
    extraction time and the one used here have drifted, and every fusion number
    below would be measured against a different ruler than every routing number.
    """
    if store.texts is None:
        store.load_texts(members)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(store.nref), size=min(n, len(store.nref)), replace=False)
    rows = []
    for row in picks:
        ref = str(store.texts.iloc[row]["reference"])
        for m in members:
            s, d, i, _ = word_counters(ref, str(store.texts.iloc[row][m]))
            rows.append({"row": int(row), "model": m, "recomputed": s + d + i,
                         "stored": float(store.errors.iloc[row][m])})
    out = pd.DataFrame(rows)
    out["match"] = out["recomputed"] == out["stored"]
    rate = out["match"].mean()
    print(f"{rate:.1%} of {len(out):,} (utterance, model) counters reproduce exactly")
    if rate < 1.0:
        bad = out[~out["match"]]
        print(f"  {len(bad)} mismatch(es); worst delta "
              f"{(bad['recomputed'] - bad['stored']).abs().max():.0f} words")
    return out

def rover_grid(store, members, fit_rows, eval_rows, weightings=None,
               penalties=(0.5, 1.0, 2.0), cache=None, verbose=True):
    """Score every (weighting, null_penalty) pair on a hold-out. Returns `(factory, table)`.

    Parameters
    ----------
    fit_rows, eval_rows : store row indices. Weights are fitted on the first and
        the fused transcript scored on the second — pass a tuning partition's
        `fit` and `val` (`HPTRun.tuner.rows`), never the evaluation rows.
    weightings : defaults to every scheme in `WEIGHT_SCHEMES`.
    penalties : the `null_penalty` grid. ROVER's only continuous knob: how much
        a system that aligned nothing at a position counts against the word
        that is there.

    The returned factory is `rover_arm`-shaped and ready for `run_cv`, with the
    alignment cache shared across every configuration — the alignment is the
    expensive part and it does not depend on the weights, so the whole grid
    costs barely more than one fit.
    """
    if store.texts is None:
        store.load_texts(members)
    weightings = tuple(weightings or WEIGHT_SCHEMES)
    shared = RoverFusion.shared_cache() if cache is None else cache
    fit_rows = np.asarray(fit_rows, dtype=int)
    eval_rows = np.asarray(eval_rows, dtype=int)
    words = store.nref[eval_rows].sum()

    rows = []
    for w in weightings:
        for pen in penalties:
            model = RoverFusion(store.texts, members, w, float(pen), shared)
            model.fit(None, store.err(members)[fit_rows], nref=store.nref[fit_rows],
                      rows=fit_rows)
            errors = model.predict(None, rows=eval_rows)
            rows.append({"weighting": w, "null_penalty": float(pen),
                         "corpus_wer": float(errors.sum() / words),
                         "mean_utt_wer": float(
                             (errors / np.maximum(store.nref[eval_rows], 1)).mean())})
    table = pd.DataFrame(rows).sort_values("corpus_wer", ignore_index=True)
    best = table.iloc[0]
    if verbose:
        print(f"ROVER grid on {len(eval_rows):,} hold-out rows "
              f"({len(weightings)} scheme(s) x {len(penalties)} penalty):")
        print(table.round(4).to_string(index=False))
        print(f"  -> {best['weighting']}, null_penalty={best['null_penalty']:g}")
    factory = rover_arm(store, members, best["weighting"],
                        float(best["null_penalty"]), shared)
    return factory, table

from dataclasses import replace

from fastt.probing import RidgeProbe
from fastt.tuning import MODELS, ModelSpec, Param


def edit_distance(a, b):
    """Word-level Levenshtein cost between two token lists. No backtrace.

    `_levenshtein_path` computes the same number on its way to an alignment, but
    the agreement features below need the cost for every ORDERED PAIR of
    hypotheses and never need the path, and keeping the O(nm) table rather than
    the O(nm) table plus a backtrace is the difference that shows up when it is
    called `k^2` times per utterance over 31k utterances.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return max(n, m)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def agreement(hyps):
    """`(k,)` mean agreement of each hypothesis with the others, in `[0, 1]`.

    `1 - d(i, j) / max(|i|, |j|)`, averaged over `j != i`. This is the `cn`
    confidence source: a system that says what everyone else says on this
    utterance is probably right on it, and a system that has wandered off is
    probably not. It is the only per-utterance confidence available with no
    learned component and no decoder scores, which is what makes it worth having
    beside the probe rather than instead of it.

    Degenerate cases are pinned rather than left to divide by zero: a single
    system agrees with nothing and scores 1.0 (there is no evidence against it),
    and two empty hypotheses agree completely.
    """
    k = len(hyps)
    if k < 2:
        return np.ones(k)
    toks = [h.split() for h in hyps]
    out = np.zeros(k)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            denom = max(len(toks[i]), len(toks[j]))
            D[i, j] = D[j, i] = (0.0 if denom == 0
                                 else edit_distance(toks[i], toks[j]) / denom)
    for i in range(k):
        out[i] = 1.0 - (D[i].sum() / (k - 1))
    return np.clip(out, 0.0, 1.0)

class FusionCache:
    """The three memo tables every combination arm shares. One per run.

    Fusion is expensive in ways that do not depend on the hyperparameters, and a
    20x2 pool fits 40 folds x 3 arms x however many trials. Each of these is
    computed once and read back:

    * `networks` — the confusion network for a store row. A function of the
      transcripts alone, so it is fold- and trial-independent.
    * `agreement` — the `cn` confidence for a store row. Likewise.
    * `scores` — `(row, transcript) -> error count`. The expensive one and the
      least obvious: the *fused string* for a row is usually identical across
      folds, because the fitted weights barely move, so scoring is memoised on
      the string rather than on the configuration that produced it. On AMI-SDM
      this turns ~624k `jiwer` calls per arm into ~31k.
    * `probes` — the fold's confidence model, keyed by the fit rows. Shared by
      every arm and every trial that sees the same fold, so the SVD is paid once
      per fold rather than once per trial.

    It lives on `BuildContext.cache` so that `hpt`, `run_cv` and a hand-built
    arm all reach the same one without a global.
    """

    def __init__(self):
        self.networks, self.agreement, self.scores, self.probes = {}, {}, {}, {}

    @classmethod
    def get(cls, cache):
        """The `FusionCache` in a `BuildContext.cache` dict, created on demand."""
        if cache is None:
            return cls()
        got = cache.get("fusion")
        if got is None:
            got = cache["fusion"] = cls()
        return got

    def __repr__(self):
        return (f"FusionCache(networks={len(self.networks):,}, "
                f"agreement={len(self.agreement):,}, scores={len(self.scores):,}, "
                f"probes={len(self.probes)})")

    def hypotheses(self, texts, members, row):
        """The `k` hypothesis strings for one store row, in member order."""
        rec = texts.iloc[int(row)]
        return [str(rec[m]) for m in members]

    def network(self, texts, members, row):
        """The confusion network for one store row, built in MEMBER order.

        The order matters — ROVER's alignment is order-dependent — and member
        order is used rather than best-system-first so that every arm and every
        configuration shares one cached network. What the seed order buys is
        argued in §9.2; what sharing the cache buys is a 40-fold pool that runs
        in minutes, and the two are not close.
        """
        key = int(row)
        if key not in self.networks:
            self.networks[key] = build_network(self.hypotheses(texts, members, key))
        return self.networks[key]

    def agree(self, texts, members, row):
        """The `cn` per-system confidence for one store row."""
        key = int(row)
        if key not in self.agreement:
            self.agreement[key] = agreement(self.hypotheses(texts, members, key))
        return self.agreement[key]

    def errors(self, texts, rows, hyps):
        """`(n,)` error counts for fused transcripts, memoised on the string.

        The reference is fixed per row, so `(row, transcript)` determines the
        count and nothing about the fold or the configuration can change it.
        """
        out = np.empty(len(rows), dtype=np.float64)
        for i, (row, hyp) in enumerate(zip(np.asarray(rows, dtype=int), hyps)):
            key = (int(row), hyp)
            got = self.scores.get(key)
            if got is None:
                ref = str(texts.iloc[int(row)]["reference"])
                got = self.scores[key] = float(sum(word_counters(ref, hyp)[:3]))
            out[i] = got
        return out

    def probe(self, X, Y, rows, alphas):
        """The fold's `(N, k)` WER-prediction model, fitted once per fold.

        Keyed by the fit rows AND a checksum of the design matrix: the rows alone
        would collide if the same fold were ever fitted through a different
        transform (a PCA control, say), and a confidence model fitted on one
        design matrix is not one for another.
        """
        key = (hashlib.sha1(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16],
               X.shape[1], round(float(X.sum()), 3), len(alphas))
        got = self.probes.get(key)
        if got is None:
            got = self.probes[key] = RidgeProbe(alphas=alphas).fit(X, Y)
        return got

def _blend(alpha, freq, conf, null_penalty, is_null):
    """One candidate's Fiscus score."""
    return (alpha * freq + (1.0 - alpha) * conf) * (null_penalty if is_null else 1.0)


def fiscus_vote(slots, mass, conf, alpha=1.0, null_penalty=1.0):
    """Vote one network with per-system mass and per-system confidence.

    `mass` and `conf` are both length `k`, in the same order as each slot's
    entries. `mass` is the system prior for this utterance (it need not sum to
    one; only ratios matter); `conf` is each system's confidence on it.

    Ties break towards emitting a word, matching `vote` — a deletion has to win
    outright, because an ensemble that deletes on a tie loses words it had.
    """
    words = []
    for slot in slots:
        freq, conf_mass = {}, {}
        for w, m, c in zip(slot, mass, conf):
            freq[w] = freq.get(w, 0.0) + m
            conf_mass[w] = conf_mass.get(w, 0.0) + m * c
        best, best_score = None, -np.inf
        for w, f in freq.items():
            score = _blend(alpha, f, (conf_mass[w] / f) if f > 0 else 0.0,
                           null_penalty, w is None)
            # `(score, w is not None)` as the key, so a word beats a deletion at
            # equal score.
            if (score, w is not None) > (best_score, best is not None):
                best, best_score = w, score
        if best is not None:
            words.append(best)
    return " ".join(words)


class ConfWeightedRover:
    """Confidence-weighted ROVER (§9.5.1) as a tunable `run_cv` arm.

    Parameters
    ----------
    texts, members : as `RoverFusion`.
    alpha : Fiscus's frequency/confidence trade. `1.0` is pure frequency.
    conf_source : `"cn"` (agreement with the other systems), `"probe"` (a
        fold-fitted ridge model's `1 - predicted WER`), or `"blend"` (their
        geometric mean — a system needs both to be trusted).
    weighting : the system prior, from the training half's WER. See
        `system_weights`.
    weight_by_conf : scale each system's vote by `conf ** conf_temp` as well.
    conf_temp : sharpens the confidence before it is used, either way.
    null_penalty : how loudly a system that aligned nothing argues for deletion.
    alphas : the ridge path for the confidence model. Short by default — this is
        a confidence estimate, not the headline model, and the LOO selection
        makes a long path mostly wasted SVD-free work.
    """

    wants_rows = True
    produces = "errors"

    def __init__(self, texts, members, alpha=1.0, conf_source="cn",
                 weighting="inverse_wer", weight_by_conf=False, conf_temp=1.0,
                 null_penalty=1.0, alphas=None, cache=None):
        self.texts, self.members = texts, tuple(members)
        self.alpha, self.conf_source = float(alpha), conf_source
        self.weighting, self.weight_by_conf = weighting, bool(weight_by_conf)
        self.conf_temp, self.null_penalty = float(conf_temp), float(null_penalty)
        self.alphas = np.logspace(-2, 6, 12) if alphas is None else alphas
        self.cache = FusionCache.get(cache) if not isinstance(cache, FusionCache) else cache
        self.weights_ = self.probe_ = None

    def _needs_probe(self):
        return self.conf_source in ("probe", "blend")

    def fit(self, X, y, nref=None, rows=None):
        if rows is None:
            raise ValueError("ConfWeightedRover needs `rows`; run_cv passes them "
                             "when `wants_rows` is set")
        E = np.asarray(y, dtype=np.float64)
        n = np.ones(len(E)) if nref is None else np.asarray(nref, dtype=np.float64)
        wer = (E * n[:, None]).sum(0) / n.sum()
        self.weights_ = system_weights(self.weighting, wer)
        if self._needs_probe():
            # Fitted on the training half only, and shared with every other arm
            # that sees this fold. `y` is already the per-utterance WER matrix,
            # so the target is exactly "how wrong is member m on utterance i".
            self.probe_ = self.cache.probe(X, E, rows, self.alphas)
        return self

    def _confidence(self, X, rows):
        """`(n, k)` per-utterance confidence, in `[0, 1]`, member order."""
        k = len(self.members)
        cn = None
        if self.conf_source in ("cn", "blend"):
            cn = np.vstack([self.cache.agree(self.texts, self.members, r)
                            for r in rows])
        pr = None
        if self._needs_probe():
            pred = np.asarray(self.probe_.predict(X), dtype=np.float64)
            pr = np.clip(1.0 - pred.reshape(len(rows), k), 0.0, 1.0)
        if self.conf_source == "cn":
            conf = cn
        elif self.conf_source == "probe":
            conf = pr
        else:
            # Geometric mean: a system has to look right to BOTH the network and
            # the probe. An arithmetic mean lets one source carry a system the
            # other has rejected, which is the failure this arm exists to avoid.
            conf = np.sqrt(np.clip(cn, 1e-6, None) * np.clip(pr, 1e-6, None))
        return np.clip(conf, 1e-6, 1.0) ** self.conf_temp

    def transcribe(self, X, rows):
        conf = self._confidence(X, rows)
        out = []
        for i, row in enumerate(rows):
            slots = self.cache.network(self.texts, self.members, row)
            mass = (self.weights_ * conf[i]) if self.weight_by_conf else self.weights_
            out.append(fiscus_vote(slots, mass, conf[i], self.alpha,
                                   self.null_penalty))
        return out

    def predict(self, X, rows=None):
        if rows is None:
            raise ValueError("ConfWeightedRover needs `rows` at predict time too")
        rows = np.asarray(rows, dtype=int)
        return self.cache.errors(self.texts, rows, self.transcribe(X, rows))

class CNMBRFusion:
    """Confusion-network MBR combination (§9.5.2) as a tunable `run_cv` arm.

    Parameters
    ----------
    decode : `"slot"` (per-slot argmax) or `"hyp"` (lowest expected error among
        candidate transcripts).
    candidates : `"members"` or `"members+consensus"`. Only read when
        `decode == "hyp"`. With `"members"` the arm selects a member and sets
        `produces = "choice"`.
    prior : the system prior, as in `system_weights`.
    temperature : sharpens the slot posterior. Below 1 commits harder; above 1
        hedges. It is the knob that decides whether a 2-1 slot is a decision or
        a coin flip.
    null_prior : multiplies the empty token's mass before normalisation. Above 1
        the combination becomes willing to delete, below 1 reluctant.
    use_conf : weight each system's vote by its per-utterance confidence as well
        as by its prior.

    `produces` is set on the **instance**, in `__init__`, because it is a
    property of the configuration rather than of the class: `run_cv` and the
    tuner both read it off the object, so a `hyp`/`members` instance is scored
    through the router path and every other instance through the fusion path.
    """

    wants_rows = True

    def __init__(self, texts, members, decode="slot", candidates="members",
                 prior="inverse_wer", temperature=1.0, null_prior=1.0,
                 use_conf=False, conf_source="cn", alphas=None, cache=None):
        self.texts, self.members = texts, tuple(members)
        self.decode, self.candidates = decode, candidates
        self.prior, self.temperature = prior, float(temperature)
        self.null_prior, self.use_conf = float(null_prior), bool(use_conf)
        self.conf_source = conf_source
        self.alphas = np.logspace(-2, 6, 12) if alphas is None else alphas
        self.cache = FusionCache.get(cache) if not isinstance(cache, FusionCache) else cache
        self.produces = ("choice" if (decode == "hyp" and candidates == "members")
                         else "errors")
        self.weights_ = self.probe_ = None

    def fit(self, X, y, nref=None, rows=None):
        if rows is None:
            raise ValueError("CNMBRFusion needs `rows`; run_cv passes them when "
                             "`wants_rows` is set")
        E = np.asarray(y, dtype=np.float64)
        n = np.ones(len(E)) if nref is None else np.asarray(nref, dtype=np.float64)
        wer = (E * n[:, None]).sum(0) / n.sum()
        self.weights_ = system_weights(self.prior, wer)
        if self.use_conf and self.conf_source in ("probe", "blend"):
            self.probe_ = self.cache.probe(X, E, rows, self.alphas)
        return self

    def _confidence(self, X, rows):
        k = len(self.members)
        if not self.use_conf:
            return np.ones((len(rows), k))
        if self.conf_source == "cn":
            # Floored, as in the `blend` branch below: `agreement` returns 0 for
            # a system that agrees with nobody, and on an utterance where the
            # members share no words it returns 0 for ALL of them. A row of
            # zeros would multiply the system prior away entirely, and "no
            # confidence anywhere" carries no information about which system to
            # believe — it must leave the prior standing, not erase it.
            return np.clip(
                np.vstack([self.cache.agree(self.texts, self.members, r)
                           for r in rows]), 1e-6, None)
        pred = np.asarray(self.probe_.predict(X), dtype=np.float64)
        pr = np.clip(1.0 - pred.reshape(len(rows), k), 1e-6, 1.0)
        if self.conf_source == "probe":
            return pr
        cn = np.vstack([self.cache.agree(self.texts, self.members, r) for r in rows])
        return np.sqrt(np.clip(cn, 1e-6, None) * pr)

    def _posterior(self, slots, mass):
        """`[{word or None: probability}]`, one dict per slot."""
        out = []
        for slot in slots:
            raw = {}
            for w, m in zip(slot, mass):
                raw[w] = raw.get(w, 0.0) + m
            top = max(raw.values(), default=0.0)
            if top <= 0.0:
                # No mass reached this slot at all. Nothing separates its
                # candidates, so it is uniform over them — never empty: an
                # empty slot has neither an argmax for `_consensus` nor a
                # maximum posterior for the consensus cost in `_decode_row`.
                raw, top = {w: 1.0 for w in raw}, 1.0
            # Tempered, then the empty token re-weighted, then normalised. The
            # order matters: applying `null_prior` before the temperature would
            # let the temperature undo it. Masses are scaled by the slot's
            # largest before tempering, which is the same posterior once
            # normalised but keeps the top candidate at exactly 1.0, so a small
            # temperature cannot underflow every candidate to zero either.
            post = {w: ((v / top) ** (1.0 / self.temperature))
                    for w, v in raw.items() if v > 0}
            if None in post:
                post[None] *= self.null_prior
            total = sum(post.values()) or 1.0
            out.append({w: v / total for w, v in post.items()})
        return out

    def _expected_error(self, slots, post, system):
        """Expected word error of system `system`'s path through the network.

        Slot-independent, which is the standard CN-MBR approximation: the cost
        of a path is the sum over slots of the posterior mass it disagrees with.
        """
        total = 0.0
        for slot, p in zip(slots, post):
            total += 1.0 - p.get(slot[system], 0.0)
        return total

    def _consensus(self, post):
        return " ".join(
            w for w in (max(p, key=lambda k: (p[k], k is not None)) for p in post)
            if w is not None)

    def _decode_row(self, row, mass):
        """`(transcript, member index or -1)` for one store row."""
        slots = self.cache.network(self.texts, self.members, row)
        post = self._posterior(slots, mass)
        if self.decode == "slot":
            return self._consensus(post), -1
        costs = [self._expected_error(slots, post, s)
                 for s in range(len(self.members))]
        pick = int(np.argmin(costs))
        if self.candidates == "members":
            return None, pick
        # The consensus competes with the members on the same ruler: its own
        # expected error is the posterior mass it disagrees with, slot by slot.
        cons_cost = sum(1.0 - max(p.values()) for p in post)
        if cons_cost < costs[pick]:
            return self._consensus(post), -1
        return self.texts.iloc[int(row)][self.members[pick]], pick

    def predict(self, X, rows=None):
        if rows is None:
            raise ValueError("CNMBRFusion needs `rows` at predict time too")
        rows = np.asarray(rows, dtype=int)
        conf = self._confidence(X, rows)
        hyps, picks = [], []
        for i, row in enumerate(rows):
            hyp, pick = self._decode_row(row, self.weights_ * conf[i])
            hyps.append(hyp)
            picks.append(pick)
        if self.produces == "choice":
            return np.asarray(picks, dtype=int)
        return self.cache.errors(self.texts, rows, [str(h) for h in hyps])

def _zscore_rows(A):
    """Standardise each ROW of `(n, k)`. Constant rows come back as zeros."""
    A = np.asarray(A, dtype=np.float64)
    mu = A.mean(axis=1, keepdims=True)
    sd = A.std(axis=1, keepdims=True)
    return np.where(sd > 1e-12, (A - mu) / np.maximum(sd, 1e-12), 0.0)


class ScoreFusion:
    """Score-based fusion (§9.5.3) as a tunable `run_cv` arm.

    Parameters
    ----------
    mode : `"hard"` selects the top-scoring system (and reports as a router);
        `"soft"` votes the network with `softmax(temperature * score)`.
    w_conf, w_consensus, w_length : the score's mixing weights. Searched over a
        range that includes zero and both signs.
    temperature : inverse temperature of the soft mode's softmax. Ignored by
        `hard`, which is the `temperature -> inf` limit of it.
    null_penalty : as everywhere in this section. Soft mode only.
    """

    wants_rows = True

    def __init__(self, texts, members, mode="hard", w_conf=1.0, w_consensus=0.5,
                 w_length=0.0, temperature=1.0, null_penalty=1.0, alphas=None,
                 cache=None):
        self.texts, self.members = texts, tuple(members)
        self.mode = mode
        self.w_conf, self.w_consensus = float(w_conf), float(w_consensus)
        self.w_length, self.temperature = float(w_length), float(temperature)
        self.null_penalty = float(null_penalty)
        self.alphas = np.logspace(-2, 6, 12) if alphas is None else alphas
        self.cache = FusionCache.get(cache) if not isinstance(cache, FusionCache) else cache
        self.produces = "choice" if mode == "hard" else "errors"
        self.probe_ = None

    def fit(self, X, y, nref=None, rows=None):
        if rows is None:
            raise ValueError("ScoreFusion needs `rows`; run_cv passes them when "
                             "`wants_rows` is set")
        self.probe_ = self.cache.probe(X, np.asarray(y, dtype=np.float64), rows,
                                       self.alphas)
        return self

    def _lengths(self, rows):
        """`(n, k)` |median length - own length|, negated so larger is better."""
        out = np.zeros((len(rows), len(self.members)))
        for i, row in enumerate(rows):
            n = np.array([len(h.split()) for h in
                          self.cache.hypotheses(self.texts, self.members, row)],
                         dtype=np.float64)
            out[i] = -np.abs(n - np.median(n))
        return out

    def scores(self, X, rows):
        """`(n, k)` per-utterance system scores."""
        k = len(self.members)
        pred = np.asarray(self.probe_.predict(X), dtype=np.float64).reshape(len(rows), k)
        conf = _zscore_rows(-pred)                       # lower predicted WER is better
        cons = _zscore_rows(np.vstack([self.cache.agree(self.texts, self.members, r)
                                       for r in rows]))
        leng = _zscore_rows(self._lengths(rows))
        return self.w_conf * conf + self.w_consensus * cons + self.w_length * leng

    def predict(self, X, rows=None):
        if rows is None:
            raise ValueError("ScoreFusion needs `rows` at predict time too")
        rows = np.asarray(rows, dtype=int)
        S = self.scores(X, rows)
        if self.mode == "hard":
            return S.argmax(axis=1)
        W = _softmax_rows(self.temperature * S)
        hyps = []
        for i, row in enumerate(rows):
            slots = self.cache.network(self.texts, self.members, row)
            hyps.append(fiscus_vote(slots, W[i], np.ones(len(self.members)),
                                    alpha=1.0, null_penalty=self.null_penalty))
        return self.cache.errors(self.texts, rows, hyps)


def _softmax_rows(S):
    """Row-wise softmax, shifted by the row max for stability."""
    Z = np.exp(np.asarray(S, dtype=np.float64) - np.max(S, axis=1, keepdims=True))
    return Z / Z.sum(axis=1, keepdims=True)

def _fusion_ctx(ctx, who):
    """`(texts, members, cache)` from a `BuildContext`, or a message saying why not."""
    if ctx is None or ctx.texts is None:
        raise ValueError(
            f"{who} works on transcripts, which the store does not load by "
            f"default. Call `store.load_texts(members)` before `hpt(...)` or "
            f"`Tuner(...)` — the fusion arms read them off `BuildContext`.")
    if not len(ctx.members):
        raise ValueError(f"{who} needs `BuildContext.members`; `Tuner` fills it")
    return ctx.texts, ctx.members, ctx.cache


def conf_rover_space():
    """Search space for `rover_conf`. `alpha=1` recovers the §9.2 majority vote."""
    return {p.name: p for p in [
        Param('alpha', 'float', low=0.0, high=1.0, hard_low=0.0, hard_high=1.0),
        Param('conf_source', 'cat', choices=('cn', 'probe', 'blend')),
        Param('weighting', 'cat', choices=WEIGHT_SCHEMES),
        Param('weight_by_conf', 'cat', choices=(False, True)),
        Param('conf_temp', 'float', low=0.25, high=4.0, log=True,
              hard_low=0.05, hard_high=20.0),
        Param('null_penalty', 'float', low=0.25, high=4.0, log=True,
              hard_low=0.05, hard_high=20.0),
    ]}


def cn_mbr_space():
    """Search space for `cn_mbr`."""
    return {p.name: p for p in [
        Param('decode', 'cat', choices=('slot', 'hyp')),
        Param('candidates', 'cat', choices=('members', 'members+consensus')),
        Param('prior', 'cat', choices=WEIGHT_SCHEMES),
        Param('temperature', 'float', low=0.1, high=10.0, log=True,
              hard_low=0.01, hard_high=100.0),
        Param('null_prior', 'float', low=0.05, high=5.0, log=True,
              hard_low=0.001, hard_high=50.0),
        Param('use_conf', 'cat', choices=(False, True)),
        Param('conf_source', 'cat', choices=('cn', 'probe', 'blend')),
    ]}


def score_fusion_space():
    """Search space for `score_fusion`. Every weight may reach zero or go negative."""
    return {p.name: p for p in [
        Param('mode', 'cat', choices=('hard', 'soft')),
        Param('w_conf', 'float', low=-1.0, high=3.0, hard_low=-10.0, hard_high=10.0),
        Param('w_consensus', 'float', low=-1.0, high=3.0, hard_low=-10.0, hard_high=10.0),
        Param('w_length', 'float', low=-1.0, high=1.0, hard_low=-10.0, hard_high=10.0),
        Param('temperature', 'float', low=0.1, high=10.0, log=True,
              hard_low=0.01, hard_high=100.0),
        Param('null_penalty', 'float', low=0.25, high=4.0, log=True,
              hard_low=0.05, hard_high=20.0),
    ]}


def suggest_conf_rover(trial, space, max_iter=None):
    p = {k: space[k].suggest(trial) for k in
         ('alpha', 'conf_source', 'weighting', 'weight_by_conf', 'null_penalty')}
    # Conditional, exactly like `suggest_alternating`: the temperature only does
    # something when a confidence is actually consulted, and sampling it when
    # `alpha == 1` and the vote is pure frequency would spend the budget
    # measuring a parameter that cannot move the score. `space_from_trials`
    # flags the resulting thin evidence with `!`; that is the intended
    # behaviour, not a gap.
    if p['conf_source'] != 'cn' or p['alpha'] < 1.0 or p['weight_by_conf']:
        p['conf_temp'] = space['conf_temp'].suggest(trial)
    return p


def suggest_cn_mbr(trial, space, max_iter=None):
    p = {k: space[k].suggest(trial) for k in
         ('decode', 'prior', 'temperature', 'null_prior', 'use_conf')}
    if p['decode'] == 'hyp':
        p['candidates'] = space['candidates'].suggest(trial)
    if p['use_conf']:
        p['conf_source'] = space['conf_source'].suggest(trial)
    return p


def suggest_score_fusion(trial, space, max_iter=None):
    p = {k: space[k].suggest(trial) for k in
         ('mode', 'w_conf', 'w_consensus', 'w_length')}
    if p['mode'] == 'soft':
        # `hard` is the temperature -> infinity limit, so neither knob exists
        # for it. Sampling them anyway would make two thirds of the space a
        # relabelling of the same model.
        p['temperature'] = space['temperature'].suggest(trial)
        p['null_penalty'] = space['null_penalty'].suggest(trial)
    return p


def build_conf_rover(p, in_features, seed=42, ctx=None):
    texts, members, cache = _fusion_ctx(ctx, "rover_conf")
    return ConfWeightedRover(texts, members, alpha=p['alpha'],
                             conf_source=p['conf_source'], weighting=p['weighting'],
                             weight_by_conf=p['weight_by_conf'],
                             conf_temp=p.get('conf_temp', 1.0),
                             null_penalty=p['null_penalty'], cache=cache)


def build_cn_mbr(p, in_features, seed=42, ctx=None):
    texts, members, cache = _fusion_ctx(ctx, "cn_mbr")
    return CNMBRFusion(texts, members, decode=p['decode'],
                       candidates=p.get('candidates', 'members'), prior=p['prior'],
                       temperature=p['temperature'], null_prior=p['null_prior'],
                       use_conf=p['use_conf'],
                       conf_source=p.get('conf_source', 'cn'), cache=cache)


def build_score_fusion(p, in_features, seed=42, ctx=None):
    texts, members, cache = _fusion_ctx(ctx, "score_fusion")
    return ScoreFusion(texts, members, mode=p['mode'], w_conf=p['w_conf'],
                       w_consensus=p['w_consensus'], w_length=p['w_length'],
                       temperature=p.get('temperature', 1.0),
                       null_penalty=p.get('null_penalty', 1.0), cache=cache)


def rover_w_space():
    """Classical weighted ROVER: `conf_rover_space` with the confidence terms off.

    `alpha` frozen at 1.0 makes the Fiscus score pure vote frequency, and
    `weight_by_conf` frozen off keeps each system's vote at its prior weight —
    which is precisely `RoverFusion`, the weighted ROVER of §9.2. Freezing
    rather than reimplementing is what makes the two arms comparable: they run
    the same code down the same path, so the difference between their rows is
    the confidence term and nothing else. `tests/t_fusion_arms.py` #3 checks the
    equivalence against `vote()` directly rather than taking it on trust.

    What stays free is exactly what `rover_grid` used to enumerate — the
    weighting scheme and the null penalty — so this arm is the same search, run
    by the same tuner as everything else instead of by a bespoke grid.
    """
    space = conf_rover_space()
    return {**space,
            "alpha": replace(space["alpha"], frozen=1.0),
            "weight_by_conf": replace(space["weight_by_conf"], frozen=False),
            "conf_source": replace(space["conf_source"], frozen="cn"),
            "conf_temp": replace(space["conf_temp"], frozen=1.0)}


MODELS.update({
    "rover_w": ModelSpec("rover_w", rover_w_space, suggest_conf_rover,
                         build_conf_rover, trials=0.3),
    "rover_conf": ModelSpec("rover_conf", conf_rover_space, suggest_conf_rover,
                            build_conf_rover, trials=0.5),
    "cn_mbr": ModelSpec("cn_mbr", cn_mbr_space, suggest_cn_mbr, build_cn_mbr,
                        trials=0.5),
    "score_fusion": ModelSpec("score_fusion", score_fusion_space,
                              suggest_score_fusion, build_score_fusion,
                              trials=0.5),
})

# The combination baselines. `RIVALS` in §7 is the selection field; this is the
# fusion field, and a table that reports one without the other is incomplete.
FUSION_RIVALS = ("rover_w", "rover_conf", "cn_mbr", "score_fusion")

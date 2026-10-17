__all__ = ['SoftmaxProbe', 'select_k_best', 'EnsembleRouter', 'top_ensembles', 'run_routing', 'push_ensemble_results']

import numpy as np
import pandas as pd
import torch

from fastt.core import active_dataset
from fastt.ensemble import best_single_model, ensemble_stats, win_rates
from fastt.hub import FasttHub
from fastt.probing import RidgeProbe


class SoftmaxProbe:
    """Multinomial logistic regression, L2-penalised, full-batch L-BFGS.

    Written here rather than pulled from scikit-learn for the same reason
    `RidgeProbe` is: no install group in §1 carries sklearn, and this is ~20
    lines that run on the GPU next to everything else.

    The penalty applies to the weights but not the intercept, so the class priors
    stay free — which matters, because the sample weights below deliberately
    reweight the classes.

    Parameters
    ----------
    alpha : L2 strength, on the same scale as the mean loss (sample weights are
        normalised to sum to one).
    """

    def __init__(self, alpha=1.0, max_iter=200, device=None, dtype=torch.float32):
        self.alpha, self.max_iter, self.dtype = alpha, max_iter, dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _t(self, a, dtype=None):
        return torch.as_tensor(np.asarray(a), dtype=dtype or self.dtype, device=self.device)

    def fit(self, X, y, sample_weight=None):
        X, y = self._t(X), self._t(y, torch.long)
        w = (torch.ones(len(y), dtype=self.dtype, device=self.device)
             if sample_weight is None else self._t(sample_weight))
        if w.sum() <= 0:
            raise ValueError("all sample weights are zero — nothing to learn from")
        w = w / w.sum()

        self.n_classes_ = int(y.max().item()) + 1
        W = torch.zeros(X.shape[1], self.n_classes_, dtype=self.dtype,
                        device=self.device, requires_grad=True)
        b = torch.zeros(self.n_classes_, dtype=self.dtype,
                        device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=self.max_iter, history_size=10,
                                tolerance_grad=1e-7, tolerance_change=1e-9,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(X @ W + b, y, reduction="none")
            loss = (ce * w).sum() + self.alpha * 0.5 * (W ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.coef_, self.intercept_ = W.detach(), b.detach()
        return self

    def decision_function(self, X):
        """(N, C) logits. `argmax` is the prediction; differences are the margin."""
        return (self._t(X) @ self.coef_ + self.intercept_).cpu().numpy()


def select_k_best(X, y, k, sample_weight=None, task="classification"):
    """Indices of the `k` most informative columns, ranked univariately.

    `task="classification"` uses the one-way ANOVA F statistic — between-class
    variance over within-class variance, weighted by `sample_weight` so the
    ranking agrees with what the classifier is actually optimising.

    `task="regression"` takes, per column, the largest absolute Pearson
    correlation with any target column. A feature that separates *one* member
    from the others is exactly as useful for routing as one that separates all of
    them, so a max is the right reduction here, not a mean.

    Univariate on purpose: it ignores redundancy between columns, but it is one
    pass over the data and needs no model fit, which is what makes the sweep in
    §6.6 cheap enough to run for every group.
    """
    X = np.asarray(X, np.float64)
    if k is None or k >= X.shape[1]:
        return np.arange(X.shape[1])

    if task == "regression":
        Y = np.asarray(y, np.float64).reshape(len(X), -1)
        Xc = X - X.mean(0)
        Yc = Y - Y.mean(0)
        denom = np.outer(np.sqrt((Xc ** 2).sum(0)), np.sqrt((Yc ** 2).sum(0)))
        score = np.abs(Xc.T @ Yc / np.maximum(denom, 1e-12)).max(axis=1)
    else:
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, np.float64)
        classes = np.unique(y)
        grand = np.average(X, axis=0, weights=w)
        between = np.zeros(X.shape[1])
        within = np.zeros(X.shape[1])
        for c in classes:
            m = np.asarray(y) == c
            if w[m].sum() <= 0:
                continue
            mean_c = np.average(X[m], axis=0, weights=w[m])
            between += w[m].sum() * (mean_c - grand) ** 2
            within += (w[m][:, None] * (X[m] - mean_c) ** 2).sum(0)
        df_b, df_w = max(len(classes) - 1, 1), max(w.sum() - len(classes), 1)
        score = (between / df_b) / np.maximum(within / df_w, 1e-12)

    # Sorted so a column subset keeps the original column order, which keeps the
    # source labels in §6.6 aligned.
    return np.sort(np.argsort(-score)[:k])

class EnsembleRouter:
    """Fit and score every router for ONE group of models.

    Parameters
    ----------
    store : a loaded `EnsembleStore`.
    members : the group, e.g. `("whisper_large_v3", "wav2vec2", "data2vec")`.
    use_extra_features : include the feature-only models (the MFCC baseline).
    use_duration : include `log(1 + n_frames)` as a column.
    centre : mean-centre the regressor's targets across members. See §6.5.

    The whole object is built for one group and holds its standardised design
    matrix, so fitting several routers over it costs one build.
    """

    def __init__(self, store, members, fit_splits=None, eval_splits=None,
                 use_extra_features=True, use_duration=True, centre=True,
                 device=None, seed=0):
        self.store, self.members = store, tuple(members)
        fit_splits = fit_splits or store.spec.fit_splits
        eval_splits = eval_splits or store.spec.eval_splits
        self.fit_splits, self.eval_splits = tuple(fit_splits), tuple(eval_splits)
        self.centre, self.seed = centre, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.fit_mask = store.mask(self.fit_splits)
        self.eval_mask = store.mask(self.eval_splits)
        self.err = store.err(self.members)                       # (N, k) all rows
        self.nref = store.nref
        self.utt_wer = self.err / np.maximum(self.nref, 1)[:, None]
        self.bsm = best_single_model(store, self.members, self.fit_splits)

        # Training label and its cost-sensitive weight, defined in §6.5.
        self.label = self.err.argmin(axis=1)
        self.weight = (self.err.max(axis=1) - self.err.min(axis=1)) / np.maximum(self.nref, 1)

        self._build_design(use_extra_features, use_duration)
        self.references = self._references()

    def __repr__(self):
        return (f"EnsembleRouter({'+'.join(self.members)}, "
                f"D={self.X.shape[1]}, n_informative={int(self._train_rows().sum())})")

    # ---------------------------------------------------------- the design --

    def _build_design(self, use_extra_features, use_duration):
        """Concatenate the feature blocks and standardise on the fit splits."""
        blocks, source = [], []
        names = list(self.members)
        if use_extra_features:
            names += list(self.store.feature_only)
        for name in names:
            F = self.store.features[name]
            blocks.append(F)
            source += [name] * F.shape[1]
        if use_duration:
            blocks.append(np.log1p(self.store.n_frames)[:, None])
            source.append("duration")

        X = np.concatenate(blocks, axis=1).astype(np.float64)
        mean = X[self.fit_mask].mean(0)
        # clamp: a constant column would otherwise divide by zero.
        std = X[self.fit_mask].std(0).clip(1e-8)
        self.X, self.source = (X - mean) / std, np.array(source)

    def columns(self, sources=None):
        """Column indices belonging to the named blocks (all of them by default)."""
        if sources is None:
            return np.arange(self.X.shape[1])
        return np.flatnonzero(np.isin(self.source, list(sources)))

    def _train_rows(self):
        """Fit-split rows the classifier can learn from: weight strictly positive."""
        return self.fit_mask & (self.weight > 0)

    # -------------------------------------------------------------- scoring --

    def _references(self):
        """oracle / bsm / random / worst, on the eval splits."""
        E, n = self.err[self.eval_mask], self.nref[self.eval_mask]
        return ensemble_stats(E, n, self.bsm)

    def _score(self, router, choice, n_features, **extra):
        """Turn a per-utterance choice into one row of the results table."""
        E, n = self.err[self.eval_mask], self.nref[self.eval_mask]
        picked = E[np.arange(len(E)), choice]
        best, worst = E.min(axis=1), E.max(axis=1)
        # Two subsets, and the difference matters. `contested` is every row where
        # the members disagree at all — the rows a router can win or lose on, and
        # exactly the rows it trains on. `sole` is the stricter one from §6.2:
        # a single member is strictly best, no shared minimum.
        contested = best < worst
        sole = (E == best[:, None]).sum(axis=1) == 1
        gap = self.references["gap_abs"]

        wer = picked.sum() / n.sum()
        row = {
            "members": "+".join(self.members), "router": router,
            "n_features": int(n_features), "wer": float(wer),
            # The headline: share of the oracle gap the router captured. Negative
            # means it did worse than simply always using the best single model.
            "gap_closed": float((self.references["bsm_wer"] - wer) / gap) if gap else np.nan,
            "regret_per_word": float((picked - best).sum() / n.sum()),
            "acc_tie_tolerant": float((picked == best).mean()),
            "acc_contested": float((picked == best)[contested].mean()),
            "acc_sole": float((picked == best)[sole].mean()),
            "macro_f1": self._macro_f1(choice[contested], E[contested].argmin(axis=1)),
            "switch_rate": float((choice != self.bsm).mean()),
            "n_train": int(self._train_rows().sum()), "n_eval": int(len(E)),
        }
        row.update(extra)
        return row

    def _wer_only_row(self, router, wer, n_features=0, **extra):
        """A row for something that has a WER but makes no per-utterance choice.

        `random` and `reg_soft` are both expectations over members, so accuracy,
        F1 and switch rate are undefined for them rather than zero.
        """
        gap = self.references["gap_abs"]
        row = {"members": "+".join(self.members), "router": router,
               "n_features": int(n_features), "wer": float(wer),
               "gap_closed": float((self.references["bsm_wer"] - wer) / gap) if gap else np.nan,
               "regret_per_word": np.nan, "acc_tie_tolerant": np.nan,
               "acc_contested": np.nan, "acc_sole": np.nan,
               "macro_f1": np.nan, "switch_rate": np.nan,
               "n_train": int(self._train_rows().sum()), "n_eval": int(self.eval_mask.sum()),
               "alpha": np.nan, "tau": np.nan}
        row.update(extra)
        return row

    def _macro_f1(self, pred, true):
        """Unweighted mean per-class F1 over the decisive utterances.

        Macro rather than micro because the classes are unbalanced and the point
        is whether the router can find *every* member's territory, not just the
        biggest one.
        """
        scores = []
        for c in range(len(self.members)):
            tp = float(((pred == c) & (true == c)).sum())
            p = tp / max((pred == c).sum(), 1)
            r = tp / max((true == c).sum(), 1)
            scores.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        return float(np.mean(scores))

    def _gate(self, choice, margin):
        """Override the BSM only where `margin` clears a fit-split-tuned threshold.

        The threshold is chosen to minimise total errors ON THE FIT SPLITS, so the
        eval numbers stay honest. Candidate thresholds are quantiles of the
        observed margins, which adapts to whatever scale the router works on —
        logits for the classifier, predicted WER for the regressor.
        """
        f = self.fit_mask
        # The incumbent to beat is "never switch", i.e. the BSM everywhere.
        best_errors = self.err[f, self.bsm].sum()
        best_tau = None
        positive = margin[f][margin[f] > 0]
        candidates = (np.quantile(positive, np.linspace(0, 0.99, 60))
                      if len(positive) else [])
        for tau in candidates:
            gated = np.where(margin[f] >= tau, choice[f], self.bsm)
            total = self.err[f][np.arange(f.sum()), gated].sum()
            if total < best_errors:
                best_errors, best_tau = total, tau

        if best_tau is None:            # no threshold beat staying put
            return np.full(len(choice), self.bsm), float("nan")
        return np.where(margin >= best_tau, choice, self.bsm), float(best_tau)

    # -------------------------------------------------------------- probes --

    def classifier(self, k=None, alphas=np.logspace(-3, 4, 8), folds=3,
                   columns=None, shuffle=False):
        """Fit the classifier router. Returns `(choice, margin, info)` for ALL rows.

        `alpha` is chosen by k-fold CV on the fit splits, scored by the **routing
        regret** the fold incurs — words lost against that fold's oracle — rather
        than by log-loss. Selecting on the quantity we care about costs nothing
        extra and is less likely to prefer a probe that is well calibrated but
        decides badly.
        """
        rows = self._train_rows()
        cols = self.columns() if columns is None else np.asarray(columns)
        X, y = self.X[rows][:, cols], self.label[rows]
        w, E = self.weight[rows], self.err[rows]
        rng = np.random.default_rng(self.seed)
        if shuffle:                                # the control: labels destroyed
            y = y[rng.permutation(len(y))]

        if k is not None:
            keep = select_k_best(X, y, k, w, task="classification")
            cols, X = cols[keep], X[:, keep]

        fold = rng.permutation(len(y)) % folds
        best_alpha, best_regret = alphas[0], np.inf
        for alpha in alphas:
            regret = 0.0
            for f in range(folds):
                tr, va = fold != f, fold == f
                probe = SoftmaxProbe(alpha, device=self.device).fit(X[tr], y[tr], w[tr])
                pick = probe.decision_function(X[va]).argmax(axis=1)
                regret += (E[va][np.arange(va.sum()), pick] - E[va].min(axis=1)).sum()
            if regret < best_regret:
                best_regret, best_alpha = regret, alpha

        probe = SoftmaxProbe(best_alpha, device=self.device).fit(X, y, w)
        logits = probe.decision_function(self.X[:, cols])
        choice = logits.argmax(axis=1)
        # Margin for the gate: how much better than staying on the BSM.
        margin = logits.max(axis=1) - logits[:, self.bsm]
        return choice, margin, {"alpha": float(best_alpha), "n_features": len(cols),
                                "columns": cols}

    def regressor(self, k=None, alphas=np.logspace(-2, 6, 30), columns=None):
        """Fit the WER-regressor router. Returns `(choice, margin, info)`.

        Both target blocks — raw per-member WER and the mean-centred version —
        share one `RidgeProbe` fit, so the quality table in §6.8 costs nothing
        beyond the routing fit itself. Routing uses whichever block `centre`
        selects; the raw block is always kept for reporting.
        """
        f = self.fit_mask
        cols = self.columns() if columns is None else np.asarray(columns)
        Y = self.utt_wer
        Yc = Y - Y.mean(axis=1, keepdims=True)
        if k is not None:
            keep = select_k_best(self.X[f][:, cols], (Yc if self.centre else Y)[f],
                                 k, task="regression")
            cols = cols[keep]

        k_m = len(self.members)
        probe = RidgeProbe(alphas, device=self.device).fit(
            self.X[f][:, cols], np.hstack([Y[f], Yc[f]]))
        pred = probe.predict(self.X[:, cols])
        pred_raw, pred_centred = pred[:, :k_m], pred[:, k_m:]

        used = pred_centred if self.centre else pred_raw
        choice = used.argmin(axis=1)
        # Margin: predicted words-per-word saved by leaving the BSM.
        margin = used[:, self.bsm] - used.min(axis=1)
        # `alpha_` holds one penalty per target column; report the mean over the
        # block that actually decided the route, not over both blocks.
        routing_block = slice(k_m, 2 * k_m) if self.centre else slice(0, k_m)
        return choice, margin, {"alpha": float(np.mean(probe.alpha_[routing_block])),
                                "n_features": len(cols), "columns": cols,
                                "pred_raw": pred_raw}

    def soft_weighted(self, pred, betas=np.logspace(-1, 5, 31)):
        """Expected WER under soft weights `softmax(-beta * pred)`, beta on the fit splits.

        With no way to fuse hypotheses, a weighted combination of models can only
        be scored as an expectation: `sum_m w_m * e_m`. beta = 0 gives random
        routing, beta -> inf gives hard routing, and the fit-split optimum lands
        wherever hedging stops paying.
        """
        def expected(mask, beta):
            w = torch.softmax(torch.as_tensor(-beta * pred[mask]), dim=1).numpy()
            return (w * self.err[mask]).sum() / self.nref[mask].sum()

        # beta running to the top of the grid is the expected result, not a
        # failure to converge: a weighted average of the members' WER is minimised
        # at a corner, so the fit splits keep asking for a harder and harder
        # softmax until it is hard routing.
        beta = min(betas, key=lambda b: expected(self.fit_mask, b))
        return float(expected(self.eval_mask, beta)), float(beta)

    # ------------------------------------------------------------ the sweep --

    def evaluate(self, ks=(64, 256, 1024, None), gate=True, shuffled=True,
                 baselines=True, verbose=False):
        """Every router at every K, as one tidy DataFrame. See §6.7 for the columns."""
        rows, E = [], self.err[self.eval_mask]

        def everywhere(column):
            return np.full(self.eval_mask.sum(), column)

        if baselines:
            for tag, choice in (("bsm", everywhere(self.bsm)),
                                ("oracle", E.argmin(axis=1)),
                                ("worst", E.argmax(axis=1))):
                rows.append(self._score(tag, choice, 0, alpha=np.nan, tau=np.nan))
            # `random` is an expectation over choices, not a choice, so it has no
            # accuracy and is written straight in.
            rows.append(self._wer_only_row("random", self.references["random_wer"]))
            # Two narrow feature sets, to show what the wide one is worth.
            for tag, srcs in (("duration_only", ["duration"]),
                              ("extra_only", list(self.store.feature_only))):
                cols = self.columns(srcs)
                if len(cols) == 0:
                    continue
                choice, _, info = self.classifier(columns=cols)
                rows.append(self._score(tag, choice[self.eval_mask], len(cols),
                                        alpha=info["alpha"], tau=np.nan))

        pred_for_soft = None
        for k in ks:
            label = "all" if k is None else str(k)

            choice, margin, info = self.classifier(k=k)
            rows.append(self._score(f"clf_k{label}", choice[self.eval_mask],
                                    info["n_features"], alpha=info["alpha"], tau=np.nan))
            if gate:
                gated, tau = self._gate(choice, margin)
                rows.append(self._score(f"clf_k{label}_gated", gated[self.eval_mask],
                                        info["n_features"], alpha=info["alpha"], tau=tau))

            choice, margin, info = self.regressor(k=k)
            rows.append(self._score(f"reg_k{label}", choice[self.eval_mask],
                                    info["n_features"], alpha=info["alpha"], tau=np.nan))
            if gate:
                gated, tau = self._gate(choice, margin)
                rows.append(self._score(f"reg_k{label}_gated", gated[self.eval_mask],
                                        info["n_features"], alpha=info["alpha"], tau=tau))
            if k is None:
                pred_for_soft = info["pred_raw"]

        if pred_for_soft is not None:
            wer, beta = self.soft_weighted(pred_for_soft)
            rows.append(self._wer_only_row("reg_soft", wer,
                                           n_features=self.X.shape[1], tau=beta))

        if shuffled:
            choice, _, info = self.classifier(shuffle=True)
            rows.append(self._score("clf_shuffled", choice[self.eval_mask],
                                    info["n_features"], alpha=info["alpha"], tau=np.nan))

        df = pd.DataFrame(rows)
        if verbose:
            print(df.round(4).to_string(index=False))
        return df

    def composition(self, ks=(64, 256, 1024), task="classification"):
        """Where the top-K features came from. One row per (K, source block).

        The interpretability payoff of §6.6: if the selected columns come almost
        entirely from one member's encoder, the router is reading that model's
        confidence rather than comparing the three.
        """
        # Score the features on the same rows the matching router trains on: the
        # classifier sees only informative utterances, the regressor sees all.
        if task == "classification":
            rows_mask = self._train_rows()
            y, w = self.label[rows_mask], self.weight[rows_mask]
        else:
            rows_mask = self.fit_mask
            y, w = (self.utt_wer - self.utt_wer.mean(axis=1, keepdims=True))[rows_mask], None

        out = []
        for k in ks:
            keep = select_k_best(self.X[rows_mask], y, k, w, task=task)
            picked = pd.Series(self.source[keep]).value_counts()
            for source, count in picked.items():
                out.append({"members": "+".join(self.members), "k": k,
                            "criterion": task, "source": source,
                            "n_selected": int(count), "share": count / len(keep),
                            "n_available": int((self.source == source).sum())})
        return pd.DataFrame(out)

def top_ensembles(fitness, n=8, by="gap_abs", include=()):
    """The `n` groups with the most to gain, as tuples, plus any you name."""
    picked = [tuple(m.split("+")) for m in fitness.nlargest(n, by)["members"]]
    for extra in include:
        extra = tuple(extra)
        if extra not in picked:
            picked.append(extra)
    return picked


def run_routing(store, ensembles, ks=(64, 256, 1024, None), fit_splits=None,
                eval_splits=None, gate=True, centre=True, verbose=True, **kw):
    """Fit and score every router for every group. Returns `(results, composition)`.

    Every group is scored on the identical protocol, so the rows are directly
    comparable across groups as well as across routers.
    """
    import time

    results, comps = [], []
    for i, members in enumerate(ensembles, 1):
        t0 = time.perf_counter()
        router = EnsembleRouter(store, members, fit_splits, eval_splits,
                                centre=centre, **kw)
        df = router.evaluate(ks=ks, gate=gate)
        results.append(df)
        comps.append(router.composition(ks=[k for k in ks if k is not None]))
        if verbose:
            best = df[df["router"].str.startswith(("clf", "reg"))].nlargest(1, "gap_closed")
            print(f"[{i}/{len(ensembles)}] {'+'.join(members):<62} "
                  f"gap {router.references['gap_abs']:.4f}  "
                  f"best {best['router'].iloc[0]} closed {best['gap_closed'].iloc[0]:+.1%}  "
                  f"({time.perf_counter() - t0:.0f}s)")
    return pd.concat(results, ignore_index=True), pd.concat(comps, ignore_index=True)


def push_ensemble_results(fitness, routing, composition, wins, store,
                          repo_id=None, pooling="mean", push=False):
    """Push the four §6 tables plus `results/ensemble.json`. Nothing moves unless `push`."""
    import json
    from datetime import datetime, timezone

    spec = store.spec
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": spec.name,
        "eval_splits": list(spec.eval_splits),
        "pooling": pooling,
        "routable_models": list(store.routable),
        "feature_only_models": list(store.feature_only),
        "n_utterances": int(len(store.index)),
        "n_ensembles": int(len(fitness)),
        "fitness": json.loads(fitness.to_json(orient="records")),
        "routing": json.loads(routing.to_json(orient="records")),
    }
    if not push:
        print("push=False — nothing uploaded. Call again with push=True when ready.")
        return payload

    hub = FasttHub(repo_id, spec=spec)
    for df, name in ((fitness, "ensemble_fitness"), (routing, "ensemble_routing"),
                     (composition, "ensemble_composition"), (wins, "ensemble_wins")):
        hub.push_dataframe(df, f"{spec.config_prefix}{name}",
                           f"Ensemble analysis on {spec.name}: {name} ({pooling} pooling)")
    hub.push_json(payload, spec.results_path("ensemble"),
                  f"Ensemble analysis summary for {spec.name}")
    return payload

__all__ = ['feature_matrix', 'load_split', 'RidgeProbe', 'run_probe']

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from fastt.core import active_dataset


def feature_matrix(ds, col, dtype=np.float64):
    """Read a pooled (N, D) column straight out of Arrow into a dense array.

    Pooled columns are `Sequence(Value("float32"), length=D)`, i.e. an Arrow
    fixed-size list: the width is recorded once in the schema and the values
    buffer is one contiguous block. Reading it this way skips the `datasets`
    formatter, which would otherwise build N x D Python floats.
    """
    if getattr(ds, "_indices", None) is not None:
        ds = ds.flatten_indices()
    chunks = ds.data.column(col).chunks
    return np.concatenate([
        c.flatten().to_numpy(zero_copy_only=False).reshape(len(c), -1) for c in chunks
    ]).astype(dtype, copy=False)


def load_split(ds, targets, feature_col, frames_col="n_frames"):
    """Return `(X, Y, n_frames)` for one split, bypassing the datasets formatter."""
    if getattr(ds, "_indices", None) is not None:
        ds = ds.flatten_indices()
    tbl = ds.data

    X = feature_matrix(ds, feature_col)

    ys = []
    for t in targets:
        chunks = tbl.column(t).chunks
        if any(c.null_count for c in chunks):
            raise ValueError(f"target column {t!r} contains nulls")
        ys.append(np.concatenate([c.to_numpy(zero_copy_only=False) for c in chunks]))
    Y = np.stack(ys, axis=1).astype(np.float64)

    n_frames = None
    if frames_col and frames_col in tbl.column_names:
        n_frames = np.concatenate([
            c.to_numpy(zero_copy_only=False) for c in tbl.column(frames_col).chunks
        ]).astype(np.float64)

    return X, Y, n_frames


class RidgeProbe:
    """Ridge regression in closed form, with exact leave-one-out alpha selection.

    Let `Xc` be the centred (and optionally standardised) design matrix with thin
    SVD `Xc = U diag(s) Vt`. For any penalty `a`,

        w(a) = V diag(s / (s^2 + a)) U' yc

    so ONE decomposition gives the entire alpha path. The leave-one-out residual
    is closed form too,

        e_i(a)  = (y_i - yhat_i(a)) / (1 - h_ii(a))
        h_ii(a) = 1/N + sum_k U_ik^2 s_k^2 / (s_k^2 + a)

    so alpha is chosen by exact LOO CV at no extra fitting cost and without
    holding data out.

    SVD of `Xc` rather than eigendecomposition of `Xc'Xc`: identical solution,
    but forming the Gram matrix squares the condition number, and pooled speech
    embeddings are badly conditioned.

    `y` may be (N,) or (N, T); extra targets share the decomposition, which is
    what makes the shuffled and length-residual controls nearly free.

    Attributes set by `fit`: `coef_` (D, T), `intercept_` (T,), `alpha_` (T,),
    `loo_mse_` (n_alphas, T), `dof_` (T,), `rank_`.
    """

    def __init__(self, alphas=np.logspace(-3, 7, 40), standardize=True,
                 fit_intercept=True, device=None, dtype=torch.float64,
                 max_elems=3 * 10 ** 7):
        self.alphas = np.atleast_1d(np.asarray(alphas, np.float64))
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_elems = max_elems          # caps the (N, T, alphas) LOO temporary

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)

    def fit(self, X, y):
        X, Y = self._t(X), self._t(y)
        self._1d = Y.ndim == 1
        if self._1d:
            Y = Y[:, None]
        N, D = X.shape
        T = Y.shape[1]
        if N != Y.shape[0]:
            raise ValueError(f"X has {N} rows but y has {Y.shape[0]}")

        zeros_d = torch.zeros(D, dtype=self.dtype, device=self.device)
        self.x_mean_ = X.mean(0) if self.fit_intercept else zeros_d
        self.y_mean_ = (Y.mean(0) if self.fit_intercept
                        else torch.zeros(T, dtype=self.dtype, device=self.device))
        Xc, Yc = X - self.x_mean_, Y - self.y_mean_
        self.x_std_ = (Xc.std(0).clamp_min(1e-8) if self.standardize
                       else torch.ones_like(self.x_mean_))
        Xc = Xc / self.x_std_

        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        s2 = S ** 2
        G = U.T @ Yc
        self.rank_ = int((S > S.max() * max(N, D) * torch.finfo(self.dtype).eps).sum())

        alphas = self._t(self.alphas)
        A = len(alphas)
        shrink = s2[:, None] / (s2[:, None] + alphas[None, :])              # (k, A)
        H = (U * U) @ shrink                                                # (N, A)
        if self.fit_intercept:
            H = H + 1.0 / N

        # Chunked over alphas: the full (N, T, A) fitted-value tensor would be
        # large enough to matter for a 5.6k-row fit with three target blocks.
        loo = torch.empty(A, T, dtype=self.dtype, device=self.device)
        step = max(1, min(A, self.max_elems // max(1, N * T)))
        for lo in range(0, A, step):
            hi = min(lo + step, A)
            fitted = torch.einsum("nk,kta->nta", U, G[:, :, None] * shrink[:, None, lo:hi])
            resid = Yc[:, :, None] - fitted
            denom = (1.0 - H[:, None, lo:hi]).clamp_min(1e-12)
            loo[lo:hi] = ((resid / denom) ** 2).mean(0).T

        best = loo.argmin(0)
        d_best = S[:, None] / (s2[:, None] + alphas[best][None, :])
        w = Vt.T @ (G * d_best)
        self.coef_ = w / self.x_std_[:, None]
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        self.alpha_ = self.alphas[best.cpu().numpy()]
        self.loo_mse_ = loo.cpu().numpy()
        self.dof_ = shrink[:, best].sum(0).cpu().numpy()
        return self

    def predict(self, X):
        out = (self._t(X) @ self.coef_ + self.intercept_).cpu().numpy()
        return out[:, 0] if self._1d else out

def _r2(y, p):
    """R^2 scored against the mean of the EVALUATION set."""
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run_probe(ds, targets=("wer",), feature_col="pool_l32_mean",
              train_splits=None, eval_splits=None, frames_col="n_frames",
              alphas=np.logspace(-3, 7, 40), length_alphas=np.logspace(-6, 3, 30),
              standardize=True, seed=0, device=None, verbose=False):
    """Fit one probe per target on the train splits, evaluate on the eval splits.

    The real probe, the shuffled-target control and the length-residualised
    probe all share X, so they share one SVD and cost 3T right-hand sides rather
    than three separate fits.

    Reported columns
    ----------------
    cv_r2             leave-one-out R^2 on the TRAINING splits. The honest
                      model-selection criterion — it never sees the eval splits.
    r2, pearson_r, spearman_r, rmse
                      the probe on the eval split. Pearson is invariant to
                      affine miscalibration, so a high r beside a poor r2 means
                      the predictions rank correctly but sit on the wrong scale.
                      Read Spearman too: utterance WER is right-skewed and a few
                      catastrophic utterances can carry Pearson.
    r2_shuffled       the same probe refitted on permuted training targets — the
                      floor reachable by memorising noise at this capacity.
    selectivity       r2 - r2_shuffled. Report this.
    r2_dummy          predicting the training mean; negative r2 means the probe
                      is worse than a constant.
    r2_length_only    log utterance duration alone.
    r2_beyond_length  the probe on length-residualised targets.
    alpha, dof        selected penalty and effective degrees of freedom.
    """
    targets = list(targets)
    T = len(targets)
    # Left at None, the fit splits come from the active dataset (§1.5) — which is
    # what makes this function corpus-agnostic.
    train_splits = list(train_splits) if train_splits else list(active_dataset().fit_splits)

    missing = [s for s in train_splits if s not in ds]
    if missing:
        raise KeyError(f"train split(s) {missing} not in {list(ds)}")
    eval_splits = list(eval_splits) if eval_splits else [s for s in ds if s not in train_splits]
    overlap = set(train_splits) & set(eval_splits)
    if overlap:
        raise ValueError(f"{sorted(overlap)} appear in BOTH train and eval")

    parts = [load_split(ds[s], targets, feature_col, frames_col) for s in train_splits]
    widths = {p[0].shape[1] for p in parts}
    if len(widths) > 1:
        raise ValueError(f"training splits disagree on feature width: {widths}")

    Xtr = np.concatenate([p[0] for p in parts])
    Ytr = np.concatenate([p[1] for p in parts])
    has_len = all(p[2] is not None for p in parts)
    Ftr = np.concatenate([p[2] for p in parts]) if has_len else None

    rng = np.random.default_rng(seed)
    if verbose:
        print(f"  {feature_col}: n={len(Xtr):,} D={Xtr.shape[1]}")

    # Length-only reference model, and length-residualised targets. log1p
    # compresses duration's long right tail; a nearly-unpenalised grid keeps the
    # BASELINE from being handicapped by shrinkage.
    if has_len:
        Ltr = np.log1p(Ftr)[:, None]
        len_model = RidgeProbe(length_alphas, standardize, device=device).fit(Ltr, Ytr)
        Rtr = Ytr - np.atleast_2d(len_model.predict(Ltr))

    blocks = [Ytr, Ytr[rng.permutation(len(Ytr))]] + ([Rtr] if has_len else [])
    probe = RidgeProbe(alphas, standardize, device=device).fit(Xtr, np.hstack(blocks))

    grid = probe.alphas
    edge = np.isin(probe.alpha_[:T], [grid.min(), grid.max()])
    if edge.any() and verbose:
        print(f"  warning: alpha hit the grid edge for "
              f"{[targets[i] for i in np.where(edge)[0]]} — widen `alphas`")

    # LOO error at the selected alpha, turned into an R^2 against the training
    # variance so it reads on the same scale as the eval `r2` column.
    loo_best = probe.loo_mse_.min(axis=0)
    cv_r2 = [1.0 - loo_best[j] / Ytr[:, j].var() for j in range(T)]

    rows = []
    for split in eval_splits:
        Xev, Yev, Fev = load_split(ds[split], targets, feature_col, frames_col)
        if Xev.shape[1] != Xtr.shape[1]:
            raise ValueError(f"{split} has D={Xev.shape[1]}, train has D={Xtr.shape[1]}")
        pred = np.atleast_2d(probe.predict(Xev))
        real, shuf = pred[:, :T], pred[:, T:2 * T]
        if has_len:
            len_pred = np.atleast_2d(len_model.predict(np.log1p(Fev)[:, None]))
            Rev, resid_pred = Yev - len_pred, pred[:, 2 * T:3 * T]

        for j, t in enumerate(targets):
            r, p = pearsonr(Yev[:, j], real[:, j])
            rho, _ = spearmanr(Yev[:, j], real[:, j])
            row = {
                "feature": feature_col, "target": t, "split": split,
                "train_on": "+".join(train_splits),
                "cv_r2": float(cv_r2[j]),
                "r2": _r2(Yev[:, j], real[:, j]),
                "pearson_r": r, "spearman_r": rho, "p_value": p,
                "rmse": float(np.sqrt(((Yev[:, j] - real[:, j]) ** 2).mean())),
                "r2_shuffled": _r2(Yev[:, j], shuf[:, j]),
                "r2_dummy": _r2(Yev[:, j], np.full(len(Yev), Ytr[:, j].mean())),
                "alpha": probe.alpha_[j], "dof": probe.dof_[j],
                "n_train": len(Xtr), "n_eval": len(Xev), "n_features": Xtr.shape[1],
            }
            row["selectivity"] = row["r2"] - row["r2_shuffled"]
            if has_len:
                row["r2_length_only"] = _r2(Yev[:, j], len_pred[:, j])
                row["r2_beyond_length"] = _r2(Rev[:, j], resid_pred[:, j])
            rows.append(row)

    return pd.DataFrame(rows)

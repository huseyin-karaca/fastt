__all__ = ['OBJECTIVE_WEIGHTS', 'TRANSFORM_REGISTRY', 'OBJECTIVES', 'wer_matrix', 'realised_wer', 'objective_weights', 'set_determinism', 'DiagonalGating', 'LinearTransform', 'LowRankTransform', 'NonlinearTransform', 'IdentityTransform', 'build_transform', 'AdaSTTObjective', 'AdaSTTXGBoost', 'FASTTAlternating', 'FASTTBoosted', 'SoftmaxProbeRouter', 'SoftProbeRouter', 'ValBestSingleRouter', 'WinRateRouter', 'PCAProjection', 'PCAPipeline', 'SelectKBestProjection', 'SelectKBestPipeline', 'ArgminClassifier', 'QERouter']

import copy

import numpy as np

# Imported here, before torch, and deliberately so. Both PyTorch and XGBoost
# ship their own OpenMP runtime, and on macOS loading torch's first makes the
# first `xgb.train` call segfault the interpreter — no exception, no traceback,
# exit 139. Importing XGBoost first is enough to avoid it. Colab is unaffected
# (one shared libomp), so this costs nothing there and saves an afternoon here.
# Guarded because §7 is the only section that needs XGBoost at all.
try:
    import xgboost as xgb
except ImportError:                                    # pragma: no cover
    xgb = None


def _require_xgboost():
    if xgb is None:
        raise ImportError("AdaSTT needs XGBoost — `pip install xgboost` (see §7).")
    return xgb


def wer_matrix(E, nref):
    """`(N, k)` error counts + `(N,)` reference words -> `(N, k)` per-utterance WER.

    The one conversion between the store's representation (§6.1) and the
    estimator contract. `float32` because that is what both XGBoost and torch
    want, and the extra precision buys nothing on a ratio of small integers.
    """
    return (np.asarray(E, dtype=np.float64)
            / np.maximum(np.asarray(nref, dtype=np.float64), 1)[:, None]).astype(np.float32)


def realised_wer(E, nref, choice):
    """Corpus WER actually incurred by routing to `choice`: sum(errors)/sum(words)."""
    picked = E[np.arange(len(E)), choice]
    return float(picked.sum() / nref.sum())


OBJECTIVE_WEIGHTS = ("uniform", "length")


def objective_weights(nref, mode="uniform", n=None):
    """Per-utterance training weights, normalised to mean 1.

    **This is a real modelling choice, not a knob.** Every estimator here
    minimises something of the form `mean_i sum_j p_ij WER_ij` where
    `WER_ij = E_ij / n_i`, while every metric reported in §7.3 is corpus WER,
    `sum_i E_ij / sum_i n_i`. Those are different objectives. Under `"uniform"`
    a one-word error on a six-word utterance costs the training objective about
    seven times what the identical error costs on a forty-word one; corpus WER
    charges them the same. The model is therefore optimising a quantity nobody
    reports.

    `"length"` weights utterance `i` by `n_i`, which cancels the per-utterance
    normalisation and leaves the objective proportional to total errors — i.e.
    exactly corpus WER. On our data the two differ by about 6x in relative weight
    between the shortest and longest quintiles, and the longest quintile carries
    40% of the words while `"uniform"` gives it 19% of the attention.

    `"uniform"` remains the default because it is what every number published so
    far was produced under, and changing a default silently is how results stop
    being comparable. Pass `"length"` to align the two, and report both.
    """
    if mode not in OBJECTIVE_WEIGHTS:
        raise ValueError(f"objective_weight must be one of {OBJECTIVE_WEIGHTS}, got {mode!r}")
    if mode == "uniform" or nref is None:
        if mode == "length":
            raise ValueError("objective_weight='length' needs `nref`; `run_cv` passes it, "
                             "but a direct `fit(X, y)` call does not")
        return None
    w = np.asarray(nref, dtype=np.float64)
    if n is not None and len(w) != n:
        raise ValueError(f"nref has {len(w)} rows but X has {n}")
    total = w.sum()
    if total <= 0:
        return None
    return (w * (len(w) / total)).astype(np.float64)


def _weighted_mean(values, weights):
    """`values` averaged under `weights`, which are already mean-1 or None."""
    return float(values.mean()) if weights is None else float((values * weights).mean())


def set_determinism(seed=0, strict=False):
    """Pin every RNG §7 can reach, and return the seed for logging.

    Covers Python's `random`, NumPy's legacy global state (which
    `FASTTAlternating.fit` sets), and torch on both CPU and CUDA. `strict=True`
    additionally forbids torch's nondeterministic kernels and fixes cuBLAS's
    workspace, which is what makes GPU runs bit-reproducible — at some cost in
    speed, and it raises if an op has no deterministic implementation.

    What this does NOT cover, and cannot: XGBoost's result depends on its thread
    count, so a run reproduces only at the same `nthread`. Pin it explicitly
    (`nthread=1` is the safe choice) rather than leaving the `-1` default, which
    resolves to whatever `os.cpu_count()` says on the machine of the day.
    """
    import os
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    return seed


def _rng_split(n, test_size, seed):
    """`(train_idx, test_idx)` from a seeded shuffle.

    Stands in for `sklearn.model_selection.train_test_split`, which the legacy
    estimators used for their internal early-stopping split. No install group in
    §1 carries scikit-learn and this is four lines, so the dependency is not
    worth taking on (same reasoning as `RidgeProbe` in §3.3).
    """
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(round(n * (1.0 - test_size)))
    return idx[:cut], idx[cut:]

import torch
import torch.nn as nn


class DiagonalGating(nn.Module):
    """Feature-wise scaling, `u = q * z`. Sparsity via combined l1/l2 on `q`."""

    def __init__(self, in_features, lambda1=0.01, lambda2=0.01):
        super().__init__()
        self.q = nn.Parameter(torch.ones(in_features))
        self.lambda1, self.lambda2 = lambda1, lambda2

    def forward(self, z):
        return self.q * z

    def regularization_loss(self):
        return self.lambda1 * self.q.abs().sum() + self.lambda2 * (self.q ** 2).sum()


class LinearTransform(nn.Module):
    """Full linear projection, `u = Wz + b`. Set `out_features < in_features` to reduce."""

    def __init__(self, in_features, out_features, weight_decay=1e-4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_decay = weight_decay

    def forward(self, z):
        return self.linear(z)

    def regularization_loss(self):
        return self.weight_decay * (self.linear.weight ** 2).sum()


class LowRankTransform(nn.Module):
    """Low-rank factorisation, `u = W2 W1 z`, with `r << min(p, d')`."""

    def __init__(self, in_features, out_features, bottleneck_dim, weight_decay=1e-4):
        super().__init__()
        self.w1 = nn.Linear(in_features, bottleneck_dim, bias=False)
        self.w2 = nn.Linear(bottleneck_dim, out_features, bias=False)
        self.weight_decay = weight_decay

    def forward(self, z):
        return self.w2(self.w1(z))

    def regularization_loss(self):
        return self.weight_decay * ((self.w1.weight ** 2).sum() + (self.w2.weight ** 2).sum())


class NonlinearTransform(nn.Module):
    """Nonlinear bottleneck, `u = W2 GELU(W1 z)`. The default in the shipped config."""

    def __init__(self, in_features, out_features, bottleneck_dim, weight_decay=1e-4):
        super().__init__()
        self.w1 = nn.Linear(in_features, bottleneck_dim)
        self.activation = nn.GELU()
        self.w2 = nn.Linear(bottleneck_dim, out_features, bias=False)
        self.weight_decay = weight_decay

    def forward(self, z):
        return self.w2(self.activation(self.w1(z)))

    def regularization_loss(self):
        return self.weight_decay * ((self.w1.weight ** 2).sum() + (self.w2.weight ** 2).sum())


class IdentityTransform(nn.Module):
    """No-op, i.e. the raw-features baseline. Has no parameters to update."""

    def forward(self, z):
        return z

    def regularization_loss(self):
        return torch.zeros(())


TRANSFORM_REGISTRY = {
    "identity": IdentityTransform,
    "diagonal": DiagonalGating,
    "linear": LinearTransform,
    "low_rank": LowRankTransform,
    "nonlinear": NonlinearTransform,
}


def build_transform(transform_type, in_features, out_features=None,
                    bottleneck_dim=None, **kwargs):
    """Factory for the transforms above. `out_features` defaults to `in_features`."""
    if transform_type not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform_type {transform_type!r}. "
                         f"Choose from {list(TRANSFORM_REGISTRY)}.")
    cls = TRANSFORM_REGISTRY[transform_type]
    out_features = in_features if out_features is None else out_features

    if transform_type == "identity":
        return cls()
    if transform_type == "diagonal":
        return cls(in_features=in_features, **kwargs)
    if transform_type == "linear":
        return cls(in_features=in_features, out_features=out_features, **kwargs)
    if bottleneck_dim is None:
        bottleneck_dim = max(in_features // 4, 8)
    return cls(in_features=in_features, out_features=out_features,
               bottleneck_dim=bottleneck_dim, **kwargs)

import os


class AdaSTTObjective:
    """Expected-WER gradient and Hessian for XGBoost's `obj` hook.

    `wer_matrix` is the `(n, k)` WER matrix for the rows in the DMatrix being
    trained on, in the same row order.
    """

    def __init__(self, wer_matrix, epsilon=0.02, weights=None):
        self.wer_matrix = wer_matrix
        self.epsilon = epsilon
        # Row weights are applied here rather than through `DMatrix(weight=...)`:
        # XGBoost scales gradients by the instance weight for its *built-in*
        # objectives, but a custom `obj` returns the final grad/hess and is taken
        # at its word. Setting the DMatrix weight and expecting it to reach this
        # function is a silent no-op — the exact kind that produces a clean run
        # and an unweighted model.
        self.weights = None if weights is None else np.asarray(weights, np.float64)

    def __call__(self, preds, dtrain):
        z = preds.reshape(self.wer_matrix.shape) if preds.ndim == 1 else preds
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        w = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        L = np.sum(w * self.wer_matrix, axis=1, keepdims=True)

        grad = w * (self.wer_matrix - L)
        # Floored because (1 - 2w) flips sign once a member holds > 1/2 the mass,
        # and XGBoost cannot take a step with a negative Hessian.
        hess = np.maximum(w * (1.0 - 2.0 * w) * (self.wer_matrix - L), self.epsilon)
        if self.weights is not None:
            # The floor is applied before scaling so a weighted Hessian stays
            # positive; scaling first and flooring after would clip the small
            # utterances back up and quietly undo the weighting.
            grad = grad * self.weights[:, None]
            hess = hess * self.weights[:, None]
        return grad.astype(np.float32).flatten(), hess.astype(np.float32).flatten()


OBJECTIVES = ("expected_wer", "cross_entropy")


def _check_objective(objective):
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    return objective


class AdaSTTXGBoost:
    """Gradient-boosted router trained directly on expected WER.

    Parameters are the `configs/model/*.yaml` defaults from the research
    project. `fit` holds out 20% of the rows internally for early stopping, so
    the caller passes the whole training half.

    `objective` selects what the trees descend:

    * `"expected_wer"` — the hand-derived `AdaSTTObjective` gradient of the
      softmax-expected WER, judged by `ExpectedWER` for early stopping.
    * `"cross_entropy"` — the **loss-ablation control**: XGBoost's own
      `multi:softprob` on the argmin-expert label (`argmin`'s first-minimum
      tie rule), early-stopped on `mlogloss`, **no** sample weights and no
      row filtering. The trees, the features, the split and the seed are
      identical; only what the model is told about WER changes — it learns
      *which* expert won and nothing about *by how much*. `objective_weight`
      is ignored on this path, deliberately: any WER-derived weight would turn
      the control back into a cost-sensitive classifier.
    """

    def __init__(self, n_estimators=1000, learning_rate=0.02, max_depth=8,
                 subsample=0.8, colsample_bytree=0.5, min_child_weight=4,
                 reg_lambda=1.0, early_stopping_rounds=50, nthread=-1,
                 random_state=42, objective_weight="uniform",
                 objective="expected_wer"):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_depth, self.subsample = max_depth, subsample
        self.colsample_bytree, self.min_child_weight = colsample_bytree, min_child_weight
        self.reg_lambda, self.early_stopping_rounds = reg_lambda, early_stopping_rounds
        self.nthread, self.random_state = nthread, random_state
        self.objective_weight = objective_weight
        self.objective = _check_objective(objective)
        self.model_ = None
        self.num_classes_ = None

    def fit(self, X, y, nref=None):
        xgb = _require_xgboost()

        self.num_classes_ = y.shape[1]
        tr, va = _rng_split(len(X), test_size=0.2, seed=self.random_state)
        X_train, Y_train, X_val, Y_val = X[tr], y[tr], X[va], y[va]

        if self.objective == "cross_entropy":
            return self._fit_cross_entropy(xgb, X_train, Y_train, X_val, Y_val)

        W = objective_weights(nref, self.objective_weight, n=len(X))
        W_train, W_val = (None, None) if W is None else (W[tr], W[va])

        # Dummy labels exist only so XGBoost can infer n_groups; the objective
        # below reads WER from the closure, never from the DMatrix label.
        rng = np.random.default_rng(self.random_state)
        dtrain = xgb.DMatrix(X_train, label=rng.integers(0, self.num_classes_, len(X_train)))
        dval = xgb.DMatrix(X_val, label=rng.integers(0, self.num_classes_, len(X_val)))
        wer_map = {id(dtrain): (Y_train, W_train), id(dval): (Y_val, W_val)}

        def eval_expected_wer(preds, dmat):
            z = preds.reshape(-1, self.num_classes_) if preds.ndim == 1 else preds
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            w = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            entry = wer_map.get(id(dmat))
            if entry is None:
                return "ExpectedWER", 0.0
            wers, weights = entry
            # Early stopping has to watch the same quantity the objective
            # descends, or a weighted fit is stopped by an unweighted judge.
            return "ExpectedWER", _weighted_mean(np.sum(w * wers, axis=1), weights)

        # `multi:softprob` is configured but overridden by `obj=` below; the
        # CE path keeps it and drops `obj=`, so the two share every other knob.
        params = {**self._xgb_params(), "disable_default_eval_metric": 1}
        self.model_ = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
            obj=AdaSTTObjective(Y_train, weights=W_train),
            evals=[(dtrain, "train"), (dval, "val")],
            custom_metric=eval_expected_wer,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        return self

    def _xgb_params(self):
        return {
            "num_class": self.num_classes_,
            "objective": "multi:softprob",
            "tree_method": "hist",
            "device": "cpu",
            "nthread": self.nthread if self.nthread > 0 else os.cpu_count(),
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda": self.reg_lambda,
            "seed": self.random_state,
        }

    def _fit_cross_entropy(self, xgb, X_train, Y_train, X_val, Y_val):
        """The CE control: built-in softmax cross-entropy on the argmin label.

        Three things are deliberately absent, and each would re-admit WER
        magnitudes if present: no `obj=` (so `multi:softprob` — softmax
        cross-entropy — is what XGBoost descends), no `DMatrix(weight=)`, and
        no custom metric (early stopping watches `mlogloss`, the loss being
        descended, not expected WER). The labels are REAL: `argmin` over the
        WER row, first minimum on ties.
        """
        dtrain = xgb.DMatrix(X_train, label=Y_train.argmin(axis=1))
        dval = xgb.DMatrix(X_val, label=Y_val.argmin(axis=1))
        params = {**self._xgb_params(), "eval_metric": "mlogloss"}
        self.model_ = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        return self

    def predict_proba(self, X):
        preds = self.model_.predict(_require_xgboost().DMatrix(X))
        if preds.ndim == 1:
            preds = preds.reshape(-1, self.num_classes_)
        exp_z = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def expected_wer(self, X, y):
        """Expected WER under the trees' routing distribution.

        A reporting method (`run_cv` records it as `soft_expected_wer`), so the
        soft-vs-hard gap is available for the XGBoost rungs of the loss ladder
        as it is for every other selector. Under `objective="cross_entropy"`
        it is a diagnostic of a distribution trained on something else — which
        is exactly the asymmetry the ablation reads.
        """
        return float((self.predict_proba(X) * np.asarray(y)).sum(axis=1).mean())

    def score(self, X, y):
        """Negative mean utterance WER — higher is better, per the sklearn convention."""
        return -y[np.arange(len(y)), self.predict(X)].mean()

class _LinearSurrogate(nn.Module):
    """Differentiable linear stand-in for the selector's `u -> logits` mapping."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, u):
        return self.linear(u)


class FASTTAlternating:
    """Alternating optimisation of a feature transform and a tree selector.

    Parameters
    ----------
    base_selector : an unfitted estimator following the §7.1 contract, deep-copied
        before each fit. Defaults to `AdaSTTXGBoost()`.
    transform_type : one of `TRANSFORM_REGISTRY`.
    num_iterations : alternating iterations T.
    transform_lr, transform_steps : the inner Adam loop for the transform update.
    surrogate_lr, surrogate_steps : the inner Adam loop that fits the surrogate.
    objective : `"expected_wer"` (the method) or `"cross_entropy"` (the
        loss-ablation control). The alternation has TWO WER-aware components —
        the selector's objective and the transform update's loss — and the
        control swaps **both**: the selector is fitted with `multi:softprob` on
        the argmin label and the transform descends cross-entropy through the
        surrogate on the same label. Swapping one and not the other would still
        optimise expected WER somewhere and prove nothing, so `_fit_selector`
        refuses a `base_selector` whose objective disagrees. The surrogate's MSE
        distillation fit is not a routing loss and is left alone. With
        `transform_type="identity"` the transform update is a no-op and the arm
        collapses to `adastt` under the same objective. Sample weights are
        not applied under cross-entropy.
    """

    def __init__(self, base_selector=None, transform_type="diagonal",
                 transform_kwargs=None, num_iterations=5, transform_lr=1e-3,
                 transform_steps=100, transform_weight_decay=1e-4,
                 transform_lambda1=0.01, transform_lambda2=0.01,
                 surrogate_lr=1e-2, surrogate_steps=200, device=None,
                 random_state=42, objective_weight="uniform",
                 objective="expected_wer"):
        self.objective_weight = objective_weight
        self.objective = _check_objective(objective)
        self.base_selector = base_selector
        self.transform_type, self.transform_kwargs = transform_type, transform_kwargs
        self.num_iterations = num_iterations
        self.transform_lr, self.transform_steps = transform_lr, transform_steps
        self.transform_weight_decay = transform_weight_decay
        self.transform_lambda1, self.transform_lambda2 = transform_lambda1, transform_lambda2
        self.surrogate_lr, self.surrogate_steps = surrogate_lr, surrogate_steps
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform_ = self.selector_ = self.num_classes_ = None
        self.nref_ = self.weights_ = None

    def _build_transform(self, in_features):
        """Merge the scalar penalty parameters into the transform's kwargs.

        `lambda1`/`lambda2` belong to `DiagonalGating` alone and `weight_decay` to
        everything else, so they cannot be passed unconditionally — the wrong one
        is an unexpected-keyword `TypeError`.
        """
        kwargs = dict(self.transform_kwargs or {})
        if self.transform_type == "diagonal":
            kwargs.setdefault("lambda1", self.transform_lambda1)
            kwargs.setdefault("lambda2", self.transform_lambda2)
        elif self.transform_type != "identity":
            kwargs.setdefault("weight_decay", self.transform_weight_decay)
        return build_transform(self.transform_type, in_features=in_features, **kwargs).to(self.device)

    def _fit_selector(self, u, y):
        """A fresh selector on the current transformed features. Trees run on CPU.

        `nref` is forwarded so the inner selector weights its objective the same
        way the transform update does — a half-weighted alternation optimises two
        different things on alternate steps and converges to neither.
        """
        selector = copy.deepcopy(self.base_selector or AdaSTTXGBoost(objective=self.objective))
        # Both swaps or neither: a selector descending a different objective
        # than the transform update is a half-swapped ablation.
        if getattr(selector, "objective", self.objective) != self.objective:
            raise ValueError(f"base_selector optimises {selector.objective!r}, "
                             f"the alternation {self.objective!r}")
        return selector.fit(u, y, nref=self.nref_)

    def _fit_surrogate(self, u_np, q_np):
        """Least-squares fit of the linear surrogate to the selector's logits."""
        surrogate = _LinearSurrogate(u_np.shape[1], self.num_classes_).to(self.device)
        u_t = torch.tensor(u_np, dtype=torch.float32, device=self.device)
        q_t = torch.tensor(q_np, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(surrogate.parameters(), lr=self.surrogate_lr)
        for _ in range(self.surrogate_steps):
            opt.zero_grad()
            nn.functional.mse_loss(surrogate(u_t), q_t).backward()
            opt.step()
        return surrogate

    def _update_transform(self, z_t, wer_t, surrogate):
        """Descend the objective + regularisation in theta, with psi held fixed.

        Expected WER by default; under `objective="cross_entropy"` the data
        term is the cross-entropy of the surrogate's logits against the argmin
        label (swap A of the ablation). Same graph, same Adam, same clipping.
        """
        params = list(self.transform_.parameters())
        if not params:                                   # identity transform
            return
        for p in surrogate.parameters():
            p.requires_grad_(False)
        label = wer_t.argmin(dim=1)

        opt = torch.optim.Adam(params, lr=self.transform_lr)
        for _ in range(self.transform_steps):
            opt.zero_grad()
            logits = surrogate(self.transform_(z_t))
            if self.objective == "cross_entropy":
                data = nn.functional.cross_entropy(logits, label)
            else:
                expected = (torch.softmax(logits, dim=1) * wer_t).sum(dim=1)
                if self.weights_ is not None:
                    expected = expected * self.weights_
                data = expected.mean()
            loss = data + self.transform_.regularization_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    def _transformed(self, z_t):
        with torch.no_grad():
            return self.transform_(z_t).cpu().numpy().astype(np.float32)

    def expected_wer(self, X, y):
        """Expected WER under the selector's own routing distribution.

        The quantity the alternating loop actually descends, and a smoother
        progress signal than realised WER because it does not pass through an
        argmax. Used as the per-iteration report for pruning.
        """
        return float((self.predict_proba(X) * y).sum(axis=1).mean())

    def fit(self, X, y, nref=None, callback=None):
        """Alternating optimisation.

        `callback(iteration, model)` runs after each alternating iteration, with
        the model in a fully usable state (selector and transform both fitted).
        Return value is ignored; raise from it to abort the fit early — which is
        how an Optuna pruner stops a hopeless trial without paying for the
        remaining iterations.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.num_classes_ = y.shape[1]
        self.nref_ = nref
        # The CE control never sees a WER-derived weight, not even a uniform
        # one computed from `nref` — the fit path must not touch `objective_weights`.
        W = (None if self.objective == "cross_entropy"
             else objective_weights(nref, self.objective_weight, n=len(X)))
        self.weights_ = (None if W is None else
                         torch.tensor(W, dtype=torch.float32, device=self.device))
        self.transform_ = self._build_transform(X.shape[1])
        z_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        wer_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        for iteration in range(self.num_iterations):
            u_np = self._transformed(z_t)
            self.selector_ = self._fit_selector(u_np, y)
            # Logits, not probabilities: the surrogate has to approximate the
            # pre-softmax scale or the gradients come out flattened.
            q_np = np.log(np.clip(self.selector_.predict_proba(u_np), 1e-10, None))
            self._update_transform(z_t, wer_t, self._fit_surrogate(u_np, q_np))
            if callback is not None:
                callback(iteration, self)

        # The loop leaves the selector one transform-update stale; refit it.
        self.selector_ = self._fit_selector(self._transformed(z_t), y)
        return self

    def _project(self, X):
        return self._transformed(torch.tensor(X, dtype=torch.float32, device=self.device))

    def predict_proba(self, X):
        return self.selector_.predict_proba(self._project(X))

    def predict(self, X):
        return self.selector_.predict(self._project(X))

    def score(self, X, y):
        """Negative mean utterance WER — higher is better."""
        return -y[np.arange(len(y)), self.predict(X)].mean()

    def feature_importances(self):
        """Per-input-dimension importance read off the transform's first layer."""
        if self.transform_type == "diagonal":
            return np.abs(self.transform_.q.detach().cpu().numpy())
        if self.transform_type == "linear":
            W = self.transform_.linear.weight
        elif self.transform_type in ("low_rank", "nonlinear"):
            W = self.transform_.w1.weight
        else:
            return None
        return torch.norm(W, p=2, dim=0).detach().cpu().numpy()

class _SDTR(nn.Module):
    """Soft decision tree ensemble with routing and L1 regularisation.

    `num_trees` trees of depth `depth`, each keeping a distribution over its
    `2**depth` leaves. Returns `(response, routing_penalty, l1_penalty)`.

    Parameters
    ----------
    tree_dim : outputs per leaf — the number of ensemble members to route between.
    lmbda : routing-balance strength; pushes splits towards 50/50.
    lmbda2 : L1 strength on the split layers.
    beta : initial inverse temperature of the sigmoid splits. Learned per node.
    """

    def __init__(self, in_features, num_trees, depth=4, tree_dim=1,
                 lmbda=0.1, lmbda2=0.01, beta=1.5, init_func="zero", eps=1e-6):
        super().__init__()
        self.depth, self.num_trees, self.tree_dim = depth, num_trees, tree_dim
        self.in_features = in_features
        self.lmbda, self.lmbda2, self.eps = lmbda, lmbda2, eps

        self.response = nn.Parameter(torch.zeros(num_trees, tree_dim, 2 ** depth))
        # Legacy deferred this to the first forward pass through a lazy-init base
        # class, but it never depended on the data, so it belongs here.
        if init_func == "uniform":
            nn.init.uniform_(self.response, -1, 1)
        elif init_func == "xuniform":
            nn.init.xavier_uniform_(self.response)
        elif init_func == "normal":
            nn.init.normal_(self.response)

        self.splits = nn.ModuleList()
        self.betas = nn.ParameterList()
        for level in range(depth):
            layer = nn.Linear(in_features, num_trees * (2 ** level))
            nn.init.xavier_uniform_(layer.weight)
            self.splits.append(layer)
            self.betas.append(nn.Parameter(torch.full((num_trees, 2 ** level), float(beta))))

    def forward(self, x):
        batch = x.size(0)
        path = torch.ones(batch, self.num_trees, 1, device=x.device, dtype=x.dtype)
        routing = torch.zeros((), device=x.device, dtype=x.dtype)
        l1 = torch.zeros((), device=x.device, dtype=x.dtype)
        reweight = 1.0

        for level, (layer, beta) in enumerate(zip(self.splits, self.betas)):
            logit = layer(x).view(batch, self.num_trees, -1)
            l1 = l1 + layer.weight.norm(1) / self.num_trees * reweight * self.lmbda2
            reweight /= 2

            go_left = torch.sigmoid(torch.einsum("bnc,nc->bnc", logit, beta))

            # Mean "go left" mass at each node, over the batch. Clamped because
            # a saturated node gives exactly 0 or 1 and log() would return -inf,
            # which then poisons the whole summed penalty.
            balance = (torch.einsum("bnc,bnc->nc", go_left, path)
                       / (path.sum(dim=0) + 1e-4)).clamp(self.eps, 1 - self.eps)
            routing = routing - (self.lmbda * 2 ** (-level) * 0.5
                                 * torch.log(balance * (1 - balance)).mean())

            left, right = go_left * path, (1 - go_left) * path
            path = torch.stack((left.unsqueeze(-1), right.unsqueeze(-1)), dim=3).view(
                batch, self.num_trees, 2 ** (level + 1))

        # Average the leaf responses under the path distribution, then sum trees.
        out = torch.einsum("bnd,ncd->bnc", path, self.response).sum(dim=1)
        return out, routing, l1


class _FASTTBoostedNet(nn.Module):
    """`num_rounds` (transform, soft-tree) pairs, summed in logit space."""

    def __init__(self, in_features, num_classes, num_rounds=3,
                 transform_type="diagonal", transform_kwargs=None,
                 num_trees=10, depth=4, lmbda=0.1, lmbda2=0.01):
        super().__init__()
        self.transforms, self.trees = nn.ModuleList(), nn.ModuleList()

        for _ in range(num_rounds):
            transform = build_transform(transform_type, in_features=in_features,
                                        **(transform_kwargs or {}))
            # How wide the transform leaves the features decides the tree's input.
            if hasattr(transform, "linear"):
                out_dim = transform.linear.out_features
            elif hasattr(transform, "w2"):
                out_dim = transform.w2.out_features
            else:                                    # identity, diagonal
                out_dim = in_features
            self.transforms.append(transform)
            self.trees.append(_SDTR(out_dim, num_trees=num_trees, depth=depth,
                                    tree_dim=num_classes, lmbda=lmbda, lmbda2=lmbda2))

    def forward(self, z):
        logits = None
        reg = torch.zeros((), device=z.device, dtype=z.dtype)
        for transform, tree in zip(self.transforms, self.trees):
            out, routing, l1 = tree(transform(z))
            logits = out if logits is None else logits + out
            reg = reg + routing + l1 + transform.regularization_loss().to(z.device)
        return logits, reg


class FASTTBoosted:
    """Boosted soft decision trees with per-round learnable transforms.

    The differentiable counterpart to `FASTTAlternating`: one end-to-end
    optimisation of expected WER, no surrogate gradient anywhere.

    Parameters
    ----------
    num_rounds : boosting rounds B, each a (transform, tree ensemble) pair.
    num_trees, depth : shape of each round's ensemble. Cost grows as
        `2**depth`, so raise `num_trees` before `depth`.
    lmbda, lmbda2 : routing-balance and L1 strengths inside the trees.
    lr, weight_decay : AdamW.
    batch_size, epochs, patience : the training loop; `patience` counts epochs
        without improvement on the internal validation fifth.
    val_frac : share of the training rows held out for early stopping.
    objective : `"expected_wer"` (the method) or `"cross_entropy"` (the
        loss-ablation control: cross-entropy of the logits against the argmin
        label, first minimum on ties, no sample weights). Everything else —
        the transforms, the trees, their regularisers, AdamW, the schedule,
        early stopping, the seed — is byte-identical. Early stopping and
        `history_` follow the objective, so histories of the two are not on
        one scale; `expected_wer()` stays expected WER either way, as the
        soft-vs-hard diagnostic.
    """

    def __init__(self, num_rounds=3, transform_type="diagonal", transform_kwargs=None,
                 num_trees=10, depth=4, lmbda=0.1, lmbda2=0.01,
                 transform_weight_decay=1e-4, transform_lambda1=0.01,
                 transform_lambda2=0.01, lr=1e-3, weight_decay=1e-4,
                 batch_size=512, epochs=100, patience=20, val_frac=0.2,
                 grad_clip=1.0, device=None, random_state=42,
                 objective_weight="uniform", objective="expected_wer"):
        self.objective_weight = objective_weight
        self.objective = _check_objective(objective)
        self.num_rounds, self.transform_type = num_rounds, transform_type
        self.transform_kwargs = transform_kwargs
        self.num_trees, self.depth = num_trees, depth
        self.lmbda, self.lmbda2 = lmbda, lmbda2
        self.transform_weight_decay = transform_weight_decay
        self.transform_lambda1, self.transform_lambda2 = transform_lambda1, transform_lambda2
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.val_frac, self.grad_clip = val_frac, grad_clip
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = self.num_classes_ = None
        self.history_ = []

    def _transform_kwargs(self):
        """Same lambda/weight-decay routing as `FASTTAlternating._build_transform`."""
        kwargs = dict(self.transform_kwargs or {})
        if self.transform_type == "diagonal":
            kwargs.setdefault("lambda1", self.transform_lambda1)
            kwargs.setdefault("lambda2", self.transform_lambda2)
        elif self.transform_type != "identity":
            kwargs.setdefault("weight_decay", self.transform_weight_decay)
        return kwargs

    def _loss(self, logits, reg, wer, weights=None):
        """The data term under `objective`, plus the model's own regularisers.

        Expected WER under the softmax, or — for the control — cross-entropy
        against `wer.argmin(1)`. `weights` are mean-1 per-utterance weights
        (§7.1). A weighted mean rather than a sum keeps the loss on the same
        scale as the unweighted one, so the learning rate and `patience` do not
        have to be re-tuned when the weighting changes — which is the difference
        between an ablation and a confound. The regulariser tail is shared: it
        is part of the architecture, not of the loss under test.
        """
        if self.objective == "cross_entropy":
            data = nn.functional.cross_entropy(logits, wer.argmin(dim=1), reduction="none")
        else:
            data = (torch.softmax(logits, dim=1) * wer).sum(dim=1)
        if weights is not None:
            data = data * weights
        # Belt and braces: the clamp inside `_SDTR` should make this a no-op, but
        # a diverged run can still produce a non-finite penalty and this keeps the
        # data term usable rather than turning the whole loss into nan.
        return data.mean() + torch.nan_to_num(reg, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y, nref=None, callback=None):
        """Train end-to-end. `callback(epoch, model)` runs after each epoch.

        Raise from `callback` to stop early — that is how an Optuna pruner drops
        a hopeless trial without paying for the remaining epochs.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.num_classes_ = y.shape[1]
        tr, va = _rng_split(len(X), self.val_frac, seed=self.random_state)

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        Xtr, Ytr, Xva, Yva = t(X[tr]), t(y[tr]), t(X[va]), t(y[va])
        # The CE control is unweighted by definition; see `AdaSTTXGBoost`.
        W = (None if self.objective == "cross_entropy"
             else objective_weights(nref, self.objective_weight, n=len(X)))
        Wtr, Wva = (None, None) if W is None else (t(W[tr]), t(W[va]))

        self.model_ = _FASTTBoostedNet(
            in_features=X.shape[1], num_classes=self.num_classes_,
            num_rounds=self.num_rounds, transform_type=self.transform_type,
            transform_kwargs=self._transform_kwargs(), num_trees=self.num_trees,
            depth=self.depth, lmbda=self.lmbda, lmbda2=self.lmbda2).to(self.device)

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                           patience=3, factor=0.5)
        # Batch order comes from a generator we own, not the global torch RNG, so
        # the epoch sequence is reproducible independently of anything else drawn.
        gen = np.random.default_rng(self.random_state)
        best, best_state, waited = float("inf"), None, 0

        for epoch in range(self.epochs):
            self.model_.train()
            order = gen.permutation(len(Xtr))
            for start in range(0, len(order), self.batch_size):
                idx = torch.as_tensor(order[start:start + self.batch_size],
                                      device=self.device, dtype=torch.long)
                opt.zero_grad()
                logits, reg = self.model_(Xtr[idx])
                self._loss(logits, reg, Ytr[idx],
                           None if Wtr is None else Wtr[idx]).backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                logits, reg = self.model_(Xva)
                val = float(self._loss(logits, reg, Yva, Wva))
            sched.step(val)
            self.history_.append(val)

            if val < best:
                best, waited = val, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                waited += 1
            if callback is not None:
                callback(epoch, self)
            if waited >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _logits(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_(torch.tensor(X, dtype=torch.float32, device=self.device))
        return logits

    def predict_proba(self, X):
        return torch.softmax(self._logits(X), dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def expected_wer(self, X, y):
        """Expected WER under the model's routing distribution — the smooth signal."""
        return float((self.predict_proba(X) * y).sum(axis=1).mean())

    def score(self, X, y):
        """Negative mean utterance WER — higher is better."""
        return -y[np.arange(len(y)), self.predict(X)].mean()

    def feature_importances(self):
        """Per-round input importance, read off each round's transform."""
        out = []
        for transform in self.model_.transforms:
            if self.transform_type == "diagonal":
                out.append(np.abs(transform.q.detach().cpu().numpy()))
            elif self.transform_type == "linear":
                out.append(torch.norm(transform.linear.weight, p=2, dim=0)
                           .detach().cpu().numpy())
            elif self.transform_type in ("low_rank", "nonlinear"):
                out.append(torch.norm(transform.w1.weight, p=2, dim=0)
                           .detach().cpu().numpy())
        return out or None

from fastt.routing import SoftmaxProbe


class SoftmaxProbeRouter:
    """Multinomial logistic router over the standardised design matrix (§6.5).

    **Deterministic**: full-batch L-BFGS from a zero initialisation, so two
    instances fitted on the same fold are bit-identical. It therefore cannot
    serve as the stochastic twin in a null-calibration check — see §7.6.
    """

    def __init__(self, alpha=1.0, max_iter=200, device=None,
                 objective_weight="uniform"):
        self.alpha, self.max_iter, self.device = alpha, max_iter, device
        self.objective_weight = objective_weight
        self.probe_ = None

    def fit(self, X, y, nref=None):
        label, weight = y.argmin(axis=1), y.max(axis=1) - y.min(axis=1)
        W = objective_weights(nref, self.objective_weight, n=len(X))
        if W is not None:
            # `weight` is already the WER spread, i.e. what routing this utterance
            # right is worth per word; multiplying by the word count makes it what
            # it is worth in words, which is the unit corpus WER counts in.
            weight = weight * W
        rows = weight > 0                       # contested utterances only
        if not rows.any():
            raise ValueError("no contested utterances — every member ties everywhere")
        self.probe_ = SoftmaxProbe(self.alpha, self.max_iter, self.device).fit(
            X[rows], label[rows], weight[rows])
        return self

    def decision_function(self, X):
        return self.probe_.decision_function(X)

    def predict(self, X):
        return self.decision_function(X).argmax(axis=1)

    def expected_wer(self, X, y, beta=1.0):
        """Expected WER under `softmax(beta * scores)` — the soft reading of §6.5.

        Reported by `run_cv` as `soft_expected_wer` beside the hard WER the
        argmax realises. `SoftProbeRouter` below turns the same quantity into a
        competing **arm** rather than a diagnostic column.
        """
        w = _softmax(beta * self.decision_function(X))
        return float((w * np.asarray(y)).sum(axis=1).mean())

from fastt.ensemble import win_rates


def _softmax(S):
    """Row-wise softmax, shifted by the row max for numerical stability."""
    Z = np.exp(np.asarray(S, dtype=np.float64) - np.max(S, axis=1, keepdims=True))
    return Z / Z.sum(axis=1, keepdims=True)


class SoftProbeRouter:
    """Soft counterpart of `SoftmaxProbeRouter`: hedge over members, do not commit.

    Parameters
    ----------
    errors : the store's `(N, k)` error-count matrix, in **store row order** —
        `store.err(members)`. Indexed by the `rows` the fold hands over, so the
        arm never sees the evaluation half's labels through `Y`.
    beta : inverse temperature on the probe's scores. Larger commits harder;
        `beta=0` is uniform mixing, i.e. the `random` arm's expectation.
    alpha : the underlying probe's ridge penalty, as in `SoftmaxProbeRouter`.

    `produces = "errors"` because no member is selected: the arm's cost on an
    utterance belongs to a mixture and not to any one column of `E`.
    """

    produces = "errors"
    wants_rows = True

    def __init__(self, errors, beta=1.0, alpha=1.0, max_iter=200, device=None,
                 objective_weight="uniform"):
        self.errors = np.asarray(errors, dtype=np.float64)
        self.beta = float(beta)
        self.hard = SoftmaxProbeRouter(alpha=alpha, max_iter=max_iter, device=device,
                                       objective_weight=objective_weight)

    def fit(self, X, y, nref=None, rows=None):
        self.hard.fit(X, y, nref=nref)
        return self

    def predict(self, X, rows=None):
        if rows is None:
            raise ValueError("SoftProbeRouter needs `rows` to reach the error matrix")
        w = _softmax(self.beta * self.hard.decision_function(X))
        return (w * self.errors[np.asarray(rows, dtype=int)]).sum(axis=1)


class ValBestSingleRouter:
    """Always route to the member that wins on a held-out slice of the fit data.

    Parameters
    ----------
    val_frac : share of the training half spent on choosing the member. Larger
        is a more reliable choice made from less data; the default quarter is
        the same split `HoldoutSplit` uses.
    random_state : which slice. Two seeds can genuinely pick different members
        on a close call, which is the variance this arm exists to expose.

    Contrast with the `bsm` reference arm, which picks on the **whole** training
    half. `bsm` is the right yardstick for "did routing beat standing still";
    this is the right yardstick for "did routing beat the model-selection
    procedure a practitioner would actually run".
    """

    def __init__(self, val_frac=0.25, random_state=0):
        if not 0 < val_frac < 1:
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
        self.val_frac, self.random_state = float(val_frac), random_state
        self.member_ = None

    def fit(self, X, y, nref=None):
        # `Y` is `E` divided row-wise by `nref` (§7.1), so this recovers the
        # error counts corpus WER is defined on. Without it the arm would be
        # minimising a per-utterance mean, i.e. answering a different question
        # from the one `bsm` answers, and the comparison would be rigged.
        y = np.asarray(y, dtype=np.float64)
        E = y if nref is None else y * np.asarray(nref, dtype=np.float64)[:, None]
        n = len(E)
        cut = int(np.clip(round(n * self.val_frac), 1, n - 1))
        val = np.random.default_rng(self.random_state).permutation(n)[:cut]
        words = 1.0 if nref is None else np.asarray(nref)[val].sum()
        self.member_ = int((E[val].sum(axis=0) / max(words, 1e-12)).argmin())
        return self

    def predict(self, X):
        return np.full(len(X), self.member_, dtype=int)


class WinRateRouter:
    """Always route to the member with the highest win rate on the fit data.

    `sole=True` ranks on outright wins only, ignoring ties; `sole=False` (the
    default) shares each tied row equally between its winners, which is
    `win_rates`' first return value and the rate §6.2 reports.

    Parameter-free in everything that matters, and deliberately so: it is the
    "popularity" baseline, and the interesting question is only ever whether it
    agrees with `bsm`.
    """

    def __init__(self, sole=False):
        self.sole = bool(sole)
        self.member_ = None

    def fit(self, X, y, nref=None):
        # Ties and argmins are identical in `Y` and in `E` — the two differ by a
        # strictly positive per-ROW factor, which cannot reorder a row — so the
        # WER matrix serves here and no error counts are needed.
        share, sole = win_rates(np.asarray(y, dtype=np.float64))
        self.member_ = int((sole if self.sole else share).argmax())
        return self

    def predict(self, X):
        return np.full(len(X), self.member_, dtype=int)

def _wrap_callback(wrapper, callback):
    """Re-point an iteration callback at the WRAPPER rather than the inner model.

    A pruning callback is called by the inner estimator as `callback(step, self)`
    and is then handed the *outer* validation matrix by whoever installed it. The
    two only agree if the object it receives is the one that knows about the
    projection, so the inner model is swapped out for the wrapper here.
    """
    import inspect

    if "callback" not in inspect.signature(wrapper.model_.fit).parameters:
        raise TypeError(
            f"{type(wrapper.model_).__name__} does not accept a `callback`, so a "
            f"{type(wrapper).__name__} around it cannot be pruned on its "
            f"iterations. Register the arm with ModelSpec(..., prunable=False).")
    return lambda step, _inner: callback(step, wrapper)


class PCAProjection:
    """Truncated PCA, fitted per fold. `fit(X)` then `transform(X)`.

    Implemented on the covariance eigendecomposition rather than an SVD of `X`:
    the design matrix is roughly 8000 x 1800, so the covariance is the smaller of
    the two and `eigh` on 1800 x 1800 is milliseconds. `X` arrives standardised
    from the fold, so this is a correlation-matrix PCA.
    """

    def __init__(self, n_components=256, whiten=False):
        self.n_components, self.whiten = int(n_components), whiten
        self.mean_ = self.components_ = self.explained_ = None

    def fit(self, X, y=None, nref=None):
        X = np.asarray(X, np.float64)
        k = min(self.n_components, X.shape[1], max(len(X) - 1, 1))
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)                  # ascending
        order = np.argsort(vals)[::-1][:k]
        self.explained_ = np.clip(vals[order], 0, None)
        self.components_ = vecs[:, order]
        return self

    def transform(self, X):
        Z = (np.asarray(X, np.float64) - self.mean_) @ self.components_
        if self.whiten:
            Z = Z / np.sqrt(np.maximum(self.explained_, 1e-12))
        return Z

    def fit_transform(self, X, y=None, nref=None):
        return self.fit(X).transform(X)

    @property
    def explained_variance_ratio_(self):
        """Share of the standardised features' total variance kept, per component."""
        return self.explained_ / max(self.explained_.sum(), 1e-12)


class PCAPipeline:
    """`PCAProjection` then any §7.1 estimator — the "is it just compression?" arm.

    Parameters
    ----------
    base : a **factory**, called once per fit, exactly like the arms in `run_cv`.
    n_components : projection width. Match it to what the tuned FASTT transform
        actually uses (`out_ratio x n_features`) or the comparison is not like
        for like.
    """

    def __init__(self, base=None, n_components=256, whiten=False):
        self.base, self.n_components, self.whiten = base, int(n_components), whiten
        self.pca_ = self.model_ = None

    def fit(self, X, y, nref=None, callback=None, **kw):
        """Fit the projection on the training half, then the wrapped estimator.

        `callback` is forwarded, but **wrapped**: the tuner hands the callback
        the outer, unprojected design matrix, so what it is given has to be an
        object that projects before it predicts. That is `self`, not the inner
        model — reporting the inner model would call `expected_wer` with the
        full feature width on an estimator fitted on the reduced one, and the
        shape error surfaces inside Optuna's trial loop rather than here.

        `self.model_` is therefore assigned **before** the inner fit runs, so
        `__getattr__` can resolve `expected_wer` while the callback is firing.
        """
        self.pca_ = PCAProjection(self.n_components, self.whiten).fit(X)
        self.model_ = (self.base or (lambda: AdaSTTXGBoost()))()
        if callback is not None:
            kw["callback"] = _wrap_callback(self, callback)
        self.model_.fit(self.pca_.transform(X), y, nref=nref, **kw)
        return self

    def predict(self, X):
        return self.model_.predict(self.pca_.transform(X))

    def score(self, X, y):
        return -y[np.arange(len(y)), self.predict(X)].mean()

    def __getattr__(self, name):
        """Mirror the wrapped estimator's optional capabilities, rather than claim them.

        §7.3 decides whether to record the soft-vs-hard gap with
        `hasattr(model, "expected_wer")`. A wrapper that defines the method
        unconditionally turns "this estimator has no routing distribution" into
        an `AttributeError` raised three frames inside the fold loop. Resolving
        the name against the wrapped model instead makes `hasattr` tell the
        truth — and it has to consult the wrapped model for BOTH routes to the
        quantity, since an estimator may expose `expected_wer` directly
        (`SoftmaxProbeRouter`, over its own softmax) or only `predict_proba`,
        from which it is derived.
        """
        if name in ("predict_proba", "expected_wer"):
            base = self.__dict__.get("model_")
            if base is not None:
                if hasattr(base, "predict_proba"):
                    proba = lambda X: base.predict_proba(self.pca_.transform(X))
                    if name == "predict_proba":
                        return proba
                    return lambda X, y: float((proba(X) * y).sum(axis=1).mean())
                if name == "expected_wer" and hasattr(base, "expected_wer"):
                    return lambda X, y: base.expected_wer(self.pca_.transform(X), y)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

from fastt.routing import select_k_best


class SelectKBestProjection:
    """Univariate ANOVA-F column filter, fitted per fold. `fit(X, y)` then `transform`.

    `y` is the (N, k) WER matrix, so the ranking target is the **argmin member** —
    the label a selector is actually asked to predict. `task="classification"`
    is therefore the right reduction and not merely the default: ranking against
    the raw WER columns would score a feature by how well it predicts *how badly*
    each member does, which is a different and easier question.

    Parameters
    ----------
    k : how many columns to keep. `None`, or a `k` at least as large as the
        input width, keeps everything and makes this a no-op — which is what
        makes `k` searchable over a range that includes "no filtering at all".
    """

    def __init__(self, k=256):
        self.k = None if k is None else int(k)
        self.columns_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, np.float64)
        if self.k is None or self.k >= X.shape[1] or y is None:
            self.columns_ = np.arange(X.shape[1])
            return self
        labels = np.asarray(y).reshape(len(X), -1).argmin(axis=1)
        self.columns_ = select_k_best(X, labels, self.k, task="classification")
        return self

    def transform(self, X):
        return np.asarray(X, np.float64)[:, self.columns_]

    @property
    def n_components(self):
        return len(self.columns_) if self.columns_ is not None else self.k


class SelectKBestPipeline:
    """`SelectKBestProjection` then any §7.1 estimator — the classical baseline.

    Deliberately the same interface as `PCAPipeline`, including the `__getattr__`
    that mirrors rather than claims the wrapped estimator's optional
    capabilities, so the two controls can be swapped in a table without any
    other code noticing.

    Parameters
    ----------
    base : a **factory**, called once per fit, exactly like the arms in `run_cv`.
        `AdaSTTXGBoost` gives the manuscript's `SelectKBest-XGBoost`; a
        `FASTTBoosted(transform_type="identity")` factory gives `SelectKBest-SDT`.
    k : columns kept by the filter.
    """

    def __init__(self, base=None, k=256):
        self.base, self.k = base, k
        self.filter_ = self.model_ = None

    def fit(self, X, y, nref=None, callback=None, **kw):
        """Fit the projection on the training half, then the wrapped estimator.

        `callback` is forwarded, but **wrapped**: the tuner hands the callback
        the outer, unprojected design matrix, so what it is given has to be an
        object that projects before it predicts. That is `self`, not the inner
        model — reporting the inner model would call `expected_wer` with the
        full feature width on an estimator fitted on the reduced one, and the
        shape error surfaces inside Optuna's trial loop rather than here.

        `self.model_` is therefore assigned **before** the inner fit runs, so
        `__getattr__` can resolve `expected_wer` while the callback is firing.
        """
        # Fitted on this fold's training half only — see the warning above.
        self.filter_ = SelectKBestProjection(self.k).fit(X, y)
        self.model_ = (self.base or (lambda: AdaSTTXGBoost()))()
        if callback is not None:
            kw["callback"] = _wrap_callback(self, callback)
        self.model_.fit(self.filter_.transform(X), y, nref=nref, **kw)
        return self

    def predict(self, X):
        return self.model_.predict(self.filter_.transform(X))

    def score(self, X, y):
        return -y[np.arange(len(y)), self.predict(X)].mean()

    def __getattr__(self, name):
        """Mirror the wrapped estimator's optional capabilities. See `PCAPipeline`."""
        if name in ("predict_proba", "expected_wer"):
            base = self.__dict__.get("model_")
            flt = self.__dict__.get("filter_")
            if base is not None and flt is not None:
                if hasattr(base, "predict_proba"):
                    proba = lambda X: base.predict_proba(flt.transform(X))
                    if name == "predict_proba":
                        return proba
                    return lambda X, y: float((proba(X) * y).sum(axis=1).mean())
                if name == "expected_wer" and hasattr(base, "expected_wer"):
                    return lambda X, y: base.expected_wer(flt.transform(X), y)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

from fastt.probing import RidgeProbe


def _pad_scores(S, k):
    """Widen an `(n, c)` score matrix to `(n, k)` when a class never appeared.

    `SoftmaxProbe` sizes itself from `label.max() + 1`, so an expert that is
    never the argmin on the training half produces a narrower matrix than there
    are members. `predict` still returns valid member indices, but anything that
    lines the scores up against the `(n, k)` WER matrix — `expected_wer`, and so
    the `soft_expected_wer` column — would broadcast wrongly or raise. Missing
    classes are padded with `-inf`, i.e. "never chosen", which is what the fit
    actually concluded about them.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.shape[1] >= k:
        return S
    pad = np.full((len(S), k - S.shape[1]), -np.inf)
    return np.hstack([S, pad])


class ArgminClassifier:
    """Plain multinomial logistic regression on the argmin-expert label.

    **The zero-WER-information control.** It is told which expert was best and
    nothing whatever about by how much, which is the whole point: the difference
    between this and `SoftmaxProbeRouter` prices the cost-sensitive weighting,
    and the difference between it and `AdaSTTXGBoost` prices the WER-aware
    objective at fixed feature access.

    Three deliberate departures from `SoftmaxProbeRouter`, each of which would
    smuggle WER magnitudes back in:

    * **no sample weights** — not the WER spread, not `objective_weights`;
    * **no row filtering** — uncontested utterances are kept, where
      `SoftmaxProbeRouter` drops them, because "is this utterance contested" is
      itself a fact about the magnitudes;
    * **ties go to the first minimum**, `argmin`'s own rule. Deterministic, and
      no tie-break that could depend on the size of the gap.

    Parameters
    ----------
    alpha : L2 penalty, as in `SoftmaxProbeRouter`. The only knob, which is the
        honest shape for a control: an arm that needed careful tuning to look
        bad would not be evidence of anything.
    """

    def __init__(self, alpha=1.0, max_iter=200, device=None):
        self.alpha, self.max_iter, self.device = alpha, max_iter, device
        self.probe_ = None
        self.n_members_ = None

    def fit(self, X, y, nref=None):
        """`nref` is accepted and ignored — this arm is not allowed to use it."""
        y = np.asarray(y, dtype=np.float64)
        self.n_members_ = y.shape[1]
        label = y.argmin(axis=1)
        self.probe_ = SoftmaxProbe(self.alpha, self.max_iter, self.device).fit(X, label)
        return self

    def decision_function(self, X):
        return _pad_scores(self.probe_.decision_function(X), self.n_members_)

    def predict(self, X):
        return self.decision_function(X).argmax(axis=1)

    def expected_wer(self, X, y, beta=1.0):
        """Expected WER under `softmax(beta * scores)`, for the soft-vs-hard gap.

        Reported by `run_cv` as `soft_expected_wer`. Note this is a *diagnostic*
        here and not the trained objective — that is the asymmetry the ablation
        is built to expose.
        """
        w = _softmax(beta * self.decision_function(X))
        return float((w * np.asarray(y)).sum(axis=1).mean())


class QERouter:
    """Predict every expert's WER, route to the lowest. The QE baseline.

    Regression where `ArgminClassifier` classifies: it estimates `WER_ij` for
    each expert and takes the `argmin` of the estimates, which is the
    quality-estimation router, the natural comparison for
    confidence-based routing. No decoder is executed and no confidence score is
    read off a lattice — the estimate comes from the same pooled encoder
    features every other arm sees.

    It is closer to FASTT's objective than a classifier is (it models the
    magnitudes) while still being trained on a **surrogate** rather than on the
    routing decision itself: squared error on WER, not expected WER under a
    routing distribution. That places it between `argmin_clf` and `adastt` in
    the ladder.

    Parameters
    ----------
    target : `"wer"` regresses the per-utterance WER matrix directly.
        `"errors"` regresses the error *counts* — `WER_ij * nref_i` — which
        weights long utterances in proportion to the words they contribute and
        so aligns the fit with corpus WER rather than with the macro average.
        The same choice `objective_weights` documents for every other arm, in
        the form a regressor can take it.
    alpha_lo, alpha_decades, n_alphas : the ridge path handed to `RidgeProbe`,
        which picks a penalty per target by exact leave-one-out CV. Expressed as
        a floor plus a width so the range cannot be inverted by the sampler.
    standardize : whether `RidgeProbe` rescales the columns before the SVD.
    """

    def __init__(self, target="wer", alpha_lo=1e-2, alpha_decades=8, n_alphas=20,
                 standardize=True):
        self.target, self.standardize = target, bool(standardize)
        self.alpha_lo = float(alpha_lo)
        self.alpha_decades = int(alpha_decades)
        self.n_alphas = int(n_alphas)
        self.probe_ = None

    def _alphas(self):
        return np.logspace(np.log10(self.alpha_lo),
                           np.log10(self.alpha_lo) + self.alpha_decades,
                           self.n_alphas)

    def fit(self, X, y, nref=None):
        Y = np.asarray(y, dtype=np.float64)
        if self.target == "errors":
            if nref is None:
                raise ValueError("target='errors' needs `nref`; run_cv passes it")
            Y = Y * np.asarray(nref, dtype=np.float64)[:, None]
        elif self.target != "wer":
            raise ValueError(f"target must be 'wer' or 'errors', got {self.target!r}")
        self.probe_ = RidgeProbe(alphas=self._alphas(),
                                 standardize=self.standardize).fit(X, Y)
        return self

    def predicted_wer(self, X):
        """`(n, k)` predicted cost per expert. Not clipped — only the order matters."""
        return np.asarray(self.probe_.predict(X), dtype=np.float64)

    def predict(self, X):
        return self.predicted_wer(X).argmin(axis=1)

    def expected_wer(self, X, y, beta=1.0):
        """Expected WER under `softmax(-beta * predicted cost)`.

        Negated because this arm scores by cost, where every other arm scores by
        preference — routing to the `argmin` of a cost is the `argmax` of its
        negation, and the soft reading has to agree with the hard one.
        """
        w = _softmax(-beta * self.predicted_wer(X))
        return float((w * np.asarray(y)).sum(axis=1).mean())

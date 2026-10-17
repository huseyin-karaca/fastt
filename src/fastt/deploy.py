__all__ = ['FittedSelector', 'fit_selector', 'RouterOutput', 'Router', 'RoverRouter', 'fit_rover', 'load_extractors', 'compare_routers']

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fastt.crossval import build_design, wants_rows
from fastt.estimators import wer_matrix
from fastt.latency import EXTRACTORS, install_group, load_extractor
from fastt.rover import RoverFusion, build_network, vote
from fastt.tuning import MODELS, BuildContext


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class FittedSelector:
    """A selector fitted once, plus everything needed to apply it to live audio.

    Built by `fit_selector`; applied by `predict` (features in hand) or through
    a `Router` (audio in hand). The design matrix is assembled exactly as
    `crossval.build_design` assembles it — members in order, then the
    feature-only blocks, then `log1p(n_frames)` — and standardised with the
    mean and sd of the rows it was fitted on, which is what `run_cv` does
    inside a fold.

    Attributes
    ----------
    arm, params : the `MODELS` entry and the parameter dict it was built from.
    members : the routable members, in the order the estimator's output indexes.
    feature_only : encoders whose features are read but never decoded (MFCC).
    frames_model : whose frame count feeds the duration column — the first
        model of the store the selector was fitted on, as in the store itself.
    pooling, layer : the pooled column the store held (`pool_l<layer>_<pooling>`).
    mu, sd : the standardiser. `transform` : an optional fitted projection
        (PCA) applied after standardising, as in `run_cv(transform=...)`.
    """

    def __init__(self, arm, params, members, estimator, mu, sd, feature_only=(),
                 use_extra_features=True, use_duration=True, pooling="mean", layer=-1,
                 frames_model=None, transform=None, meta=None):
        self.arm, self.params = arm, dict(params)
        self.members, self.feature_only = tuple(members), tuple(feature_only)
        self.estimator = estimator
        self.mu, self.sd = np.asarray(mu, dtype=np.float64), np.asarray(sd, dtype=np.float64)
        self.use_extra_features, self.use_duration = bool(use_extra_features), bool(use_duration)
        self.pooling, self.layer = pooling, int(layer)
        self.frames_model = frames_model or self.members[0]
        self.transform = transform
        self.meta = dict(meta or {})

    def __repr__(self):
        return (f"FittedSelector({self.arm}, members={self.members}, "
                f"{len(self.mu)} features, fitted on {self.meta.get('n_rows', '?')} rows)")

    @property
    def encoders(self):
        """Every model whose encoder a `Router` must run for this selector."""
        extra = self.feature_only if self.use_extra_features else ()
        return self.members + tuple(m for m in extra if m not in self.members)

    def recipe(self):
        """The JSON-able description: enough to refit with `fit_selector`."""
        return {"arm": self.arm, "params": self.params, "members": list(self.members),
                "feature_only": list(self.feature_only),
                "use_extra_features": self.use_extra_features,
                "use_duration": self.use_duration, "pooling": self.pooling,
                "layer": self.layer, "frames_model": self.frames_model,
                "n_features": int(len(self.mu)), "transform": self.transform is not None,
                **self.meta}

    def design(self, features, n_frames=None):
        """`{model: (B, D)}` pooled features (+ frame counts) -> standardised `(B, F)`."""
        blocks = [np.asarray(features[m], dtype=np.float64) for m in self.encoders]
        if self.use_duration:
            if n_frames is None:
                raise ValueError("this selector uses the duration column: pass n_frames")
            blocks.append(np.log1p(np.asarray(n_frames, dtype=np.float64))[:, None])
        X = np.concatenate(blocks, axis=1)
        if X.shape[1] != len(self.mu):
            raise ValueError(f"{X.shape[1]} features assembled, selector expects {len(self.mu)}")
        X = (X - self.mu) / self.sd
        return self.transform.transform(X) if self.transform is not None else X

    def predict(self, features, n_frames=None):
        """Member index per utterance (into `members`)."""
        return np.asarray(self.estimator.predict(self.design(features, n_frames)), dtype=int)

    def choose(self, features, n_frames=None):
        """Member name per utterance."""
        return [self.members[i] for i in self.predict(features, n_frames)]

    # ------------------------------------------------------------- storage --

    def save(self, path):
        """`<path>.pkl` (the whole object) and `<path>.json` (the recipe). Returns both."""
        path = Path(path)
        pkl, rec = path.with_suffix(".pkl"), path.with_suffix(".json")
        pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl, "wb") as fh:
            pickle.dump(self, fh)
        rec.write_text(json.dumps(self.recipe(), indent=2, default=str))
        return pkl, rec

    @classmethod
    def load(cls, path):
        with open(Path(path).with_suffix(".pkl"), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} holds a {type(obj).__name__}, not a FittedSelector")
        return obj

    def push(self, hub, name=None, message=None):
        """Upload pickle + recipe to `results/<corpus>/deploy/<name>.{pkl,json}`."""
        name = name or self.arm
        base = hub.spec.results_path(f"deploy/{name}")[:-len(".json")]
        pkl, rec = self.save(Path(base.replace("/", "_")))
        hub.push_files({f"{base}.pkl": pkl, f"{base}.json": rec},
                       message or f"deploy: {name} on {hub.spec.name}")
        return base

    @classmethod
    def pull(cls, hub, name):
        """The selector pushed under `name`, or `None` if there is none."""
        base = hub.spec.results_path(f"deploy/{name}")[:-len(".json")]
        local = hub.pull_file(f"{base}.pkl", verbose=False)
        return None if local is None else cls.load(local)


def fit_selector(store, arm, params, members=None, rows=None, seed=42,
                 use_extra_features=True, use_duration=True, nthread=4,
                 epochs_full=200, transform=None, verbose=True):
    """Fit one arm on `rows` of `store` and keep everything: a `FittedSelector`.

    Parameters
    ----------
    arm, params : a `MODELS` key and the parameter dict to build it with —
        `main_<dataset>.json` carries the selected `params[arm]` of every arm.
    members : default `store.routable`, in store order.
    rows : store row indices to fit on; every row when None. Pass a 5x2
        training half to reproduce that fold, or the tuner's `eval` rows to
        deploy the model Table 3 reports on.
    seed : `Config.model_seed` in the main notebooks.
    transform : as `run_cv(transform=...)`: a factory for a fitted projection.

    The steps are `run_cv`'s, in its order: `build_design` over all rows,
    standardise on `rows`, project, build, fit. Nothing is scored here.
    """
    members = tuple(members or store.routable)
    X_all, _ = build_design(store, members, use_extra_features, use_duration)
    rows = np.arange(len(X_all)) if rows is None else np.asarray(rows, dtype=int)
    X_tr = X_all[rows]
    mu, sd = X_tr.mean(0), X_tr.std(0).clip(1e-8)
    X_tr = (X_tr - mu) / sd
    fitted = None
    if transform is not None:
        fitted = transform().fit(X_tr)
        X_tr = fitted.transform(X_tr)
    E, nref = store.err(members)[rows], store.nref[rows]
    Y = wer_matrix(E, nref)

    ctx = BuildContext(nthread=nthread, epochs_full=epochs_full, members=members,
                       errors=store.errors, texts=store.texts)
    model = MODELS[arm].build(dict(params), X_tr.shape[1], seed=seed, ctx=ctx)
    if wants_rows(model) or getattr(model, "produces", "choice") != "choice":
        raise ValueError(f"{arm} is a fusion arm: it needs every member's transcript "
                         "and cannot be deployed as a selector — see RoverRouter")
    t0 = time.perf_counter()
    model.fit(X_tr, Y, nref=nref)
    seconds = time.perf_counter() - t0

    feature_only = tuple(store.feature_only) if use_extra_features else ()
    meta = {"corpus": store.spec.name, "n_rows": int(len(rows)), "seed": int(seed),
            "fit_seconds": seconds, "nthread": nthread}
    layer = getattr(store, "layer", None)
    sel = FittedSelector(arm, params, members, model, mu, sd, feature_only=feature_only,
                         use_extra_features=use_extra_features, use_duration=use_duration,
                         pooling=getattr(store, "pooling", "mean"),
                         layer=-1 if layer is None else layer,
                         frames_model=getattr(store, "models", members)[0],
                         transform=fitted, meta=meta)
    if verbose:
        print(f"{arm}: fitted on {len(rows):,} rows x {X_tr.shape[1]:,} features "
              f"in {seconds:.1f}s")
    return sel


class RouterOutput:
    """What a router returns for one batch: transcripts, choices and the clock."""

    def __init__(self, text, chosen, seconds, n):
        self.text, self.chosen, self.seconds, self.n = list(text), list(chosen), dict(seconds), int(n)

    def __repr__(self):
        ms = {k: f"{v * 1e3 / max(self.n, 1):.1f}" for k, v in self.seconds.items()}
        return f"RouterOutput({self.n} utt, ms/utt {ms})"

    @property
    def per_utterance(self):
        """Seconds per utterance, per step, plus `total`."""
        out = {k: v / max(self.n, 1) for k, v in self.seconds.items()}
        out["total"] = sum(out.values())
        return out


class Router:
    """FASTT at run time: every encoder, the selector, ONE decoder.

    Parameters
    ----------
    selector : a `FittedSelector`.
    extractors : `{model: ASRProbeExtractor}` covering `selector.encoders`;
        `load_extractors` builds them from the registry.

    `transcribe(arrays)` returns a `RouterOutput` whose `seconds` splits the
    batch's wall-clock into `encode` (all members + feature-only blocks),
    `select` (pooling, design matrix and the estimator) and `decode` (the
    chosen members only, each on its own sub-batch). Device time, not queue
    time: every step ends in a synchronise.
    """

    name = "fastt"

    def __init__(self, selector, extractors):
        self.selector, self.extractors = selector, dict(extractors)
        missing = [m for m in selector.encoders if m not in self.extractors]
        if missing:
            raise KeyError(f"no extractor for {missing}")

    def encode(self, arrays):
        """`{model: Encoded}` for every encoder the selector needs."""
        return {m: self.extractors[m].encode(arrays) for m in self.selector.encoders}

    def select(self, encoded):
        """Member index per utterance from a batch of `Encoded`s."""
        s = self.selector
        feats = {m: self.extractors[m].features(encoded[m], pooling=s.pooling, layer=s.layer)
                 for m in s.encoders}
        n_frames = encoded[s.frames_model].lengths.cpu().numpy() if s.use_duration else None
        return s.predict(feats, n_frames)

    def transcribe(self, arrays):
        arrays = list(arrays)
        _sync(); t0 = time.perf_counter()
        enc = self.encode(arrays)
        _sync(); t1 = time.perf_counter()
        choice = self.select(enc)
        _sync(); t2 = time.perf_counter()
        text = [None] * len(arrays)
        for j, m in enumerate(self.selector.members):
            idx = [i for i in range(len(arrays)) if choice[i] == j]
            if idx:
                for i, h in zip(idx, self.extractors[m].decode(enc[m].take(idx))):
                    text[i] = h
        _sync(); t3 = time.perf_counter()
        return RouterOutput(text, [self.selector.members[c] for c in choice],
                            {"encode": t1 - t0, "select": t2 - t1, "decode": t3 - t2},
                            len(arrays))


class RoverRouter:
    """The comparison: every encoder, EVERY decoder, then ROVER over the transcripts.

    Parameters
    ----------
    members : the systems fused, and the order their transcripts are aligned
        in (best system first, as `RoverFusion` orders them).
    weights : one per member, as `RoverFusion.fit` sets `weights_`.
    null_penalty : see `rover.vote`.
    extractors : `{model: ASRProbeExtractor}` covering `members`.

    `fit_rover` derives `members`/`weights` from a store the way the `rover_w`
    arm does, so the fused system here is the one Table 3 scores.
    """

    name = "rover"

    def __init__(self, members, weights, null_penalty, extractors):
        self.members, self.weights = tuple(members), np.asarray(weights, dtype=np.float64)
        self.null_penalty, self.extractors = float(null_penalty), dict(extractors)
        missing = [m for m in self.members if m not in self.extractors]
        if missing:
            raise KeyError(f"no extractor for {missing}")

    def fuse(self, hyps_by_member):
        """`{member: [text per utterance]}` -> fused text per utterance."""
        n = len(next(iter(hyps_by_member.values())))
        out = []
        for i in range(n):
            slots = build_network([str(hyps_by_member[m][i]) for m in self.members])
            out.append(vote(slots, self.weights, self.null_penalty))
        return out

    def transcribe(self, arrays):
        arrays = list(arrays)
        _sync(); t0 = time.perf_counter()
        enc = {m: self.extractors[m].encode(arrays) for m in self.members}
        _sync(); t1 = time.perf_counter()
        hyps = {m: self.extractors[m].decode(enc[m]) for m in self.members}
        _sync(); t2 = time.perf_counter()
        text = self.fuse(hyps)
        t3 = time.perf_counter()
        return RouterOutput(text, ["rover"] * len(arrays),
                            {"encode": t1 - t0, "decode": t2 - t1, "fuse": t3 - t2},
                            len(arrays))


def fit_rover(store, members, weighting="inverse_wer", null_penalty=1.0, rows=None,
              extractors=None):
    """A `RoverRouter` weighted like the `rover_w` arm fitted on `rows`.

    `RoverFusion.fit` needs only the members' WER on the rows, so no
    transcript is aligned here. `extractors` may be added later
    (`RoverRouter(..., extractors)`); `None` returns the fitted pieces as a
    dict for that.
    """
    members = tuple(members)
    rows = np.arange(len(store.nref)) if rows is None else np.asarray(rows, dtype=int)
    E, nref = store.err(members)[rows], store.nref[rows]
    fused = RoverFusion(store.texts, members, weighting, null_penalty)
    fused.fit(None, wer_matrix(E, nref), nref=nref, rows=rows)
    order = [members[i] for i in fused.order_]
    weights = fused.weights_[fused.order_]
    if extractors is None:
        return {"members": order, "weights": weights.tolist(), "null_penalty": float(null_penalty),
                "weighting": weighting}
    return RoverRouter(order, weights, null_penalty, extractors)


def load_extractors(names, groups=None, **kw):
    """`{name: extractor}` from the registry, restricted to the installed groups.

    `groups` is the set of install groups this kernel has (`("A", "C")` for the
    transformers + ESPnet stack); a model from another group is skipped with a
    message rather than failing on import. Nothing is loaded onto the device
    until a model's first `encode`.
    """
    out = {}
    for n in names:
        if n not in EXTRACTORS:
            raise KeyError(f"{n} is not in latency.EXTRACTORS")
        g = install_group(n)
        if groups is not None and g not in groups:
            print(f"  {n}: install group {g} not in this kernel — skipped")
            continue
        out[n] = load_extractor(n, **kw)
    return out


def compare_routers(routers, arrays, refs=None, scorer=None, batch_size=1, warmup=1):
    """Time several routers on the same clips. One row per router.

    Parameters
    ----------
    routers : `{label: Router | RoverRouter}`.
    arrays : the clips; `refs` their reference texts, to add a WER column.
    scorer : a `WerScorer`; the first extractor's when None.
    batch_size : 1 for online latency.
    warmup : untimed leading batches per router, for CUDA warm-up.

    Returns per-utterance seconds per step (`encode`, `select`, `decode`,
    `fuse`, `total`), the selection distribution for a `Router`, and `wer`.
    """
    arrays = list(arrays)
    rows = []
    for label, router in routers.items():
        if scorer is None and refs is not None:
            scorer = next(iter(router.extractors.values())).scorer
        batches = [arrays[i:i + batch_size] for i in range(0, len(arrays), batch_size)]
        for b in batches[:warmup]:
            router.transcribe(b)
        outs = [router.transcribe(b) for b in batches]
        secs = {}
        for o in outs:
            for k, v in o.seconds.items():
                secs[k] = secs.get(k, 0.0) + v
        row = {"router": label, **{k: v / len(arrays) for k, v in secs.items()}}
        row["total"] = sum(secs.values()) / len(arrays)
        chosen = [c for o in outs for c in o.chosen]
        if isinstance(router, Router):
            for m in router.selector.members:
                row[f"share_{m}"] = chosen.count(m) / len(chosen)
        if refs is not None:
            hyps = [t for o in outs for t in o.text]
            cols = scorer.score(list(refs), hyps)
            row["wer"] = (sum(cols["sub"]) + sum(cols["dele"]) + sum(cols["ins"])) \
                / max(sum(cols["nref"]), 1)
        rows.append(row)
    return pd.DataFrame(rows)

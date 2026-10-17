__all__ = ['ERROR_PARTS', 'ERROR_COLUMNS', 'default_cache_dir', 'EnsembleStore', 'ensemble_stats', 'win_rates', 'best_single_model', 'describe_ensemble', 'pair_diversity_table', 'error_correlation_matrix', 'win_rate_table']

import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


from fastt.hub import FasttHub


def default_cache_dir():
    """Where `EnsembleStore` keeps its local copy: `$FASTT_CACHE` or `~/.cache/fastt`.

    On Colab this lives on the VM disk, which survives a kernel restart — which
    is the case that matters, because restarting the kernel is how you switch
    dependency groups and it is exactly when you do not want to re-download
    400 MB.
    """
    return Path(os.environ.get("FASTT_CACHE") or (Path.home() / ".cache" / "fastt"))

# The three counters that make up an error count, plus the reference length.
ERROR_PARTS = ("sub", "dele", "ins")
ERROR_COLUMNS = ERROR_PARTS + ("nref",)


def _fixed_list_matrix(column):
    """Arrow fixed-size-list column -> dense (N, D) array, without the formatter.

    Same trick as `feature_matrix` in §3.3: the width lives in the schema and the
    values are one contiguous buffer, so this is a reshape rather than N x D
    Python floats.
    """
    chunks = column.chunks if hasattr(column, "chunks") else [column]
    return np.concatenate([
        c.flatten().to_numpy(zero_copy_only=False).reshape(len(c), -1) for c in chunks
    ])


class EnsembleStore:
    """Pooled features + per-utterance error counts for every model, aligned.

    Parameters
    ----------
    models : short model names, e.g. `("whisper_large_v3", "wav2vec2")`. Default
        is every `model_<name>` config in the repo.
    pooling : which pooled column to read. `"mean"` is the default and, per §5.4,
        usually the right one; the parameter exists so the whole section can be
        re-run on `"std"` or any other pooling without editing anything.
    layer : only needed if a config carries more than one extracted layer, which
        none currently do. Left `None`, the single pooled column is found by name.
    stream : forwarded to `FasttHub`. `False` — the default — downloads every
        shard this store needs in one parallel pass and then reads locally;
        `True` restores the per-column HTTP range reads. See §2.1.

    Attributes set by `load()`
    --------------------------
    index      : `MultiIndex(split, id)` — the canonical row order (from the
                 first model, i.e. `base` order: sorted by duration).
    routable   : models with WER counters; these are the ones an ensemble picks from.
    feature_only : models with embeddings but no WER (the MFCC baseline).
    errors     : (N, n_routable) DataFrame of `sub + dele + ins`.
    nref, n_frames : (N,) arrays — reference words and encoder frames per utterance.
    features   : dict model -> (N, D) float32.
    """

    def __init__(self, models=None, pooling="mean", layer=None,
                 repo_id=None, splits=None, cache_dir=None, spec=None, stream=False):
        self.hub = FasttHub(repo_id, spec=spec, stream=stream)
        self.spec = self.hub.spec
        self.repo_id, self.pooling, self.layer = self.hub.repo_id, pooling, layer
        self.splits = tuple(splits or self.spec.splits)
        self.models = tuple(models) if models is not None else tuple(self.hub.model_names())
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.cache_path = None                  # set by load(); where it read/wrote
        self._reset()

    def _reset(self):
        """Clear everything `load()` sets, so a failed load never leaves half a store."""
        self.routable = self.feature_only = ()
        self.features = {}
        self.index = self.errors = self.nref = self.n_frames = None
        self.texts = None                       # filled by load_texts(), not load()

    def __repr__(self):
        state = (f"{len(self.index):,} utterances, {len(self.routable)} routable"
                 if self.index is not None else "not loaded")
        return f"EnsembleStore({self.spec.name}, {self.pooling!r}, {state})"

    # --------------------------------------------------------------- load --

    def _pooled_column(self, config, names):
        """The one `pool_l<layer>_<pooling>` column to read from this config."""
        import re

        pattern = (rf"pool_l{self.layer}_{self.pooling}" if self.layer is not None
                   else rf"pool_l\d+_{self.pooling}")
        found = [n for n in names if re.fullmatch(pattern, n)]
        if not found:
            raise KeyError(f"{config} has no column matching {pattern!r}; "
                           f"it has {[n for n in names if n.startswith('pool_l')]}")
        if len(found) > 1:
            raise KeyError(f"{config} has several: {found}. Pass layer=... to choose.")
        return found[0]

    def _read_model(self, model):
        """One model, all splits: `(meta DataFrame, (N, D) features)`."""
        config = self.spec.model_config(model)
        names = self.hub.column_names(config, self.splits[0])
        pooled = self._pooled_column(config, names)
        has_wer = all(c in names for c in ERROR_COLUMNS)
        wanted = (["id"] + [c for c in ("n_frames",) + ERROR_COLUMNS if c in names]
                  + [pooled])

        metas, feats = [], []
        for split in self.splits:
            table = self.hub.read_columns(config, split, wanted)
            meta = pd.DataFrame({c: table.column(c).to_pandas()
                                 for c in wanted if c != pooled})
            meta.insert(0, "split", split)
            metas.append(meta)
            feats.append(_fixed_list_matrix(table.column(pooled)).astype(np.float32))

        meta = pd.concat(metas, ignore_index=True)
        meta.index = pd.MultiIndex.from_arrays([meta["split"], meta["id"]])
        return meta, np.concatenate(feats), has_wer, pooled

    # -------------------------------------------------------------- cache --

    def fingerprint(self):
        """`(payload, digest)` — everything this store's contents depend on.

        The repo revision is in the payload, so pushing a new `model_*` config
        changes the digest and the next `load()` re-fetches instead of serving
        features that no longer match the Hub.
        """
        # `dataset` is in the key so two corpora never share a cache entry: the
        # model names are identical across them and everything else here could
        # match by coincidence.
        payload = {"repo_id": self.repo_id, "dataset": self.spec.name,
                   "pooling": self.pooling, "layer": self.layer,
                   "splits": list(self.splits), "models": list(self.models),
                   "revision": self.hub.revision()}
        blob = json.dumps(payload, sort_keys=True).encode()
        return payload, hashlib.sha1(blob).hexdigest()[:16]

    def _write_cache(self, path, payload):
        """Write the loaded state to `path`, atomically.

        Built in a sibling directory and renamed into place, because the usual
        way to interrupt a Colab cell is to interrupt it, and a half-written
        cache that still looks valid is worse than no cache at all.
        """
        tmp = path.with_name(path.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        meta = pd.DataFrame({
            "split": self.index.get_level_values("split").to_numpy(),
            "id": self.index.get_level_values("id").to_numpy(),
            "nref": self.nref, "n_frames": self.n_frames})
        for m in self.routable:
            meta[f"err_{m}"] = self.errors[m].to_numpy(np.float64)
        meta.to_parquet(tmp / "meta.parquet", index=False)
        np.savez(tmp / "features.npz", **self.features)
        (tmp / "manifest.json").write_text(json.dumps(
            {**payload, "routable": list(self.routable),
             "feature_only": list(self.feature_only),
             "widths": {m: int(X.shape[1]) for m, X in self.features.items()}}, indent=2))

        shutil.rmtree(path, ignore_errors=True)
        tmp.rename(path)

    def _read_cache(self, path):
        """Restore the state `_write_cache` wrote. Raises if anything is missing."""
        manifest = json.loads((path / "manifest.json").read_text())
        meta = pd.read_parquet(path / "meta.parquet")

        self.index = pd.MultiIndex.from_arrays(
            [meta["split"].to_numpy(), meta["id"].to_numpy()], names=["split", "id"])
        self.nref = meta["nref"].to_numpy(np.float64)
        self.n_frames = meta["n_frames"].to_numpy(np.float64)
        self.routable = tuple(manifest["routable"])
        self.feature_only = tuple(manifest["feature_only"])
        self.errors = pd.DataFrame(
            {m: meta[f"err_{m}"].to_numpy(np.float64) for m in self.routable},
            index=self.index)
        with np.load(path / "features.npz") as z:
            self.features = {k: z[k] for k in z.files}

    def load(self, verbose=True, cache=True, refresh=False):
        """Fetch every model, align on `(split, id)`, and validate. Returns self.

        Parameters
        ----------
        cache : read from and write to the local cache under `cache_dir`. The
            Hub read is ~40 HTTP-backed parquet reads and takes minutes; the
            cache turns a reload into about a second, which is the difference
            between "run the experiment" and "think about running it".
        refresh : re-fetch and overwrite the cache entry even if it is valid.
        """
        payload, digest = self.fingerprint() if cache else (None, None)
        self.cache_path = (self.cache_dir / f"store_{digest}") if cache else None

        if cache and not refresh and self.cache_path.exists():
            try:
                self._read_cache(self.cache_path)
                if verbose:
                    print(f"{len(self.index):,} utterances x {len(self.routable)} routable "
                          f"model(s), from cache {self.cache_path}")
                return self
            except Exception as e:                         # noqa: BLE001
                print(f"  note: cache at {self.cache_path} unusable "
                      f"({type(e).__name__}: {e}); re-fetching.")
                self._reset()

        self._fetch(verbose=verbose)
        if cache and payload["revision"] is not None:
            # No revision means the Hub could not be reached, so what was just
            # read came from somewhere unverified. Do not stamp it as canonical.
            self._write_cache(self.cache_path, payload)
            if verbose:
                print(f"cached to {self.cache_path}")
        return self

    def _fetch(self, verbose=True):
        """The uncached read: every model, aligned on `(split, id)` and validated."""
        # One bulk transfer for everything the loop below is about to read. In
        # streaming mode there is nothing to prefetch, and the loop's range
        # requests are the transfer.
        if not self.hub.streaming():
            self.hub.prefetch([self.spec.model_config(m) for m in self.models],
                              self.splits, verbose=verbose)

        routable, feature_only, metas = [], [], {}
        for model in self.models:
            meta, X, has_wer, pooled = self._read_model(model)
            if self.index is None:                  # first model fixes the row order
                self.index = meta.index
                self.n_frames = meta["n_frames"].to_numpy(np.float64)

            # Align by key rather than trusting row order (see the note above).
            order = meta.index.get_indexer(self.index)
            if (order < 0).any():
                missing = self.index[order < 0]
                raise KeyError(f"{self.spec.model_config(model)} is missing "
                               f"{len(missing)} utterance(s), "
                               f"e.g. {list(missing[:3])}")
            metas[model] = meta.iloc[order]
            self.features[model] = X[order]
            (routable if has_wer else feature_only).append(model)
            if has_wer and self.nref is None:       # any routable model defines it
                self.nref = metas[model]["nref"].to_numpy(np.float64)
            if verbose:
                print(f"  {model:<24} {X.shape[1]:>5}-d {pooled:<16} "
                      f"{'WER' if has_wer else 'features only'}")

        self.routable, self.feature_only = tuple(routable), tuple(feature_only)
        if not self.routable:
            raise ValueError("no model carries WER counters — nothing to route between")

        errors = {m: sum(metas[m][c].to_numpy(np.float64) for c in ERROR_PARTS)
                  for m in self.routable}
        self.errors = pd.DataFrame(errors, index=self.index)

        # Same reference and the same normaliser everywhere, so the reference
        # word count must agree. If it does not, the WER columns are not on a
        # common denominator and nothing below is comparable.
        for m in self.routable:
            delta = np.abs(metas[m]["nref"].to_numpy(np.float64) - self.nref).max()
            if delta > 0:
                raise ValueError(f"{self.spec.model_config(m)} disagrees with "
                                 f"{self.routable[0]} on `nref` "
                                 f"(max difference {delta:g}) — refusing to compare them")
        if verbose:
            print(f"\n{len(self.index):,} utterances x {len(self.routable)} routable model(s)"
                  + (f" (+ {', '.join(self.feature_only)} as features)"
                     if self.feature_only else ""))
        return self

    # --------------------------------------------------------------- text --

    def load_texts(self, models=None, verbose=True, cache=True, refresh=False):
        """Fetch the normalised hypotheses and the reference, aligned to `index`.

        Fusion (§9) works on transcripts, not on embeddings, and they are already
        on the Hub in `transcription_norm` — so ROVER needs no GPU and no ASR
        dependency, exactly like the rest of §6-§8. Sets and returns `self.texts`,
        a DataFrame indexed like the store with one column per model plus
        `reference`.

        `text_norm` is the reference under the same Whisper normaliser the WER
        counters used, so a WER recomputed from these strings reproduces the
        stored `sub + dele + ins` exactly. That identity is worth keeping: it is
        what makes a fused transcript comparable to a routed one.
        """
        models = tuple(models) if models is not None else self.routable
        path = None
        if cache and self.cache_path is not None:
            path = self.cache_path / f"texts_{hashlib.sha1('+'.join(models).encode()).hexdigest()[:12]}.parquet"
            if not refresh and path.exists():
                self.texts = pd.read_parquet(path)
                self.texts.index = self.index
                if verbose:
                    print(f"texts for {len(models)} model(s), from cache")
                return self.texts

        cols = {}
        for model in models:
            frames = []
            for split in self.splits:
                table = self.hub.read_columns(self.spec.model_config(model), split,
                                              ["id", "transcription_norm", "text_norm"])
                d = table.to_pandas()
                d.index = pd.MultiIndex.from_arrays(
                    [np.full(len(d), split), d["id"].to_numpy()], names=["split", "id"])
                frames.append(d)
            d = pd.concat(frames).reindex(self.index)
            if d["transcription_norm"].isna().any():
                raise KeyError(f"{self.spec.model_config(model)} is missing "
                               "transcripts for some utterances")
            cols[model] = d["transcription_norm"].to_numpy()
            if "reference" not in cols:
                cols["reference"] = d["text_norm"].to_numpy()
            if verbose:
                print(f"  {model:<24} transcripts")

        self.texts = pd.DataFrame(cols, index=self.index)[list(models) + ["reference"]]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.texts.to_parquet(path, index=False)
        return self.texts

    # ------------------------------------------------------------- access --

    def mask(self, splits):
        """Boolean row mask for a list of split names."""
        return np.isin(self.index.get_level_values("split").to_numpy(), list(splits))

    def err(self, models, splits=None):
        """(N, len(models)) error counts, optionally restricted to some splits."""
        E = self.errors[list(models)].to_numpy(np.float64)
        return E if splits is None else E[self.mask(splits)]

    def words(self, splits=None):
        """(N,) reference word counts, optionally restricted to some splits."""
        return self.nref if splits is None else self.nref[self.mask(splits)]

    def corpus_wer(self, models=None, splits=None):
        """Corpus WER per model over `splits` — total errors over total words.

        `splits=None` means the dataset's eval splits, NOT every row: `err()`
        and `words()` read a bare `None` as "all rows", so the default is
        resolved here rather than passed through.
        """
        models = list(models or self.routable)
        splits = splits or self.spec.eval_splits
        E, n = self.err(models, splits), self.words(splits)
        return pd.Series(E.sum(0) / n.sum(), index=models, name="wer")

    def summary(self, splits=None):
        """Per-model reference table: corpus WER, mean utterance WER, clean rate."""
        models = list(self.routable)
        splits = splits or self.spec.eval_splits
        E, n = self.err(models, splits), self.words(splits)
        return pd.DataFrame({
            "model": models,
            "corpus_wer": E.sum(0) / n.sum(),
            # Reported next to corpus WER precisely because it disagrees with it:
            # the mean over utterances weights a five-word utterance like a
            # fifty-word one.
            "mean_utt_wer": (E / np.maximum(n, 1)[:, None]).mean(0),
            "clean_rate": (E == 0).mean(0),          # utterances with no error at all
            "n_utts": len(n),
        }).sort_values("corpus_wer", ignore_index=True)


def ensemble_stats(E, n, bsm):
    """Oracle arithmetic for one group of models on one set of rows.

    Parameters
    ----------
    E : (N, k) error counts, one column per member.
    n : (N,) reference word counts.
    bsm : column index of the best single model, **chosen on the fit splits**.

    Returns a flat dict of scalars, so a list of them is a DataFrame.
    """
    total = n.sum()
    best = E.min(axis=1)
    is_best = E == best[:, None]
    n_best = is_best.sum(axis=1)

    # Words the oracle saves over always using the BSM. This is the whole prize.
    gain = E[:, bsm] - best
    ranked = np.sort(gain)[::-1]

    def top_share(frac):
        """Share of the total gain sitting in the worst `frac` of utterances."""
        if gain.sum() <= 0:
            return float("nan")
        return ranked[:max(1, int(round(len(ranked) * frac)))].sum() / gain.sum()

    wer = E.sum(axis=0) / total
    oracle, bsm_wer = best.sum() / total, E[:, bsm].sum() / total
    return {
        "oracle_wer": oracle,
        "bsm_wer": bsm_wer,
        "gap_abs": bsm_wer - oracle,
        "gap_rel": (bsm_wer - oracle) / bsm_wer if bsm_wer else float("nan"),
        "random_wer": float(wer.mean()),          # expected WER of a coin flip
        "worst_wer": float(wer.max()),
        "best_member_wer": float(wer.min()),      # hindsight: may beat bsm_wer
        "all_tie_rate": float((n_best == E.shape[1]).mean()),
        "decisive_rate": float((n_best == 1).mean()),
        "gain_utts": float((gain > 0).mean()),
        "gain_top1pct": top_share(0.01),
        "gain_top5pct": top_share(0.05),
        "gain_top10pct": top_share(0.10),
        "n_utts": int(len(n)),
    }


def win_rates(E):
    """`(win_rate, sole_win_rate)`, each (k,), for one group's error matrix.

    Ties share the win equally, so `win_rate` sums to 1 and reads as "how often
    is this member the one to route to". `sole_win_rate` counts outright wins
    only, and sums to `decisive_rate`.
    """
    is_best = E == E.min(axis=1, keepdims=True)
    return (is_best / is_best.sum(axis=1, keepdims=True)).mean(axis=0), \
           (is_best & (is_best.sum(axis=1, keepdims=True) == 1)).mean(axis=0)


def best_single_model(store, members, splits=None):
    """Index of the lowest-WER member, chosen on `splits` (the fit splits)."""
    return int(store.corpus_wer(members, splits or store.spec.fit_splits)
               .to_numpy().argmin())


def describe_ensemble(store, members, fit_splits=None, eval_splits=None):
    """One group of models, fully described: stats + per-member win rates."""
    fit_splits = fit_splits or store.spec.fit_splits
    eval_splits = eval_splits or store.spec.eval_splits
    bsm = best_single_model(store, members, fit_splits)
    E, n = store.err(members, eval_splits), store.words(eval_splits)
    stats = {"members": "+".join(members), "bsm_model": members[bsm]}
    stats.update(ensemble_stats(E, n, bsm))

    share, sole = win_rates(E)
    wins = pd.DataFrame({
        "model": list(members),
        "wer": store.corpus_wer(members, eval_splits).to_numpy(),
        "win_rate": share, "sole_win_rate": sole,
        "is_bsm": [i == bsm for i in range(len(members))],
    })
    # A one-row frame rather than a Series: it mixes names and numbers, and only
    # a DataFrame rounds the numeric columns and leaves the strings alone.
    return pd.DataFrame([stats]), wins

from scipy.stats import pearsonr, spearmanr


def _pair_stats(err_a, err_b, n):
    """Correlation and binary-diversity measures for one pair of models."""
    wer_a, wer_b = err_a / np.maximum(n, 1), err_b / np.maximum(n, 1)
    ok_a, ok_b = err_a == 0, err_b == 0          # "transcribed exactly right"

    # 2x2 contingency table of correctness.
    n11 = float((ok_a & ok_b).mean())            # both right
    n00 = float((~ok_a & ~ok_b).mean())          # both wrong  -> the double fault
    n10 = float((ok_a & ~ok_b).mean())
    n01 = float((~ok_a & ok_b).mean())

    ad, bc = n11 * n00, n01 * n10
    observed = n11 + n00
    # Chance agreement from the marginals, which is what kappa corrects for.
    expected = (ok_a.mean() * ok_b.mean() + (1 - ok_a.mean()) * (1 - ok_b.mean()))
    return {
        "pearson_r": float(pearsonr(wer_a, wer_b)[0]),
        "spearman_r": float(spearmanr(wer_a, wer_b)[0]),
        "disagreement": n10 + n01,
        "double_fault": n00,
        "both_clean": n11,
        "q_statistic": (ad - bc) / (ad + bc) if ad + bc > 0 else float("nan"),
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else float("nan"),
    }


def pair_diversity_table(store, splits=None, models=None):
    """One row per model pair. The input to every "is this group diverse?" question."""
    models = list(models or store.routable)
    splits = splits or store.spec.eval_splits
    E, n = store.err(models, splits), store.words(splits)
    wer = E.sum(axis=0) / n.sum()

    rows = []
    for i, j in itertools.combinations(range(len(models)), 2):
        row = {"model_a": models[i], "model_b": models[j],
               "wer_a": wer[i], "wer_b": wer[j]}
        row.update(_pair_stats(E[:, i], E[:, j], n))
        pair_oracle = np.minimum(E[:, i], E[:, j]).sum() / n.sum()
        row["oracle_wer"] = pair_oracle
        row["gap_abs"] = min(wer[i], wer[j]) - pair_oracle
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gap_abs", ascending=False, ignore_index=True)


def error_correlation_matrix(pairs, metric="pearson_r"):
    """Fold `pair_diversity_table` into a symmetric model x model matrix."""
    models = sorted(set(pairs["model_a"]) | set(pairs["model_b"]))
    M = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
    if metric in ("disagreement", "double_fault"):
        np.fill_diagonal(M.to_numpy(), 0.0)       # a model never disagrees with itself
    for _, r in pairs.iterrows():
        M.loc[r["model_a"], r["model_b"]] = M.loc[r["model_b"], r["model_a"]] = r[metric]
    return M

def win_rate_table(store, ensembles, eval_splits=None, fit_splits=None):
    """Long-format win rates: one row per (group, member). Tidy enough to plot."""
    fit_splits = fit_splits or store.spec.fit_splits
    eval_splits = eval_splits or store.spec.eval_splits
    rows = []
    for members in ensembles:
        bsm = best_single_model(store, members, fit_splits)
        E = store.err(members, eval_splits)
        share, sole = win_rates(E)
        for i, m in enumerate(members):
            rows.append({"members": "+".join(members), "model": m,
                         "win_rate": share[i], "sole_win_rate": sole[i],
                         "is_bsm": i == bsm})
    return pd.DataFrame(rows)

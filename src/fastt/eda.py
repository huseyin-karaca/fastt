__all__ = ['gain_profile', 'eda', 'selection_accuracy', 'base_model_report', 'split_sizes', 'feature_dims', 'partition_sizes', 'fold_brief', 'eda_report', 'win_rates_tolerant', 'tolerance_profile', 'split_durations', 'dataset_table']

import numpy as np
import pandas as pd

from fastt.ensemble import (best_single_model, ensemble_stats,
                            error_correlation_matrix, pair_diversity_table,
                            win_rates)


def gain_profile(E, nref, bsm, quantiles=5):
    """Where the achievable gain lives, by reference length. One row per quantile.

    `share_of_gain` against `share_of_words` is the comparison that matters: a
    bin holding more gain than words is one the length-weighted objective of §7.1
    would *under*-serve, and vice versa.
    """
    gain = E[:, bsm] - E.min(axis=1)
    if gain.sum() <= 0:
        return pd.DataFrame()
    # Ranked, not cut on values: reference lengths are lumpy and equal-width bins
    # come out wildly unbalanced.
    ranks = np.argsort(np.argsort(nref))
    bins = np.minimum((ranks * quantiles) // len(nref), quantiles - 1)
    rows = []
    for b in range(quantiles):
        m = bins == b
        rows.append({
            "length_bin": b,
            "n_utts": int(m.sum()),
            "mean_nref": float(nref[m].mean()),
            "share_of_gain": float(gain[m].sum() / gain.sum()),
            "share_of_words": float(nref[m].sum() / nref.sum()),
            "contested_rate": float((E[m].min(1) < E[m].max(1)).mean()),
        })
    out = pd.DataFrame(rows)
    out["gain_per_word"] = out["share_of_gain"] / out["share_of_words"].clip(1e-12)
    return out


def eda(store, members, splits=None, rows=None, quantiles=5, verbose=True):
    """The pre-flight briefing for one member set. Returns a dict of frames.

    `splits` restricts to named splits (the §6 protocol); `rows` restricts to
    store row indices (the §7 protocol). Pass neither to use everything, which is
    what `run_cv` does.
    """
    members = tuple(members)
    E, nref = store.err(members), store.nref
    if splits is not None:
        mask = store.mask(splits)
        E, nref = E[mask], nref[mask]
    if rows is not None:
        rows = np.asarray(rows, dtype=int)
        E, nref = E[rows], nref[rows]

    bsm = int((E.sum(0) / nref.sum()).argmin())
    stats = ensemble_stats(E, nref, bsm)
    share, sole = win_rates(E)
    best = E.min(axis=1)

    per_model = pd.DataFrame({
        "model": list(members),
        "corpus_wer": E.sum(0) / nref.sum(),
        "mean_utt_wer": (E / np.maximum(nref, 1)[:, None]).mean(0),
        "clean_rate": (E == 0).mean(0),
        "win_rate": share,
        "sole_win_rate": sole,
        "is_bsm": [i == bsm for i in range(len(members))],
    }).sort_values("corpus_wer", ignore_index=True)

    trivial = pd.DataFrame([{
        # The three ways an utterance can be outside a router's reach, which are
        # not the same thing and are routinely conflated.
        "all_members_perfect": float((E == 0).all(axis=1).mean()),
        "all_members_tied": float((E == best[:, None]).all(axis=1).mean()),
        "bsm_already_optimal": float((E[:, bsm] == best).mean()),
        "gainable": float((E[:, bsm] > best).mean()),
    }])

    pairs = pair_diversity_table(store, splits or store.splits, list(members))
    out = {
        "stats": pd.DataFrame([{"members": "+".join(members),
                                "bsm_model": members[bsm], **stats}]),
        "per_model": per_model,
        "trivial": trivial,
        "gain_profile": gain_profile(E, nref, bsm, quantiles),
        "pairs": pairs,
        "error_correlation": error_correlation_matrix(pairs, "pearson_r"),
    }

    if verbose:
        s = out["stats"].iloc[0]
        print(f"=== {'+'.join(members)}   {len(nref):,} utterances, "
              f"{nref.sum():,.0f} reference words")
        print(f"    bsm {s['bsm_model']} {s['bsm_wer']:.4f}   oracle {s['oracle_wer']:.4f}"
              f"   gap {s['gap_abs']:.4f} ({s['gap_rel']:.1%} relative)"
              f"   random {s['random_wer']:.4f}")
        t = trivial.iloc[0]
        print(f"    unroutable: {t['all_members_perfect']:.1%} all perfect, "
              f"{t['all_members_tied']:.1%} all tied, "
              f"{t['bsm_already_optimal']:.1%} bsm already optimal "
              f"-> only {t['gainable']:.1%} of rows are winnable")
        print(f"    prize concentration: {s['gain_top10pct']:.0%} of the gain is in "
              f"the worst 10% of utterances", end="")
        print("   (>0.85 means no router will find it)"
              if s["gain_top10pct"] > 0.85 else "")
        print("\n" + per_model.round(4).to_string(index=False))
        if len(out["gain_profile"]):
            print("\nwhere the gain sits, by reference length:")
            print(out["gain_profile"].round(3).to_string(index=False))
        print("\npairwise error correlation:")
        print(out["error_correlation"].round(3).to_string())
    return out

def selection_accuracy(E, choice):
    """`(strict, lenient)` accuracy of one per-utterance choice vector.

    **lenient** counts a row as correct whenever the chosen member achieves that
    row's minimum, ties included — on a tied row every choice really is correct,
    so anything stricter would penalise an arm for a decision that could not have
    been wrong.

    **strict** is the same rule applied only to the *decisive* rows, those with a
    single best member. Those are the rows where a wrong answer exists, and they
    are a minority of this corpus, so the two numbers can be far apart. Quote
    both or neither. `strict` is `nan` when no row is decisive.
    """
    picked = E[np.arange(len(E)), np.asarray(choice, dtype=int)]
    best = E.min(axis=1)
    lenient = float((picked <= best).mean())
    decisive = (E == best[:, None]).sum(axis=1) == 1
    strict = (float((picked[decisive] <= best[decisive]).mean())
              if decisive.any() else float("nan"))
    return strict, lenient


def base_model_report(E, nref, members, bsm=None):
    """Per-member statistics for the models INSIDE the ensemble. One row each.

    `win_rate` shares a tied row equally between its winners and so sums to 1;
    `sole_win_rate` counts outright wins only and sums to the decisive rate.
    `acc_strict` / `acc_lenient` are `selection_accuracy` for the constant policy
    "always route to this member" — the baseline a learned router is beating, or
    not.
    """
    members = list(members)
    share, sole = win_rates(E)
    rows = []
    for i, m in enumerate(members):
        strict, lenient = selection_accuracy(E, np.full(len(E), i))
        rows.append({
            "model": m,
            "corpus_wer": float(E[:, i].sum() / nref.sum()),
            "mean_utt_wer": float((E[:, i] / np.maximum(nref, 1)).mean()),
            "clean_rate": float((E[:, i] == 0).mean()),
            "win_rate": float(share[i]),
            "sole_win_rate": float(sole[i]),
            "acc_strict": strict,
            "acc_lenient": lenient,
            "is_bsm": (bsm is not None and i == int(bsm)),
        })
    return pd.DataFrame(rows).sort_values("corpus_wer", ignore_index=True)


def split_sizes(store, splits=None, rows=None):
    """One row per corpus split: utterances, reference words, and its role.

    `role` names what the active `DatasetSpec` does with the split — `fit` for
    the splits §6 trains on, `eval` for the ones it scores on, `both` where a
    spec uses one split for both. §7 ignores the roles entirely (it pools
    everything and re-splits at random), which is exactly why they are worth
    printing: the table shows what was merged.
    """
    splits = tuple(splits or store.splits)
    level = store.index.get_level_values("split").to_numpy()
    keep = np.ones(len(level), dtype=bool)
    if rows is not None:
        keep = np.zeros(len(level), dtype=bool)
        keep[np.asarray(rows, dtype=int)] = True

    fit_splits = set(store.spec.fit_splits)
    eval_splits = set(store.spec.eval_splits)
    out = []
    for s in splits:
        m = (level == s) & keep
        role = ("both" if s in fit_splits and s in eval_splits else
                "fit" if s in fit_splits else "eval" if s in eval_splits else "unused")
        out.append({"split": s, "role": role, "n_utts": int(m.sum()),
                    "n_words": float(store.nref[m].sum()),
                    "mean_words": float(store.nref[m].mean()) if m.any() else np.nan})
    df = pd.DataFrame(out)
    total = df["n_utts"].sum()
    df["share"] = df["n_utts"] / max(total, 1)
    return df


def feature_dims(store, members, use_extra_features=True, use_duration=True):
    """One row per block of the design matrix `build_design` would assemble.

    Kept in step with `build_design` by construction — same member order, same
    `feature_only` block, same trailing duration column — so the widths printed
    before a run are the widths the run actually fits on.
    """
    names = list(members) + (list(store.feature_only) if use_extra_features else [])
    rows = [{"block": n, "kind": "member" if n in tuple(members) else "feature_only",
             "n_features": int(store.features[n].shape[1])} for n in names]
    if use_duration:
        rows.append({"block": "duration", "kind": "derived", "n_features": 1})
    df = pd.DataFrame(rows)
    df["share"] = df["n_features"] / max(df["n_features"].sum(), 1)
    return df


def partition_sizes(n, parts):
    """`parts` is `{name: row indices}` — one row each, plus a containment check.

    Two partitions may be **disjoint** (`tune` and `eval`) or **nested** (`fit`
    and `val` inside `tune`); what they may not be is partially overlapping.
    That is the check, and it is the entire reason this is a function rather
    than four `len()` calls: a tuning hold-out that leaks a handful of
    utterances into the evaluation rows invalidates every p-value downstream and
    is invisible in any table that only prints sizes.

    The `within` column names the smallest partition each one sits inside, so
    the nesting is on the record rather than merely tolerated.
    """
    named = {k: np.asarray(v, dtype=int) for k, v in parts.items()}
    sets = {k: set(v.tolist()) for k, v in named.items()}
    keys = list(named)

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = sets[a] & sets[b]
            if shared and not (sets[a] <= sets[b] or sets[b] <= sets[a]):
                extra = sorted(shared)[:3]
                raise ValueError(
                    f"partitions {a!r} and {b!r} overlap without either containing "
                    f"the other: {len(shared)} shared row(s), e.g. {extra}. "
                    "Partitions must be disjoint or nested.")

    rows = []
    for k, v in named.items():
        # The smallest OTHER partition that strictly contains this one.
        parents = [o for o in keys if o != k and sets[k] < sets[o]]
        within = min(parents, key=lambda o: len(sets[o])) if parents else ""
        rows.append({"partition": k, "n_utts": len(v),
                     "share": len(v) / max(n, 1), "within": within})
    return pd.DataFrame(rows)

def fold_brief(E, nref, members, train_idx, eval_idx, seed=None, fold=None,
               verbose=True):
    """The per-fold briefing, printed by `run_cv(fold_eda=True)` before fitting.

    Deliberately takes arrays rather than a store: `run_cv` has already applied
    `rows=`, so the store's row numbering is no longer the one in play, and
    handing it a store would invite indexing the wrong thing.

    The BSM is recomputed here exactly as `fold_arms` computes it — on the
    training half, by corpus WER — so the member printed is the member scored
    against, not a corpus-wide favourite that this fold may not agree with.
    """
    tr, ev = np.asarray(train_idx, dtype=int), np.asarray(eval_idx, dtype=int)
    E_tr, E_ev = E[tr], E[ev]
    n_tr, n_ev = nref[tr], nref[ev]
    bsm = int((E_tr.sum(0) / n_tr.sum()).argmin())

    stats = ensemble_stats(E_ev, n_ev, bsm)
    out = {
        "sizes": pd.DataFrame([{"half": "train", "n_utts": len(tr),
                                "n_words": float(n_tr.sum())},
                               {"half": "eval", "n_utts": len(ev),
                                "n_words": float(n_ev.sum())}]),
        "bsm": members[bsm],
        "stats": pd.DataFrame([stats]),
        "per_model": base_model_report(E_ev, n_ev, members, bsm),
    }
    if verbose:
        label = "fold" if seed is None else f"seed={seed}"
        print(f"  [{label} f{fold}]  train {len(tr):,} / eval {len(ev):,} utts   "
              f"bsm={members[bsm]}  "
              f"bsm_wer {stats['bsm_wer']:.4f}  oracle {stats['oracle_wer']:.4f}  "
              f"gap {stats['gap_abs']:.4f} ({stats['gap_rel']:.1%})  "
              f"winnable {stats['gain_utts']:.1%}")
        cols = ["model", "corpus_wer", "win_rate", "acc_strict", "acc_lenient", "is_bsm"]
        for line in out["per_model"][cols].round(4).to_string(index=False).splitlines():
            print(f"      {line}")
    return out


def eda_report(store, members, splits=None, rows=None, partitions=None,
               use_extra_features=True, use_duration=True, quantiles=5,
               verbose=True):
    """The once-per-run briefing. Everything `fold_brief` is NOT.

    Parameters
    ----------
    partitions : `{name: row indices}` — the tuning/evaluation cut, if there is
        one. Printed as sizes and **checked for overlap**; see `partition_sizes`.
    rows : restrict every statistic to these store rows, the way `run_cv(rows=)`
        does. Pass the evaluation partition here to see the data the test will
        actually run on rather than the corpus it was drawn from.

    Returns a dict of frames, and prints the screenful. Composes `split_sizes`,
    `feature_dims`, `base_model_report` and `eda` rather than recomputing any of
    them, so there is one definition of each number in the notebook.
    """
    members = tuple(members)
    E, nref = store.err(members), store.nref
    if rows is not None:
        idx = np.asarray(rows, dtype=int)
        E, nref = E[idx], nref[idx]
    bsm = int((E.sum(0) / nref.sum()).argmin())

    out = {
        "splits": split_sizes(store, splits, rows),
        "features": feature_dims(store, members, use_extra_features, use_duration),
        "base_models": base_model_report(E, nref, members, bsm),
    }
    if partitions is not None:
        out["partitions"] = partition_sizes(len(store.nref), partitions)

    if verbose:
        print("=" * 78)
        print(f"RUN BRIEFING   {store.spec.name}   members: {'+'.join(members)}")
        print("=" * 78)
        print(f"\nsplits on the Hub ({len(store.nref):,} utterances pooled):")
        print(out["splits"].round(4).to_string(index=False))
        if "partitions" in out:
            print("\nrandom re-split (§7 pools the splits above and cuts afresh):")
            print(out["partitions"].round(4).to_string(index=False))
        d = out["features"]
        print(f"\ndesign matrix: {int(d['n_features'].sum()):,} features "
              f"in {len(d)} block(s)")
        print(d.round(4).to_string(index=False))
        scope = "all pooled rows" if rows is None else f"{len(nref):,} selected rows"
        print(f"\nbase models in the ensemble, on {scope}:")
        print(out["base_models"].round(4).to_string(index=False))
        b = out["base_models"]
        top = b.iloc[0]
        print(f"    -> the bar for a router is {top['model']}'s "
              f"{top['acc_strict']:.1%} strict / {top['acc_lenient']:.1%} lenient "
              f"selection accuracy, NOT 1/{len(members)} = {1 / len(members):.1%}")

    # The member-set briefing proper, on the same rows.
    out["ensemble"] = eda(store, members, splits=splits, rows=rows,
                          quantiles=quantiles, verbose=verbose)
    return out

def win_rates_tolerant(E, tolerances=(0, 1, 2), members=None):
    """Per-member acceptability rate at each word margin. One row per member.

    A member is *acceptable* on a row when its error is within `N` words of the
    row's minimum. `N = 0` is `win_rate_lenient` — ties count for every member
    that holds them, so it is at least `base_model_report`'s tie-shared
    `win_rate` and usually larger.

    Columns beyond `N = 0` do not sum to 1. `n_acceptable_mean` in `.attrs`
    records how many members clear each margin on the average row, which is the
    context those columns have to be read in: if three of three members are
    acceptable at `N = 2`, there is no routing decision left to get right.
    """
    E = np.asarray(E, dtype=np.float64)
    k = E.shape[1]
    names = list(members) if members is not None else [f"m{i}" for i in range(k)]
    if len(names) != k:
        raise ValueError(f"{len(names)} member names for {k} columns")

    best = E.min(axis=1, keepdims=True)
    share, sole = win_rates(E)
    out = {"model": names, "win_rate_shared": share, "sole_win_rate": sole}
    context = {}
    for n in tolerances:
        ok = E <= best + n
        out[f"win_rate_tol{n}"] = ok.mean(axis=0)
        context[n] = float(ok.sum(axis=1).mean())

    df = pd.DataFrame(out)
    df.attrs["n_acceptable_mean"] = context
    df.attrs["tolerances"] = tuple(tolerances)
    return df


def tolerance_profile(E, tolerances=(0, 1, 2)):
    """How much routing headroom survives each margin. One row per tolerance.

    `decisive_rate` is the share of rows with a single acceptable member — the
    rows where a router can actually be wrong. `n_acceptable_mean` is how many
    members clear the margin on the average row. Together they say whether a
    reported routing accuracy is measuring skill or measuring how many rows had
    no decision in them.
    """
    E = np.asarray(E, dtype=np.float64)
    best = E.min(axis=1, keepdims=True)
    rows = []
    for n in tolerances:
        ok = E <= best + n
        cnt = ok.sum(axis=1)
        rows.append({"tolerance_words": n,
                     "decisive_rate": float((cnt == 1).mean()),
                     "all_acceptable_rate": float((cnt == E.shape[1]).mean()),
                     "n_acceptable_mean": float(cnt.mean())})
    return pd.DataFrame(rows)

def split_durations(store, seconds_per_frame, splits=None, rows=None):
    """`split_sizes` plus encoder frames and an explicit hours estimate.

    `seconds_per_frame` is required and has no default: hours are derived from
    `store.n_frames`, not measured, and the multiplier belongs in the caller's
    provenance record. `hours_est` is named for what it is.
    """
    if not seconds_per_frame > 0:
        raise ValueError(f"seconds_per_frame must be positive, got {seconds_per_frame}")

    df = split_sizes(store, splits, rows)
    level = store.index.get_level_values("split").to_numpy()
    keep = np.ones(len(level), dtype=bool)
    if rows is not None:
        keep = np.zeros(len(level), dtype=bool)
        keep[np.asarray(rows, dtype=int)] = True

    frames = [float(store.n_frames[(level == s) & keep].sum()) for s in df["split"]]
    df["n_frames"] = frames
    df["hours_est"] = np.asarray(frames) * seconds_per_frame / 3600.0
    df["mean_seconds_est"] = np.where(
        df["n_utts"] > 0, np.asarray(frames) * seconds_per_frame / df["n_utts"].clip(lower=1), np.nan)
    df.attrs["seconds_per_frame"] = float(seconds_per_frame)
    return df


def dataset_table(store, partitions=None, n_features=None, seconds_per_frame=None):
    """The dataset-statistics rows of the manuscript (Table 1), one corpus.

    `partitions` is the tuner's `rows` dict — `tune` is the hyperparameter
    hold-out and `eval` the rows the 5x2 test runs on. The train / test sizes
    are those of ONE 5x2 fold: the evaluation rows are cut into complementary
    halves, so every fold trains on one half and scores the other, and the two
    halves differ by at most one utterance. `n_features` is the design width
    (`feature_dims(...)["n_features"].sum()`), passed in rather than derived so
    the table states what the arms were actually fitted on.
    """
    n = int(len(store.index))
    rows = [("Total clips", n),
            ("Features", int(n_features) if n_features is not None else np.nan)]
    if partitions is not None:
        tune, ev = len(partitions["tune"]), len(partitions["eval"])
        rows += [("Hyperparameter hold-out", int(tune)),
                 ("Evaluation rows (5x2)", int(ev)),
                 ("Train size (per fold)", int(ev // 2)),
                 ("Test size (per fold)", int(ev - ev // 2))]
    if seconds_per_frame:
        rows.append(("Hours (est.)", float(store.n_frames.sum() * seconds_per_frame / 3600)))
    rows.append(("Reference words", int(store.nref.sum())))
    # `object` dtype so the integer rows print as integers beside the float ones.
    return pd.DataFrame({"statistic": [r[0] for r in rows],
                         store.spec.name: pd.Series([r[1] for r in rows], dtype=object)})

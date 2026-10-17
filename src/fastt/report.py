__all__ = ['coverage', 'wer_table', 'probe_table', 'pooling_table']

import pandas as pd

from fastt.core import active_dataset
from fastt.hub import FasttHub


def coverage(repo_id=None, spec=None):
    """Which artefacts exist for which model, **for the active dataset**.

    Returns one row per model with three flags — the model config, the probe
    config, and the results JSON — plus how many pooled columns were actually
    pushed. A model missing `probe` or showing `n_pooled=1` was pushed under an
    older or narrower protocol and should be re-run for comparability.

    Every name here is resolved through the spec, so running this after
    `use_dataset("ami")` reports AMI coverage and nothing else — an empty table
    means that corpus has no extractions yet, not that the repo is broken.
    """
    hub = FasttHub(repo_id, spec=spec)
    spec = hub.spec
    configs = set(hub.configs())
    files = set(hub.files())

    rows = []
    for name in hub.model_names():
        n_pooled = None
        if spec.results_path(name) in files:
            info = hub.load_results(name)
            n_pooled = len([c for c in info["probe"]["pushed_columns"]
                            if c.startswith("pool_l")])
        rows.append({
            "model": name,
            "model_config": spec.model_config(name) in configs,
            "probe_config": spec.probe_config(name) in configs,
            "results_json": spec.results_path(name) in files,
            "n_pooled": n_pooled,
        })
    df = pd.DataFrame(rows, columns=["model", "model_config", "probe_config",
                                     "results_json", "n_pooled"])
    if df.empty:
        print(f"no models extracted for {spec.name!r} yet "
              f"(looking for {spec.model_prefix}* in {hub.repo_id})")
    return df.sort_values("model", ignore_index=True)


def wer_table(repo_id=None, models=None, which="norm", spec=None):
    """Corpus WER per split, one row per model, one column per split.

    which : `"norm"` (Whisper-normalised — comparable to the model cards) or
        `"raw"` (as emitted — comparable only among models that share a text
        convention, i.e. the CTC and ESPnet recipes).
    """
    hub = FasttHub(repo_id, spec=spec)
    spec = hub.spec
    files = set(hub.files())
    rows = []
    for name in (models or hub.model_names()):
        if spec.results_path(name) not in files:
            print(f"  skipping {name}: no {spec.results_path(name)}")
            continue
        info = hub.load_results(name)
        if not info.get("corpus_wer"):
            print(f"  skipping {name}: no WER of its own (feature-only model)")
            continue
        row = {"model": name, "model_id": info["model_id"]}
        row.update({r["split"]: r[f"wer_{which}"] for r in info["corpus_wer"]})
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Split order comes from the spec, so the table reads fit-splits-then-eval
    # for whichever corpus is active instead of a LibriSpeech-shaped list.
    cols = [c for c in spec.splits if c in df]
    df = df[["model", "model_id"] + cols]
    return df.sort_values(cols[-1], ignore_index=True) if cols else df


def probe_table(repo_id=None, models=None, target="wer", split=None,
                criterion="cv_r2", spec=None):
    """Best pooled feature set per model, chosen on the fit splits.

    For each model the winner is the column with the highest `criterion`
    (leave-one-out R^2 on validation), and the row reported is that column's
    score on `split` — the active dataset's first eval split when not given. Selection never touches the eval splits, so these numbers
    are directly comparable across models.

    Columns to read, in order of usefulness: `selectivity` (r2 above the
    shuffled-target control), `r2_beyond_length` (r2 after removing what
    duration alone explains), then `r2`.
    """
    hub = FasttHub(repo_id, spec=spec)
    spec = hub.spec
    split = split or spec.eval_splits[0]
    configs = set(hub.configs())
    rows = []
    for name in (models or hub.model_names()):
        if spec.probe_config(name) not in configs:
            print(f"  skipping {name}: no {spec.probe_config(name)} config — "
                  "re-run it to compare")
            continue
        df = hub.load_probe(name)
        df = df[(df["target"] == target) & (df["split"] == split)]
        if df.empty:
            continue
        best = df.loc[df[criterion].idxmax()]
        rows.append({
            "model": name,
            "pooling": best.get("pooling"),
            "layer": best.get("layer"),
            "n_features": best.get("n_features"),
            criterion: best[criterion],
            "r2": best["r2"],
            "selectivity": best.get("selectivity"),
            "r2_beyond_length": best.get("r2_beyond_length"),
            "r2_length_only": best.get("r2_length_only"),
            "pearson_r": best.get("pearson_r"),
            "spearman_r": best.get("spearman_r"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("selectivity", ascending=False, ignore_index=True)


def pooling_table(repo_id=None, models=None, target="wer", split=None,
                  metric="cv_r2", spec=None):
    """Every pooling for every model, as a model x pooling matrix.

    Useful for the question "is `mean` always the right choice?" — it usually
    is, but `std` sometimes wins, and `sum` winning is a warning sign that the
    probe is reading duration.
    """
    hub = FasttHub(repo_id, spec=spec)
    spec = hub.spec
    split = split or spec.eval_splits[0]
    configs = set(hub.configs())
    frames = []
    for name in (models or hub.model_names()):
        if spec.probe_config(name) not in configs:
            continue
        df = hub.load_probe(name)
        df = df[(df["target"] == target) & (df["split"] == split)]
        if not df.empty:
            frames.append(df.assign(model=name))
    if not frames:
        return pd.DataFrame()
    return (pd.concat(frames, ignore_index=True)
              .pivot_table(index="model", columns="pooling", values=metric))

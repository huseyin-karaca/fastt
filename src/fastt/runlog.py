"""A content-addressed cache for expensive notebook steps, mirrored to the records repository.

Every result is stored under a name hashed from what it depends on (`RunLog.digest`), so a re-run with the same
inputs reads instead of recomputing, and a changed input computes a new file beside the old one. `strict=True`
never computes: a result the cache does not hold raises `CacheMiss`.
"""

__all__ = ['RunLog', 'CacheMiss']

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fastt.hub import read_only
from fastt.pretty import state


class CacheMiss(LookupError):
    """A strict run needed a result the cache does not hold."""


class RunLog:
    """A content-addressed cache and timing log for expensive notebook steps.

    Parameters
    ----------
    name : the run's stem. Files land in `<dirname>/<name>/` and, when synced,
        under `results/<corpus>/runlog/<name>/` in the dataset repo.
    dirname : local directory. Defaults to the working directory.
    spec, repo_id : where the mirror lives; both default the way `FasttHub` does.
    sync : mirror to the Hub. `False` keeps everything local, which is the right
        setting on a machine whose disk persists between sessions.
    refresh : ignore every hit and recompute. The escape hatch for "I changed
        something the key does not cover" — and if you need it twice, the key is
        missing a field.
    strict : never compute — a result the cache lacks raises `CacheMiss`.
    """

    def __init__(self, name, dirname=None, spec=None, repo_id=None, sync=True,
                 refresh=False, verbose=True, strict=False):
        self.name = name
        self.dir = Path(dirname or ".") / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spec, self.repo_id = spec, repo_id
        self.sync, self.refresh, self.verbose = bool(sync), bool(refresh), bool(verbose)
        self.strict = bool(strict)
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = (json.loads(self.manifest_path.read_text())
                         if self.manifest_path.exists() else {})
        self._hub_files = None       # the repo listing, fetched once per session
        self._hub_manifest = None    # the Hub's copy of the manifest, read once
        self._pending = {}           # {path in repo: (local path, message)} for one commit
        self._batching = 0

    def __repr__(self):
        hits = sum(1 for e in self.manifest.values() if e.get("kind") == "frames")
        return (f"RunLog({self.name!r}, {self.dir}, {len(self.manifest)} entr(ies), "
                f"{hits} cached step(s), {'sync' if self.sync else 'local only'})")

    # ------------------------------------------------------------- plumbing --

    def hub(self):
        from fastt.hub import FasttHub                   # local: optional dep

        return FasttHub(self.repo_id, spec=self.spec)

    def _repo_path(self, filename):
        corpus = getattr(self.spec, "name", None)
        stem = f"results/{corpus}/runlog" if corpus else "results/runlog"
        return f"{stem}/{self.name}/{filename}"

    @staticmethod
    def digest(key):
        """Eight hex characters over a key dict. Order-insensitive, type-tolerant."""
        return hashlib.sha1(
            json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def _write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, default=str))

    def _on_hub(self, filename):
        """Whether the Hub lists this artefact. One listing per session, so a
        miss costs no request at all; `True` when the listing is unavailable,
        so an unknown Hub is tried rather than assumed empty."""
        if self._hub_files is None:
            files = getattr(self.hub(), "files", None)
            try:
                self._hub_files = set(files()) if files else False
            except Exception:                                  # noqa: BLE001
                self._hub_files = False
        return self._hub_files is False or self._repo_path(filename) in self._hub_files

    def _pull(self, path):
        """Fetch one artefact from the Hub into the local directory, if it is there."""
        if not self.sync or path.exists():
            return path.exists()
        if not self._on_hub(path.name):
            return False
        try:
            got = self.hub().pull_file(self._repo_path(path.name), path,
                                       verbose=False)
        except Exception as e:                                 # noqa: BLE001
            # A cache is an optimisation. A Hub that is down, private or simply
            # missing the file must degrade to a recompute, never to a crash.
            if self.verbose:
                print(f"  runlog: Hub read failed ({e}); computing locally")
            return False
        return got is not None and path.exists()

    def _push(self, path, message):
        """Queue one artefact for the Hub; sent by `flush`, at once unless batching."""
        if not self.sync or read_only() or not path.exists():
            return
        self._pending[self._repo_path(path.name)] = (path, message)
        if not self._batching:
            self.flush()

    def flush(self, message=None):
        """Send every queued artefact — as ONE commit where the Hub layer can.

        A seed's worth of per-arm frames is thirty files, and thirty commits
        for what is logically one write is what gets a repo rate-limited.
        """
        if not self._pending:
            return
        pending, self._pending = self._pending, {}
        hub = self.hub()
        try:
            if len(pending) > 1 and hasattr(hub, "push_files"):
                hub.push_files({repo: str(p) for repo, (p, _) in pending.items()},
                               message or f"{self.name}: {len(pending)} artefact(s)",
                               verbose=False)
            else:
                for repo, (p, msg) in pending.items():
                    hub.push_file(p, repo, msg, verbose=False)
        except Exception as e:                                 # noqa: BLE001
            if self.verbose:
                print(f"  runlog: Hub write failed ({e}); the local copies are kept")
            return
        if self._hub_files:
            self._hub_files |= set(pending)

    def mirror(self, message=None):
        """Push every local artefact of this run the Hub lacks, in one commit.

        The recovery from a push that failed mid-run — a Hub rate limit, a
        dropped connection. A cache hit never re-pushes, so a frame that missed
        its commit would otherwise stay local for exactly as long as the
        runtime lives and be refitted on the next machine. Called at the end of
        `per_arm` and `per_seed`; call it by hand once a rate limit has passed.
        Returns the paths sent. Does nothing when the Hub cannot be listed.
        """
        if not self.sync or read_only():
            return []
        self._hub_files = None                                  # list afresh
        local = [p for p in sorted(self.dir.iterdir())
                 if p.is_file() and p.name not in ("manifest.json", "manifest.hub.json")]
        missing = [p for p in local if not self._on_hub(p.name)]
        if self._hub_files is False or not missing:
            return []
        if self.verbose:
            print(f"  runlog: mirroring {len(missing)} local artefact(s) the Hub lacks")
        with self.batch():
            for p in missing:
                self._push(p, f"{self.name}: {p.name}")
            self._push(self.manifest_path, f"{self.name}: manifest")
        self.flush(message or f"{self.name}: mirror {len(missing)} artefact(s)")
        return [str(p) for p in missing]

    class _Batch:
        def __init__(self, log):
            self.log = log

        def __enter__(self):
            self.log._batching += 1

        def __exit__(self, *exc):
            self.log._batching -= 1
            if not self.log._batching:
                self.log.flush()
            return False

    def batch(self):
        """Context manager: hold every `_push` until the block ends, then one commit."""
        return RunLog._Batch(self)

    # ---------------------------------------------------------------- steps --

    def step(self, tag, **key):
        """Time a block and record it. Use as a context manager.

        ```python
        with log.step("load store"):
            store = EnsembleStore(...).load()
        ```

        Recording is all it does — a step is not cached, because the thing it
        wraps usually binds a name the cache cannot return. It is what makes the
        phase-level wall-clock table possible.
        """
        return _Step(self, tag, key)

    def record(self, entry, seconds, kind="step", **extra):
        """Write one manifest entry by hand.

        `entry` is the manifest key. A step uses its own tag; a cached frame
        uses `tag[digest]`, so `extra` may legitimately carry a `tag` of its own
        — which is why this parameter is not called `tag`.
        """
        self.manifest[entry] = {"kind": kind, "seconds": float(seconds),
                                "at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
        self._write_manifest()
        return self.manifest[entry]

    # ------------------------------------------------------------- frames --

    @staticmethod
    def expand_summary(summary, scores=None):
        """A picks frame rebuilt from its per-fold summary.

        The same `(model, repeat, fold, choice)` counts as the frame the summary
        was taken from, with `row` set to -1 and `errors` to NaN: enough for
        every share, distribution and switch-rate table, which is all the
        manuscript reads, and not for a per-utterance join. `scores` stamps
        `members` and `repeat_seed` on the rows when it is given.
        """
        idx = np.repeat(summary.index.values, summary["n"].values.astype(int))
        s = summary.loc[idx]
        out = pd.DataFrame({"model": s["model"].values, "repeat": s["repeat"].values,
                            "fold": s["fold"].values, "row": -1, "errors": np.nan,
                            "choice": s["choice"].values})
        if scores is not None:
            stamp = [c for c in ("members", "repeat_seed") if c in scores.columns]
            if stamp:
                per_fold = scores.drop_duplicates(["model", "repeat", "fold"])
                out = out.merge(per_fold[["model", "repeat", "fold", *stamp]],
                                on=["model", "repeat", "fold"], how="left")
        return out

    def _read_picks(self, picks_p, summary_p, scores):
        """The raw picks if they can be had, else the summary expanded; `None` if neither."""
        if self._pull(picks_p):
            return pd.read_parquet(picks_p)
        if self._pull(summary_p):
            return self.expand_summary(pd.read_parquet(summary_p), scores)
        return None

    @staticmethod
    def _arm_seconds(scores):
        """Fit plus predict wall-clock summed over the score frame's folds and arms."""
        cols = [c for c in ("fit_seconds", "predict_seconds") if c in scores.columns]
        if not cols:
            return float("nan")
        per_fold = scores.drop_duplicates(["model", "repeat", "fold"])
        return float(per_fold[cols].sum().sum())

    def frames(self, tag, fn, key=None, picks_summary=True, refresh=None,
               push_picks=True):
        """Cached `(fold_scores, fold_picks)`. Runs `fn()` only on a miss.

        `fn` must return the pair `run_cv` returns. `key` is everything that
        makes this call distinct — the study name, the arms, the seeds, the row
        count. It is digested into the filenames, so a changed key is a miss.

        `picks_summary` also writes the compact `(model, repeat, fold, choice)`
        counts; `push_picks=False` keeps the raw picks frame local and mirrors
        only the summary, for a run whose picks are millions of rows. A hit
        needs the scores and either picks file: with only the summary on the
        Hub, the picks come back expanded from it (`expand_summary`).
        """
        refresh = self.refresh if refresh is None else refresh
        key = dict(key or {})
        h = self.digest(key)
        scores_p = self.dir / f"{tag}_{h}_scores.parquet"
        picks_p = self.dir / f"{tag}_{h}_picks.parquet"
        summary_p = self.dir / f"{tag}_{h}_picks_summary.parquet"

        # The manifest entry is qualified by the digest, not by the tag alone.
        # Two configurations of one step have distinct parquet files already;
        # letting them share a manifest row would make the reported saving —
        # and the phase table — quote whichever ran last, which is a different
        # configuration's wall-clock presented as this one's.
        entry = f"{tag}[{h}]"

        if not refresh and self._pull(scores_p):
            scores = pd.read_parquet(scores_p)
            picks = self._read_picks(picks_p, summary_p, scores)
            if picks is not None:
                if self.verbose:
                    saved = self.manifest.get(entry, {}).get("seconds")
                    print(f"  runlog {state('HIT')}  {tag} [{h}] — {len(scores):,} score rows, "
                          f"nothing refitted"
                          + (f" (saving ~{saved / 60:.1f} min)" if saved else ""))
                return scores, picks

        if self.strict:
            raise CacheMiss(f"runlog {self.name}: {tag} [{h}] is not cached (strict run)")
        if self.verbose:
            print(f"  runlog {state('MISS')} {tag} [{h}] — computing")
        t0 = time.perf_counter()
        scores, picks = fn()
        secs = time.perf_counter() - t0
        self._write(entry, tag, h, key, scores, picks, secs,
                    picks_summary=picks_summary, push_picks=push_picks)
        if self.verbose:
            print(f"  runlog {state('WROTE')} {tag} [{h}] — {secs / 60:.1f} min, "
                  f"{len(scores):,} score rows")
        return scores, picks

    def _write(self, entry, tag, h, key, scores, picks, secs, picks_summary=True,
               push_picks=True, **extra):
        scores_p = self.dir / f"{tag}_{h}_scores.parquet"
        picks_p = self.dir / f"{tag}_{h}_picks.parquet"
        summary_p = self.dir / f"{tag}_{h}_picks_summary.parquet"
        scores.to_parquet(scores_p, index=False)
        picks.to_parquet(picks_p, index=False)
        self._push(scores_p, f"{self.name}: {tag} scores")
        if push_picks:
            self._push(picks_p, f"{self.name}: {tag} picks")
        files = [scores_p.name, picks_p.name]
        if picks_summary:
            summary = (picks.groupby(["model", "repeat", "fold", "choice"])
                       .size().rename("n").reset_index())
            summary.to_parquet(summary_p, index=False)
            self._push(summary_p, f"{self.name}: {tag} pick summary")
            files.append(summary_p.name)
        self.record(entry, secs, kind="frames", digest=h, key=key, tag=tag,
                    n_score_rows=int(len(scores)), n_pick_rows=int(len(picks)),
                    files=files, **extra)
        self._push(self.manifest_path, f"{self.name}: manifest")

    def per_seed(self, tag, seeds, fn, key=None, refresh=None):
        """Cached one repetition at a time. Returns concatenated `(scores, picks)`.

        `fn(seed)` must run ONE repetition — `run.evaluate(scheme=
        RepetitionSplit(seed), ...)` — and return the pair `run_cv` returns.
        Each seed is stored under its own key (`key` plus the seed), so:

        * a seed is never fitted twice — re-running with the same tuple reads
          every repetition back;
        * adding a seed to the tuple runs only the new one;
        * a partial run that died after three seeds resumes at the fourth.

        `repeat` is renumbered to the seed's position in `seeds` so the frames
        pivot exactly as a single `FiveByTwoSplit(seeds=seeds)` run would;
        `repeat_seed` still names the seed itself. Every arm is refitted on a
        miss; `per_arm` is the same cache one arm at a time.
        """
        scores, picks = [], []
        for i, seed in enumerate(seeds):
            with self.batch():
                s, p = self.frames(f"{tag}_seed{int(seed)}", lambda: fn(int(seed)),
                                   key={**(key or {}), "seed": int(seed)}, refresh=refresh)
            if s["repeat"].nunique() != 1:
                raise ValueError(f"fn({seed}) returned {s['repeat'].nunique()} "
                                 "repetitions; per_seed runs exactly one per call")
            scores.append(s.assign(repeat=i))
            picks.append(p.assign(repeat=i))
        self.mirror()
        return (pd.concat(scores, ignore_index=True),
                pd.concat(picks, ignore_index=True))

    # ------------------------------------------------------------ per arm --

    REFERENCES = "references"

    def per_arm(self, tag, seeds, estimators, fn, key=None, arm_keys=None,
                refresh=None):
        """Cached per **arm** and seed. Returns concatenated `(scores, picks)`.

        `fn(seed, estimators)` must run ONE repetition of the arms it is handed
        — `run.evaluate(estimators=..., scheme=RepetitionSplit(seed), ...)` —
        and return the pair `run_cv` returns, references included. `estimators`
        is the full `{arm: factory}`; `arm_keys[arm]` is what that arm's scores
        depend on given the folds (`HPTRun.arm_keys()`: the selected
        configuration and the arm's version), and `key` is everything shared —
        the partition, the reference-arm settings, the metric set.

        Every arm is cached under its own key, so changing one arm's search —
        or its code, by bumping its version — refits **that arm** on the folds
        and reads every other one back. The pairing the 5x2 tests need is
        untouched: the folds are a function of the seed and the rows, and
        `fold_arms` fits each arm independently on them. The parameter-free
        reference arms (`bsm`, `oracle`, the members, ...) form one more group,
        `references`, keyed on the shared `key` alone.

        A group that is absent under its key is looked for in the per-seed
        frames of earlier runs (`_adopt`): an old whole-run entry whose study
        selected the same configuration for this arm on the same rows is the
        same result, and its rows are copied across rather than refitted.
        `repeat` is renumbered to the seed's position in `seeds`.
        """
        refresh = self.refresh if refresh is None else refresh
        key, arm_keys = dict(key or {}), dict(arm_keys or {})
        estimators = dict(estimators)
        missing = [a for a in estimators if a not in arm_keys]
        if missing:
            raise ValueError(f"no arm key for {missing}; pass `arm_keys=` "
                             "(`HPTRun.arm_keys()`) for every estimator")
        if self.REFERENCES in estimators:
            raise ValueError(f"{self.REFERENCES!r} is the reference group's name")

        out_scores, out_picks = [], []
        for i, seed in enumerate(seeds):
            seed = int(seed)
            groups = {self.REFERENCES: {"arm": self.REFERENCES},
                      **{a: arm_keys[a] for a in estimators}}
            have, need, adopted = {}, [], []
            with self.batch():
                for g, gk in groups.items():
                    gkey = {**key, **gk, "seed": seed}
                    got = None if refresh else self._read_group(tag, seed, g, gkey)
                    if got is None and not refresh:
                        got = self._adopt(tag, seed, g, gkey, estimators)
                        if got is not None:
                            adopted.append(g)
                    if got is None:
                        need.append(g)
                    else:
                        have[g] = got
                if self.verbose:
                    n = len(groups)
                    word = state("HIT ") if not need else state("MISS")
                    print(f"  runlog {word} {tag} seed {seed} — {n - len(need)}/{n} "
                          f"group(s) cached"
                          + (f", {len(adopted)} adopted from earlier runs" if adopted else "")
                          + (f"; computing {', '.join(need)}" if need else ", nothing refitted"))
                if need and self.strict:
                    raise CacheMiss(f"runlog {self.name}: {tag} seed {seed}: {', '.join(need)} not cached "
                                    "(strict run)")
                if need:
                    fit = {a: estimators[a] for a in need if a in estimators}
                    t0 = time.perf_counter()
                    scores, picks = fn(seed, fit)
                    secs = time.perf_counter() - t0
                    if scores["repeat"].nunique() != 1:
                        raise ValueError(f"fn({seed}) returned {scores['repeat'].nunique()} "
                                         "repetitions; per_arm runs exactly one per call")
                    for g in need:
                        s, p = self._split(scores, picks, g, estimators)
                        if s.empty:
                            raise ValueError(f"fn({seed}) returned no rows for {g!r}")
                        gkey = {**key, **groups[g], "seed": seed}
                        h = self.digest(gkey)
                        stem = f"{tag}_seed{seed}_{g}"
                        self._write(f"{stem}[{h}]", stem, h, gkey, s, p,
                                    self._arm_seconds(s), picks_summary=False,
                                    group=g, fitted_with=list(fit))
                        have[g] = (s, p)
                    if self.verbose:
                        print(f"  runlog {state('WROTE')} {tag} seed {seed} — {secs / 60:.1f} min "
                              f"for {', '.join(need)}")
            order = [self.REFERENCES] + list(estimators)
            out_scores += [have[g][0].assign(repeat=i) for g in order]
            out_picks += [have[g][1].assign(repeat=i) for g in order]
        self.mirror()
        return (pd.concat(out_scores, ignore_index=True),
                pd.concat(out_picks, ignore_index=True))

    def _split(self, scores, picks, group, estimators):
        """One group's rows: the arm's own, or everything that is not an estimator."""
        if group == self.REFERENCES:
            fitted = set(estimators)
            m_s, m_p = ~scores["model"].isin(fitted), ~picks["model"].isin(fitted)
        else:
            m_s, m_p = scores["model"] == group, picks["model"] == group
        return scores[m_s].reset_index(drop=True), picks[m_p].reset_index(drop=True)

    def _read_group(self, tag, seed, group, gkey):
        h = self.digest(gkey)
        stem = f"{tag}_seed{seed}_{group}"
        scores_p = self.dir / f"{stem}_{h}_scores.parquet"
        if not self._pull(scores_p):
            return None
        scores = pd.read_parquet(scores_p)
        picks = self._read_picks(self.dir / f"{stem}_{h}_picks.parquet",
                                 self.dir / f"{stem}_{h}_picks_summary.parquet", scores)
        return None if picks is None else (scores, picks)

    def hub_manifest(self):
        """The Hub's copy of this run's manifest — entries written from other machines."""
        if self._hub_manifest is None:
            self._hub_manifest = {}
            staged = self.dir / "manifest.hub.json"
            if self.sync and self._on_hub("manifest.json"):
                try:
                    got = self.hub().pull_file(self._repo_path("manifest.json"), staged,
                                               verbose=False)
                    if got is not None and staged.exists():
                        self._hub_manifest = json.loads(staged.read_text())
                except Exception:                              # noqa: BLE001
                    pass
        return self._hub_manifest

    def _adopt(self, tag, seed, group, gkey, estimators):
        """A group's frames sliced out of an earlier whole-run entry, if one matches.

        Per-seed entries (`per_seed`, and every run before per-arm caching) key
        on the *study names*; a search that moved one arm renames the study and
        orphans the other arms' folds, though nothing about them changed. The
        `hpt` record of each named study still holds what that search selected
        for every arm and on which rows, so the per-arm key the old run *would*
        have written can be rebuilt and compared with this one. A match means
        the same configuration on the same folds — the same numbers — and the
        arm's rows are copied under the new key. The reference group matches on
        the rows alone.
        """
        from fastt.tuning import cv_arm_key, load_hpt_json, partition_identity

        legacy_tag = f"{tag}_seed{seed}"
        h = self.digest(gkey)
        stem = f"{tag}_seed{seed}_{group}"
        entries = {**self.hub_manifest(), **self.manifest}
        for name, entry in entries.items():
            if entry.get("kind") != "frames" or entry.get("tag") != legacy_tag:
                continue
            lkey = entry.get("key") or {}
            if group != self.REFERENCES and "arms" in lkey and group not in lkey["arms"]:
                continue
            studies = [v for k, v in lkey.items()
                       if (k == "study" or k.endswith("_study")) and isinstance(v, str)]
            shared = {k: v for k, v in lkey.items()
                      if k not in ("seed", "arms") and k != "study" and not k.endswith("_study")}
            for study in studies:
                payload = load_hpt_json(self.spec, study, self.dir.parent, self.sync)
                if not payload:
                    continue
                ident = payload.get("hpt") or {}
                if group == self.REFERENCES:
                    cand = {**shared, "partition": partition_identity(ident),
                            "arm": self.REFERENCES, "seed": seed}
                else:
                    res = (payload.get("results") or {}).get(group)
                    if res is None:
                        continue
                    cand = {**shared, "partition": partition_identity(ident),
                            **cv_arm_key(ident, group, res["params"]), "seed": seed}
                if self.digest(cand) != h:
                    continue
                ld = entry.get("digest") or name.split("[")[-1].rstrip("]")
                l_scores = self.dir / f"{legacy_tag}_{ld}_scores.parquet"
                if not self._pull(l_scores):
                    continue
                scores = pd.read_parquet(l_scores)
                picks = self._read_picks(self.dir / f"{legacy_tag}_{ld}_picks.parquet",
                                         self.dir / f"{legacy_tag}_{ld}_picks_summary.parquet",
                                         scores)
                if picks is None:
                    continue
                # The old entry's own arm list, so an arm dropped since then is
                # not mistaken for a reference.
                s, p = self._split(scores, picks, group,
                                   set(estimators) | set(lkey.get("arms") or []))
                if s.empty:
                    continue
                self._write(f"{stem}[{h}]", stem, h, gkey, s, p, self._arm_seconds(s),
                            picks_summary=False, group=group, adopted_from=name,
                            adopted_study=study)
                return s, p
        return None

    # --------------------------------------------------------------- report --

    def timings(self):
        """Every recorded step, longest first. The phase wall-clock table."""
        rows = [{"step": v.get("tag", k), "entry": k,
                 "kind": v.get("kind", "step"),
                 "seconds": v.get("seconds", np.nan),
                 "minutes": v.get("seconds", np.nan) / 60.0,
                 "digest": v.get("digest", ""), "at": v.get("at", "")}
                for k, v in self.manifest.items()]
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out = out.sort_values("seconds", ascending=False, ignore_index=True)
        total = out["seconds"].sum()
        out["share"] = out["seconds"] / total if total else np.nan
        return out


class _Step:
    """`RunLog.step`'s context manager. Records wall-clock even when the block raises."""

    def __init__(self, log, tag, key):
        self.log, self.tag, self.key = log, tag, key

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        secs = time.perf_counter() - self.t0
        self.log.record(self.tag, secs, kind="step",
                        failed=exc_type is not None, **self.key)
        if self.log.verbose:
            print(f"  runlog step {self.tag}: {secs:.1f}s"
                  + (" (raised)" if exc_type is not None else ""))
        return False

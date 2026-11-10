# The package

`pip install "fastt[paper] @ git+https://github.com/huseyin-karaca/fastt"` installs the package `fastt`.

## The paper

| Module | What |
|---|---|
| `configs` | every configuration the paper reports, fixed and named: `MAIN[<corpus>]` (the trio, the protocol, the seeds), the spaces of Table 2 (`SPACES`), the tie-breaks of the search (`SELECTED`) |
| `experiment` | the main experiment on one corpus (`MainExperiment`): the search, the 5×2 cross-validation, Tables 1, 2, 3 and 6, the record; `python -m fastt.experiment <corpus>` |
| `paper` | the manuscript's tables from the records, shown and written as LaTeX (`ManuscriptTables`) |
| `ablation`, `timing` | Table 5 (`LossAblation`, the transformation trials) and Table 4 (`Timing`) |
| `rebuild` | level 3: the plan per toolkit group, the pass over a corpus, the comparison with the published outputs |
| `records` | the records the notebooks read, from the Hub or from `FASTT_RECORDS_DIR` |

## The method and the baselines

| Module | What |
|---|---|
| `estimators` | the feature transformations (diagonal gating, linear, low-rank, nonlinear), the expected-WER objective, FASTT-Boosted and FASTT-Alternating, and the learned baselines (raw and SelectKBest selectors, the Hard-Selectors, the QE-Router) |
| `tuning` | every method as a searchable model, its space, the hold-out search and the names its studies are stored under |
| `crossval` | the splits (5×2, k-fold, the hold-out), the metrics, one fold of every method, Table 3 |
| `significance` | the paired 5×2 cv tests of Table 6 (Dietterich's t, Alpaydın's F) and their table |
| `rover` | the fusion baselines: weighted and confidence-weighted ROVER, CN-MBR, Score-Fusion |
| `routing`, `probing` | the ANOVA F-value filter of SelectKBest; the ridge probes from which CN-MBR and Score-Fusion estimate confidences |

## Data and experts

| Module | What |
|---|---|
| `core` | the corpus registry (`DatasetSpec`): splits, shards, the trio's Hub configurations; the Hub repositories (`FASTT_HUB`, `FASTT_RECORDS`) |
| `hub`, `data` | the dataset repository named the active corpus's way, a local folder in the same layout standing in for it; the audio bases |
| `ensemble` | `EnsembleStore`: the trio's features and error counters for one corpus, cached locally (`FASTT_CACHE`); the oracle arithmetic |
| `scoring` | the WER scorer, with Whisper's English normaliser |
| `extractor`, `models.*`, `pooling` | one extractor per expert: decode, word-error counters, the pooled encoder states |
| `latency`, `deploy` | the per-expert clocks of Table 4 and the toolkit group of each expert; a selector deployed on raw audio |
| `runlog` | a content-addressed cache of the expensive steps (one fold of one method), stored with the records; a strict mode that never computes |
| `eda`, `report`, `pretty` | the dataset table and per-corpus summaries; tables in a notebook and in a terminal |

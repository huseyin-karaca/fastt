# Reproducibility report

What we have checked, against what, and what is still to be run. This page states only what was measured.

## At a glance

| What we ran | Result |
|---|---|
| **Level 1**, every main notebook, from the public package and the public dataset | **identical to the paper**: every cell of Tables 1, 3 and 6 on all five corpora, and the selected configuration of every method |
| **Level 1**, the `ablation` and `timing` notebooks | **identical**: Table 5b from the stored folds; Table 4 recomposed from the stored per-expert clocks |
| **Level 1**, the `manuscript_tables` notebook | **byte for byte**: all nine LaTeX tables equal to the ones in `tables/`, which the manuscript inputs |
| **Level 2** and **level 3** | not yet re-run from the public package |

## Level 1: exact

Level 1 recomputes every table from the stored hyperparameter searches and the stored 5×2 folds. It trains nothing,
so it reproduces **exactly**: we compared every numeric cell of each main notebook's record with the record the paper
was built from, and every LaTeX table byte for byte, on all five corpora. The `manuscript_tables` notebook writes the
paper's tables to `tables/`, so after a run `git diff tables/` is the check.

## Levels 2 and 3

Level 2 refits every method on every fold from the published features, and level 3 does the same on features rebuilt
from the audio. Both run the code that produced the paper's records, but we have not yet re-run them from the public
package on a fresh machine; this page will report them when we have. Until then, read their results against the
fold-to-fold standard deviations in the main notebooks rather than against the last digit.

## Hardware

| Level | Hardware | Time |
|---|---|---|
| 1 | any CPU | about ten seconds per notebook once the data is downloaded |
| 2 | a CPU | hours per corpus |
| 3 | a GPU for `extract`, one runtime per toolkit group | minutes to an hour per expert and corpus |

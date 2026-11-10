# The notebooks

| Notebook | What it reproduces | Open |
|---|---|---|
| `main_librispeech` | Tables 1, 2, 3 and 6 on LibriSpeech | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/main_librispeech.ipynb) |
| `main_voxpopuli` | the same on VoxPopuli | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/main_voxpopuli.ipynb) |
| `main_ami_sdm` | the same on AMI (single distant microphone) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/main_ami_sdm.ipynb) |
| `main_earnings22` | the same on Earnings-22 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/main_earnings22.ipynb) |
| `main_peoples_speech` | the same on People's Speech | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/main_peoples_speech.ipynb) |
| `ablation` | the feature transformation and the training objective (Table 5) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/ablation.ipynb) |
| `timing` | the inference cost (Table 4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/timing.ipynb) |
| `manuscript_tables` | every table of the paper, as LaTeX in `tables/` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/manuscript_tables.ipynb) |
| `extract` | the experts' outputs, rebuilt from the audio (level 3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/fastt/blob/main/notebooks/extract.ipynb) |

Every configuration is fixed in `fastt.configs` and imported by its notebook; the notebooks hold the calls and the
results, the package holds the code. They are committed with the outputs of a level-1 run.

## A main notebook, section by section

| § | What it does | Shows |
|---|---|---|
| — | `MainExperiment(MAIN["<corpus>"], level=LEVEL)`: the corpus, its trio, the protocol and the seeds | |
| 1 | the hyperparameter spaces of every method (`fastt.configs.SPACES`) | **Table 2** |
| 2 | the three experts' error counters (and, from level 2, their features); a check that a sample of the stored counters reproduces from the stored transcripts | |
| 3 | the search: a random 25 % of the clips is held out, and every method's hyperparameters are chosen there from 10 random draws, each fitted on 70 % of the hold-out and scored on the rest | **Table 1**, the selected configurations |
| 4 | the 5×2 cross-validation on the other 75 %: each of five repetitions halves the clips with its own seed, every method is trained on one half and tested on the other, then the other way round | |
| 5 | every method's corpus WER, mean utterance WER, gap closed and selection accuracy at 0, 1 and 2 words of tolerance, as mean ± sd over the ten folds; how often each method picks each expert | **Table 3** |
| 6 | every method against the FASTT anchor on the same folds: Dietterich's 5×2 cv t test and Alpaydın's combined 5×2 cv F test, on corpus WER and on mean utterance WER | **Table 6** |
| 7 | the record: everything above in one JSON file | |

When several configurations tie on the hold-out WER, the one the paper's run kept is used
(`fastt.configs.SELECTED`), after checking that it is among the tied best.

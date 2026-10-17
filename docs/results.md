---
hide:
  - navigation
---

# Results

<p class="ft-lede">FASTT is compared with the best single model, the per-utterance oracle, four fusion ensembles that
decode every expert, three learned routers and the classical ways of handling high-dimensional features, on five
English corpora, each with its own trio of experts. Every number on this page is generated from the paper's
tables.</p>

## Setup

Each corpus is split once, with a fixed seed: a random quarter of the clips is held out for the hyperparameter
search, where every method draws ten configurations at random from its space with the same sampler and seed, each
fitted on 70 % of the hold-out and scored on the other 30 %. The other 75 % of the clips are never seen by the search:
on them, every method is trained and tested on the same five repetitions of two-fold cross-validation (5×2), and the
tables give the mean over the ten test folds. Every method is compared fold by fold with the corpus's *FASTT anchor*,
the better of the two FASTT variants on that corpus, by Alpaydın's combined 5×2 cv F test.

--8<-- "snippets/table_datasets.html"

On four corpora the experts are trained on LibriSpeech only, so these corpora are out of domain for all of them, and
their ranking changes from corpus to corpus. LibriSpeech itself is the control: its trio is three general-purpose
models for which read speech is in domain, and one of them is expected to dominate.

| Corpus | The three experts |
|---|---|
| LibriSpeech | Parakeet-TDT-0.6b-v2, Canary-Qwen-2.5b, Whisper-large-v3 |
| VoxPopuli | ESPnet Conformer-RNN-T, ESPnet Transformer, data2vec-audio-base |
| AMI (SDM) | ESPnet Conformer-RNN-T, ESPnet Transformer, wav2vec2-large-960h |
| Earnings-22 | data2vec-audio-base, ESPnet Conformer-RNN-T, wav2vec2-large-960h |
| People's Speech | data2vec-audio-base, ESPnet Transformer, wav2vec2-base-960h |

The methods compared:

- **References.** *Best-Single-Model*, the expert with the lowest WER on the training half, used for every clip;
  *Random-Ensemble*, an expert drawn at random for every clip; the *oracles*, the expert with the fewest errors on each
  clip, and the same with a tolerance of one or two words (another expert replaces the best single model only if it
  saves more than one or two words).
- **Fusion ensembles**, which decode every expert: *ROVER (weighted)*, *ROVER (confidence-weighted)*, *CN-MBR*
  (confusion-network minimum Bayes risk) and *Score-Fusion*.
- **Learned routers** on the same features, without the FASTT objective: *Hard-Selector (plain CE)*, an arg min
  classifier trained with cross-entropy against the best expert on exactly FASTT's features and transformation;
  *Hard-Selector (cost-sensitive CE)*, the same with the cross-entropy weighted by the cost of a wrong choice; and the
  *QE-Router*, which predicts each expert's error count and routes to the smallest estimate.
- **Classical feature handling**, with FASTT's objective: *Raw XGBoost* and *Raw SDT* on all the features, Raw XGBoost
  being exactly the AdaSTT selector, and *SelectKBest XGBoost* and *SelectKBest SDT* after an ANOVA F-value filter.
- **FASTT**: *FASTT-Boosted* (boosted soft decision trees) and *FASTT-Alternating* (XGBoost), each with its
  transformation chosen by the search.

## Main results

--8<-- "snippets/results_main.md"

--8<-- "snippets/fig_gap.html"

--8<-- "snippets/table_main.html"

The share of the oracle gap each method closes, and how often it picks a best expert:

--8<-- "snippets/table_main_gc.html"

### Against the baselines

--8<-- "snippets/results_baselines.md"

### Significance

Every method against the corpus's FASTT anchor on the same ten folds:

--8<-- "snippets/table_significance.html"

## Cost

Measured on VoxPopuli, one clip at a time on one GPU, with each expert's encoder and decoder timed separately and
every system composed from those clocks and the main run's choices.

--8<-- "snippets/results_cost.md"

--8<-- "snippets/fig_cost.html"

--8<-- "snippets/table_systems.html"

Under greedy decoding, the cost of every method of the main table, with the time it takes to train its selector:

--8<-- "snippets/table_overhead.html"

The clocks the systems are composed from:

--8<-- "snippets/table_clocks.html"

## Ablation

### The feature transformation

An extended exploration on the hold-out data of the main search, with the same experts, features and partition,
gives the best hold-out WER each transformation reaches for each FASTT variant.

--8<-- "snippets/results_transform.md"

--8<-- "snippets/table_transform.html"

### The training objective

The same architecture, features, search budget, partition and folds, trained once with cross-entropy against the best
expert and once with the expected-WER objective.

--8<-- "snippets/results_ablation.md"

--8<-- "snippets/fig_objective.html"

--8<-- "snippets/table_objective.html"

## Hyperparameter spaces

--8<-- "snippets/table_spaces.html"

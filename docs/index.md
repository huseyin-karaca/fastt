---
template: home.html
title: Home
hide:
  - navigation
  - toc
---

## No single recognizer wins

Each corpus in the study comes with three pretrained speech recognizers, the experts. None of them is best on every
clip. Choosing the best of the three for each clip in hindsight, the per-utterance oracle, gives a much lower word
error rate than the best single expert (corpus WER on the test folds):

--8<-- "snippets/home_stats.html"

Reaching part of that headroom is a selection problem, and it is hard for three reasons. The features that could
tell the experts apart are high-dimensional, of the same order as the number of labelled clips, so a selector
trained on them directly overfits. What matters is not which expert is right on a clip but how much a wrong choice
costs, and on many clips the experts tie. And transcribing with every expert to keep the best output, as fusion
ensembles do, pays every decoder on every clip.

## The idea

<ul class="ft-points">
<li><b>Read what the experts compute</b><span>The features are the experts' own encoder states, averaged over the clip, with the clip's duration: the encoders run anyway, so no separate front-end runs at inference.</span></li>
<li><b>Learn the feature space with the selector</b><span>A learned transformation, from a feature-wise gate to a nonlinear bottleneck, reshapes the features for the choice, and is trained together with the selector.</span></li>
<li><b>Train on the error you pay</b><span>The selector is trained to minimise the expected word error rate under its own choice probabilities, so a clip counts in proportion to what a wrong choice would cost.</span></li>
<li><b>All encoders, one decoder</b><span>The choice is made from the encoder outputs, before any decoder runs; only the chosen expert's decoder runs.</span></li>
</ul>

The [method page](method.md) gives the objective, the transformations and the two ways of training them with a
selector: boosted soft decision trees by backpropagation, and gradient-boosted trees by alternating updates.

## Results at a glance

--8<-- "snippets/home_result.md"

--8<-- "snippets/fig_gap.html"

<div class="ft-cards">
<a class="ft-card" href="method/"><b>Method</b><span>The problem, the features, the learned transformation, the objective and the two selectors.</span></a>
<a class="ft-card" href="results/"><b>Results</b><span>Five corpora against fusion ensembles and routers, the significance tests, the cost, the ablations.</span></a>
<a class="ft-card" href="reproduce/"><b>Reproduce</b><span>Every table from the stored results on a CPU, every model refitted, the features rebuilt from the audio.</span></a>
<a class="ft-card" href="data/"><b>Data</b><span>The audio, the experts' transcripts, error counts and encoder features, and the stored results, on the Hugging Face Hub.</span></a>
</div>

## Citation

```bibtex
@article{karaca2026fastt,
  title  = {{FASTT}: Feature-Adaptive Speech-to-Text Model Selection},
  author = {Karaca, Huseyin and Namli, A. Samil and Kozat, Suleyman S.},
  year   = {2026},
  note   = {Under review at Expert Systems with Applications}
}
```

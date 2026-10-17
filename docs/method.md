---
hide:
  - navigation
---

# Method

<p class="ft-lede">FASTT chooses one pretrained speech recognizer for each clip. It reads the experts' own encoder
states, reshapes them with a learned transformation, and scores the experts with a selector; the transformation and
the selector are trained together to minimise the word error rate the choice will incur. Only the chosen expert's
decoder runs.</p>

## The problem

Take a pool of \(L\) pretrained recognizers, the experts \(E_1, \dots, E_L\). On clip \(\mathbf{s}_i\) with reference
transcript \(y_i^{\mathrm{ref}}\), expert \(E_j\) produces the transcript \(y_{i,j}\) with word error rate
\(\mathrm{WER}_{i,j}\). The clip is described by a feature vector \(\mathbf{z}_i \in \mathbb{R}^p\), and a selection
policy \(\pi\) maps it to one expert. Its cost over \(N\) clips is

\[
J(\pi) \;=\; \frac{1}{N} \sum_{i=1}^{N} \mathrm{WER}\big(y_{i,\pi(\mathbf{z}_i)},\, y_i^{\mathrm{ref}}\big).
\]

Two policies bound the problem. The **best single model** sends every clip to the expert with the lowest WER on
labelled data; every adaptive policy has to beat it. The **per-utterance oracle**,
\(\pi^\star(\mathbf{z}_i) = \arg\min_j \mathrm{WER}_{i,j}\), reads the reference, so it cannot be deployed, but no
policy that picks one expert per clip does better. The results report how much of the distance between the two a
method covers, the *gap closed*,

\[
\mathrm{GC} \;=\; \frac{\mathrm{WER}_{\text{BSM}} - \mathrm{WER}_{\text{method}}}{\mathrm{WER}_{\text{BSM}} - \mathrm{WER}_{\text{oracle}}},
\]

and its *selection accuracy*, the share of clips routed to an expert with the lowest WER on that clip. How much of
the gap is recoverable is a property of the pool: where one expert is best on nearly every clip, there is little to
select.

## The features

FASTT accepts any fixed-length description of the clip. In the paper, \(\mathbf{z}_i\) is the mean over time of the
final-layer encoder states of each of the three experts, concatenated, with the logarithm of the clip's duration:

\[
\mathbf{z}_i \;=\; \big[\,\bar{\mathbf{h}}^{(1)}_i;\; \bar{\mathbf{h}}^{(2)}_i;\; \bar{\mathbf{h}}^{(3)}_i;\; \log d_i\,\big],
\qquad
\bar{\mathbf{h}}^{(j)}_i = \frac{1}{T_{i,j}} \sum_{t=1}^{T_{i,j}} \mathbf{h}^{(j)}_{i,t}.
\]

Its width \(p\) is one plus the sum of the experts' encoder widths, between 1,793 and 3,329 depending on the corpus,
which is of the same order as the number of clips a selector is trained on. The encoders have to run anyway to
decode, so the features cost nothing beyond the encoder passes, and they are available before any decoder runs.

## The learned transformation

A selector trained directly on thousands of correlated features with a few thousand labelled clips has to find the
informative directions on its own. FASTT puts a parametric map in front of it,

\[
\mathbf{u}_i \;=\; \mathbb{T}_\theta(\mathbf{z}_i),
\]

and learns \(\theta\) with the selector, under the same objective. The family of \(\mathbb{T}_\theta\) is a
hyperparameter, with the identity as one of its values:

/// tab | Identity

\[
\mathbf{u}_i = \mathbf{z}_i
\]

The raw features, for comparison: no parameters.

///

/// tab | Diagonal gating

\[
\mathbf{u}_i = \mathbf{g} \odot \mathbf{z}_i,
\qquad
\lambda_1 \lVert \mathbf{g} \rVert_1 + \lambda_2 \lVert \mathbf{g} \rVert_2^2
\]

One weight per feature, with an \(\ell_1\)/\(\ell_2\) penalty: a differentiable feature selection, \(p\) parameters.

///

/// tab | Linear

\[
\mathbf{u}_i = \mathbf{W}\mathbf{z}_i + \mathbf{b},
\qquad
\mathbf{W} \in \mathbb{R}^{d' \times p}
\]

A full projection that mixes the features; with \(d' < p\) it also reduces the dimension.

///

/// tab | Low-rank

\[
\mathbf{u}_i = \mathbf{W}_2 \mathbf{W}_1 \mathbf{z}_i,
\qquad
\mathbf{W}_1 \in \mathbb{R}^{r \times p},\;
\mathbf{W}_2 \in \mathbb{R}^{d' \times r}
\]

The projection factored through a bottleneck of width \(r\), fewer parameters than the full one.

///

/// tab | Nonlinear

\[
\mathbf{u}_i = \mathbf{W}_2\, \sigma(\mathbf{W}_1 \mathbf{z}_i)
\]

A bottleneck with an element-wise nonlinearity \(\sigma\), for interactions between features.

///

## The objective

The selector \(f_\psi\) maps the transformed features to one logit per expert, \(\mathbf{q}_i = f_\psi(\mathbf{u}_i)\),
and a softmax turns them into choice probabilities \(p_{i,j}\). FASTT minimises the expected word error rate under
those probabilities:

\[
\min_{\theta,\,\psi}\; \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{L} p_{i,j}\, \mathrm{WER}_{i,j},
\qquad
p_{i,j} = \frac{\exp(q_{i,j})}{\sum_{m} \exp(q_{i,m})}.
\]

The error rates are constants, computed once by running every expert on every training clip. At inference the
expert with the largest probability is chosen; the soft objective stands in for that choice because the error of an
arg max has no gradient, while the expected error is smooth. It is also cost sensitive: unlike a cross-entropy loss
against the best expert, it weights every candidate by its actual error, so it ignores clips on which the experts
tie and concentrates on clips where a wrong choice is expensive. As the probabilities approach a one-hot vector the
two coincide. The [ablation](results.md#ablation) trains the same architectures with cross-entropy instead.

## Two selectors

The transformation can be trained with any selector. The paper uses two, one on each optimisation path.

### FASTT-Boosted: backpropagation

When the selector is differentiable, \(\theta\) and \(\psi\) are trained together by backpropagation. FASTT-Boosted is
an additive ensemble of \(B\) soft decision trees, each with a transformation of its own:

\[
\mathbf{q}_i \;=\; \sum_{b=1}^{B} h_b\big(\mathbb{T}_{\theta_b}(\mathbf{z}_i)\big).
\]

The rounds are fitted one after another: each round's tree and transformation are trained jointly on the expected
WER of the ensemble so far, with an \(\ell_1\) penalty on \(\theta_b\), so that different rounds can attend to
different combinations of the features.

### FASTT-Alternating: gradient-boosted trees

Gradient-boosted trees are piecewise constant in their input, so no gradient reaches \(\theta\) through them. FASTT
trains XGBoost on the expected WER through a custom objective, with the gradient and a diagonal Hessian of the
per-clip loss \(\mathcal{L}_i = \sum_j p_{i,j} \mathrm{WER}_{i,j}\):

\[
g_{i,j} = p_{i,j}\big(\mathrm{WER}_{i,j} - \mathcal{L}_i\big),
\qquad
h_{i,j} = p_{i,j}\,(1 - 2p_{i,j})\big(\mathrm{WER}_{i,j} - \mathcal{L}_i\big),
\]

with negative Hessian entries clipped to a small positive constant. It then alternates for a fixed number of
iterations:

1. **Selector update.** With \(\theta\) fixed, transform the features and fit the trees on them.
2. **Surrogate fit.** Fit a linear map \(\tilde f_\phi(\mathbf{u}) = \mathbf{A}\mathbf{u} + \mathbf{c}\) by least
   squares to the trees' own logits: a differentiable stand-in for how the fixed trees respond to the features.
3. **Transformation update.** With \(\phi\) fixed, descend the expected WER at the logits
   \(\tilde f_\phi(\mathbb{T}_\theta(\mathbf{z}_i))\), plus the transformation's penalty, in \(\theta\).

The trees are refitted on the updated features in the next iteration. The number of iterations, and the steps and
learning rates of the surrogate and the transformation, are hyperparameters.

## Convergence

For a fixed transformation, both selector training procedures converge to a first-order stationary point of the
training objective under mild smoothness conditions: the boosted trees with a descent inequality at every round, the
neural selector by the standard non-convex results for gradient descent and SGD. For the differentiable path the
same argument covers \(\theta\) and \(\psi\) together; for the alternating path, whose transformation update descends
a surrogate, a guarantee for the joint system is open. The paper gives the derivations.

## What runs at inference

--8<-- "snippets/fig_pipeline.html"

| | Encoders | Selector | Decoders |
|---|---|---|---|
| Best single model | one | — | one |
| Decode every expert, then fuse | all three | — | all three |
| FASTT | all three | microseconds | one |

FASTT adds the other experts' encoders and the selector to the cost of one expert, and saves every decoder but one.
How that balances depends on how much of each expert's time is spent decoding, which grows with the beam width; the
[cost results](results.md#cost) measure it.

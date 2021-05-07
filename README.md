<img src="./images/spins.png" width="400px"></img>

## Deep Implicit Attention

Experimental implementation of deep implicit attention in PyTorch.

**Summary:** Using deep equilibrium models to implicitly solve a set of self-consistent mean-field equations of a random Ising model implements attention as a collective response 🤗 and provides insight into the transformer architecture, connecting it to mean-field theory, message-passing algorithms, and Boltzmann machines.

**Blog post: [Deep Implicit Attention: A Mean-Field Theory Perspective on Attention Mechanisms](https://mcbal.github.io/post/deep-implicit-attention-a-mean-field-theory-perspective-on-attention-mechanisms/)**

## Mean-field theory framework for transformer architectures

Transformer architectures can be understood as particular approximations/parametrizations of the blueprint mean-field description of a vector Ising model being probed by incoming data `x_i`:
```
z_i = sum_j J_ij z_j + f(z_i) + x_i
```
where `f` is a neural network acting on every vector `z_i` and the `z_i` are solved for iteratively. Explicit transformers take the couplings `J_ij` to depend on `x_i` parametrically and consider the fixed-point equation above as a single-step update equation.

### `DEQMLPMixerAttention`

A deep equilibrium version of MLP-Mixer transformer attention (https://arxiv.org/abs/2105.02723, https://arxiv.org/abs/2105.01601):
```
z_i = g({z_j}) - f(z_i) + x_i
```
where `g` is an MLP acting across the sequence dimension instead of
the feature dimension (so across patches). The network `f` acts
across the feature dimension (so individually on every sequence).

Compared to a vanilla softmax attention transformer module, the
sum over couplings has been "amortized" and parametrized by an MLP.
The fixed-point variables z_i's are also fed straight into the
feed-forward self-correction term. One could feed `spin_mean_mf`
instead to fully mimic the residual connection in the explicit
MLP-Mixer architecture.

### `DEQVanillaSoftmaxAttention`

A deep equilibrium version of vanilla softmax transformer attention (https://arxiv.org/abs/1706.03762):
```
z_i = sum_j J_ij z_j - f(z_i) + x_i
```
where
```
J_ij = [softmax(X W_Q W_K^T X^T / sqrt(dim))]_ij
```
Compared to the explicit vanilla softmax attention transformer module,
there's no values and the fixed-point variables `z_i`'s are fed straight
into the feed-forward self-correction term.

### `DEQMeanFieldAttention`
Fast and neural deep implicit attention as introduced in https://mcbal.github.io/post/deep-implicit-attention-a-mean-field-theory-perspective-on-attention-mechanisms/.

Schematically, the fixed-point mean-field equations including the Onsager self-correction term look like:
```
z_i = sum_j J_ij z_j - f(z_i) + x_i
```
where `f` is a neural network parametrizing the self-correction term for
every site and `X_i` denote the input injection or magnetic fields applied
at site `i`. Mean-field results are obtained by dropping the self-
correction term. This model generalizes the current generation of transformers in that its couplings are free parameters independent of the incoming data `x_i`.

### `DEQAdaTAPMeanFieldAttention`

Slow and explicit deep implicit attention as introduced in https://mcbal.github.io/post/deep-implicit-attention-a-mean-field-theory-perspective-on-attention-mechanisms/.

Ising-like vector model with multivariate Gaussian prior over spins. Generalization of the application of the adaptive TAP mean-field approach
from a system of binary/scalar spins to vector spins. Schematically, the
fixed-point mean-field equations including the Onsager term look like:
```
S_i ~ sum_j J_ij S_j - V_i S_i + x_i
```
where the V_i are self-corrections obtained self-consistently and `x_i`
denote the input injection or magnetic fields applied at site `i`. The
linear response correction step involves solving a system of equations,
leading to a complexity ~ `O(N^3*d^3)`. Mean-field results are obtained
by setting `V_i = 0`.

Given the couplings between spins and a prior distribution for the single-
spin partition function, the adaptive TAP framework provides a closed-form
solution in terms of sets of equations that should be solved for a fixed
point. The algorithm is related to expectation propagation (see Section
4.3 in https://arxiv.org/abs/1409.6179) and boils down to matching the
first and second moments assuming a Gaussian cavity distribution.

## Setup

Install package in editable mode:

```bash
$ pip install -e .
```

Run tests with:

```bash
$ python -m unittest
```

## References

### Selection of literature
On variational inference, iterative approximation algorithms, expectation propagation, mean-field methods and belief propagation:
- [Expectation Propagation](https://arxiv.org/abs/1409.6179) (2014) by Jack Raymond, Andre Manoel, Manfred Opper

On the adaptive Thouless-Anderson-Palmer (TAP) mean-field approach in disorder physics:
- [Adaptive and self-averaging Thouless-Anderson-Palmer mean-field theory for probabilistic modeling](https://link.aps.org/doi/10.1103/PhysRevE.64.056131) (2001) by Manfred Opper and Ole Winther


On Boltzmann machines and mean-field theory:
- [Efficient Learning in Boltzmann Machines Using Linear Response Theory](https://doi.org/10.1162/089976698300017386) (1998) by H. J. Kappen and
F. B. Rodríguez
- [Mean-field theory of Boltzmann machine learning](https://link.aps.org/doi/10.1103/PhysRevE.58.2302) (1998) by Toshiyuki Tanaka

On deep equilibrium models:
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) (2019) by Shaojie Bai, Zico Kolter, Vladlen Koltun
- [Chapter 4: Deep Equilibrium Models](https://implicit-layers-tutorial.org/deep_equilibrium_models/) of the [Deep Implicit Layers - Neural ODEs, Deep Equilibrium Models, and Beyond](http://implicit-layers-tutorial.org/), created by Zico Kolter, David Duvenaud, and Matt Johnson


### Code inspiration

- http://implicit-layers-tutorial.org/
- https://github.com/locuslab/deq
- https://github.com/lucidrains?tab=repositories

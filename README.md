<img src="./images/spins.png" width="400px"></img>

## Deep Implicit Attention
Experimental implementation of deep implicit attention in PyTorch.

**Blog post:** <a href="https://mcbal.github.io/">Deep Implicit Attention: A Mean-Field Theory Perspective on Attention Mechanisms</a>

> **This is an experimental proof-of-concept repository. No attempt has been made to optimize for speed. It would probably make sense to switch to Jax and use its `vmap`/`pmap` magic given the nature of the steps in the algorithm.**

## Install

```bash
$ pip install -e .
```

## Examples



## Code acknowledgements and inspiration

- https://github.com/locuslab/deq: Reference implementation of Deep Equilibrium Models to find fixed points of arbitrary modules and backpropagate efficiently using Broyden's method and implicit differentiation. To make this repository as small and self-contained as possible, we removed dropout and weight normalization (which requires resetting and syncing between `func` and `func_copy`) and swapped Broyden's method for a simpler Anderson fixed-point solver.
- https://implicit-layers-tutorial.org/deep_equilibrium_models/: Implementation of batched Anderson fixed-point solver.
- https://github.com/lucidrains: Clean transformers for everyone.


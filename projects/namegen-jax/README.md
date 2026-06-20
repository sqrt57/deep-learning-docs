# namegen-jax

**Repo:** [github.com/sqrt57/namegen-jax](https://github.com/sqrt57/namegen-jax)
**Language:** Python / JAX + Flax (nnx)

JAX rewrite of [namegen](../namegen/README.md). Same architecture progression, exploring JAX's functional style vs PyTorch's imperative style for the same models.

## Plan

1. ✅ Get several names datasets
2. ✅ Simple bigrams model
3. ✅ Simple one-layer NN equivalent to bigrams
4. MLP

## Progress Notes

- Bigram statistical model: train_loss ~2.55, validate_loss ~2.55 — matches PyTorch identically
- BigramNN (single linear layer via `flax.nnx`): defined and initialized; training loop has a JIT tracing bug (`DynamicJaxprTracer` error when accessing `batch.features` inside `@nnx.jit`) — not yet fixed
- Training loop in `3.02` references undefined `eval_every`, `train_steps`, `test_ds` — carried over from a Flax example, needs cleanup before MLP

## Training (where working)

- Optimizer: optax.adamw, lr=1e-3
- Loss: softmax_cross_entropy_with_integer_labels
- Train/val split: 90/10
- train_step and eval_step use @nnx.jit

## Data

- UK towns and counties — scraped from townscountiespostcodes.co.uk

## Papers

- [Bengio et al. 2003 — A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

## References

- [Andrej Karpathy: makemore](https://github.com/karpathy/makemore)
- [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)

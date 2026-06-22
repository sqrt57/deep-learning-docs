# Training Playbook

Heuristics and failure modes distilled from practice and Karpathy's [Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).

---

## Before writing any model code

- Stare at thousands of raw examples. Outliers almost always reveal preprocessing bugs or label errors.
- Understand what information the task actually requires — local vs. global features, noise tolerance, required resolution.

---

## Building the pipeline

- Fix the random seed first.
- Verify initialization loss matches theory. For a character-level model over vocab size `V`: expected loss ≈ `ln(V)`. If it's off, something is wrong before training even starts.
- Set final-layer bias to match data distribution (e.g. character frequencies), not zero.
- **Overfit a single batch to near-zero loss before anything else.** If you can't, there's a bug.
- Visualize inputs right before the model — decode tensors into readable form. This is the only source of truth about what the model actually sees.
- To verify data dependencies: zero a single example's input, run backward pass, confirm gradients are nonzero only for that example. Catches batch-dimension bugs silently introduced by wrong `view`/`transpose`.

---

## Training

- Start with Adam @ lr=3e-4. It's forgiving. Switch to SGD only when tuning for final performance.
- Disable LR decay initially. Many implementations decay based on epoch counts copied from ImageNet — wrong domain, wrong schedule. Add decay last, after everything else works.
- Complexify one thing at a time. Each added signal, layer, or feature should produce a measurable improvement. If it doesn't, remove it.
- "Don't be a hero" on architecture. Copy-paste a known-good design from a related paper first. Novel architectures only after baselines are solid.

---

## Regularization priority

1. More real data (most reliable)
2. Data augmentation
3. Pretrained weights
4. Smaller input / reduced features
5. Dropout (spatial dropout for convnets; careful with batchnorm)
6. Weight decay
7. Early stopping

Larger models early-stopped often beat smaller models regularized — try a larger model before aggressively regularizing a smaller one.

---

## Silent failure modes

These don't raise errors. They just quietly degrade results.

| Failure | Example |
|---|---|
| Off-by-one in label masking | Padding token included in loss computation |
| Off-by-one in autoregressive generation | Future token leaks into context |
| Batch dimension mixed by wrong `view` | Gradient bleeds across examples |
| Missing activation on a gate | Raw linear output used as sigmoid gate |
| LR decay from wrong domain | ImageNet schedule applied to character model |
| Loss evaluated only on train batch | Overfitting goes undetected |
| Initialization loss not verified | Misconfigured head, wrong vocab size |

---

## Squeezing performance

- Ensembles give ~2% improvement almost for free.
- Leave training running past apparent plateaus — networks often keep improving.
- Random hyperparameter search beats grid search; sensitivity is uneven across parameters.

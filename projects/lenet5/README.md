# lenet5

**Repo:** [github.com/sqrt57/lenet5](https://github.com/sqrt57/lenet5)
**Language:** Python / PyTorch

Recreation of the full LeNet 1–5 model family from LeCun's original paper.

## Plan

1. Data exploration
2. Data preprocessing
3. Simple model creation
4. LeNet 1–5 models

## Models

Five progressive architectures matching LeCun 1989:

| Model | Architecture | Params |
|---|---|---|
| Net1 | FC: 256 → 12 → 10 | ~3k |
| Net2 | FC: 256 → 12 → 10 (two-layer) | ~3k |
| Net3 | Conv(1→8, 3×3, s2) → Conv(8→4, 5×5, s2) → FC(16→10) | ~1k |
| Net4 | Conv(1→2, 3×3) → Conv(2→4, 5×5, s2) → FC(16→10) | ~1k |
| Net5 | Conv(1→2, 3×3, s2) → Conv(2→4, 5×5, s2) → FC(16→10) | 1,060 |

## Architecture Notes

- Activation: normalized tanh — `tanh(2x/3) / tanh(2/3)` (faithful to original)
- Convolutions implemented with manual `unfold` rather than `F.conv2d`, matching LeCun's design
- Padding: -1 value (original paper convention)
- Loss: MSE scaled by `num_classes / 2`; targets one-hot encoded to [-1, 1]

## Dataset

- 16×16 grayscale hand-drawn digits (0–9)
- 480 samples total; augmented 4× via padding variations from 120 originals; 320 train / 160 test
- Normalized to [-1, 1]

## Training

- Optimizer: Adam, lr=5e-4, betas=(0.9, 0.999)
- Epochs: 2,000, batch size 32

## Papers

- [LeCun 1989 — Generalization and Network Design Strategies](http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf)

## References

- [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)

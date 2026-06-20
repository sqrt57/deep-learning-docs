# solve-cifar10

**Repo:** [github.com/sqrt57/solve-cifar10](https://github.com/sqrt57/solve-cifar10)
**Language:** Python / PyTorch

ResNet reimplementation for CIFAR-10 classification.

## Plan

1. Data exploration
2. Simple one-layer model
3. ResNet model

## Models

| Model | Description | Params |
|---|---|---|
| SimpleModel | Linear(3072→10) + ReLU | — |
| SimpleModelBatchNorm | Linear + BatchNorm1d + ReLU | — |
| SimpleModelBatchNormPrelu | Linear + BatchNorm1d + PReLU | — |
| Net | LeNet-style CNN (2 conv + 3 FC) | — |
| Resnet8 | 6 residual blocks | — |
| Resnet14 | 9 residual blocks | — |
| Resnet20 | 9 residual blocks, wider | 272,474 |

## Architecture Notes

- Channel progression: 3 → 16 → 32 → 64, with stride-2 ResizeBlocks at each transition
- ResBlock: two Conv(ch→ch, 3×3, pad=1, bias=False) + BN + skip connection
- ResizeBlock: Conv(1×1, stride=s) projection on skip path for spatial downsampling
- All conv layers: bias=False (BatchNorm provides bias)
- Global average pooling (8×8) before classification head

## Training

- Dataset: CIFAR-10, 45k train / 5k val / 10k test, normalized to [0, 1]
- Optimizer: AdamW, lr=1e-3
- LR schedule: MultiStepLR, milestones=[10, 20, 30, 40, 50], γ=1/√10
- Loss: CrossEntropyLoss
- Batch size: 64, epochs: 60, CUDA

## Papers

- [He et al. 2015 — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [He et al. 2015 — Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
- [Xie et al. 2017 — Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

## References

- [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)

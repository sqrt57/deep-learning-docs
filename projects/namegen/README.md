# namegen

**Repo:** [github.com/sqrt57/namegen](https://github.com/sqrt57/namegen)
**Language:** Python / PyTorch

Character-level language model for name generation.

## Plan

1. ✅ Get several names datasets
2. ✅ Simple bigrams model
3. ✅ Simple one-layer NN equivalent to bigrams
4. ✅ MLP

See [plan.md](plan.md) for detailed work items. Reference legend:
- `I#` — Infrastructure (config, build, logging, libraries)
- `T#` — Training
- `D#` — Data
- `M#` — Models
- `B#` — Bugs
- `E#` — Explore (reading, research)

## Models

Progressive architecture study from statistical baseline to deep RNNs:

| Model | Params | Test loss |
|---|---|---|
| BigramsModel (statistical) | — | 2.55 |
| OneLayerBigramModel | 1,024 | 2.61 |
| EmbeddingMLP small | 2,046 | 2.46 |
| EmbeddingMLP big (8-gram context) | 39,088 | 1.97 |
| RNN small | 4,346 | 2.25 |
| RNN big (8-gram context) | 176,688 | 1.71 |
| LSTM small | 19,796 | 2.10 |
| LSTM big (8-gram context) | 1,147,488 | **1.39** |

## Architecture Notes

See [architecture.md](architecture.md) for the full overview.

## Data

- UK towns and counties — scraped from townscountiespostcodes.co.uk
- US names — SSA baby names dataset
- Russian names and surnames

## Papers

- [Bengio et al. 2003 — A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

## References

- [Andrej Karpathy: makemore](https://github.com/karpathy/makemore)
- [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) — project structure and notebook naming convention

# Architecture

Character-level language model for name generation. All models share a common interface: `context_size() → int` and `forward(x: Tensor[batch, time]) → Tensor[batch, time, nalphabet]`.

## Overall

```
data/raw/uk-towns.csv
    └── dataset.py → Dataset (features, labels tensors)
                         ├── InfiniteDataLoader → Trainer (train.py)
                         └── calculate_loss (predict.py)
model.py (BigramsModel | OneLayerBigramModel | EmbeddingMLP | RNN | LSTM)
    ├── Trainer.run_scenario() → Result
    └── generate() → list[str]
```

- Padding token `_` is always index 0; used as start-of-sequence context and end-of-sequence label
- All sequences padded to `max_word_length + 1` (extra position for the end token)
- Labels use `-1` as ignore mask; `CrossEntropyLoss(ignore_index=-1)` skips those positions

## Data

- Source: UK towns CSV, lowercased, split on `/` `(` `)`, deduped — 37,238 names
- Alphabet: ordered by character frequency (most common first), `_` prepended as index 0
- `Dataset.get_features_and_labels()`:
  - `features[i]`: `[0, c1, c2, ..., cn, 0, 0, ...]` — zero-padded, leading zero = start token
  - `labels[i]`: `[c1, c2, ..., cn, 0, -1, -1, ...]` — chars + one end token, rest masked
  - Known off-by-one: `labels_line[len(line)+1:] = -1` should be `labels_line[len(line):]` (B1)
- `InfiniteDataLoader`: samples random batches via `torch.multinomial` with uniform weights

## Models

### BigramsModel
- Statistical baseline: co-occurrence count matrix → row-normalized log-probabilities
- Smoothing via additive `prior` parameter
- `context_size = 1`

### OneLayerBigramModel
- Learnable weight matrix `W[nalphabet, nalphabet]`; initialized uniform
- Equivalent to BigramsModel but trained by gradient descent
- `context_size = 1`

### EmbeddingMLP
- `Embedding(nalphabet, nembedding)` → concat `context_size` embeddings → `Linear → tanh → Linear`
- Left-pads short sequences with zeros internally (bug B4: double-padding during generation when input is already shorter than `context_size`)
- `context_size` configurable; big variant uses 8

### RNN
- Per-step: `h_t = tanh(Linear([x_t, h_{t-1}]))`; processes full sequence in a Python loop
- Separate `output` linear maps final hidden states to logits
- `context_size` parameter stored but sequence always processed from step 0

### LSTM
- Per-step state: `(cell, hidden)`, both `(batch, nstate)`
- Gate inputs include `cell` alongside `hidden` and `x_t` — non-standard vs. canonical LSTM
- Known bugs:
  - B2: output gate `hidden = hidden_gate * tanh(cell)` missing `sigmoid` on `hidden_gate`
  - B3: forget and input gates share one `Linear(emb+2*nstate, 2*nstate)` layer, split by slice

## Training

- `Hyper` namedtuple: model class, context_size, batch_size, nsteps, optimizer, lr, seed
- `Trainer.run_scenario()`: instantiates model, runs loop, logs loss every 10 steps
- Loss: `CrossEntropyLoss(ignore_index=-1)`, sequences flattened to `(batch*time, nalphabet)` before loss
- Eval at each log step uses the current training batch, not a held-out set (T1 — no test split yet)
- Default optimizer: Adam lr=3e-4; production runs use AdamW lr=1e-3

## Inference

- `generate(dataset, model, N, T, max_len)`: auto-regressive from zero context
  - Sliding window: `x[:, -context_size:]` if sequence long enough, else raw `x` (bug B4)
  - Temperature `T`: scales logits before softmax; lower = sharper, higher = more random
  - Stops when all N sequences have produced a `_` token or `max_len` is reached
  - Strips leading zero (start token) and truncates at first `_` in output
- `calculate_loss(dataset, model)`: full-dataset loss, no-grad

## Notebooks

Naming follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) convention: `<phase>.<sequence>-<initials>-<description>.ipynb`.

| Phase | Meaning |
|---|---|
| 0 | Data exploration |
| 1 | Data processing |
| 2 | Feature engineering |
| 3 | Modeling |
| 4 | Analysis / visualization |
| 5 | Reporting |

| Notebook | Description |
|---|---|
| `0.01-dg-uk-towns` | Data exploration — UK towns CSV structure and character distribution |
| `3.01-dg-bigrams-model` | Statistical bigrams baseline |
| `3.02-dg-nn-bigram-model` | One-layer NN equivalent to bigrams |
| `3.03-dg-nn-probabilistic-embedding-model` | EmbeddingMLP experiments |
| `3.04-dg-models-compare` | Side-by-side comparison of all trained models |

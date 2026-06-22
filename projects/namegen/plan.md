# Work Plan

Legend: `I` Infrastructure (config, build, logging, libraries) · `T` Training · `D` Data · `B` Bug · `M` Model · `E` Explore (reading, research, external resources)

Status: `todo` · `in progress` · `done` · `blocked`

---

## E1 — Read and process Karpathy recipe article

**Status:** done

**Description:** Read Karpathy's "A Recipe for Training Neural Networks" article. Extract key findings, failure modes, and debugging heuristics relevant to this project. Codify reusable rules into this repo (e.g. a `rules.md` at the repo root).

**Acceptance criteria:**
- Reusable rules and heuristics written up and committed to this repo
- Actionable items linked to or added as new plan tasks

**Result:** [playbook.md](../../playbook.md) written at repo root.

**Estimate:** 2h

---

## D1 — Extract CharTokenizer from Dataset

**Status:** todo

**Description:** Refactor `Dataset` to extract a `CharTokenizer` class, mirroring the structure in namegen-jax. Currently the alphabet construction and character↔index mappings live inside `Dataset`; they should be a standalone, reusable object.

**Architecture:**

```python
class CharTokenizer:
    def __init__(self, strings: list[str])
    def dict_size(self) -> int          # vocabulary size (nalphabet)
    def str_to_indices(self, s: str) -> list[int]
    def indices_to_str(self, indices) -> str
```

- Alphabet built from input strings, ordered by character frequency (most common first), `_` prepended as index 0.
- `Dataset.__init__` takes a `CharTokenizer`; alphabet construction removed from it.
- `predict.py` / `generate()` switches from `dataset.itoc` to `tokenizer.indices_to_str()`.
- Prerequisite for T1: tokenizer should be built from train names only, not the full dataset.

**Acceptance criteria:**
- `CharTokenizer` in its own file (`tokenizer.py`)
- `Dataset` accepts a tokenizer; no alphabet logic inside it
- `generate()` and `calculate_loss()` use tokenizer for encode/decode
- All existing notebooks and training scripts work unchanged

**Testing:** Encode then decode a known string; assert round-trip is identical.

**Estimate:** 2h

---

## T1 — Train/test split

**Status:** todo

**Description:** Hold out a test set and report test loss after training. Currently eval runs on the training batch, so no metric is meaningful until this is fixed.

**Architecture:** Split dataset before any training; pass test set to trainer separately. Trainer computes and logs test loss after final step.

**Acceptance criteria:**
- Test loss reported at end of training run
- Test set never seen during training or hyperparameter selection

**Testing:** Verify test set size matches expected fraction; confirm test loss is higher than train loss (sanity check for no leakage).

**Results:** Retrain all existing models and update the README models table with corrected test losses.

**Estimate:** 2h

---

## B1 — Label masking off-by-one

**Status:** todo

**Description:** `labels_line[len(line)+1:] = -1` leaves `labels_line[len(line)]` as 0 (padding token), so the loss trains on a spurious prediction at the word boundary.

**Architecture:** `dataset.py:65-66` only. One-line fix.

**Acceptance criteria:**
- `labels_line[len(line):]` is all `-1` for every sample
- No padding token appears in unmasked label positions

**Testing:** Unit test: construct a sample, assert `labels[len(word):] == -1` and `labels[:len(word)] != -1`.

**Results:** Retrain all models; update README table. Expect improvement across the board since all models trained on corrupted signal.

**Estimate:** 30min

---

## B2 — LSTM output gate missing sigmoid

**Status:** todo

**Description:** `hidden = hidden_gate * F.tanh(cell)` uses raw linear output as the gate — no sigmoid means it's not actually gating.

**Architecture:** `model.py:164`. Fix: `hidden = F.sigmoid(hidden_gate) * F.tanh(cell)`.

**Acceptance criteria:**
- Output gate is sigmoid-activated
- LSTM hidden state values stay in `(-1, 1)`

**Testing:** Retrain LSTM; test loss should improve over the pre-fix baseline.

**Results:** Retrain LSTM small and big; update README table with new losses.

**Estimate:** 30min

---

## B3 — LSTM forget/input gates share one linear layer

**Status:** todo

**Description:** Single `forget_input` layer outputs `2*nstate` then is split — both halves come from the same computation, preventing independent gate learning.

**Architecture:** `model.py:149`. Replace with two separate linear layers `W_f` and `W_i`, each outputting `nstate`.

**Acceptance criteria:**
- Forget and input gates computed from independent weight matrices
- Parameter count increases by `nstate * (nembedding + nstate)` per LSTM layer

**Testing:** Retrain; compare test loss against B2 baseline. Gate activations should show distinct patterns.

**Results:** Retrain LSTM small and big; update README table. Document new parameter counts.

**Estimate:** 1h

---

## B4 — Generation context window double-padding

**Status:** todo

**Description:** When `x.shape[1] < context_size`, the short sequence is passed as-is, then `EmbeddingMLP` pads it internally — double-padding the first few generation steps.

**Architecture:** `predict.py:7-19`. Pre-pad input to exactly `context_size` with padding token before every model call.

**Acceptance criteria:**
- First generated token uses the same context representation as subsequent tokens
- MLP internal padding path is no longer exercised during generation

**Testing:** Generate with MLP model; verify first character distribution matches expected unigram frequencies.

**Results:** Generation quality only — no loss metric affected. Document qualitative change in generated samples if noticeable.

**Estimate:** 30min

---

## B5–B8 — Minor fixes

**Status:** todo

**Description:** Four small cleanup items.

| ID | Issue | File | Fix |
|---|---|---|---|
| B5 | Stray `model.train()` in `no_grad` block | `train.py:74` | Remove the call |
| B6 | `torch.zeros(...)` always on CPU | `predict.py:10` | Pass `device` from model |
| B7 | `max()` crashes on empty dataset | `dataset.py:47` | Add `default=0` |
| B8 | `CrossEntropyLoss` created per eval call | `predict.py:33` | Move to module init |

**Architecture:** Isolated one-liners, no structural changes.

**Acceptance criteria:** Each line changed as described; no regressions.

**Testing:** Existing tests pass; B6 verified on GPU run.

**Results:** No model metrics affected.

**Estimate:** 1h total

---

## M1 — Implement GRU

**Status:** todo

**Description:** Add a GRU model following the same custom-gate pattern as the existing RNN/LSTM (not `nn.GRU`).

**Architecture:**
- Gates: reset gate `r = sigmoid(W_r @ [h, x])`, update gate `z = sigmoid(W_z @ [h, x])`
- Candidate: `h̃ = tanh(W_h @ [r*h, x])`
- Output: `h = (1-z)*h + z*h̃`
- Same embedding + linear head as RNN/LSTM

**Acceptance criteria:**
- GRU trains to lower test loss than RNN big (1.71)
- Small and big (8-gram) variants benchmarked

**Testing:** Train small and big variants; sanity-check gate activations are in (0,1).

**Results:** Add GRU small and GRU big rows to README models table with params and test loss.

**Estimate:** 4h

---

## M2 — Implement Transformer

**Status:** todo

**Description:** Add a decoder-only Transformer for character-level generation.

**Architecture:**
- Learned positional embeddings up to `context_size`
- N stacked decoder blocks: causal self-attention + feed-forward + layer norm
- No cross-attention (decoder-only)
- Same token embedding + linear head as other models

**Acceptance criteria:**
- Trains to lower test loss than LSTM big (1.39)
- Small and large variants benchmarked

**Testing:** Verify causal mask prevents attending to future tokens; check loss decreases monotonically early in training.

**Results:** Add Transformer small and big rows to README models table with params and test loss.

**Estimate:** 1–2 days

---

## T2 — Add validation split

**Status:** todo

**Description:** Add a val split between train and test for hyperparameter tuning and early stopping. Depends on T1.

**Architecture:** Three-way split of the dataset. Val loss logged during training; test loss reported only at the end.

**Acceptance criteria:**
- Val loss logged every N steps during training
- Test set untouched until final evaluation
- Early stopping optionally triggered on val loss plateau

**Testing:** Confirm val and test losses are correlated but val is always computed mid-training and test only once.

**Results:** No new model entries; update README training section to document the split ratios.

**Estimate:** 2h

---

## E2 — Explore Karpathy blog

**Status:** todo

**Description:** Browse Karpathy's blog for posts relevant to character-level language models, training practices, and architecture choices. Identify posts worth deep-reading.

**Acceptance criteria:**
- List of relevant posts noted with one-line summaries
- High-value posts queued for deeper reading (E-type tasks or paper notes)

**Estimate:** 1h

---

## E3 — Explore Christopher Olah blog

**Status:** todo

**Description:** Browse Christopher Olah's blog (colah.github.io) for posts on RNNs, LSTMs, and neural network internals. Identify posts worth deep-reading.

**Acceptance criteria:**
- List of relevant posts noted with one-line summaries
- High-value posts queued for deeper reading

**Estimate:** 1h

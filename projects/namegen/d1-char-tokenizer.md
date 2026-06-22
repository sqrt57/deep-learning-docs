# D1 — Extract CharTokenizer from Dataset

**Prerequisite for:** T1 (tokenizer must be built from train names only, not the full dataset).

## Problem

Alphabet construction and character↔index mappings live inside `Dataset`. This couples tokenization to dataset loading, prevents building the vocabulary from a subset of names (required for T1's train/test split), and duplicates logic already refactored in namegen-jax.

## Spec

### CharTokenizer interface

```python
class CharTokenizer:
    def __init__(self, strings: list[str])
    def dict_size(self) -> int          # vocabulary size (nalphabet)
    def str_to_indices(self, s: str) -> list[int]
    def indices_to_str(self, indices) -> str
```

- Alphabet built from input strings, ordered by character frequency (most common first).
- `_` prepended as index 0 (padding token).
- Lives in its own file: `tokenizer.py`.

### Interface changes

**`tokenizer.py`** — new file containing `CharTokenizer`.

**`dataset.py`**

- `Dataset.__init__` accepts a `CharTokenizer` instead of building the alphabet internally.
- Remove alphabet construction logic from `Dataset`.

**`predict.py`**

- `generate()` switches from `dataset.itoc` to `tokenizer.indices_to_str()`.
- `calculate_loss()` uses tokenizer for encode/decode where applicable.

### What does not change

- `Dataset` API beyond the tokenizer argument — feature/label tensor construction is unchanged.
- No training runs required; this is a pure refactor with identical behavior.

## Acceptance criteria

- `CharTokenizer` in its own file (`tokenizer.py`).
- `Dataset` accepts a tokenizer; no alphabet logic inside it.
- `generate()` and `calculate_loss()` use tokenizer for encode/decode.
- All existing notebooks and training scripts work unchanged.

## Verification

```python
tokenizer = CharTokenizer(["alice", "bob", "charlie"])
assert tokenizer.indices_to_str(tokenizer.str_to_indices("alice")) == "alice"
```

## Implementation steps

1. Create `tokenizer.py` with `CharTokenizer`.
2. Update `Dataset.__init__` to accept a `CharTokenizer`; remove internal alphabet construction.
3. Update `predict.py` to use `tokenizer.indices_to_str()`.
4. Add round-trip unit test.

## Estimate

2h

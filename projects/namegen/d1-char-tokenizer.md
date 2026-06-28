# D1 — Extract CharTokenizer from Dataset

**Prerequisite for:** T1 (tokenizer must be built from train names only, not the full dataset).

**Status: done.**

## Problem

Alphabet construction and character↔index mappings lived inside `Dataset`. This coupled tokenization to dataset loading, prevented building the vocabulary from a subset of names (required for T1's train/test split), and duplicated logic already refactored in namegen-jax.

## Design

`Dataset` class eliminated entirely. Replaced by:

- `CharTokenizer` — builds and owns the vocabulary.
- `Batch` namedtuple — pure tensor container `(features, labels)`, no tokenizer attached.
- `make_dataset(strings, tokenizer) -> Batch` — factory function; `strings` and `max_word_length` are local variables, nothing stored beyond the tensors.

Callers hold the tokenizer separately and pass it where needed (`generate`, `calculate_loss`, `Trainer`, `train_bigram_model`).

## CharTokenizer interface

```python
class CharTokenizer:
    alphabet: str                              # public; '_' at index 0

    def __init__(self, strings: list[str])
    def dict_size(self) -> int
    def str_to_indices(self, s: str) -> list[int]
    def indices_to_str(self, indices) -> str
```

- Alphabet ordered by character frequency (most common first), `_` prepended as index 0.
- `str_to_indices` returns `list[int]` — no torch dependency on the tokenizer.

## Files changed

| File | Change |
|---|---|
| `namegen/tokenizer.py` | New — `CharTokenizer` |
| `namegen/dataset.py` | `Dataset` replaced by `Batch` + `make_dataset`; `InfiniteDataLoader.next()` returns `Batch`; `uk_towns_and_counties` returns `tuple[Batch, CharTokenizer]` |
| `namegen/modeling/predict.py` | `generate` and `calculate_loss` take `Batch` + `tokenizer` separately |
| `namegen/modeling/train.py` | `train_bigram_model` and `Trainer` take `Batch` + `nalphabet: int` (not tokenizer) |
| `pyproject.toml` | Added pytest dev dependency; fixed setuptools package discovery |
| `tests/test_tokenizer.py` | New — round-trip, alphabet, dict_size tests |
| `tests/test_dataset.py` | New — shape, padding, masking tests |

## Verification

```python
t = CharTokenizer(["alice", "bob", "charlie"])
assert t.indices_to_str(t.str_to_indices("alice")) == "alice"
assert t.alphabet[0] == '_'
assert t.dict_size() == 4  # _, a, b, c (for CharTokenizer(["abc"]))
```

## Estimate

2h

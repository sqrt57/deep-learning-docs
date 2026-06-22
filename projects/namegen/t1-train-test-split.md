# T1 — Train/Test Split

**Depends on:** D1 (CharTokenizer extraction) — tokenizer must be built from train names only.

## Problem

`Trainer` evaluates loss on the current training batch. All reported losses in the models table are train losses — meaningless for comparing generalization.

## Spec

### Split

- Shuffle the raw name list with a fixed seed, then slice: 90% train, 10% test.
- Split at the name level (before feature/label construction) so no name appears in both sets.
- Split ratio and seed configurable; defaults: `train_frac=0.9`, `split_seed=42`.

### Interface changes

**`dataset.py`**

```python
# New: returns two Dataset instances
def split_dataset(dataset: Dataset, train_frac: float, seed: int) -> tuple[Dataset, Dataset]:
    ...
```

`Dataset` already holds a list of names and builds tensors from them — `split_dataset` slices that list and constructs two `Dataset` instances.

**`train.py`**

`Hyper` gains an optional `test_dataset` field (or `Trainer` receives it separately). After the final training step, `Trainer.run_scenario()` calls `calculate_loss(test_dataset, model)` and includes test loss in `Result`.

**`Result`**

Add `test_loss: float` field alongside existing `train_loss`.

### What does not change

- `InfiniteDataLoader` samples only from the train `Dataset`.
- `calculate_loss` is unchanged — already does full-dataset no-grad loss; just called on the test split.
- No val split yet (T2).

## Acceptance criteria

- Test loss reported at end of every training run.
- Test set never touched during training or hyperparameter selection.
- `test_loss > train_loss` for all models (basic leakage sanity check).
- Test set size ≈ 3,724 names (10% of 37,238).

## Verification

```python
train_ds, test_ds = split_dataset(full_ds, train_frac=0.9, seed=42)
assert len(set(train_ds.names) & set(test_ds.names)) == 0  # no overlap
assert abs(len(test_ds.names) / len(full_ds.names) - 0.1) < 0.01
```

## Implementation steps

1. Add `split_dataset(dataset, train_frac, seed)` to `dataset.py`.
2. Add unit test for no-overlap and correct sizes.
3. Update `Trainer` / `Hyper` to accept and use test dataset.
4. Add `test_loss` to `Result`; log it after final step.
5. Retrain all models; update README models table with corrected test losses.

## Estimate

2h

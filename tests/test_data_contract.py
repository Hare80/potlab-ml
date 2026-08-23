"""Data contract tests (M4): split validation + the batch label shape.

Everything here runs without the QM9 download - _validate_splits is a
pure function and GetTarget runs on synthetic Data objects. Checklist
item 7 (split discipline: no trajectory sharing across splits) is NOT
testable yet because no trajectory dataset exists in the project; it
becomes testable when a vasp.py-style module lands, and no empty test
should be written for an absent feature.

torch_geometric is a hard project dependency (potlab.data.qm9 imports it
at module level), so it is imported normally - a broken environment must
fail loudly at collection, never skip.
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from potlab.data.qm9 import _validate_splits
from potlab.data.transforms import GetTarget


# --- splits guard (the M1 fix for the silently-ignored-third-value bug) ---

def test_splits_mixed_types_rejected():
    with pytest.raises(ValueError, match="all int or all float"):
        _validate_splits([100, 0.5, 100], n_mols=1000)


def test_splits_wrong_length_rejected():
    with pytest.raises(ValueError, match="Expected 3 values"):
        _validate_splits([800, 100], n_mols=1000)


def test_splits_train_val_overflow_rejected():
    # train + val consume the whole dataset: nothing left for the test set.
    with pytest.raises(ValueError, match="must be less than the dataset size"):
        _validate_splits([800, 200, 0], n_mols=1000)


def test_splits_subset_guard_message():
    # The classic mistake: full-size splits against a --subset-size run.
    # The message teaches how to fix it - that is part of the contract.
    with pytest.raises(ValueError, match="subset_size"):
        _validate_splits([800, 100, 100], n_mols=100)


def test_splits_int_counts_pass_through():
    assert _validate_splits([800, 100, 100], n_mols=1000) == [800, 100, 100]


def test_splits_float_proportions_resolved():
    # Proportions resolve against the dataset size; truncation is fine
    # (the test split takes the remainder at setup time).
    assert _validate_splits([0.8, 0.1, 0.1], n_mols=1000) == [800, 100, 100]


# --- batch contract (DESIGN.md §3): one label row per molecule ---
#
# GetTarget consumes PyG Data objects - that is the format QM9 yields
# its molecules in, so the tests feed it minimal stand-ins of the same
# shape (the same idea as ToyModel/ToyLoader in test_trainer.py: fake
# the INPUT, assert the PROJECT class). The class under test in every
# assertion below is potlab.data.transforms.GetTarget, never PyG.

def test_get_target_keeps_column_dimension():
    # [1, 19] -> [1, 1], never [1]: the batch contract wants
    # [N_graphs, num_outputs], and a squeezed label would break every
    # consumer downstream (trainer loss, standardizer).
    data = Data(y=torch.rand(1, 19))  # one molecule, all 19 QM9 properties
    out = GetTarget(7)(data)
    assert out.y.shape == (1, 1)


def test_get_target_batch_label_shape():
    # Batch.from_data_list replicates what the DataLoader does on every
    # iteration: merge batch_size molecules into one batch, concatenating
    # y along dim 0. After that merge, GetTarget must still yield one
    # label row per MOLECULE - [2, 1], the shape the trainer divides by.
    batch = Batch.from_data_list(
        [Data(y=torch.rand(1, 19)), Data(y=torch.rand(1, 19))]
    )
    out = GetTarget(7)(batch)
    assert out.y.shape == (2, 1)

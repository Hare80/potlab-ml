"""Roundtrip tests for Qm9Standardizer.

These tests pin down the mathematical contract of the standardizer:
transform then inverse must restore the original labels exactly. They run
on hand-crafted tensors only - no dataset download, no GPU. fit() itself
is exercised end-to-end by the M1 acceptance run (MAE vs the baseline).
"""

import pytest
import torch

# qm9.py imports torch_geometric at module level; skip cleanly when it is
# not installed instead of failing collection.
pytest.importorskip("torch_geometric")

from potlab.data.qm9 import Qm9Standardizer


def _make_standardizer():
    """A standardizer with hand-set statistics (contract-shaped, no fit)."""
    std = Qm9Standardizer()
    std.mean = torch.tensor(1.0)
    std.std = torch.tensor(2.0)
    # Real atom_refs are [num_atom_types, 1]; keep that 2D contract shape -
    # a 1D table would break the per-graph sums downstream.
    std.atom_refs = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
    return std


def _make_batch():
    """Two molecules with DIFFERENT atom counts (8 and 3).

    The different counts are the point: the per-molecule divide/multiply
    by n_atoms only cancels correctly if both sizes round-trip. A test
    with a single molecule cannot catch a buggy global divide.
    """
    z = torch.tensor([6, 6, 1, 1, 1, 1, 1, 1, 8, 1, 1])  # C2H6 + H2O
    graph_indexes = torch.tensor([0] * 8 + [1] * 3)
    y = torch.tensor([[10.0], [20.0]])
    return z, graph_indexes, y


def test_transform_inverse_roundtrip():
    """transform -> inverse must restore the original physical labels."""
    std = _make_standardizer()
    z, graph_indexes, y = _make_batch()

    y_back = std.inverse(std.transform(y, z, graph_indexes), z, graph_indexes)
    assert torch.allclose(y_back, y), f"Expected {y}, but got {y_back}"


def test_transform_shape():
    """transform keeps the [N_graphs, num_outputs] batch contract shape."""
    std = _make_standardizer()
    z, graph_indexes, y = _make_batch()

    transformed = std.transform(y, z, graph_indexes)
    assert transformed.shape == y.shape, (
        f"Expected shape {y.shape}, but got {transformed.shape}"
    )


def test_state_dict_roundtrip():
    """load_state_dict(state_dict()) restores the exact statistics."""
    std = _make_standardizer()
    state = std.state_dict()

    restored = Qm9Standardizer()
    restored.load_state_dict(state)

    assert torch.allclose(restored.mean, std.mean)
    assert torch.allclose(restored.std, std.std)
    assert torch.allclose(restored.atom_refs, std.atom_refs)

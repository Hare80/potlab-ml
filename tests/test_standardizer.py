"""Roundtrip tests for Qm9Standardizer.

These tests pin down the mathematical contract of the standardizer:
transform then inverse must restore the original labels exactly. They run
on hand-crafted tensors only - no dataset download, no GPU. fit() itself
is exercised end-to-end by the M1 acceptance run (MAE vs the baseline).
"""

import torch

# torch_geometric is a hard project dependency (qm9.py imports it at
# module level) - a broken environment must fail loudly, never skip.
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


def test_inverse_is_per_atom_aggregation():
    """inverse == broadcast pooled value -> inverse_per_atom -> per-graph sum.

    The M5 refactor contract: inverse_per_atom is the base primitive and
    inverse composes it (the export path and the training path must be
    the SAME math at two granularities).
    """
    std = _make_standardizer()
    z, graph_indexes, y = _make_batch()
    energy_pred = std.transform(y, z, graph_indexes)  # per-molecule standardized

    per_atom = std.inverse_per_atom(energy_pred[graph_indexes], z)
    out = torch.zeros(energy_pred.shape[0], 1)
    out.index_add_(dim=0, index=graph_indexes, source=per_atom)
    assert torch.allclose(out, std.inverse(energy_pred, z, graph_indexes))


def test_inverse_per_atom_closed_form():
    """inverse_per_atom is the documented unstandardization: c*std + mean + refs[z]."""
    std = _make_standardizer()
    z, graph_indexes, y = _make_batch()
    energy_pred = std.transform(y, z, graph_indexes)

    # Per-atom inputs, fabricated without a model: the pooled value
    # broadcast to each atom (identical within a molecule) - NOT the
    # core's real per-atom contributions, hence the honest name below.
    # Degenerate but still exercises the full formula - refs[z] varies
    # per element.
    broadcast = energy_pred[graph_indexes]
    expected = broadcast * std.std + std.mean + std.atom_refs[z]
    assert torch.allclose(std.inverse_per_atom(broadcast, z), expected)

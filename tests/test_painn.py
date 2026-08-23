"""PaiNN model contract tests (M2): registration wiring + rotation invariance.

Rotation invariance is the physical symmetry every geometric model must
have: the energy of a rotated molecule is unchanged. The permanent M4
suite will extend this file (force equivariance, gradient checks); what
lands now is the M2 acceptance piece.
"""

import importlib

import pytest
import torch

torch_geometric = pytest.importorskip("torch_geometric")

import potlab.models.painn.model  # noqa: E402 - side effect: registers "painn"
import potlab.registry as registry
from potlab.models.painn.model import PaiNNModel


def _small_model() -> PaiNNModel:
    """A fast config: few features, few RBFs, two rounds. Seconds on CPU."""
    return PaiNNModel(
        num_message_passing_layers=2,
        num_features=16,
        num_outputs=1,
        num_rbf_features=4,
        num_unique_atoms=100,
        cutoff_dist=5.0,
    )


def _random_rotation() -> torch.Tensor:
    """A proper rotation matrix (orthogonal, det = +1).

    QR of a random matrix gives an orthogonal Q, but det(Q) may be -1
    (a reflection); flipping one column fixes the handedness. A test run
    with a reflection would fail "invariance" for the wrong reason.
    """
    # float64: the invariance test runs the model in double precision.
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_painn_registered_in_models():
    # MODELS is module-level global state shared across tests, so the
    # test clears it and re-runs the real registration path (reload
    # re-executes the decorator) - order-independent and honest.
    registry.MODELS.clear()
    model_module = importlib.reload(potlab.models.painn.model)
    assert registry.MODELS["painn"] is model_module.PaiNNModel


def test_energy_rotation_invariant():
    model = _small_model().double()  # float64: the test measures the math,
    z = torch.tensor([1, 6, 8, 1, 6, 6, 8])  # not float32 rounding noise
    pos = torch.rand(7, 3, dtype=torch.float64)  # two molecules: 4 + 3 atoms
    graph_indexes = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    rotation = _random_rotation()
    pos_rot = pos @ rotation.T

    energy = model.energy(z, pos, graph_indexes)
    energy_rot = model.energy(z, pos_rot, graph_indexes)

    # The batch contract: one row per molecule.
    assert energy.shape == (2, 1)
    assert torch.allclose(energy, energy_rot, atol=1e-8)

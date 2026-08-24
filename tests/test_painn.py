"""PaiNN contract tests (M2+M4): registration, symmetries, gradients.

M2 landed registration wiring + rotation invariance of the energy. M4
added the rest of the permanent exam: force equivariance under rotation
and the autograd-vs-finite-differences gradient check. Every future
model must pass this file before being merged. torch_geometric is a hard
project dependency, so it is imported normally - a broken environment
fails loudly, never skips.
"""

import importlib

import torch

import potlab.models.painn.model  # side effect: registers "painn"
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


def test_forces_rotation_equivariant():
    model = _small_model().double()  # float64, as in the invariance test
    z = torch.tensor([1, 6, 8, 1, 6, 6, 8])
    pos = torch.rand(7, 3, dtype=torch.float64)
    graph_indexes = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    rotation = _random_rotation()
    pos_rot = pos @ rotation.T

    _, forces = model.energy_and_forces(z, pos, graph_indexes)
    _, forces_rot = model.energy_and_forces(z, pos_rot, graph_indexes)

    # Forces are vectors, so they transform like positions: the gradient
    # w.r.t. pos' = pos @ R.T picks up a factor of R.T (chain rule, R
    # orthogonal -> inverse transpose = R.T). f' = f @ R.T.
    assert forces.shape == (7, 3)
    assert torch.allclose(forces_rot, forces @ rotation.T, atol=1e-8)


def test_autograd_forces_match_finite_differences():
    # Central differences with eps=1e-6: in float64 the truncation error
    # (~eps^2) sits far below the 1e-4 bound, so the bound really measures
    # the autograd path. Note the forces here are the gradient of the
    # MEAN-pooled per-molecule energies - the test locks autograd SELF-
    # consistency, not physical forces (mean pooling rescales them by
    # n_atoms; QM9 never uses forces, and the M6 force-bearing toy model
    # will make its own pooling choice).
    model = _small_model().double()
    z = torch.tensor([1, 6, 8, 1, 6, 6, 8])
    pos = torch.rand(7, 3, dtype=torch.float64)
    graph_indexes = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    _, forces_auto = model.energy_and_forces(z, pos, graph_indexes)
    forces_auto = forces_auto.detach()
    # energy_and_forces sets requires_grad_ on pos in place; the FD loop
    # below wants clean leaf tensors.
    pos = pos.detach().clone()

    eps = 1e-6
    forces_fd = torch.zeros_like(forces_auto)
    for i in range(pos.shape[0]):
        for j in range(3):
            pos_plus = pos.clone()
            pos_plus[i, j] += eps
            pos_minus = pos.clone()
            pos_minus[i, j] -= eps
            e_plus = model.energy(z, pos_plus, graph_indexes).sum()
            e_minus = model.energy(z, pos_minus, graph_indexes).sum()
            forces_fd[i, j] = -(e_plus - e_minus) / (2 * eps)

    # Guard: if the random geometry were gradient-flat, the ratio below
    # would be noise over ~0 - fail loudly instead of "passing" vacuously.
    assert forces_auto.abs().max() > 1e-6
    # Relative to the largest component: per-component division blows up
    # on near-zero force components.
    rel_error = (forces_fd - forces_auto).abs().max() / forces_fd.abs().max()
    assert rel_error < 1e-4


def test_forces_differentiate_energy_column_only():
    # Protocol convention (M5 alignment): column 0 is the energy. With
    # num_outputs = 2, forces must be the gradient of the energy column
    # ALONE - the second column is a property, and the gradient of the
    # two-column sum is not a force. FD on energy[:, 0].sum() proves the
    # differentiation target exactly.
    model = PaiNNModel(
        num_message_passing_layers=2,
        num_features=16,
        num_outputs=2,
        num_rbf_features=4,
        num_unique_atoms=100,
        cutoff_dist=5.0,
    ).double()
    z = torch.tensor([1, 6, 8, 1, 6, 6, 8])
    pos = torch.rand(7, 3, dtype=torch.float64)
    graph_indexes = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    _, forces_auto = model.energy_and_forces(z, pos, graph_indexes)
    forces_auto = forces_auto.detach()
    pos = pos.detach().clone()

    eps = 1e-6
    forces_fd = torch.zeros_like(forces_auto)
    for i in range(pos.shape[0]):
        for j in range(3):
            pos_plus = pos.clone()
            pos_plus[i, j] += eps
            pos_minus = pos.clone()
            pos_minus[i, j] -= eps
            e_plus = model.energy(z, pos_plus, graph_indexes)[:, 0].sum()
            e_minus = model.energy(z, pos_minus, graph_indexes)[:, 0].sum()
            forces_fd[i, j] = -(e_plus - e_minus) / (2 * eps)

    assert forces_auto.abs().max() > 1e-6
    rel_error = (forces_fd - forces_auto).abs().max() / forces_fd.abs().max()
    assert rel_error < 1e-4

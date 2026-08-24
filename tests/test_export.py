"""M5 export tests: LammpsWrapper parity with the training pipeline.

The wrapper's promise, pinned here: its scalar total energy equals the
training pipeline's physical energy (model mean-pool + standardizer.
inverse), and its forces are the physical gradient. Everything runs on a
small real PaiNNCore + hand-set statistics - no QM9 download, no LAMMPS,
following the suite's zero-data philosophy. The M5 acceptance's 1e-6
parity is exactly test_wrapper_energy_matches_training_pipeline below.
"""

import torch

from potlab.data.qm9 import Qm9Standardizer
from potlab.export.lammps import LammpsWrapper
from potlab.models.painn.core import PaiNNCore
from potlab.models.painn.model import PaiNNModel


def _small_model_kwargs():
    """A fast config: few features, few RBFs, two rounds. Seconds on CPU."""
    return dict(
        num_message_passing_layers=2,
        num_features=16,
        num_outputs=1,
        num_rbf_features=4,
        num_unique_atoms=100,
        cutoff_dist=5.0,
    )


def _make_standardizer():
    """Hand-set statistics (contract-shaped, no fit) - float64 throughout.

    The values are arbitrary but deliberately far from the identity
    (std=2, mean=1, nonzero refs): a wrapper that forgot any of the three
    would fail every assertion below.
    """
    std = Qm9Standardizer()
    std.mean = torch.tensor(1.0, dtype=torch.float64)
    std.std = torch.tensor(2.0, dtype=torch.float64)
    std.atom_refs = torch.arange(10, dtype=torch.float64).unsqueeze(-1)
    return std


def _system():
    """Two molecules (4 + 3 atoms) with graph and edge indices."""
    z = torch.tensor([1, 6, 8, 1, 6, 6, 8])
    pos = torch.rand(7, 3, dtype=torch.float64)
    graph_indexes = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    return z, pos, graph_indexes


def _make_wrapper(model, standardizer, energy_index=0):
    """A wrapper over the model's core, with an EXPLICIT energy column.

    The real assembly passes dm.energy_index; the tests always spell the
    value out so no test leans on the default.
    """
    return LammpsWrapper(model.painn_core, standardizer, energy_index=energy_index)


def test_wrapper_energy_matches_training_pipeline():
    # Same weights, same statistics, two paths: the wrapper (per-atom
    # inverse, summed) vs the training pipeline (mean-pool + per-molecule
    # inverse). float64: the M5 acceptance's 1e-6 parity is measured on
    # the math, not on float32 rounding.
    model = PaiNNModel(**_small_model_kwargs()).double()
    std = _make_standardizer()
    wrapper = _make_wrapper(model, std)
    z, pos, graph_indexes = _system()
    idx_i, idx_j = model._radius_graph(pos, graph_indexes)

    pipeline = std.inverse(model.energy(z, pos, graph_indexes), z, graph_indexes)
    wrapped = wrapper.energy(z, pos, idx_i, idx_j)

    assert torch.allclose(wrapped, pipeline.sum(), atol=1e-8)


def test_wrapper_forces_match_finite_differences():
    model = PaiNNModel(**_small_model_kwargs()).double()
    wrapper = _make_wrapper(model, _make_standardizer())
    z, pos, graph_indexes = _system()
    idx_i, idx_j = model._radius_graph(pos, graph_indexes)

    _, forces_auto = wrapper.energy_and_forces(z, pos, idx_i, idx_j)
    forces_auto = forces_auto.detach()
    # energy_and_forces sets requires_grad_ on pos in place; the FD loop
    # wants clean leaf tensors.
    pos = pos.detach().clone()

    eps = 1e-6
    forces_fd = torch.zeros_like(forces_auto)
    for i in range(pos.shape[0]):
        for j in range(3):
            pos_plus = pos.clone()
            pos_plus[i, j] += eps
            pos_minus = pos.clone()
            pos_minus[i, j] -= eps
            e_plus = wrapper.energy(z, pos_plus, idx_i, idx_j)
            e_minus = wrapper.energy(z, pos_minus, idx_i, idx_j)
            forces_fd[i, j] = -(e_plus - e_minus) / (2 * eps)

    # Guard: if the random geometry were gradient-flat, the ratio below
    # would be noise over ~0 - fail loudly instead of "passing" vacuously.
    assert forces_auto.abs().max() > 1e-6
    # Relative to the largest component: per-component division blows up
    # on near-zero force components.
    rel_error = (forces_fd - forces_auto).abs().max() / forces_fd.abs().max()
    assert rel_error < 1e-4


def test_wrapper_forces_are_physical_not_mean_pooled():
    # The M4 caveat, resolved and pinned: PaiNNModel.energy_and_forces
    # returns the mean-pooled gradient (physical / n_atoms, standardized
    # scale); the wrapper returns PHYSICAL forces. The relationship is
    # f_wrapper = f_model * n_atoms[atom] * std - the n cancels in the
    # wrapper because inverse multiplies the pooled value back by n.
    model = PaiNNModel(**_small_model_kwargs()).double()
    std = _make_standardizer()
    wrapper = _make_wrapper(model, std)
    z, pos, graph_indexes = _system()
    idx_i, idx_j = model._radius_graph(pos, graph_indexes)

    _, f_model = model.energy_and_forces(z, pos, graph_indexes)
    _, f_wrapper = wrapper.energy_and_forces(z, pos, idx_i, idx_j)

    n_atoms = torch.bincount(graph_indexes)[graph_indexes].unsqueeze(-1)
    assert torch.allclose(f_model * n_atoms * std.std, f_wrapper, atol=1e-8)


def test_wrapper_core_is_shared_with_model():
    # The wrapper and the model must hold the SAME core object (the M5
    # assembly wires model.painn_core in) - sharing is what makes the
    # parity tests above meaningful, not two separately-trained copies.
    model = PaiNNModel(**_small_model_kwargs())
    wrapper = _make_wrapper(model, _make_standardizer())
    assert wrapper.core is model.painn_core


def test_wrapper_energy_index_selects_the_column():
    # The wrapper's column choice is driven by the energy_index argument
    # (the export assembly fills it from dm.energy_index) - not by a
    # hardcoded 0. A 2-output core wrapped with energy_index=1 must sum
    # column 1's physical values.
    kwargs = _small_model_kwargs()
    kwargs["num_outputs"] = 2
    model = PaiNNModel(**kwargs).double()
    std = _make_standardizer()
    wrapper = _make_wrapper(model, std, energy_index=1)
    z, pos, graph_indexes = _system()
    idx_i, idx_j = model._radius_graph(pos, graph_indexes)

    contribs = model.painn_core(z, pos, idx_i, idx_j)
    expected = std.inverse_per_atom(contribs, z)[:, 1].sum()

    assert torch.allclose(wrapper.energy(z, pos, idx_i, idx_j), expected, atol=1e-8)

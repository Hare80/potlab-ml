"""LAMMPS-facing wrapper: trained core + standardizer (M5, MLIAP-Python).

The MLIAP-Python plugin glue (Phase B, a separate file) hands LAMMPS'
own neighbor list to this wrapper as ``(z, pos, idx_i, idx_j)`` and gets
back physical energies and forces. No LAMMPS import lives in this module:
the math here is pure PyTorch and fully testable without LAMMPS.

Why the wrapper sits on the core: the model layer mean-pools per
molecule (standardized space), but LAMMPS has per-atom outputs and no
molecule boundaries. ``standardizer.inverse_per_atom`` (the M5 protocol
addition) bridges the granularity gap - the wrapper never touches
graph_indexes and never re-implements the standardizer's math.
"""

import torch
import torch.nn as nn
from torch import Tensor

from potlab.data.base import Standardizer


class LammpsWrapper(nn.Module):
    """A trained core + its standardizer under the (z, pos, idx_i, idx_j) interface.

    ``energy`` returns the scalar total energy of the WHOLE system: the
    core's per-atom contributions become absolute per-atom energies via
    ``inverse_per_atom`` (c_i * std + mean + refs[z_i]) and are summed.
    That sum is algebraically the training pipeline's physical energy:
    sum_i e_i = (mean_pool(c) * std + mean) * n + refs. Forces are its
    gradient - PHYSICAL forces: no mean-pooling 1/n survives (the
    inverse's * n cancels it, and refs are constant), so the M4 caveat
    about PaiNNModel.energy_and_forces does not apply here.

    The standardizer is a plain attribute, not a module: its statistics
    travel with the training checkpoint (ckpt["standardizer"]) and it
    moves them to the caller's device itself (Qm9Standardizer._stats_on).
    """

    def __init__(self, core: nn.Module, standardizer: Standardizer) -> None:
        super().__init__()
        self.core = core
        self.standardizer = standardizer

    def energy(self, z: Tensor, pos: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """Scalar total energy (0-dim). Column 0 of a multi-output core is the energy column."""
        contribs = self.core(z, pos, idx_i, idx_j)  # [N_atoms, num_outputs]
        e_per_atom = self.standardizer.inverse_per_atom(contribs, z)  # absolute, per atom
        return e_per_atom[:, 0].sum()  # LAMMPS wants ONE number

    def forces(self, z: Tensor, pos: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """Per-atom forces [N_atoms, 3]: -d(total energy)/dpos.

        ``pos.requires_grad_(True)`` mutates the input in place (the same
        convention as BaseModel.energy_and_forces). Inference-only: no
        create_graph, the force graph is not reused.
        """
        pos = pos.requires_grad_(True)
        energy = self.energy(z, pos, idx_i, idx_j)
        return -torch.autograd.grad(energy, pos)[0]

    def energy_and_forces(
        self, z: Tensor, pos: Tensor, idx_i: Tensor, idx_j: Tensor
    ) -> tuple[Tensor, Tensor]:
        """(energy, forces) in one forward + one backward - the mliap hot path."""
        pos = pos.requires_grad_(True)
        energy = self.energy(z, pos, idx_i, idx_j)
        forces = -torch.autograd.grad(energy, pos)[0]
        return energy, forces

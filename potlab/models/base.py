"""BaseModel protocol (DESIGN.md §1)."""

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module):
    """The protocol every model implements.

    Total energy is the common denominator: trainers, metrics and the
    standardizer call ``energy`` (and ``energy_and_forces`` when the
    dataset provides forces). Per-atom contributions are an optional
    capability, never a requirement - a model that regresses the total
    energy directly is a first-class citizen.
    """

    def energy(self, z: Tensor, pos: Tensor, graph_indexes: Tensor) -> Tensor:
        """Per-molecule property. [N_atoms]/[N_atoms,3]/[N_atoms] -> [N_graphs, num_outputs]."""
        raise NotImplementedError

    def energy_and_forces(
        self, z: Tensor, pos: Tensor, graph_indexes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Total energy + per-atom forces via autograd (DESIGN.md §1).

        Forces are the negative gradient of the total energy with respect
        to positions. Grad of a sum = sum of grads, and a molecule's energy
        never depends on another molecule's positions (edges are intra-graph
        only), so the batch sum computes every molecule's forces exactly as
        a per-molecule loop would - no cross-molecule mixing.

        ``create_graph=True`` keeps the force graph alive: a force loss
        must itself be differentiable.
        """
        pos = pos.requires_grad_(True)
        energy = self.energy(z, pos, graph_indexes)
        forces = -torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
        return energy, forces

    def atomic_contributions(
        self, z: Tensor, pos: Tensor, graph_indexes: Tensor
    ) -> Tensor:
        """Optional per-atom decomposition, [N_atoms, num_outputs].

        Diagnostics and decomposition plots only - unsupported models raise
        NotImplementedError, and that is fine: trainers never require it.
        """
        raise NotImplementedError

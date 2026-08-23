"""PaiNN core: pure PyTorch, graph-agnostic, TorchScript-safe (PLAN.md M2).

The core takes ``(z, pos, idx_i, idx_j)`` - edges come from OUTSIDE. That
split exists for export: LAMMPS owns the neighbor list at inference time,
so the exported artifact is the core only (no PyG, no Python control flow
in ``forward``; every op here is a plain tensor op).

Module layout mirrors the original ``src.models.PaiNN`` exactly
(atom_embedding / cosine_cut / radial_basis / message_blocks /
update_blocks / readout_network), so a state_dict trained by the old
monolithic model loads 1:1 - that is the basis of the M2 parity test.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


def build_readout_network(
    num_in_features: int,
    num_out_features: int = 1,
    num_layers: int = 2,
    activation: nn.Module = nn.SiLU,
) -> nn.Sequential:
    """Per-atom MLP compressing [num_features] into [num_outputs] contributions.

    Hidden widths halve layer by layer - a funnel that compresses
    information gradually while the parameter count shrinks. No activation
    after the last Linear: regression outputs need the whole real line.
    """
    num_neurons = [
        num_in_features,
        *[
            max(num_out_features, num_in_features // 2 ** (i + 1))
            for i in range(num_layers - 1)
        ],
        num_out_features,
    ]
    readout_network = nn.Sequential()
    for i, (n_in, n_out) in enumerate(zip(num_neurons[:-1], num_neurons[1:])):
        readout_network.append(nn.Linear(n_in, n_out))
        if i < num_layers - 1:
            readout_network.append(activation())
    return readout_network


class SinusoidalRBFLayer(nn.Module):
    """Sinusoidal radial basis: one distance -> ``num_basis`` fingerprint values.

    ``sin(freq * d) / d`` at frequencies pi/cutoff, 2*pi/cutoff, ... The
    division keeps the features finite and smooth at d -> 0 (sin(x) ~ x).
    Frequencies are fixed, so they live in a buffer (moves with the model,
    never trained).
    """

    def __init__(self, num_basis: int = 20, cutoff_dist: float = 5.0) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_dist = cutoff_dist
        self.register_buffer(
            "freqs",
            math.pi * torch.arange(1, self.num_basis + 1) / self.cutoff_dist,
        )

    def forward(self, distances: Tensor) -> Tensor:
        distances = distances.unsqueeze(-1)  # [E] -> [E, 1], broadcasts over freqs
        return torch.sin(self.freqs * distances) / distances


class CosineCutoff(nn.Module):
    """Smooth switch fading interactions from 1 (d=0) to 0 (d=cutoff).

    A hard cutoff is discontinuous - the gradient jumps at the boundary and
    training destabilizes. 0.5 * (cos(pi*d/cutoff) + 1) decays smoothly and
    is exactly 0 beyond cutoff.
    """

    def __init__(self, cutoff_dist: float = 5.0) -> None:
        super().__init__()
        self.cutoff_dist = cutoff_dist

    def forward(self, distances: Tensor) -> Tensor:
        # torch.pi, not math.pi: the forward must stay TorchScript-safe.
        return torch.where(
            distances < self.cutoff_dist,
            0.5 * (torch.cos(distances * torch.pi / self.cutoff_dist) + 1),
            0,
        )


class PaiNNMessageBlock(nn.Module):
    """Message block (Schuett et al. 2021, fig. 1 left): neighbor aggregation.

    Each edge carries a message = phi[idx_j] * W: phi comes from the
    sender atom's scalar features, W weights the edge by its RBF-encoded
    distance (modulated by the cosine cutoff - nearby neighbors talk
    louder). Messages accumulate into the receiver atoms with
    ``index_add_``, the core op of message passing. The 3*num_features
    output splits into the paper's three channels: ss (scalar->scalar),
    vv (vector->vector), vs (scalar->vector - scalar information creating
    directional information along the bond, the polarization source).
    """

    def __init__(self, num_features: int = 128, num_rbf_features: int = 20) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features

        self.scalar_network = nn.Sequential(
            nn.Linear(
                in_features=self.num_features,
                out_features=self.num_features,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self.num_features,
                out_features=3 * self.num_features,
            ),
        )
        self.rbf_network = nn.Linear(
            in_features=self.num_rbf_features,
            out_features=3 * self.num_features,
        )

    def forward(
        self,
        idx_i: Tensor,
        idx_j: Tensor,
        rel_dir: Tensor,
        rel_dist_cut: Tensor,
        rbf_features: Tensor,
        scalar_features: Tensor,
        vector_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        phi = self.scalar_network(scalar_features)  # [N, 3F]
        W = self.rbf_network(rbf_features) * rel_dist_cut.unsqueeze(-1)  # [E, 3F]
        phi_W = phi[idx_j] * W  # [E, 3F]: the sender's phi, gathered per edge
        phi_W_vv, phi_W_ss, phi_W_vs = torch.split(phi_W, self.num_features, dim=-1)

        # Scalar residual: sum the ss channel into the receiver atoms.
        scalar_residuals = torch.zeros_like(scalar_features)
        scalar_residuals.index_add_(dim=0, index=idx_i, source=phi_W_ss)

        # Vector residual: neighbor vectors scaled by the vv channel, plus
        # the vs channel creating new directional information along the
        # unit direction between the two atoms.
        vector_residuals_per_edge = (
            vector_features[idx_j] * phi_W_vv.unsqueeze(-1)
            + phi_W_vs.unsqueeze(-1) * rel_dir.unsqueeze(-2)
        )
        vector_residuals = torch.zeros_like(vector_features)
        vector_residuals.index_add_(
            dim=0, index=idx_i, source=vector_residuals_per_edge
        )

        # Residual connections: gradients get a shortcut back, training
        # stays stable through deep stacks.
        return (
            scalar_features + scalar_residuals,
            vector_features + vector_residuals,
        )


class PaiNNUpdateBlock(nn.Module):
    """Update block (fig. 1 right): per-atom coupling of scalar and vector features.

    The key invariant: only the NORM of a transformed vector feeds back
    into scalars - norms are rotation-invariant, raw vector components are
    not, and reading components would break rotational invariance.
    """

    def __init__(self, num_features: int = 128) -> None:
        super().__init__()
        self.num_features = num_features

        # Two bias-free linear maps mixing vector channels (nn.Linear acts
        # on the last dim, hence the movedim dance).
        self.U = nn.Linear(
            in_features=self.num_features, out_features=self.num_features, bias=False
        )
        self.V = nn.Linear(
            in_features=self.num_features, out_features=self.num_features, bias=False
        )
        # Input: [norm of V-transformed vectors; scalar features]; output
        # splits into avv / asv / ass (the paper's three update channels).
        self.scalar_vector_network = nn.Sequential(
            nn.Linear(
                in_features=2 * self.num_features,
                out_features=self.num_features,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self.num_features,
                out_features=3 * self.num_features,
            ),
        )

    def forward(
        self, scalar_features: Tensor, vector_features: Tensor
    ) -> tuple[Tensor, Tensor]:
        U_vector_features = self.U(vector_features.movedim(-2, -1)).movedim(-2, -1)
        V_vector_features = self.V(vector_features.movedim(-2, -1)).movedim(-2, -1)

        a = self.scalar_vector_network(
            torch.cat(
                [
                    torch.linalg.vector_norm(V_vector_features, dim=-1),
                    scalar_features,
                ],
                dim=-1,
            )
        )
        a_vv, a_sv, a_ss = torch.split(a, self.num_features, dim=-1)

        # Vector residual: U-transformed vectors scaled by a_vv (vector
        # updates only from vectors, direction preserved).
        vector_residuals = U_vector_features * a_vv.unsqueeze(-1)
        # Scalar residual: own update a_ss plus the vector->scalar channel
        # a_sv * <U, V> - the dot product is rotation-invariant.
        scalar_residuals = (
            a_ss + a_sv * torch.sum(U_vector_features * V_vector_features, dim=-1)
        )

        return (
            scalar_features + scalar_residuals,
            vector_features + vector_residuals,
        )


class PaiNNCore(nn.Module):
    """The graph-agnostic PaiNN network: (z, pos, idx_i, idx_j) -> per-atom contributions.

    Edge geometry (directions / distances / RBF features) is computed here
    from positions - it is pure tensor math, so it belongs to the core.
    Only the neighbor SEARCH (radius_graph) stays outside in model.py; at
    export time LAMMPS supplies idx_i/idx_j itself. idx_i/idx_j must not
    contain self-loops (a zero rel_dist would divide by zero in rel_dir).
    """

    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist

        # Row 0 is padding (QM9 atom numbers start at 1).
        self.atom_embedding = nn.Embedding(
            num_embeddings=self.num_unique_atoms + 1,
            embedding_dim=num_features,
            padding_idx=0,
        )
        self.cosine_cut = CosineCutoff(cutoff_dist=self.cutoff_dist)
        self.radial_basis = SinusoidalRBFLayer(
            num_basis=self.num_rbf_features, cutoff_dist=self.cutoff_dist
        )
        # One independent (message + update) pair per round: each round
        # learns a different scale - round 1 sees direct neighbors, round 2
        # sees the neighbors' neighbors, and so on.
        self.message_blocks = nn.ModuleList()
        self.update_blocks = nn.ModuleList()
        for _ in range(self.num_message_passing_layers):
            self.message_blocks.append(
                PaiNNMessageBlock(
                    num_features=self.num_features,
                    num_rbf_features=self.num_rbf_features,
                )
            )
            self.update_blocks.append(
                PaiNNUpdateBlock(num_features=self.num_features)
            )

        self.readout_network = build_readout_network(
            num_in_features=self.num_features,
            num_out_features=self.num_outputs,
            num_layers=2,
            activation=nn.SiLU,
        )

    def forward(
        self, z: Tensor, pos: Tensor, idx_i: Tensor, idx_j: Tensor
    ) -> Tensor:
        """Per-atom contributions, [N_atoms, num_outputs]."""
        scalar_features = self.atom_embedding(z)  # [N, F]
        vector_features = torch.zeros(  # [N, F, 3]: starts empty, polarization
            scalar_features.size() + (3,),  # accumulates through the rounds
            dtype=scalar_features.dtype,
            device=scalar_features.device,
        )

        # Edge geometry, computed once and shared by all rounds (geometry
        # does not change between rounds).
        rel_pos = pos[idx_j] - pos[idx_i]  # [E, 3]
        rel_dist = torch.linalg.vector_norm(rel_pos, dim=1)  # [E]
        rel_dir = rel_pos / rel_dist.unsqueeze(-1)  # [E, 3] unit vectors
        rel_dist_cut = self.cosine_cut(rel_dist)  # [E] 0..1, fading with d
        rbf_features = self.radial_basis(rel_dist)  # [E, num_rbf_features]

        # zip over two ModuleLists: enumeration is the TorchScript-safe way
        # to iterate modules (indexing with a loop variable is NOT - the
        # M2 script smoke caught that the other way around).
        for message, update in zip(self.message_blocks, self.update_blocks):
            scalar_features, vector_features = message(
                idx_i,
                idx_j,
                rel_dir,
                rel_dist_cut,
                rbf_features,
                scalar_features,
                vector_features,
            )
            scalar_features, vector_features = update(
                scalar_features, vector_features
            )

        # Only scalar features feed the readout (vector features are an
        # internal messenger for directional information).
        return self.readout_network(scalar_features)  # [N, num_outputs]

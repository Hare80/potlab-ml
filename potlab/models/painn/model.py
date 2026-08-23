"""PaiNN model: PyG graph builder + PaiNNCore (DESIGN.md §1 protocol).

The graph builder (radius_graph) is the only place PyG appears in the
model layer. LAMMPS owns the neighbor list at inference time (M5:
pair_style mliap), so the LAMMPS-side entry point is PaiNNCore with
edges supplied by the plugin - the model's own radius_graph is
training-side only.
"""

import torch
from torch import Tensor
from torch_geometric.nn import radius_graph

from potlab.models.base import BaseModel
from potlab.models.painn.core import PaiNNCore
from potlab.registry import register_model


@register_model("painn")
class PaiNNModel(BaseModel):
    """PaiNN under the BaseModel protocol: graph builder + core + per-molecule mean.

    ``energy`` returns the per-molecule MEAN of the core's per-atom
    contributions. Mean pooling makes the output the standardized-space
    prediction directly: the training target is ((y - refs) / n_atoms -
    mean) / std per molecule, so the standardizer pairs with this model
    unchanged (M2 step 5). Sum vs mean is each model's own choice - a
    model regressing total energies (MD17) would sum here.
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

        self.painn_core = PaiNNCore(
            num_message_passing_layers=num_message_passing_layers,
            num_features=num_features,
            num_outputs=num_outputs,
            num_rbf_features=num_rbf_features,
            num_unique_atoms=num_unique_atoms,
            cutoff_dist=cutoff_dist,
        )

    def _radius_graph(self, atom_positions, graph_indexes):
        """Edge indices, replicating the old monolithic forward exactly.

        batch=graph_indexes keeps edges inside molecules - an
        inter-molecular edge would let one molecule's energy leak into
        another's (and into its forces).
        """
        _, num_nodes_per_graph = torch.unique(graph_indexes, return_counts=True)
        idx_i, idx_j = radius_graph(
            x=atom_positions,
            r=self.cutoff_dist,
            batch=graph_indexes,
            loop=False,  # no self-loops: a zero rel_dist would divide by zero
            max_num_neighbors=torch.max(num_nodes_per_graph),
            flow="target_to_source",
            batch_size=len(num_nodes_per_graph),
        )
        return idx_i, idx_j

    def energy(self, z: Tensor, pos: Tensor, graph_indexes: Tensor) -> Tensor:
        """Per-molecule contribution means. [N_atoms]/[N_atoms,3]/[N_atoms] -> [N_graphs, num_outputs]."""
        contribs = self.atomic_contributions(z, pos, graph_indexes)
        # One row per MOLECULE, filled by index_add_ along the atom axis
        # (PyG guarantees graph indices are contiguous 0..N-1).
        n_graphs = int(graph_indexes.max().item()) + 1
        out = torch.zeros(
            n_graphs,
            self.num_outputs,
            device=contribs.device,
            dtype=contribs.dtype,
        )
        out.index_add_(dim=0, index=graph_indexes, source=contribs)
        # Per-atom average: bincount of the contiguous batch indices gives
        # the molecule sizes (the same math the data layer's standardizer
        # mirrors on the label side - the pair is what makes transform/
        # inverse work unchanged).
        n_atoms = torch.bincount(graph_indexes).unsqueeze(-1)  # [N_graphs, 1]
        return out / n_atoms

    def atomic_contributions(
        self, z: Tensor, pos: Tensor, graph_indexes: Tensor
    ) -> Tensor:
        """Per-atom decomposition, [N_atoms, num_outputs] (diagnostics)."""
        idx_i, idx_j = self._radius_graph(pos, graph_indexes)
        return self.painn_core(z, pos, idx_i, idx_j)

"""Per-sample transforms shared across datasets.

Transforms run on each molecule's Data object when the dataset yields it,
before batching. They may rely on the batch contract (DESIGN.md §3) but
never on one specific dataset.
"""

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class GetTarget(BaseTransform):
    """Keep one target column from a multi-property label matrix.

    QM9 stores all 19 properties per molecule in ``data.y`` ([1, 19] for
    one molecule); this keeps column ``target`` only. List indexing
    ``y[:, [target]]`` (not ``y[:, target]``) preserves the 2D shape
    [1, 1] that the batch contract requires ([N_graphs, num_outputs]).
    """

    def __init__(self, target: int) -> None:
        # Stored as a list on purpose: list indexing keeps the column
        # dimension alive, integer indexing would squeeze it away.
        self.target = [target]

    def forward(self, data: Data) -> Data:
        # All rows (molecules) x the one target column: [1, 19] -> [1, 1].
        data.y = data.y[:, self.target]
        return data

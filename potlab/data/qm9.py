"""QM9 data module and standardizer.

Loads QM9 (target property, seeded shuffle, three-way split), yields
batches under the DESIGN.md §3 contract, and standardizes targets with
QM9's built-in per-element atom references.
"""

import warnings
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from potlab import ROOT
from potlab.data.base import BaseDataModule, Standardizer
from potlab.data.transforms import GetTarget
from potlab.registry import register_dataset

# QM9 properties that are not energies (dipole moment, heat capacity, ...):
# their native units are kept for display instead of converting eV -> meV.
NO_UNIT_CONVERSION = {0, 1, 5, 11, 16, 17, 18}

# QM9 targets that are POTENTIAL-ENERGY-SURFACE energies: conservative
# quantities whose negative position gradient IS the interatomic force.
# Only U0 (7) qualifies today. Deliberately excluded: U/H/G (contain
# ZPE + thermochemistry - nuclear-motion contributions, not electronic
# forces), ZPVE itself, HOMO/LUMO/gap (orbital energies - their gradient
# is a property gradient, not a force), and the non-energy properties.
# Note U0_atom (12) differs from U0 by a constant and would have the
# same gradient - add it here when a run actually trains it.
PES_ENERGY_TARGETS = {7}


def _num_atoms_per_graph(graph_indexes: Tensor) -> Tensor:
    """Atom count per molecule, [N_graphs, 1]. Batch indices are contiguous 0..N-1."""
    return torch.bincount(graph_indexes).unsqueeze(-1)


def _sum_per_graph(values: Tensor, graph_indexes: Tensor, n_graphs: int) -> Tensor:
    """Sum per-atom values into [N_graphs, num_outputs] rows."""
    out = torch.zeros(
        n_graphs, values.shape[-1], device=values.device, dtype=values.dtype
    )
    out.index_add_(dim=0, index=graph_indexes, source=values)
    return out


def _validate_splits(
    splits: Union[List[int], List[float]], n_mols: int
) -> list[int]:
    """Resolve and validate a splits spec into [train, val, test] counts.

    All ints = molecule counts; all floats = proportions of the dataset.
    Anything mixed is a config error, not a silent fallthrough. Extracted
    from QM9DataModule.setup so the guard is testable without the QM9
    download (the M4 data-contract tests pin the error paths).
    """
    if all(type(split) == int for split in splits):
        split_sizes = splits
    elif all(type(split) == float for split in splits):
        split_sizes = [int(n_mols * prop) for prop in splits]
    else:
        raise ValueError(
            f"Invalid splits: {splits}. Must be all int or all float."
        )
    if len(split_sizes) != 3:
        raise ValueError(
            f"Invalid splits: {splits}. Expected 3 values [train, val, test]."
        )

    split_idx = np.cumsum(split_sizes)
    # Guard: train+val must leave at least one molecule for the test set.
    if split_idx[1] >= n_mols:
        raise ValueError(
            f"Invalid splits: {splits}. train + val ({split_idx[1]}) "
            f"must be less than the dataset size ({n_mols}). "
            "When using subset_size, scale the splits down to match "
            "(e.g. [800, 100, 100] for a 1000-molecule subset, or use "
            "proportions like [0.8, 0.1, 0.1])."
        )
    return split_sizes


class Qm9Standardizer(Standardizer):
    """QM9 target standardization (the original get_target_stats, rehomed).

    Pipeline, mirrored exactly by ``inverse``:

    1. subtract per-element atom references (QM9's atomref table)
    2. divide by atom count (per-molecule average)
    3. shift/scale with train-set mean and std
    """

    def fit(self, train_data: "QM9DataModule") -> None:
        # Statistics come from the TRAIN split only - touching val/test
        # here would leak information.
        atom_refs = train_data.data_train.atomref(train_data.target)

        ys = []
        for batch in train_data.train_dataloader(shuffle=False):
            y = batch.y.clone()  # clone: never mutate the dataset in place
            # Step 1: subtract the molecule's atom-reference sum.
            y.index_add_(dim=0, index=batch.batch, source=-atom_refs[batch.z])
            # Step 2: divide by atom count (per-atom average).
            _, num_atoms = torch.unique(batch.batch, return_counts=True)
            y = y / num_atoms.unsqueeze(-1)
            ys.append(y)

        y = torch.cat(ys, dim=0)
        # Step 3: shift/scale statistics, computed on the transformed labels.
        self.mean = y.mean()
        self.std = y.std()
        self.atom_refs = atom_refs

    def _stats_on(self, device: torch.device):
        """Statistics moved to the caller's device.

        fit() runs on CPU; transform/inverse may be called with GPU tensors,
        and cross-device arithmetic raises. Not an nn.Module, so the move is
        explicit here (a no-op when the input is already on CPU).
        """
        return self.mean.to(device), self.std.to(device), self.atom_refs.to(device)

    def transform(self, y: Tensor, z: Tensor, graph_indexes: Tensor) -> Tensor:
        """Labels -> standardized space: ((y - refs) / n_atoms - mean) / std."""
        mean, std, atom_refs = self._stats_on(y.device)
        n_graphs = y.shape[0]
        refs = _sum_per_graph(atom_refs[z], graph_indexes, n_graphs)
        per_atom = (y - refs) / _num_atoms_per_graph(graph_indexes)
        return (per_atom - mean) / std

    def inverse_per_atom(self, contribs: Tensor, z: Tensor) -> Tensor:
        """Per-atom standardized contributions -> per-atom physical energies.

        ``c_i * std + mean + refs[z_i]``: unstandardize, then add the
        atom's element reference. The base of the per-molecule ``inverse``
        (M5: the LAMMPS mliap wrapper calls this directly - LAMMPS has
        per-atom outputs and no molecule boundaries).
        """
        mean, std, atom_refs = self._stats_on(contribs.device)
        return contribs * std + mean + atom_refs[z]

    def inverse(self, energy_pred: Tensor, z: Tensor, graph_indexes: Tensor) -> Tensor:
        """Model energies -> physical units: inverse_per_atom, aggregated per molecule.

        Each molecule's pooled value is broadcast to its atoms, transformed
        per atom, and summed per graph. The algebra: n_atoms * (E * std +
        mean) + refs - exactly the inverse of transform.
        """
        per_atom = self.inverse_per_atom(energy_pred[graph_indexes], z)
        return _sum_per_graph(per_atom, graph_indexes, energy_pred.shape[0])

    def state_dict(self) -> dict:
        """Statistics needed to reproduce transform/inverse - saved with checkpoints."""
        return {"mean": self.mean, "std": self.std, "atom_refs": self.atom_refs}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.std = state["std"]
        self.atom_refs = state["atom_refs"]


@register_dataset("qm9")
class QM9DataModule(BaseDataModule):
    """QM9 under the BaseDataModule protocol.

    Splits: seeded global shuffle, then train/val/test by index. The third
    split number documents the expected test size; the test set actually
    takes everything train+val leave over (the original behavior, kept).
    """

    def __init__(
        self,
        target: int = 7,
        data_dir: str = "data/",
        batch_size_train: int = 100,
        batch_size_eval: int = 1000,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [110000, 10000, 10831],
        seed: int = 0,
        subset_size: Optional[int] = None,
    ) -> None:
        self.target = target
        # Anchor relative dirs to ROOT: the config's 'data/' means the
        # repo-root data folder, never CWD-relative (a run started from
        # scripts/ used to download QM9 into scripts/data/).
        self.data_dir = ROOT / data_dir if not Path(data_dir).is_absolute() else Path(data_dir)
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size  # debug: cap the dataset to N molecules

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None:
        # First call downloads (~100 MB) and preprocesses; later calls skip.
        QM9(root=self.data_dir)

    def setup(self) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

        # Seeded shuffle: same seed -> same order -> reproducible splits.
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        if self.subset_size is not None:
            dataset = dataset[: self.subset_size]

        # The guard lives in _validate_splits (pure, unit-tested in M4);
        # setup() only slices.
        split_sizes = _validate_splits(self.splits, len(dataset))
        split_idx = np.cumsum(split_sizes)

        self.data_train = dataset[: split_idx[0]]
        self.data_val = dataset[split_idx[0] : split_idx[1]]
        self.data_test = dataset[split_idx[1] :]  # test = remainder

        # The third number is documentation of the expected test size; warn
        # when the actual remainder differs (a typo here used to silently
        # change the test set). Exempt: float proportions (truncation means
        # they never sum exactly) and subset runs (every number is
        # deliberately approximate there).
        if all(type(split) == int for split in self.splits) and self.subset_size is None:
            actual_test = len(self.data_test)
            if actual_test != split_sizes[2]:
                warnings.warn(
                    f"Test set has {actual_test} molecules, not the {split_sizes[2]} "
                    "requested in splits (test takes whatever train+val leave over)."
                )

    def make_standardizer(self) -> Standardizer:
        standardizer = Qm9Standardizer()
        standardizer.fit(self)
        return standardizer

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    @property
    def has_forces(self) -> bool:
        return False  # QM9 ships energies/properties, no forces

    @property
    def num_targets(self) -> int:
        return 1  # GetTarget always slices ONE property column

    @property
    def energy_index(self) -> Optional[int]:
        # Single-target QM9: the model's only output column holds the
        # target. It IS the energy only for PES targets (see the module
        # comment on PES_ENERGY_TARGETS).
        return 0 if self.target in PES_ENERGY_TARGETS else None

    @property
    def unit_conversion(self) -> Callable:
        # Display only: energies are stored in eV and shown in meV;
        # non-energy properties keep their native units.
        if self.target in NO_UNIT_CONVERSION:
            return lambda t: t
        return lambda t: 1000.0 * t

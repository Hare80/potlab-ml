"""Data layer contracts: Standardizer and BaseDataModule.

These two classes are the only entry points the trainer knows about. The
batch contract (z / pos / y / batch / forces) is specified in DESIGN.md §3;
everything else about a dataset is private to its concrete implementation.
"""

from typing import Callable

from torch import Tensor
from torch.utils.data import DataLoader


class Standardizer:
    """Owns every target transformation.

    It operates on **labels**, so it is independent of what the model
    outputs. The canonical pipeline, mirrored exactly by ``inverse``:

    1. subtract atom references (dataset-provided, or fit by lstsq)
    2. optionally divide by atom count (property datasets like QM9)
    3. shift/scale to zero mean, unit variance (train statistics only)

    The network therefore always trains on small, zero-centered targets.
    """

    def fit(self, train_data) -> None:
        """Compute and store all statistics from train data.

        What ``train_data`` is depends on the concrete standardizer - for
        QM9 it is the data module itself. Must be called before
        ``transform`` / ``inverse``.
        """
        raise NotImplementedError

    def transform(self, y: Tensor, z: Tensor, graph_indexes: Tensor) -> Tensor:
        """Labels -> standardized space (subtract refs, shift/scale)."""
        raise NotImplementedError

    def inverse(self, energy_pred: Tensor, z: Tensor, graph_indexes: Tensor) -> Tensor:
        """Model energies -> physical units (the exact reverse of transform).

        The LAMMPS plugin wrapper (M5: pair_style mliap) calls this on
        model outputs - LAMMPS needs absolute energies, so this direction
        is what LAMMPS sees.
        """
        raise NotImplementedError

    def state_dict(self) -> dict:
        """Everything needed to reproduce the transforms - saved with checkpoints."""
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        """Restore the state saved by ``state_dict`` (used by --resume)."""
        raise NotImplementedError


class BaseDataModule:
    """Protocol every dataset implements.

    Splitting strategy is private to each dataset (QM9: seeded global
    shuffle; MD/VASP trajectories: split by simulation run). The trainer
    calls only the methods below and never touches the subsets directly.
    """

    def prepare_data(self) -> None:
        """One-time download / raw-file parsing. Safe to call repeatedly."""
        raise NotImplementedError

    def setup(self) -> None:
        """Split the dataset and build the train/val/test subsets."""
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """Batches of train data (shuffled), following the DESIGN.md §3 contract."""
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        """Batches of validation data (not shuffled)."""
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        """Batches of test data (not shuffled)."""
        raise NotImplementedError

    def make_standardizer(self) -> Standardizer:
        """Build and fit the dataset-specific standardizer.

        Fitting happens on train data only - statistics must never touch
        val/test (that would leak).
        """
        raise NotImplementedError

    @property
    def has_forces(self) -> bool:
        """Whether the dataset provides forces - drives the loss composition."""
        raise NotImplementedError

    @property
    def unit_conversion(self) -> Callable:
        """Display-only conversion (eV -> meV etc.). Never used for training."""
        raise NotImplementedError

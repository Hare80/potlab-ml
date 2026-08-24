"""Data layer contracts: Standardizer and BaseDataModule.

These two classes are the only entry points the trainer knows about. The
batch contract (z / pos / y / batch / forces) is specified in DESIGN.md §3;
everything else about a dataset is private to its concrete implementation.
"""

from typing import Callable, Optional

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

        The per-molecule aggregation of ``inverse_per_atom``: broadcast
        each molecule's pooled value to its atoms, apply the per-atom
        transform, sum per molecule. One implementation, two granularities.
        """
        raise NotImplementedError

    def inverse_per_atom(self, contribs: Tensor, z: Tensor) -> Tensor:
        """Per-atom standardized contributions -> per-atom physical energies.

        The granularity the export path needs: LAMMPS' mliap plugin works
        per atom and has no molecule boundaries (no graph_indexes), so the
        M5 wrapper calls this method directly - LAMMPS gets per-atom
        absolute energies, and their sum is the total energy.
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
    def num_targets(self) -> int:
        """How many target columns the dataset yields (y is [N_graphs, num_targets]).

        Every model output column is a target column: the assembly checks
        model.num_outputs == num_targets and refuses the rest - an extra
        output column would have no training signal, a missing one would
        drop targets. QM9 is single-target; a future multi-target dataset
        overrides this with len(targets).
        """
        raise NotImplementedError

    @property
    def energy_index(self) -> Optional[int]:
        """Which output column is the energy, or None if the target is not one.

        Energy means precisely: a conservative potential-energy-surface
        quantity whose negative position gradient IS the interatomic
        force. The dataset knows its target's semantics (QM9's U0 is a
        PES energy; its HOMO, ZPVE or dipole moment are not - even when
        measured in eV), so the knowledge lives here, next to has_forces
        and unit_conversion. The model never decides it.

        ``has_forces`` implies ``energy_index is not None`` (a dataset
        with forces has an energy); the reverse does not hold (QM9 is
        energies without force data). Consumers: energy_and_forces
        differentiates column 0; the LAMMPS export refuses runs whose
        energy_index is None.
        """
        return None

    @property
    def unit_conversion(self) -> Callable:
        """Display-only conversion (eV -> meV etc.). Never used for training."""
        raise NotImplementedError

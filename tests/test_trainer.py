"""Trainer contract tests (M3): algorithm selection, resume semantics, fit smoke.

Smoke philosophy: these tests pin MECHANISMS, never numbers - which class
the config selects, where T_max comes from, what resume restores. No data
download, no GPU, no touching the real runs/ directory (every test uses
tmp_path). scripts/train.py itself is not tested here: it needs the old
project and a QM9 download, so it stays in the acceptance checklist.
"""

from pathlib import Path
import csv

import pytest
import torch
import torch.nn as nn

from potlab.config import Config
from potlab.data.base import BaseDataModule, Standardizer
from potlab.training import make_run_dir
from potlab.training.metrics import METRICS_HEADER
from potlab.training.trainer import Trainer


class MiniBatch:
    """The DESIGN.md §3 batch contract in miniature: z/pos/batch/y + .to().

    A plain container, not a torch_geometric object - this also documents
    exactly which attributes the trainer touches (nothing else).
    """

    def __init__(self, z, pos, graph_indexes, y):
        self.z = z
        self.pos = pos
        self.batch = graph_indexes
        self.y = y

    def to(self, device):
        self.z = self.z.to(device)
        self.pos = self.pos.to(device)
        self.batch = self.batch.to(device)
        self.y = self.y.to(device)
        return self


class ToyLoader:
    """A minimal iterable of batches: the trainer needs len() and iteration only.

    A real DataLoader would add batching/shuffling/workers - none of which
    is under test here, so none of it is faked.
    """

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _make_batches():
    """Three static toy batches: a few atoms, a few molecules each.

    y is [N_graphs, 1] - one label per MOLECULE (len(batch.y) is the
    molecule count the trainer divides by).
    """
    return [
        MiniBatch(
            z=torch.tensor([1, 6, 8]),
            pos=torch.tensor([[0.0, 0, 0], [1.0, 1, 1], [2.0, 2, 2]]),
            graph_indexes=torch.tensor([0, 0, 1]),
            y=torch.tensor([[0.5], [1.5]]),
        ),
        MiniBatch(
            z=torch.tensor([1, 1, 6, 6]),
            pos=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [1.0, 1, 1]]),
            graph_indexes=torch.tensor([0, 0, 0, 1]),
            y=torch.tensor([[0.75], [1.25]]),
        ),
        MiniBatch(
            z=torch.tensor([6, 8]),
            pos=torch.tensor([[0.0, 0, 0], [2.0, 2, 2]]),
            graph_indexes=torch.tensor([0, 1]),
            y=torch.tensor([[1.0], [2.0]]),
        ),
    ]


class ToyModel(nn.Module):
    """The energy() protocol in miniature: per-atom features -> per-molecule sum.

    Mirrors the real pipeline's shape (embedding + coordinate features +
    readout + index_add aggregation) with a handful of parameters, so the
    trainer's optimizer/checkpoint machinery has real state to work with.
    """

    def __init__(self, num_features: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(100, num_features)  # z -> scalar features
        self.pos_linear = nn.Linear(3, num_features)      # coordinates -> features
        self.out_linear = nn.Linear(2 * num_features, 1)  # features -> contribution

    def energy(self, z, pos, graph_indexes):
        z_features = self.embedding(z)
        pos_features = self.pos_linear(pos)
        per_atom = self.out_linear(torch.cat([z_features, pos_features], dim=1))
        n_graphs = int(graph_indexes.max().item()) + 1
        out = torch.zeros(
            n_graphs, 1, device=per_atom.device, dtype=per_atom.dtype
        )
        out.index_add_(dim=0, index=graph_indexes, source=per_atom)
        return out


class ToyDataModule(BaseDataModule):
    """Toy data module: only what the trainer calls is implemented.

    BaseDataModule is a loose protocol (not ABC), so prepare_data/setup/
    make_standardizer/has_forces stay untouched - the trainer never calls
    them.
    """

    def __init__(self):
        batches = _make_batches()  # train = 2 batches, val = 1 batch
        self._train = ToyLoader(batches[:2])
        self._val = ToyLoader(batches[2:])

    def train_dataloader(self):
        return self._train

    def val_dataloader(self):
        return self._val

    @property
    def unit_conversion(self):
        return lambda t: t  # display-only; the toy has no units


class ToyStandardizer(Standardizer):
    """Toy standardizer: only the two checkpoint methods are implemented.

    The M3 trainer's loss lives in physical space, so fit/transform/inverse
    are never called - but state_dict/load_state_dict go in and out of
    every checkpoint.
    """

    def __init__(self, mean: float = 0.0):
        self.mean = torch.tensor(mean)

    def state_dict(self):
        return {"mean": self.mean}

    def load_state_dict(self, state):
        self.mean = state["mean"]


def _make_config(**training_overrides):
    """A Config built directly (never configs/default.yaml - tests are standalone)."""
    training = {
        "num_epochs": 2,
        # lr / weight_decay differ from AdamW's defaults on purpose: a test
        # that passes proves the config values are forwarded, not defaults.
        "optimizer": {"name": "AdamW", "lr": 0.01, "weight_decay": 0.25},
        "scheduler": {"name": "CosineAnnealingLR"},
        "early_stopping": {"patience": 30, "min_epochs": 1000},
    }
    training.update(training_overrides)
    config = Config()
    config.training = training
    return config


def _make_trainer(tmp_path, config=None, resume=False):
    """Fresh trainer over tmp_path: every test gets its own run dir."""
    return Trainer(
        model=ToyModel(),
        data_module=ToyDataModule(),
        standardizer=ToyStandardizer(),
        run_dir=make_run_dir("test_run", root=tmp_path),
        config=config or _make_config(),
        resume=resume,
    )


# --- A: algorithm selection (construction only, no fit) ---

def test_optimizer_selected_by_name_and_kwargs_forwarded(tmp_path):
    trainer = _make_trainer(tmp_path)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.01
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.25


def test_scheduler_tmax_derived_from_total_steps(tmp_path):
    trainer = _make_trainer(tmp_path)
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    # T_max = num_epochs * steps_per_epoch, computed inside the trainer
    # (steps_per_epoch = len of the toy train loader = 2).
    assert trainer.scheduler.T_max == 2 * len(trainer.train_loader)


def test_scheduler_explicit_tmax_respected(tmp_path):
    trainer = _make_trainer(
        tmp_path, _make_config(scheduler={"name": "CosineAnnealingLR", "T_max": 7})
    )
    assert trainer.scheduler.T_max == 7


def test_unknown_optimizer_name_raises(tmp_path):
    with pytest.raises(ValueError):
        _make_trainer(
            tmp_path, _make_config(optimizer={"name": "not_a_real_optimizer"})
        )


def test_unknown_scheduler_name_raises(tmp_path):
    with pytest.raises(ValueError):
        _make_trainer(
            tmp_path, _make_config(scheduler={"name": "not_a_real_scheduler"})
        )


# --- B: resume semantics ---

def test_resume_uses_checkpoint_algorithms_not_config(tmp_path):
    run_dir = make_run_dir("test_run", root=tmp_path)
    # First run: train one epoch with SGD + StepLR - the checkpoint carries
    # that config snapshot inside itself.
    first = Trainer(
        model=ToyModel(),
        data_module=ToyDataModule(),
        standardizer=ToyStandardizer(),
        run_dir=run_dir,
        config=_make_config(
            num_epochs=1,
            optimizer={"name": "SGD", "lr": 0.01},
            scheduler={"name": "StepLR", "step_size": 10},
        ),
    )
    first.fit()

    # Resume with a DIFFERENT config: the checkpoint is authoritative for
    # the run's algorithms - a config change is ignored, not applied.
    resumed = Trainer(
        model=ToyModel(),
        data_module=ToyDataModule(),
        standardizer=ToyStandardizer(),
        run_dir=run_dir,
        config=_make_config(num_epochs=1),
        resume=True,
    )
    assert isinstance(resumed.optimizer, torch.optim.SGD)
    assert isinstance(resumed.scheduler, torch.optim.lr_scheduler.StepLR)
    assert resumed.start_epoch == 1  # checkpoint epoch 0 -> continue at 1


def test_resume_without_checkpoint_raises(tmp_path):
    # Explicit resume with no checkpoint must fail at CONSTRUCTION time
    # (before any training), never silently start over.
    with pytest.raises(FileNotFoundError):
        _make_trainer(tmp_path, resume=True)


def test_resume_restores_model_weights_and_best_mae(tmp_path):
    run_dir = make_run_dir("test_run", root=tmp_path)
    first = Trainer(
        model=ToyModel(),
        data_module=ToyDataModule(),
        standardizer=ToyStandardizer(),
        run_dir=run_dir,
        config=_make_config(num_epochs=1),
    )
    first.fit()
    saved_params = {k: v.clone() for k, v in first.model.state_dict().items()}
    saved_best = first.best_val_mae

    # A freshly initialized model would have different weights: matching
    # against the saved ones proves the checkpoint was actually loaded.
    resumed = _make_trainer(tmp_path, _make_config(num_epochs=1), resume=True)
    for key, value in resumed.model.state_dict().items():
        assert torch.allclose(value, saved_params[key])
    assert resumed.best_val_mae == saved_best


def test_resume_appends_metrics_without_duplicating_header(tmp_path):
    run_dir = make_run_dir("test_run", root=tmp_path)
    first = Trainer(
        model=ToyModel(),
        data_module=ToyDataModule(),
        standardizer=ToyStandardizer(),
        run_dir=run_dir,
        config=_make_config(num_epochs=1),
    )
    first.fit()

    # Second run continues from epoch 1 (num_epochs=2); the logger appends
    # to the same CSV and must not write a second header row.
    resumed = Trainer(
        model=ToyModel(),
        data_module=ToyDataModule(),
        standardizer=ToyStandardizer(),
        run_dir=run_dir,
        config=_make_config(num_epochs=2),
        resume=True,
    )
    resumed.fit()

    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == list(METRICS_HEADER)
    assert len(rows) == 3  # header + epoch 0 + epoch 1


# --- C: minimal fit smoke (one epoch, end to end) ---

def test_fit_writes_complete_run_layout(tmp_path):
    run_dir = make_run_dir("test_run", root=tmp_path)
    trainer = _make_trainer(tmp_path, _make_config(num_epochs=1))
    trainer.fit()

    # The five-piece run layout; best.pt exists because the first epoch
    # always beats the initial best of +inf.
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "lr_steps.csv").is_file()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "checkpoints" / "latest.pt").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "plots" / "latest.png").is_file()

    # Exactly one data row: the header is written once and the epoch count
    # is the molecule-count-division contract in action.
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == list(METRICS_HEADER)
    assert len(rows) == 2  # header + epoch 0

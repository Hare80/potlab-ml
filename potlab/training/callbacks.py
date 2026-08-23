"""Training callbacks: early stopping, TensorBoard, matplotlib panel.

All callbacks implement one contract: on_epoch_end(epoch, metrics, trainer).
The trainer's loop is a thin shell that builds the metrics dict and calls
every callback - it never grows feature flags.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: must precede pyplot import
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from potlab.training.metrics import METRICS_CSV


class Callback:
    """Contract every callback implements (DESIGN.md §7)."""

    def on_epoch_end(self, epoch: int, metrics: dict, trainer) -> None:
        raise NotImplementedError


class EarlyStoppingCallback(Callback):
    """Stop training when val MAE stops improving.

    Decision only: sets ``trainer.stop = True``. Saving the best model is
    the trainer's job (checkpointing), never the callback's - the original
    EarlyStopping mixed those two roles.
    """

    def __init__(self, patience: int = 30, min_epochs: int = 1000) -> None:
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_val_mae = float("inf")
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, metrics: dict, trainer) -> None:
        val_mae = metrics["val_mae"]
        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
            self.best_epoch = epoch

        # No special case for min_epochs=1000: inside num_epochs=1000 the
        # guard below can never fire, so stopping is simply inert.
        if epoch >= self.min_epochs and epoch - self.best_epoch >= self.patience:
            trainer.stop = True


class TensorBoardCallback(Callback):
    """Live scalar view of the run (the CSV stays the source of truth)."""

    def __init__(self, run_dir: Path) -> None:
        self.writer = SummaryWriter(log_dir=run_dir)

    def on_epoch_end(self, epoch: int, metrics: dict, trainer) -> None:
        self.writer.add_scalar("train/loss", metrics["train_loss"], epoch)
        self.writer.add_scalar("val/mae", metrics["val_mae"], epoch)
        self.writer.add_scalar("train/lr", metrics["lr"], epoch)

    def close(self) -> None:
        self.writer.close()


class PlotCallback(Callback):
    """Redraw a 2x2 panel from metrics.csv every N epochs.

    Reads the CSV, not in-memory state: the plot is only correct if the
    CSV is complete, so this doubles as a standing check of the
    "CSV = source of truth" rule. The lr panel is a wiring self-check -
    a wrong T_max shows up as a malformed cosine curve.
    """

    def __init__(self, run_dir: Path, every_n_epochs: int = 10) -> None:
        self.run_dir = run_dir
        self.every_n_epochs = every_n_epochs

    def _read_metrics(self):
        """Load all logged epochs from metrics.csv (the only data source)."""
        epochs, train_loss, val_mae, lr, grad_norm = [], [], [], [], []
        with open(self.run_dir / METRICS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_mae.append(float(row["val_mae"]))
                lr.append(float(row["lr"]))
                grad_norm.append(float(row["grad_norm"]))
        return epochs, train_loss, val_mae, lr, grad_norm

    def on_epoch_end(self, epoch: int, metrics: dict, trainer) -> None:
        if epoch % self.every_n_epochs != 0:
            return

        epochs, train_loss, val_mae, lr, grad_norm = self._read_metrics()
        if not epochs:  # defensive: nothing logged yet
            return

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"Run: {self.run_dir.name}")

        # Train loss: log scale - it spans orders of magnitude, a linear
        # axis hides the early drop entirely.
        axes[0, 0].plot(epochs, train_loss)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Train loss (log y)")
        axes[0, 0].set_xlabel("epoch")

        # Val MAE with a vertical line at the best epoch (picking the
        # early-stopping patience is a matter of reading this curve).
        best_idx = min(range(len(val_mae)), key=lambda i: val_mae[i])
        axes[0, 1].plot(epochs, val_mae)
        axes[0, 1].axvline(
            epochs[best_idx], color="gray", linestyle="--",
            label=f"best: {val_mae[best_idx]:.1f}",
        )
        axes[0, 1].legend()
        axes[0, 1].set_title("Val MAE (meV)")
        axes[0, 1].set_xlabel("epoch")

        axes[1, 0].plot(epochs, lr)
        axes[1, 0].set_title("LR (epoch-end snapshot)")
        axes[1, 0].set_xlabel("epoch")

        axes[1, 1].plot(epochs, grad_norm)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Grad norm (log y)")
        axes[1, 1].set_xlabel("epoch")

        fig.tight_layout()
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(self.run_dir / "plots" / "latest.png")
        plt.close(fig)  # matplotlib keeps figures alive otherwise

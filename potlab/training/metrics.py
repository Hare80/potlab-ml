"""Metrics logging: the CSVs are the single source of truth.

Every epoch appends one row to metrics.csv and every K-th training step
appends to lr_steps.csv. Figures are always re-plottable from these files
and data survives a killed process (every write is flushed); TensorBoard
events are only a view on top, never the record.
"""

import csv
from pathlib import Path

# File names are part of the contract too: writers and readers (the
# PlotCallback) share these constants instead of duplicating the strings.
METRICS_CSV = "metrics.csv"
LR_STEPS_CSV = "lr_steps.csv"
METRICS_HEADER = ("epoch", "train_loss", "val_mae", "lr", "epoch_time", "grad_norm")
LR_STEPS_HEADER = ("step", "lr")


class MetricsLogger:
    """Append per-epoch metrics and per-step lr values to CSV files.

    ``run_dir`` must already exist (the trainer creates it via
    make_run_dir). File handles stay open for the logger's lifetime, so no
    ``with`` here - a with-block would close them immediately.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

        # The header goes in only when the file starts empty: on --resume
        # the logger appends to the same CSV and must not duplicate it.
        metrics_path = run_dir / METRICS_CSV
        fresh = not (metrics_path.exists() and metrics_path.stat().st_size > 0)
        self.metrics_file = open(metrics_path, "a", newline="")
        self.metrics_writer = csv.writer(self.metrics_file)
        if fresh:
            self.metrics_writer.writerow(METRICS_HEADER)
            self.metrics_file.flush()

        lr_steps_path = run_dir / LR_STEPS_CSV
        fresh = not (lr_steps_path.exists() and lr_steps_path.stat().st_size > 0)
        self.lr_steps_file = open(lr_steps_path, "a", newline="")
        self.lr_steps_writer = csv.writer(self.lr_steps_file)
        if fresh:
            self.lr_steps_writer.writerow(LR_STEPS_HEADER)
            self.lr_steps_file.flush()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_mae: float,
        lr: float,
        epoch_time: float,
        grad_norm: float,
    ) -> None:
        """Append one row to metrics.csv.

        flush() makes the row durable immediately: a killed training
        process must not lose the last rows (the M3 acceptance scenario).
        """
        self.metrics_writer.writerow(
            [epoch, train_loss, val_mae, lr, epoch_time, grad_norm]
        )
        self.metrics_file.flush()

    def log_lr_step(self, step: int, lr: float) -> None:
        """Append one row to lr_steps.csv (cosine changes the lr every step)."""
        self.lr_steps_writer.writerow([step, lr])
        self.lr_steps_file.flush()

    def close(self) -> None:
        """Close both files (called on clean exit; harmless to skip)."""
        self.metrics_file.close()
        self.lr_steps_file.close()

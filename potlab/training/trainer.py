"""The training loop (DESIGN.md §7).

A thin shell: run epochs over the train loader, build the metrics dict and
call every callback at epoch end. The loop never grows feature flags -
anything extra is a callback's job. The model is used only through the
``energy(z, pos, graph_indexes)`` protocol, so the registry-backed
PaiNNModel replaced the legacy adapter without touching this file.

Loss space (M2 step 5): the model outputs per-molecule MEAN contributions,
i.e. standardized-space predictions. The loss compares them to
``standardizer.transform(y)`` - ~N(0,1) targets are numerically friendly,
and mean pooling removes the per-molecule size weighting that physical-
space MSE carried (the (n_atoms * std)^2 factor per molecule). Val MAE
converts predictions back through ``standardizer.inverse`` and stays in
physical units - the baseline number (~5.4 meV) lives in that space.
"""

import dataclasses
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from tqdm import trange

from potlab.config import Config
from potlab.data.base import BaseDataModule, Standardizer
from potlab.training.callbacks import (
    EarlyStoppingCallback,
    PlotCallback,
    TensorBoardCallback,
)
from potlab.training.metrics import MetricsLogger

CHECKPOINT_DIR = "checkpoints"
LATEST_CHECKPOINT = "latest.pt"
BEST_CHECKPOINT = "best.pt"
LR_LOG_EVERY_STEPS = 10


class Trainer:
    """Train a model under the energy() protocol, with logging and checkpoints.

    The device is decided at assembly time (train.py moves the model before
    handing it over); the trainer only follows where the model lives.
    """

    def __init__(
        self,
        model,
        data_module: BaseDataModule,
        standardizer: Standardizer,
        run_dir: Path,
        config: Config,
        resume: bool = False,
    ) -> None:
        self.model = model
        self.data_module = data_module
        # The loss targets live in standardized space (transform), and val
        # MAE converts predictions back (inverse); the state_dict also goes
        # into every checkpoint (resume must restore it).
        self.standardizer = standardizer
        self.run_dir = run_dir
        self.config = config
        self.stop = False
        self.device = next(model.parameters()).device

        # DataLoaders are built once and reused - rebuilding one per epoch
        # respawns worker pools for nothing.
        self.train_loader = data_module.train_dataloader()
        self.val_loader = data_module.val_dataloader()

        # Built from the config now; on resume, _load_checkpoint may rebuild
        # them from the CHECKPOINT's config instead - the checkpoint is the
        # source of truth for the run's algorithms (its state_dicts are
        # then loaded over whatever stands here).
        self._build_optimizer_scheduler(config.training)

        self.logger = MetricsLogger(run_dir)

        self.early_stopping = EarlyStoppingCallback(
            patience=config.training["early_stopping"]["patience"],
            min_epochs=config.training["early_stopping"]["min_epochs"],
        )
        self.callbacks = [
            self.early_stopping,
            TensorBoardCallback(run_dir),
            PlotCallback(run_dir),
        ]

        # The checkpoint's `epoch` is the LAST COMPLETED epoch, so training
        # continues at epoch + 1 (the classic off-by-one of resume logic).
        self.best_val_mae = float("inf")
        self.best_epoch = -1
        if resume:
            latest = run_dir / CHECKPOINT_DIR / LATEST_CHECKPOINT
            if not latest.is_file():
                # Explicit resume with no checkpoint must fail loudly:
                # silently starting over would pollute an existing run.
                raise FileNotFoundError(f"Checkpoint not found: {latest}")
            self.start_epoch = self._load_checkpoint(latest)
        else:
            # The reproducibility snapshot belongs to the run's initial
            # state: written only when the run starts fresh, never on
            # resume.
            self.start_epoch = 0
            self._dump_config_snapshot()

    def fit(self) -> None:
        """Run the training loop until num_epochs or early stopping."""
        num_epochs = self.config.training["num_epochs"]
        steps_per_epoch = len(self.train_loader)

        pbar = trange(self.start_epoch, num_epochs)
        for epoch in pbar:
            t0 = time.time()

            self.model.train()
            loss_epoch_sum = 0.0
            n_mols_epoch = 0
            for batch_idx, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)

                preds = self.model.energy(batch.z, batch.pos, batch.batch)
                # Standardized space (M2 step 5): the model mean-pools, so
                # the loss target is transform(y), not the raw label.
                standardized_y = self.standardizer.transform(
                    batch.y, batch.z, batch.batch
                )
                # Defense in depth behind train.py's assembly check: any
                # other assembly path (tests, future scripts) still gets a
                # self-diagnosing error instead of a torch broadcast
                # failure deep in the loss.
                if preds.shape != standardized_y.shape:
                    raise ValueError(
                        f"Model output shape {tuple(preds.shape)} does not "
                        f"match target shape {tuple(standardized_y.shape)}: "
                        "model.num_outputs must equal the dataset's "
                        "num_targets."
                    )
                # sum-then-divide: accumulate reduction='sum' losses, divide
                # once by the total molecule count (batches differ in size).
                loss_sum = F.mse_loss(preds, standardized_y, reduction="sum")
                loss_mean = loss_sum / len(batch.y)

                self.optimizer.zero_grad(set_to_none=True)  # frees grad memory; _grad_norm already skips None grads
                loss_mean.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_epoch_sum += loss_sum.detach().item()
                n_mols_epoch += len(batch.y)

                # lr changes every step under cosine: log on a global step
                # counter so a resumed run continues the same numbering.
                global_step = epoch * steps_per_epoch + batch_idx
                if global_step % LR_LOG_EVERY_STEPS == 0:
                    self.logger.log_lr_step(global_step, self._current_lr())

            loss_epoch_mean = loss_epoch_sum / n_mols_epoch
            val_mae = self.compute_mae(self.val_loader)  # physical units
            val_mae_display = self.data_module.unit_conversion(val_mae)

            metrics = {
                "epoch": epoch,
                "train_loss": loss_epoch_mean,
                "val_mae": val_mae_display,  # display units (meV for QM9)
                "lr": self._current_lr(),
                "epoch_time": time.time() - t0,
                "grad_norm": self._grad_norm(),
            }

            self.logger.log_epoch(**metrics)
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics, self)

            # Update the best FIRST, then save both checkpoints BEFORE
            # checking stop. latest.pt must carry this epoch's evaluation:
            # saving it before the update left a stale best_val_mae in it,
            # so a resume restored inf (an off-by-one the test suite
            # caught). A stopped epoch must still be in the history too -
            # both saves happen before the stop check.
            if val_mae_display < self.best_val_mae:
                self.best_val_mae = val_mae_display
                self.best_epoch = epoch
                self._save_checkpoint(BEST_CHECKPOINT, epoch)
            self._save_checkpoint(LATEST_CHECKPOINT, epoch)

            pbar.set_postfix_str(
                f"Train loss: {loss_epoch_mean:.3e}, "
                f"Val. MAE: {val_mae_display:.3f}, "
                f"LR: {self._current_lr():.2e}"
            )

            if self.stop:
                print(f"Early stopping after epoch {epoch}.")
                break

        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val. MAE: {self.best_val_mae:.3f}")
        self._close()

    def compute_mae(self, loader: DataLoader) -> float:
        """Sum-then-divide MAE in PHYSICAL units (display conversion is the caller's job).

        One definition for every MAE this framework reports: the fit loop
        calls it on the val loader each epoch, and the assembly's final
        report (train.py) calls it on the test loader after loading
        best.pt. Predictions leave the model in standardized space;
        inverse() maps them back so the MAE compares physical energies -
        the baseline number lives in this space.
        """
        self.model.eval()
        mae_sum = 0.0
        n_mols = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self.model.energy(batch.z, batch.pos, batch.batch)
                preds = self.standardizer.inverse(preds, batch.z, batch.batch)
                mae_sum += F.l1_loss(preds, batch.y, reduction="sum").item()
                n_mols += len(batch.y)
        return mae_sum / n_mols

    def _grad_norm(self) -> float:
        """Global gradient norm after the last backward (explosion diagnostics)."""
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "standardizer": self.standardizer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,  # the last COMPLETED epoch
            "best_val_mae": self.best_val_mae,
            "config": dataclasses.asdict(self.config),
        }
        # Atomic write: save to a temp file, then rename over the real one.
        # A kill mid-save leaves the previous checkpoint intact - at worst
        # a stale .tmp file - instead of a truncated .pt.
        path = self.run_dir / CHECKPOINT_DIR / filename
        tmp = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)

    def _build_optimizer_scheduler(self, training: dict) -> None:
        """Build optimizer + scheduler from a ``training`` section.

        Each section carries a ``name`` looked up in torch.optim /
        torch.optim.lr_scheduler; every other key is forwarded to the class,
        so each algorithm configures its own arguments (SGD carries
        momentum, cosine carries eta_min). Called from __init__ with the
        current config and from _load_checkpoint with the checkpoint's.
        """
        opt_cfg = dict(training["optimizer"])
        self.optimizer_name = opt_cfg.pop("name")
        optimizer_cls = getattr(torch.optim, self.optimizer_name, None)
        if optimizer_cls is None:
            # The name must match torch.optim exactly (case included): list
            # what IS available so a typo like 'Adamw' is self-diagnosing.
            available = sorted(
                n for n in dir(torch.optim)
                if isinstance(getattr(torch.optim, n), type) and not n.startswith("_")
            )
            raise ValueError(
                f"Unknown optimizer {self.optimizer_name!r}. "
                f"Name must match torch.optim exactly; available: {available}"
            )
        self.optimizer = optimizer_cls(self.model.parameters(), **opt_cfg)

        sched_cfg = dict(training["scheduler"])
        self.scheduler_name = sched_cfg.pop("name")
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.scheduler_name, None)
        if scheduler_cls is None:
            available = sorted(
                n for n in dir(torch.optim.lr_scheduler)
                if isinstance(getattr(torch.optim.lr_scheduler, n), type)
                and not n.startswith("_")
            )
            raise ValueError(
                f"Unknown scheduler {self.scheduler_name!r}. "
                "Name must match torch.optim.lr_scheduler exactly; "
                f"available: {available}"
            )
        # T_max is derived, not configured: the cosine covers every training
        # step, and steps_per_epoch changes with --subset-size. Only cosine
        # receives it; a config may still override it explicitly.
        if self.scheduler_name == "CosineAnnealingLR":
            sched_cfg.setdefault(
                "T_max", self.config.training["num_epochs"] * len(self.train_loader)
            )
        self.scheduler = scheduler_cls(self.optimizer, **sched_cfg)

    def _load_checkpoint(self, path: Path) -> int:
        """Restore the checkpoint state; returns the epoch to CONTINUE from."""
        checkpoint = torch.load(path, map_location=self.device)
        # The checkpoint decides the run's algorithms, not the config: a
        # resume continues the SAME run, and swapping its optimizer is a
        # new run (train.py's --warm-start), never a resume. num_epochs,
        # however, stays the current config's - extending a run is legal.
        ckpt_training = checkpoint["config"]["training"]
        ckpt_opt = ckpt_training["optimizer"]["name"]
        ckpt_sched = ckpt_training["scheduler"]["name"]
        if ckpt_opt != self.optimizer_name or ckpt_sched != self.scheduler_name:
            print(
                f"Resume: using the checkpoint's {ckpt_opt}/{ckpt_sched} "
                f"(the config requests {self.optimizer_name}/{self.scheduler_name}, "
                "which is ignored - changing algorithms is a new run, not a resume)."
            )
            self._build_optimizer_scheduler(ckpt_training)
        self.model.load_state_dict(checkpoint["model"])
        self.standardizer.load_state_dict(checkpoint["standardizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_val_mae = checkpoint["best_val_mae"]
        return checkpoint["epoch"] + 1

    def _dump_config_snapshot(self) -> None:
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(dataclasses.asdict(self.config), f)

    def _close(self) -> None:
        self.logger.close()
        for callback in self.callbacks:
            close = getattr(callback, "close", None)
            if close is not None:
                close()

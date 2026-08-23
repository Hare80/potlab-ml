"""M3 assembly: config -> data (registry) -> legacy model -> Trainer.

The old-project PaiNN + AtomwisePostProcessing are wrapped in a small
adapter that speaks the Trainer's ``energy(z, pos, graph_indexes)``
protocol. The adapter is temporary glue - M2 registers the real
PaiNNModel in the registry and this file drops the old-project import.

Two ways to reuse previous work (mutually exclusive by semantics):

- ``--resume``: continue the SAME run. The checkpoint is authoritative
  for the run's algorithms; the config may only extend num_epochs.
- ``--warm-start <checkpoint>``: start a NEW run (new run_dir, fresh
  optimizer/scheduler from the current config) but initialize the model
  weights from an old checkpoint. This is the sanctioned way to change
  the optimizer while keeping the model's learned knowledge.
"""

import argparse
import sys
from pathlib import Path

from lightning_fabric import seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F

# The old project is a sibling of the repo root: __file__ -> scripts/ ->
# repo root -> Codes/ (three parents up). Anchoring to __file__ makes this
# work regardless of the current working directory.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "02456_painn_project-main"))

from potlab import ROOT
import potlab.config as config
import potlab.data.qm9  # side effect: registers "qm9" in DATASETS
import potlab.registry as registry
import potlab.training as training
from potlab.training.trainer import Trainer
from src.models import PaiNN, AtomwisePostProcessing  # type: ignore


class LegacyPaiNNAdapter(nn.Module):
    """Old PaiNN + post-processing under the Trainer's energy() protocol.

    Only two members need writing: ``energy`` (the protocol) and
    ``parameters`` (the optimizer must see painn's parameters only - the
    post-processing's frozen atom_refs embedding is technically an
    nn.Parameter, and M1 trained with AdamW(painn.parameters())). .to() /
    .train() / .eval() / state_dict round-trips are inherited from
    nn.Module, which recurses children, not this generator.
    """

    def __init__(self, painn, post_processing):
        super().__init__()
        self.painn = painn
        self.post_processing = post_processing

    def energy(self, z, pos, graph_indexes):
        """Molecule energies in PHYSICAL units, [N_graphs, num_outputs]."""
        atomic_contributions = self.painn(
            atoms=z, atom_positions=pos, graph_indexes=graph_indexes
        )
        return self.post_processing(
            atomic_contributions=atomic_contributions,
            atoms=z,
            graph_indexes=graph_indexes,
        )

    def parameters(self, recurse: bool = True):
        # Excludes post_processing's frozen atom_refs Embedding weight from
        # the optimizer (baseline parity with M1). .to() and state_dict()
        # still cover it: they recurse children, not this generator.
        return self.painn.parameters(recurse=recurse)


def build_parser():
    """Build the argument parser (separate from main so tests can reuse it)."""
    parser = argparse.ArgumentParser(description="Train a model.")
    # Default anchored to ROOT so the script works from any CWD; a
    # user-supplied --config path is used as given (relative to CWD).
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"),
                        help="Path to the config file.")
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Override config values using dotted paths, e.g., 'training.optimizer.lr=1e-3'.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the run named by run_name from checkpoints/latest.pt.")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Start a NEW run but initialize model weights from this checkpoint.")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Debug: cap the dataset to N molecules.")
    return parser


def test_mae(model, dataloader, device):
    """Sum-then-divide MAE in PHYSICAL units (display conversion is the caller's job)."""
    model.eval()
    mae_sum = 0.0
    n_mols = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            preds = model.energy(batch.z, batch.pos, batch.batch)
            mae_sum += F.l1_loss(preds, batch.y, reduction="sum").item()
            n_mols += len(batch.y)
    return mae_sum / n_mols


def main():
    """Parse CLI args, assemble the run from config, train, evaluate on test."""
    parser = build_parser()
    args = parser.parse_args()
    config_data = config.load_config(args.config, args.override)
    run_dir = training.make_run_dir(config_data.run_name)
    print(f"Run directory: {run_dir}")
    print(f"Configuration: {config_data}")

    seed_everything(config_data.seed)  # fixed seed -> reproducible run
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data module via the registry: the config's data.name picks the class
    # (importing potlab.data.qm9 above performed the registration).
    data_cfg = dict(config_data.data)
    data_name = data_cfg.pop("name")
    dm = registry.DATASETS[data_name](
        **data_cfg,
        seed=config_data.seed,       # top-level Config field, not in data:
        subset_size=args.subset_size,
    )
    dm.prepare_data()  # downloads QM9 on first run, skips afterwards
    dm.setup()  # loads, shuffles, splits

    standardizer = dm.make_standardizer()  # fitted on the train split only

    # Model: the legacy old-project PaiNN (models do NOT go through the
    # registry yet - MODELS stays empty until M2 registers PaiNNModel).
    model_cfg = dict(config_data.model)
    model_cfg.pop("name")
    painn = PaiNN(**model_cfg)
    post_processing = AtomwisePostProcessing(
        model_cfg["num_outputs"],
        standardizer.mean, standardizer.std, standardizer.atom_refs,
    )
    model = LegacyPaiNNAdapter(painn, post_processing).to(device)

    if args.warm_start is not None:
        # Warm start: inherit the model's knowledge (weights + standardizer
        # statistics) from an old checkpoint, but start a NEW run - fresh
        # optimizer/scheduler from the current config, epoch 0, new run_dir.
        # The checkpoint's own config is deliberately ignored here.
        ckpt_path = Path(args.warm_start)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        standardizer.load_state_dict(checkpoint["standardizer"])
        print(
            f"Warm-started model weights from {ckpt_path} "
            f"(source epoch {checkpoint['epoch']}, "
            f"best val MAE {checkpoint['best_val_mae']:.3f})"
        )

    # The Trainer speaks only the energy() protocol: data, logging,
    # optimizer/scheduler selection, checkpoints and resume are its job.
    Trainer(
        model=model,
        data_module=dm,
        standardizer=standardizer,
        run_dir=run_dir,
        config=config_data,
        resume=args.resume,
    ).fit()

    # Final test evaluation with the best checkpoint (baseline gate: MAE ~= 5.4 meV).
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    mae = test_mae(model, dm.test_dataloader(), device)
    print(f"Test MAE: {dm.unit_conversion(mae):.3f}")

    return 0


if __name__ == "__main__":
    # True only when executed directly (python scripts/train.py), not when
    # imported. SystemExit carries main's return code out as the exit code.
    raise SystemExit(main())

"""Assembly: config -> data (registry) -> model (registry) -> Trainer.

M2 step 5 retired the old-project adapter: the model comes from
MODELS["painn"] (registered by importing potlab.models.painn.model), and
its mean-pooled output is the standardized-space prediction - the
standardizer pairs with it unchanged (trainer loss uses transform; the
final Test MAE is Trainer.compute_mae, which converts back via inverse).

Two ways to reuse previous work (mutually exclusive by semantics):

- ``--resume``: continue the SAME run. The checkpoint is authoritative
  for the run's algorithms; the config may only extend num_epochs.
- ``--warm-start <checkpoint>``: start a NEW run (new run_dir, fresh
  optimizer/scheduler from the current config) but initialize the model
  weights from an old checkpoint. This is the sanctioned way to change
  the optimizer while keeping the model's learned knowledge.
"""

import argparse
from pathlib import Path

from lightning_fabric import seed_everything
import torch

from potlab import ROOT
import potlab.config as config
import potlab.data.qm9  # side effect: registers "qm9" in DATASETS
import potlab.models.painn.model  # side effect: registers "painn" in MODELS
import potlab.registry as registry
import potlab.training as training
from potlab.training.trainer import Trainer


def build_parser():
    """Build the argument parser (separate from main so tests can reuse it)."""
    parser = argparse.ArgumentParser(description="Train a model.")
    # Default anchored to ROOT so the script works from any CWD; a
    # user-supplied --config path is used as given (relative to CWD).
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"),
                        help="Path to the config file.")
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Override config values using dotted paths, e.g., 'training.optimizer.lr=1.0e-3'.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the run named by run_name from checkpoints/latest.pt.")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Start a NEW run but initialize model weights from this checkpoint.")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Debug: cap the dataset to N molecules.")
    return parser


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

    # Model via the registry (importing potlab.models.painn.model above
    # performed the registration). The model mean-pools per molecule, so
    # its output IS the standardized prediction - the standardizer works
    # with it unchanged.
    model_cfg = dict(config_data.model)
    model_name = model_cfg.pop("name")
    model = registry.MODELS[model_name](**model_cfg).to(device)

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
    trainer = Trainer(
        model=model,
        data_module=dm,
        standardizer=standardizer,
        run_dir=run_dir,
        config=config_data,
        resume=args.resume,
    )
    trainer.fit()

    # Final test evaluation with the best checkpoint (baseline gate: MAE ~= 5.4 meV).
    # Trainer.compute_mae is the one MAE definition the framework has -
    # best.pt lands in the same model object the trainer holds, so the
    # final report is literally the same math as every val MAE.
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    mae = trainer.compute_mae(dm.test_dataloader())
    print(f"Test MAE: {dm.unit_conversion(mae):.3f}")

    return 0


if __name__ == "__main__":
    # True only when executed directly (python scripts/train.py), not when
    # imported. SystemExit carries main's return code out as the exit code.
    raise SystemExit(main())

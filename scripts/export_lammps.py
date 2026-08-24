"""M5 assembly: a trained run -> LammpsWrapper, verified against the pipeline.

Loads the run's config snapshot and best checkpoint, rebuilds the model
via the registry, restores the standardizer, wraps the core, and prints
two checks:

  1. energy parity: wrapper total vs the training pipeline (model energy
     + standardizer.inverse) on the first val batch, atol 1e-6
  2. force check: wrapper forces vs central differences on the first
     molecule of that batch, rel < 1e-4

The LAMMPS-side plugin glue (Phase B) is not here - this script only
verifies the math LAMMPS will call. float64 throughout: the M5
acceptance's 1e-6 parity is about the math, not float32 rounding.
"""

import argparse

import torch

from potlab import ROOT
import potlab.config as config
import potlab.data.qm9  # side effect: registers "qm9" in DATASETS
import potlab.models.painn.model  # side effect: registers "painn" in MODELS
import potlab.registry as registry
from potlab.data.qm9 import Qm9Standardizer
from potlab.export.lammps import LammpsWrapper

DEFAULT_RUN = "baseline"


def build_parser():
    """Build the argument parser (separate from main so tests can reuse it)."""
    parser = argparse.ArgumentParser(
        description="Verify a trained run under the LAMMPS wrapper."
    )
    parser.add_argument(
        "--run", type=str, default=DEFAULT_RUN,
        help=f"Run name under runs/ (default: {DEFAULT_RUN}).",
    )
    return parser


def main():
    """Assemble the wrapper from a run and print the two verification checks."""
    parser = build_parser()
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device)
    config_data = config.load_config(run_dir / "config.yaml")

    # The same assembly as train.py: registry model + checkpoint weights;
    # float64 so the parity tolerances measure the math.
    model_cfg = dict(config_data.model)
    model_name = model_cfg.pop("name")
    model = registry.MODELS[model_name](**model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).double().eval()

    standardizer = Qm9Standardizer()
    standardizer.load_state_dict(checkpoint["standardizer"])
    wrapper = LammpsWrapper(model.painn_core, standardizer)

    # One val batch from the run's own config is enough for the check.
    data_cfg = dict(config_data.data)
    data_name = data_cfg.pop("name")
    dm = registry.DATASETS[data_name](**data_cfg, seed=config_data.seed)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.val_dataloader())).to(device)
    batch.pos = batch.pos.double()

    # Check 1: energy parity against the training pipeline.
    with torch.no_grad():
        pipeline = standardizer.inverse(
            model.energy(batch.z, batch.pos, batch.batch), batch.z, batch.batch
        )
    idx_i, idx_j = model._radius_graph(batch.pos, batch.batch)
    wrapped = wrapper.energy(batch.z, batch.pos, idx_i, idx_j)

    energy_diff = (wrapped - pipeline.sum()).abs().item()
    energy_ok = energy_diff < 1e-6
    print(f"Energy parity: |wrapper - pipeline| = {energy_diff:.3e} eV "
          f"({'PASS' if energy_ok else 'FAIL'})")

    # Check 2: forces vs central differences on the first molecule.
    mask = batch.batch == 0
    z_mol = batch.z[mask]
    pos_mol = batch.pos[mask].detach().clone()
    graph_mol = torch.zeros(len(z_mol), dtype=torch.long, device=device)
    idx_i, idx_j = model._radius_graph(pos_mol, graph_mol)

    _, forces_auto = wrapper.energy_and_forces(z_mol, pos_mol, idx_i, idx_j)
    forces_auto = forces_auto.detach()
    pos_mol = pos_mol.detach().clone()

    eps = 1e-6
    forces_fd = torch.zeros_like(forces_auto)
    for i in range(pos_mol.shape[0]):
        for j in range(3):
            pos_plus = pos_mol.clone()
            pos_plus[i, j] += eps
            pos_minus = pos_mol.clone()
            pos_minus[i, j] -= eps
            e_plus = wrapper.energy(z_mol, pos_plus, idx_i, idx_j)
            e_minus = wrapper.energy(z_mol, pos_minus, idx_i, idx_j)
            forces_fd[i, j] = -(e_plus - e_minus) / (2 * eps)

    rel_error = (forces_fd - forces_auto).abs().max() / forces_fd.abs().max()
    force_ok = rel_error < 1e-4
    print(f"Force check: rel error = {rel_error:.3e} "
          f"({'PASS' if force_ok else 'FAIL'})")

    return 0 if (energy_ok and force_ok) else 1


if __name__ == "__main__":
    # True only when executed directly, not when imported. SystemExit
    # carries main's return code out as the exit code.
    raise SystemExit(main())

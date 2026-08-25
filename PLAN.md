# potlab-ml — Development Plan

This document is the roadmap for rebuilding the PaiNN course project into potlab-ml.
Work through the milestones in order; each ends with acceptance criteria you can check.
Interfaces referenced here are specified in [DESIGN.md](DESIGN.md).

## Frozen design decisions

These three decisions are made up front and are not revisited later:

1. **Core / graph-builder split.** Every geometric model is split into
   - a **core** (pure PyTorch, no PyG) that takes `(z, pos, idx_i, idx_j)` and returns per-atom contributions, and
   - a **graph-builder adapter** (PyG `radius_graph` or ASE neighbor lists) that produces the edge indices.

   Rationale: LAMMPS owns the neighbor list at inference time (pair_style mliap feeds it into the Python plugin); the inference entry point is the core only. Keeping PyG out of the core is what makes that split possible.

   (2026-08, M4): the original "TorchScript-safe" wording is dropped — `torch.jit.script` is deprecated upstream (torch ≥ 2.9), and the project abandoned the TorchScript export path. The split itself survives unchanged: MLIAP-Python needs the same graph-agnostic core.

2. **Unified model protocol.** All models implement `BaseModel` ([DESIGN.md](DESIGN.md#basemodel-protocol)): `energy()`, `energy_and_forces()`, and optionally `atomic_contributions()`. Total energy is the interface common denominator — forces come from autograd for any differentiable model, and per-atom decomposition is an optional capability, not a requirement.

3. **Config-driven + registries.** All hyperparameters live in a YAML config; models and datasets are registered by name. `scripts/train.py` never imports a concrete model or dataset.

## Target layout

```
potlab-ml/
├── configs/default.yaml
├── scripts/{train,export_lammps,make_mliap_pickle}.py
├── potlab/
│   ├── config.py, registry.py
│   ├── data/{base,transforms,qm9}.py        (+ vasp.py in the future)
│   ├── models/{base.py, painn/{core,model}.py}
│   ├── training/{trainer,metrics,callbacks}.py
│   └── export/{lammps,mliappy}.py
├── tests/
└── runs/<run_name>/
```

## Milestones

### M0 — Scaffold

- Create the folder skeleton, `pyproject.toml`/`setup.py`, `configs/default.yaml`, a dummy `tests/test_smoke.py`, and empty registry + config loader.
- Port the config values from the original CLI defaults ([`../02456_painn_project-main/minimal_example.py`](../02456_painn_project-main/minimal_example.py)): 3 message-passing layers, 128 features, 20 RBFs, cutoff 5.0 Å, `lr=5e-4`, `weight_decay=0.01`, AdamW + cosine annealing, split `[110000, 10000, 10831]`, seed 0.
- Fix the inherited bugs while porting (list below).

**Acceptance:** `pytest` collects and passes the dummy test; `scripts/train.py --config configs/default.yaml --help` prints the resolved config; `runs/` layout helper works.

Inherited bugs to fix (from the original `minimal_example.py` / `qm9.py`):
1. `args.subset_size` is used but never defined in the CLI → `AttributeError` at startup.
2. `splits` values are never validated — if they do not sum to the dataset size, the third value is silently ignored. Add an explicit check.
3. `Best val. MAE` is printed without the unit conversion (eV) while the progress bar and `Test MAE` use meV. Unify on meV.
4. `GetTarget` has a dead None-branch (`self.target = [target]` is never `None`); clean it up.
5. The epoch loss accumulator mixes `reduction='sum'` values and per-batch means — keep the sum-then-divide-once pattern but name variables to make it explicit (`loss_sum`, `loss_mean`).

### M1 — Data layer (behavior-identical)

- Port `QM9DataModule` to the new structure, implementing `BaseDataModule` ([DESIGN.md](DESIGN.md#basedatamodule-protocol)).
- Move `GetTarget` into shared `transforms.py`; split `get_target_stats` into a `Qm9Standardizer` ([DESIGN.md](DESIGN.md#standardizer)).
- Keep behavior bit-identical (same shuffle seed, same stats, same transforms).

**Acceptance:** with the *old* PaiNN model imported unchanged, the new training entry point reproduces the baseline **test MAE ≈ 5.4 meV** (val ≈ 5.6 meV at the best epoch). Standardizer roundtrip test passes.

### M2 — Model split (core + adapter)

- Split PaiNN into `painn/core.py` (embeddings, message blocks, update blocks, readout — inputs `(z, pos, idx_i, idx_j)`, no PyG imports) and `painn/model.py` (a `BaseModel` that builds the graph with `radius_graph` and calls the core).
- Move the post-processing logic so the standardizer owns it (the model no longer knows about meV, atom refs, or standardization).

**Acceptance:** `PaiNNCore` output matches the old monolithic model to `atol=1e-6` on the same inputs; energy is invariant under rotation of the molecule.

### M3 — Trainer (checkpoints, resume, visualization)

- Implement `training/trainer.py` around the callback contract ([DESIGN.md](DESIGN.md#trainer-and-callbacks)): per-epoch callbacks, early stopping, gradient-norm hook (optional).
- Checkpointing: save per epoch (or top-k) — model state dict, standardizer state, optimizer + scheduler state, config snapshot, epoch, best val MAE. `--resume <run>` restores everything and continues.
- `training/metrics.py`: append-per-epoch CSV (`metrics.csv`) plus a per-step learning-rate table (`lr_steps.csv`); CSV is the source of truth.
- `training/callbacks.py`: `EarlyStoppingCallback`, `TensorBoardCallback`, `PlotCallback` (2×2 matplotlib panel). Details in [docs/training.md](docs/training.md).
- Loss: energy MSE + optional force MSE term `λ·MSE(forces)`, engaged automatically when the dataset provides forces and the config sets `training.loss.forces`.

**Acceptance:** kill the process mid-training; `--resume` continues from the same epoch and reproduces the same metrics. `runs/<name>/` contains `metrics.csv`, `lr_steps.csv`, `plots/latest.png`, `checkpoints/`, `config.yaml`. TensorBoard shows train loss / val MAE / lr curves.

### M4 — Tests that matter

Implement the suite described in [docs/training.md](docs/training.md#test-checklist):
- rotation invariance of the energy; equivariance of forces under rotation
- autograd forces vs finite differences (relative error < 1e-4)
- standardizer roundtrip (transform → inverse restores original labels)
- split validation and dataset contract checks

**Acceptance:** `pytest` is green. These tests are the permanent "exam" any future model must pass before being merged.

### M5 — LAMMPS integration (MLIAP-Python)

- **The one path: MLIAP-Python.** Run any `BaseModel` in `pair_style mliap` via a small Python plugin on `PYTHONPATH` — no export artifact, no libtorch, no per-model pair style. The plugin wraps the trained model + standardizer: it calls `standardizer.inverse` (LAMMPS needs absolute energies, not standardized residuals) and feeds LAMMPS' own neighbor list to the core's `(z, pos, idx_i, idx_j)` interface. Details in [docs/export.md](docs/export.md).
- LAMMPS-side testing on Windows is awkward: use WSL2 or a Linux box.

**Acceptance:** the mliap plugin wrapper reproduces the eager model's energies/forces to 1e-6; (environment permitting) a short LAMMPS MD run of methane completes and matches Python-side energies.

TorchScript was dropped at M4 (`torch.jit.script` deprecated upstream); the MLIAP-Python path was always model-agnostic and now carries the whole export story.

### M6 — Extensibility proof

- Write a **toy model** (e.g., a small MLP pooling the atom embedding, or a linear model) and a **toy dataset** (random molecules with a known analytic energy and forces) registered under new names.
- Train them end-to-end **by changing only the YAML config** — no trainer, loss, or script edits.
- Confirm the force-loss term engages automatically for the force-bearing toy dataset.

**Acceptance:** `--config configs/toy.yaml` trains, visualizes, checkpoints, and passes the M4 test suite. This proves the registry + protocol design, and is the template for every future model and dataset.

### M7 — VaspDataModule (planned, not started)

- Implement `potlab/data/vasp.py`: `VaspDataModule` (`@register_dataset("vasp")`) reading VASP
  output (`vasprun.xml` / `OUTCAR`) via ASE into the [DESIGN.md](DESIGN.md#3-data-contract-the-batch)
  batch contract — PBC neighbor lists (ASE `primitive_neighbor_list` with shift vectors `S`),
  per-frame batching for varying cells, and trajectory-aware splitting.
- Implement `VaspStandardizer`: fit per-element references by `np.linalg.lstsq`, then mean/std.
- The full design is already written in [docs/data.md](docs/data.md) ("Periodic systems: VASP
  via ASE"); this milestone is the implementation of that section.

**Acceptance:** a VASP trajectory trains end-to-end (the force-loss term engages via
`has_forces`), and the model passes the M4 suite on the periodic data.

### M8 — Self-active learning (planned, not started)

- Add a loop that proposes new structures, scores them (e.g. force-error / uncertainty),
  selects the informative ones, and feeds them back into the training set for the next round.
- No design decided yet — this is a placeholder until M7 lands and the labeling source
  (DFT / a trusted reference potential) is chosen.

**Acceptance:** to be defined when the approach is chosen.

## Recommended execution order

```
M0 → M1 → M3 → M2 → M4 → M5 → M6 → M7 → M8
```

Note the swap: **trainer (M3) before model split (M2)**. Rationale: M1 leaves you with a working end-to-end pipeline that reproduces the baseline — the old code is your regression oracle. Only then do you open the model's internals (M2), re-verify against the oracle, and let the permanent test suite (M4) lock behavior in. Every step keeps a runnable, comparable system.

## Risks

| Risk | Mitigation |
|---|---|
| MLIAP-Python plugin not found by LAMMPS (PYTHONPATH / Python-enabled build) | Test the plugin early on WSL2/Linux with the conda env's Python; keep the plugin thin (BaseModel + standardizer.inverse only) |
| Units baked incorrectly into the export | Export must go through the standardizer inverse; the parity check (M5) compares against the training-time pipeline, not the raw core |
| PyG version drift (2.6.1 vs 2.8 `radius_graph` padding behavior differs) | Pin `torch-geometric` in `pyproject.toml`; the trainer only ever depends on the batch contract, never on graph-building internals |
| MD-frame leakage in future periodic datasets | Split by trajectory/simulation run, never by shuffling frames (see [docs/data.md](docs/data.md)) |
| Scope creep | Each milestone ends with checkable acceptance criteria; visualization stays simple (CSV + one panel figure + TensorBoard) |

# potlab-ml — Development Plan

This document is the roadmap for rebuilding the PaiNN course project into potlab-ml.
Work through the milestones in order; each ends with acceptance criteria you can check.
Interfaces referenced here are specified in [DESIGN.md](DESIGN.md).

## Frozen design decisions

These three decisions are made up front and are not revisited later:

1. **Core / graph-builder split.** Every geometric model is split into
   - a **core** (pure PyTorch, no PyG, no Python control flow — TorchScript-safe) that takes `(z, pos, idx_i, idx_j)` and returns per-atom contributions, and
   - a **graph-builder adapter** (PyG `radius_graph` or ASE neighbor lists) that produces the edge indices.

   Rationale: LAMMPS owns the neighbor list at inference time; the exported artifact must be the core only. Keeping PyG out of the core is what makes export possible.

2. **Unified model protocol.** All models implement `BaseModel` ([DESIGN.md](DESIGN.md#basemodel-protocol)): `energy()`, `energy_and_forces()`, and optionally `atomic_contributions()`. Total energy is the interface common denominator — forces come from autograd for any differentiable model, and per-atom decomposition is an optional capability, not a requirement.

3. **Config-driven + registries.** All hyperparameters live in a YAML config; models and datasets are registered by name. `scripts/train.py` never imports a concrete model or dataset.

## Target layout

```
potlab-ml/
├── configs/default.yaml
├── scripts/{train,evaluate,export_lammps}.py
├── src/potlab/
│   ├── config.py, registry.py
│   ├── data/{base,transforms,qm9}.py        (+ vasp.py in the future)
│   ├── models/{base.py, painn/{core,model}.py}
│   ├── training/{trainer,metrics,callbacks}.py
│   └── export/lammps.py
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
- Run the **TorchScript smoke test here, not at M5**: `torch.jit.script(PaiNNCore)` must succeed before this milestone is done. If it fails, fix the core now.
- Move the post-processing logic so the standardizer owns it (the model no longer knows about meV, atom refs, or standardization).

**Acceptance:** `PaiNNCore` output matches the old monolithic model to `atol=1e-6` on the same inputs; `torch.jit.script` succeeds; energy is invariant under rotation of the molecule.

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
- TorchScript-vs-eager parity on the core
- split validation and dataset contract checks

**Acceptance:** `pytest` is green. These tests are the permanent "exam" any future model must pass before being merged.

### M5 — LAMMPS export (two paths)

- **Primary: TorchScript + ML-PAINN.** Bake the standardizer's inverse transform into the exported module (LAMMPS needs absolute energies, not standardized residuals). Export via `torch.jit.script` + `torch.jit.save`; verify scripted vs eager parity to 1e-6. Details in [docs/export.md](docs/export.md).
- **Alternative: MLIAP-Python.** Document the plugin structure for running any `BaseModel` in `pair_style mliap` without an export step — the generic escape hatch for future models.
- LAMMPS-side testing on Windows is awkward: use WSL2 or a Linux box. libtorch must match the torch version used for training (cu126 wheels → cu126 libtorch).

**Acceptance:** `export_lammps.py` produces a `.pt` whose energies/forces match the eager model to 1e-6; (environment permitting) a short LAMMPS MD run of methane completes and matches Python-side energies.

### M6 — Extensibility proof

- Write a **toy model** (e.g., a small MLP pooling the atom embedding, or a linear model) and a **toy dataset** (random molecules with a known analytic energy and forces) registered under new names.
- Train them end-to-end **by changing only the YAML config** — no trainer, loss, or script edits.
- Confirm the force-loss term engages automatically for the force-bearing toy dataset.

**Acceptance:** `--config configs/toy.yaml` trains, visualizes, checkpoints, and passes the M4 test suite. This proves the registry + protocol design, and is the template for every future model and dataset.

## Recommended execution order

```
M0 → M1 → M3 → M2 → M4 → M5 → M6
```

Note the swap: **trainer (M3) before model split (M2)**. Rationale: M1 leaves you with a working end-to-end pipeline that reproduces the baseline — the old code is your regression oracle. Only then do you open the model's internals (M2), re-verify against the oracle, and let the permanent test suite (M4) lock behavior in. Every step keeps a runnable, comparable system.

## Risks

| Risk | Mitigation |
|---|---|
| TorchScript incompatibilities in the core | Split at M2 and run the `torch.jit.script` smoke test immediately; restrict the core to plain tensor ops (indexing, `index_add_`, `ModuleList`, `Embedding` are all scriptable) |
| Units baked incorrectly into the export | Export must go through the standardizer inverse; the parity check (M5) compares against the training-time pipeline, not the raw core |
| PyG version drift (2.6.1 vs 2.8 `radius_graph` padding behavior differs) | Pin `torch-geometric` in `environment.yml`; the trainer only ever depends on the batch contract, never on graph-building internals |
| MD-frame leakage in future periodic datasets | Split by trajectory/simulation run, never by shuffling frames (see [docs/data.md](docs/data.md)) |
| Scope creep | Each milestone ends with checkable acceptance criteria; visualization stays simple (CSV + one panel figure + TensorBoard) |

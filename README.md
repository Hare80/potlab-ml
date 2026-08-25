# potlab-ml

A small, model-agnostic framework for training machine-learned potentials (MLPs) on
molecular and periodic systems.

PaiNN is the first supported model, but it is deliberately **one plugin among many**:
models, datasets, and visualizations are all behind small interfaces, so adding a new
model (or a new data source) means writing one file and registering it — nothing else.

**potlab** = *potential* + *lab*. This is a self-driven learning project: the code is
written by hand, module by module, against the specs in this folder. No framework hides
the interesting parts — the training loop, the neighbor building, the autograd forces,
and the export path are all yours.

> 📖 **Start here:** [docs/user-guide.md](docs/user-guide.md) — train, monitor, and export a
> model. Extend it with [docs/developer-guide.md](docs/developer-guide.md); look up the full
> API in [docs/api.md](docs/api.md).

## Features

- **Pluggable models** — a unified `BaseModel` protocol (`energy`, `energy_and_forces`,
  optional per-atom contributions); PaiNN ships first.
- **Pluggable datasets** — a single batch data contract; QM9 (molecular properties)
  implemented, VASP trajectories (energies + forces, periodic) designed in
  [docs/data.md](docs/data.md) (M7).
- **Training ergonomics** — checkpoints with `--resume` / `--warm-start`, early stopping,
  YAML config with `-o key=value` overrides, seeded reproducibility.
- **Visualization** — TensorBoard for live monitoring, CSV + matplotlib for
  report-ready figures.
- **Tests that matter** — rotation invariance, force equivariance, autograd-vs-finite-
  difference gradient checks, standardizer roundtrips (the permanent M4 suite).
- **LAMMPS integration** — MLIAP-Python (`pair_style mliap unified`): a trained model +
  standardizer are pickled and run inside LAMMPS through its Python coupling. See
  [docs/export.md](docs/export.md).

## Quick start

```bash
# in a Python env with torch, torch-geometric, and lightning-fabric installed
python scripts/train.py --config configs/default.yaml
```

This downloads QM9 on first run, trains, and **prints the Test MAE at the end** (the
baseline to match is ≈ 5.4 meV on QM9 target 7). There is no separate `evaluate.py`.

```bash
python scripts/train.py --config configs/default.yaml -o training.num_epochs=5   # quick smoke
python scripts/train.py --config configs/default.yaml --resume                   # continue a run
python scripts/export_lammps.py --run baseline                                   # wrapper-vs-pipeline parity
```

Dependencies are declared in `pyproject.toml` (torch / torch-geometric / lightning-fabric
are *not* pinned there — install them into the same environment you run from).

## Layout

```
potlab-ml/
├── configs/default.yaml          # default training config
├── scripts/
│   ├── train.py                  # train / resume / warm-start from config
│   ├── export_lammps.py          # wrapper-vs-pipeline parity checker
│   └── make_mliap_pickle.py      # build the pair_style mliap unified pickle
├── potlab/
│   ├── config.py                 # dataclass config + YAML loader + -o overrides
│   ├── registry.py               # model + dataset registries
│   ├── data/                     # BaseDataModule, Standardizer, transforms, QM9 (+ VASP in M7)
│   ├── models/                   # BaseModel + painn/ (core + adapter)
│   ├── training/                 # trainer, metrics logger, callbacks
│   └── export/                   # lammps.py (LammpsWrapper) + mliappy.py (MliapPaiNN)
├── examples/lammps/methane/      # turnkey methane MD smoke (data + input + runbook)
├── tests/                        # pytest suite (see docs/training.md)
└── runs/<run_name>/              # metrics.csv, plots/, checkpoints/, config.yaml
```

## Documentation

- [docs/user-guide.md](docs/user-guide.md) — **start here**: train, monitor, and export.
- [docs/api.md](docs/api.md) — the complete API reference (per-parameter docs + examples).
- [docs/developer-guide.md](docs/developer-guide.md) — architecture and how to extend the framework.
- [DESIGN.md](DESIGN.md) — interfaces, data contract, registries, config schema.
- [PLAN.md](PLAN.md) — development roadmap with milestones (M0–M8) and acceptance criteria.
- [docs/data.md](docs/data.md) — QM9 and periodic VASP data handling.
- [docs/training.md](docs/training.md) — metrics, visualization, and the test checklist.
- [docs/export.md](docs/export.md) — the MLIAP-Python LAMMPS integration path.

## Baseline to beat

The refactor is successful when the new structure reproduces the reference result of the
original course project on QM9 target 7 (internal energy at 0 K): **test MAE ≈ 5.4 meV**
(≈ 5.6 meV validation MAE at the best epoch, 110 000 / 10 000 / 10 831 split, seed 0).
The original script remains the regression oracle throughout the rewrite.

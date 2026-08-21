# potlab-ml

A small, model-agnostic framework for training machine-learned potentials (MLPs) on molecular and periodic systems.

PaiNN is the first supported model, but it is deliberately **one plugin among many**: models, datasets, and visualizations are all behind small interfaces, so adding a new model (or a new data source) means writing one file and registering it — nothing else.

**potlab** = *potential* + *lab*. This is a self-driven learning project: the code is written by hand, module by module, against the specs in this folder. No framework hides the interesting parts — the training loop, the neighbor building, the autograd forces, and the export path are all yours.

## Features (target)

- **Pluggable models** — unified `BaseModel` protocol (`energy`, `energy_and_forces`, optional per-atom contributions); PaiNN ships first
- **Pluggable datasets** — a single batch data contract; QM9 (molecular properties) and VASP trajectories (energies + forces, periodic) via ASE
- **Training ergonomics** — checkpoints with resume, early stopping, YAML config, seeded reproducibility
- **Visualization** — TensorBoard for live monitoring, CSV + matplotlib for report-ready figures
- **Tests that matter** — rotation invariance, force equivariance, autograd-vs-finite-difference gradient checks, standardizer roundtrips, TorchScript parity
- **LAMMPS export** — two documented paths: TorchScript + ML-PAINN (`pair_style painn`, primary, no Python in the MD loop) and MLIAP-Python (model-agnostic, no export step). See [docs/export.md](docs/export.md).

## Layout (target)

```
potlab-ml/
├── configs/default.yaml      # default training config
├── scripts/
│   ├── train.py              # train / resume from config
│   ├── evaluate.py           # test-set MAE of a checkpoint
│   └── export_lammps.py      # TorchScript export
├── potlab/
│   ├── config.py             # dataclass config + YAML loader
│   ├── registry.py           # model + dataset registries
│   ├── data/                 # BaseDataModule, transforms, QM9, (future) VASP
│   ├── models/               # BaseModel + painn/ (core + adapter)
│   ├── training/             # trainer, metrics logger, callbacks
│   └── export/               # LAMMPS export (TorchScript primary, MLIAP alternative)
├── tests/                    # pytest suite (see docs/training.md)
└── runs/<run_name>/          # metrics.csv, plots/, checkpoints/, config.yaml
```

## Quick start (to be filled in after implementation)

```bash
conda env create -f environment.yml   # or reuse an existing torch + PyG env
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --run runs/baseline
python scripts/export_lammps.py --run runs/baseline --out model.pt
tensorboard --logdir runs
```

## Documentation

- [PLAN.md](PLAN.md) — development roadmap with milestones and acceptance criteria
- [DESIGN.md](DESIGN.md) — interfaces, data contract, registries, config schema
- [docs/data.md](docs/data.md) — QM9 and periodic VASP data handling
- [docs/training.md](docs/training.md) — metrics, visualization, and the test checklist
- [docs/export.md](docs/export.md) — TorchScript/ML-PAINN and MLIAP-Python export paths

## Baseline to beat

The refactor is successful when the new structure reproduces the reference result of the
original course project on QM9 target 7 (internal energy at 0 K): **test MAE ≈ 5.4 meV**
(≈ 5.6 meV validation MAE at the best epoch, 110 000 / 10 000 / 10 831 split, seed 0).
The original script remains the regression oracle throughout the rewrite.

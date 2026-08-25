# potlab-ml — User Guide

The task-oriented guide for *using* the framework: install, train, monitor, and export a
model to LAMMPS. For the exact signatures of everything mentioned here, see
[api.md](api.md); for how it works under the hood, see [developer-guide.md](developer-guide.md).

## Core concepts before you start

Three ideas cause most of the confusion in this codebase. Read them once.

### Don't conflate three numbers: `num_outputs` / `target` / `energy_index`

| Field | Where | Default | Question it answers |
|---|---|---|---|
| `num_outputs` | `model.num_outputs` (config `model.num_outputs`) | `1` | How **wide** is the model output? |
| `target` | `data.target` (config `data.target`) | `7` | **Which physical property** are we predicting (QM9 column 0–18)? |
| `energy_index` | `BaseDataModule.energy_index` | `0` or `None` | **Which output column** is the PES energy? |

- **`num_outputs` = the width.** The number of columns of the model output — nothing to do
  with which property those columns hold. On QM9 it is always 1.
- **`target` = the property.** Which QM9 column to predict (`7` = U0). Independent of
  `num_outputs`.
- **`energy_index` = the column selector.** Which output column is the physical energy:
  `0` iff `target in {7}`, else `None`. Only `target: 7` is exportable to LAMMPS.

### The word "energy" means three different things

| Thing | What it actually is |
|---|---|
| `BaseModel.energy(z, pos, graph_indexes)` | The model's whole output `[N_graphs, num_outputs]`, standardized space, all columns. |
| `LammpsWrapper.energy(z, pos, idx_i, idx_j)` | A scalar physical total energy (0-dim). |
| `energy_index` | The column selector that maps between the two. |

(The model's output layer is `readout_network`, not "energy".)

### Element identity: atomic numbers vs `ELEMENT_Z`

The model embeds **atomic numbers** directly (it has no `element_types` parameter). Only
the LAMMPS glue has the name→number map `ELEMENT_Z = {"H":1,"C":6,"N":7,"O":8}`. The
`--elements C,H` flag on `make_mliap_pickle.py` declares the LAMMPS atom types in
`pair_coeff` order — it does **not** configure the model.

---

## Install & environment

`pyproject.toml` declares the pure-Python dependencies (`numpy`, `pyyaml`, `tqdm`,
`tensorboard`, `matplotlib`, `ase`). It does **not** pin the three ML stack packages —
install them into the same environment you run from:

- `torch`
- `torch-geometric` (and its backend: `torch-cluster` on 2.6.x, `pyg-lib` on 2.8.x)
- `lightning-fabric` (used only for `seed_everything`)

On the reference machine this is the conda env `pytorch` (Python 3.13, CUDA torch); run
everything with that env's interpreter, not a bare `python` (which on Windows is a
non-functional Store stub).

## Quick start

```bash
python scripts/train.py --config configs/default.yaml
```

Training downloads QM9 on first run, then trains `num_epochs` epochs and **prints the
Test MAE at the end** — there is no separate `evaluate.py`. The baseline to match is
**test MAE ≈ 5.4 meV** on QM9 target 7 (seed 0, split `[110000, 10000, 10831]`).

```bash
python scripts/train.py --config configs/default.yaml -o training.num_epochs=5        # quick smoke
python scripts/train.py --config configs/default.yaml -o data.target=7 -o data.splits='[800,100,100]' --subset-size 1000
python scripts/train.py --config configs/default.yaml --resume                        # continue the same run
python scripts/train.py --config configs/default.yaml --warm-start runs/baseline/checkpoints/best.pt
```

## Config walkthrough

The config ([`configs/default.yaml`](../configs/default.yaml)) has four fixed top-level
scalars (`run_name`, `seed`) and four **open sections** — `model`, `data`, `training`,
`export`. Unknown keys inside the open sections are forwarded verbatim to the registered
constructor.

```yaml
model:
  name: painn                # key in registry.MODELS
  num_message_passing_layers: 3
  num_features: 128
  num_outputs: 1             # output WIDTH
  num_unique_atoms: 100
  cutoff_dist: 5.0

data:
  name: qm9                  # key in registry.DATASETS
  target: 7                  # which QM9 property
  splits: [110000, 10000, 10831]
  batch_size_train: 100
  batch_size_eval: 1000

training:
  num_epochs: 1000
  optimizer: { name: AdamW, lr: 5.0e-4, weight_decay: 0.01 }
  scheduler: { name: CosineAnnealingLR }      # T_max is derived, not configured
  loss: { energy: 1.0, forces: null }         # set forces to engage the force term
  early_stopping: { patience: 30, min_epochs: 1000 }
```

Override any value with dotted paths:

```bash
-o training.optimizer.lr=1.0e-3
-o data.target=7
-o model.num_outputs=1
```

Values are parsed with `yaml.safe_load` (`"3"` → `int`, `"1.0e-3"` → `float`, `"null"` →
`None`). **`optimizer.name` / `scheduler.name` are case-sensitive** — `AdamW`, not `adamw`.

## Training artifacts & monitoring

Each run writes to `runs/<run_name>/`:

```
runs/<run_name>/
├── metrics.csv        # per-epoch rows — the source of truth
├── lr_steps.csv       # per-step learning rate
├── plots/latest.png   # 2×2 panel, redrawn every 10 epochs
├── checkpoints/
│   ├── best.pt        # best val-MAE checkpoint
│   └── latest.pt      # most recent epoch (resume point)
└── config.yaml        # config snapshot (written only on a fresh start)
```

`metrics.csv` columns: `epoch, train_loss, val_mae, lr, epoch_time, grad_norm`. `val_mae`
is in display units (meV for QM9 energy targets). TensorBoard (`tensorboard --logdir
runs`) is a live view on top; the CSV is the record. See [training.md](training.md) for
reading the curves.

## Export to LAMMPS

Once you have a trained run with `best.pt` (and it trains `target: 7`), export it as a
pickle LAMMPS can run:

```bash
python scripts/make_mliap_pickle.py --run <run_name> --elements C,H --out model.pkl
```

`--elements` declares the LAMMPS atom types in `pair_coeff` order. The input script then
uses:

```
pair_style mliap unified model.pkl 0
pair_coeff * * C H
```

The turnkey methane example (data + input + exact Windows run commands) is in
[`examples/lammps/methane/README.md`](../examples/lammps/methane/README.md); the math and
parity check are in [export.md](export.md).

## FAQ / troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ValueError: model.num_outputs (X) must match ... num_targets (Y)` | Output width ≠ dataset target count. On QM9 `num_targets` is always 1. |
| `ValueError: ... trains a non-energy target ... only PES-energy targets can be exported` | `data.target` is not in `PES_ENERGY_TARGETS = {7}`. Only `target: 7` exports. |
| `ValueError: Unknown element names [...] - ELEMENT_Z covers [...]` | Add the element to `ELEMENT_Z` in [`potlab/export/mliappy.py`](../potlab/export/mliappy.py). |
| `pair_style mliap unified: could not unpickle / No module named 'potlab'` | LAMMPS' embedded Python can't import potlab — set `PYTHONPATH` (see the methane README). |
| `Buffer dtype mismatch / Float did not Match Double` | The LAMMPS C++ side is double; you are bypassing `MliapPaiNN`'s float64 casts. |
| `TypeError: ... unexpected keyword argument` at startup | An unknown **top-level** YAML key; keys must match the `Config` dataclass fields. |
| `ValueError: override must look like KEY=VALUE` | An `-o` flag missing its `=`. |

## Glossary

| Term | What it means |
|---|---|
| **registry** | A dict mapping a name → class (`MODELS`, `DATASETS`), filled by a decorator. |
| **standardizer** | Owns all target transforms (refs, size, mean/std) and their inverses. |
| **`energy_index`** | Which output column is the PES energy (or `None`). |
| **`num_outputs` / `num_targets`** | Model output width / dataset target width (must match). |
| **`target`** | Which property column the dataset predicts (e.g. QM9 U0 = 7). |
| **graph_indexes (`batch`)** | Per-atom index of which molecule each atom belongs to. |
| **mean pooling** | The model averages per-atom contributions to a per-molecule prediction. |
| **`inverse_per_atom`** | Per-atom standardized → per-atom physical energy (the export granularity). |
| **`LammpsWrapper`** | Core + standardizer under the `(z, pos, idx_i, idx_j)` interface. |
| **`MliapPaiNN`** | The pickled object `pair_style mliap unified` drives inside LAMMPS. |
| **`pair_style mliap unified`** | The LAMMPS pair style that loads a pickled Python model. |
| **PES energy** | A conservative quantity whose negative gradient is the force (QM9 U0). |
| **atom reference** | A per-element offset subtracted from energies so the model learns small corrections. |

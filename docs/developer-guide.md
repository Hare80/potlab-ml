# potlab-ml — Developer Guide

How the framework is put together, and how to extend it. For the exact signatures of every
class below, see [api.md](api.md); for the formal rules, see [DESIGN.md](../DESIGN.md).

## Architecture & data-flow

```
config.yaml (YAML)
    │  config.load_config  →  Config dataclass (fixed scalars + open model/data/training/export dicts)
    ▼
registry.DATASETS[data.name]  →  BaseDataModule   (prepare → setup → dataloaders)
registry.MODELS[model.name]   →  BaseModel        (energy(z, pos, graph_indexes))
    │
    ▼  standardizer = dm.make_standardizer()      (fit on the train split ONLY)
Trainer  →  model.energy  vs  standardizer.transform(y)   →  MSE (sum-then-divide)
    │
    ▼  best.pt = {model, standardizer, optimizer, scheduler, epoch, best_val_mae, config}
scripts/make_mliap_pickle.py  →  MliapPaiNN  →  pair_style mliap unified <file> 0
```

Four layers, each with one contract:

```
data/       BaseDataModule + Standardizer        — the batch (z/pos/y/batch/forces) and all target math
models/     BaseModel (+ PaiNN core/adapter)     — pure function (z, pos, graph_indexes) → energies
training/   Trainer + callbacks + metrics        — the loop is a thin shell; features are callbacks
export/     LammpsWrapper + MliapPaiNN           — the same core, wrapped, running in LAMMPS
```

The one rule that keeps everything decoupled: **the model never knows about data, units,
or LAMMPS** ([DESIGN.md §10](../DESIGN.md#10-what-the-model-must-never-know-about)).

## Config & registry internals

- **`potlab/config.py`** — `Config` is a dataclass with fixed scalars plus four open dict
  sections. `load_config` reads YAML, applies `-o` dotted-path overrides **before** building
  the dataclass (so `run_name`/`seed` are overrideable), and unknown *top-level* keys raise
  `TypeError`. `apply_overrides` walks `key.subkey=value` and parses values with
  `yaml.safe_load`. See [`load_config`](api.md#load_config).
- **`potlab/registry.py`** — two plain dicts (`MODELS`, `DATASETS`) and two decorator
  factories. Registration is a side effect of import: `@register_model("painn")` on
  `PaiNNModel`, and `scripts/train.py` imports the module just for that side effect.
  Duplicate names raise `ValueError`. See [`register_model`](api.md#register_model).

This is why "adding a model" is one file — write the class, decorate it, import it.

## Data contract & Standardizer

**The batch** ([DESIGN.md §3](../DESIGN.md#3-data-contract-the-batch)) — one big
concatenated graph per batch:

| key | shape | meaning |
|---|---|---|
| `z` | `[N_atoms]` | atomic numbers, molecules concatenated |
| `pos` | `[N_atoms, 3]` | Cartesian coordinates |
| `y` | `[N_graphs, num_outputs]` | per-molecule targets |
| `batch` | `[N_atoms]` | graph index per atom (`0,0,…,1,1,…`) |
| `forces` | `[N_atoms, 3]` | optional — present only when the dataset provides forces |

No padding, no ragged tensors; `len(batch.y)` is the molecule count and the last batch is
usually smaller, so metrics use **sum-then-divide**.

**The standardizer** ([DESIGN.md §5](../DESIGN.md#5-standardizer)) owns every target
transformation and has **two granularities, one implementation**:

1. `inverse(energy_pred, z, graph_indexes)` — the per-**molecule** physical energy, used by
   `Trainer.compute_mae`.
2. `inverse_per_atom(contribs, z)` — the per-**atom** physical energy, used by
   `LammpsWrapper` (LAMMPS has per-atom outputs, no molecule boundaries).

`inverse` is literally `inverse_per_atom` broadcast over a molecule's atoms and summed.
The three-step QM9 pipeline (subtract atom refs → divide by atom count → shift/scale with
train-only mean/std) is mirrored by both, so the exported total equals the training total.

## Model protocol & PaiNN core/adapter split

`BaseModel` is the protocol: `energy`, `energy_and_forces` (autograd, differentiates
**column 0** only), and optional `atomic_contributions`. Forces come from
`-torch.autograd.grad(E, pos)` for any differentiable model.

PaiNN is split ([DESIGN.md §6](../DESIGN.md#6-painn-core--graph-builder-split)):

```
PaiNNCore(z, pos, idx_i, idx_j)            # graph-agnostic, NO PyG
    _edge_geometry(rel_pos)                # dir / cosine cutoff / RBF
    _message_pass(...)                     # embedding → (message, update)×3 → readout_network
    forward_with_edges(z, rel_pos, ...)    # public — the LAMMPS inference entry

PaiNNModel(BaseModel)                      # training adapter
    _radius_graph(pos, batch)              # PyG radius_graph — training-side only
    energy(...)  = mean-pool over per-atom contributions
```

The core takes edge indices as **inputs** because LAMMPS owns the neighbor list at
inference time. `forward_with_edges` is the public hook: the glue hands it LAMMPS'
minimum-image displacements (`rij`) directly, bypassing `forward` (which rebuilds
`rel_pos` from absolute coordinates it does not have).

## Training loop & callbacks

`Trainer` is a thin shell around the `Callback` contract (`on_epoch_end(epoch, metrics,
trainer)`). Per epoch it computes the standardized loss, builds a `metrics` dict, and
calls every callback — the loop never grows feature flags:

- `EarlyStoppingCallback` — sets `trainer.stop` (saving is the trainer's job).
- `TensorBoardCallback` — live scalars.
- `PlotCallback` — redraws the 2×2 panel from `metrics.csv`.

Two concepts worth knowing: **sum-then-divide** (accumulate `reduction="sum"` losses,
divide once by the total molecule count); and **`--resume` vs `--warm-start`** — resume
continues the *same* run (checkpoint authoritative for optimizer/scheduler), warm-start
starts a *new* run with old weights. Full detail in [training.md](training.md).

## Export path in depth

Two modules + one script, documented in [export.md](export.md):

- [`potlab/export/lammps.py`](../potlab/export/lammps.py) — `LammpsWrapper`: wraps the
  **core** + standardizer, exposes `(z, pos, idx_i, idx_j) → (energy, forces)`.
- [`potlab/export/mliappy.py`](../potlab/export/mliappy.py) — `MliapPaiNN`: the duck-typed
  `pair_style mliap unified` object. Translates LAMMPS' data object into the wrapper's
  interface, writes `data.energy`/`data.eatoms`, and scatters forces via the C++
  `update_pair_forces` (the pair interface *is* the chain rule, ghost reverse-communication
  included). It deliberately does **not** import `lammps`.
- `scripts/export_lammps.py` — the **parity checker** (wrapper vs eager pipeline, 1e-6
  energy / 1e-4 forces), not part of the plugin.
- `scripts/make_mliap_pickle.py` — assembles a `MliapPaiNN` from a run and pickles it.

## Adding a model / dataset

The whole design is proven by M6: a toy model + toy dataset train end-to-end **by changing
only the YAML config**. The minimal-change template:

- **New model:** subclass `BaseModel`, `@register_model("name")`, add `import
  potlab.models.your_model` to `scripts/train.py` (the side effect), set
  `model.name: your_name` in config. Expose `num_outputs`; the assembly checks it against
  `num_targets`.
- **New dataset:** subclass `BaseDataModule`, `@register_dataset("name")`, implement the
  six methods + four properties (`has_forces`, `num_targets`, `energy_index`,
  `unit_conversion`) and a `Standardizer`. Set `data.name: your_name` in config.

See [`register_model`](api.md#register_model) / [`register_dataset`](api.md#register_dataset)
and [DESIGN.md §2](../DESIGN.md#2-registries).

## Roadmap

The two next features are **planned, not implemented** — the authoritative home is
[PLAN.md](../PLAN.md):

- **M7 — `VaspDataModule`** (ASE VASP output → datamodule). The full design is in
  [data.md](data.md) ("Periodic systems: VASP via ASE").
- **M8 — self-active learning.** Propose structures, score them, feed the informative ones
  back into training. No design decided yet.

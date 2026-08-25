# potlab-ml — Design

This is the interface specification the code is written against. If a question is not
answered here, prefer the simplest design that keeps the interfaces below unchanged.

## 1. BaseModel protocol

Every model (PaiNN now, others later) implements:

```python
class BaseModel(nn.Module):
    def energy(self, z, pos, graph_indexes) -> Tensor:
        """Per-molecule total energy. [N_atoms] / [N_atoms,3] / [N_atoms] -> [N_graphs, num_outputs]."""

    def energy_and_forces(self, z, pos, graph_indexes) -> tuple[Tensor, Tensor]:
        """Total energy + per-atom forces via autograd. Returns ([N_graphs, out], [N_atoms, 3])."""

    def atomic_contributions(self, z, pos, graph_indexes) -> Tensor:
        """Optional per-atom decomposition. [N_atoms, num_outputs]. Raise NotImplementedError if unsupported."""
```

Rules:

- **Total energy is the common denominator.** Trainers, metrics, and the standardizer only
  ever call `energy` (and `energy_and_forces` when the dataset has forces). Per-atom
  contributions are an optional capability (diagnostics, decomposition plots), never a
  requirement — a model that regresses the total energy directly is a first-class citizen.
  *Energy* means precisely a conservative potential-energy-surface quantity whose negative
  position gradient IS the interatomic force (QM9's U0; not its HOMO, ZPVE or dipole).
- **Column 0 is the energy.** Multi-output models may carry other properties in the
  remaining columns; `energy_and_forces` differentiates column 0 only. Whether the target
  is an energy at all is the dataset's knowledge (`BaseDataModule.energy_index`), never
  the model's.
- **Outputs and targets align 1:1.** Models expose `num_outputs`; the dataset declares
  `num_targets`. The assembly checks equality and refuses the rest — an extra output
  column has no training signal, a missing one drops targets.
- **Forces come from autograd**, for any differentiable model: `-torch.autograd.grad(E.sum(), pos)`.
  The model's output granularity is irrelevant to this. (`create_graph=True` when the force
  loss itself must be differentiated.)
- Inputs are a PyG-style batch: atom types `z`, Cartesian positions `pos`, per-atom graph
  membership `graph_indexes`. Outputs are per-graph tensors (one row per molecule), matching
  the data contract below.

## 2. Registries

Simple dicts with decorators — no plugin framework:

```python
MODELS: dict[str, type[BaseModel]] = {}
DATASETS: dict[str, type[BaseDataModule]] = {}

def register_model(name):
    def deco(cls): MODELS[name] = cls; return cls
    return deco
```

`@register_model("painn")` on `PaiNNModel`; the config selects by name. Trainers and scripts
import the registry only, never concrete classes. The concrete machinery lives in
`potlab/registry.py` (the dicts + decorators) and `potlab/config.py` (the `Config` dataclass
with open `model`/`data`/`training`/`export` sections and dotted-path `-o key=value` overrides).

## 3. Data contract (the batch)

Every dataloader yields batches with exactly these keys:

| key | shape | dtype | meaning |
|---|---|---|---|
| `z` | `[N_atoms]` | long | atomic numbers, molecules concatenated |
| `pos` | `[N_atoms, 3]` | float | Cartesian coordinates |
| `y` | `[N_graphs, num_outputs]` | float | per-molecule targets (energy / property) |
| `batch` | `[N_atoms]` | long | graph index of each atom (`0,0,…,1,1,…`) |
| `forces` | `[N_atoms, 3]` | float | optional — present only when the dataset provides forces |

Notable points:

- One big concatenated graph per batch — no padding, no ragged tensors; molecule boundaries
  live in `batch` (the same flat-list-plus-index philosophy as edge lists in message passing).
- `len(batch.y)` is the molecule count; batch sizes may differ between batches (the last one
  is usually smaller) — metrics must use sum-then-divide-by-total, never per-batch averaging.
- `cell` (`[N_graphs, 3, 3]`) will be added to the contract when periodic datasets land;
  it rides along in the batch like any other attribute.

## 4. BaseDataModule protocol

```python
class BaseDataModule:
    def prepare_data(self) -> None: ...        # one-time: download / parse raw files
    def setup(self) -> None: ...               # split + build subsets
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader: ...
    def test_dataloader(self) -> DataLoader: ...
    def make_standardizer(self) -> Standardizer: ...
    @property
    def has_forces(self) -> bool: ...          # drives loss composition
    @property
    def num_targets(self) -> int: ...          # how many target columns y carries
    @property
    def energy_index(self) -> Optional[int]: ...  # which output column is the energy (None = not one)
    @property
    def unit_conversion(self) -> Callable: ... # display only (eV -> meV etc.)
```

Splitting strategy is private to each dataset (QM9: seeded global shuffle; MD/VASP
trajectories: split by simulation run — see [docs/data.md](docs/data.md)).

## 5. Standardizer

The standardizer owns every target transformation. It operates on **labels**, so it is
independent of what the model outputs.

```python
class Standardizer:
    def fit(self, train_data) -> None: ...
    def transform(self, y, z, graph_indexes) -> Tensor: ...     # labels -> standardized space
    def inverse(self, energy_pred, z, graph_indexes) -> Tensor: # model energies -> physical units
    def inverse_per_atom(self, contribs, z) -> Tensor:          # per-atom standardized -> per-atom physical
    def state_dict(self) / def load_state_dict(self, ...): ...  # saved with checkpoints
```

The canonical pipeline (three steps, mirrored by `inverse`):

1. **Subtract atom references.** If the dataset ships reference values (QM9's
   `atomref`), use them. Otherwise fit per-element references by linear regression on the
   *training set only* — solve `min ||E − X c||²` where `X[m, Z]` counts element `Z` in
   molecule `m` (`np.linalg.lstsq`). This is standard MLIP practice (MD17, NequIP, MACE,
   ANI's self-energies are the same idea); it removes the bulk of the energy so the model
   learns small corrections around zero.
2. **Divide by atom count** (optional, property datasets like QM9) — removes size dependence.
3. **Shift/scale to zero mean, unit variance** using training-set statistics only
   (no validation/test statistics — that would leak).

The network therefore always trains on small, zero-centered targets regardless of dataset.
`AtomwisePostProcessing` from the original project disappears — `Standardizer.inverse` /
`inverse_per_atom` play its role and are data-specific by construction.

**Two granularities, one implementation.** `inverse` is the per-molecule aggregation of
`inverse_per_atom` (broadcast the pooled value to its atoms, apply the per-atom transform,
sum per graph). The LAMMPS mliap wrapper (M5) calls `inverse_per_atom` directly: LAMMPS has
per-atom outputs and no molecule boundaries, so the export path needs the per-atom
granularity — and its sum is algebraically the same total the training pipeline predicts.

## 6. PaiNN: core / graph-builder split

```
PaiNNCore(nn.Module)                      # graph-agnostic, no PyG imports
    forward(z, pos, idx_i, idx_j) -> atomic_contributions [N, out]
        # embedding -> (message block, update block) x3 -> readout
        # edge features computed internally from pos and the indices:
        #   rel_pos = pos[idx_j] - pos[idx_i]; dist, dir, rbf, cosine cutoff

PaiNNModel(BaseModel)                     # training-time adapter
    forward(z, pos, graph_indexes):
        idx_i, idx_j = radius_graph(pos, r=cutoff, batch=graph_indexes, ...)
        return PaiNNCore.forward(z, pos, idx_i, idx_j)
```

Rationale and rules:

- LAMMPS owns neighbor lists at inference time (pair_style mliap feeds them into the
  Python plugin); the inference entry point is the core alone. The core therefore takes
  edge indices as **inputs**.
- Core-only ops: tensor indexing, `index_add_`, `nn.Embedding`, `nn.Linear`,
  `nn.ModuleList`, norms, `torch.where`. No PyG, no dict/object plumbing in `forward`.
- `radius_graph` (and later ASE neighbor lists) lives only in the adapter — training-side
  code that is never exported. Pin the PyG version: graph-building backends differ between
  versions (see [docs/data.md](docs/data.md)).
- Periodic adapters use the same core: with shift vectors `S`,
  `rel_pos = pos[idx_j] - pos[idx_i] + S @ cell` — after that, distances/directions/RBF are
  identical, and the core never learns that the system was periodic.

## 7. Trainer and callbacks

```python
class Callback:
    def on_epoch_end(self, epoch: int, metrics: dict, trainer) -> None: ...
```

The training loop is a thin shell: load data → forward/backward/step → at epoch end, build a
`metrics` dict and call every callback. Early stopping, TensorBoard, plotting, and
checkpointing are all callbacks; the loop never grows feature flags.

```
training/
├── trainer.py    # loop + checkpoint save/restore (model, standardizer, optimizer,
│                 #   scheduler, config snapshot, epoch, best val MAE)
├── metrics.py    # MetricsLogger: append row to CSV (source of truth); also writes
│                 #   the per-step lr table (cosine schedule changes every step)
└── callbacks.py  # EarlyStoppingCallback, TensorBoardCallback, PlotCallback
```

Loss composition: `loss = λ_E · MSE(energy_pred, y) [+ λ_F · MSE(forces_pred, forces)]`;
the force term engages iff the dataset has forces and the config sets `training.loss.forces`.
Losses use `reduction='sum'` and are divided once by the molecule count per batch
(`sum-then-divide` — keeps gradient scale batch-size independent and epoch metrics exact;
see [docs/training.md](docs/training.md)).

## 8. Config schema

```yaml
run_name: baseline
seed: 0

model:
  name: painn                      # key in MODELS registry
  num_message_passing_layers: 3
  num_features: 128
  num_rbf_features: 20
  num_unique_atoms: 100
  cutoff_dist: 5.0
  num_outputs: 1

data:
  name: qm9                        # key in DATASETS registry
  target: 7                        # QM9 property (U0)
  data_dir: data/
  splits: [110000, 10000, 10831]
  batch_size_train: 100
  batch_size_eval: 1000
  num_workers: 0

training:
  num_epochs: 1000
  optimizer:
    name: AdamW                  # any class in torch.optim (case-sensitive getattr)
    lr: 5.0e-4
    weight_decay: 0.01
  scheduler:
    name: CosineAnnealingLR      # any class in torch.optim.lr_scheduler (case-sensitive getattr)
  loss:
    energy: 1.0
    forces: null                    # set e.g. 0.1 to engage the force term
  early_stopping:
    patience: 30
    min_epochs: 1000

export:
  path: mliap                      # LAMMPS integration: the Python plugin path
```

Unknown keys in `model.*` / `data.*` are forwarded to the registered constructor — new models
and datasets add their own sections without touching the config loader. `optimizer.name` /
`scheduler.name` are looked up verbatim with `getattr(torch.optim, name)` /
`getattr(torch.optim.lr_scheduler, name)`, so they must match the class names exactly (case
sensitive): `AdamW`, `CosineAnnealingLR` — not `adamw` / `cosine`.

## 9. Directory tree

```
potlab/
├── config.py         # dataclass mirroring the YAML + loader + validation
├── registry.py       # MODELS / DATASETS dicts + decorators
├── data/
│   ├── base.py       # BaseDataModule + Standardizer
│   ├── transforms.py # GetTarget and other per-sample transforms
│   ├── qm9.py        # QM9DataModule + Qm9Standardizer (atomref-based)
│   └── vasp.py       # future: VaspDataModule (ASE + PBC, refs fitted by lstsq)
├── models/
│   ├── base.py       # BaseModel
│   └── painn/
│       ├── core.py   # PaiNNCore (graph-agnostic, no PyG)
│       └── model.py  # PaiNNModel adapter (radius_graph)
├── training/
│   ├── trainer.py
│   ├── metrics.py
│   └── callbacks.py
└── export/
    ├── lammps.py     # LammpsWrapper: core + standardizer -> physical (energy, forces)
    └── mliappy.py    # MliapPaiNN: the pair_style mliap unified glue (M5 Phase B)
```

## 10. What the model must never know about

Models are pure functions of `(z, pos, graph_indexes)` → energies. They never import, see,
or care about: standardization statistics, atom references, unit conversions, dataset names,
meV, checkpoints, or LAMMPS. Everything outside that contract belongs to the data layer
(standardizer), the trainer, or the export layer.

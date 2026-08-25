# potlab-ml — API Reference

The complete public API, organized by module. Each entry lists the full signature, then
`Parameters` / `Returns` / `Raises` in the format used by the generated API docs of
projects like e3nn and MACE, with a usage example for the key functions.

Private helpers (`_edge_geometry`, `_message_pass`, `_radius_graph`, `_grad_norm`, …) are
implementation detail and are named with a one-line note rather than documented in full.
For the *rules* behind these interfaces (what "energy" means, why the core is split from
the adapter) see [developer-guide.md](developer-guide.md) and [DESIGN.md](../DESIGN.md).

---

## `potlab.config`

### `Config`

```python
@dataclass
class Config:
    run_name: str = "baseline"
    seed: int = 0
    model: dict = field(default_factory=dict)     # open section → registry.MODELS[model.name](**model)
    data: dict = field(default_factory=dict)      # open section → registry.DATASETS[data.name](**data)
    training: dict = field(default_factory=dict)  # open section → Trainer / optimizer / scheduler
    export: dict = field(default_factory=dict)    # open section → export assembly
```

The four open dict sections forward their keys verbatim to the registered constructor, so
a new model/dataset adds its own keys without touching this class.

### `apply_overrides`

```python
def apply_overrides(raw: dict, overrides: list[str]) -> dict
```

**Parameters:**
- `raw (dict)` — the parsed YAML dict (mutated in place).
- `overrides (list[str])` — strings of the form `KEY=VALUE`, where `KEY` is a dotted path.

**Returns:**
- `dict` — `raw`, after applying every override.

**Raises:**
- `ValueError` — an override has no `=` separator.

Values are parsed with `yaml.safe_load`, so `"3"` → `int`, `"1.0e-3"` → `float`,
`"null"` → `None`.

### `load_config`

```python
def load_config(path: str | Path, overrides: list[str] | None = None) -> Config
```

**Parameters:**
- `path (str | Path)` — path to the YAML config file.
- `overrides (list[str] | None)` — dotted `KEY=VALUE` overrides, applied before the dataclass is built.

**Returns:**
- `Config` — the resolved config.

**Raises:**
- `TypeError` — an unknown top-level YAML key (keys must match the dataclass fields).

**Example:**

```python
from potlab.config import load_config

cfg = load_config("configs/default.yaml", overrides=["training.num_epochs=5"])
cfg.training["num_epochs"]   # 5
cfg.model["name"]            # "painn"
```

---

## `potlab.registry`

```python
MODELS: dict[str, type[BaseModel]] = {}      # name -> class
DATASETS: dict[str, type[BaseDataModule]] = {}
```

### `register_model`

```python
def register_model(name)
```

**Parameters:**
- `name (str)` — the key under which the class is stored in `MODELS`.

**Returns:**
- the decorator `deco(cls)`, which stores the class and returns it unchanged.

**Raises:**
- `ValueError` — `name` is already registered.

**Example:**

```python
from potlab.registry import register_model, MODELS

@register_model("painn")
class PaiNNModel(BaseModel):
    ...

assert MODELS["painn"] is PaiNNModel
```

### `register_dataset`

Identical to `register_model`, writing into `DATASETS` instead of `MODELS`.

---

## `potlab.data.base`

### `Standardizer`

Owns every target transformation; operates on labels, independent of the model.

```python
class Standardizer:
    def fit(self, train_data) -> None
    def transform(self, y: Tensor, z: Tensor, graph_indexes: Tensor) -> Tensor
    def inverse(self, energy_pred: Tensor, z: Tensor, graph_indexes: Tensor) -> Tensor
    def inverse_per_atom(self, contribs: Tensor, z: Tensor) -> Tensor
    def state_dict(self) -> dict
    def load_state_dict(self, state: dict) -> None
```

| Method | Effect |
|---|---|
| `fit(train_data)` | Compute all statistics from the **train** split only. |
| `transform(y, z, graph_indexes)` | Labels → standardized space (subtract refs, shift/scale). |
| `inverse(energy_pred, z, graph_indexes)` | Model energies → physical units, **per molecule**. |
| `inverse_per_atom(contribs, z)` | Standardized contributions → physical energies, **per atom** (the LAMMPS granularity). |
| `state_dict` / `load_state_dict` | Persist / restore the statistics (`mean`, `std`, `atom_refs`). |

`inverse` is literally `inverse_per_atom` broadcast over a molecule's atoms and summed —
one implementation, two granularities. `Trainer.compute_mae` uses `inverse`;
`LammpsWrapper` uses `inverse_per_atom`.

### `BaseDataModule`

```python
class BaseDataModule:
    def prepare_data(self) -> None
    def setup(self) -> None
    def train_dataloader(self) -> DataLoader
    def val_dataloader(self) -> DataLoader
    def test_dataloader(self) -> DataLoader
    def make_standardizer(self) -> Standardizer
    @property has_forces(self) -> bool
    @property num_targets(self) -> int
    @property energy_index(self) -> Optional[int]
    @property unit_conversion(self) -> Callable
```

| Member | Meaning |
|---|---|
| `prepare_data` | One-time download / raw parse (safe to call repeatedly). |
| `setup` | Shuffle + split into train/val/test. |
| `*_dataloader` | Batches under the [DESIGN.md §3](../DESIGN.md#3-data-contract-the-batch) contract. |
| `make_standardizer` | Build **and fit** the standardizer on the train split. |
| `has_forces` | Whether the dataset provides forces (drives the force-loss term). |
| `num_targets` | Target column count; must equal `model.num_outputs`. |
| `energy_index` | Which output column is the PES energy, or `None`. |
| `unit_conversion` | Display-only conversion (eV → meV). |

---

## `potlab.data.qm9`

### `QM9DataModule`

```python
@register_dataset("qm9")
class QM9DataModule(BaseDataModule):
    def __init__(self, target: int = 7, data_dir: str = "data/",
                 batch_size_train: int = 100, batch_size_eval: int = 1000,
                 num_workers: int = 0, splits=[110000, 10000, 10831],
                 seed: int = 0, subset_size: Optional[int] = None)
```

**Parameters:**
- `target (int)` — which QM9 property to predict (default 7 = U0).
- `data_dir (str)` — where QM9 is/will be downloaded (relative paths anchor to `ROOT`).
- `batch_size_train / batch_size_eval (int)` — loader batch sizes.
- `num_workers (int)` — dataloader workers.
- `splits (list[int] | list[float])` — `[train, val, test]` counts (all int) or proportions (all float).
- `seed (int)` — shuffle seed (reproducible split).
- `subset_size (int | None)` — debug: cap the dataset to N molecules.

**Properties:** `has_forces → False`; `num_targets → 1`; `energy_index → 0` iff
`target in PES_ENERGY_TARGETS` else `None`.

**Example:**

```python
from potlab.data.qm9 import QM9DataModule

dm = QM9DataModule(target=7, subset_size=1000, splits=[800, 100, 100], seed=0)
dm.prepare_data()   # downloads QM9 on first call
dm.setup()          # shuffle + split
dm.energy_index     # 0
```

### `Qm9Standardizer`

```python
class Qm9Standardizer(Standardizer):
    def fit(self, train_data: QM9DataModule) -> None
    # transform / inverse / inverse_per_atom / state_dict / load_state_dict as in Standardizer
```

Three-step pipeline, mirrored by `inverse`: subtract QM9's per-element `atomref` table →
divide by atom count → shift/scale with train-set mean/std.

### Constants

```python
PES_ENERGY_TARGETS = {7}                        # QM9 properties whose gradient IS a force (U0 only)
NO_UNIT_CONVERSION = {0, 1, 5, 11, 16, 17, 18}  # non-energy properties keep native units
```

### Helpers

```python
def _validate_splits(splits, n_mols) -> list[int]      # all-int or all-float, else ValueError
def _sum_per_graph(values, graph_indexes, n_graphs) -> Tensor
def _num_atoms_per_graph(graph_indexes) -> Tensor
```

---

## `potlab.models.base`

### `BaseModel`

```python
class BaseModel(nn.Module):
    def energy(self, z, pos, graph_indexes) -> Tensor
    def energy_and_forces(self, z, pos, graph_indexes) -> tuple[Tensor, Tensor]
    def atomic_contributions(self, z, pos, graph_indexes) -> Tensor
```

| Method | Signature → result |
|---|---|
| `energy(z, pos, graph_indexes)` | `[N_atoms]/[N_atoms,3]/[N_atoms]` → `[N_graphs, num_outputs]` standardized predictions (all columns). |
| `energy_and_forces(z, pos, graph_indexes)` | total energy + per-atom forces `[N_atoms, 3]` via autograd; differentiates **column 0** only. |
| `atomic_contributions(z, pos, graph_indexes)` | `[N_atoms, num_outputs]` per-atom decomposition (optional; may raise `NotImplementedError`). |

Forces come from `-torch.autograd.grad(E[:, 0].sum(), pos)` for any differentiable model.

---

## `potlab.models.painn.core`

### `PaiNNCore`

```python
class PaiNNCore(nn.Module):
    def __init__(self, num_message_passing_layers=3, num_features=128, num_outputs=1,
                 num_rbf_features=20, num_unique_atoms=100, cutoff_dist=5.0)
    def forward(self, z, pos, idx_i, idx_j) -> Tensor
    def forward_with_edges(self, z, rel_pos, idx_i, idx_j) -> Tensor
```

**Parameters (`__init__`):**
- `num_message_passing_layers (int)` — number of (message + update) rounds.
- `num_features (int)` — scalar feature width.
- `num_outputs (int)` — readout width (see `energy_index`).
- `num_rbf_features (int)` — radial basis size.
- `num_unique_atoms (int)` — embedding table width is `num_unique_atoms + 1` (row 0 = padding).
- `cutoff_dist (float)` — neighbor cutoff in Å.

| Method | Effect |
|---|---|
| `forward(z, pos, idx_i, idx_j)` | Builds `rel_pos = pos[idx_j] - pos[idx_i]`, then the full pass → `[N_atoms, num_outputs]`. |
| `forward_with_edges(z, rel_pos, idx_i, idx_j)` | Same math, but the caller supplies per-edge displacements directly — the LAMMPS inference entry (displacements are already minimum-image corrected). |

Private split: `_edge_geometry(rel_pos) → (rel_dir, rel_dist_cut, rbf_features)` and
`_message_pass(z, idx_i, idx_j, rel_dir, rel_dist_cut, rbf_features) → Tensor`.

### `build_readout_network`

```python
def build_readout_network(num_in_features, num_out_features=1, num_layers=2,
                          activation=nn.SiLU) -> nn.Sequential
```

Per-atom MLP compressing `num_features → num_outputs`; hidden widths halve each layer, no
activation after the last `Linear`.

### Sub-modules

```python
class SinusoidalRBFLayer(nn.Module):  __init__(self, num_basis=20, cutoff_dist=5.0)  # sin(freq*d)/d
class CosineCutoff(nn.Module):        __init__(self, cutoff_dist=5.0)                # 0..1 smooth switch
class PaiNNMessageBlock(nn.Module):   __init__(self, num_features=128, num_rbf_features=20)
class PaiNNUpdateBlock(nn.Module):    __init__(self, num_features=128)
```

---

## `potlab.models.painn.model`

### `PaiNNModel`

```python
@register_model("painn")
class PaiNNModel(BaseModel):
    def __init__(self, num_message_passing_layers=3, num_features=128, num_outputs=1,
                 num_rbf_features=20, num_unique_atoms=100, cutoff_dist=5.0)
    def energy(self, z, pos, graph_indexes) -> Tensor
    def atomic_contributions(self, z, pos, graph_indexes) -> Tensor
```

Same `__init__` parameters as `PaiNNCore`. `energy` mean-pools the core's per-atom
contributions into `[N_graphs, num_outputs]` (standardized-space predictions); private
`_radius_graph(pos, batch)` builds the PyG neighbor list (training-side only).

---

## `potlab.training`

### `Trainer`

```python
class Trainer:
    def __init__(self, model, data_module, standardizer, run_dir, config, resume=False)
    def fit(self) -> None
    def compute_mae(self, loader) -> float
```

**Parameters (`__init__`):**
- `model (BaseModel)` — the registry-built model.
- `data_module (BaseDataModule)` — provides train/val loaders.
- `standardizer (Standardizer)` — fitted on the train split.
- `run_dir (Path)` — the run directory (created via `make_run_dir`).
- `config (Config)` — the resolved config.
- `resume (bool)` — restore `latest.pt` and continue (else fresh start + config snapshot).

| Method | Effect |
|---|---|
| `fit()` | Epoch loop: standardized loss (`sum-then-divide`), metrics dict, every callback, checkpoints. |
| `compute_mae(loader)` | Sum-then-divide MAE in **physical** units (via `standardizer.inverse`). |

Private: `_build_optimizer_scheduler(training)`, `_load_checkpoint(path)`,
`_save_checkpoint(filename, epoch)`, `_dump_config_snapshot()`, `_grad_norm()`,
`_current_lr()`, `_close()`.

### Callbacks

```python
class Callback:
    def on_epoch_end(self, epoch, metrics, trainer) -> None

class EarlyStoppingCallback(Callback):  __init__(self, patience=30, min_epochs=1000)
class TensorBoardCallback(Callback):    __init__(self, run_dir);  close(self)
class PlotCallback(Callback):           __init__(self, run_dir, every_n_epochs=10)
```

### `MetricsLogger`

```python
class MetricsLogger:
    def __init__(self, run_dir)
    def log_epoch(self, epoch, train_loss, val_mae, lr, epoch_time, grad_norm) -> None
    def log_lr_step(self, step, lr) -> None
    def close(self) -> None
```

Appends one row per epoch to `metrics.csv` (the source of truth) and per-step lr rows to
`lr_steps.csv`; every write is flushed so a killed process keeps its rows.

### `make_run_dir`

```python
def make_run_dir(run_name, root=RUNS_DIR) -> Path
```

Creates `root/run_name/checkpoints/` and `root/run_name/plots/`, returns the run dir.

---

## `potlab.export.lammps`

### `LammpsWrapper`

```python
class LammpsWrapper(nn.Module):
    def __init__(self, core, standardizer, energy_index=0)
    def energy(self, z, pos, idx_i, idx_j) -> Tensor
    def forces(self, z, pos, idx_i, idx_j) -> Tensor
    def energy_and_forces(self, z, pos, idx_i, idx_j) -> tuple[Tensor, Tensor]
```

**Parameters (`__init__`):**
- `core (nn.Module)` — the graph-agnostic core (`PaiNNCore`), not the full model.
- `standardizer (Standardizer)` — fitted; provides `inverse_per_atom`.
- `energy_index (int)` — which output column is the physical energy (the dataset's declared value).

| Method | Effect |
|---|---|
| `energy(z, pos, idx_i, idx_j)` | Scalar (0-dim) physical total: `inverse_per_atom(...)[:, energy_index].sum()`. |
| `forces(z, pos, idx_i, idx_j)` | `[N_atoms, 3]` physical forces via autograd. |
| `energy_and_forces(...)` | Both in one forward + one backward. |

**Example:**

```python
from potlab.export.lammps import LammpsWrapper

wrapper = LammpsWrapper(model.painn_core, standardizer, energy_index=0)
total = wrapper.energy(z, pos, idx_i, idx_j)   # scalar total energy
```

---

## `potlab.export.mliappy`

### `ELEMENT_Z`

```python
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8}
```

Element name → atomic number. The set a QM9-trained PaiNN can know about; unknown names
fail loudly at pickle build time. Extend it when training on other elements.

### `MliapPaiNN`

```python
class MliapPaiNN:
    def __init__(self, wrapper, element_types)
    def compute_gradients(self, data)
    def compute_descriptors(self, data)
    def compute_forces(self, data)
    def pickle(self, filename)
```

**Parameters (`__init__`):**
- `wrapper (LammpsWrapper)` — the wrapped core + standardizer.
- `element_types (list[str])` — element names in LAMMPS `pair_coeff` order.

`compute_gradients(data)` is the whole model: it translates the LAMMPS data object
(`elems`, `pair_i`, `pair_j`, `rij`) into the wrapper's interface, writes `data.energy` /
`data.eatoms`, and scatters forces via the C++ `update_pair_forces`. The other two hooks
are unused. `pickle(filename)` serializes the object for `pair_style mliap unified`.

### `build_from_run`

```python
def build_from_run(run_dir, element_types, device="cpu") -> MliapPaiNN
```

**Parameters:**
- `run_dir (str | Path)` — the run name under `runs/` (or an absolute path).
- `element_types (list[str])` — LAMMPS atom types in `pair_coeff` order.
- `device (str)` — device for the model (default `cpu`).

**Returns:**
- `MliapPaiNN` — a picklable model assembled from `config.yaml` + `best.pt`.

**Raises:**
- `ValueError` — the run trains a non-energy target (`energy_index is None`).

**Example:**

```python
from potlab.export.mliappy import build_from_run

model = build_from_run("baseline", ["C", "H"])   # type 1=C, type 2=H
model.pickle("model.pkl")
```

---

## `scripts` (CLI)

```bash
# scripts/train.py
--config PATH        # default configs/default.yaml
-o/--override K=V    # repeatable; dotted paths (training.optimizer.lr=1.0e-3)
--resume             # continue the run named by run_name from latest.pt
--warm-start PATH    # new run, weights initialized from PATH
--subset-size N      # debug: cap the dataset to N molecules

# scripts/make_mliap_pickle.py
--run NAME           # run under runs/; loads checkpoints/best.pt
--elements C,H       # LAMMPS atom types in pair_coeff order
--out FILE.pkl       # output pickle for pair_style mliap unified
--device cpu         # default cpu

# scripts/export_lammps.py
--run NAME           # default baseline; prints energy parity + force check
```

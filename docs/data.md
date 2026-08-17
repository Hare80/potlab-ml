# potlab-ml — Data Guide

How datasets plug into the framework, and how to handle the two reference cases:
QM9 (molecular properties) and periodic VASP trajectories.

The data layer contract lives in [DESIGN.md](../DESIGN.md) (§3 batch contract, §4
`BaseDataModule`, §5 `Standardizer`). This guide covers the *how*.

## QM9: porting the existing module

The original implementation is the specification; map it 1:1 into the new structure,
keeping behavior bit-identical (same shuffle seed, same transforms, same statistics).

| Original code (`02456_painn_project-main`) | New home in potlab-ml | Notes |
|---|---|---|
| `GetTarget` (`src/data/qm9.py`) | `src/potlab/data/transforms.py` | Generic per-sample transform; keep the column-selection behavior, drop the dead None-branch |
| `QM9DataModule` | `src/potlab/data/qm9.py` | Implement `BaseDataModule`; drop the `pl.LightningDataModule` dependency |
| `unit_conversion` dict | `qm9.py` property | Display only (eV → meV for energy targets) |
| `get_target_stats` | `Qm9Standardizer.fit/transform` | See below |
| `AtomwisePostProcessing` | removed — replaced by `Qm9Standardizer.inverse` | The standardizer owns the inverse transform now |
| split logic (`setup`) | `qm9.py` private method | Add the missing validation: `sum(splits)` must equal the dataset size (130 831) |

### QM9 pipeline, end to end

1. `QM9(root, transform=GetTarget(target))` — labels trimmed from `[1, 19]` to `[1, 1]`.
2. Seeded shuffle (`np.random.default_rng(seed)` + `rng.permutation`) — **mandatory**: the
   raw ordering is sorted by molecular size; splitting without shuffling gives
   train/val/test different size distributions.
3. Split `[110000, 10000, 10831]` (they sum exactly to 130 831).
4. `Qm9Standardizer.fit(train)`:
   - references = the dataset's own `atomref` table (shipped with QM9, indexed by atomic number);
   - `y' = (y − Σ_z atomref(z)) / n_atoms`;
   - mean/std of `y'` over the training set.
5. Training labels = `(y' − mean) / std`; `inverse` = `(Σ_i c_i · std + mean) · n_atoms + Σ_z atomref(z)`.

Why subtract references at all: molecular total energies are large offsets (≈ −1000 eV
for methane) dominated by per-atom terms; the chemically interesting correction is a few eV.
Removing the reference part shrinks the learning task by orders of magnitude. This step
applies to **labels** — it never depends on whether the model outputs per-atom contributions.

## Periodic systems: VASP via ASE

### Reading VASP output

Prefer `vasprun.xml` (`OUTCAR` works but is slower):

```python
from ase.io import read
frames = read("vasprun.xml", index=":")     # all frames of a trajectory

z    = atoms.get_atomic_numbers()           # [N]
pos  = atoms.get_positions()                # [N, 3] Cartesian
cell = atoms.cell[:]                        # [3, 3], may differ per frame (NPT!)
pbc  = atoms.pbc                            # per-axis periodicity
E    = atoms.get_potential_energy()         # eV
F    = atoms.get_forces()                   # eV/Å, [N, 3]
```

Gotchas:

- **Wrap first**: VASP may place atoms outside the cell — `atoms.wrap()`.
- **Absolute energies are meaningless**: VASP totals depend on pseudopotentials and cutoff
  settings. This is exactly why the standardizer fits per-element references (below) — the
  fitted refs absorb the arbitrary offsets, and same-settings runs become comparable.
- **Units**: ASE hands you eV and eV/Å — record them in the dataset docstring and convert
  at display time only.

### Neighbor lists under periodic boundary conditions

`torch_geometric.nn.radius_graph` has **no PBC support**. For periodic data, build neighbors
with ASE instead:

```python
from ase.neighborlist import primitive_neighbor_list
i, j, S = primitive_neighbor_list("ijS", atoms, cutoff=5.0)
# i, j: receiver / sender indices per edge, [E]
# S:    integer shift vectors [E, 3] — which periodic image j lives in
```

The only difference from the molecular case is `S`. Real edge vectors become

```
rel_pos  = pos[j] − pos[i] + S @ cell      # cell rows are the lattice vectors
rel_dist = ||rel_pos||                     # -> rel_dir, RBF, cosine cutoff as usual
```

The model **core never sees `S`** — the adapter converts shifts into correct positions and
the core consumes `(idx_i, idx_j)` and positions exactly as for QM9. Two physical notes:

- An atom can be a neighbor of *its own periodic image* (distance = a lattice translation,
  never zero) — no self-loop / 0-distance degeneracy.
- Keep the cutoff below half the shortest lattice vector so edges never span more than one
  image (ASE handles multi-image correctly anyway; this is a sanity rule, not a limit).

### Batching frames with different cells

Cells vary per frame, so edges cannot be built once over a whole batch. Two options:

**A. Precompute at load time (recommended to start).** Build `edge_index` + `S` per frame
once in `setup()`, store them in the `Data` objects. The trainer and loss see nothing new;
the cost is fixed edges for the epoch (fine for small datasets). This is the fastest way to
a working pipeline.

**B. Build on the fly in the model adapter.** Loop over the frames of the batch (grouped by
`batch`), run the ASE neighbor list per frame, and merge into one big edge table by adding
node offsets (`+ N_frames_before`) to each frame's indices. The core stays untouched.
Choose B when you need to change the cutoff without reprocessing, or for large datasets.

### Trajectory-aware splitting (critical)

Adjacent MD frames are highly correlated. Never split trajectories by shuffling frames —
the validation frames would be near-duplicates of training frames (leakage). Split **by
simulation run**: e.g. 10 independent runs → 8 train / 1 val / 1 test, whole runs at a time.

### The standardizer for VASP data

`VaspStandardizer.fit(train)`:
1. Fit per-element reference energies by linear regression on the training set —
   `np.linalg.lstsq(X, E)` where `X[m, Z]` counts element `Z` in frame `m`. (For
   single-composition sets this degenerates to subtracting the per-atom mean energy —
   still worth doing.)
2. Mean/std of the reference-subtracted energies (optionally per-atom).
3. `inverse` adds `Σ_z ref(z)` back — exactly as for QM9, so the exported model speaks
   absolute VASP-consistent energies.

This "fit your own references" step is standard MLIP practice (MD17, NequIP, MACE all do
some form of it); datasets shipping precomputed refs (QM9) are the exception, not the rule.

### PyG version note

Graph-building backends differ between PyG versions (2.6.x uses `torch-cluster`, 2.8.x uses
`pyg-lib`; padding behavior around `max_num_neighbors` is not identical). Pin
`torch-geometric` in `environment.yml` and keep graph building confined to adapters so the
rest of the code never notices.

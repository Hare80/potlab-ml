# potlab-ml — LAMMPS Integration Guide (MLIAP-Python)

The project uses exactly one integration path: **`pair_style mliap` with a Python
plugin** (`mliappy`). There is no export artifact — the trained model + standardizer
run inside LAMMPS through its Python coupling.

TorchScript was the original primary path (`torch.jit.script` + ML-PAINN + libtorch).
It was dropped at M4: `torch.jit.script` is deprecated upstream (torch >= 2.9) and
scheduled for removal, and the MLIAP-Python path needs none of it — the same
model-agnostic design that made it the "escape hatch" makes it the whole story.

| | MLIAP-Python (`mliappy`) |
|---|---|
| LAMMPS pair style | `pair_style mliap model mliappy` |
| Model runs in | Python / PyTorch (LAMMPS Python coupling) |
| Export step | none — training code runs directly |
| Model restrictions | none (any `BaseModel`) |
| PBC neighbors / virial | handled by LAMMPS |
| Speed | Python boundary every step (fine for prototyping / small MD) |
| Build requirements | Python-enabled LAMMPS + plugin on `PYTHONPATH` |

## Why the core/adapter split exists

LAMMPS builds the neighbor list itself (with PBC and virial support). The plugin
therefore hands `(z, pos, idx_i, idx_j)` to the model — which is exactly the
`PaiNNCore` interface from [DESIGN.md](../DESIGN.md#6-painn-core--graph-builder-split).
The graph-builder adapter (`radius_graph`, ASE) is training-side code and never runs
in LAMMPS.

## Bake the standardizer into the plugin

LAMMPS needs **absolute energies**, not standardized residuals. The plugin wraps:

```
wrapped.forward(z, pos, idx_i, idx_j):
    c = core.forward(z, pos, idx_i, idx_j)          # [N, out] standardized residuals
    e_per_atom = standardizer.inverse(c, z, ...)    # physical units per molecule
    return e_per_atom.sum()                         # scalar total energy
```

Forces: `-torch.autograd.grad(E, pos)` — the same math as
`BaseModel.energy_and_forces`, applied to the wrapped energy.

## Plugin structure

The unified model lives in `potlab/export/mliappy.py` (`MliapPaiNN`): the pickle
loaded by `pair_style mliap unified <file> 0`. `scripts/make_mliap_pickle.py`
builds it from a trained run; `examples/lammps/methane/` is the turnkey smoke
(data + input + runbook). The wrapper math stays in `potlab/export/lammps.py` -
the glue only translates the coupling's data object (element indices, pair
displacements) into the wrapper's interface and writes energy/forces back.

## Verification

The wrapper's energies/forces must match the eager training pipeline (standardizer
included) to `1e-6` — the same definition as `Trainer.compute_mae`, so a wrapper
that disagrees with the validation MAE fails this check. Achieved end to end in
M5: `tests/test_export.py` pins the wrapper/glue math (42-test suite), and the
methane smoke ran a 100-step NVE MD whose step-0 `pe` matches the Python-side
wrapper energy to float32 precision (1.6e-5 eV on -1103 eV), with `etotal`
conserved across the run.

## LAMMPS side

- Build LAMMPS with Python enabled (`-D BUILD_SHARED_LIBS=on` + the Python package);
  the plugin must be importable from LAMMPS' embedded Python (conda env on
  `PYTHONPATH`).
- Read the MLIAP-Python docs for the exact plugin interface (method names, unit
  conventions, atom-type indexing) and verify against the version you actually build —
  the interface has evolved between versions.
- Windows note: building LAMMPS on Windows is painful; use **WSL2 or a Linux box** for
  the LAMMPS part. Training stays where it is.

# potlab-ml — Export Guide (LAMMPS)

Two integration paths are supported. **TorchScript + ML-PAINN is primary** (production
performance for PaiNN, no Python in the MD loop); **MLIAP-Python is the generic
alternative** (model-agnostic, no export step, slower). Both are worth understanding
because they make opposite trade-offs.

| | TorchScript + ML-PAINN | MLIAP-Python (`mliappy`) |
|---|---|---|
| LAMMPS pair style | `pair_style painn` (ML-PAINN package) | `pair_style mliap model mliappy` |
| Model runs in | C++ / libtorch | Python / PyTorch (LAMMPS Python coupling) |
| Export step | yes (`torch.jit.script` + `torch.jit.save`) | none — training code runs directly |
| Model restrictions | TorchScript-safe core required | none (any PyTorch model) |
| PBC neighbors / virial | handled by LAMMPS | handled by LAMMPS |
| Speed | high (no Python per step) | lower (Python boundary every step) |
| Works for future models | per-model pair style needed | any `BaseModel`, one plugin |
| Build requirements | LAMMPS + ML-PAINN + libtorch (version-matched) | Python-enabled LAMMPS + plugin on `PYTHONPATH` |

## Path A (primary): TorchScript + ML-PAINN

### Why the core/adapter split exists

LAMMPS builds the neighbor list itself (with PBC and virial support). The exported
artifact therefore must compute energies from `(z, pos, idx_i, idx_j)` — which is exactly
the `PaiNNCore` interface from [DESIGN.md](../DESIGN.md#6-painn-core--graph-builder-split).
The graph-builder adapter (`radius_graph`, ASE) is training-side code and is never exported.

### Bake the standardizer into the export

LAMMPS needs **absolute energies**, not standardized residuals. Wrap before scripting:

```
exported.forward(z, pos, idx_i, idx_j):
    c = core.forward(z, pos, idx_i, idx_j)          # [N, out] standardized residuals
    e_per_atom = c * std + mean + atomref(z)        # standardizer inverse, per atom
    return e_per_atom.sum()                         # scalar total energy
```

The wrapper is a thin `nn.Module` holding the core plus frozen buffers (std, mean,
atom-ref table as a frozen `nn.Embedding`). Everything in it must be scriptable —
this is why the M2 smoke test (`torch.jit.script`) runs long before export day.

### Export and verify

```python
scripted = torch.jit.script(exported_wrapper)
torch.jit.save(scripted, "model.pt")
```

Verification (also in the M4 test checklist):

- **Parity**: scripted energies/forces must match the eager training pipeline (standardizer
  included) to `1e-6` — if they don't, the export disagrees with your validation MAE.
- **Forces**: compute `-torch.autograd.grad(E, pos)` on the scripted module; compare with
  the training-time `energy_and_forces`.

### LAMMPS side

- Build LAMMPS with the **ML-PAINN** package and a **libtorch matching the torch version
  used for training** (cu126 wheels ↔ cu126 libtorch) — version mismatch is the #1 failure
  mode here.
- **Read the ML-PAINN README for the exact model interface it expects** (module method
  names, unit conventions, atom-type indexing) and adapt the wrapper above to it — the
  interface has evolved between versions, so verify against the docs of the version you
  actually build.
- Windows note: building this stack on Windows is painful; use **WSL2 or a Linux box** for
  the LAMMPS part. Training stays where it is.

## Path B (generic): MLIAP-Python

The escape hatch for future models: LAMMPS' `pair_style mliap` with a Python plugin. The
plugin is a small file on `PYTHONPATH` that wraps your trained `BaseModel`:

- LAMMPS passes positions, atom types, and its own neighbor list into the plugin;
- the plugin calls `model.energy_and_forces(...)` with those inputs (no PyG involved —
  LAMMPS already built the graph) and returns energy + forces;
- no TorchScript, no per-model pair style — any PyTorch model plugs in unchanged.

Requirements: a Python-enabled LAMMPS build (`-D BUILD_SHARED_LIBS=on` + Python package),
the plugin file following the MLIAP-Python conventions (model loading, unit and
energy-convention declarations), and the standardizer baked into the wrapper the same way
as Path A (LAMMPS still needs absolute energies).

Document the plugin template in `potlab/export/lammps.py` alongside the TorchScript
exporter, so the two paths share the standardizer-baking code and differ only in the
artifact they produce (`.pt` file vs. Python plugin).

## Choosing a path

- **PaiNN, long production MD**: Path A.
- **Any future model, quick integration**: Path B.
- Both export from the same trained checkpoint; the standardizer baking is shared code.

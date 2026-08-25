"""mliappy glue: run a trained potlab wrapper inside pair_style mliap unified.

M5 Phase B. This class implements the MLIAPUnified contract
(compute_gradients / compute_descriptors / compute_forces) WITHOUT
importing LAMMPS: the unified coupling unpickles an object and calls its
methods duck-typed, and potlab's test suite must stay runnable without a
LAMMPS install - so the lammps.mliap ABC is deliberately not imported.

The unified data object (probed from the C++ coupling; see
src/ML-IAP/mliap_unified_couple.pyx in the LAMMPS source) provides:

  pair_i [npairs]      center atom of each pair (local list index)
  pair_j [npairs]      neighbor of each pair (extended index, ghosts incl.)
  rij    [npairs, 3]   displacement x[j] - x[i], minimum-image applied
  elems  [ntotal]      element index of every atom (ghosts included)
  iatoms [nlistatoms]  the local atom list (order of eatoms writes)
  f      [ntotal, 3]   WRITE-THROUGH view of the LAMMPS force array
  eflag  int           whether per-atom energies are requested
  energy (write-only)  scalar total energy
  eatoms (write-only)  per-atom energies, nlistatoms entries

The energy path: PaiNNCore.forward_with_edges (public - the core takes
edge displacements directly) -> standardizer.inverse_per_atom -> physical
per-atom energies; the total goes to data.energy and the per-atom values
to data.eatoms. The force path: autograd on the total w.r.t. rij, handed
to the C++ update_pair_forces - the pair interface IS the chain rule
(f[i] += g; f[j] -= g over the full pair list, ghost forces returned by
reverse communication), so no manual scatter is needed and the many-body
model never invents a pair decomposition.
"""

import numpy as np
import torch

# The element set a PaiNN-on-QM9 checkpoint can know about. Extend when
# training on other elements - unknown names fail loudly at model build
# time (in the LAMMPS input script), not silently at MD time.
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8}


class MliapPaiNN:
    """The mliappy unified model for a trained LammpsWrapper (PaiNN).

    Built at pickle time in the LAMMPS input script's python block::

        wrapper = LammpsWrapper(model.painn_core, standardizer)
        model = MliapPaiNN(wrapper, ["H", "C"])
        model.pickle("paiNN.pkl")

    then ``pair_style mliap unified paiNN.pkl 0`` loads it.
    """

    def __init__(self, wrapper, element_types):
        self.wrapper = wrapper
        self.element_types = list(element_types)
        # The unified coupling's cutoff is 2.0 * rcutfac (its dummy
        # descriptor uses radelem = 1), so derive rcutfac from the
        # core's real cutoff - the neighbor list must reach exactly
        # as far as the model looks.
        self.rcutfac = wrapper.core.cutoff_dist / 2.0
        # Consistency-check metadata read by the C++ side (there are no
        # real descriptors in this model; the values mirror the shipped
        # unified examples).
        self.ndescriptors = 1
        self.nparams = 3

        unknown = sorted(set(self.element_types) - set(ELEMENT_Z))
        if unknown:
            raise ValueError(
                f"Unknown element names {unknown} - ELEMENT_Z covers "
                f"{sorted(ELEMENT_Z)}."
            )
        # Element index -> atomic number, one entry per element type in
        # the order LAMMPS passes them (pair_coeff).
        self._z_by_elem = torch.tensor(
            [ELEMENT_Z[e] for e in self.element_types], dtype=torch.long
        )
        self._device = next(wrapper.core.parameters()).device

    def _as_long(self, array):
        return torch.from_numpy(np.asarray(array, dtype=np.int64)).to(self._device)

    def compute_gradients(self, data):
        """The whole model: energy into data.energy/eatoms, forces into data.f."""
        z = self._z_by_elem[np.asarray(data.elems)]  # [ntotal] atomic numbers
        idx_i = self._as_long(data.pair_i)
        idx_j = self._as_long(data.pair_j)
        # rij IS the edge geometry (same convention as the core: x[j]-x[i]).
        # requires_grad: the force path differentiates the total energy
        # w.r.t. these displacements, then scatters per atom. The dtype
        # follows the model's parameters - never hardcoded (the tests
        # run the core in float64).
        model_dtype = next(self.wrapper.core.parameters()).dtype
        rij = (
            torch.from_numpy(np.asarray(data.rij))
            .to(device=self._device, dtype=model_dtype)
            .requires_grad_(True)
        )

        contribs = self.wrapper.core.forward_with_edges(z, rij, idx_i, idx_j)
        e_phys = self.wrapper.standardizer.inverse_per_atom(contribs, z)

        local = self._as_long(data.iatoms)
        e_local = e_phys[local, self.wrapper.energy_index]  # [nlistatoms]
        total = e_local.sum()
        data.energy = float(total.detach().cpu())
        if data.eflag:
            # The Cython setters require float64 buffers regardless of
            # the model's dtype (the C++ side is double everywhere).
            data.eatoms = e_local.detach().cpu().numpy().astype(np.float64)

        # Forces: the full chain rule. F_k = sum(+g) over the pairs whose
        # CENTER is k, minus sum(+g) over the pairs whose NEIGHBOR is k
        # (rij = x[j] - x[i] contributes dE/dx_i = -g and dE/dx_j = +g,
        # F = -dE/dx). That is exactly what the C++ update_pair_forces
        # does: f[i] += g; f[j] -= g - including ghost atoms, whose
        # forces LAMMPS' reverse communication sends back to their
        # owners. No manual scatter: the pair interface IS the chain
        # rule here.
        grad = torch.autograd.grad(total, rij)[0]  # [npairs, 3]
        data.update_pair_forces(grad.double().cpu().numpy())

    def compute_descriptors(self, data):
        """Descriptor-side hook: unused - this model computes everything itself."""

    def compute_forces(self, data):
        """Descriptor-side hook: unused - forces are written in compute_gradients."""

    def pickle(self, filename):
        """Save the model for ``pair_style mliap unified <filename> 0``.

        Mirrors MLIAPUnified.pickle: the whole object (wrapper weights +
        standardizer statistics) travels in the file.
        """
        import pickle

        with open(filename, "wb") as fp:
            pickle.dump(self, fp)


def build_from_run(run_dir, element_types, device="cpu"):
    """Assemble a picklable MliapPaiNN from a trained run directory.

    Mirrors scripts/export_lammps.py's assembly: the run's config
    snapshot + best checkpoint -> registry model -> LammpsWrapper ->
    MliapPaiNN. Heavy imports stay inside the function: the embedded
    python imports this module when LAMMPS unpickles the model, and it
    must not drag PyG into LAMMPS startup.
    """
    from pathlib import Path

    import torch

    from potlab import ROOT
    import potlab.config as config
    import potlab.models.painn.model  # side effect: registers "painn"
    import potlab.registry as registry
    from potlab.data.qm9 import QM9DataModule, Qm9Standardizer
    from potlab.export.lammps import LammpsWrapper

    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / "runs" / run_dir
    checkpoint = torch.load(
        run_dir / "checkpoints" / "best.pt", map_location=device
    )
    config_data = config.load_config(run_dir / "config.yaml")

    model_cfg = dict(config_data.model)
    model_name = model_cfg.pop("name")
    model = registry.MODELS[model_name](**model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    standardizer = Qm9Standardizer()
    standardizer.load_state_dict(checkpoint["standardizer"])
    # The dataset declares which column is the PES energy (constructor
    # only - no data touched). A non-energy target has nothing LAMMPS
    # can integrate.
    data_cfg = dict(config_data.data)
    energy_index = QM9DataModule(target=data_cfg["target"]).energy_index
    if energy_index is None:
        raise ValueError(
            f"Run {run_dir.name} trains a non-energy target "
            f"(data.target={data_cfg['target']}): LAMMPS integrates "
            "forces, so only PES-energy targets can be exported."
        )
    wrapper = LammpsWrapper(
        model.painn_core, standardizer, energy_index=energy_index
    )
    return MliapPaiNN(wrapper, element_types)

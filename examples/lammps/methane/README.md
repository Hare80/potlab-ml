# Methane MD with a PaiNN model (pair_style mliap unified)

Turnkey M5 Phase B smoke: a trained potlab run drives a short methane
NVE MD inside LAMMPS via the mliappy unified coupling.

## Prerequisites

- The self-built LAMMPS with ML-IAP python support:
  `D:\Program Files\LAMMPS-64bit-Python-MLIAP-4Jul2026` (see the build
  notes in the project history; `bin\run-env.bat` sets up the runtime).
- A trained run under `runs/<name>` with best.pt + config.yaml (any PES
  target; the smoke_m25 debug run works mechanically).

## Step 1 - build the model pickle (conda pytorch env)

```bash
python scripts/make_mliap_pickle.py --run smoke_m25 --elements C,H --out methane_painn.pkl
```

`--elements` declares what the LAMMPS atom types mean, in type order:
type 1 = C, type 2 = H. It must match the `pair_coeff * *` line in
`in.methane.lmp` exactly - the glue translates LAMMPS' element indices
into atomic numbers with this list.

## Step 2 - run the MD (cmd)

```bat
call "D:\Program Files\LAMMPS-64bit-Python-MLIAP-4Jul2026\bin\run-env.bat"
set PYTHONPATH=D:\Documents\Codes\potlab-ml;D:\Documents\Codes\lammps\python
cd /d D:\Documents\Codes\potlab-ml\examples\lammps\methane
lmp.exe -in in.methane.lmp
```

`PYTHONPATH` carries two things into LAMMPS' embedded Python (the conda
pytorch env's 3.13): the repo root, so potlab imports when the model is
unpickled, and the LAMMPS source tree's `python/` directory, which the
built-in mliap coupling module itself imports (`lammps` package).

## Acceptance check

The step-0 `pe` printed in the thermo output is the model's total
energy on the initial geometry. Compare it with the Python side:

```bash
python scripts/export_lammps.py --run smoke_m25   # wrapper vs pipeline parity
```

plus a direct evaluation of `LammpsWrapper.energy` on the coordinates
in `methane.data` - the two must agree to the model's precision
(1e-6 relative in float64). The MD itself completes 100 steps of NVE.

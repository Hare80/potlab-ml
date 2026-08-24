"""LAMMPS export: the MLIAP-Python wrapper (M5).

``lammps.py`` holds LammpsWrapper - a trained core + its standardizer
behind a ``(z, pos, idx_i, idx_j)`` interface, pure PyTorch. The
LAMMPS-side plugin glue (Phase B, mliappy conventions) is a separate
file and never imports into this package's math.
"""

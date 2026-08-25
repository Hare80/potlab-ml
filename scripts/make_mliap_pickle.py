"""Build the mliappy pickle a LAMMPS run consumes (M5 Phase B).

Assembles a trained run into a MliapPaiNN and pickles it; the LAMMPS
input script then loads it with ``pair_style mliap unified <file> 0``.
Run this from the conda pytorch env (torch + potlab live there); the
LAMMPS side only needs the file plus potlab on its embedded python's
PYTHONPATH to unpickle it.

The --elements list does NOT configure the model - it declares what the
LAMMPS input script's atom types mean (the same list that will appear
in ``pair_coeff * * C H``). It must match the input script's type
order: type 1 = first element, type 2 = second, ...

Usage:
  python scripts/make_mliap_pickle.py --run smoke_m25 --elements C,H --out methane_painn.pkl
"""

import argparse

from potlab.export.mliappy import build_from_run


def build_parser():
    """Build the argument parser (separate from main so tests can reuse it)."""
    parser = argparse.ArgumentParser(
        description="Build a mliappy model pickle from a trained run."
    )
    parser.add_argument("--run", type=str, required=True,
                        help="Run name under runs/ (its best.pt is loaded).")
    parser.add_argument("--elements", type=str, required=True,
                        help="Element names of the LAMMPS atom types, comma-separated "
                             "in type order (e.g. C,H for type 1=C, type 2=H). Must "
                             "match the input script's pair_coeff.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output pickle filename.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for the model (default: cpu).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    elements = [e.strip() for e in args.elements.split(",") if e.strip()]
    model = build_from_run(args.run, elements, device=args.device)
    model.pickle(args.out)
    print(
        f"Wrote {args.out}: PaiNN wrapper from run {args.run!r}, "
        f"elements {model.element_types}, cutoff "
        f"{2.0 * model.rcutfac:.1f} Angstrom."
    )
    return 0


if __name__ == "__main__":
    # True only when executed directly, not when imported. SystemExit
    # carries main's return code out as the exit code.
    raise SystemExit(main())

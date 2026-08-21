"""Training entry point.

M0 version: parse the CLI, load and override the config, create the run
directory, and print what was resolved. The real training loop lands in M2.
"""

import argparse

import potlab.config as config
import potlab.training as training


def build_parser():
    """Build the argument parser (separate from main so tests can reuse it)."""
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the config file.")
    parser.add_argument("-o", "--override", action="append", default=[], help="Override config values using dotted paths, e.g., 'training.lr=1e-3'.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")
    return parser


def main():
    """Parse CLI args, load + override the config, create the run directory."""
    parser = build_parser()
    args = parser.parse_args()
    config_data = config.load_config(args.config, args.override)
    run_dir = training.make_run_dir(config_data.run_name)
    print(f"Run directory: {run_dir}")
    print(f"Configuration: {config_data}")
    return 0


if __name__ == "__main__":
    # True only when executed directly (python scripts/train.py), not when
    # imported. SystemExit carries main's return code out as the exit code.
    raise SystemExit(main())

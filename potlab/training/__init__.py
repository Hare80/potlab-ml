"""Training package: run directory management (the trainer itself lands in M2)."""

from pathlib import Path

from potlab import RUNS_DIR


def make_run_dir(run_name: str, root: Path = RUNS_DIR) -> Path:
    """Create a run directory (checkpoints/ and plots/) under root, return its path.

    ``root`` defaults to the project-wide RUNS_DIR; tests pass tmp_path instead.
    """
    run_dir = root / run_name
    # exist_ok=True makes repeated calls safe (idempotent).
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir
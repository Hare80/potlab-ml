"""potlab: config-driven, registry-based ML potential framework.

This module is the single source of truth for project paths: all filesystem
access (runs/, data/, configs/) anchors to ROOT, never to the current working
directory, so scripts work regardless of where they are launched from.
"""

from pathlib import Path

__version__ = "0.1.0"

# __file__ is this file's own path; going up two parents is the project root.
# resolve() makes it absolute, so ROOT stays correct no matter the CWD.
ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

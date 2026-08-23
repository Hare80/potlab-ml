"""Config loading: YAML file + dotted-path command-line overrides."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    """Resolved configuration.

    Top-level scalars are fixed fields; the ``model`` / ``data`` /
    ``training`` / ``export`` sections stay open dicts so registered
    classes can add their own keys without touching this module.
    """

    run_name: str = "baseline"
    seed: int = 0
    # Open dict sections: unknown keys are forwarded to the registered
    # constructors. default_factory=dict is called once per instance, so
    # two Configs never share one mutable dict (a plain {} default would).
    model: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    export: dict = field(default_factory=dict)


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply dotted-path overrides like ``training.optimizer.lr=1e-3`` onto the raw dict.

    Mutates ``raw`` in place: dicts are shared by reference, so the caller
    sees every change without needing the return value (kept for convenience).
    """
    for override in overrides:
        # partition always returns a 3-tuple (before, sep, after); a missing
        # "=" gives sep == "", which we reject instead of guessing a value.
        path, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"override must look like KEY=VALUE, got: {override!r}")
        # Walk one nesting level at a time, rebinding the local cursor d.
        d = raw
        keys = path.split(".")
        for key in keys[:-1]:
            d = d[key]
        # Write the final key in place. yaml.safe_load parses the raw string
        # into the right type (int / float / bool / None): "3" becomes 3.
        d[keys[-1]] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file, apply overrides, and return a Config."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Overrides must land before Config(**raw) is built, so top-level fields
    # (run_name, seed) are overrideable too. The mutation happens in place,
    # so apply_overrides' return value is intentionally not used here.
    apply_overrides(raw, overrides or [])
    # **raw unpacks the dict into keyword arguments: keys must match the
    # dataclass field names exactly; unknown top-level keys raise TypeError.
    return Config(**raw)

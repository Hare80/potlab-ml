"""Smoke tests for the M0 scaffold.

These tests pin down *mechanisms and structure*, never hyperparameter
values: configs will be tuned freely, but load_config / the registry /
make_run_dir must keep working exactly like this.
"""

from pathlib import Path

import pytest
import yaml

import potlab
from potlab import config, registry, training


def test_import():
    # The package must be importable and carry a version string. This is
    # what proves the editable install is wired up correctly.
    assert potlab.__version__ == "0.1.0"


def test_load_config(tmp_path):
    # Write a throwaway YAML into a temporary directory - tests must never
    # touch the real configs/ or rely on configs/default.yaml.
    config_content = {
        "run_name": "test_run",
        "seed": 42,
        "model": {"name": "TestModel"},
        "data": {
            "name": "TestData",
            "data_dir": str(tmp_path),
        },
        "training": {"lr": 0.001, "batch_size": 32},
        "export": {"out": "model.pt"},
    }
    # Three different paths through apply_overrides in one go: a nested
    # key, a top-level field, and a plain string value.
    overrides = [
        "training.lr=0.01",
        "run_name=exp2",
        "data.data_dir=/tmp/data",
    ]

    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(config_content, f)

    loaded = config.load_config(yaml_path, overrides)

    # Top-level fields are overrideable too: run_name changed in place
    # before Config was built...
    assert loaded.run_name == "exp2"
    # ...while untouched fields survive the round-trip unchanged.
    assert loaded.seed == 42
    # Nested override + type conversion: "0.01" became the float 0.01,
    # not the string "0.01".
    assert isinstance(loaded.training["lr"], float)
    assert loaded.training["lr"] == 0.01
    assert loaded.data["data_dir"] == "/tmp/data"

    # A bare KEY without "=" is rejected loudly, not silently ignored.
    with pytest.raises(ValueError):
        config.apply_overrides(config_content, ["invalid_override"])


def test_registry():
    # MODELS is module-level global state, so the test clears it both
    # before AND after: it must neither depend on nor leave behind
    # anything for other tests, regardless of run order.
    registry.MODELS.clear()

    @registry.register_model("smoke")
    class Dummy:
        pass

    # The registry stores the class object itself: the same identity, not
    # a copy - that's what makes a name in the config resolve back to it.
    assert registry.MODELS["smoke"] is Dummy

    # Registering the same name twice must fail fast instead of silently
    # overwriting the first entry. The decorator raises the moment it is
    # applied, i.e. while the class definition inside pytest.raises runs.
    with pytest.raises(ValueError):
        @registry.register_model("smoke")
        class DuplicateModel:
            pass

    registry.MODELS.clear()


def test_make_run_dir(tmp_path):
    # tmp_path is passed as root so the real runs/ directory is never
    # touched. Calling twice is the idempotency check itself: the second
    # call must not raise thanks to exist_ok=True.
    run_dir = training.make_run_dir("demo", root=tmp_path)
    assert isinstance(run_dir, Path)
    assert run_dir == tmp_path / "demo"
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "plots").is_dir()

    # Second call: same directory, no exception.
    run_dir = training.make_run_dir("demo", root=tmp_path)
    assert run_dir == tmp_path / "demo"

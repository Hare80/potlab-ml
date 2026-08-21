"""Model and dataset registries.

Configs select entries by name (``model.name`` / ``data.name``); classes
register themselves here with a decorator. Scripts import only this module,
never concrete model or dataset classes.
"""

# name -> class; a plain dict is the whole registry, no plugin framework.
MODELS = {}
DATASETS = {}


def register_model(name):
    """Return a decorator that registers a model class under ``name``.

    ``register_model("painn")`` is a decorator factory: it is called once
    with the name (captured by the closure), and the ``deco`` it returns is
    what Python then applies to the class.
    """

    def deco(cls):
        if name in MODELS:
            raise ValueError(f"duplicate model name: {name!r}")
        # Store the class object itself (not an instance) - constructing it
        # is the caller's job, later, with the config as arguments.
        MODELS[name] = cls
        # Returning cls keeps the class usable under its original name.
        return cls

    return deco


def register_dataset(name):
    """Return a decorator that registers a dataset class under ``name``.

    Same pattern as register_model, writing into DATASETS instead.
    """

    def deco(cls):
        if name in DATASETS:
            raise ValueError(f"duplicate dataset name: {name!r}")
        DATASETS[name] = cls
        return cls

    return deco

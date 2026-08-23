"""Model layer: the BaseModel protocol and concrete models.

Models register themselves by name (potlab.registry.MODELS); scripts and
the trainer import the registry only, never concrete model classes.
"""

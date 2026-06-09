"""Typed config *schema* — the YAML in ``configs/`` is the single source of truth.

These dataclasses name and type every field but deliberately hold **no default values**: a
config file is meant to be read as a complete, reproducible record of exactly what ran, so the
values live in one place (the YAML), never two. ``from_yaml`` ignores unknown keys (so ``name:``
and comments are free) and raises if a required field is missing — a partial config fails loudly
instead of silently falling back to a stale default.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


def _load(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class _FromYaml:
    """Mixin: build the dataclass from a YAML file, ignoring unknown keys, requiring the rest."""

    @classmethod
    def from_yaml(cls, path: str | Path):
        data = _load(path)
        fields = {f.name for f in dataclasses.fields(cls)}
        known = {k: v for k, v in data.items() if k in fields}
        missing = fields - known.keys()
        if missing:
            raise ValueError(f"{path}: config is missing required fields {sorted(missing)}")
        return cls(**known)


@dataclasses.dataclass
class ZooConfig(_FromYaml):
    """A population of classifiers + the velocity-field ΔK pairs probed from them."""
    out: str                                   # memmap prefix
    corruptions: tuple[float, ...]
    seeds: int
    train_n: int
    hidden: int
    epochs: int
    probe_k: int                               # samples per Gram probe
    probe_reps: int
    k_flow: int                                # steps along the one-step val flow
    val_lr: float


@dataclasses.dataclass
class CriticConfig(_FromYaml):
    zoo: str
    out: str
    epochs: int
    batch: int
    lr: float
    grad_clip: float
    val_config_frac: float                     # config-disjoint val for an honest cosine
    d_model: int


@dataclasses.dataclass
class DeployConfig(_FromYaml):
    critic: str
    corruption: float
    reg: str                                   # l2 | oracle | amortized | control | ellipsoid | compare
    metric: str                                # Gram-match geometry: frobenius | logeuclid | power | ...
    reg_lambda: float
    l2_lambda: float
    hidden: int
    train_n: int
    epochs: int
    seeds: int
    probe_k: int


@dataclasses.dataclass
class AblationConfig(_FromYaml):
    """Oracle deliveries of the validation signal — how much survives the Gram projection."""
    methods: tuple[str, ...]
    corruptions: tuple[float, ...]
    l2_lambdas: tuple[float, ...]              # tuned weight-decay per corruption (parallel to corruptions)
    reg_lambda: float
    hidden: int
    train_n: int
    epochs: int
    seeds: int
    probe_k: int


@dataclasses.dataclass
class VizConfig(_FromYaml):
    critic: str
    kind: str                                  # transport | compare | collapse
    corruption: float
    reg_lambda: float
    l2_lambda: float                           # weight decay kept in the L2/critic panels (0 disables)
    hidden: int
    train_n: int
    epochs: int
    probe_k: int
    out: str

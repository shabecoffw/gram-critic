"""Shared seeding/helpers — CPU-only and deterministic so the suite is reproducible."""

from __future__ import annotations

import torch


def seeded(seed: int = 0) -> torch.Generator:
    """A fixed-seed CPU generator so every random tensor is reproducible across runs."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def random_orthogonal(d: int, gen: torch.Generator) -> torch.Tensor:
    """A random ``d×d`` orthogonal matrix (QR of a Gaussian), for rotation-invariance tests."""
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
    return q

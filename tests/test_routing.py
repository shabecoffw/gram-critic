"""``routing.py`` — the Gram-matching metrics."""

from __future__ import annotations

import math

import torch

from gram_critic.gram import gram
from gram_critic.routing import (
    frobenius,
    logdet_barrier,
    logeuclid,
    match,
    power,
)

from .conftest import seeded

# Pure matching metrics — zero on self-match (frobenius_logdet carries a standalone barrier
# term, so it is excluded here).
PURE_METRICS = {"frobenius": frobenius, "logeuclid": logeuclid, "power": power}


def _two_grams():
    gen = seeded()
    return gram(torch.randn(24, 12, generator=gen)), gram(torch.randn(24, 12, generator=gen))


def test_self_match_is_zero():
    k, _ = _two_grams()
    for name, fn in PURE_METRICS.items():
        assert float(fn(k, k)) < 1e-6, f"{name} self-match should be ~0"


def test_metrics_are_nonnegative_and_finite():
    k, kt = _two_grams()
    for name, fn in PURE_METRICS.items():
        v = float(fn(k, kt))
        assert math.isfinite(v) and v >= 0.0, f"{name} should be finite and >= 0"


def test_power_alpha_one_reduces_to_frobenius():
    """power(α=1) == frobenius."""
    k, kt = _two_grams()
    assert torch.allclose(power(k, kt, alpha=1.0), frobenius(k, kt), atol=1e-4)


def test_logdet_barrier_explodes_toward_singular():
    """At fixed trace the barrier is minimized by a flat spectrum and grows as a λ→0."""
    flat = torch.diag(torch.tensor([0.5, 0.5]))
    peaked = torch.diag(torch.tensor([0.999, 0.001]))   # same trace, near-singular
    assert float(logdet_barrier(peaked)) > float(logdet_barrier(flat))


def test_collapse_asymmetry_logeuclid_vs_frobenius():
    """Same absolute error costs equally under Frobenius, but far more on a small eigenvalue
    under log-Euclidean — why Frobenius tolerates rank collapse and log-Euclidean doesn't."""
    delta = 0.05
    target = torch.diag(torch.tensor([1.0, 0.1]))
    perturb_large = torch.diag(torch.tensor([1.0 + delta, 0.1]))
    perturb_small = torch.diag(torch.tensor([1.0, 0.1 + delta]))

    assert torch.allclose(
        frobenius(perturb_large, target), frobenius(perturb_small, target), atol=1e-6
    )
    assert float(logeuclid(perturb_small, target)) > 10 * float(logeuclid(perturb_large, target))


def test_match_dispatches_by_name_and_callable():
    k, kt = _two_grams()
    assert torch.allclose(match(k, kt, "frobenius"), frobenius(k, kt))
    assert torch.allclose(match(k, kt, frobenius), frobenius(k, kt))   # callable passed through

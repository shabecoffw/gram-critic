"""``gram.py`` — the Gram invariances and effective-rank metric."""

from __future__ import annotations

import torch

from gram_critic.gram import effective_rank, gram, label_gram

from .conftest import random_orthogonal, seeded


def test_gram_is_symmetric():
    g = gram(torch.randn(16, 8, generator=seeded()))
    assert torch.allclose(g, g.T, atol=1e-6)


def test_gram_is_centered():
    """Every row and column sums to ~0."""
    g = gram(torch.randn(16, 8, generator=seeded()))
    assert torch.allclose(g.sum(0), torch.zeros(16), atol=1e-5)
    assert torch.allclose(g.sum(1), torch.zeros(16), atol=1e-5)


def test_gram_is_psd():
    """K is PSD — required for the log-Euclidean routing's matrix log/eigh."""
    eigs = torch.linalg.eigvalsh(gram(torch.randn(16, 8, generator=seeded())))
    assert float(eigs.min()) > -1e-5


def test_gram_normalization_divides_by_sqrt_trace():
    """K is the centered Gram scaled by 1/sqrt(raw trace)."""
    k, d = 16, 8
    h = torch.randn(k, d, generator=seeded())
    center = torch.eye(k) - 1.0 / k
    raw = center @ (h @ h.T) @ center
    expected = raw / (raw.diagonal().sum().abs() + 1e-6).sqrt()
    assert torch.allclose(gram(h), expected, atol=1e-6)


def test_gram_permutation_invariance():
    """Reordering samples permutes both axes of K the same way."""
    h = torch.randn(16, 8, generator=seeded())
    perm = torch.randperm(16, generator=seeded(1))
    assert torch.allclose(gram(h[perm]), gram(h)[perm][:, perm], atol=1e-6)


def test_gram_rotation_invariance():
    """Rotating the feature axes (h @ Q, Q orthogonal) leaves K identical."""
    gen = seeded()
    h = torch.randn(16, 8, generator=gen)
    q = random_orthogonal(8, gen)
    assert torch.allclose(gram(h @ q), gram(h), atol=1e-5)


def test_label_gram_symmetric_and_shaped():
    g = label_gram(torch.randint(0, 10, (16,), generator=seeded()), n_classes=10)
    assert g.shape == (16, 16)
    assert torch.allclose(g, g.T, atol=1e-6)


def test_effective_rank_collapsed_is_near_one():
    """A rank-1 cloud has effective rank ~1."""
    direction = torch.randn(1, 32, generator=seeded())
    coords = torch.randn(128, 1, generator=seeded(1))
    assert effective_rank(coords @ direction) < 1.5


def test_effective_rank_isotropic_is_near_full():
    """An isotropic Gaussian in D dims has effective rank near D."""
    d = 32
    er = effective_rank(torch.randn(4096, d, generator=seeded()))
    assert 0.7 * d < er <= d + 1e-3


def test_effective_rank_monotone_in_spread():
    """Concentrating variance into fewer directions lowers effective rank."""
    base = torch.randn(2048, 16, generator=seeded())
    squashed = base * torch.tensor([1.0] + [1e-3] * 15)
    assert effective_rank(squashed) < effective_rank(base)

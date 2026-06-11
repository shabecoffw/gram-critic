"""``critic.py`` — output symmetry and permutation equivariance of the set-transformer."""

from __future__ import annotations

import torch

from gram_critic.critic import GramCritic
from gram_critic.gram import gram

from .conftest import seeded

N_CHANNELS = 4
K = 12


def _inputs(b: int = 2, k: int = K, n_channels: int = N_CHANNELS, seed: int = 0):
    """A batch of Gram channels plus the current hidden Gram ``k_out``."""
    gen = seeded(seed)
    grams = [
        torch.stack([gram(torch.randn(k, 8, generator=gen)) for _ in range(b)])
        for _ in range(n_channels)
    ]
    k_out = torch.stack([gram(torch.randn(k, 8, generator=gen)) for _ in range(b)])
    return grams, k_out


def _critic(seed: int = 0) -> GramCritic:
    torch.manual_seed(seed)
    return GramCritic(n_channels=N_CHANNELS).eval()


def test_output_shape():
    critic = _critic()
    grams, k_out = _inputs(b=3)
    with torch.no_grad():
        out = critic(grams, k_out)
    assert out.shape == (3, K, K)


def test_output_is_symmetric():
    critic = _critic()
    grams, k_out = _inputs()
    with torch.no_grad():
        out = critic(grams, k_out)
    assert torch.allclose(out, out.transpose(1, 2), atol=1e-6)


def test_permutation_equivariance():
    """Permuting the K rows/cols of every input permutes the predicted ΔK the same way."""
    critic = _critic()
    grams, k_out = _inputs(b=1)
    perm = torch.randperm(K, generator=seeded(99))

    grams_p = [g[:, perm][:, :, perm] for g in grams]
    k_out_p = k_out[:, perm][:, :, perm]

    with torch.no_grad():
        out = critic(grams, k_out)
        out_p = critic(grams_p, k_out_p)

    expected = out[:, perm][:, :, perm]
    assert torch.allclose(out_p, expected, atol=1e-5)

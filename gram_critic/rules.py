"""Heuristic Gram-rule zoo — soft, regularizer-form analogs of normalization layers.

A normalization *layer* enforces a geometric constraint *hard* every forward pass; each rule here
enforces the same kind of constraint *softly*, as a differentiable loss added during training (L2
weight decay is always kept alongside). They are the handcrafted baselines the learned critic is
measured against. All are functions ``activations (n, d) -> scalar loss`` and are Gram-native
(rotation + permutation invariant). Sample-side rules act on the Gram ``K = HHᵀ`` (LayerNorm-like);
feature-side rules act on the covariance ``C = HᵀH`` (BatchNorm-like).
"""

from __future__ import annotations

import torch


def _center(h):
    return h - h.mean(0)


def ellipsoid(h, eps=1e-2):
    """Soft LayerNorm in the Mahalanobis metric: equalize each sample's Mahalanobis radius so the
    cloud lies on the covariance ellipsoid (Var of Gram leverage scores). NB: the inverse covariance
    rewards deleting weak directions, so this one has a hidden collapse incentive."""
    hc = _center(h)
    cov = hc.t() @ hc + eps * torch.eye(h.shape[1], device=h.device)
    lev = (hc.t() * torch.linalg.solve(cov, hc.t())).sum(0)
    return lev.var()


def soft_ln(h):
    """Soft LayerNorm (Euclidean): equalize each sample's norm so the cloud lies on a sphere.
    No inverse covariance, so — unlike ``ellipsoid`` — no hidden collapse incentive."""
    n2 = (_center(h) ** 2).sum(1)
    return n2.var() / (n2.mean() ** 2 + 1e-8)            # squared coefficient of variation (scale-free)


def decorrelation(h, eps=1e-5):
    """Barlow-Twins-style: drive the feature correlation matrix toward the identity (decorrelate)."""
    z = _center(h) / (_center(h).std(0) + eps)
    corr = (z.t() @ z) / h.shape[0]
    off = corr - torch.diag(torch.diagonal(corr))
    return off.pow(2).mean()


def log_det(h, alpha=10.0, eps=1e-6):
    """MCR²-style coding volume: MAXIMIZE ``log det(I + α·C)`` of unit-norm features (minimize its
    negative). Rank-PROMOTING by construction — the direct opposite of ``ellipsoid``'s collapse pull."""
    z = h / (h.norm(dim=1, keepdim=True) + eps)
    zc = _center(z)
    cov = (zc.t() @ zc) / h.shape[0]
    return -torch.logdet(torch.eye(h.shape[1], device=h.device) + alpha * cov)


def vicreg(h, gamma=1.0, eps=1e-4):
    """VICReg: per-feature variance floor (anti-collapse) + decorrelation."""
    std = torch.sqrt(_center(h).var(0) + eps)
    return torch.relu(gamma - std).mean() + decorrelation(h)


def uniformity(h, t=2.0, eps=1e-6):
    """Wang–Isola: spread unit-norm features evenly on the sphere via a Gaussian potential.
    Pairwise sq-distances come straight from the Gram: ``‖z_i−z_j‖² = 2 − 2·G_ij`` for unit z."""
    z = h / (h.norm(dim=1, keepdim=True) + eps)
    d2 = 2.0 - 2.0 * (z @ z.t())
    mask = ~torch.eye(h.shape[0], dtype=torch.bool, device=h.device)
    return torch.log(torch.exp(-t * d2[mask]).mean() + 1e-9)


RULES = {
    "ellipsoid": ellipsoid,        # sample-side, Mahalanobis sphere (collapse-prone)
    "soft_ln": soft_ln,            # sample-side, Euclidean sphere
    "decorrelation": decorrelation,  # feature-side, Barlow Twins
    "log_det": log_det,            # feature-side, MCR² volume (rank-promoting)
    "vicreg": vicreg,              # feature-side, variance floor + decorrelation
    "uniformity": uniformity,      # sample-side, even spread on the sphere
}

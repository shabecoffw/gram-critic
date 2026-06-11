"""Gram geometry of activations — the representation the critic reads and pushes.

The centered, trace-normalized Gram matrix ``K = HH^T`` of a batch of activations is
*permutation-invariant* over samples and *rotation-invariant* over feature axes. That
invariance is exactly what makes a critic trained on one network's activations transfer
to another of a different width — the critic never sees raw features, only their geometry.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gram(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Centered, trace-normalized Gram matrix.

    Args:
        h: activations, shape ``(K, D)`` (K samples, D features).
    Returns:
        ``(K, K)`` matrix, mean-centered across samples and scaled by ``1/sqrt(trace)`` (so its
        own trace is ``sqrt(trace)``, not 1) — a scale that keeps the Gram entries O(1) across
        widths without fully discarding the magnitude.
    """
    k = h.shape[0]
    center = torch.eye(k, device=h.device, dtype=h.dtype) - 1.0 / k
    g = center @ (h @ h.T) @ center
    return g / (g.diagonal().sum().abs() + eps).sqrt()


def label_gram(y: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    """Gram matrix of one-hot labels — the 'target geometry' the task implies."""
    return gram(F.one_hot(y.long(), n_classes).float())


def effective_rank(h: torch.Tensor) -> float:
    """Participation ratio ``exp(H(λ̂))`` of the activation covariance spectrum.

    A smooth, differentiable-in-spirit stand-in for rank: ~1 when activations collapse
    onto a line, ~D when variance is spread evenly. Rank *collapse* is the failure mode
    that kills naive activation regularizers, so this is the metric we watch.
    """
    hc = (h - h.mean(0)).float().cpu()
    spectrum = torch.linalg.eigvalsh(hc.T @ hc).clamp(min=0)
    p = spectrum / (spectrum.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def activation_gradient(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``g_h = ∂CE(model(x), y) / ∂h`` — the validation signal in activation space.

    This is the quantity the critic learns to *amortize*: how a label-informed step would
    reshape the features. It needs labels, so it is only ever computed to build training
    data — never at deploy time.
    """
    h = model.features(x).detach().requires_grad_(True)
    loss = F.cross_entropy(model.head(h), y)
    return torch.autograd.grad(loss, h)[0].detach()

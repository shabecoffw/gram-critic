"""Routing the validation gradient through the Gram — *Gram approximations of `g_h`*.

This is the axis orthogonal to ``rules.py``. A rule imposes label-free structure; a **routing**
delivers the validation signal `g_h = ∂L_val/∂h` *through* the Gram, and the question is how
faithfully. Two things determine the fidelity:

1. **The target.** The oracle target is `ΔK = gram(h − g_h) − gram(h)` — how one true-label step
   reshapes the Gram. (The amortized critic predicts this `ΔK` instead.)
2. **The matching metric** — and this turns out to matter enormously. The Gram is a PSD matrix, so
   matching it in the *Euclidean* `‖K − K_target‖²_F` is the wrong geometry: a low-rank `K` matches
   cheaply, so Frobenius *rewards rank collapse*. The PSD-manifold metrics make sending an eigenvalue
   to zero ~infinitely costly, so they route the same `g_h` far more faithfully and rank-preservingly.

Empirically (oracle `ΔK`, L2 kept): Frobenius gives +0.213 @rank4.6 at c=0.9; **log-Euclidean gives
+0.548 @rank8.8** — same signal, >2× the win, no collapse. The collapse was a metric artifact.
"""

from __future__ import annotations

import torch


def _matlog(k: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Differentiable matrix logarithm of a PSD Gram (routes through CPU — MPS lacks ``eigh``)."""
    kc = (k + eps * torch.eye(k.shape[0], device=k.device)).cpu().double()
    w, v = torch.linalg.eigh(kc)
    return ((v * w.clamp(min=eps).log()) @ v.t()).float().to(k.device)


def frobenius(k: torch.Tensor, k_target: torch.Tensor) -> torch.Tensor:
    """Euclidean matching ``‖K − K_target‖²_F`` — cheap, but rewards rank collapse."""
    return ((k - k_target.detach()) ** 2).sum()


def logeuclid(k: torch.Tensor, k_target: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Log-Euclidean (PSD-manifold) matching — collapsing an eigenvalue to 0 costs ~∞."""
    return ((_matlog(k, eps) - _matlog(k_target, eps).detach()) ** 2).sum()


def affine_invariant(k: torch.Tensor, k_target: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Affine-invariant (canonical SPD) matching ``‖log(T^{-½} K T^{-½})‖²_F = Σ (log γᵢ)²``.

    The *exact* geodesic distance on the SPD manifold — congruence-invariant (``K ↦ MKMᵀ``), of
    which ``logeuclid`` is the cheaper flatten-once-with-log approximation. Both send the cone
    boundary to infinity (so rank-preserving); they agree iff ``K`` and ``T`` commute and diverge by
    the non-commuting part. ``γᵢ`` are the generalized eigenvalues of ``(K, T)`` = eigenvalues of the
    congruence ``T^{-½} K T^{-½}``. Target detached, so only the inner ``K`` carries gradient."""
    n = k.shape[0]
    eye = torch.eye(n, device=k.device)
    t = (k_target.detach() + eps * eye).cpu().double()
    wt, vt = torch.linalg.eigh(t)
    t_isqrt = (vt * wt.clamp(min=eps).rsqrt()) @ vt.t()             # detached constant (cpu, double)
    kc = (k + eps * eye).cpu().double()                            # differentiable in k
    m = t_isqrt @ kc @ t_isqrt
    m = 0.5 * (m + m.t())                                          # symmetrize (kill fp drift)
    wm, vm = torch.linalg.eigh(m)
    logm = (vm * wm.clamp(min=eps).log()) @ vm.t()
    return (logm ** 2).sum().float().to(k.device)


def _matpow(k: torch.Tensor, alpha: float, eps: float = 1e-2) -> torch.Tensor:
    """Differentiable matrix power ``Kᵅ`` of a PSD Gram (spectral function → CPU eigh, like ``_matlog``)."""
    kc = (k + eps * torch.eye(k.shape[0], device=k.device)).cpu().double()
    w, v = torch.linalg.eigh(kc)
    return ((v * w.clamp(min=eps).pow(alpha)) @ v.t()).float().to(k.device)


def power(k: torch.Tensor, k_target: torch.Tensor, alpha: float = 0.5, eps: float = 1e-2) -> torch.Tensor:
    """Power-metric matching ``‖Kᵅ − K_targetᵅ‖²_F / α²`` — a magnitude-normalized dial that
    interpolates Frobenius (α=1) → log-Euclidean (α→0, since ``Kᵅ ≈ I + α·logK`` ⇒ the raw norm
    ``≈ α²·‖logK−logKt‖²``). The ``/α²`` holds the push *magnitude* roughly constant so that lowering
    α isolates the *geometry* (more relative/ratio error) rather than just shrinking the gradient.

    Empirically (oracle ΔK, c=0.9) the win climbs smoothly +0.21→+0.55 as α:1→0 while rank saturates
    by α=0.5 — i.e. the gain past rank-saturation is the relative geometry, not collapse-prevention."""
    raw = ((_matpow(k, alpha, eps) - _matpow(k_target, alpha, eps).detach()) ** 2).sum()
    return raw / (alpha * alpha)


def logdet_barrier(k: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """``−logdet(K + εI) = −Σ log(λ_i + ε)`` — an anti-collapse barrier that explodes as any λ→0.

    MPS-native (Cholesky, no CPU eigh). At fixed trace it is minimized by a *flat* spectrum, i.e. it
    pushes the representation toward full rank — the pure collapse-prevention effect, no direction."""
    kpd = k + eps * torch.eye(k.shape[0], device=k.device, dtype=k.dtype)
    return -2.0 * torch.linalg.cholesky(kpd).diagonal().clamp(min=1e-12).log().sum()


def frobenius_logdet(k: torch.Tensor, k_target: torch.Tensor, barrier: float = 1e-3,
                     eps: float = 1e-2) -> torch.Tensor:
    """Euclidean direction-matching **plus** a separable anti-collapse barrier — the decomposition
    test: does flooring collapse rescue the cheap Frobenius match, without log-Euclidean's reweighting?"""
    return frobenius(k, k_target) + barrier * logdet_barrier(k, eps)


METRICS = {"frobenius": frobenius, "logeuclid": logeuclid, "affine_invariant": affine_invariant,
           "power": power, "frobenius_logdet": frobenius_logdet}


def match(k: torch.Tensor, k_target: torch.Tensor, metric="frobenius") -> torch.Tensor:
    """Pull the (differentiable) Gram ``k`` toward ``k_target`` under the chosen matching metric.

    ``metric`` is a name in ``METRICS`` or a callable ``(k, k_target) -> scalar`` — the latter lets an
    experiment pass a parameterized variant, e.g. ``partial(power, alpha=0.25)``, with no new key."""
    fn = metric if callable(metric) else METRICS[metric]
    return fn(k, k_target)

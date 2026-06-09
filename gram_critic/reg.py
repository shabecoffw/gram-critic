"""The ``Regularizer`` base class — a training-time penalty, built to mirror a PyTorch optimizer.

This module is just the mechanism. Every concrete condition (weight decay, the handcrafted Gram
rules, the learned critic) lives in ``regularizers.py``; the training loop never changes, only which
Regularizer(s) you attach.

The lifecycle copies ``torch.optim.Optimizer`` deliberately, because the analogy is exact:

    opt = Adam(model.parameters())     reg = make_reg(..., model)         # register what you govern
    opt.zero_grad()                    reg.clear()                        # drop last step's state
    loss.backward()   # fills .grad    model(x)  # hook fills the cache    # forward populates it
    opt.step()        # uses .grad     reg.penalty(...)  # uses the cache  # consume it

Parameters are persistent, so a parameter penalty (weight decay) just *reads its registered params* —
no hook. Activations are ephemeral (they exist only during a forward and change every batch), so the
only thing a Regularizer can register for them is the *layer*, and reading that layer's output at
forward time is a forward hook. ``clear()`` is the ``zero_grad`` analog: it drops the captured
activations each step, which (1) releases last step's autograd graph instead of pinning it and (2)
turns a missing forward into a loud ``KeyError`` rather than a silent penalty on a stale activation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch

from .models import MLP


class Regularizer:
    """A representation-geometry penalty, registered with what it governs like an optimizer.

    Subclasses implement ``_compute(cache, x, y, y_true) -> scalar``; the base handles registration
    (hooking the tapped layers), the per-step ``clear``, probe subsampling, λ scaling, the
    warmup/stride gate, and turning the penalty into a transport field for visualization.

    Class attributes ``warmup``/``every`` set the schedule: the experimental push applies after
    ``warmup`` epochs, every ``every``-th step. Weight decay overrides them to ``0, 1`` (always on).

    Args:
        taps: ``(name, module)`` layers whose forward output to capture (the activations the penalty
            reads). Empty for parameter-only regularizers.
        params: parameters the penalty reads directly (weight decay). Empty for activation penalties.
        model: kept for the few deliveries that recompute a label gradient (``activation_gradient``).
        lam: penalty weight λ.
        probe_k: if set, each call penalizes a random ``probe_k``-row subset of the batch — the critic
            needs a *fixed* Gram size; ``None`` uses the whole batch.
    """

    warmup, every = 4, 2          # experimental push: after `warmup` epochs, every `every`-th step

    def __init__(self, *, taps=(), params=(), model: MLP | None = None,
                 lam: float = 1.0, probe_k: int | None = None):
        self.lam = lam
        self.probe_k = probe_k
        self.model = model
        self.taps = list(taps)
        self.params = list(params)
        self._cache: dict[str, torch.Tensor] = {}

    # --- registration & per-step lifecycle (the optimizer analogy) ---------------------------
    @contextmanager
    def registered(self):
        """Install the capture hooks for the duration of training (once), then remove them."""
        handles = [m.register_forward_hook(self._capture(name)) for name, m in self.taps]
        try:
            yield self
        finally:
            for h in handles:
                h.remove()
            self.clear()

    def _capture(self, name: str) -> Callable:
        return lambda _m, _inp, out: self._cache.__setitem__(name, out)

    def clear(self) -> None:
        """Drop captured activations — call once per step, the analog of ``optimizer.zero_grad()``."""
        self._cache.clear()

    def active(self, epoch: int, step: int) -> bool:
        return epoch >= self.warmup and step % self.every == 0

    # --- the penalty -------------------------------------------------------------------------
    def _compute(self, cache: dict, x, y, y_true) -> torch.Tensor:
        raise NotImplementedError

    def _select(self, x, y, y_true):
        """The (activations, batch) the penalty sees — a probe subset if ``probe_k`` is set."""
        if self.probe_k is None or self.probe_k >= x.shape[0]:
            return self._cache, x, y, y_true
        idx = torch.randperm(x.shape[0], device=x.device)[: self.probe_k]
        return {k: v[idx] for k, v in self._cache.items()}, x[idx], y[idx], y_true[idx]

    def penalty(self, x, y, y_true) -> torch.Tensor:
        cache, x, y, y_true = self._select(x, y, y_true)
        return self.lam * self._compute(cache, x, y, y_true)

    # --- the same penalty, as a field on the activations (for visualization) -----------------
    def transport(self, xp, yp, yp_true) -> "np.ndarray":  # noqa: F821
        """``−∂penalty/∂h`` on every probe point — the arrow the transport animation draws.

        Registers its own hooks for one forward (no subsampling, so each arrow stays aligned with
        its point), then differentiates the penalty w.r.t. the captured penultimate activation.
        """
        with self.registered():
            self.clear()
            self.model(xp)                                # forward fills self._cache via the hooks
            pen = self._compute(self._cache, xp, yp, yp_true)
            h = self._cache["h2"]
            g = torch.autograd.grad(pen, h, allow_unused=True)[0]
            return (torch.zeros_like(h) if g is None else -g).detach().cpu().numpy()

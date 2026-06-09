"""The classifier under study.

Everything the critic reads lives on ``MLP.features`` (the penultimate activations), so the Gram
geometry is width-agnostic: a critic trained on 64-wide models applies unchanged to any width. The
network is split into ``features`` (penultimate) and ``head`` (readout) so regularizers and the
critic can act on the representation directly. The two ReLUs are kept as named submodules (``act1``,
``act2``) so a ``Regularizer`` can register a forward hook on them and capture the post-activation
values — ``taps()`` exposes those points. Training lives in ``harness.py``; this module is just the
model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Three-layer MLP split into ``features`` (penultimate) and ``head`` (readout)."""

    def __init__(self, in_dim: int = 784, hidden: int = 64, n_classes: int = 10):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, n_classes)
        self.act1 = nn.ReLU()        # named tap points: a Regularizer hooks these to read h1 / h2
        self.act2 = nn.ReLU()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.l2(self.act1(self.l1(x))))

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.l3(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def taps(self) -> list[tuple[str, nn.Module]]:
        """The ``(name, module)`` activation tap points a Regularizer can capture on the forward pass."""
        return [("h1", self.act1), ("h2", self.act2)]

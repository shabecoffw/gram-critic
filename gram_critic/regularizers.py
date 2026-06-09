"""The concrete regularizer zoo, the ΔK delivery strategies, and the name -> object factory.

Every experimental condition in this repo is a ``Regularizer`` (the base class lives in ``reg.py``):
weight decay, the handcrafted Gram rules, and the learned critic are the *same kind of object*,
swapped in one line. The project stated as a type: "can a *learned* Regularizer beat the
*weight-decay* Regularizer".

  * ``WeightDecay`` — the baseline, a genuine optimizer peer: registered with its params, read
    directly, no hook, every step.
  * ``RuleReg``     — a handcrafted Gram rule (``rules.RULES``) on the captured activation.
  * ``GramMatchReg``— pull the hidden Gram toward ``K + ΔK``; two axes compose and mirror the paper:
      - *how ΔK is delivered* — a ``delta`` strategy: oracle / critic / control (below).
      - *how the Gram is matched* — a ``metric`` from ``routing.py``: Frobenius (rank-collapsing) or
        log-Euclidean (PSD-manifold, rank-preserving).
  * ``DirectReg`` / ``LabelGramReg`` — the ablation deliveries.
"""

from __future__ import annotations

from typing import Callable

import torch

from . import rules
from .gram import activation_gradient, gram, label_gram
from .models import MLP
from .reg import Regularizer
from .routing import match


# ----------------------------------------------------------------------------------------------
# Concrete regularizers
# ----------------------------------------------------------------------------------------------
class WeightDecay(Regularizer):
    """L2 / weight decay — a genuine optimizer peer: registered with the params it governs and read
    directly, no hook, every step (no warmup)."""

    warmup, every = 0, 1

    def __init__(self, params, lam: float = 1e-4):
        super().__init__(params=params, lam=lam)

    def _compute(self, cache, x, y, y_true):
        return sum(p.pow(2).sum() for p in self.params)


class RuleReg(Regularizer):
    """A handcrafted Gram rule (``rules.RULES``) as a penalty on the captured hidden activation."""

    def __init__(self, fn: Callable, *, taps, lam: float = 1.0):
        super().__init__(taps=taps, lam=lam)
        self.fn = fn

    def _compute(self, cache, x, y, y_true):
        return self.fn(cache["h2"])


class GramMatchReg(Regularizer):
    """Pull the hidden Gram toward ``K + ΔK`` under a PSD-aware ``metric``.

    ``delta(reg, cache, x, y, y_true) -> (probe_k, probe_k)`` supplies ΔK; the ``metric`` decides how
    faithfully the match preserves rank. Covers deploy's oracle / amortized / control regimes and the
    ablation's ``gram_gradient`` delivery.
    """

    def __init__(self, delta: Callable, *, taps, model, lam: float = 1.0,
                 probe_k: int = 64, metric: str = "frobenius"):
        super().__init__(taps=taps, model=model, lam=lam, probe_k=probe_k)
        self.delta = delta
        self.metric = metric

    def _compute(self, cache, x, y, y_true):
        k = gram(cache["h2"])
        target = (k + self.delta(self, cache, x, y, y_true)).detach()
        return match(k, target, self.metric)


class DirectReg(Regularizer):
    """Ablation ``direct``: push activations straight along ``−g_h`` (the full, non-Gram signal)."""

    def __init__(self, *, taps, model, lam: float = 1.0, probe_k: int = 64):
        super().__init__(taps=taps, model=model, lam=lam, probe_k=probe_k)

    def _compute(self, cache, x, y, y_true):
        h = cache["h2"]
        g = activation_gradient(self.model, x, y_true)
        return ((h - (h.detach() - g)) ** 2).sum()


class LabelGramReg(Regularizer):
    """Ablation ``label_gram``: match the hidden Gram to the (true-)label Gram — full supervision."""

    def __init__(self, *, taps, lam: float = 1.0, probe_k: int = 64):
        super().__init__(taps=taps, lam=lam, probe_k=probe_k)

    def _compute(self, cache, x, y, y_true):
        return ((gram(cache["h2"]) - label_gram(y_true).detach()) ** 2).sum()


# ----------------------------------------------------------------------------------------------
# ΔK delivery strategies (the "routing" axis, orthogonal to the metric)
# ----------------------------------------------------------------------------------------------
def oracle_delta(reg, cache, x, y, y_true) -> torch.Tensor:
    """True one-step ΔK from the label gradient — the ceiling (uses labels, not amortizable)."""
    g = activation_gradient(reg.model, x, y_true)
    h = cache["h2"]
    return gram(h.detach() - g) - gram(h).detach()


def critic_delta(critic) -> Callable:
    """The learned, label-free ΔK: the critic reads the Gram channels and predicts the push."""
    def delta(reg, cache, x, y, y_true):
        k = gram(cache["h2"])
        grams = [gram(x).unsqueeze(0), gram(cache["h1"]).unsqueeze(0),
                 k.unsqueeze(0), label_gram(y).unsqueeze(0)]
        return critic(grams, k.unsqueeze(0))[0]
    return delta


def control_delta(critic) -> Callable:
    """A random symmetric ΔK, magnitude-matched to the critic's — isolates *direction* from strength."""
    base = critic_delta(critic)
    def delta(reg, cache, x, y, y_true):
        d = base(reg, cache, x, y, y_true)
        r = torch.randn_like(d)
        r = 0.5 * (r + r.transpose(0, 1))
        return r * (d.norm() / (r.norm() + 1e-8))
    return delta


# ----------------------------------------------------------------------------------------------
# Factory: (name, model) -> Regularizer.  Adding a condition is one entry here, not an `if` in
# three files.  The model is needed because, like an optimizer, a Regularizer is registered with
# what it governs (this model's tap points and/or params).
# ----------------------------------------------------------------------------------------------
_GRAM_DELTAS = {
    "oracle": lambda critic: oracle_delta,
    "gram_gradient": lambda critic: oracle_delta,     # ablation alias for the oracle Gram push
    "amortized": critic_delta,
    "control": control_delta,
}


def make_reg(name: str, model: MLP, *, lam: float, probe_k: int,
             critic=None, metric: str = "frobenius") -> Regularizer:
    """Build the Regularizer for a named condition, registered against ``model``.

    ``oracle``/``gram_gradient``/``amortized``/``control`` -> ``GramMatchReg`` (``amortized`` and
    ``control`` require ``critic``); any key in ``rules.RULES`` -> ``RuleReg``; ``direct`` /
    ``label_gram`` -> their ablation regs.
    """
    taps = model.taps()
    if name in _GRAM_DELTAS:
        if name in ("amortized", "control") and critic is None:
            raise ValueError(f"condition {name!r} needs a critic")
        return GramMatchReg(_GRAM_DELTAS[name](critic), taps=taps, model=model,
                            lam=lam, probe_k=probe_k, metric=metric)
    if name in rules.RULES:
        return RuleReg(rules.RULES[name], taps=taps, lam=lam)
    if name == "direct":
        return DirectReg(taps=taps, model=model, lam=lam, probe_k=probe_k)
    if name == "label_gram":
        return LabelGramReg(taps=taps, lam=lam, probe_k=probe_k)
    raise ValueError(f"unknown regularizer: {name!r}")

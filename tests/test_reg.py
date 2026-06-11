"""``reg.py`` — the Regularizer lifecycle: hook capture, clear, gating, probe subsampling."""

from __future__ import annotations

import pytest
import torch

from gram_critic.models import MLP
from gram_critic.regularizers import RuleReg, WeightDecay
from gram_critic.rules import soft_ln

from .conftest import seeded


def _rule_reg(model: MLP) -> RuleReg:
    return RuleReg(soft_ln, taps=model.taps())


def test_hook_captures_activations_then_clears():
    model = MLP()
    reg = _rule_reg(model)
    x = torch.randn(16, 784, generator=seeded())
    with reg.registered():
        reg.clear()
        model(x)
        assert set(reg._cache) == {"h1", "h2"}
        assert reg._cache["h2"].shape == (16, 64)     # hidden width
        reg.clear()
        assert reg._cache == {}


def test_hooks_removed_after_context():
    """After ``registered()`` exits, a later forward captures nothing."""
    model = MLP()
    reg = _rule_reg(model)
    with reg.registered():
        pass
    model(torch.randn(4, 784, generator=seeded()))
    assert reg._cache == {}


def test_missing_forward_is_loud_keyerror():
    """No forward => empty cache => penalty raises instead of scoring stale activations."""
    model = MLP()
    reg = _rule_reg(model)
    x = torch.randn(8, 784, generator=seeded())
    y = torch.randint(0, 10, (8,), generator=seeded())
    with pytest.raises(KeyError):
        reg.penalty(x, y, y)


def test_active_gate_respects_warmup_and_stride():
    reg = _rule_reg(MLP())          # base default: warmup=4, every=2
    assert reg.active(epoch=3, step=0) is False       # warming up
    assert reg.active(epoch=4, step=0) is True
    assert reg.active(epoch=4, step=1) is False       # off-stride
    assert reg.active(epoch=5, step=2) is True


def test_weight_decay_is_always_active():
    """Weight decay overrides the schedule to (warmup=0, every=1)."""
    wd = WeightDecay(params=list(MLP().parameters()))
    assert wd.warmup == 0 and wd.every == 1
    assert wd.active(epoch=0, step=0) is True


def test_probe_subsampling_fixes_gram_size():
    """``probe_k`` selects a fixed-size row subset across cache and batch."""
    reg = _rule_reg(MLP())
    reg.probe_k = 8
    reg._cache = {"h2": torch.randn(32, 64, generator=seeded())}
    x = torch.randn(32, 784, generator=seeded())
    y = torch.randint(0, 10, (32,), generator=seeded())
    cache, xs, ys, ys_true = reg._select(x, y, y)
    assert cache["h2"].shape == (8, 64)
    assert xs.shape == (8, 784)
    assert ys.shape == (8,)


def test_probe_noop_when_batch_smaller_than_k():
    """probe_k >= batch leaves the whole batch unchanged."""
    reg = _rule_reg(MLP())
    reg.probe_k = 64
    reg._cache = {"h2": torch.randn(16, 64, generator=seeded())}
    x = torch.randn(16, 784, generator=seeded())
    y = torch.randint(0, 10, (16,), generator=seeded())
    cache, xs, _, _ = reg._select(x, y, y)
    assert cache["h2"].shape == (16, 64)
    assert xs.shape == (16, 784)

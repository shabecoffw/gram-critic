"""The single training loop, and the seed-averaged evaluation built on it.

Every place a classifier-under-study is trained — the zoo, the deploy comparison, the ablation,
the visualizer — goes through ``train``. It is a generator that yields ``(epoch, model)`` so
callers can probe mid-training. Regularizers are the *only* thing that distinguishes an L2 baseline
from the critic from a handcrafted rule, so the loop takes a ``make_regs(model)`` factory: because a
Regularizer is registered with the model it governs (its tap points / params), exactly like
``Adam(model.parameters())``, it can only be built once the model exists — which is here.

The per-step lifecycle mirrors the optimizer's: ``reg.clear()`` (drop last step's captured
activations, the ``zero_grad`` analog), forward (the hooks fill each reg's cache), then add each
active reg's penalty.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch import nn

from .data import make_run
from .gram import effective_rank
from .models import MLP
from .reg import Regularizer
from .regularizers import WeightDecay, make_reg

MakeRegs = Callable[[MLP], Sequence[Regularizer]]


def _mps_gc() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def train(x, y, *, make_regs: MakeRegs | None = None, y_true=None, hidden: int = 64,
          epochs: int = 40, lr: float = 1e-3, batch: int = 256, seed: int = 0,
          device: str = "cpu") -> Iterator[tuple[int, MLP]]:
    """Train an MLP under the regularizers ``make_regs`` builds for it, yielding ``(epoch, model)``.

    Each regularizer decides when it applies (``active``) and what it costs (``penalty``); the loop
    clears their caches, runs one forward (which fills those caches via hooks), and sums the active
    penalties onto the cross-entropy. ``y_true`` (clean labels) is threaded to the regularizers that
    need the validation direction; it defaults to ``y`` when there is none.
    """
    torch.manual_seed(seed)
    model = MLP(x.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = x.shape[0]
    if y_true is None:
        y_true = y
    regs = list(make_regs(model)) if make_regs else []
    with contextlib.ExitStack() as stack:
        for r in regs:
            stack.enter_context(r.registered())          # install capture hooks once, for the run
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for step, i in enumerate(range(0, n, batch)):
                idx = perm[i : i + batch]
                opt.zero_grad()                          # clear last step's gradients...
                for r in regs:
                    r.clear()                            # ...and the regularizers' captured activations
                logits = model(x[idx])                   # forward fills each reg's cache via its hooks
                loss = criterion(logits, y[idx])
                for r in regs:
                    if r.active(epoch, step):
                        loss = loss + r.penalty(x[idx], y[idx], y_true[idx])
                loss.backward()
                opt.step()
            yield epoch, model
            _mps_gc()


def evaluate(model: MLP, x_test, y_test) -> tuple[float, float]:
    """Test accuracy and effective rank — the two numbers every experiment reports."""
    model.eval()
    with torch.no_grad():
        acc = (model(x_test).argmax(1) == y_test).float().mean().item()
        rank = effective_rank(model.features(x_test[:512]))
    return acc, rank


def run_condition(name, *, cfg, x_full, y_full, x_test, y_test, corruption, l2_lambda,
                  critic=None, metric: str = "frobenius", device: str) -> tuple[float, float]:
    """Train one condition across ``cfg.seeds`` seeds and return mean (accuracy, effective rank).

    ``name == "l2"`` is the baseline (weight decay only); any other name adds that regularizer
    *on top of* weight decay — the way these experiments are always run.
    """
    def make_regs(model):
        regs: list[Regularizer] = []
        if name != "l2":
            regs.append(make_reg(name, model, lam=cfg.reg_lambda, probe_k=cfg.probe_k,
                                 critic=critic, metric=metric))
        regs.append(WeightDecay(model.parameters(), l2_lambda))
        return regs

    accs, ranks = [], []
    for seed in range(cfg.seeds):
        x, y_true, y = make_run(x_full, y_full, train_n=cfg.train_n, corruption=corruption,
                                seed=seed, device=device)
        for _, model in train(x, y, make_regs=make_regs, y_true=y_true, hidden=cfg.hidden,
                              epochs=cfg.epochs, seed=42 + seed, device=device):
            pass
        acc, rank = evaluate(model, x_test, y_test)
        accs.append(acc)
        ranks.append(rank)
    return float(np.mean(accs)), float(np.mean(ranks))

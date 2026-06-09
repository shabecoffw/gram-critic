"""Deploy regularizers and measure whether they beat weight decay.

A fresh classifier trains on corrupted labels under weight decay plus one experimental regularizer
that pulls its hidden-activation Gram toward ``K + ΔK`` under ``cfg.metric`` (the matching geometry —
``logeuclid`` for the rank-preserving headline result, ``frobenius`` for the collapse-prone contrast).
The conditions differ only in that regularizer (built by ``reg.make_reg``):
  * ``l2``        — weight decay only; the baseline.
  * ``oracle``    — the *true* one-step ΔK from true labels; the ceiling (uses labels).
  * ``amortized`` — the *critic's* prediction from Gram geometry alone; NO validation labels.
  * ``control``   — a random ΔK, magnitude-matched to the critic; isolates the learned direction.
  * any ``rules.RULES`` key — a handcrafted Gram rule.
We report test accuracy and effective rank (to watch for the rank-collapse failure mode).
"""

from __future__ import annotations

from .config import DeployConfig
from .harness import run_condition


def run(cfg: DeployConfig, x_full, y_full, x_test, y_test, device: str):
    """Run one regularizer, or — if ``reg: compare`` — the full L2 / oracle / amortized table."""
    regimes = ["l2", "oracle", "amortized", "control", "ellipsoid"] if cfg.reg == "compare" else [cfg.reg]
    critic = None
    if {"amortized", "control"} & set(regimes):
        from .train import load_critic
        critic = load_critic(cfg.critic, device)

    print(f"  corruption={cfg.corruption}  (l2_lambda={cfg.l2_lambda}, reg_lambda={cfg.reg_lambda})",
          flush=True)
    results, baseline = {}, None
    for reg in regimes:
        acc, rank = run_condition(reg, cfg=cfg, x_full=x_full, y_full=y_full, x_test=x_test,
                                  y_test=y_test, corruption=cfg.corruption, l2_lambda=cfg.l2_lambda,
                                  critic=critic, metric=cfg.metric, device=device)
        if reg == "l2":
            baseline = acc
        delta = f"  (vs L2 {acc - baseline:+.3f})" if baseline is not None and reg != "l2" else ""
        print(f"    {reg:<9s}: acc={acc:.3f}  eff_rank={rank:.1f}{delta}", flush=True)
        results[reg] = (acc, rank)
    return results

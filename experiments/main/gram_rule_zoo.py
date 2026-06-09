"""Gram-rule zoo — head-to-head of handcrafted Gram regularizers (soft-normalization analogs) vs L2
and the learned critic, under label noise. Which is the best generalizer, and does any beat L2 while
PRESERVING rank? L2 weight decay is kept in every condition. Each rule gets its own λ grid (their
loss scales differ by ~4 orders of magnitude)."""
import numpy as np
import torch
from gram_critic.config import DeployConfig
from gram_critic import deploy as D, rules
from gram_critic.data import load_mnist
from gram_critic.train import load_critic
from gram_critic.__main__ import get_device

GRIDS = {"soft_ln": [10, 30, 100], "ellipsoid": [10, 30, 100], "decorrelation": [30, 100, 300],
         "vicreg": [10, 30, 100], "log_det": [0.03, 0.1, 0.3], "uniformity": [0.1, 0.3, 1.0]}

dev = get_device(); print("device", dev, flush=True)
x, y, xt, yt = load_mnist(); x, xt, yt = x.to(dev), xt.to(dev), yt.to(dev)
critic = load_critic("runs/critic.pt", dev)

for c, l2 in [(0.9, 3e-3), (0.6, 1e-2)]:
    print(f"\n=== c={c}  (L2 kept in every row, l2_lambda={l2}) ===", flush=True)
    cfg = DeployConfig(critic="runs/critic.pt", reg="compare", metric="frobenius", corruption=c,
                       l2_lambda=l2, reg_lambda=1.0, seeds=3, epochs=40, hidden=64, train_n=3000,
                       probe_k=64)
    base, _ = D._eval("l2", cfg, x, y, xt, yt, None, dev)
    print(f"  {'L2 (baseline)':<22} acc={base:.3f}", flush=True)
    a, r = D._eval("oracle", cfg, x, y, xt, yt, critic, dev)
    print(f"  {'oracle (ceiling)':<22} acc={a:.3f}@{r:.1f}  ({a-base:+.3f})", flush=True)
    a, r = D._eval("amortized", cfg, x, y, xt, yt, critic, dev)
    print(f"  {'learned critic':<22} acc={a:.3f}@{r:.1f}  ({a-base:+.3f})", flush=True)
    for rule, grid in GRIDS.items():
        for lam in grid:
            cfg.reg_lambda = lam
            a, r = D._eval(rule, cfg, x, y, xt, yt, None, dev)
            print(f"  {f'{rule}(λ={lam})':<22} acc={a:.3f}@{r:.1f}  ({a-base:+.3f})", flush=True)

"""Oracle metric-geometry comparison: the SAME oracle ΔK, matched under different geometries, to
decompose the log-Euclidean win into (a) anti-collapse barrier vs (b) relative mode-reweighting.

  frobenius          α=1 absolute matching            (rank-collapsing baseline)
  power α=0.5,0.25   ‖Kᵅ−Ktᵅ‖²  interpolates toward log-E  (Option 1: bundles barrier+reweighting)
  frobenius_logdet   Frobenius + barrier·(−logdet)    (Option 2: isolates the barrier alone)
  logeuclid          full log-Euclidean               (barrier + reweighting together)

L2 kept in every row (tuned per corruption). Oracle = true labels, the ceiling — isolates the metric
from critic-fidelity noise. Reports Δacc vs L2 @ effective-rank, 3 seeds, c=0.9 and c=0.6."""
from functools import partial
import numpy as np
from gram_critic.config import DeployConfig
from gram_critic.data import load_mnist
from gram_critic.routing import power, frobenius_logdet
from gram_critic.harness import run_condition
from gram_critic.__main__ import get_device

dev = get_device(); print("device", dev, flush=True)
cfg = DeployConfig.from_yaml("configs/deploy_c09.yaml")          # seeds/reg_lambda/probe_k/etc.
x, y, xt, yt = load_mnist(); x = x.to(dev); xt = xt.to(dev); yt = yt.to(dev)

METRICS = [
    ("frobenius",          "frobenius"),
    ("power α=0.5",        partial(power, alpha=0.5)),
    ("power α=0.25",       partial(power, alpha=0.25)),
    ("logeuclid",          "logeuclid"),
    ("frob+logdet b=1e-3", partial(frobenius_logdet, barrier=1e-3)),
    ("frob+logdet b=1e-2", partial(frobenius_logdet, barrier=1e-2)),
    ("frob+logdet b=1e-1", partial(frobenius_logdet, barrier=1e-1)),
]

for corruption, l2_lambda in [(0.9, 3.0e-3), (0.6, 1.0e-2)]:
    print(f"\n=== corruption={corruption}  (L2 λ={l2_lambda}, 3 seeds, oracle ΔK) ===", flush=True)
    base, _ = run_condition("l2", cfg=cfg, x_full=x, y_full=y, x_test=xt, y_test=yt,
                            corruption=corruption, l2_lambda=l2_lambda, device=dev)
    print(f"    {'l2 (baseline)':<20s}: acc={base:.3f}", flush=True)
    for label, metric in METRICS:
        acc, rank = run_condition("oracle", cfg=cfg, x_full=x, y_full=y, x_test=xt, y_test=yt,
                                  corruption=corruption, l2_lambda=l2_lambda, metric=metric, device=dev)
        print(f"    {label:<20s}: acc={acc:.3f}  Δ={acc-base:+.3f}  @rank {rank:.1f}", flush=True)

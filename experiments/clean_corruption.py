"""Does our BEST (log-Euclidean) deployable critic help at LOW (0.3) and NO (0.0) corruption?
At high noise it won (+0.094) by delivering the validation direction, rank-preserving. At clean/low
noise the training gradient already points right, so the question is whether the amortized val signal
still adds anything once there's little noise to correct. Tune L2 per corruption (fair), critic on top.
oracle-logeuclid = ceiling (true ΔK); critic-frobenius = contrast (the bad metric)."""
import numpy as np
from gram_critic.config import DeployConfig
from gram_critic.data import load_mnist
from gram_critic.train import load_critic
from gram_critic.harness import run_condition
from gram_critic.__main__ import get_device

dev = get_device(); print("device", dev, flush=True)
cfg = DeployConfig.from_yaml("configs/deploy_c09.yaml")
critic = load_critic("runs/critic.pt", dev)
x, y, xt, yt = load_mnist(); x = x.to(dev); xt = xt.to(dev); yt = yt.to(dev)

def rc(name, corruption, l2, metric="logeuclid"):
    return run_condition(name, cfg=cfg, x_full=x, y_full=y, x_test=xt, y_test=yt,
                         corruption=corruption, l2_lambda=l2, critic=critic, metric=metric, device=dev)

for corruption in [0.3, 0.0]:
    print(f"\n=== corruption={corruption} ===", flush=True)
    sweep = {l2: rc("l2", corruption, l2)[0] for l2 in [1e-3, 3e-3, 1e-2, 3e-2]}
    best_l2 = max(sweep, key=sweep.get); base = sweep[best_l2]
    print(f"    L2 sweep {{{', '.join(f'{l:.0e}:{a:.3f}' for l,a in sweep.items())}}}  -> best λ={best_l2:.0e} acc={base:.3f}", flush=True)
    for label, name, metric in [("CRITIC (log-Euclid)", "amortized", "logeuclid"),
                                ("oracle (log-Euclid)", "oracle", "logeuclid"),
                                ("critic (frobenius) ", "amortized", "frobenius")]:
        acc, rank = rc(name, corruption, best_l2, metric)
        print(f"    {label:<20s}: acc={acc:.3f}  Δ={acc-base:+.3f}  @rank {rank:.1f}", flush=True)

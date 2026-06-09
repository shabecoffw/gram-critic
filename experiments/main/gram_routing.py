"""Better Gram routing: does the MATCHING METRIC cause the collapse?

Same oracle ΔK (true-label activation gradient g_h through the Gram), delivered two ways:
  frobenius  — ‖K − K_target‖²_F   (Euclidean; a low-rank K matches cheaply -> collapse incentive)
  logeuclid  — ‖log(K+εI) − log(K_target+εI)‖²_F  (PSD-manifold; sending an eigenvalue->0 costs ~∞)
Hypothesis: log-Euclidean preserves rank and routes the same signal more faithfully, because the
Frobenius collapse was a metric artifact, not a property of the Gram. L2 kept in every row."""
import numpy as np
import torch
import torch.nn.functional as F
from gram_critic.data import load_mnist, corrupt_labels, subset
from gram_critic.gram import gram, activation_gradient, effective_rank
from gram_critic.models import MLP
from gram_critic.__main__ import get_device

dev = get_device(); print("device", dev, flush=True)
xf, yf, xt, yt = load_mnist(); xt, yt = xt.to(dev), yt.to(dev)
PK = 64

def matlog(K, eps=1e-2):
    Kc = (K + eps * torch.eye(K.shape[0], device=K.device)).cpu().double()
    w, V = torch.linalg.eigh(Kc)
    return ((V * w.clamp(min=eps).log()) @ V.t()).float().to(K.device)

def push(metric, model, xs, ys_true):
    h1 = torch.relu(model.l1(xs)); h2 = torch.relu(model.l2(h1))
    K = gram(h2)
    g = activation_gradient(model, xs, ys_true)
    K_tgt = gram(h2.detach() - g)
    if metric == "frobenius":
        return ((K - K_tgt.detach()) ** 2).sum()
    return ((matlog(K) - matlog(K_tgt).detach()) ** 2).sum()

def train(metric, reg_lambda, c, l2, seed):
    x, yc = subset(xf, yf, 3000, np.random.default_rng(31000 + seed)); x = x.to(dev)
    y = corrupt_labels(yc, c, np.random.default_rng(42 + seed)).to(dev)
    ytrue = yc.to(dev)
    torch.manual_seed(42 + seed); m = MLP(784, 64).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(40):
        perm = torch.randperm(x.shape[0], device=dev)
        for bi, i in enumerate(range(0, x.shape[0], 256)):
            ix = perm[i:i+256]; xb = x[ix]
            loss = F.cross_entropy(m(xb), y[ix])
            if metric != "l2" and ep >= 4 and bi % 2 == 0:
                sub = torch.randperm(xb.shape[0], device=dev)[:PK]
                loss = loss + reg_lambda * push(metric, m, xb[sub], ytrue[ix][sub])
            loss = loss + l2 * sum(p.pow(2).sum() for p in m.parameters())
            opt.zero_grad(); loss.backward(); opt.step()
        if hasattr(torch, "mps"): torch.mps.empty_cache()
    m.eval()
    with torch.no_grad():
        return (m(xt).argmax(1) == yt).float().mean().item(), effective_rank(m.features(xt[:512]))

for c, l2 in [(0.9, 3e-3), (0.6, 1e-2)]:
    print(f"=== c={c} (L2 kept everywhere) ===", flush=True)
    base = np.mean([train("l2", 0, c, l2, s)[0] for s in range(3)])
    print(f"  {'L2':<22} acc={base:.3f}", flush=True)
    for name, metric, lam in [("oracle frobenius", "frobenius", 1.0),
                              ("oracle logeuclid λ0.3", "logeuclid", 0.3),
                              ("oracle logeuclid λ1", "logeuclid", 1.0),
                              ("oracle logeuclid λ3", "logeuclid", 3.0)]:
        r = [train(metric, lam, c, l2, s) for s in range(3)]
        a = np.mean([v for v,_ in r]); rk = np.mean([v for _,v in r])
        print(f"  {name:<22} acc={a:.3f}@{rk:.1f}  ({a-base:+.3f})", flush=True)

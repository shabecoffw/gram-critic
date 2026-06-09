"""Does the labels-free critic inherit the log-Euclidean routing win? The critic already predicts ΔK;
we only change the DEPLOY matching metric (no retrain). L2 kept in every row."""
import numpy as np
import torch
import torch.nn.functional as F
from gram_critic.data import load_mnist, corrupt_labels, subset
from gram_critic.gram import gram, label_gram, activation_gradient, effective_rank
from gram_critic.models import MLP
from gram_critic.train import load_critic
from gram_critic import routing
from gram_critic.__main__ import get_device

dev = get_device(); print("device", dev, flush=True)
xf, yf, xt, yt = load_mnist(); xt, yt = xt.to(dev), yt.to(dev)
critic = load_critic("runs/critic.pt", dev); PK = 64

def push(reg, metric, model, xb, yb, yb_true):
    sub = torch.randperm(xb.shape[0], device=dev)[:PK]; xs = xb[sub]
    h1 = torch.relu(model.l1(xs)); h2 = torch.relu(model.l2(h1)); K = gram(h2)
    if reg == "oracle":
        K_tgt = gram(h2.detach() - activation_gradient(model, xs, yb_true[sub]))
    else:
        grams = [gram(xs)[None], gram(h1)[None], K[None], label_gram(yb[sub])[None]]
        K_tgt = (K + critic(grams, K[None])[0]).detach()
    return routing.match(K, K_tgt, metric)

def train(reg, metric, lam, c, l2, seed):
    x, yc = subset(xf, yf, 3000, np.random.default_rng(31000 + seed)); x = x.to(dev)
    y = corrupt_labels(yc, c, np.random.default_rng(42 + seed)).to(dev); ytrue = yc.to(dev)
    torch.manual_seed(42 + seed); m = MLP(784, 64).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(40):
        perm = torch.randperm(x.shape[0], device=dev)
        for bi, i in enumerate(range(0, x.shape[0], 256)):
            ix = perm[i:i+256]; xb = x[ix]
            loss = F.cross_entropy(m(xb), y[ix])
            if reg != "l2" and ep >= 4 and bi % 2 == 0:
                loss = loss + lam * push(reg, metric, m, xb, y[ix], ytrue[ix])
            loss = loss + l2 * sum(p.pow(2).sum() for p in m.parameters())
            opt.zero_grad(); loss.backward(); opt.step()
        if hasattr(torch, "mps"): torch.mps.empty_cache()
    m.eval()
    with torch.no_grad():
        return (m(xt).argmax(1) == yt).float().mean().item(), effective_rank(m.features(xt[:512]))

for c, l2 in [(0.9, 3e-3), (0.6, 1e-2)]:
    print(f"=== c={c} (L2 kept everywhere) ===", flush=True)
    base = np.mean([train("l2", None, 0, c, l2, s)[0] for s in range(3)])
    print(f"  {'L2':<26} acc={base:.3f}", flush=True)
    for name, reg, metric, lam in [("amortized frobenius λ1", "amortized", "frobenius", 1.0),
                                   ("amortized logeuclid λ1", "amortized", "logeuclid", 1.0),
                                   ("amortized logeuclid λ3", "amortized", "logeuclid", 3.0),
                                   ("oracle logeuclid λ3 (ceil)", "oracle", "logeuclid", 3.0)]:
        r = [train(reg, metric, lam, c, l2, s) for s in range(3)]
        a = np.mean([v for v,_ in r]); rk = np.mean([v for _,v in r])
        print(f"  {name:<26} acc={a:.3f}@{rk:.1f}  ({a-base:+.3f})", flush=True)

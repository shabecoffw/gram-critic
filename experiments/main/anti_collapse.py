"""Can the critic's win survive WITHOUT the rank collapse?

Keep the gradient-amortized critic push (no labels) but add an activation-only anti-collapse term:
minimize ||C/tr(C)||_F^2 of the activation covariance (minimized by a uniform spectrum -> high rank).
Neither term uses labels at deploy. If critic+anti-collapse holds the c=0.9 win while preserving rank
(so c=0.6 stops hurting), the goal is met; if preventing collapse also kills the win, that's decisive
evidence the meaningful win is intrinsically collapse-mediated.
"""
import numpy as np
import torch
import torch.nn.functional as F
from gram_critic.data import load_mnist, corrupt_labels, subset
from gram_critic.gram import gram, label_gram, effective_rank
from gram_critic.models import MLP
from gram_critic.train import load_critic
from gram_critic.__main__ import get_device

dev = get_device(); print("device", dev, flush=True)
xf, yf, xt, yt = load_mnist(); xt, yt = xt.to(dev), yt.to(dev)
critic = load_critic("runs/critic.pt", dev)

def anti_collapse(h):
    hc = h - h.mean(0); C = hc.t() @ hc
    return ((C / (C.diagonal().sum() + 1e-6)) ** 2).sum()   # low <-> uniform spectrum <-> high rank

def train(c, l2, use_critic, reg_lambda, rank_lambda, seed):
    x, yc = subset(xf, yf, 3000, np.random.default_rng(31000 + seed)); x = x.to(dev)
    y = corrupt_labels(yc, c, np.random.default_rng(42 + seed)).to(dev)
    torch.manual_seed(42 + seed); m = MLP(784, 64).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(40):
        perm = torch.randperm(x.shape[0], device=dev)
        for bi, i in enumerate(range(0, x.shape[0], 256)):
            ix = perm[i:i+256]; xb = x[ix]
            loss = F.cross_entropy(m(xb), y[ix])
            if use_critic and ep >= 4 and bi % 2 == 0:
                sub = torch.randperm(xb.shape[0], device=dev)[:64]; xs = xb[sub]
                h1 = torch.relu(m.l1(xs)); h2 = torch.relu(m.l2(h1)); k = gram(h2)
                g = [gram(xs)[None], gram(h1)[None], k[None], label_gram(y[ix][sub])[None]]
                loss = loss + reg_lambda * ((k - (k + critic(g, k[None])[0]).detach()) ** 2).sum()
            if rank_lambda > 0:
                loss = loss + rank_lambda * anti_collapse(m.features(xb))
            loss = loss + l2 * sum(p.pow(2).sum() for p in m.parameters())
            opt.zero_grad(); loss.backward(); opt.step()
        if hasattr(torch, "mps"): torch.mps.empty_cache()
    m.eval()
    with torch.no_grad():
        return (m(xt).argmax(1) == yt).float().mean().item(), effective_rank(m.features(xt[:512]))

conds = [("L2", False, 0, 0), ("critic", True, 1.0, 0),
         ("critic+rank.01", True, 1.0, 0.01), ("critic+rank.03", True, 1.0, 0.03),
         ("critic+rank.1", True, 1.0, 0.1), ("critic+rank.3", True, 1.0, 0.3)]
for c, l2 in [(0.9, 3e-3), (0.6, 1e-2)]:
    print(f"=== c={c} ===", flush=True); base = None
    for name, uc, rl, rk in conds:
        r = [train(c, l2, uc, rl, rk, s) for s in range(3)]
        a = np.mean([v for v, _ in r]); rr = np.mean([v for _, v in r])
        if name == "L2": base = a
        print(f"  {name:<18}: acc={a:.3f}@{rr:.1f}  (vs L2 {a-base:+.3f})", flush=True)

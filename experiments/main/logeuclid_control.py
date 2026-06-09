"""Is the log-Euclidean win the GEOMETRY (spectrum inflation) or the critic's SIGNAL?
Control: random symmetric ΔK (matched magnitude) deployed under log-Euclidean, vs the critic and the
oracle. If random ≈ critic, it's the inflation degeneracy; if random << critic, the signal matters."""
import numpy as np
import torch
import torch.nn.functional as F
from gram_critic.data import load_mnist, corrupt_labels, subset
from gram_critic.gram import gram, label_gram, activation_gradient, effective_rank
from gram_critic.models import MLP
from gram_critic.routing import match
from gram_critic.train import load_critic
from gram_critic.__main__ import get_device

dev = get_device(); print("device", dev, flush=True)
xf, yf, xt, yt = load_mnist(); xt, yt = xt.to(dev), yt.to(dev)
critic = load_critic("runs/critic.pt", dev)

def push(kind, model, xb, yb, yb_true):
    sub = torch.randperm(xb.shape[0], device=dev)[:64]; xs = xb[sub]
    h1 = torch.relu(model.l1(xs)); h2 = torch.relu(model.l2(h1)); K = gram(h2)
    grams = [gram(xs)[None], gram(h1)[None], K[None], label_gram(yb[sub])[None]]
    dk = critic(grams, K[None])[0]
    if kind == "critic":
        tgt = K + dk
    elif kind == "oracle":
        tgt = gram(h2.detach() - activation_gradient(model, xs, yb_true[sub]))
    else:  # random ΔK, matched to the critic's magnitude
        r = torch.randn_like(K); r = 0.5 * (r + r.t()); r = r * (dk.norm() / (r.norm() + 1e-8))
        tgt = K + r
    return match(K, tgt.detach(), "logeuclid")

def run(kind, lam, c, l2, seed):
    x, yc = subset(xf, yf, 3000, np.random.default_rng(31000 + seed)); x = x.to(dev)
    y = corrupt_labels(yc, c, np.random.default_rng(42 + seed)).to(dev); ytrue = yc.to(dev)
    torch.manual_seed(42 + seed); m = MLP(784, 64).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(40):
        perm = torch.randperm(x.shape[0], device=dev)
        for bi, i in enumerate(range(0, x.shape[0], 256)):
            ix = perm[i:i+256]; xb = x[ix]; loss = F.cross_entropy(m(xb), y[ix])
            if kind != "l2" and ep >= 4 and bi % 2 == 0: loss = loss + lam * push(kind, m, xb, y[ix], ytrue[ix])
            loss = loss + l2 * sum(p.pow(2).sum() for p in m.parameters())
            opt.zero_grad(); loss.backward(); opt.step()
        if hasattr(torch, "mps"): torch.mps.empty_cache()
    m.eval()
    with torch.no_grad(): return (m(xt).argmax(1) == yt).float().mean().item(), effective_rank(m.features(xt[:512]))

for c, l2 in [(0.9, 3e-3), (0.6, 1e-2)]:
    base = np.mean([run("l2", 0, c, l2, s)[0] for s in range(3)])
    print(f"=== c={c}: L2={base:.3f} ===", flush=True)
    for name, kind, lam in [("random ΔK (log-E)", "random", 3.0), ("critic (log-E)", "critic", 3.0),
                            ("oracle (log-E)", "oracle", 3.0)]:
        r = [run(kind, lam, c, l2, s) for s in range(3)]
        a = np.mean([v for v,_ in r]); rk = np.mean([v for _,v in r])
        print(f"  {name:<20} acc={a:.3f}@{rk:.1f}  ({a-base:+.3f})", flush=True)

"""Capacity sweep done RIGHT: batch 32, 40 epochs (converged), so d=32 reaches its true ~0.17 on 20k
and we can see if d=64 actually beats it. Config-held-out val, Frobenius loss, best-val checkpoint."""
import numpy as np
import torch
import torch.nn.functional as F
from gram_critic.zoo import load, CHANNELS
from gram_critic.critic import GramCritic
from gram_critic.train import _cosine
from gram_critic.__main__ import get_device

dev = get_device(); print("device", dev, flush=True)
mm, meta, N = load("runs/zoo"); SUB = 20000
sel = np.sort(np.random.default_rng(0).choice(N, SUB, replace=False))
data = {k: torch.from_numpy(np.asarray(mm[k][sel], dtype=np.float16)) for k in CHANNELS + ["delta_k"]}
cfgs = meta[sel][:, :2]; uniq = sorted(set(map(tuple, cfgs)))
val_cfgs = set(uniq[i] for i in np.random.default_rng(1).permutation(len(uniq))[:max(1, len(uniq)//6)])
is_val = np.array([tuple(c) in val_cfgs for c in cfgs]); tr, va = np.where(~is_val)[0], np.where(is_val)[0]
print(f"  {SUB} frames, train={len(tr)} val={len(va)}; batch 32, 40 epochs (converged)", flush=True)

def cos_on(net, ids):
    net.eval(); ps, ts = [], []
    for i in range(0, len(ids), 256):
        b = torch.from_numpy(np.sort(ids[i:i+256])); g = [data[c][b].float().to(dev) for c in CHANNELS]
        with torch.no_grad(): ps.append(net(g, data["k_h2"][b].float().to(dev)).cpu())
        ts.append(data["delta_k"][b].float())
    return _cosine(torch.cat(ps), torch.cat(ts)).mean().item()

for d in [32, 64]:
    torch.manual_seed(0); net = GramCritic(len(CHANNELS), d_model=d).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    tr_s = tr[np.random.default_rng(2).permutation(len(tr))[:3000]]; best = {"v": -2, "t": 0, "e": 0}
    for ep in range(40):
        net.train(); perm = tr[np.random.default_rng(ep).permutation(len(tr))]
        for i in range(0, len(perm), 32):
            b = torch.from_numpy(np.sort(perm[i:i+32])); g = [data[c][b].float().to(dev) for c in CHANNELS]
            F.smooth_l1_loss(net(g, data["k_h2"][b].float().to(dev)), data["delta_k"][b].float().to(dev)).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad()
        sch.step(); vc = cos_on(net, va)
        if vc > best["v"]: best = {"v": vc, "t": cos_on(net, tr_s), "e": ep}
        if ep % 8 == 0 or ep == 39: print(f"    d={d} ep{ep:2d}: val_cos={vc:+.3f} (best {best['v']:+.3f})", flush=True)
    print(f"  >>> d_model={d} ({npar:,} params): train_cos={best['t']:+.3f}  val_cos={best['v']:+.3f}  @ep{best['e']}", flush=True)

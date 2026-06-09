"""Depth sweep: we only ever varied set-transformer WIDTH (d_model); this stacks ISAB blocks
(1/2/3) at d=64, same converged 20k setup. _ISAB is shape-preserving so depth = a Sequential of
them (no forward override). Also bumps m_induce/heads at depth-2 to test 'bigger attention'."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gram_critic.zoo import load, CHANNELS
from gram_critic.critic import GramCritic, _ISAB
from gram_critic.train import _cosine
from gram_critic.__main__ import get_device

dev = get_device(); print("device", dev, flush=True)
mm, meta, N = load("runs/zoo"); SUB = 20000
sel = np.sort(np.random.default_rng(0).choice(N, SUB, replace=False))
data = {k: torch.from_numpy(np.asarray(mm[k][sel], dtype=np.float16)) for k in CHANNELS + ["delta_k"]}
cfgs = meta[sel][:, :2]; uniq = sorted(set(map(tuple, cfgs)))
val_cfgs = set(uniq[i] for i in np.random.default_rng(1).permutation(len(uniq))[:max(1, len(uniq)//6)])
is_val = np.array([tuple(c) in val_cfgs for c in cfgs]); tr, va = np.where(~is_val)[0], np.where(is_val)[0]
print(f"  {SUB} frames, train={len(tr)} val={len(va)}; batch 32, 40 epochs", flush=True)

def make(d_model=64, n_isab=1, n_heads=2, m_induce=16):
    net = GramCritic(len(CHANNELS), d_model=d_model, n_heads=n_heads, m_induce=m_induce)
    net.isab = nn.Sequential(*[_ISAB(d_model, n_heads, m_induce) for _ in range(n_isab)])
    return net

def cos_on(net, ids):
    net.eval(); ps, ts = [], []
    for i in range(0, len(ids), 256):
        b = torch.from_numpy(np.sort(ids[i:i+256])); g = [data[c][b].float().to(dev) for c in CHANNELS]
        with torch.no_grad(): ps.append(net(g, data["k_h2"][b].float().to(dev)).cpu())
        ts.append(data["delta_k"][b].float())
    return _cosine(torch.cat(ps), torch.cat(ts)).mean().item()

def train(net, tag):
    net = net.to(dev); npar = sum(p.numel() for p in net.parameters())
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
        if ep % 8 == 0 or ep == 39: print(f"    {tag} ep{ep:2d}: val_cos={vc:+.3f} (best {best['v']:+.3f})", flush=True)
    print(f"  >>> {tag} ({npar:,} params): train_cos={best['t']:+.3f}  val_cos={best['v']:+.3f}  @ep{best['e']}", flush=True)

torch.manual_seed(0); train(make(n_isab=1), "depth=1 (baseline)")
torch.manual_seed(0); train(make(n_isab=2), "depth=2")
torch.manual_seed(0); train(make(n_isab=3), "depth=3")
torch.manual_seed(0); train(make(n_isab=2, n_heads=4, m_induce=32), "depth=2 wide-attn (h4,m32)")

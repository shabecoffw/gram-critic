"""Structured rank-r critic head: emit per-row Phi (B,K,r) + symmetric core C (r,r), reconstruct
DeltaK_hat = Phi C Phi^T -- symmetric & rank<=r BY CONSTRUCTION. The target DeltaK provably lives in
a rank<=2C=20 subspace (DeltaK=[F E]S[F E]^T, F=logits/E=errors, C=10 classes), and empirically 99%
of its energy is in ~18-22 dims. So a free B*B pairwise head wastes its error budget on the ~44 null
dims; the structured head predicts only the subspace the signal lives in. Hypothesis: higher cosine.
Head-to-head vs the d=64 pairwise baseline, identical converged training (batch 32, 40 epochs, 20k)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gram_critic.zoo import load, CHANNELS
from gram_critic.critic import GramCritic
from gram_critic.train import _cosine
from gram_critic.__main__ import get_device


class StructuredGramCritic(GramCritic):
    """Same DeepSet+ISAB trunk as GramCritic, but a low-rank factored head instead of the pairwise MLP."""
    def __init__(self, n_channels, d_model=64, rank=24, **kw):
        super().__init__(n_channels, d_model=d_model, **kw)
        del self.pair
        self.rank = rank
        self.factor = nn.Linear(d_model, rank)                         # per-row Phi
        self.core = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, rank * rank))     # global symmetric core

    def forward(self, grams, k_out):
        b, k, _ = grams[0].shape
        eye = torch.eye(k, device=grams[0].device, dtype=grams[0].dtype).expand(b, -1, -1)
        emb = self.embed(torch.stack([*grams, eye], dim=-1))
        rows = torch.cat([emb.mean(2), emb.var(2), emb.amax(2)], dim=-1)
        e = self.isab(self.row_proj(rows))                            # (B,K,d)
        phi = self.factor(e)                                          # (B,K,r)
        c = self.core(e.mean(1)).view(b, self.rank, self.rank)        # (B,r,r)
        c = 0.5 * (c + c.transpose(1, 2))                             # symmetric core
        return phi @ c @ phi.transpose(1, 2)                         # (B,K,K) symmetric, rank<=r


dev = get_device(); print("device", dev, flush=True)
mm, meta, N = load("runs/zoo"); SUB = 20000
sel = np.sort(np.random.default_rng(0).choice(N, SUB, replace=False))
data = {k: torch.from_numpy(np.asarray(mm[k][sel], dtype=np.float16)) for k in CHANNELS + ["delta_k"]}
cfgs = meta[sel][:, :2]; uniq = sorted(set(map(tuple, cfgs)))
val_cfgs = set(uniq[i] for i in np.random.default_rng(1).permutation(len(uniq))[:max(1, len(uniq)//6)])
is_val = np.array([tuple(c) in val_cfgs for c in cfgs]); tr, va = np.where(~is_val)[0], np.where(is_val)[0]
print(f"  {SUB} frames, train={len(tr)} val={len(va)}; batch 32, 40 epochs", flush=True)

def cos_on(net, ids):
    net.eval(); ps, ts = [], []
    for i in range(0, len(ids), 256):
        b = torch.from_numpy(np.sort(ids[i:i+256])); g = [data[c][b].float().to(dev) for c in CHANNELS]
        with torch.no_grad(): ps.append(net(g, data["k_h2"][b].float().to(dev)).cpu())
        ts.append(data["delta_k"][b].float())
    return _cosine(torch.cat(ps), torch.cat(ts)).mean().item()

def train(net, tag):
    net = net.to(dev)
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
        if ep % 8 == 0 or ep == 39: print(f"    {tag} ep{ep:2d}: val_cos={vc:+.3f} (best {best['v']:+.3f})", flush=True)
    print(f"  >>> {tag} ({npar:,} params): train_cos={best['t']:+.3f}  val_cos={best['v']:+.3f}  @ep{best['e']}", flush=True)

torch.manual_seed(0); train(GramCritic(len(CHANNELS), d_model=64), "pairwise d=64")
for r in [20, 32]:
    torch.manual_seed(0); train(StructuredGramCritic(len(CHANNELS), d_model=64, rank=r), f"structured r={r}")

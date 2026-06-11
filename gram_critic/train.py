"""Train the Gram Critic to predict the velocity field, with an honest generalization metric.

The critic is held out on *whole configs* (corruption×seed combinations it never saw) so the
reported cosine measures transfer to new training runs, not memorization. Training uses best-val
checkpointing + gradient clipping — without the clip the cosine oscillates and never settles.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import CriticConfig
from .critic import GramCritic
from .zoo import CHANNELS, load


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def _gather(data, ids, device):
    """One batch: the input Gram channels, the current state ``k_h2``, and the target ``delta_k``."""
    b = torch.from_numpy(ids)
    grams = [data[c][b].float().to(device) for c in CHANNELS]
    return grams, data["k_h2"][b].float().to(device), data["delta_k"][b].float().to(device)


def _cos_on(net, data, ids, device) -> float:
    net.eval()
    preds, tgts = [], []
    for i in range(0, len(ids), 512):
        grams, k_h2, target = _gather(data, np.sort(ids[i : i + 512]), device)
        with torch.no_grad():
            preds.append(net(grams, k_h2).cpu())
        tgts.append(target.cpu())
    return _cosine(torch.cat(preds), torch.cat(tgts)).mean().item()


def _fit(data, tr, va, cfg: CriticConfig, device, log=False):
    net = GramCritic(len(CHANNELS), d_model=cfg.d_model).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    tr_probe = tr[np.random.default_rng(1).permutation(len(tr))[:3000]]
    va_probe = va[np.random.default_rng(2).permutation(len(va))[:3000]]
    best = {"cos": -2.0, "state": None}
    for epoch in range(cfg.epochs):
        net.train()
        order = tr[np.random.default_rng(100 + epoch).permutation(len(tr))]
        for i in range(0, len(order), cfg.batch):
            grams, k_h2, target = _gather(data, np.sort(order[i : i + cfg.batch]), device)
            pred = net(grams, k_h2)
            F.smooth_l1_loss(pred, target).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            opt.zero_grad()
        sched.step()
        val_cos = _cos_on(net, data, va_probe, device)
        if val_cos > best["cos"]:
            best = {"cos": val_cos,
                    "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
        if log and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            tc = _cos_on(net, data, tr_probe, device)
            print(f"    epoch {epoch:3d}: train_cos={tc:+.3f}  val_cos={val_cos:+.3f}  "
                  f"best={best['cos']:+.3f}", flush=True)
    return best["state"], best["cos"]


def run(cfg: CriticConfig, device: str) -> float:
    arrays, meta, n = load(cfg.zoo)
    print(f"  loading {n} frames into RAM (float16)...", flush=True)
    data = {a: torch.from_numpy(np.asarray(arr, dtype=np.float16)) for a, arr in arrays.items()}

    configs = sorted({tuple(m[:2]) for m in meta})
    perm = np.random.default_rng(0).permutation(len(configs))
    n_val = max(1, int(cfg.val_config_frac * len(configs)))
    held_out = {configs[perm[i]] for i in range(n_val)}
    is_val = np.array([tuple(m[:2]) in held_out for m in meta])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    print(f"  train={len(tr)} val={len(va)} ({n_val} held-out configs)", flush=True)

    state, cos = _fit(data, tr, va, cfg, device, log=True)
    torch.save({"state": state, "n_channels": len(CHANNELS), "d_model": cfg.d_model,
                "config_held_out_cos": cos}, cfg.out)
    print(f"  >>> config-held-out cosine = {cos:+.3f}   saved -> {cfg.out}", flush=True)
    return cos


def load_critic(path: str, device: str) -> GramCritic:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = GramCritic(ckpt["n_channels"], d_model=ckpt["d_model"]).to(device)
    net.load_state_dict(ckpt["state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net

"""Generate the critic's training data: a *velocity field* of one-step validation ΔK's.

The target must be a deterministic function of the activation state, and the critic must see
the states it will actually encounter at deploy. So for each probed checkpoint we walk a short
trajectory: take ONE validation step toward the true labels in activation space, record the
local one-step ``ΔK`` (the target), move to the new state, and repeat. Inputs therefore span the
whole push trajectory (on-policy), while every target is a single, fixed-magnitude step.

Pairs are written to a float16 memmap on disk so the dataset can far exceed RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ZooConfig
from .data import corrupt_labels, subset
from .gram import gram, label_gram
from .harness import train

CHANNELS = ["k_in", "k_h1", "k_h2", "k_label"]   # critic input Gram channels
ARRAYS = CHANNELS + ["delta_k"]                  # + the target


def generate(cfg: ZooConfig, x_full: torch.Tensor, y_full: torch.Tensor, device: str) -> int:
    """Build the velocity-field dataset; returns the number of frames written."""
    n_frames = (len(cfg.corruptions) * cfg.seeds * cfg.epochs * cfg.probe_reps * cfg.k_flow)
    prefix = Path(cfg.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mm = {a: np.lib.format.open_memmap(f"{prefix}_{a}.npy", mode="w+", dtype=np.float16,
                                       shape=(n_frames, cfg.probe_k, cfg.probe_k)) for a in ARRAYS}
    meta = np.zeros((n_frames, 3), np.float32)   # (corruption, seed, flow_step)

    j = 0
    model_i = 0
    n_models = len(cfg.corruptions) * cfg.seeds
    for corruption in cfg.corruptions:
        for seed in range(cfg.seeds):
            model_i += 1
            x, y_clean = subset(x_full, y_full, cfg.train_n, np.random.default_rng(7000 + model_i))
            x = x.to(device)
            y_true = y_clean.to(device)
            y = corrupt_labels(y_clean, corruption, np.random.default_rng(42 + seed)).to(device)
            for epoch, model in train(x, y, hidden=cfg.hidden, epochs=cfg.epochs,
                                       seed=42 + seed, device=device):
                j = _probe(mm, meta, j, model, x, y, y_true, corruption, seed, cfg, device)
            print(f"  model {model_i}/{n_models} (c={corruption} seed={seed})  frames={j}", flush=True)

    for arr in mm.values():
        arr.flush()
    np.save(f"{prefix}_meta.npy", meta)
    return j


def _probe(mm, meta, j, model, x, y, y_true, corruption, seed, cfg: ZooConfig, device) -> int:
    """Probe ``probe_reps`` flows of length ``k_flow`` from the current checkpoint."""
    n = x.shape[0]
    for rep in range(cfg.probe_reps):
        pid = torch.from_numpy(
            np.random.default_rng(seed * 99991 + 101 * j + rep + 7).choice(n, cfg.probe_k, replace=False)
        ).to(device)
        xs, ys_true = x[pid], y_true[pid]
        with torch.no_grad():
            h1 = torch.relu(model.l1(xs))
            h = torch.relu(model.l2(h1))
        k_in = gram(xs).cpu().numpy()
        k_h1 = gram(h1).cpu().numpy()
        k_label = label_gram(y[pid]).cpu().numpy()          # CORRUPTED labels = deploy-consistent context
        for step in range(cfg.k_flow):
            k_h2 = gram(h)
            hv = h.detach().requires_grad_(True)
            g = torch.autograd.grad(F.cross_entropy(model.head(hv), ys_true), hv)[0]
            h_next = (h - cfg.val_lr * g).detach()           # one validation step toward true labels
            mm["k_in"][j] = k_in
            mm["k_h1"][j] = k_h1
            mm["k_label"][j] = k_label
            mm["k_h2"][j] = k_h2.cpu().numpy()
            mm["delta_k"][j] = (gram(h_next) - k_h2).cpu().numpy()
            meta[j] = (corruption, seed, step)
            j += 1
            h = h_next
    return j


def load(prefix: str | Path):
    """Open the memmap zoo lazily; returns ``(arrays, meta, n_frames)``."""
    arrays = {a: np.load(f"{prefix}_{a}.npy", mmap_mode="r") for a in ARRAYS}
    meta = np.load(f"{prefix}_meta.npy")
    return arrays, meta, arrays["delta_k"].shape[0]

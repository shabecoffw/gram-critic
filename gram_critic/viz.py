"""Render activation-geometry animations from recorded trajectories.

This module only *renders*; the expensive part (training + capture) lives in ``trajectory.py``, so a
single recorded run can be re-rendered any number of ways without retraining. ``run`` records the
trajectory (or loads a saved one), then draws:

  transport — one cloud (PCA of penultimate activations, colored by true class) with a quiver arrow
              on every point = the critic's transport ``−∂(Gram-match)/∂h``.
  compare   — L2 baseline vs critic clouds side by side under heavy label noise.
  collapse  — the headline: the *same* critic under Frobenius vs log-Euclidean matching, with a live
              effective-rank readout — the matching geometry is what decides collapse vs preservation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter     # noqa: E402

from .config import VizConfig
from .data import make_run
from .gram import effective_rank
from .regularizers import make_reg
from .trajectory import Trajectory, record
from .train import load_critic

CMAP = "tab10"


def _pca_basis(activations: np.ndarray) -> np.ndarray:
    """Top-2 principal directions of an activation matrix (N, D) -> basis (D, 2)."""
    _, _, vt = np.linalg.svd(activations - activations.mean(0), full_matrices=False)
    return vt[:2].T


def render_transport(traj: Trajectory, out: Path) -> None:
    basis = _pca_basis(traj[-1].activations)
    scale = [None]
    fig, ax = plt.subplots(figsize=(7, 7))

    def draw(i):
        ax.clear()
        frame = traj[i]
        p = frame.activations @ basis
        a = frame.transport @ basis
        if scale[0] is None:
            scale[0] = 0.15 * (p.max(0) - p.min(0)).mean() / (np.linalg.norm(a, axis=1).mean() + 1e-9)
        ax.scatter(p[:, 0], p[:, 1], c=traj.labels, cmap=CMAP, s=18, alpha=0.85, edgecolors="none")
        ax.quiver(p[:, 0], p[:, 1], a[:, 0] * scale[0], a[:, 1] * scale[0],
                  angles="xy", scale_units="xy", scale=1, color="black", alpha=0.5, width=0.003)
        ax.set_title(f"critic transport on activations — epoch {frame.epoch}", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])

    _save(fig, draw, len(traj), out)


def render_compare(traj_l2: Trajectory, traj_critic: Trajectory, out: Path) -> None:
    bases = (_pca_basis(traj_l2[-1].activations), _pca_basis(traj_critic[-1].activations))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    def draw(i):
        for ax, traj, basis, title in (
            (axes[0], traj_l2, bases[0], "L2 baseline"),
            (axes[1], traj_critic, bases[1], "Gram critic"),
        ):
            ax.clear()
            frame = traj[i]
            p = frame.activations @ basis
            ax.scatter(p[:, 0], p[:, 1], c=traj.labels, cmap=CMAP, s=18, alpha=0.85, edgecolors="none")
            ax.set_title(f"{title} — epoch {frame.epoch}", fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])

    _save(fig, draw, len(traj_l2), out)


def render_collapse(traj_train: Trajectory, traj_l2: Trajectory, traj_frob: Trajectory,
                    traj_le: Trajectory, out: Path) -> None:
    """The headline animation: four runs sharing an init, differing only in regularizer.

    Left to right: **just training** (no weight decay, no push — what the labels alone teach),
    **weight decay only** (the L2 baseline the deploy tables keep in every row), then the critic push
    *on top of L2* under **Frobenius** and under **log-Euclidean** matching. Each panel fixes its axes
    to its highest-rank (pre-collapse) frame, so the Frobenius cloud is seen *contracting* into a line
    while the others stay full — with a live effective-rank readout under each. The two baselines show
    neither training nor weight decay builds class structure; the Frobenius-vs-log-Euclidean split
    shows the *matching geometry*, not the predicted signal, decides collapse-vs-preservation.
    """
    panels = []
    for traj, title in ((traj_train, "just training — no L2, no push"),
                        (traj_l2, "weight decay only — L2 baseline"),
                        (traj_frob, "L2 + critic, Frobenius — rank collapses"),
                        (traj_le, "L2 + critic, log-Euclidean — preserved + clustered")):
        ranks = [effective_rank(torch.from_numpy(f.activations)) for f in traj]
        basis = _pca_basis(traj[-1].activations)                           # final-frame axes: collapse -> a line
        lim = 1.1 * max(np.abs(f.activations @ basis).max() for f in traj)  # fit every frame so contraction shows
        panels.append((traj, title, basis, lim, ranks))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))

    def draw(i):
        for ax, (traj, title, basis, lim, ranks) in zip(axes, panels):
            ax.clear()
            p = traj[i].activations @ basis
            ax.scatter(p[:, 0], p[:, 1], c=traj.labels, cmap=CMAP, s=20, alpha=0.85, edgecolors="none")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_title(f"{title}\nepoch {traj[i].epoch}  ·  eff. rank {ranks[i]:.1f}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

    _save(fig, draw, min(len(t) for t in (traj_train, traj_l2, traj_frob, traj_le)), out)


def _save(fig, draw, n_frames: int, out: Path) -> None:
    FuncAnimation(fig, draw, frames=n_frames, interval=200).save(
        out.with_suffix(".gif"), writer=PillowWriter(fps=5))
    draw(n_frames - 1)
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def run(cfg: VizConfig, x_full, y_full, device: str) -> None:
    critic = load_critic(cfg.critic, device)
    x, y_true, y = make_run(x_full, y_full, train_n=cfg.train_n, corruption=cfg.corruption,
                            seed=0, device=device)
    kw = dict(hidden=cfg.hidden, epochs=cfg.epochs, probe_k=cfg.probe_k, device=device)

    def critic_reg(metric="logeuclid"):
        """A model -> Regularizer factory (the critic push under ``metric``), for the recorder."""
        return lambda model: make_reg("amortized", model, lam=cfg.reg_lambda, probe_k=cfg.probe_k,
                                      critic=critic, metric=metric)

    if cfg.kind == "transport":
        traj = record(x, y, y_true, make_reg=critic_reg(), l2_lambda=cfg.l2_lambda, transport=True, **kw)
        traj.save(f"{cfg.out}_transport_traj")
        render_transport(traj, Path(f"{cfg.out}_transport"))
        print(f"  wrote {cfg.out}_transport.gif / .png (+ _traj.npz)", flush=True)
    elif cfg.kind == "compare":
        traj_l2 = record(x, y, y_true, l2_lambda=cfg.l2_lambda, **kw)                              # L2 baseline
        traj_critic = record(x, y, y_true, make_reg=critic_reg(), l2_lambda=cfg.l2_lambda, **kw)   # L2 + critic
        traj_l2.save(f"{cfg.out}_compare_l2_traj")
        traj_critic.save(f"{cfg.out}_compare_critic_traj")
        render_compare(traj_l2, traj_critic, Path(f"{cfg.out}_compare"))
        print(f"  wrote {cfg.out}_compare.gif / .png (+ _traj.npz)", flush=True)
    elif cfg.kind == "collapse":
        L2 = cfg.l2_lambda
        traj_train = record(x, y, y_true, **kw)                                                       # just training
        traj_l2 = record(x, y, y_true, l2_lambda=L2, **kw)                                            # L2 baseline
        traj_frob = record(x, y, y_true, make_reg=critic_reg("frobenius"), l2_lambda=L2, **kw)        # L2 + critic, frob
        traj_le = record(x, y, y_true, make_reg=critic_reg("logeuclid"), l2_lambda=L2, **kw)          # L2 + critic, log-E
        for tag, t in (("train", traj_train), ("l2", traj_l2), ("frob", traj_frob), ("le", traj_le)):
            t.save(f"{cfg.out}_collapse_{tag}_traj")
        render_collapse(traj_train, traj_l2, traj_frob, traj_le, Path(f"{cfg.out}_collapse"))
        print(f"  wrote {cfg.out}_collapse.gif / .png (+ _traj.npz)", flush=True)
    else:
        raise ValueError(f"unknown viz kind: {cfg.kind!r}")

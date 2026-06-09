"""Record activation-geometry trajectories during training; replay them for any visualization.

Producing a trajectory is expensive (a whole training run); consuming one is cheap (iterate frames).
So we split them: **record once → save → render many** figures offline, with no retraining. The
recorder is just another consumer of a ``(epoch, model)`` training stream — the training loop never
knows it is being observed, which is what lets the zoo, the deploy evaluation, and this recorder all
reuse the exact same trainer.

A ``Trajectory`` is a list of per-epoch ``Frame``s (penultimate activations on a fixed probe, plus
the critic's transport vectors if a critic was supplied) and the probe's true labels. It serializes
to a single compressed ``.npz``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .harness import train
from .models import MLP
from .reg import Regularizer
from .regularizers import WeightDecay

# Builds the (model-bound) Regularizer for a recorded run, the way an optimizer is built from a model.
MakeReg = Callable[[MLP], Regularizer]


@dataclasses.dataclass
class Frame:
    """One epoch's snapshot of the probe set's activation geometry."""
    epoch: int
    activations: np.ndarray                 # (K, D) penultimate features
    transport: np.ndarray | None = None     # (K, D) critic push, −∂(Gram-match)/∂h  (None if no critic)


class Trajectory:
    """An ordered set of activation ``Frame``s + the probe's true labels; one ``.npz`` on disk."""

    def __init__(self, labels: np.ndarray, frames: list[Frame] | None = None):
        self.labels = labels
        self.frames = frames if frames is not None else []

    def add(self, frame: Frame) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, i: int) -> Frame:
        return self.frames[i]

    @property
    def has_transport(self) -> bool:
        return bool(self.frames) and self.frames[0].transport is not None

    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "labels": self.labels,
            "epochs": np.array([f.epoch for f in self.frames]),
            "activations": np.stack([f.activations for f in self.frames]),
        }
        if self.has_transport:
            arrays["transport"] = np.stack([f.transport for f in self.frames])
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        d = np.load(Path(path).with_suffix(".npz"))
        transport = d["transport"] if "transport" in d.files else None
        frames = [
            Frame(int(ep), d["activations"][i], None if transport is None else transport[i])
            for i, ep in enumerate(d["epochs"])
        ]
        return cls(d["labels"], frames)


def record(x, y, y_true, *, hidden, epochs, probe_k, device, make_reg: MakeReg | None = None,
           l2_lambda: float = 0.0, transport: bool = False, seed: int = 0) -> Trajectory:
    """Train one MLP under ``make_reg(model)`` (+ optional weight decay) and capture a ``Frame`` per epoch.

    The recorder knows nothing about critics or oracles — it trains through the shared ``harness``
    loop and observes. ``make_reg`` builds the Gram Regularizer for a given model (it must be
    model-bound, like an optimizer); ``l2_lambda`` adds weight decay on top, so a recorded run can
    mirror the deploy tables, which keep L2 in *every* row. The four collapse panels are exactly the
    four combinations: neither (just training), L2 only, L2 + critic/Frobenius, L2 + critic/log-Euclidean.
    With ``transport=True`` each epoch also records the critic's ``−∂penalty/∂h`` arrow field.
    """
    probe = torch.from_numpy(np.random.default_rng(0).choice(x.shape[0], probe_k, replace=False)).to(device)
    xp, yp, yp_true = x[probe], y[probe], y_true[probe]
    traj = Trajectory(yp_true.cpu().numpy())

    def build_regs(model):
        regs: list[Regularizer] = []
        if make_reg is not None:
            regs.append(make_reg(model))
        if l2_lambda > 0:
            regs.append(WeightDecay(model.parameters(), l2_lambda))
        return regs

    make_regs = build_regs if (make_reg is not None or l2_lambda > 0) else None
    for epoch, model in train(x, y, make_regs=make_regs, y_true=y_true, hidden=hidden, epochs=epochs,
                              seed=seed, device=device):
        arrows = make_reg(model).transport(xp, yp, yp_true) if (make_reg is not None and transport) else None
        model.eval()
        with torch.no_grad():
            traj.add(Frame(epoch, model.features(xp).cpu().numpy(), arrows))
    return traj

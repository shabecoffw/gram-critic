"""MNIST loading and label corruption.

Label corruption is the stress test: flipping a fraction of training labels to random classes
creates a regime where the *training* signal is partly poisoned, so a regularizer that injects
the *validation* direction can meaningfully beat weight decay.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def download(data_dir: str | Path = "data") -> None:
    """Fetch and decompress the raw MNIST idx files into ``<data_dir>/MNIST/``."""
    out = Path(data_dir) / "MNIST"
    out.mkdir(parents=True, exist_ok=True)
    for stem in FILES.values():
        raw = out / stem
        if raw.exists():
            print(f"  have {raw.name}")
            continue
        print(f"  downloading {stem}.gz ...", flush=True)
        gz = out / (stem + ".gz")
        urllib.request.urlretrieve(MIRROR + stem + ".gz", gz)
        with gzip.open(gz, "rb") as f_in, open(raw, "wb") as f_out:
            f_out.write(f_in.read())
        gz.unlink()
    print(f"MNIST ready in {out}")


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, = struct.unpack(">H", f.read(2))  # noqa: F841
        dims = f.read(2)[1]
        shape = struct.unpack(">" + "I" * dims, f.read(4 * dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(data_dir: str | Path = "data") -> tuple[torch.Tensor, ...]:
    """Return ``(x_train, y_train, x_test, y_test)`` as flat, [0,1]-scaled float tensors."""
    root = Path(data_dir) / "MNIST"
    def find(stem: str) -> Path:
        for cand in (root / stem, root / (stem + ".gz")):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"{stem} not found under {root}; run `make data`.")

    xtr = _read_idx(find(FILES["train_images"])).reshape(-1, 784) / 255.0
    ytr = _read_idx(find(FILES["train_labels"]))
    xte = _read_idx(find(FILES["test_images"])).reshape(-1, 784) / 255.0
    yte = _read_idx(find(FILES["test_labels"]))
    return (
        torch.from_numpy(xtr.astype(np.float32)),
        torch.from_numpy(ytr.astype(np.int64)),
        torch.from_numpy(xte.astype(np.float32)),
        torch.from_numpy(yte.astype(np.int64)),
    )


def corrupt_labels(y: torch.Tensor, fraction: float, rng: np.random.Generator,
                   n_classes: int = 10) -> torch.Tensor:
    """Flip ``fraction`` of labels to uniformly random classes (returns a copy)."""
    y = y.clone()
    if fraction <= 0:
        return y
    n = len(y)
    idx = rng.choice(n, int(fraction * n), replace=False)
    y[idx] = torch.from_numpy(rng.integers(0, n_classes, len(idx)).astype(np.int64))
    return y


def subset(x: torch.Tensor, y: torch.Tensor, n: int, rng: np.random.Generator):
    """Draw a random ``n``-sample subset (a fresh 'dataset' for zoo diversity)."""
    idx = rng.choice(x.shape[0], n, replace=False)
    return x[idx], y[idx]


def make_run(x_full, y_full, *, train_n: int, corruption: float, seed: int, device: str,
             subset_seed: int = 31000, corrupt_seed: int = 42):
    """One deploy/ablation training run: a fresh subset, on device, with corrupted + clean labels.

    Returns ``(x, y_true, y)`` — inputs, clean labels (the validation direction the regularizers
    amortize), and the corrupted labels actually trained on. The two seed bases keep the *data*
    draw and the *corruption* draw independent across runs.
    """
    x, y_clean = subset(x_full, y_full, train_n, np.random.default_rng(subset_seed + seed))
    x = x.to(device)
    y_true = y_clean.to(device)
    y = corrupt_labels(y_clean, corruption, np.random.default_rng(corrupt_seed + seed)).to(device)
    return x, y_true, y

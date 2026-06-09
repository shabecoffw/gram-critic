"""Single entry point — every stage runs from a YAML config: ``python -m gram_critic <stage> <config>``.

Stages: ``zoo`` (generate ΔK pairs) · ``train`` (fit the critic) · ``deploy`` (eval as regularizer)
· ``viz`` (render activation animations). ``data`` downloads MNIST.
"""

from __future__ import annotations

import argparse

import torch

from .config import AblationConfig, CriticConfig, DeployConfig, VizConfig, ZooConfig
from .data import load_mnist


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(prog="gram_critic")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("zoo", "train", "deploy", "viz", "ablation"):
        sub.add_parser(name).add_argument("config")
    sub.add_parser("data").add_argument("--dir", default="data")
    args = parser.parse_args()

    if args.stage == "data":
        from .data import download
        download(args.dir)
        return

    device = get_device()
    print(f"[{args.stage}] device={device}", flush=True)

    if args.stage == "zoo":
        from . import zoo
        cfg = ZooConfig.from_yaml(args.config)
        x, y, _, _ = load_mnist()
        zoo.generate(cfg, x.to(device), y, device)

    elif args.stage == "train":
        from . import train
        train.run(CriticConfig.from_yaml(args.config), device)

    elif args.stage == "deploy":
        from . import deploy
        cfg = DeployConfig.from_yaml(args.config)
        x, y, xt, yt = load_mnist()
        deploy.run(cfg, x.to(device), y, xt.to(device), yt.to(device), device)

    elif args.stage == "ablation":
        from . import ablation
        cfg = AblationConfig.from_yaml(args.config)
        x, y, xt, yt = load_mnist()
        ablation.run(cfg, x.to(device), y, xt.to(device), yt.to(device), device)

    elif args.stage == "viz":
        from . import viz
        cfg = VizConfig.from_yaml(args.config)
        x, y, _, _ = load_mnist()
        viz.run(cfg, x.to(device), y, device)


if __name__ == "__main__":
    main()

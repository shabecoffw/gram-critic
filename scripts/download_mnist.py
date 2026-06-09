"""Download the raw MNIST idx files. Usage: ``python scripts/download_mnist.py [data_dir]``."""

import sys

from gram_critic.data import download

if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else "data")

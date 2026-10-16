"""Deterministic seeding + reproducibility hashes for the normative package.

Light port of LNGModel src/seeding.py: a single seeding source and a code
hash over the package files so resume-safe runs can detect stale artifacts.
"""
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

PKG_DIR = Path(__file__).resolve().parent


def seed_all(seed: int) -> None:
    """Seed python, numpy, and torch global RNGs from a single int."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def code_hash() -> str:
    """sha256 over the normative package's .py files (sorted paths)."""
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def determinism_env_info() -> dict:
    """Environment facts recorded with every run for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "code_hash": code_hash(),
    }

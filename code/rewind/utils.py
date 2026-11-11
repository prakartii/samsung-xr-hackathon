"""Small shared helpers: seeding and logging."""

import importlib.util
import logging
import random
import sys

import numpy as np


def seed_all(seed: int) -> None:
    """Seed Python, numpy and (if installed) torch, for byte-identical output."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seeds legacy global users
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)


def get_logger(name: str = "rewind") -> logging.Logger:
    """A logger that writes to stderr (stdout stays clean)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

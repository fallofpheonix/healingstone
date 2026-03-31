"""Centralized seed management for deterministic execution."""

import logging
import random
import os

import numpy as np
import torch

LOG = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """MANDATORY: Fix all random seeds and environment flags for determinism."""
    LOG.info("event=seed_set seed=%d", seed)
    
    # 1. Standard Python random
    random.seed(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch (if available)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Newer torch versions:
        # torch.use_deterministic_algorithms(True)
    except (ImportError, NameError):
        LOG.debug("Torch not found or cuda unavailable, skipping torch-specific seeds.")
    
    # 4. Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)


def deterministic_listdir(path: str) -> list[str]:
    """Return a sorted list of files to ensure consistent iteration order."""
    return sorted(os.listdir(path))

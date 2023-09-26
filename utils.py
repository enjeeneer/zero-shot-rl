# pylint: disable=protected-access
"""Utility functions for the project."""
import random
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

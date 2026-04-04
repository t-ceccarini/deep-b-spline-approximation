# -*- coding: utf-8 -*-
"""Helpers to make training/evaluation runs reproducible."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

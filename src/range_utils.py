"""
src/range_utils.py
------------------
Shared Bragg-peak localization utilities for both training and evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


SOFTARGMAX_TEMPERATURE = 80.0


def peak_position_soft_torch(profile: torch.Tensor, temperature: float = SOFTARGMAX_TEMPERATURE) -> torch.Tensor:
    """
    Soft peak index for tensors.
    profile: [B, L]
    returns: [B] in index units
    """
    idx = torch.arange(profile.shape[-1], dtype=profile.dtype, device=profile.device)
    weights = F.softmax(profile * temperature, dim=-1)
    return (weights * idx).sum(dim=-1)


def peak_position_soft_numpy(profile: np.ndarray, temperature: float = SOFTARGMAX_TEMPERATURE) -> float:
    """
    Soft peak index for 1D numpy profile.
    Returns index in [0, L-1].
    """
    p = np.asarray(profile, dtype=np.float64)
    p = p - float(p.max())
    w = np.exp(p * float(temperature))
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        return float(np.argmax(profile))
    w = w / w_sum
    idx = np.arange(p.shape[0], dtype=np.float64)
    return float((w * idx).sum())

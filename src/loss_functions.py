"""
src/loss_functions.py
----------------------
Physics-informed loss functions for training the 1D and 2D U-Nets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from range_utils import peak_position_soft_torch

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def find_peak_position(dose: torch.Tensor) -> torch.Tensor:
    """1D Bragg peak localization via soft-argmax, normalized to [0, 1]."""
    L = dose.shape[-1]
    d = dose.squeeze(1)
    pos = peak_position_soft_torch(d)
    return pos / max(L - 1, 1)

def compute_spatial_gradient(dose: torch.Tensor) -> torch.Tensor:
    """1D depth gradient"""
    return dose[:, :, 1:] - dose[:, :, :-1]

def find_peak_position_2d(dose_2d: torch.Tensor) -> torch.Tensor:
    """2D Bragg peak localization on central axis, normalized to [0, 1]."""
    B, _, H, W = dose_2d.shape
    central_row = H // 2
    central_profile = dose_2d[:, 0, central_row, :]  # [B, W]
    pos = peak_position_soft_torch(central_profile)
    return pos / max(W - 1, 1)

def compute_depth_gradient(dose_2d: torch.Tensor) -> torch.Tensor:
    """2D depth gradient (dD/dx) along W axis"""
    return dose_2d[:, :, :, 1:] - dose_2d[:, :, :, :-1]

def compute_lateral_gradient(dose_2d: torch.Tensor) -> torch.Tensor:
    """2D lateral gradient (dD/dy) along H axis"""
    return dose_2d[:, :, 1:, :] - dose_2d[:, :, :-1, :]


def central_axis_profile_2d(dose_2d: torch.Tensor) -> torch.Tensor:
    """Extract central-axis depth-dose profile [B, W]."""
    H = dose_2d.shape[2]
    return dose_2d[:, 0, H // 2, :]

# ══════════════════════════════════════════════════════════════════════════════
# Core Loss Modules
# ══════════════════════════════════════════════════════════════════════════════

class MSELoss(nn.Module):
    def __init__(self, focus_alpha: float = 0.0):
        super().__init__()
        self.focus_alpha = float(focus_alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.focus_alpha <= 0.0:
            return F.mse_loss(pred, target)
        # Focus learning on clinically relevant high-dose region while keeping
        # full-field supervision.
        weights = 1.0 + self.focus_alpha * target.detach()
        return ((pred - target) ** 2 * weights).mean()

# ── 1D ────────────────────────────────────────────────────────────────────────

class RangeLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(find_peak_position(pred), find_peak_position(target))

class GradientLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(compute_spatial_gradient(pred), compute_spatial_gradient(target))

class PhysicsInformedLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_range=2.0, w_grad=0.5):
        super().__init__()
        self.w_mse, self.w_range, self.w_grad = w_mse, w_range, w_grad
        self.mse_fn   = MSELoss(focus_alpha=0.0)
        self.range_fn = RangeLoss()
        self.grad_fn  = GradientLoss()

    def forward(self, pred, target):
        mse   = self.mse_fn(pred, target)
        rng   = self.range_fn(pred, target)
        grad  = self.grad_fn(pred, target)
        total = self.w_mse * mse + self.w_range * rng + self.w_grad * grad
        return total, {"mse": mse.item(), "range": rng.item(), "grad": grad.item(), "total": total.item()}

# ── 2D ────────────────────────────────────────────────────────────────────────

class RangeLoss2D(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(find_peak_position_2d(pred), find_peak_position_2d(target))

class GradientLoss2D(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(compute_depth_gradient(pred), compute_depth_gradient(target))

class PenumbraLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(compute_lateral_gradient(pred), compute_lateral_gradient(target))

class RangeBiasLoss2D(nn.Module):
    """
    Penalize systematic signed shift in Bragg-peak position.
    Complements RangeLoss2D, which is symmetric per-sample.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        signed_bias = find_peak_position_2d(pred) - find_peak_position_2d(target)
        return signed_bias.mean().pow(2)

class DistalEdgeLoss2D(nn.Module):
    """
    Match distal fall-off using central-axis depth gradients.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = central_axis_profile_2d(pred)
        t = central_axis_profile_2d(target)
        gp = p[:, 1:] - p[:, :-1]
        gt = t[:, 1:] - t[:, :-1]
        # Focus on distal drop (negative target gradient).
        weights = 1.0 + 20.0 * torch.relu(-gt)
        return ((gp - gt).pow(2) * weights).mean()

class PhysicsInformedLoss2D(nn.Module):
    def __init__(
        self,
        w_mse=1.0,
        w_range=5.0,
        w_grad=1.0,
        w_pen=2.0,
        w_bias=2.0,
        w_distal=2.0,
        mse_focus=20.0,
    ):
        super().__init__()
        self.w_mse, self.w_range, self.w_grad, self.w_pen = w_mse, w_range, w_grad, w_pen
        self.w_bias, self.w_distal = w_bias, w_distal
        self.mse_fn      = MSELoss(focus_alpha=mse_focus)
        self.range_fn    = RangeLoss2D()
        self.grad_fn     = GradientLoss2D()
        self.penumbra_fn = PenumbraLoss()
        self.bias_fn     = RangeBiasLoss2D()
        self.distal_fn   = DistalEdgeLoss2D()

    def forward(self, pred, target):
        mse   = self.mse_fn(pred, target)
        rng   = self.range_fn(pred, target)
        grad  = self.grad_fn(pred, target)
        pen   = self.penumbra_fn(pred, target)
        bias  = self.bias_fn(pred, target)
        distal = self.distal_fn(pred, target)
        total = (
            self.w_mse * mse
            + self.w_range * rng
            + self.w_grad * grad
            + self.w_pen * pen
            + self.w_bias * bias
            + self.w_distal * distal
        )
        return total, {
            "mse": mse.item(),
            "range": rng.item(),
            "grad": grad.item(),
            "penumbra": pen.item(),
            "bias": bias.item(),
            "distal": distal.item(),
            "total": total.item(),
        }

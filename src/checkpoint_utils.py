"""
src/checkpoint_utils.py
-----------------------
Helpers to save/load U-Net checkpoints with architecture metadata.
Supports legacy checkpoints that only store a raw state_dict.
"""

from __future__ import annotations

from typing import Any

import torch

from model import UNet1D, UNet2D


def _get_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Unsupported checkpoint format.")


def _infer_depth_from_state(state_dict: dict[str, torch.Tensor], prefix: str = "encoders.") -> int:
    encoder_ids = set()
    for key in state_dict.keys():
        if key.startswith(prefix):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                encoder_ids.add(int(parts[1]))
    if not encoder_ids:
        raise ValueError("Could not infer model depth from checkpoint.")
    return max(encoder_ids) + 1


def _infer_base_ch_1d(state_dict: dict[str, torch.Tensor]) -> int:
    w = state_dict.get("encoders.0.conv.block.0.weight")
    if w is None or w.ndim != 3:
        raise ValueError("Could not infer 1D base channels from checkpoint.")
    return int(w.shape[0])


def _infer_base_ch_2d(state_dict: dict[str, torch.Tensor]) -> int:
    w = state_dict.get("encoders.0.conv.block.0.weight")
    if w is None or w.ndim != 4:
        raise ValueError("Could not infer 2D base channels from checkpoint.")
    return int(w.shape[0])


def load_unet1d_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[UNet1D, dict[str, int]]:
    checkpoint_obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = _get_state_dict(checkpoint_obj)

    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        base_ch = int(checkpoint_obj.get("base_ch", _infer_base_ch_1d(state_dict)))
        depth = int(checkpoint_obj.get("depth", _infer_depth_from_state(state_dict)))
    else:
        base_ch = _infer_base_ch_1d(state_dict)
        depth = _infer_depth_from_state(state_dict)

    model = UNet1D(base_ch=base_ch, depth=depth).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, {"base_ch": base_ch, "depth": depth}


def load_unet2d_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[UNet2D, dict[str, int]]:
    checkpoint_obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = _get_state_dict(checkpoint_obj)

    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        base_ch = int(checkpoint_obj.get("base_ch", _infer_base_ch_2d(state_dict)))
        depth = int(checkpoint_obj.get("depth", _infer_depth_from_state(state_dict)))
    else:
        base_ch = _infer_base_ch_2d(state_dict)
        depth = _infer_depth_from_state(state_dict)

    model = UNet2D(base_ch=base_ch, depth=depth).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, {"base_ch": base_ch, "depth": depth}


def build_checkpoint_payload(model: torch.nn.Module, mode: str, base_ch: int, depth: int) -> dict[str, Any]:
    return {
        "mode": mode,
        "base_ch": int(base_ch),
        "depth": int(depth),
        "state_dict": model.state_dict(),
    }

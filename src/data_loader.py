"""
src/data_loader.py
-------------------
Reads the binary files produced by the C++ Monte Carlo engine and builds
a PyTorch Dataset of (noisy, reference) pairs for training the U-Net.

Supports both 1D (v1 format) and 2D (v2 format).
"""

import struct
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Binary I/O
# ──────────────────────────────────────────────────────────────────────────────

def read_bin(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Read a 1D .bin file produced by the C++ engine (v1 format).
    """
    with open(filepath, "rb") as f:
        grid_size  = struct.unpack("d", f.read(8))[0]
        step_size  = struct.unpack("d", f.read(8))[0]
        n_hist     = struct.unpack("i", f.read(4))[0]
        num_bins   = struct.unpack("i", f.read(4))[0]
        data       = np.frombuffer(f.read(num_bins * 8), dtype=np.float64)

    meta = {
        "grid_size_cm": grid_size,
        "step_size_cm": step_size,
        "n_histories":  n_hist,
        "num_bins":     num_bins,
    }
    return data.astype(np.float32), meta

def read_bin_2d(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Read a 2D .bin file produced by the C++ engine (v2 format).
    
    Header layout:
      int32  version = 2
      int32  nx
      int32  ny
      double grid_size_cm
      double step_size_cm
      int32  n_histories
    """
    with open(filepath, "rb") as f:
        version = struct.unpack("i", f.read(4))[0]
        if version != 2:
            raise ValueError(f"Expected v2 binary format, got version {version}")
        
        nx        = struct.unpack("i", f.read(4))[0]
        ny        = struct.unpack("i", f.read(4))[0]
        grid_size = struct.unpack("d", f.read(8))[0]
        step_size = struct.unpack("d", f.read(8))[0]
        n_hist    = struct.unpack("i", f.read(4))[0]
        
        n_elements = nx * ny
        data = np.frombuffer(f.read(n_elements * 8), dtype=np.float64)
        data = data.reshape(ny, nx)

    meta = {
        "version":      version,
        "nx":           nx,
        "ny":           ny,
        "grid_size_cm": grid_size,
        "step_size_cm": step_size,
        "n_histories":  n_hist,
    }
    return data.astype(np.float32), meta


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale by peak dose. Returns (arr_norm, 0.0, max_val)."""
    a_min = arr.min()
    a_max = arr.max()
    if a_max - a_min < 1e-12:
        return np.zeros_like(arr), a_min, a_max
    # Use max-only scaling to preserve physical dose magnitudes.
    return arr / a_max, 0.0, a_max


def denormalize(arr: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    """Reverse scale produced by `normalize`."""
    return arr * (a_max - a_min) + a_min


def normalize_with_reference(
    noisy_arr: np.ndarray,
    ref_arr: np.ndarray,
    clip_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize a noisy/reference pair using the reference maximum.
    Returns (noisy_norm, ref_norm, ref_max).
    """
    ref_norm, _, ref_max = normalize(ref_arr)
    if ref_max < 1e-12:
        noisy_norm = np.zeros_like(noisy_arr)
    else:
        noisy_norm = noisy_arr / ref_max
        if clip_max is not None:
            noisy_norm = np.clip(noisy_norm, 0.0, clip_max)
    return noisy_norm.astype(np.float32), ref_norm.astype(np.float32), float(ref_max)


def pad_or_crop_1d(arr: np.ndarray, target_length: int = 300) -> np.ndarray:
    """Pad or crop a 1D array to target_length."""
    if len(arr) >= target_length:
        return arr[:target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[:len(arr)] = arr
    return padded


def pad_or_crop_2d(arr: np.ndarray, target_ny: int = 100, target_nx: int = 300) -> np.ndarray:
    """Pad or crop a 2D [ny, nx] array to [target_ny, target_nx]."""
    arr = arr[:min(arr.shape[0], target_ny), :min(arr.shape[1], target_nx)]
    if arr.shape[0] < target_ny or arr.shape[1] < target_nx:
        padded = np.zeros((target_ny, target_nx), dtype=np.float32)
        padded[:arr.shape[0], :arr.shape[1]] = arr
        return padded
    return arr


def prepare_pair_2d(
    noisy_raw: np.ndarray,
    ref_raw: np.ndarray,
    target_ny: int = 100,
    target_nx: int = 300,
    clip_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared 2D preprocessing for train/eval/inference.
    Returns padded/cropped (noisy_norm, ref_norm).
    """
    noisy_norm, ref_norm, _ = normalize_with_reference(noisy_raw, ref_raw, clip_max=clip_max)
    return (
        pad_or_crop_2d(noisy_norm, target_ny=target_ny, target_nx=target_nx),
        pad_or_crop_2d(ref_norm, target_ny=target_ny, target_nx=target_nx),
    )


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Datasets
# ──────────────────────────────────────────────────────────────────────────────

class DoseDataset(Dataset):
    """Loads 1D (noisy, reference) dose pairs from a split directory."""

    def __init__(self, split_dir: str, target_length: int = 300):
        self.target_length = target_length
        self.samples: list[tuple[str, str]] = []

        for case_name in sorted(os.listdir(split_dir)):
            case_path = os.path.join(split_dir, case_name)
            if not os.path.isdir(case_path): continue
            n_path, r_path = os.path.join(case_path, "noisy_output.bin"), os.path.join(case_path, "reference_output.bin")
            if os.path.exists(n_path) and os.path.exists(r_path):
                self.samples.append((n_path, r_path))

        if not self.samples: raise FileNotFoundError(f"No 1D cases in: {split_dir}")
        print(f"[DoseDataset 1D] Loaded {len(self.samples)} pairs from '{split_dir}'")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_raw, _ = read_bin(self.samples[idx][0])
        ref_raw,   _ = read_bin(self.samples[idx][1])

        noisy_norm, ref_norm, _ = normalize_with_reference(noisy_raw, ref_raw)

        noisy_norm = pad_or_crop_1d(noisy_norm, target_length=self.target_length)
        ref_norm   = pad_or_crop_1d(ref_norm, target_length=self.target_length)

        return torch.tensor(noisy_norm).unsqueeze(0), torch.tensor(ref_norm).unsqueeze(0)


class DoseDataset2D(Dataset):
    """Loads 2D (noisy, reference) dose pairs from a split directory."""

    def __init__(self, split_dir: str, target_nx: int = 300, target_ny: int = 100):
        self.target_nx = target_nx
        self.target_ny = target_ny
        self.samples: list[tuple[str, str]] = []

        for case_name in sorted(os.listdir(split_dir)):
            case_path = os.path.join(split_dir, case_name)
            if not os.path.isdir(case_path): continue
            n_path, r_path = os.path.join(case_path, "noisy_output.bin"), os.path.join(case_path, "reference_output.bin")
            if os.path.exists(n_path) and os.path.exists(r_path):
                self.samples.append((n_path, r_path))

        if not self.samples: raise FileNotFoundError(f"No 2D cases in: {split_dir}")
        print(f"[DoseDataset 2D] Loaded {len(self.samples)} pairs from '{split_dir}'")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_raw, _ = read_bin_2d(self.samples[idx][0])
        ref_raw,   _ = read_bin_2d(self.samples[idx][1])

        noisy_norm, ref_norm = prepare_pair_2d(
            noisy_raw,
            ref_raw,
            target_ny=self.target_ny,
            target_nx=self.target_nx,
            clip_max=None,
        )

        # [1, H, W]
        return torch.tensor(noisy_norm).unsqueeze(0), torch.tensor(ref_norm).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader Factories
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(data_root: str, batch_size: int = 16, target_length: int = 300, num_workers: int = 0):
    splits = {}
    for split in ("train", "val", "test"):
        dataset = DoseDataset(os.path.join(data_root, split), target_length=target_length)
        splits[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, pin_memory=True)
    return splits["train"], splits["val"], splits["test"]

def build_dataloaders_2d(data_root: str, batch_size: int = 16, target_nx: int = 300, target_ny: int = 100, num_workers: int = 0):
    splits = {}
    for split in ("train", "val", "test"):
        dataset = DoseDataset2D(os.path.join(data_root, split), target_nx=target_nx, target_ny=target_ny)
        splits[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, pin_memory=True)
    return splits["train"], splits["val"], splits["test"]


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pass  # Used mainly as a module

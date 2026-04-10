"""
evaluate.py
-----------
Evaluate the trained model on the held-out test set.
Reports clinically relevant metrics: Range Error (mm), MSE, Max Absolute Error.
In 2D mode, also reports lateral penumbra width error.

Usage:
    .\\venv_projeto\\Scripts\\python.exe evaluate.py --mode 1d
    .\\venv_projeto\\Scripts\\python.exe evaluate.py --mode 2d
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader import DoseDataset, DoseDataset2D
from checkpoint_utils import load_unet1d_checkpoint, load_unet2d_checkpoint
from range_utils import peak_position_soft_numpy
from torch.utils.data import DataLoader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["1d", "2d"], default="1d")
    return p.parse_args()

def calculate_penumbra_width(profile: np.ndarray, dy_cm: float = 0.1) -> float:
    """Calculate 80%-20% penumbra width from a lateral profile."""
    max_dose = profile.max()
    if max_dose < 1e-6: return 0.0
    
    # Left penumbra
    p80_idx = np.argmax(profile >= 0.8 * max_dose)
    p20_idx = np.argmax(profile >= 0.2 * max_dose)
    
    # Right penumbra
    rev_profile = profile[::-1]
    rp80_idx = len(profile) - 1 - np.argmax(rev_profile >= 0.8 * max_dose)
    rp20_idx = len(profile) - 1 - np.argmax(rev_profile >= 0.2 * max_dose)
    
    # Average of left and right, in mm
    w_left  = abs(p80_idx - p20_idx) * dy_cm * 10
    w_right = abs(rp80_idx - rp20_idx) * dy_cm * 10
    
    return (w_left + w_right) / 2.0

def main():
    args = parse_args()
    BASE       = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    range_errors_mm = []
    range_signed_bias_mm = []
    mse_list        = []
    max_err_list    = []
    penumbra_err_list = []

    if args.mode == "1d":
        MODEL_PATH = os.path.join(BASE, "models", "best_unet1d.pth")
        TEST_DIR   = os.path.join(BASE, "data", "train_set", "test")
        SEQ_LEN, GRID_CM = 300, 0.1

        model, _ = load_unet1d_checkpoint(MODEL_PATH, device)

        dataset = DoseDataset(TEST_DIR, target_length=SEQ_LEN)
        loader  = DataLoader(dataset, batch_size=1, shuffle=False)

        depth_axis = np.arange(SEQ_LEN) * GRID_CM

        with torch.no_grad():
            for noisy, ref in loader:
                noisy = noisy.to(device)
                pred  = model(noisy).squeeze().cpu().numpy()
                ref_n = ref.squeeze().numpy()

                bragg_ref  = peak_position_soft_numpy(ref_n) * GRID_CM
                bragg_pred = peak_position_soft_numpy(pred) * GRID_CM
                range_err  = abs(bragg_pred - bragg_ref) * 10
                range_bias = (bragg_pred - bragg_ref) * 10

                mse_list.append(float(np.mean((pred - ref_n)**2)))
                max_err_list.append(float(np.abs(pred - ref_n).max()))
                range_errors_mm.append(range_err)
                range_signed_bias_mm.append(range_bias)

    else:
        MODEL_PATH = os.path.join(BASE, "models", "best_unet2d.pth")
        TEST_DIR   = os.path.join(BASE, "dataset_2d", "test")
        NX, NY, GRID_CM = 300, 100, 0.1

        model, _ = load_unet2d_checkpoint(MODEL_PATH, device)

        dataset = DoseDataset2D(TEST_DIR, target_nx=NX, target_ny=NY)
        loader  = DataLoader(dataset, batch_size=1, shuffle=False)

        depth_axis = np.arange(NX) * GRID_CM
        central_row = NY // 2

        with torch.no_grad():
            for noisy, ref in loader:
                noisy = noisy.to(device)
                # Output is [1, 1, NY, NX]
                pred  = model(noisy).squeeze().cpu().numpy()
                ref_n = ref.squeeze().numpy()

                # Central axis range
                ref_center  = ref_n[central_row, :]
                pred_center = pred[central_row, :]
                bragg_ref   = peak_position_soft_numpy(ref_center) * GRID_CM
                bragg_pred  = peak_position_soft_numpy(pred_center) * GRID_CM
                range_err   = abs(bragg_pred - bragg_ref) * 10
                range_bias  = (bragg_pred - bragg_ref) * 10

                # Penumbra roughly at half-depth
                mid_idx = NX // 2
                ref_lat  = ref_n[:, mid_idx]
                pred_lat = pred[:, mid_idx]
                p_ref  = calculate_penumbra_width(ref_lat, GRID_CM)
                p_pred = calculate_penumbra_width(pred_lat, GRID_CM)
                pen_err = abs(p_pred - p_ref)

                mse_list.append(float(np.mean((pred - ref_n)**2)))
                max_err_list.append(float(np.abs(pred - ref_n).max()))
                range_errors_mm.append(range_err)
                range_signed_bias_mm.append(range_bias)
                penumbra_err_list.append(pen_err)

    print("=" * 60)
    print(f"  TEST SET EVALUATION ({args.mode.upper()}) | {len(range_errors_mm)} cases")
    print("=" * 60)
    print(f"  Mean Range Error   : {np.mean(range_errors_mm):.2f} mm")
    print(f"  Mean Range Bias    : {np.mean(range_signed_bias_mm):.2f} mm")
    print(f"  Max Range Error    : {np.max(range_errors_mm):.2f} mm")
    print(f"  Cases RE < 2 mm    : {sum(e < 2 for e in range_errors_mm)}/{len(range_errors_mm)}")
    print(f"  Mean MSE           : {np.mean(mse_list):.6f}")
    print(f"  Mean Max Abs Error : {np.mean(max_err_list):.4f}")
    if args.mode == "2d":
        print(f"  Mean Penumbra Err  : {np.mean(penumbra_err_list):.2f} mm")
    print("=" * 60)


if __name__ == "__main__":
    main()

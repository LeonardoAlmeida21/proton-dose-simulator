r"""
diagnose_offset.py
------------------
Diagnose systematic depth offset (range bias) without applying any post-hoc shift.

Usage:
  .\venv_projeto\Scripts\python.exe diagnose_offset.py --mode 2d --split val
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader import DoseDataset, DoseDataset2D
from checkpoint_utils import load_unet1d_checkpoint, load_unet2d_checkpoint
from range_utils import peak_position_soft_numpy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["1d", "2d"], default="2d")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_cm = 0.1

    if args.mode == "1d":
        model, _ = load_unet1d_checkpoint(os.path.join(base, "models", "best_unet1d.pth"), device)
        dataset = DoseDataset(os.path.join(base, "data", "train_set", args.split), target_length=300)
    else:
        model, _ = load_unet2d_checkpoint(os.path.join(base, "models", "best_unet2d.pth"), device)
        dataset = DoseDataset2D(os.path.join(base, "dataset_2d", args.split), target_nx=300, target_ny=100)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    signed_bias_mm = []
    abs_error_mm = []
    with torch.no_grad():
        for noisy, ref in loader:
            pred = model(noisy.to(device)).squeeze().cpu().numpy()
            ref_n = ref.squeeze().numpy()

            if args.mode == "1d":
                pred_prof = pred
                ref_prof = ref_n
            else:
                cr = ref_n.shape[0] // 2
                pred_prof = pred[cr, :]
                ref_prof = ref_n[cr, :]

            bragg_pred = peak_position_soft_numpy(pred_prof) * grid_cm
            bragg_ref = peak_position_soft_numpy(ref_prof) * grid_cm
            bias_mm = (bragg_pred - bragg_ref) * 10.0
            signed_bias_mm.append(bias_mm)
            abs_error_mm.append(abs(bias_mm))

    signed_bias_mm = np.array(signed_bias_mm, dtype=np.float32)
    abs_error_mm = np.array(abs_error_mm, dtype=np.float32)
    print("=" * 60)
    print(f"OFFSET DIAGNOSTIC ({args.mode.upper()} | {args.split} | n={len(dataset)})")
    print("=" * 60)
    print(f"Mean signed bias : {signed_bias_mm.mean():.2f} mm")
    print(f"Median bias      : {np.median(signed_bias_mm):.2f} mm")
    print(f"Mean abs error   : {abs_error_mm.mean():.2f} mm")
    print(f"Std signed bias  : {signed_bias_mm.std():.2f} mm")
    print("=" * 60)


if __name__ == "__main__":
    main()

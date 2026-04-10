"""
visualize.py
-------------
Generate physics-validation plots after training the U-Net.
Supports 1D and 2D models.

Usage:
    .\\venv_projeto\\Scripts\\python.exe visualize.py --mode 1d
    .\\venv_projeto\\Scripts\\python.exe visualize.py --mode 2d
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader import read_bin, read_bin_2d, normalize_with_reference, pad_or_crop_1d, prepare_pair_2d
from checkpoint_utils import load_unet1d_checkpoint, load_unet2d_checkpoint

STYLE = {
    "noisy":  dict(color="#e07b54", lw=1.2, alpha=0.7, label="MC Noisy"),
    "pred":   dict(color="#5ab4ac", lw=2.0, alpha=0.9, label="U-Net Denoised"),
    "ref":    dict(color="#2c7fb8", lw=2.5, alpha=1.0, label="Ground Truth", ls="--"),
    "error":  dict(color="#fc8d59", lw=1.5, alpha=0.85, label="|Pred - Ref|"),
    "grad_p": dict(color="#5ab4ac", lw=1.5, alpha=0.85, label="Pred Gradient"),
    "grad_r": dict(color="#2c7fb8", lw=1.5, alpha=0.85, label="Ref Gradient", ls="--"),
}

def parse_args():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["1d", "2d"], default="1d")
    p.add_argument("--case", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "1d":
        case_dir = args.case or os.path.join(base, "data", "train_set", "test", "case_0000")
        out_path = args.output or os.path.join(base, "figures", "bragg_peak_validation.png")
        model, cfg = load_unet1d_checkpoint(os.path.join(base, "models", "best_unet1d.pth"), device)
        print(f"Loaded UNet1D checkpoint (base_ch={cfg['base_ch']}, depth={cfg['depth']})")

        noisy_raw, meta = read_bin(os.path.join(case_dir, "noisy_output.bin"))
        ref_raw, _      = read_bin(os.path.join(case_dir, "reference_output.bin"))
        dx = meta["grid_size_cm"]

        noisy_norm, ref_norm, _ = normalize_with_reference(noisy_raw, ref_raw, clip_max=None)
        noisy = pad_or_crop_1d(noisy_norm, target_length=300)
        ref   = pad_or_crop_1d(ref_norm, target_length=300)
        
        with torch.no_grad():
            inp = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(inp).squeeze().cpu().numpy()

        depth = np.arange(300) * dx
        fig = plt.figure(figsize=(14, 10), facecolor="#0f1117")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(depth, noisy, **STYLE["noisy"])
        ax1.plot(depth, pred,  **STYLE["pred"])
        ax1.plot(depth, ref,   **STYLE["ref"])
        ax1.set_title("1D Dose Profile", color="white", fontsize=13)
        ax1.set_facecolor("#1a1d27")
        ax1.tick_params(colors="#aaaaaa")
        ax1.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(depth, np.abs(pred - ref), **STYLE["error"])
        ax2.set_title("Residual Error", color="white")
        ax2.set_facecolor("#1a1d27")
        ax2.tick_params(colors="#aaaaaa")

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(depth[:-1], np.diff(pred)/dx, **STYLE["grad_p"])
        ax3.plot(depth[:-1], np.diff(ref)/dx,  **STYLE["grad_r"])
        ax3.set_title("Distal Gradient", color="white")
        ax3.set_facecolor("#1a1d27")
        ax3.tick_params(colors="#aaaaaa")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {out_path}")

    else:
        case_dir = args.case or os.path.join(base, "dataset_2d", "test", "case_0240")
        out_path = args.output or os.path.join(base, "figures", "bragg_peak_validation_2d.png")
        model, cfg = load_unet2d_checkpoint(os.path.join(base, "models", "best_unet2d.pth"), device)
        print(f"Loaded UNet2D checkpoint (base_ch={cfg['base_ch']}, depth={cfg['depth']})")

        noisy_raw, meta = read_bin_2d(os.path.join(case_dir, "noisy_output.bin"))
        ref_raw, _      = read_bin_2d(os.path.join(case_dir, "reference_output.bin"))
        dx, dy = meta["grid_size_cm"], meta["step_size_cm"]  # step_size_cm holds dy? Wait, v2 format dx=dy

        noisy, ref = prepare_pair_2d(
            noisy_raw,
            ref_raw,
            target_ny=100,
            target_nx=300,
            clip_max=None,
        )

        with torch.no_grad():
            inp = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(inp).squeeze().cpu().numpy()

        depth = np.arange(300) * dx
        cr = 50

        fig = plt.figure(figsize=(15, 12), facecolor="#0f1117")
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2)
        
        def format_map(ax, arr, title):
            im = ax.imshow(arr, aspect="auto", extent=[0, 300*dx, 100*dx, 0], cmap="inferno", vmin=0, vmax=1.0)
            ax.set_title(title, color="white")
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="#aaaaaa")
            for spine in ax.spines.values(): spine.set_edgecolor("#333344")
            return im

        # Heatmaps
        format_map(fig.add_subplot(gs[0, 0]), noisy, "Noisy 2D Dose")
        format_map(fig.add_subplot(gs[0, 1]), pred,  "Predicted 2D Dose")
        im = format_map(fig.add_subplot(gs[1, 0]), ref, "Reference 2D Dose")
        err_ax = fig.add_subplot(gs[1, 1])
        im_err = format_map(err_ax, np.abs(pred - ref), "Absolute Error")
        
        # Central Profile
        ax_prof = fig.add_subplot(gs[2, :])
        ax_prof.plot(depth, noisy[cr, :], **STYLE["noisy"])
        ax_prof.plot(depth, pred[cr, :],  **STYLE["pred"])
        ax_prof.plot(depth, ref[cr, :],   **STYLE["ref"])
        ax_prof.set_title("Central Axis Profile", color="white")
        ax_prof.set_facecolor("#1a1d27")
        ax_prof.tick_params(colors="#aaaaaa")
        ax_prof.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

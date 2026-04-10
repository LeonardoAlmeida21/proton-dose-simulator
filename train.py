"""
train.py
---------
Full training loop for the 1D and 2D U-Net proton dose denoising models.

Usage:
    .\\venv_projeto\\Scripts\\python.exe train.py --mode 1d --epochs 100
    .\\venv_projeto\\Scripts\\python.exe train.py --mode 2d --epochs 50 --batch_size 4
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src/ to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model          import UNet1D, UNet2D
from loss_functions import PhysicsInformedLoss, PhysicsInformedLoss2D
from data_loader    import build_dataloaders, build_dataloaders_2d
from checkpoint_utils import build_checkpoint_payload
from range_utils import peak_position_soft_numpy
from benchmark_framework import load_framework_config, clinical_composite_score


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the U-Net denoiser")
    p.add_argument("--mode",         type=str,   choices=["1d", "2d"], default="1d",
                   help="Mode: train the 1D model or the new 2D model")
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--base_ch",      type=int,   default=None,
                   help="Base channel count for the U-Net")
    p.add_argument("--depth",        type=int,   default=None)
    p.add_argument("--seq_len",      type=int,   default=300,
                   help="Fixed-length dose vector size (1D) or Nx (2D)")
    p.add_argument("--data_root",    type=str,   default=None)
    p.add_argument("--models_dir",   type=str,
                   default=os.path.join(os.path.dirname(__file__), "models"))
    p.add_argument("--framework_config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "framework_config.json"),
                   help="Path to framework config (score/gates contract)")
    # Loss Weights
    p.add_argument("--w_mse",        type=float, default=1.0)
    p.add_argument("--w_range",      type=float, default=None)
    p.add_argument("--w_grad",       type=float, default=0.5)
    p.add_argument("--w_pen",        type=float, default=2.0,
                   help="Weight for penumbra preservation loss (2D only)")
    p.add_argument("--w_bias",       type=float, default=2.0,
                   help="Weight for systematic range-bias loss (2D only)")
    p.add_argument("--w_distal",     type=float, default=2.0,
                   help="Weight for distal-edge preservation loss (2D only)")
    p.add_argument("--mse_focus_2d", type=float, default=20.0,
                   help="Extra MSE focus on high-dose region (2D only)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Train / Validate (single epoch)
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    is_train: bool,
    mode: str
) -> dict:
    model.train(is_train)
    totals = {}

    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for noisy, ref in loader:
            noisy, ref = noisy.to(device), ref.to(device)

            pred = model(noisy)
            loss, breakdown = loss_fn(pred, ref)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def calculate_penumbra_width(profile: np.ndarray, dy_cm: float = 0.1) -> float:
    max_dose = profile.max()
    if max_dose < 1e-6:
        return 0.0

    p80_idx = np.argmax(profile >= 0.8 * max_dose)
    p20_idx = np.argmax(profile >= 0.2 * max_dose)

    rev_profile = profile[::-1]
    rp80_idx = len(profile) - 1 - np.argmax(rev_profile >= 0.8 * max_dose)
    rp20_idx = len(profile) - 1 - np.argmax(rev_profile >= 0.2 * max_dose)

    w_left = abs(p80_idx - p20_idx) * dy_cm * 10
    w_right = abs(rp80_idx - rp20_idx) * dy_cm * 10
    return (w_left + w_right) / 2.0


def clinical_metrics(model, loader, device, mode: str, grid_cm: float = 0.1) -> dict:
    model.eval()
    range_err = []
    range_bias = []
    mse = []
    pen = []

    with torch.no_grad():
        for noisy, ref in loader:
            noisy = noisy.to(device)
            pred_batch = model(noisy).detach().cpu().numpy()
            ref_batch = ref.numpy()

            if mode == "1d":
                pred_batch = pred_batch[:, 0, :]
                ref_batch = ref_batch[:, 0, :]
                for pred, ref_n in zip(pred_batch, ref_batch):
                    bragg_ref = peak_position_soft_numpy(ref_n) * grid_cm
                    bragg_pred = peak_position_soft_numpy(pred) * grid_cm
                    range_err.append(abs(bragg_pred - bragg_ref) * 10)
                    range_bias.append((bragg_pred - bragg_ref) * 10)
                    mse.append(float(np.mean((pred - ref_n) ** 2)))
            else:
                pred_batch = pred_batch[:, 0, :, :]
                ref_batch = ref_batch[:, 0, :, :]
                for pred, ref_n in zip(pred_batch, ref_batch):
                    ny, nx = ref_n.shape
                    cr = ny // 2
                    ref_center = ref_n[cr, :]
                    pred_center = pred[cr, :]
                    bragg_ref = peak_position_soft_numpy(ref_center) * grid_cm
                    bragg_pred = peak_position_soft_numpy(pred_center) * grid_cm
                    range_err.append(abs(bragg_pred - bragg_ref) * 10)
                    range_bias.append((bragg_pred - bragg_ref) * 10)
                    mse.append(float(np.mean((pred - ref_n) ** 2)))

                    mid_idx = nx // 2
                    pen_ref = calculate_penumbra_width(ref_n[:, mid_idx], dy_cm=grid_cm)
                    pen_pred = calculate_penumbra_width(pred[:, mid_idx], dy_cm=grid_cm)
                    pen.append(abs(pen_pred - pen_ref))

    out = {
        "range_error_mm": float(np.mean(range_err)),
        "range_bias_mm": float(np.mean(range_bias)),
        "mse": float(np.mean(mse)),
        "percent_cases_re_below_2mm": float(np.mean(np.array(range_err) < 2.0)),
    }
    if mode == "2d":
        out["penumbra_error_mm"] = float(np.mean(pen))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    framework_cfg = load_framework_config(args.framework_config)

    if args.base_ch is None:
        args.base_ch = 32 if args.mode == "1d" else 16
    if args.depth is None:
        args.depth = 4 if args.mode == "1d" else 2
    if args.w_range is None:
        args.w_range = 2.0 if args.mode == "1d" else 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.data_root is None:
        if args.mode == "1d":
            args.data_root = os.path.join(os.path.dirname(__file__), "data", "train_set")
        else:
            args.data_root = os.path.join(os.path.dirname(__file__), "dataset_2d")

    print(f"\nMode: {args.mode.upper()} | Loading dataset from {args.data_root}...")

    if args.mode == "1d":
        train_loader, val_loader, _ = build_dataloaders(args.data_root, args.batch_size, args.seq_len)
        model = UNet1D(base_ch=args.base_ch, depth=args.depth).to(device)
        loss_fn = PhysicsInformedLoss(args.w_mse, args.w_range, args.w_grad)
        model_name = "unet1d"
    else:
        # Override batch size for 2D since it takes more memory
        bs = min(args.batch_size, 8) if args.batch_size > 8 else args.batch_size
        train_loader, val_loader, _ = build_dataloaders_2d(args.data_root, bs, target_nx=args.seq_len, target_ny=100)
        model = UNet2D(base_ch=args.base_ch, depth=args.depth).to(device)
        loss_fn = PhysicsInformedLoss2D(
            args.w_mse,
            args.w_range,
            args.w_grad,
            args.w_pen,
            args.w_bias,
            args.w_distal,
            args.mse_focus_2d,
        )
        model_name = "unet2d"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name.upper()} | Parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_clinical_score = float("inf")
    print(f"\nStarting training: {args.epochs} epochs\n" + "-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, is_train=True, mode=args.mode)
        val_metrics   = run_epoch(model, val_loader,   loss_fn, optimizer, device, is_train=False, mode=args.mode)
        val_clinical = clinical_metrics(model, val_loader, device, mode=args.mode)
        clinical_score = clinical_composite_score(val_clinical, mode=args.mode, config=framework_cfg)

        scheduler.step()
        elapsed = time.time() - t0

        val_loss = val_metrics["total"]
        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                build_checkpoint_payload(model, args.mode, args.base_ch, args.depth),
                os.path.join(args.models_dir, f"best_loss_{model_name}.pth"),
            )

        if clinical_score < best_clinical_score:
            best_clinical_score = clinical_score
            torch.save(
                build_checkpoint_payload(model, args.mode, args.base_ch, args.depth),
                os.path.join(args.models_dir, f"best_{model_name}.pth"),
            )
            flag = " * best_clinical"

        if args.mode == "1d":
            print(
                f"[{epoch:03d}/{args.epochs}] "
                f"T={train_metrics['total']:.4f} (ms={train_metrics['mse']:.4f} rg={train_metrics['range']:.4f}) | "
                f"V={val_metrics['total']:.4f} "
                f"C(re={val_clinical['range_error_mm']:.2f} rb={val_clinical['range_bias_mm']:.2f} mse={val_clinical['mse']:.4f}) "
                f"t={elapsed:.1f}s{flag}"
            )
        else:
            print(
                f"[{epoch:03d}/{args.epochs}] "
                f"T={train_metrics['total']:.4f} (ms={train_metrics['mse']:.4f} rg={train_metrics['range']:.4f} "
                f"pn={train_metrics['penumbra']:.4f} bs={train_metrics['bias']:.4f} ds={train_metrics['distal']:.4f}) | "
                f"V={val_metrics['total']:.4f} "
                f"C(re={val_clinical['range_error_mm']:.2f} rb={val_clinical['range_bias_mm']:.2f} "
                f"pe={val_clinical['penumbra_error_mm']:.2f} mse={val_clinical['mse']:.4f}) "
                f"t={elapsed:.1f}s{flag}"
            )

    torch.save(
        build_checkpoint_payload(model, args.mode, args.base_ch, args.depth),
        os.path.join(args.models_dir, f"last_{model_name}.pth"),
    )
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.5f} | Best clinical score: {best_clinical_score:.5f}")


if __name__ == "__main__":
    main()

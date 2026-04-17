"""Detailed clinical analysis for 2D UNet checkpoints.

This script reports:
- legacy metrics (same definition used in evaluate.py/benchmark_unet.py)
- robust Bragg alignment metrics (beam-window depth profile)
- robust penumbra metrics (adaptive depth and local threshold crossings)
- per-split and per-energy-band summaries
- worst-case table for fast root-cause inspection
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from checkpoint_utils import load_unet2d_checkpoint
from data_loader import prepare_pair_2d, read_bin_2d
from range_utils import peak_position_soft_numpy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detailed UNet2D clinical diagnostics.")
    p.add_argument("--checkpoint", type=str, default=os.path.join(BASE_DIR, "models", "best_unet2d.pth"))
    p.add_argument("--manifest", type=str, default=os.path.join(BASE_DIR, "golden_set_2d.json"))
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--window_rows", type=int, default=7, help="Rows used for robust beam-window depth profile.")
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def energy_band(energy_mev: float) -> str:
    if energy_mev < 110.0:
        return "low"
    if energy_mev < 170.0:
        return "medium"
    return "high"


def penumbra_width_legacy(profile: np.ndarray, dy_cm: float = 0.1) -> float:
    max_dose = float(profile.max())
    if max_dose < 1e-8:
        return 0.0
    p80_idx = int(np.argmax(profile >= 0.8 * max_dose))
    p20_idx = int(np.argmax(profile >= 0.2 * max_dose))
    rev_profile = profile[::-1]
    rp80_idx = len(profile) - 1 - int(np.argmax(rev_profile >= 0.8 * max_dose))
    rp20_idx = len(profile) - 1 - int(np.argmax(rev_profile >= 0.2 * max_dose))
    w_left = abs(p80_idx - p20_idx) * dy_cm * 10.0
    w_right = abs(rp80_idx - rp20_idx) * dy_cm * 10.0
    return (w_left + w_right) * 0.5


def penumbra_width_local(profile: np.ndarray, dy_cm: float = 0.1, min_peak: float = 1e-6) -> float:
    p = np.asarray(profile, dtype=np.float64)
    peak = float(p.max())
    if peak < min_peak:
        return 0.0

    ip = int(np.argmax(p))
    t80 = 0.8 * peak
    t20 = 0.2 * peak

    l80 = ip
    while l80 > 0 and p[l80] >= t80:
        l80 -= 1
    r80 = ip
    while r80 < len(p) - 1 and p[r80] >= t80:
        r80 += 1

    l20 = ip
    while l20 > 0 and p[l20] >= t20:
        l20 -= 1
    r20 = ip
    while r20 < len(p) - 1 and p[r20] >= t20:
        r20 += 1

    w_left = abs(l80 - l20) * dy_cm * 10.0
    w_right = abs(r80 - r20) * dy_cm * 10.0
    return (w_left + w_right) * 0.5


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def band_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["energy_band"]].append(row)

    out: dict[str, dict[str, float]] = {}
    for band, items in sorted(grouped.items()):
        abs_mm = [float(x["bragg_abs_mm_robust"]) for x in items]
        out[band] = {
            "n": float(len(items)),
            "bragg_abs_mm_mean": float(np.mean(abs_mm)),
            "bragg_abs_mm_p95": float(np.percentile(np.asarray(abs_mm, dtype=np.float64), 95)),
            "percent_bragg_abs_lt_2mm": float(np.mean(np.asarray(abs_mm) < 2.0)),
        }
    return out


def analyze_cases(
    cases: list[dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    window_rows: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    half = max(1, window_rows // 2)
    for c in cases:
        case_dir = os.path.join(BASE_DIR, c["case_path"])
        noisy_raw, _ = read_bin_2d(os.path.join(case_dir, "noisy_output.bin"))
        ref_raw, _ = read_bin_2d(os.path.join(case_dir, "reference_output.bin"))
        noisy, ref = prepare_pair_2d(noisy_raw, ref_raw, target_ny=100, target_nx=300, clip_max=None)

        inp = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp).squeeze().cpu().numpy()

        ny, nx = ref.shape
        center_row = ny // 2
        legacy_ref_prof = ref[center_row, :]
        legacy_pred_prof = pred[center_row, :]
        legacy_noisy_prof = noisy[center_row, :]
        br_ref_legacy = peak_position_soft_numpy(legacy_ref_prof)
        br_pred_legacy = peak_position_soft_numpy(legacy_pred_prof)
        br_noisy_legacy = peak_position_soft_numpy(legacy_noisy_prof)

        y_star = int(np.argmax(ref.sum(axis=1)))
        y0 = max(0, y_star - half)
        y1 = min(ny, y_star + half + 1)
        robust_ref_prof = ref[y0:y1, :].sum(axis=0)
        robust_pred_prof = pred[y0:y1, :].sum(axis=0)
        robust_noisy_prof = noisy[y0:y1, :].sum(axis=0)
        br_ref_robust = peak_position_soft_numpy(robust_ref_prof)
        br_pred_robust = peak_position_soft_numpy(robust_pred_prof)
        br_noisy_robust = peak_position_soft_numpy(robust_noisy_prof)

        depth_idx = int(np.clip(round(0.7 * br_ref_robust), 0, nx - 1))
        pen_ref_legacy = penumbra_width_legacy(ref[:, nx // 2])
        pen_pred_legacy = penumbra_width_legacy(pred[:, nx // 2])
        pen_noisy_legacy = penumbra_width_legacy(noisy[:, nx // 2])
        pen_ref_robust = penumbra_width_local(ref[:, depth_idx])
        pen_pred_robust = penumbra_width_local(pred[:, depth_idx])
        pen_noisy_robust = penumbra_width_local(noisy[:, depth_idx])

        out.append(
            {
                "split": c["split"],
                "case_id": c["case_id"],
                "energy_mev": float(c["energy_mev"]),
                "energy_band": energy_band(float(c["energy_mev"])),
                "heterogeneity_score": float(c["heterogeneity_score"]),
                "shift_magnitude_cm": float(c["shift_magnitude_cm"]),
                "bragg_signed_mm_legacy": float(br_pred_legacy - br_ref_legacy),
                "bragg_abs_mm_legacy": float(abs(br_pred_legacy - br_ref_legacy)),
                "bragg_signed_mm_robust": float(br_pred_robust - br_ref_robust),
                "bragg_abs_mm_robust": float(abs(br_pred_robust - br_ref_robust)),
                "bragg_abs_mm_noisy_legacy": float(abs(br_noisy_legacy - br_ref_legacy)),
                "bragg_abs_mm_noisy_robust": float(abs(br_noisy_robust - br_ref_robust)),
                "penumbra_err_mm_legacy": float(abs(pen_pred_legacy - pen_ref_legacy)),
                "penumbra_err_mm_robust": float(abs(pen_pred_robust - pen_ref_robust)),
                "penumbra_err_mm_noisy_legacy": float(abs(pen_noisy_legacy - pen_ref_legacy)),
                "penumbra_err_mm_noisy_robust": float(abs(pen_noisy_robust - pen_ref_robust)),
            }
        )
    return out


def split_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    robust_abs = [float(r["bragg_abs_mm_robust"]) for r in rows]
    robust_signed = [float(r["bragg_signed_mm_robust"]) for r in rows]
    legacy_abs = [float(r["bragg_abs_mm_legacy"]) for r in rows]
    noisy_robust_abs = [float(r["bragg_abs_mm_noisy_robust"]) for r in rows]
    noisy_legacy_abs = [float(r["bragg_abs_mm_noisy_legacy"]) for r in rows]
    pen_legacy = [float(r["penumbra_err_mm_legacy"]) for r in rows]
    pen_robust = [float(r["penumbra_err_mm_robust"]) for r in rows]
    pen_noisy_legacy = [float(r["penumbra_err_mm_noisy_legacy"]) for r in rows]
    pen_noisy_robust = [float(r["penumbra_err_mm_noisy_robust"]) for r in rows]
    return {
        "n_cases": int(len(rows)),
        "bragg_abs_mm_legacy": summarize(legacy_abs),
        "bragg_abs_mm_robust": summarize(robust_abs),
        "bragg_abs_mm_noisy_legacy": summarize(noisy_legacy_abs),
        "bragg_abs_mm_noisy_robust": summarize(noisy_robust_abs),
        "bragg_signed_mm_robust": summarize(robust_signed),
        "percent_robust_bragg_abs_lt_1mm": float(np.mean(np.asarray(robust_abs) < 1.0)),
        "percent_robust_bragg_abs_lt_2mm": float(np.mean(np.asarray(robust_abs) < 2.0)),
        "percent_robust_bragg_abs_lt_3mm": float(np.mean(np.asarray(robust_abs) < 3.0)),
        "penumbra_err_mm_legacy_mean": float(np.mean(np.asarray(pen_legacy))),
        "penumbra_err_mm_robust_mean": float(np.mean(np.asarray(pen_robust))),
        "penumbra_err_mm_noisy_legacy_mean": float(np.mean(np.asarray(pen_noisy_legacy))),
        "penumbra_err_mm_noisy_robust_mean": float(np.mean(np.asarray(pen_noisy_robust))),
        "by_energy_band": band_table(rows),
    }


def main() -> None:
    args = parse_args()
    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
    all_cases = manifest["cases"]
    if args.split != "all":
        all_cases = [c for c in all_cases if c["split"] == args.split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, arch = load_unet2d_checkpoint(args.checkpoint, device)

    rows = analyze_cases(all_cases, model, device, window_rows=args.window_rows)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    report: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "arch": arch,
        "window_rows": args.window_rows,
        "summary": {split: split_report(srows) for split, srows in sorted(by_split.items())},
        "worst_robust_bragg_cases": sorted(rows, key=lambda r: float(r["bragg_abs_mm_robust"]), reverse=True)[:20],
        "worst_robust_penumbra_cases": sorted(rows, key=lambda r: float(r["penumbra_err_mm_robust"]), reverse=True)[:20],
    }

    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Arch      : {arch}")
    print(f"Rows win  : {args.window_rows}")
    print("=" * 80)
    for split, s in sorted(report["summary"].items()):
        print(f"[{split.upper()}] n={s['n_cases']}")
        print(
            "  Bragg abs mm (legacy/robust): "
            f"{s['bragg_abs_mm_legacy']['mean']:.3f} / {s['bragg_abs_mm_robust']['mean']:.3f}"
        )
        print(
            "  Bragg abs mm noisy (legacy/robust): "
            f"{s['bragg_abs_mm_noisy_legacy']['mean']:.3f} / {s['bragg_abs_mm_noisy_robust']['mean']:.3f}"
        )
        print(
            "  Robust Bragg success: "
            f"<1mm={100*s['percent_robust_bragg_abs_lt_1mm']:.1f}% "
            f"<2mm={100*s['percent_robust_bragg_abs_lt_2mm']:.1f}% "
            f"<3mm={100*s['percent_robust_bragg_abs_lt_3mm']:.1f}%"
        )
        print(
            "  Penumbra err mean mm (legacy/robust): "
            f"{s['penumbra_err_mm_legacy_mean']:.3f} / {s['penumbra_err_mm_robust_mean']:.3f}"
        )
        print(
            "  Penumbra noisy mean mm (legacy/robust): "
            f"{s['penumbra_err_mm_noisy_legacy_mean']:.3f} / {s['penumbra_err_mm_noisy_robust_mean']:.3f}"
        )
    print("=" * 80)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()

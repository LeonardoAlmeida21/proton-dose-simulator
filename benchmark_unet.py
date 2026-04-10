"""Benchmark U-Net checkpoints on the frozen 2D golden set."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from data_loader import read_bin_2d, prepare_pair_2d
from checkpoint_utils import load_unet2d_checkpoint
from benchmark_framework import (
    load_framework_config,
    clinical_composite_score,
    evaluate_acceptance_gates,
    stratify_case,
)
from range_utils import peak_position_soft_numpy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark U-Net on the frozen 2D golden set.")
    p.add_argument("--checkpoint", type=str, default=os.path.join(BASE_DIR, "models", "best_unet2d.pth"))
    p.add_argument("--baseline_checkpoint", type=str, default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--mode", choices=["summary", "full", "stratified"], default="summary")
    p.add_argument("--framework_config", type=str, default=os.path.join(BASE_DIR, "framework_config.json"))
    p.add_argument("--golden_manifest", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "benchmark_outputs"))
    p.add_argument("--batch", action="store_true", help="Run current, baseline and diff decision.")
    return p.parse_args()


def load_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def penumbra_width(profile: np.ndarray, dy_cm: float = 0.1) -> float:
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


def case_metrics(pred: np.ndarray, ref: np.ndarray, grid_cm: float = 0.1) -> dict[str, float]:
    ny, nx = ref.shape
    central_row = ny // 2
    depth_pred = pred[central_row, :]
    depth_ref = ref[central_row, :]

    bragg_pred_cm = peak_position_soft_numpy(depth_pred) * grid_cm
    bragg_ref_cm = peak_position_soft_numpy(depth_ref) * grid_cm
    range_bias_mm = (bragg_pred_cm - bragg_ref_cm) * 10.0
    range_error_mm = abs(range_bias_mm)

    mid_idx = nx // 2
    pen_pred_mm = penumbra_width(pred[:, mid_idx], dy_cm=grid_cm)
    pen_ref_mm = penumbra_width(ref[:, mid_idx], dy_cm=grid_cm)
    pen_err_mm = abs(pen_pred_mm - pen_ref_mm)

    mse = float(np.mean((pred - ref) ** 2))
    max_abs = float(np.max(np.abs(pred - ref)))
    return {
        "range_error_mm": float(range_error_mm),
        "range_bias_mm": float(range_bias_mm),
        "penumbra_error_mm": float(pen_err_mm),
        "mse": float(mse),
        "max_abs_error": float(max_abs),
    }


def aggregate_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    if not rows:
        raise ValueError(f"No rows to aggregate for prefix '{prefix}'.")

    re = np.array([float(r[f"{prefix}_range_error_mm"]) for r in rows], dtype=np.float64)
    rb = np.array([float(r[f"{prefix}_range_bias_mm"]) for r in rows], dtype=np.float64)
    pe = np.array([float(r[f"{prefix}_penumbra_error_mm"]) for r in rows], dtype=np.float64)
    mse = np.array([float(r[f"{prefix}_mse"]) for r in rows], dtype=np.float64)
    mae = np.array([float(r[f"{prefix}_max_abs_error"]) for r in rows], dtype=np.float64)

    return {
        "range_error_mm": float(re.mean()),
        "range_bias_mm": float(rb.mean()),
        "abs_range_bias_mm": float(np.abs(rb).mean()),
        "penumbra_error_mm": float(pe.mean()),
        "mse": float(mse.mean()),
        "max_abs_error": float(mae.mean()),
        "percent_cases_re_below_2mm": float((re < 2.0).mean()),
        "num_cases": int(len(rows)),
    }


def stratified_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("energy_band", "shift_band", "heterogeneity_band"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(str(r[key]), []).append(r)
        out[key] = {label: aggregate_metrics(group_rows, prefix) for label, group_rows in sorted(groups.items())}
    return out


def write_cases_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def append_experiment_log(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    fieldnames = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_model_on_cases(
    cases: list[dict[str, Any]],
    model: torch.nn.Module | None,
    device: torch.device,
    framework_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_dir = os.path.join(BASE_DIR, case["case_path"])
        noisy_raw, _ = read_bin_2d(os.path.join(case_dir, "noisy_output.bin"))
        ref_raw, _ = read_bin_2d(os.path.join(case_dir, "reference_output.bin"))
        noisy, ref = prepare_pair_2d(noisy_raw, ref_raw, target_ny=100, target_nx=300, clip_max=None)

        row: dict[str, Any] = dict(case)
        row.update(stratify_case(case, framework_cfg))

        noisy_m = case_metrics(noisy, ref)
        for k, v in noisy_m.items():
            row[f"noisy_{k}"] = float(v)

        if model is None:
            pred = noisy
        else:
            inp = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).squeeze().cpu().numpy()
        cur_m = case_metrics(pred, ref)
        for k, v in cur_m.items():
            row[f"current_{k}"] = float(v)
            row[f"delta_current_vs_noisy_{k}"] = float(cur_m[k] - noisy_m[k])

        rows.append(row)
    return rows


def add_baseline_columns(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    baseline_model: torch.nn.Module,
    device: torch.device,
) -> None:
    per_case = {r["case_id"]: r for r in rows}
    for case in cases:
        case_dir = os.path.join(BASE_DIR, case["case_path"])
        noisy_raw, _ = read_bin_2d(os.path.join(case_dir, "noisy_output.bin"))
        ref_raw, _ = read_bin_2d(os.path.join(case_dir, "reference_output.bin"))
        noisy, ref = prepare_pair_2d(noisy_raw, ref_raw, target_ny=100, target_nx=300, clip_max=None)

        inp = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = baseline_model(inp).squeeze().cpu().numpy()
        base_m = case_metrics(pred, ref)
        row = per_case[case["case_id"]]
        for k, v in base_m.items():
            row[f"baseline_{k}"] = float(v)
            row[f"delta_current_vs_baseline_{k}"] = float(row[f"current_{k}"] - v)


def main():
    args = parse_args()
    framework_cfg = load_framework_config(args.framework_config)
    manifest_path = args.golden_manifest or os.path.join(BASE_DIR, framework_cfg["golden_set_manifest"])
    manifest = load_manifest(manifest_path)

    selected_cases = [c for c in manifest["cases"] if c["split"] == args.split]
    if not selected_cases:
        raise RuntimeError(f"No cases for split '{args.split}' in manifest '{manifest_path}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model, current_cfg = load_unet2d_checkpoint(args.checkpoint, device)

    rows = run_model_on_cases(selected_cases, current_model, device, framework_cfg)
    current_summary = aggregate_metrics(rows, "current")
    noisy_summary = aggregate_metrics(rows, "noisy")
    current_score = clinical_composite_score(current_summary, mode="2d", config=framework_cfg)
    noisy_score = clinical_composite_score(noisy_summary, mode="2d", config=framework_cfg)
    gates_pass, gates_detail = evaluate_acceptance_gates(current_summary, mode="2d", config=framework_cfg)

    baseline_summary = None
    baseline_score = None
    if args.baseline_checkpoint:
        baseline_model, _ = load_unet2d_checkpoint(args.baseline_checkpoint, device)
        add_baseline_columns(rows, selected_cases, baseline_model, device)
        baseline_summary = aggregate_metrics(rows, "baseline")
        baseline_score = clinical_composite_score(baseline_summary, mode="2d", config=framework_cfg)

    decision = {
        "gates_pass": bool(gates_pass),
        "improves_vs_noisy": bool(current_score < noisy_score),
    }
    if baseline_score is not None:
        decision["improves_vs_baseline"] = bool(current_score < baseline_score)
    decision["overall_pass"] = bool(all(decision.values()))

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "framework_config_version": framework_cfg["version"],
        "manifest_path": manifest_path,
        "manifest_version": manifest.get("manifest_version", "unknown"),
        "split": args.split,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "checkpoint_arch": current_cfg,
        "current": {
            "metrics": current_summary,
            "composite_score": current_score,
            "gates": gates_detail,
        },
        "noisy_baseline": {
            "metrics": noisy_summary,
            "composite_score": noisy_score,
        },
        "decision": decision,
    }

    if baseline_summary is not None:
        summary["baseline_checkpoint"] = args.baseline_checkpoint
        summary["baseline"] = {
            "metrics": baseline_summary,
            "composite_score": baseline_score,
        }

    if args.mode in ("stratified", "full"):
        summary["stratified_current"] = stratified_summary(rows, "current")
        summary["stratified_noisy"] = stratified_summary(rows, "noisy")
        if baseline_summary is not None:
            summary["stratified_baseline"] = stratified_summary(rows, "baseline")

    if args.batch:
        diff = {
            "current_minus_noisy_score": float(current_score - noisy_score),
            "current_minus_noisy_range_error_mm": float(current_summary["range_error_mm"] - noisy_summary["range_error_mm"]),
            "current_minus_noisy_penumbra_error_mm": float(current_summary["penumbra_error_mm"] - noisy_summary["penumbra_error_mm"]),
        }
        if baseline_summary is not None:
            diff["current_minus_baseline_score"] = float(current_score - baseline_score)
            diff["current_minus_baseline_range_error_mm"] = float(
                current_summary["range_error_mm"] - baseline_summary["range_error_mm"]
            )
            diff["current_minus_baseline_penumbra_error_mm"] = float(
                current_summary["penumbra_error_mm"] - baseline_summary["penumbra_error_mm"]
            )
        summary["diff"] = diff

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    cases_path = os.path.join(args.output_dir, "benchmark_cases.csv")
    log_path = os.path.join(args.output_dir, "experiments_log.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_cases_csv(cases_path, rows)

    log_row = {
        "timestamp_utc": summary["timestamp_utc"],
        "framework_version": framework_cfg["version"],
        "split": args.split,
        "checkpoint": args.checkpoint,
        "score_current": float(current_score),
        "score_noisy": float(noisy_score),
        "range_error_mm": float(current_summary["range_error_mm"]),
        "range_bias_mm": float(current_summary["range_bias_mm"]),
        "penumbra_error_mm": float(current_summary["penumbra_error_mm"]),
        "mse": float(current_summary["mse"]),
        "percent_cases_re_below_2mm": float(current_summary["percent_cases_re_below_2mm"]),
        "gates_pass": int(gates_pass),
        "overall_pass": int(decision["overall_pass"]),
    }
    if baseline_score is not None:
        log_row["baseline_checkpoint"] = args.baseline_checkpoint
        log_row["score_baseline"] = float(baseline_score)
    append_experiment_log(log_path, log_row)

    print("=" * 72)
    print("Benchmark complete")
    print(f"Summary: {summary_path}")
    print(f"Cases  : {cases_path}")
    print(f"Log    : {log_path}")
    print("-" * 72)
    print(
        f"Current score={current_score:.4f} | Noisy score={noisy_score:.4f} | "
        f"Gates={'PASS' if gates_pass else 'FAIL'} | Decision={'PASS' if decision['overall_pass'] else 'FAIL'}"
    )
    if baseline_score is not None:
        print(f"Baseline score={baseline_score:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""Build a frozen golden-set manifest from dataset_2d."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader import read_bin_2d


def parse_params(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key] = float(value)
    return out


def heterogeneity_score(ref: np.ndarray) -> float:
    g_lat = np.abs(np.diff(ref, axis=0)).mean()
    g_dep = np.abs(np.diff(ref, axis=1)).mean()
    return float(g_lat + g_dep)


def build_manifest(dataset_root: str) -> dict:
    cases = []
    counts = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue
        split_count = 0
        for case_name in sorted(os.listdir(split_dir)):
            case_dir = os.path.join(split_dir, case_name)
            if not os.path.isdir(case_dir):
                continue

            noisy_path = os.path.join(case_dir, "noisy_output.bin")
            ref_path = os.path.join(case_dir, "reference_output.bin")
            params_path = os.path.join(case_dir, "params.txt")
            if not (os.path.exists(noisy_path) and os.path.exists(ref_path) and os.path.exists(params_path)):
                continue

            p = parse_params(params_path)
            ref, meta = read_bin_2d(ref_path)
            shift_depth = float(p.get("setup_shift_depth_cm", 0.0))
            shift_lat = float(p.get("setup_shift_lateral_cm", 0.0))
            shift_mag = float(np.sqrt(shift_depth * shift_depth + shift_lat * shift_lat))

            cases.append(
                {
                    "split": split,
                    "case_id": case_name,
                    "case_path": os.path.relpath(case_dir, os.path.dirname(dataset_root)).replace("\\", "/"),
                    "energy_mev": float(p.get("energy_mev", 0.0)),
                    "setup_shift_depth_cm": shift_depth,
                    "setup_shift_lateral_cm": shift_lat,
                    "shift_magnitude_cm": shift_mag,
                    "phantom_seed": int(p.get("phantom_seed", 0.0)),
                    "heterogeneity_score": heterogeneity_score(ref),
                    "nx": int(meta["nx"]),
                    "ny": int(meta["ny"]),
                }
            )
            split_count += 1
        counts[split] = split_count

    return {
        "manifest_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": os.path.relpath(dataset_root, os.path.dirname(dataset_root)).replace("\\", "/"),
        "counts": counts,
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default=os.path.join(base, "dataset_2d"))
    p.add_argument("--output", default=os.path.join(base, "golden_set_2d.json"))
    return p.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(args.dataset_root)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Golden set manifest saved: {args.output}")
    print(f"Counts: {manifest['counts']}")


if __name__ == "__main__":
    main()

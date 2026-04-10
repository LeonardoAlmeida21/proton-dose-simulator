"""Shared score, gate, and stratification helpers for benchmarking."""

from __future__ import annotations

import json
import os
from typing import Any


def default_framework_config_path(project_root: str) -> str:
    return os.path.join(project_root, "framework_config.json")


def load_framework_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "version" not in cfg:
        raise ValueError("Framework config must include 'version'.")
    return cfg


def _mode_key(mode: str) -> str:
    m = mode.lower()
    if m not in ("1d", "2d"):
        raise ValueError(f"Unsupported mode: {mode}")
    return m


def clinical_composite_score(metrics: dict[str, float], mode: str, config: dict[str, Any]) -> float:
    m = _mode_key(mode)
    weights = config["score"][m]

    score = (
        float(weights["range_error_mm"]) * float(metrics["range_error_mm"])
        + float(weights["abs_range_bias_mm"]) * abs(float(metrics["range_bias_mm"]))
        + float(weights["mse"]) * float(metrics["mse"])
    )
    if m == "2d":
        score += float(weights["penumbra_error_mm"]) * float(metrics["penumbra_error_mm"])
    return float(score)


def evaluate_acceptance_gates(metrics: dict[str, float], mode: str, config: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    m = _mode_key(mode)
    gates = config["gates"][m]

    checks = {
        "mean_range_error_mm_max": float(metrics["range_error_mm"]) <= float(gates["mean_range_error_mm_max"]),
        "mean_abs_range_bias_mm_max": abs(float(metrics["range_bias_mm"])) <= float(gates["mean_abs_range_bias_mm_max"]),
        "percent_cases_re_below_2mm_min": float(metrics["percent_cases_re_below_2mm"]) >= float(gates["percent_cases_re_below_2mm_min"]),
    }
    if m == "2d":
        checks["mean_penumbra_error_mm_max"] = float(metrics["penumbra_error_mm"]) <= float(gates["mean_penumbra_error_mm_max"])

    return all(checks.values()), checks


def _bucket(value: float, bins: list[float], labels: list[str], default_label: str) -> str:
    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if lo <= value < hi:
            return labels[i]
    if value == float(bins[-1]):
        return labels[-1]
    return default_label


def stratify_case(case_meta: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    strat = config["stratification"]
    energy = float(case_meta["energy_mev"])
    shift_mag = float(case_meta["shift_magnitude_cm"])
    hetero = float(case_meta["heterogeneity_score"])

    energy_label = _bucket(
        energy,
        strat["energy_mev_bins"],
        strat["energy_labels"],
        default_label="energy_out_of_range",
    )
    shift_label = _bucket(
        shift_mag,
        strat["shift_magnitude_cm_bins"],
        strat["shift_labels"],
        default_label="shift_out_of_range",
    )
    hetero_label = _bucket(
        hetero,
        strat["heterogeneity_score_bins"],
        strat["heterogeneity_labels"],
        default_label="hetero_out_of_range",
    )
    return {
        "energy_band": energy_label,
        "shift_band": shift_label,
        "heterogeneity_band": hetero_label,
    }

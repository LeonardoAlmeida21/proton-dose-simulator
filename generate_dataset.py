"""
generate_dataset.py
-------------------
Invokes the C++ Monte Carlo engine (mc_engine.exe) in a loop to generate a
training dataset with varied physical scenarios (energy and setup uncertainty).

Each scenario produces a pair of binary files:
  - reference_output.bin  (100k histories = Ground Truth)
  - noisy_output.bin      (100 histories  = Noisy input for AI)

Pairs are saved to data/train_set/{train,val,test}/case_XXXX/
"""

import subprocess
import os
import random
import sys
from tqdm import tqdm

# --- Dataset Configuration ---
N_CASES_TRAIN = 200   # Training cases
N_CASES_VAL   = 40    # Validation cases (monitored during training)
N_CASES_TEST  = 20    # Held-out test cases (final evaluation only)

# Physical parameter ranges (clinically realistic variations)
ENERGY_RANGE_MEV  = (80.0, 220.0)    # Typical clinical proton energy range
SHIFT_RANGE_CM    = (-1.0, 1.0)      # Setup uncertainty ±1 cm

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "mc_engine.exe")
DATA_ROOT   = os.path.join(os.path.dirname(__file__), "data")


def run_single_case(energy_mev: float, shift_cm: float, output_dir: str) -> bool:
    """Invoke the C++ engine for a single physical scenario."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [ENGINE_PATH, f"{energy_mev:.4f}", f"{shift_cm:.4f}", output_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"[ERROR] Engine failed: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def generate_split(split_name: str, n_cases: int):
    """Generate a set of cases (train/val/test) with randomised physical parameters."""
    split_dir = os.path.join(DATA_ROOT, "train_set", split_name)
    print(f"\n[{split_name.upper()}] Generating {n_cases} cases in '{split_dir}'")

    successful = 0
    with tqdm(total=n_cases, unit="case", desc=split_name) as pbar:
        case_idx = 0
        while successful < n_cases:
            energy = random.uniform(*ENERGY_RANGE_MEV)
            shift  = random.uniform(*SHIFT_RANGE_CM)

            case_dir = os.path.join(split_dir, f"case_{case_idx:04d}")
            if run_single_case(energy, shift, case_dir):
                # Save case metadata
                with open(os.path.join(case_dir, "params.txt"), "w") as f:
                    f.write(f"energy_mev={energy:.4f}\n")
                    f.write(f"shift_cm={shift:.4f}\n")
                successful += 1
                pbar.update(1)
            case_idx += 1

    print(f"  -> {successful} cases generated successfully.")


if __name__ == "__main__":
    random.seed(42)  # Reproducibility

    if not os.path.exists(ENGINE_PATH):
        print(f"[ERROR] C++ engine not found: {ENGINE_PATH}")
        print("  Please run 'build_vs.bat' first.")
        sys.exit(1)

    generate_split("train", N_CASES_TRAIN)
    generate_split("val",   N_CASES_VAL)
    generate_split("test",  N_CASES_TEST)

    print("\n Dataset complete! Ready to train the U-Net.")

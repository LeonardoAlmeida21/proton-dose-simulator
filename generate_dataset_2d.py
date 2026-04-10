"""
generate_dataset_2d.py
----------------------
Orchestrates the generation of the 2D training dataset.

For each case:
  1. Generates a synthetic 2D thoracic phantom.
  2. Samples a random beam energy.
  3. Samples random sub-voxel rigid shifts (setup errors).
  4. Applies the shifts to the phantom property maps.
  5. Saves the shifted phantom to a temporary binary file.
  6. Executes the 2D C++ Monte Carlo engine.
  7. Stores parameters.
"""

import os
import shutil
import random
import subprocess
import argparse
import numpy as np
from tqdm import tqdm

from generate_phantom import generate_thoracic_phantom, save_phantom
from generate_setup_errors import apply_rigid_shift

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

NUM_TRAIN = 1000
NUM_VAL   = 200
NUM_TEST  = 100
TOTAL     = NUM_TRAIN + NUM_VAL + NUM_TEST

OUTPUT_DIR = "dataset_2d"
TEMP_PHANTOM = "temp_phantom.bin"

# Physical parameters
GRID_CM = 0.1
NX, NY = 300, 100

# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def create_case(case_id: int, split: str):
    case_dir = os.path.join(OUTPUT_DIR, split, f"case_{case_id:04d}")
    os.makedirs(case_dir, exist_ok=True)

    # 1. Randomise physical parameters
    # Energy: 80 to 220 MeV covers most depths in a 30 cm phantom
    energy_mev = random.uniform(80.0, 220.0)

    # Setup uncertainty: realistic interfraction shifts are small (e.g., +/- 3 mm)
    dx_shift_cm = random.uniform(-0.3, 0.3)
    dy_shift_cm = random.uniform(-0.3, 0.3)

    # Phantom anatomical variations
    seed = case_id + 1000  # Ensure deterministic but distinct phantoms
    
    # 2. Generate Base Phantom
    density, I_val, Z_A = generate_thoracic_phantom(
        nx=NX, ny=NY, dx=GRID_CM, dy=GRID_CM, seed=seed
    )

    # 3. Apply setup errors (sub-voxel shifts)
    shifted_density = apply_rigid_shift(density, dx_shift_cm, dy_shift_cm, GRID_CM)
    shifted_I_val   = apply_rigid_shift(I_val,   dx_shift_cm, dy_shift_cm, GRID_CM)
    shifted_Z_A     = apply_rigid_shift(Z_A,     dx_shift_cm, dy_shift_cm, GRID_CM)

    # 4. Save shifted phantom to a temporary file for the MC engine
    temp_phantom_path = os.path.join(case_dir, TEMP_PHANTOM)
    save_phantom(temp_phantom_path, shifted_density, shifted_I_val, shifted_Z_A, GRID_CM, GRID_CM)

    # 5. Run the 2D Monte Carlo engine
    # mc_engine.exe [energy] 0.0 [output_dir] --density-map [phantom.bin]
    # Note: we pass 0.0 for the internal shift since we already shifted the map via Python!
    mc_exe_path = os.path.join(".", "mc_engine.exe")
    cmd = [
        mc_exe_path,
        f"{energy_mev:.2f}",
        "0.0",
        case_dir,
        "--density-map",
        temp_phantom_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] MC Engine failed for case {case_id}: {e}")
        return

    # 6. Save parameters for later analysis
    params_path = os.path.join(case_dir, "params.txt")
    with open(params_path, "w") as f:
        f.write(f"energy_mev={energy_mev:.3f}\n")
        f.write(f"setup_shift_depth_cm={dx_shift_cm:.3f}\n")
        f.write(f"setup_shift_lateral_cm={dy_shift_cm:.3f}\n")
        f.write(f"phantom_seed={seed}\n")
    
    # Clean up the temp phantom to save disk space
    os.remove(temp_phantom_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D MC dose dataset.")
    parser.add_argument("--num_train", type=int, default=NUM_TRAIN)
    parser.add_argument("--num_val", type=int, default=NUM_VAL)
    parser.add_argument("--num_test", type=int, default=NUM_TEST)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    total = args.num_train + args.num_val + args.num_test

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    print(f"--- Generating 2D Dataset ({total} cases) ---")
    print(f"Output directory: {OUTPUT_DIR}")

    if os.path.exists(OUTPUT_DIR):
        print("Warning: Output directory exists. Overwriting...")
        shutil.rmtree(OUTPUT_DIR)

    splits = [
        ("train", args.num_train),
        ("val",   args.num_val),
        ("test",  args.num_test)
    ]

    case_id = 0
    for split_name, count in splits:
        print(f"\nGenerating {split_name} split ({count} cases):")
        for _ in tqdm(range(count)):
            create_case(case_id, split_name)
            case_id += 1

    print("\nDataset generation complete!")


if __name__ == "__main__":
    # Ensure determinism across full dataset generation
    random.seed(42)
    main()

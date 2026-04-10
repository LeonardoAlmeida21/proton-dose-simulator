"""
generate_phantom.py
-------------------
Generates synthetic 2D thoracic CT-like density phantoms for training
the 2D Monte Carlo engine.

Each phantom consists of three tissue regions on a [ny, nx] grid:
  - Anterior soft tissue wall   (0–5 cm depth)
  - Lung parenchyma region      (5–15 cm depth, curved boundaries with noise)
  - Posterior soft tissue       (15–30 cm depth, optional tumour insert)

Material properties per region (ICRU Report 37 / PDG):
  Soft tissue: rho=1.04 g/cm^3, I=75.0 eV, Z/A=0.5500
  Lung:        rho=0.26 g/cm^3, I=85.7 eV, Z/A=0.5494
  Bone (ribs): rho=1.60 g/cm^3, I=106.0 eV, Z/A=0.5218
  Air:         rho=0.001205 g/cm^3, I=85.7 eV, Z/A=0.4992

Output binary format (read by density_map.cpp):
  int32  nx, ny
  double dx, dy
  double density[ny*nx]   (row-major)
  double I_value[ny*nx]
  double Z_over_A[ny*nx]

Usage:
    python generate_phantom.py
    python generate_phantom.py --nx 300 --ny 100 --seed 42 --output data/phantom.bin --visualize
"""

import argparse
import os
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ──────────────────────────────────────────────────────────────────────────────
# Material property tables
# ──────────────────────────────────────────────────────────────────────────────

MATERIALS = {
    "soft_tissue": {"density": 1.04,    "I": 75.0,  "ZA": 0.5500},
    "lung":        {"density": 0.26,    "I": 85.7,  "ZA": 0.5494},
    "bone":        {"density": 1.60,    "I": 106.0, "ZA": 0.5218},
    "air":         {"density": 0.001205,"I": 85.7,  "ZA": 0.4992},
}


# ──────────────────────────────────────────────────────────────────────────────
# Phantom generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_thoracic_phantom(
    nx:               int   = 300,    # Depth bins  (30 cm at 1 mm)
    ny:               int   = 100,    # Lateral bins (10 cm at 1 mm)
    dx:               float = 0.1,    # Depth voxel size [cm]
    dy:               float = 0.1,    # Lateral voxel size [cm]
    lung_depth_start: float = 5.0,    # Nominal lung start [cm]
    lung_depth_end:   float = 15.0,   # Nominal lung end [cm]
    boundary_noise:   float = 0.4,    # Lung boundary spatial noise amplitude [cm]
    density_noise:    float = 0.015,  # Intra-tissue density Gaussian noise σ [g/cm^3]
    add_ribs:         bool  = True,   # Insert rib-like high-density structures
    seed:             int   = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single 2D thoracic phantom.

    Returns:
        density  : [ny, nx] float64 array [g/cm^3]
        I_value  : [ny, nx] float64 array [eV]
        Z_over_A : [ny, nx] float64 array
    """
    rng = np.random.default_rng(seed)

    # Coordinate arrays
    depth  = np.arange(nx) * dx   # cm
    lat    = np.arange(ny) * dy   # cm

    # ── Lung boundaries ────────────────────────────────────────────────────────
    # Randomise the per-lateral-row lung entry/exit depth with smooth noise
    def smooth_boundary(nominal_cm, amp_cm):
        raw  = rng.standard_normal(ny) * amp_cm
        smooth = gaussian_filter(raw, sigma=5.0)         # Correlate laterally
        return nominal_cm + smooth

    lung_start = smooth_boundary(lung_depth_start, boundary_noise)  # [ny]
    lung_end   = smooth_boundary(lung_depth_end,   boundary_noise)  # [ny]
    # Ensure start < end always
    for j in range(ny):
        if lung_start[j] >= lung_end[j]:
            lung_start[j], lung_end[j] = lung_end[j] - 0.5, lung_start[j] + 0.5

    # ── Base material assignment ───────────────────────────────────────────────
    density  = np.full((ny, nx), MATERIALS["soft_tissue"]["density"])
    I_value  = np.full((ny, nx), MATERIALS["soft_tissue"]["I"])
    Z_over_A = np.full((ny, nx), MATERIALS["soft_tissue"]["ZA"])

    for j in range(ny):
        # Find depth indices for this lateral row's lung region
        ix_start = int(np.clip(lung_start[j] / dx, 0, nx - 1))
        ix_end   = int(np.clip(lung_end[j]   / dx, 0, nx - 1))
        density [j, ix_start:ix_end] = MATERIALS["lung"]["density"]
        I_value [j, ix_start:ix_end] = MATERIALS["lung"]["I"]
        Z_over_A[j, ix_start:ix_end] = MATERIALS["lung"]["ZA"]

    # ── Rib inserts ────────────────────────────────────────────────────────────
    if add_ribs:
        # Two ribs: one near the anterior wall, one just posterior to the lung
        rib_configs = [
            {"depth_cm": 2.5, "width_cm": 0.4, "lat_center": ny * dy * 0.35, "lat_hw": 0.6},
            {"depth_cm": 2.5, "width_cm": 0.4, "lat_center": ny * dy * 0.65, "lat_hw": 0.6},
            {"depth_cm": 16.5,"width_cm": 0.5, "lat_center": ny * dy * 0.30, "lat_hw": 0.7},
            {"depth_cm": 16.5,"width_cm": 0.5, "lat_center": ny * dy * 0.70, "lat_hw": 0.7},
        ]
        for rib in rib_configs:
            ix0 = int(np.clip((rib["depth_cm"] - rib["width_cm"]/2) / dx, 0, nx-1))
            ix1 = int(np.clip((rib["depth_cm"] + rib["width_cm"]/2) / dx, 0, nx-1))
            jy0 = int(np.clip((rib["lat_center"] - rib["lat_hw"]) / dy, 0, ny-1))
            jy1 = int(np.clip((rib["lat_center"] + rib["lat_hw"]) / dy, 0, ny-1))
            density [jy0:jy1, ix0:ix1] = MATERIALS["bone"]["density"]
            I_value [jy0:jy1, ix0:ix1] = MATERIALS["bone"]["I"]
            Z_over_A[jy0:jy1, ix0:ix1] = MATERIALS["bone"]["ZA"]

    # ── Intra-tissue density noise (anatomical heterogeneity) ─────────────────
    if density_noise > 0.0:
        noise = rng.standard_normal((ny, nx)) * density_noise
        noise = gaussian_filter(noise, sigma=2.0)
        density = np.clip(density + noise, 0.001, 2.5)  # Physical bounds

    return density, I_value, Z_over_A


# ──────────────────────────────────────────────────────────────────────────────
# Binary writer (matches density_map.cpp binary format)
# ──────────────────────────────────────────────────────────────────────────────

def save_phantom(filepath: str,
                 density: np.ndarray,
                 I_value: np.ndarray,
                 Z_over_A: np.ndarray,
                 dx: float = 0.1,
                 dy: float = 0.1):
    """Write the three property maps to the binary format read by density_map.cpp."""
    ny, nx = density.shape
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "wb") as f:
        f.write(struct.pack("ii", nx, ny))          # int32 nx, ny
        f.write(struct.pack("dd", dx, dy))          # double dx, dy
        f.write(density.astype(np.float64).tobytes())
        f.write(I_value.astype(np.float64).tobytes())
        f.write(Z_over_A.astype(np.float64).tobytes())

    print(f"Phantom saved: {filepath}  [{nx}x{ny}]  dx={dx} cm  dy={dy} cm")


def load_phantom(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reload a phantom from its binary file (for verification)."""
    with open(filepath, "rb") as f:
        nx, ny  = struct.unpack("ii", f.read(8))
        dx, dy  = struct.unpack("dd", f.read(16))
        n       = nx * ny
        density  = np.frombuffer(f.read(n * 8), dtype=np.float64).reshape(ny, nx)
        I_value  = np.frombuffer(f.read(n * 8), dtype=np.float64).reshape(ny, nx)
        Z_over_A = np.frombuffer(f.read(n * 8), dtype=np.float64).reshape(ny, nx)
    return density, I_value, Z_over_A, {"nx": nx, "ny": ny, "dx": dx, "dy": dy}


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def visualize_phantom(density: np.ndarray, I_value: np.ndarray,
                      dx: float, dy: float, output_path: str):
    depth  = np.arange(density.shape[1]) * dx
    lateral = np.arange(density.shape[0]) * dy

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    im0 = axes[0].imshow(density, extent=[0, depth[-1], lateral[-1], 0],
                         aspect="auto", cmap="viridis", vmin=0, vmax=1.8)
    axes[0].set_title("Density [g/cm³]", color="white")
    axes[0].set_xlabel("Depth (cm)", color="#aaaaaa")
    axes[0].set_ylabel("Lateral (cm)", color="#aaaaaa")
    plt.colorbar(im0, ax=axes[0]).ax.yaxis.set_tick_params(color="white")

    im1 = axes[1].imshow(I_value, extent=[0, depth[-1], lateral[-1], 0],
                         aspect="auto", cmap="plasma", vmin=70, vmax=110)
    axes[1].set_title("I-value [eV]", color="white")
    axes[1].set_xlabel("Depth (cm)", color="#aaaaaa")
    axes[1].set_ylabel("Lateral (cm)", color="#aaaaaa")
    plt.colorbar(im1, ax=axes[1]).ax.yaxis.set_tick_params(color="white")

    fig.suptitle("Synthetic 2D Thoracic Phantom", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Visualisation saved: {output_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Generate a synthetic 2D thoracic phantom")
    p.add_argument("--nx",        type=int,   default=300)
    p.add_argument("--ny",        type=int,   default=100)
    p.add_argument("--dx",        type=float, default=0.1,  help="Depth voxel size [cm]")
    p.add_argument("--dy",        type=float, default=0.1,  help="Lateral voxel size [cm]")
    p.add_argument("--seed",      type=int,   default=None)
    p.add_argument("--output",    type=str,   default=os.path.join(base, "data", "phantom_base.bin"))
    p.add_argument("--visualize", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    density, I_value, Z_over_A = generate_thoracic_phantom(
        nx=args.nx, ny=args.ny, dx=args.dx, dy=args.dy, seed=args.seed
    )

    save_phantom(args.output, density, I_value, Z_over_A, dx=args.dx, dy=args.dy)

    if args.visualize:
        fig_path = args.output.replace(".bin", "_preview.png")
        visualize_phantom(density, I_value, args.dx, args.dy, fig_path)

    # Quick reload verification
    d, I, ZA, meta = load_phantom(args.output)
    print(f"Verification: density range [{d.min():.4f}, {d.max():.4f}] g/cm^3")
    print(f"             I range       [{I.min():.1f}, {I.max():.1f}] eV")

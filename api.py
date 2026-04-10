import os
import sys
import gc
import json
import base64
import io
import uuid
import struct
import numpy as np
import subprocess
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add local modules to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from generate_phantom import generate_thoracic_phantom, save_phantom
from generate_setup_errors import apply_rigid_shift
from data_loader import read_bin_2d, prepare_pair_2d
from checkpoint_utils import load_unet2d_checkpoint
from evaluate import calculate_penumbra_width
from range_utils import peak_position_soft_numpy

# ──────────────────────────────────────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Proton Therapy Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow Vite frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load PyTorch Model globally to save time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model_path = os.path.join(BASE_DIR, "models", "best_unet2d.pth")
    if os.path.exists(model_path):
        UNET_MODEL, cfg = load_unet2d_checkpoint(model_path, DEVICE)
        print(f"Loaded UNet2D model onto {DEVICE} (base_ch={cfg['base_ch']}, depth={cfg['depth']}).")
    else:
        UNET_MODEL = None
        print("[WARNING] best_unet2d.pth not found. API predictions will be random.")
except Exception as e:
    print(f"Failed to load model: {e}")
    UNET_MODEL = None


# ──────────────────────────────────────────────────────────────────────────────
# Request Schema
# ──────────────────────────────────────────────────────────────────────────────
class SimulationRequest(BaseModel):
    energy_mev: float = 150.0
    shift_x_cm: float = 0.0
    shift_y_cm: float = 0.0
    seed: int = 42

# ──────────────────────────────────────────────────────────────────────────────
# Plotting Utility
# ──────────────────────────────────────────────────────────────────────────────
def array_to_base64(arr: np.ndarray, title: str, cmap="inferno", vmin=0, vmax=1.0) -> str:
    fig, ax = plt.subplots(figsize=(6, 2), facecolor="#1a1d27")
    ax.set_facecolor("#1a1d27")
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 30, 10, 0])
    ax.set_title(title, color="white", pad=4, fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_profiles(depth, noisy, pred, ref) -> str:
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1a1d27")
    ax.set_facecolor("#1a1d27")
    ax.plot(depth, noisy, color="#e07b54", lw=1.2, alpha=0.7, label="MC Noisy")
    ax.plot(depth, pred,  color="#5ab4ac", lw=2.0, alpha=0.9, label="U-Net Denoised")
    ax.plot(depth, ref,   color="#2c7fb8", lw=2.5, alpha=1.0, label="Ground Truth", ls="--")
    
    ax.set_title("Central Axis Depth-Dose Profile", color="white", fontsize=11)
    ax.set_xlabel("Depth (cm)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Normalised Dose", color="#aaaaaa", fontsize=9)
    ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# API Endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    GRID_CM = 0.1
    NX, NY = 300, 100
    
    # 1. Create a unique temporary directory for this request
    req_id = str(uuid.uuid4())[:8]
    tmp_dir = os.path.join(BASE_DIR, "data", f"tmp_{req_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_phantom = os.path.join(tmp_dir, "phantom.bin")
    
    try:
        # 2. Generate Base Phantom & Apply Shifts
        density, I_val, Z_A = generate_thoracic_phantom(nx=NX, ny=NY, dx=GRID_CM, dy=GRID_CM, seed=req.seed)
        d_shifted = apply_rigid_shift(density, req.shift_x_cm, req.shift_y_cm, GRID_CM)
        I_shifted = apply_rigid_shift(I_val,   req.shift_x_cm, req.shift_y_cm, GRID_CM)
        Z_shifted = apply_rigid_shift(Z_A,     req.shift_x_cm, req.shift_y_cm, GRID_CM)
        save_phantom(temp_phantom, d_shifted, I_shifted, Z_shifted, dx=GRID_CM, dy=GRID_CM)
        
        # Plot anatomical phantom
        phantom_b64 = array_to_base64(d_shifted, "Setup Error CT Phantom [g/cm³]", cmap="bone", vmin=0, vmax=1.8)
        
        # 3. Call MC Engine
        mc_exe = os.path.join(BASE_DIR, "mc_engine.exe")
        cmd = [mc_exe, str(req.energy_mev), "0.0", tmp_dir, "--density-map", temp_phantom]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        noisy_path = os.path.join(tmp_dir, "noisy_output.bin")
        ref_path   = os.path.join(tmp_dir, "reference_output.bin")
        
        if not os.path.exists(noisy_path) or not os.path.exists(ref_path):
            raise HTTPException(status_code=500, detail="MC Engine did not output files.")
            
        noisy_raw, _ = read_bin_2d(noisy_path)
        ref_raw, _   = read_bin_2d(ref_path)
        
        # 4. PyTorch Inference
        noisy_norm, ref_final = prepare_pair_2d(
            noisy_raw,
            ref_raw,
            target_ny=NY,
            target_nx=NX,
            clip_max=None,
        )
        noisy_tensor = torch.tensor(noisy_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        if UNET_MODEL is not None:
            with torch.no_grad():
                pred = UNET_MODEL(noisy_tensor).squeeze().cpu().numpy()
        else:
            pred = np.zeros((NY, NX))
        
        # 5. Calculate Metrics
        cr = NY // 2
        ref_center  = ref_final[cr, :]
        pred_center = pred[cr, :]
        depth_axis  = np.arange(NX) * GRID_CM
        
        bragg_ref  = peak_position_soft_numpy(ref_center) * GRID_CM
        bragg_pred = peak_position_soft_numpy(pred_center) * GRID_CM
        range_err  = abs(bragg_pred - bragg_ref) * 10  # mm
        
        mse = float(np.mean((pred - ref_final)**2))
        
        # Penumbra at 50% depth
        mid_idx = NX // 2
        p_ref  = calculate_penumbra_width(ref_final[:, mid_idx], GRID_CM)
        p_pred = calculate_penumbra_width(pred[:, mid_idx], GRID_CM)
        pen_err = abs(p_pred - p_ref)
        
        # 6. Generate Base64 Images
        noisy_b64 = array_to_base64(noisy_norm, "Noisy MC (100 histories)")
        pred_b64  = array_to_base64(pred, "U-Net 2D Prediction")
        ref_b64   = array_to_base64(ref_final, "Ground Truth (100k histories)")
        prof_b64  = plot_profiles(depth_axis, noisy_norm[cr, :], pred[cr, :], ref_final[cr, :])
        
    finally:
        # Cleanup temp directory
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    return {
        "metrics": {
            "range_error_mm": round(range_err, 2),
            "mse": round(mse, 4),
            "penumbra_error_mm": round(pen_err, 2)
        },
        "images": {
            "phantom": "data:image/png;base64," + phantom_b64,
            "noisy":   "data:image/png;base64," + noisy_b64,
            "pred":    "data:image/png;base64," + pred_b64,
            "ref":     "data:image/png;base64," + ref_b64,
            "profile": "data:image/png;base64," + prof_b64,
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Make sure this runs correctly standalone for testing
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

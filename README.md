# Proton Dose Simulator

A research-oriented 2D proton dose pipeline that combines a fast C++ Monte Carlo engine with a PyTorch U-Net denoiser and a clinically oriented benchmark framework.

The project targets thoracic cases (high heterogeneity: soft tissue, lung, ribs) and focuses on preserving clinically relevant characteristics such as Bragg peak depth and lateral penumbra.

## What This Repository Includes

- **C++ Monte Carlo engine** (`src/mc_engine`) with:
  - material-aware stopping power and straggling
  - 2D transport with Highland multiple Coulomb scattering
  - binary export format for model training/inference
- **PyTorch denoiser pipeline** with:
  - `UNet2D` model definition
  - physics-informed loss components
  - training, evaluation, and visualization scripts
- **Dashboard/API runtime**:
  - FastAPI backend (`api.py`)
  - frontend app (`frontend/`)
  - one-command startup via `dashboard.bat`
- **Continuous model-improvement framework**:
  - frozen golden set manifest
  - versioned scoring and acceptance gates
  - reproducible benchmark and pass/fail decision flow

## Repository Layout

- `src/mc_engine/`: Monte Carlo core and binary I/O
- `src/`: model, data loading, loss, checkpoint, benchmark utilities
- `train.py`: model training with clinical-score checkpoint selection
- `evaluate.py`: held-out evaluation metrics
- `benchmark_unet.py`: framework benchmark (`summary/full/stratified`, optional batch diff)
- `build_golden_set_manifest.py`: frozen golden-set manifest generator
- `framework_config.json`: versioned score/gate contract
- `golden_set_2d.json`: current frozen benchmark manifest
- `api.py`: backend for dashboard inference
- `dashboard.bat`: starts backend + frontend

## Setup

### 1) Python environment

```powershell
python -m venv venv_projeto
.\venv_projeto\Scripts\activate
pip install -r requirements.txt
```

### 2) Build the C++ engine (Windows / VS2022)

```powershell
.\build_vs.bat
```

This generates `mc_engine.exe` in the project root.

## Main Workflows

### Train (2D)

```powershell
.\venv_projeto\Scripts\python.exe train.py --mode 2d --framework_config .\framework_config.json
```

Fine-tune from an existing checkpoint:

```powershell
.\venv_projeto\Scripts\python.exe train.py --mode 2d --resume_checkpoint .\models\best_unet2d.pth --epochs 30 --lr 1e-4 --models_dir .\models\ft_run_01
```

Output checkpoints:
- `models/best_unet2d.pth` (best clinical composite score)
- `models/best_loss_unet2d.pth` (best validation loss)
- `models/last_unet2d.pth`

### Published Weights (in repository)

- `models/best_unet2d.pth` (default inference model used by `api.py` and dashboard)
- `models/best_loss_unet2d.pth` (reference baseline)
- `models/ft_bragg_v1/best_unet2d.pth` (fine-tuned variant focused on Bragg alignment)

To use the fine-tuned model in the dashboard/API, replace:
`models/best_unet2d.pth` with `models/ft_bragg_v1/best_unet2d.pth`.

### Evaluate

```powershell
.\venv_projeto\Scripts\python.exe evaluate.py --mode 2d
```

### Run benchmark framework

Summary:
```powershell
.\venv_projeto\Scripts\python.exe benchmark_unet.py --split test --mode summary --output_dir .\benchmark_outputs\summary_run
```

Batch decision (current vs noisy vs baseline):
```powershell
.\venv_projeto\Scripts\python.exe benchmark_unet.py --split test --mode stratified --batch --baseline_checkpoint .\models\best_loss_unet2d.pth --output_dir .\benchmark_outputs\batch_run
```

Generated artifacts:
- `benchmark_summary.json`
- `benchmark_cases.csv`
- `experiments_log.csv`

### Deep clinical analysis (Bragg + penumbra diagnostics)

```powershell
.\venv_projeto\Scripts\python.exe analyze_unet2d.py --checkpoint .\models\best_unet2d.pth --split all --window_rows 7 --output_json .\benchmark_outputs\analysis\analysis_best_unet2d.json
```

This report includes both:
- legacy metrics (same definitions as `evaluate.py` and `benchmark_unet.py`)
- robust diagnostics for Bragg alignment and penumbra stability

### Regenerate frozen manifest (only when versioning a new golden set)

```powershell
.\venv_projeto\Scripts\python.exe build_golden_set_manifest.py
```

## Clinical Benchmark Contract

The benchmark contract is versioned in `framework_config.json` and covers:
- composite score weights
- acceptance gates
- stratification bins

Primary metrics:
- Mean Range Error (mm)
- Mean |Range Bias| (mm)
- Mean Penumbra Error (mm)
- Mean MSE
- % cases with Range Error < 2 mm

## Dashboard

Run:
```powershell
.\dashboard.bat
```

The dashboard executes:
1. phantom generation
2. Monte Carlo simulation
3. U-Net inference
4. metric and profile visualization


## License

MIT

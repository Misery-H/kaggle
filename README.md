# ROGII Wellbore Geology Prediction

CPU-first, GPU-compatible workspace for the Kaggle competition:
https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

## Local setup

Keep `kaggle.json` in the repository root. It is ignored by Git.

```powershell
$env:KAGGLE_CONFIG_DIR = (Get-Location).Path
uv sync
uv run python scripts/check_env.py --device auto
```

## Download competition assets

```powershell
$env:KAGGLE_CONFIG_DIR = (Get-Location).Path
uv run python scripts/download_assets.py
uv run python scripts/smoke_read_data.py
```

This downloads:

- competition data to `data/raw/rogii-wellbore-geology-prediction/`
- DWT top kernel source to `baselines/dwt_top_kernel/`
- DWT top kernel output to `outputs/dwt_top_kernel/`

Downloaded data, outputs, zips, and Kaggle credentials are ignored by Git.

## Competition standard and target score

The competition uses RMSE over submitted `tvt` predictions, so lower is better.
See `docs/competition-standard.md` for the task summary, scoring rule, submission
format, and the current 7.2-level target baseline notes.

Validate a candidate submission before uploading:

```powershell
uv run python scripts/validate_submission.py outputs/rogii_lb7201_public_gold_conservative/submission.csv
```

This is a code competition. Kaggle rejects direct CSV uploads with
`Submission not allowed: This competition only accepts Submissions from Notebooks`.
Prepare and push a Kaggle notebook package, wait for it to complete, then submit
one of its output files:

```powershell
$env:KAGGLE_CONFIG_DIR = (Get-Location).Path
$env:PYTHONUTF8 = "1"
uv run python scripts/prepare_kernel_submit.py `
  --source-dir baselines/rogii_lb7201_public_gold_conservative `
  --code-file rogii-lb7201-public-gold-conservative.ipynb `
  --slug rogii-lb7201-public-gold-conservative-codex `
  --title "ROGII LB7201 Public Gold Conservative Codex" `
  --out-dir .kaggle_submit_7201
uv run kaggle kernels push -p .kaggle_submit_7201
uv run kaggle kernels status sumo1290/rogii-lb7201-public-gold-conservative-codex
uv run kaggle competitions submit rogii-wellbore-geology-prediction `
  -k sumo1290/rogii-lb7201-public-gold-conservative-codex `
  -v 1 `
  -f submission.csv `
  -m "public 7.201 conservative notebook baseline"
```

## Kaggle GPU smoke test

This creates or updates a private Kaggle kernel named `rogii-gpu-smoke-test` under
the Kaggle username in `kaggle.json`.

```powershell
$env:KAGGLE_CONFIG_DIR = (Get-Location).Path
uv run python scripts/kaggle_gpu_smoke.py
```

The script pushes the kernel with `--accelerator gpu`, polls status, and prints logs.
In the current test run, Kaggle assigned a `Tesla P100-PCIE-16GB`. `nvidia-smi`
was visible, but Kaggle's preinstalled `torch 2.10.0+cu128` reported that P100
`sm_60` is below the supported architecture range, so a tiny CUDA tensor probe
failed. Treat Kaggle GPU scheduling as working, and treat PyTorch-on-P100 as a
separate compatibility issue for any future torch-based model.

## Docker through WSL

Docker is available in the local `Ubuntu-22.04` WSL distribution.

```powershell
wsl -d Ubuntu-22.04 -- docker build -f Dockerfile.cpu -t rogii:cpu .
wsl -d Ubuntu-22.04 -- docker run --rm rogii:cpu uv run python scripts/check_env.py --device auto --skip-kaggle
```

If Docker Hub is unreachable from WSL, keep the Dockerfile default portable and
override only the build base image locally:

```powershell
wsl -d Ubuntu-22.04 -- docker build -f Dockerfile.cpu -t rogii:cpu --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.12-slim .
```

For a future SSH GPU machine with NVIDIA Container Toolkit:

```bash
docker build -f Dockerfile.gpu -t rogii:gpu .
docker run --rm --gpus all \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/kaggle.json:/workspace/kaggle.json:ro" \
  -e KAGGLE_CONFIG_DIR=/workspace \
  rogii:gpu uv run python scripts/check_env.py --device auto
```

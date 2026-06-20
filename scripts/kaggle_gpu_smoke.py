from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


COMPETITION = "rogii-wellbore-geology-prediction"


SMOKE_SCRIPT = r'''
import json
import os
import shutil
import subprocess
import sys


def run(command):
    if shutil.which(command[0]) is None:
        return {"available": False, "command": command[0]}
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


try:
    import torch
    cuda_available = bool(torch.cuda.is_available())
    cuda_probe = {"attempted": False}
    if cuda_available:
        cuda_probe = {"attempted": True}
        try:
            tensor = torch.tensor([1.0], device="cuda")
            cuda_probe.update({"ok": bool((tensor + 1).item() == 2.0)})
        except Exception as exc:
            cuda_probe.update({"ok": False, "error": str(exc)})

    torch_info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_usable": bool(cuda_probe.get("ok", False)),
        "cuda_probe": cuda_probe,
        "device_count": torch.cuda.device_count(),
        "devices": [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ],
    }
except Exception as exc:
    torch_info = {"installed": False, "error": str(exc)}

report = {
    "python": sys.version,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_smi": run(["nvidia-smi", "-L"]),
    "torch": torch_info,
}

print("KAGGLE_GPU_SMOKE_REPORT_START")
print(json.dumps(report, indent=2))
print("KAGGLE_GPU_SMOKE_REPORT_END")

if not report["nvidia_smi"].get("available"):
    raise SystemExit("Kaggle GPU smoke test did not detect nvidia-smi.")
'''


def run(command: list[str], env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, env=env, text=True, capture_output=False)


def kaggle_username(repo_root: Path) -> str:
    config_path = repo_root / "kaggle.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    username = config.get("username")
    if not username:
        raise ValueError("kaggle.json does not contain a username.")
    return username


def prepare_kernel(repo_root: Path, kernel_dir: Path, username: str) -> None:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": f"{username}/rogii-gpu-smoke-test",
        "title": "ROGII GPU Smoke Test",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (kernel_dir / "kernel.py").write_text(SMOKE_SCRIPT.strip() + "\n", encoding="utf-8")


def poll_status(kernel_ref: str, env: dict[str, str], timeout_seconds: int, interval_seconds: int) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_output = ""
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ["uv", "run", "kaggle", "kernels", "status", kernel_ref],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        last_output = (completed.stdout + completed.stderr).strip()
        print(last_output, flush=True)
        upper = last_output.upper()
        if "COMPLETE" in upper:
            return "COMPLETE"
        if any(state in upper for state in ("ERROR", "FAILED", "CANCELLED")):
            return "FAILED"
        time.sleep(interval_seconds)
    raise TimeoutError(f"Timed out waiting for {kernel_ref}. Last status: {last_output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and run a private Kaggle GPU smoke-test kernel.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--kernel-dir", type=Path, default=Path(".kaggle_gpu_smoke"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    kernel_dir = args.kernel_dir
    if not kernel_dir.is_absolute():
        kernel_dir = repo_root / kernel_dir

    env = os.environ.copy()
    env.setdefault("KAGGLE_CONFIG_DIR", str(repo_root))
    username = kaggle_username(repo_root)
    kernel_ref = f"{username}/rogii-gpu-smoke-test"

    prepare_kernel(repo_root, kernel_dir, username)
    print(f"Prepared {kernel_ref} in {kernel_dir}")

    if args.prepare_only:
        return 0

    run(["uv", "run", "kaggle", "kernels", "push", "-p", str(kernel_dir), "--accelerator", "gpu"], env)
    status = poll_status(kernel_ref, env, args.timeout_seconds, args.interval_seconds)
    run(["uv", "run", "kaggle", "kernels", "logs", kernel_ref], env, check=False)
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

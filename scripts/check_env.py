from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rogii.device import choose_device, torch_status


def run_command(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command[0]}

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"available": True, "error": str(exc)}

    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def kaggle_status() -> dict[str, Any]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        files = api.competition_list_files("rogii-wellbore-geology-prediction")
        return {"authenticated": True, "competition_file_count_first_page": len(files.files)}
    except Exception as exc:
        return {"authenticated": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local/Kaggle/GPU environment health.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle API authentication checks.",
    )
    args = parser.parse_args()

    torch_info = torch_status()
    device_choice = choose_device(args.device, torch_info)
    exit_code = 2 if args.device == "cuda" and device_choice.resolved == "unavailable" else 0
    repo_root = Path(__file__).resolve().parents[1]

    report = {
        "requested_device": args.device,
        "resolved_device": device_choice.resolved,
        "device_reason": device_choice.reason,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "repo_root": str(repo_root),
        "kaggle_config_dir": os.environ.get("KAGGLE_CONFIG_DIR"),
        "packages": {
            "kaggle": package_version("kaggle"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "PyWavelets": package_version("PyWavelets"),
            "torch": package_version("torch"),
        },
        "commands": {
            "uv": run_command(["uv", "--version"]),
            "nvidia-smi": run_command(["nvidia-smi", "-L"]),
        },
        "torch": torch_info,
    }

    if not args.skip_kaggle:
        report["kaggle"] = kaggle_status()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.device == "cuda" and exit_code != 0:
        print("CUDA was requested but is not usable.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path


COMPETITION = "rogii-wellbore-geology-prediction"
DWT_KERNEL = "nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based"
PUBLIC_TARGET_KERNEL = "curvecowboy/rogii-lb7201-public-gold-conservative"


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def unzip_archives(target_dir: Path) -> None:
    for archive in sorted(target_dir.glob("*.zip")):
        marker = target_dir / f".{archive.stem}.extracted"
        if marker.exists():
            continue
        print(f"Extracting {archive} -> {target_dir}", flush=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
        marker.write_text("ok\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ROGII data and DWT baseline assets.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-kernel", action="store_true")
    parser.add_argument("--skip-output", action="store_true")
    parser.add_argument("--include-public-target", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    env = os.environ.copy()
    env.setdefault("KAGGLE_CONFIG_DIR", str(repo_root))

    data_dir = repo_root / "data" / "raw" / COMPETITION
    baseline_dir = repo_root / "baselines" / "dwt_top_kernel"
    output_dir = repo_root / "outputs" / "dwt_top_kernel"
    target_baseline_dir = repo_root / "baselines" / "rogii_lb7201_public_gold_conservative"
    target_output_dir = repo_root / "outputs" / "rogii_lb7201_public_gold_conservative"
    for directory in (data_dir, baseline_dir, output_dir, target_baseline_dir, target_output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.skip_data:
        run(
            [
                "uv",
                "run",
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(data_dir),
            ],
            env,
        )
        unzip_archives(data_dir)

    if not args.skip_kernel:
        run(
            [
                "uv",
                "run",
                "kaggle",
                "kernels",
                "pull",
                DWT_KERNEL,
                "-p",
                str(baseline_dir),
                "-m",
            ],
            env,
        )

    if not args.skip_output:
        run(
            [
                "uv",
                "run",
                "kaggle",
                "kernels",
                "output",
                DWT_KERNEL,
                "-p",
                str(output_dir),
            ],
            env,
        )

    if args.include_public_target:
        run(
            [
                "uv",
                "run",
                "kaggle",
                "kernels",
                "pull",
                PUBLIC_TARGET_KERNEL,
                "-p",
                str(target_baseline_dir),
                "-m",
            ],
            env,
        )
        run(
            [
                "uv",
                "run",
                "kaggle",
                "kernels",
                "output",
                PUBLIC_TARGET_KERNEL,
                "-p",
                str(target_output_dir),
            ],
            env,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

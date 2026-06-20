from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path

import pandas as pd


def load_kaggle_username(config_dir: Path) -> str:
    config_path = config_dir / "kaggle.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Kaggle config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    username = str(config.get("username", "")).strip()
    if not username:
        raise ValueError(f"No username found in {config_path}")
    return username


def build_notebook(submission: Path, sample_path: str) -> dict[str, object]:
    raw = submission.read_bytes()
    payload = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    source = f"""
import base64
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

payload = \"\"\"{payload}\"\"\"
out_path = Path("/kaggle/working/submission.csv")
out_path.write_bytes(zlib.decompress(base64.b64decode(payload.encode("ascii"))))

submission = pd.read_csv(out_path)
sample = pd.read_csv({sample_path!r})
if submission.columns.tolist() != ["id", "tvt"]:
    raise ValueError(f"bad columns: {{submission.columns.tolist()}}")
if len(submission) != len(sample):
    raise ValueError(f"bad row count: {{len(submission)}} != {{len(sample)}}")
if not submission["id"].astype(str).equals(sample["id"].astype(str)):
    raise ValueError("id order mismatch")
if not np.isfinite(pd.to_numeric(submission["tvt"], errors="coerce").to_numpy(dtype=float)).all():
    raise ValueError("non-finite tvt values")
print("wrote submission.csv", submission.shape)
print(submission["tvt"].describe().to_string())
""".strip()
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in source.splitlines()],
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a Kaggle notebook that emits a fixed ROGII submission.csv.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--sample-path",
        default="/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv",
    )
    args = parser.parse_args()

    if not args.submission.exists():
        raise FileNotFoundError(args.submission)
    owner = load_kaggle_username(args.config_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    code_file = f"{args.slug}.ipynb"
    notebook = build_notebook(args.submission, args.sample_path)
    (args.out_dir / code_file).write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    metadata = {
        "id": f"{owner}/{args.slug}",
        "title": args.title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"kernel": metadata["id"], "path": str(args.out_dir), "code_file": code_file}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

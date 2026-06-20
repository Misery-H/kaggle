from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(submission_path: Path, sample_path: Path) -> dict[str, object]:
    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)

    errors: list[str] = []
    if submission.columns.tolist() != ["id", "tvt"]:
        errors.append(f"Expected columns ['id', 'tvt'], got {submission.columns.tolist()}")
    if len(submission) != len(sample):
        errors.append(f"Expected {len(sample)} rows, got {len(submission)}")
    if "id" in submission and "id" in sample and not submission["id"].astype(str).equals(sample["id"].astype(str)):
        errors.append("Submission ids do not match sample_submission.csv order")
    if "tvt" in submission:
        tvt = pd.to_numeric(submission["tvt"], errors="coerce")
        non_finite = int((~np.isfinite(tvt.to_numpy(dtype=float))).sum())
        if non_finite:
            errors.append(f"Found {non_finite} non-finite tvt values")
    else:
        tvt = pd.Series(dtype=float)

    report = {
        "submission": str(submission_path),
        "sample": str(sample_path),
        "valid": not errors,
        "errors": errors,
        "rows": int(len(submission)),
        "columns": submission.columns.tolist(),
    }
    if "tvt" in submission:
        report["tvt"] = {
            "min": float(tvt.min()),
            "max": float(tvt.max()),
            "mean": float(tvt.mean()),
            "std": float(tvt.std()),
            "missing": int(tvt.isna().sum()),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ROGII submission file.")
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/raw/rogii-wellbore-geology-prediction/sample_submission.csv"),
    )
    args = parser.parse_args()

    report = validate_submission(args.submission, args.sample)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

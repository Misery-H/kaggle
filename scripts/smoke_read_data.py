from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Read a small slice of the ROGII dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/rogii-wellbore-geology-prediction"),
    )
    args = parser.parse_args()

    data_root = args.data_root
    sample_submission = data_root / "sample_submission.csv"
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not sample_submission.exists():
        raise FileNotFoundError(f"Missing {sample_submission}")
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing {test_dir}")

    train_csv = sorted(train_dir.glob("*__horizontal_well.csv"))
    typewell_csv = sorted(train_dir.glob("*__typewell.csv"))
    test_csv = sorted(test_dir.glob("*__horizontal_well.csv"))
    if not train_csv or not typewell_csv or not test_csv:
        raise FileNotFoundError("Expected train/test CSV files were not found.")

    sample_df = pd.read_csv(sample_submission)
    train_df = pd.read_csv(train_csv[0], nrows=5)
    typewell_df = pd.read_csv(typewell_csv[0], nrows=5)
    test_df = pd.read_csv(test_csv[0], nrows=5)

    report = {
        "data_root": str(data_root),
        "sample_submission": {
            "rows": int(len(sample_df)),
            "columns": sample_df.columns.tolist(),
        },
        "first_train_horizontal": {
            "path": str(train_csv[0]),
            "columns": train_df.columns.tolist(),
        },
        "first_train_typewell": {
            "path": str(typewell_csv[0]),
            "columns": typewell_df.columns.tolist(),
        },
        "first_test_horizontal": {
            "path": str(test_csv[0]),
            "columns": test_df.columns.tolist(),
        },
        "counts": {
            "train_horizontal_csv": len(train_csv),
            "train_typewell_csv": len(typewell_csv),
            "test_horizontal_csv": len(test_csv),
            "train_png": len(list(train_dir.glob("*.png"))),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

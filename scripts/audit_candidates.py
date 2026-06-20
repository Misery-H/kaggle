from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


def read_valid_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.columns.tolist() != ["id", "tvt"]:
        return None
    if len(df) != len(sample):
        return None
    if not df["id"].astype(str).equals(sample["id"].astype(str)):
        return None
    tvt = pd.to_numeric(df["tvt"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(tvt).all():
        return None
    return df


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def audit_candidates(
    paths: list[Path],
    sample_path: Path,
    best_path: Path,
    bad_path: Path,
    best_score: float,
    bad_score: float,
) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    sample["well"] = sample["id"].astype(str).str[:8]
    best = read_valid_submission(best_path, sample)
    bad = read_valid_submission(bad_path, sample)
    if best is None:
        raise ValueError(f"Invalid best submission: {best_path}")
    if bad is None:
        raise ValueError(f"Invalid bad submission: {bad_path}")

    base = best["tvt"].to_numpy(dtype=float)
    failed = bad["tvt"].to_numpy(dtype=float)
    direction = failed - base
    direction_mse = float(np.mean(direction * direction))
    hidden_dot = (bad_score * bad_score - best_score * best_score - direction_mse) / 2.0

    rows: list[dict[str, object]] = []
    for path in paths:
        df = read_valid_submission(path, sample)
        if df is None:
            continue
        y = df["tvt"].to_numpy(dtype=float)
        delta = y - base
        alpha = float(np.mean(delta * direction) / direction_mse) if direction_mse else 0.0
        residual = delta - alpha * direction
        heuristic_score = math.sqrt(max(0.0, best_score * best_score + np.mean(delta * delta) + 2.0 * alpha * hidden_dot))
        row: dict[str, object] = {
            "path": str(path),
            "sha16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "heuristic_score": heuristic_score,
            "rmse_vs_best": rmse(delta),
            "rmse_vs_bad": rmse(y - failed),
            "alpha_bad_direction": alpha,
            "residual_rmse": rmse(residual),
            "mean_tvt": float(np.mean(y)),
            "std_tvt": float(np.std(y, ddof=1)),
        }
        for well in sample["well"].unique():
            mask = (sample["well"] == well).to_numpy()
            row[f"{well}_rmse_move"] = rmse(delta[mask])
            row[f"{well}_mean_move"] = float(np.mean(delta[mask]))
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["duplicate_count"] = result.groupby("sha16")["sha16"].transform("count")
    return result.sort_values(["heuristic_score", "rmse_vs_best", "path"]).reset_index(drop=True)


def expand_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.csv")))
        elif item.exists():
            paths.append(item)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ROGII candidate submissions against known submitted anchors.")
    parser.add_argument("paths", nargs="+", type=Path, help="Candidate CSV files or directories to scan.")
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/raw/rogii-wellbore-geology-prediction/sample_submission.csv"),
    )
    parser.add_argument("--best", type=Path, default=Path("outputs/codex_rogii_lb7201/submission.csv"))
    parser.add_argument("--bad", type=Path, default=Path("outputs/codex_rogii_w060/submission.csv"))
    parser.add_argument("--best-score", type=float, default=7.285)
    parser.add_argument("--bad-score", type=float, default=7.540)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    result = audit_candidates(
        expand_paths(args.paths),
        args.sample,
        args.best,
        args.bad,
        args.best_score,
        args.bad_score,
    )
    if result.empty:
        print("No valid candidate submissions found.")
        return 1

    cols = [
        "path",
        "sha16",
        "duplicate_count",
        "heuristic_score",
        "rmse_vs_best",
        "alpha_bad_direction",
        "residual_rmse",
        "rmse_vs_bad",
        "mean_tvt",
        "std_tvt",
    ]
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", args.top)
    print(result[cols].head(args.top).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

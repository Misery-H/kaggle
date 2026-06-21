from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    submission = pd.read_csv(path)
    if submission.columns.tolist() != ["id", "tvt"]:
        raise ValueError(f"{path} must have columns ['id', 'tvt']")
    if len(submission) != len(sample) or not submission["id"].astype(str).equals(sample["id"].astype(str)):
        raise ValueError(f"{path} does not match sample_submission id order")
    values = pd.to_numeric(submission["tvt"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite tvt values")
    return submission


def parse_scored_arg(value: str) -> tuple[str, Path, float]:
    parts = value.split("=")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected scored candidate as name=path=score")
    name, path, score = parts
    return name, Path(path), float(score)


def audit_scored_geometry(
    sample: pd.DataFrame,
    anchor_path: Path,
    anchor_score: float,
    scored: list[tuple[str, Path, float]],
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    anchor = read_submission(anchor_path, sample)
    anchor_values = anchor["tvt"].to_numpy(dtype=float)
    wells = sample["id"].astype(str).str[:8].to_numpy()
    rows: list[dict[str, object]] = []

    for name, path, score in scored:
        candidate = read_submission(path, sample)
        delta = candidate["tvt"].to_numpy(dtype=float) - anchor_values
        move_rmse = rmse(delta)
        score_delta = abs(float(score) - float(anchor_score))
        abs_delta = np.abs(delta)
        row = {
            "name": name,
            "path": str(path),
            "score": float(score),
            "score_delta_vs_anchor": float(score - anchor_score),
            "abs_score_delta": score_delta,
            "full_move_rmse_vs_anchor": move_rmse,
            "full_move_mean_vs_anchor": float(np.mean(delta)),
            "full_move_p95_abs_vs_anchor": float(np.quantile(abs_delta, 0.95)),
            "full_move_p99_abs_vs_anchor": float(np.quantile(abs_delta, 0.99)),
            "full_move_max_abs_vs_anchor": float(np.max(abs_delta)),
            "full_rows_triangle_margin": move_rmse - score_delta,
            "full_rows_triangle_warning": bool((move_rmse + tolerance) < score_delta),
        }
        for well in pd.unique(wells):
            mask = wells == well
            well_delta = delta[mask]
            row[f"{well}_move_rmse"] = rmse(well_delta)
            row[f"{well}_move_max_abs"] = float(np.max(np.abs(well_delta)))
            row[f"{well}_move_mean"] = float(np.mean(well_delta))
        rows.append(row)

    summary: dict[str, object] = {
        "anchor_path": str(anchor_path),
        "anchor_score": float(anchor_score),
        "public_subset_note": (
            "Kaggle publicScore may be computed on an unknown subset of submitted rows. "
            "Triangle and projection checks over all submission rows are diagnostics only, "
            "not strict proof that a score/file pairing is impossible on the public subset."
        ),
        "full_rows_triangle_warnings": [row["name"] for row in rows if row["full_rows_triangle_warning"]],
        "best_scored_candidate": min(rows, key=lambda row: row["score"])["name"] if rows else None,
        "worst_scored_candidate": max(rows, key=lambda row: row["score"])["name"] if rows else None,
    }

    return pd.DataFrame(rows), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit scored ROGII candidates against all-row local movement.")
    parser.add_argument("--sample", type=Path, default=Path("data/raw/rogii-wellbore-geology-prediction/sample_submission.csv"))
    parser.add_argument("--anchor", type=Path, default=Path("outputs/codex_rogii_lb7201/submission.csv"))
    parser.add_argument("--anchor-score", type=float, default=7.285)
    parser.add_argument(
        "--scored",
        action="append",
        type=parse_scored_arg,
        default=[
            parse_scored_arg("w060=outputs/codex_rogii_w060/submission.csv=7.540"),
            parse_scored_arg("light=outputs/codex_light_u_smoother/submission.csv=7.523"),
            parse_scored_arg("anti=outputs/codex_anti_light_u_smoother/submission.csv=7.628"),
        ],
        help="Scored candidate as name=path=score. Can be repeated.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/score_geometry_audit"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(args.sample)
    report, summary = audit_scored_geometry(sample, args.anchor, args.anchor_score, args.scored, args.tolerance)
    report.to_csv(args.out_dir / "score_geometry_report.csv", index=False, lineterminator="\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

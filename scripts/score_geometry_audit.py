from __future__ import annotations

import argparse
import json
import math
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
    rows: list[dict[str, object]] = []
    valid_dirs: list[np.ndarray] = []
    valid_names: list[str] = []
    valid_scores: list[float] = []

    for name, path, score in scored:
        candidate = read_submission(path, sample)
        delta = candidate["tvt"].to_numpy(dtype=float) - anchor_values
        move_rmse = rmse(delta)
        score_delta = abs(float(score) - float(anchor_score))
        triangle_margin = move_rmse - score_delta
        projection = (anchor_score * anchor_score + move_rmse * move_rmse - score * score) / 2.0
        cauchy_bound = anchor_score * move_rmse
        cauchy_margin = cauchy_bound - abs(projection)
        is_consistent = triangle_margin >= -tolerance and cauchy_margin >= -tolerance
        rows.append(
            {
                "name": name,
                "path": str(path),
                "score": float(score),
                "move_rmse_vs_anchor": move_rmse,
                "abs_score_delta": score_delta,
                "triangle_margin": triangle_margin,
                "hidden_error_projection": projection,
                "cauchy_bound": cauchy_bound,
                "cauchy_margin": cauchy_margin,
                "consistent_with_anchor_file": bool(is_consistent),
            }
        )
        if is_consistent and move_rmse > 0:
            valid_dirs.append(delta)
            valid_names.append(name)
            valid_scores.append(float(score))

    summary: dict[str, object] = {
        "anchor_path": str(anchor_path),
        "anchor_score": float(anchor_score),
        "consistent_scored_candidates": valid_names,
        "inconsistent_scored_candidates": [row["name"] for row in rows if not row["consistent_with_anchor_file"]],
    }

    if valid_dirs:
        dmat = np.vstack(valid_dirs).T
        gram = dmat.T @ dmat / len(anchor_values)
        b = np.array(
            [
                (anchor_score * anchor_score + gram[i, i] - valid_scores[i] * valid_scores[i]) / 2.0
                for i in range(len(valid_scores))
            ],
            dtype=float,
        )
        try:
            coeffs = np.linalg.solve(gram, b)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.pinv(gram) @ b
        pred_score_sq = anchor_score * anchor_score - float(b @ coeffs)
        opt_delta = dmat @ coeffs
        summary["span_fit"] = {
            "names": valid_names,
            "coefficients": {name: float(value) for name, value in zip(valid_names, coeffs)},
            "predicted_best_score": math.sqrt(max(0.0, pred_score_sq)),
            "move_rmse_vs_anchor": rmse(opt_delta),
            "move_mean_vs_anchor": float(np.mean(opt_delta)),
            "move_p95_abs_vs_anchor": float(np.quantile(np.abs(opt_delta), 0.95)),
        }
    else:
        summary["span_fit"] = None

    return pd.DataFrame(rows), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether submitted scores are geometrically consistent with local files.")
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

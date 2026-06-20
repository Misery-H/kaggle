from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def robust_polyfit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < degree + 4:
        return y.copy()
    xx = x[finite]
    yy = y[finite]
    coef = np.polyfit(xx, yy, degree)
    for _ in range(4):
        residual = yy - np.polyval(coef, xx)
        scale = float(np.median(np.abs(residual)) * 1.4826 + 1e-6)
        weights = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        coef = np.polyfit(xx, yy, degree, w=weights)
    smoothed = y.copy()
    smoothed[finite] = np.polyval(coef, xx)
    return smoothed


def make_variant(
    submission: pd.DataFrame,
    data_root: Path,
    blend: float,
    degree: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = submission.copy()
    work["well"] = work["id"].astype(str).str[:8]
    work["row_idx"] = work["id"].astype(str).str[9:].astype(int)
    predictions = dict(zip(work["id"].astype(str), work["tvt"].astype(float)))
    report_rows: list[dict[str, object]] = []

    for well, group in work.groupby("well", sort=False):
        horizontal_path = data_root / "test" / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            continue
        horizontal = pd.read_csv(horizontal_path)
        ordered = group.sort_values("row_idx")
        idx = ordered["row_idx"].to_numpy(dtype=int)
        valid = (idx >= 0) & (idx < len(horizontal))
        idx = idx[valid]
        ids = ordered["id"].astype(str).to_numpy()[valid]
        if len(idx) < 25:
            continue

        md = horizontal["MD"].to_numpy(dtype=float)[idx]
        z = horizontal["Z"].to_numpy(dtype=float)[idx]
        tvt = ordered["tvt"].to_numpy(dtype=float)[valid]
        x = (md - md.min()) / max(float(md.max() - md.min()), 1.0)
        u = tvt + z
        u_fit = robust_polyfit(x, u, degree)
        tvt_smoothed = (1.0 - blend) * tvt + blend * (u_fit - z)
        for row_id, value in zip(ids, tvt_smoothed):
            predictions[row_id] = float(value)

        delta = tvt_smoothed - tvt
        report_rows.append(
            {
                "well": well,
                "rows": int(len(idx)),
                "blend": float(blend),
                "degree": int(degree),
                "rmse_delta": float(np.sqrt(np.mean(delta * delta))),
                "p95_abs_delta": float(np.quantile(np.abs(delta), 0.95)),
                "mean_delta": float(np.mean(delta)),
            }
        )

    out = work[["id"]].copy()
    out["tvt"] = out["id"].astype(str).map(predictions).astype(float)
    if not np.isfinite(out["tvt"].to_numpy(dtype=float)).all():
        raise RuntimeError("Smoother produced non-finite predictions.")
    return out, pd.DataFrame(report_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate contact U-smoother variants for ROGII submissions.")
    parser.add_argument("submission", type=Path, help="Base submission CSV.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/rogii-wellbore-geology-prediction"),
    )
    parser.add_argument("--blend", type=float, action="append", required=True, help="Smoother blend weight.")
    parser.add_argument("--degree", type=int, default=3)
    args = parser.parse_args()

    base = pd.read_csv(args.submission)
    if base.columns.tolist() != ["id", "tvt"]:
        raise ValueError(f"Expected columns ['id', 'tvt'], got {base.columns.tolist()}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for blend in args.blend:
        label = f"b{blend:.3f}".replace(".", "p")
        out, report = make_variant(base, args.data_root, blend, args.degree)
        out_path = args.out_dir / f"submission_contact_smoother_{label}.csv"
        report_path = args.out_dir / f"contact_smoother_{label}_report.csv"
        out.to_csv(out_path, index=False, lineterminator="\n")
        report.to_csv(report_path, index=False, lineterminator="\n")
        summary.append(
            {
                "blend": float(blend),
                "degree": int(args.degree),
                "submission": str(out_path),
                "report": str(report_path),
                "rows": int(len(out)),
                "mean_tvt": float(out["tvt"].mean()),
                "std_tvt": float(out["tvt"].std()),
            }
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

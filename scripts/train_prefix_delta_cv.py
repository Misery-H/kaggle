from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 20 or float(np.nanstd(x[mask])) < 1e-9:
        return 0.0, float(np.nanmedian(y[mask])) if mask.any() else 0.0
    xx = x[mask]
    yy = y[mask]
    coef = np.polyfit(xx, yy, 1)
    for _ in range(4):
        residual = yy - np.polyval(coef, xx)
        scale = float(np.nanmedian(np.abs(residual)) * 1.4826 + 1e-6)
        weights = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        coef = np.polyfit(xx, yy, 1, w=weights)
    return float(coef[0]), float(coef[1])


def interp_typewell_gr(typewell: pd.DataFrame, tvt: np.ndarray) -> np.ndarray:
    tw = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
    if len(tw) < 5:
        return np.full_like(tvt, np.nan, dtype=float)
    return np.interp(tvt, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float))


def build_well_frame(data_root: Path, horizontal_path: Path, stride: int) -> pd.DataFrame | None:
    well = horizontal_path.name.split("__")[0]
    typewell_path = data_root / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    known = horizontal["TVT_input"].notna()
    hidden = horizontal["TVT_input"].isna() & horizontal["TVT"].notna()
    if int(known.sum()) < 100 or int(hidden.sum()) < 100:
        return None

    known_idx = np.flatnonzero(known.to_numpy())
    last_known_idx = int(known_idx[-1])
    known_df = horizontal.iloc[known_idx]
    last = horizontal.iloc[last_known_idx]
    n_rows = len(horizontal)
    md_min = float(horizontal["MD"].min())
    md_span = max(float(horizontal["MD"].max() - md_min), 1.0)
    md0 = float(known_df["MD"].iloc[0])
    prefix_span = max(float(known_df["MD"].iloc[-1] - md0), 1.0)
    prefix_x = (known_df["MD"].to_numpy(float) - md0) / prefix_span
    slope_norm, intercept = robust_line(prefix_x, known_df["TVT_input"].to_numpy(float))
    prefix_fit = intercept + slope_norm * prefix_x
    prefix_rmse = rmse(known_df["TVT_input"].to_numpy(float), prefix_fit)

    idx = np.flatnonzero(hidden.to_numpy())
    if stride > 1:
        idx = idx[::stride]
    h = horizontal.iloc[idx].copy()
    md = h["MD"].to_numpy(float)
    x_norm = (md - md0) / prefix_span
    linear_pred = intercept + slope_norm * x_norm
    last_tvt = float(last["TVT_input"])
    last_md = float(last["MD"])
    last_x = float(last["X"])
    last_y = float(last["Y"])
    last_z = float(last["Z"])
    gr = h["GR"].to_numpy(float)
    gr_median = float(horizontal["GR"].median())
    gr_filled = np.where(np.isfinite(gr), gr, gr_median)
    tw_gr_at_linear = interp_typewell_gr(typewell, linear_pred)

    tw_gr = pd.to_numeric(typewell["GR"], errors="coerce")
    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce")
    prefix_gr = pd.to_numeric(known_df["GR"], errors="coerce")
    out = pd.DataFrame(
        {
            "well": well,
            "row_idx": idx.astype(np.int32),
            "n_rows": n_rows,
            "known_rows": int(known.sum()),
            "known_frac": float(known.mean()),
            "row_frac": idx / max(n_rows - 1, 1),
            "hidden_frac": (idx - int(known.sum())) / max(int(hidden.sum()) - 1, 1),
            "md": md,
            "md_rel": (md - md_min) / md_span,
            "md_since_known": md - last_md,
            "x_since_known": h["X"].to_numpy(float) - last_x,
            "y_since_known": h["Y"].to_numpy(float) - last_y,
            "xy_dist_since_known": np.hypot(h["X"].to_numpy(float) - last_x, h["Y"].to_numpy(float) - last_y),
            "z": h["Z"].to_numpy(float),
            "z_since_known": h["Z"].to_numpy(float) - last_z,
            "gr": gr_filled,
            "gr_missing": (~np.isfinite(gr)).astype(np.int8),
            "last_known_tvt": last_tvt,
            "linear_pred": linear_pred,
            "linear_delta": linear_pred - last_tvt,
            "prefix_slope_per_md": slope_norm / prefix_span,
            "prefix_rmse": prefix_rmse,
            "prefix_tvt_span": float(known_df["TVT_input"].max() - known_df["TVT_input"].min()),
            "prefix_z_span": float(known_df["Z"].max() - known_df["Z"].min()),
            "prefix_gr_mean": float(prefix_gr.mean()),
            "prefix_gr_std": float(prefix_gr.std()),
            "type_tvt_min": float(tw_tvt.min()),
            "type_tvt_max": float(tw_tvt.max()),
            "type_tvt_span": float(tw_tvt.max() - tw_tvt.min()),
            "type_gr_mean": float(tw_gr.mean()),
            "type_gr_std": float(tw_gr.std()),
            "type_gr_at_linear": tw_gr_at_linear,
            "gr_minus_type_gr_at_linear": gr_filled - tw_gr_at_linear,
            "target": h["TVT"].to_numpy(float),
        }
    )
    for col in out.columns:
        if col not in {"well"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [c for c in out.columns if c != "well"]
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median(numeric_only=True))
    out["target_delta"] = out["target"] - out["last_known_tvt"]
    return out


def build_dataset(data_root: Path, stride: int, max_wells: int | None) -> pd.DataFrame:
    paths = sorted((data_root / "train").glob("*__horizontal_well.csv"))
    if max_wells:
        paths = paths[:max_wells]
    frames: list[pd.DataFrame] = []
    for i, path in enumerate(paths, 1):
        frame = build_well_frame(data_root, path, stride)
        if frame is not None:
            frames.append(frame)
        if i % 50 == 0:
            print(f"processed {i}/{len(paths)} wells; usable={len(frames)}", flush=True)
    if not frames:
        raise RuntimeError("No usable wells found")
    return pd.concat(frames, ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"well", "target", "target_delta"}
    return [col for col in df.columns if col not in excluded]


def run_cv(df: pd.DataFrame, folds: int, seed: int, max_estimators: int) -> dict[str, object]:
    features = feature_columns(df)
    groups = df["well"].to_numpy()
    y_delta = df["target_delta"].to_numpy(float)
    target = df["target"].to_numpy(float)
    last = df["last_known_tvt"].to_numpy(float)
    linear = df["linear_pred"].to_numpy(float)
    oof_delta = np.zeros(len(df), dtype=float)
    fold_rows: list[dict[str, object]] = []

    params = dict(
        objective="regression",
        learning_rate=0.04,
        n_estimators=max_estimators,
        num_leaves=128,
        min_child_samples=100,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )

    splitter = GroupKFold(n_splits=folds)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df[features], y_delta, groups), 1):
        model = LGBMRegressor(**params)
        model.fit(
            df.iloc[tr_idx][features],
            y_delta[tr_idx],
            eval_set=[(df.iloc[va_idx][features], y_delta[va_idx])],
            eval_metric="rmse",
            callbacks=[early_stopping(120, verbose=False), log_evaluation(100)],
        )
        pred_delta = model.predict(df.iloc[va_idx][features], num_iteration=model.best_iteration_)
        oof_delta[va_idx] = pred_delta
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(len(va_idx)),
                "wells": int(pd.Series(groups[va_idx]).nunique()),
                "hold_rmse": rmse(target[va_idx], last[va_idx]),
                "linear_rmse": rmse(target[va_idx], linear[va_idx]),
                "model_rmse": rmse(target[va_idx], last[va_idx] + pred_delta),
                "best_iteration": int(model.best_iteration_ or max_estimators),
            }
        )
        row = fold_rows[-1]
        print(
            f"fold {fold}: hold={row['hold_rmse']:.4f} linear={row['linear_rmse']:.4f} model={row['model_rmse']:.4f}",
            flush=True,
        )

    return {
        "features": features,
        "overall": {
            "rows": int(len(df)),
            "wells": int(df["well"].nunique()),
            "hold_rmse": rmse(target, last),
            "linear_rmse": rmse(target, linear),
            "model_rmse": rmse(target, last + oof_delta),
        },
        "folds": fold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a prefix-delta model using only test-visible fields.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/rogii-wellbore-geology-prediction"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/prefix_delta_cv"))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-estimators", type=int, default=1200)
    parser.add_argument("--max-wells", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset(args.data_root, args.stride, args.max_wells)
    dataset_path = args.out_dir / f"dataset_stride{args.stride}.parquet"
    df.to_parquet(dataset_path, index=False)
    print(f"dataset: rows={len(df)} wells={df['well'].nunique()} path={dataset_path}", flush=True)
    print(
        json.dumps(
            {
                "target_delta": df["target_delta"].describe().to_dict(),
                "known_frac_by_well": df.groupby("well")["known_frac"].first().describe().to_dict(),
            },
            indent=2,
        ),
        flush=True,
    )
    report = run_cv(df, args.folds, args.seed, args.max_estimators)
    report["dataset"] = {
        "path": str(dataset_path),
        "stride": args.stride,
        "rows": int(len(df)),
        "wells": int(df["well"].nunique()),
    }
    report_path = args.out_dir / f"cv_report_stride{args.stride}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["folds"]).to_csv(args.out_dir / f"cv_folds_stride{args.stride}.csv", index=False)
    print(json.dumps(report["overall"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

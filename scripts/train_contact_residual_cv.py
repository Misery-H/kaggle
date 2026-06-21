from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold


SURFACE_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def tvt_from_contacts(horizontal: pd.DataFrame, typewell: pd.DataFrame, ref_col: str = "EGFDU") -> np.ndarray:
    if "Geology" not in typewell.columns:
        raise ValueError("typewell must contain Geology for contact reconstruction")
    tw_geo = typewell.dropna(subset=["Geology"])
    if tw_geo.empty:
        raise ValueError("typewell has no geology labels")
    if ref_col not in horizontal.columns:
        ref_col = next(col for col in SURFACE_COLS if col in horizontal.columns)
    ref_rows = tw_geo[tw_geo["Geology"] == ref_col]
    if ref_rows.empty:
        ref_col = str(tw_geo["Geology"].iloc[0])
        ref_rows = tw_geo[tw_geo["Geology"] == ref_col]
    if ref_col not in horizontal.columns:
        ref_col = next(col for col in SURFACE_COLS if col in horizontal.columns)
    ref_tvt = float(ref_rows["TVT"].min())
    physical = ref_tvt - (horizontal["Z"].to_numpy(float) - horizontal[ref_col].to_numpy(float))
    offset_source = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=float)
    offset_mask = np.isfinite(offset_source) & np.isfinite(physical)
    if int(offset_mask.sum()) < 20:
        offset_source = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=float)
        offset_mask = np.isfinite(offset_source) & np.isfinite(physical)
    offset = float(np.nanmean(offset_source[offset_mask] - physical[offset_mask]))
    return physical + offset


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


def prefix_calibrated_contact(horizontal: pd.DataFrame, physical: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    known = horizontal[horizontal["TVT_input"].notna()].copy()
    if len(known) < 50:
        return physical.copy(), {
            "prefix_rows": float(len(known)),
            "prefix_rmse": np.nan,
            "prefix_median": 0.0,
            "prefix_mad": 0.0,
            "prefix_slope": 0.0,
        }

    known_idx = known.index.to_numpy(dtype=int)
    known_md = known["MD"].to_numpy(dtype=float)
    known_y = known["TVT_input"].to_numpy(dtype=float)
    known_phys = physical[known_idx]
    residual = known_y - known_phys
    md0 = float(known_md[0])
    span = max(float(known_md[-1] - known_md[0]), 1.0)
    x_known = (known_md - md0) / span
    slope, intercept = robust_line(x_known, residual)
    median = float(np.nanmedian(residual))
    mad = float(np.nanmedian(np.abs(residual - median)) * 1.4826 + 1e-6)
    limit = float(np.clip(4.0 * mad + abs(median), 0.15, 3.0))
    x_all = (horizontal["MD"].to_numpy(dtype=float) - md0) / span
    correction = 0.60 * (intercept + slope * x_all) + 0.40 * median
    correction = np.clip(correction, -limit, limit)
    pred = physical + correction
    return pred, {
        "prefix_rows": float(len(known)),
        "prefix_rmse": rmse(known_y, known_phys),
        "prefix_median": median,
        "prefix_mad": mad,
        "prefix_slope": slope,
        "prefix_corr_limit": limit,
    }


def build_well_frame(data_root: Path, horizontal_path: Path, stride: int) -> pd.DataFrame | None:
    well = horizontal_path.name.split("__")[0]
    typewell_path = data_root / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    hidden = horizontal["TVT_input"].isna() & horizontal["TVT"].notna()
    if int(hidden.sum()) < 100:
        return None

    try:
        physical = tvt_from_contacts(horizontal, typewell)
    except Exception:
        return None
    contact_pred, prefix_stats = prefix_calibrated_contact(horizontal, physical)

    known_count = int(horizontal["TVT_input"].notna().sum())
    last_known_idx = max(0, known_count - 1)
    last_known_tvt = float(horizontal.loc[last_known_idx, "TVT_input"])
    last_known_md = float(horizontal.loc[last_known_idx, "MD"])
    last_known_z = float(horizontal.loc[last_known_idx, "Z"])
    n_rows = len(horizontal)

    idx = np.flatnonzero(hidden.to_numpy())
    if stride > 1:
        idx = idx[::stride]
    h = horizontal.iloc[idx].copy()
    out = pd.DataFrame(
        {
            "well": well,
            "row_idx": idx.astype(np.int32),
            "n_rows": n_rows,
            "known_frac": known_count / n_rows,
            "md": h["MD"].to_numpy(float),
            "md_rel": (h["MD"].to_numpy(float) - float(horizontal["MD"].min()))
            / max(float(horizontal["MD"].max() - horizontal["MD"].min()), 1.0),
            "md_since_known": h["MD"].to_numpy(float) - last_known_md,
            "z": h["Z"].to_numpy(float),
            "z_since_known": h["Z"].to_numpy(float) - last_known_z,
            "gr": h["GR"].to_numpy(float),
            "last_known_tvt": last_known_tvt,
            "contact_pred": contact_pred[idx],
            "physical_pred": physical[idx],
            "target": h["TVT"].to_numpy(float),
        }
    )
    out["contact_delta"] = out["contact_pred"] - out["last_known_tvt"]
    out["physical_delta"] = out["physical_pred"] - out["last_known_tvt"]
    out["row_frac"] = out["row_idx"] / max(n_rows - 1, 1)
    out["hidden_frac"] = (out["row_idx"] - known_count) / max(n_rows - known_count - 1, 1)
    out["gr_missing"] = out["gr"].isna().astype(np.int8)
    out["gr"] = out["gr"].fillna(float(horizontal["GR"].median()))
    for col in SURFACE_COLS:
        if col in horizontal.columns:
            values = horizontal[col].to_numpy(float)[idx]
            out[f"{col.lower()}_surface"] = values
            out[f"z_minus_{col.lower()}"] = out["z"].to_numpy(float) - values
            out[f"contact_minus_{col.lower()}"] = out["contact_pred"].to_numpy(float) - values
    for key, value in prefix_stats.items():
        out[key] = value
    out["residual"] = out["target"] - out["contact_pred"]
    return out


def build_dataset(data_root: Path, stride: int, max_wells: int | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    paths = sorted((data_root / "train").glob("*__horizontal_well.csv"))
    if max_wells:
        paths = paths[:max_wells]
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
    excluded = {"well", "target", "residual"}
    return [col for col in df.columns if col not in excluded]


def run_cv(df: pd.DataFrame, folds: int, seed: int, max_estimators: int) -> dict[str, object]:
    features = feature_columns(df)
    groups = df["well"].to_numpy()
    y = df["residual"].to_numpy(float)
    base = df["contact_pred"].to_numpy(float)
    target = df["target"].to_numpy(float)
    oof_residual = np.zeros(len(df), dtype=float)
    fold_rows: list[dict[str, object]] = []

    params = dict(
        objective="regression",
        learning_rate=0.035,
        n_estimators=max_estimators,
        num_leaves=96,
        min_child_samples=80,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )

    splitter = GroupKFold(n_splits=folds)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df[features], y, groups), 1):
        model = LGBMRegressor(**params)
        model.fit(
            df.iloc[tr_idx][features],
            y[tr_idx],
            eval_set=[(df.iloc[va_idx][features], y[va_idx])],
            eval_metric="rmse",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(100)],
        )
        pred_residual = model.predict(df.iloc[va_idx][features], num_iteration=model.best_iteration_)
        oof_residual[va_idx] = pred_residual
        fold_base = rmse(target[va_idx], base[va_idx])
        fold_model = rmse(target[va_idx], base[va_idx] + pred_residual)
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(len(va_idx)),
                "wells": int(pd.Series(groups[va_idx]).nunique()),
                "base_contact_rmse": fold_base,
                "model_rmse": fold_model,
                "gain": fold_base - fold_model,
                "best_iteration": int(model.best_iteration_ or max_estimators),
            }
        )
        print(f"fold {fold}: contact={fold_base:.5f} model={fold_model:.5f}", flush=True)

    base_rmse = rmse(target, base)
    model_rmse = rmse(target, base + oof_residual)
    return {
        "features": features,
        "overall": {
            "rows": int(len(df)),
            "wells": int(df["well"].nunique()),
            "base_contact_rmse": base_rmse,
            "model_rmse": model_rmse,
            "gain": base_rmse - model_rmse,
        },
        "folds": fold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a contact-residual model with GroupKFold.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/rogii-wellbore-geology-prediction"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/contact_residual_cv"))
    parser.add_argument("--stride", type=int, default=5, help="Use every Nth hidden row for faster CV.")
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
                "target": df["target"].describe().to_dict(),
                "residual": df["residual"].describe().to_dict(),
                "known_frac": df.groupby("well")["known_frac"].first().describe().to_dict(),
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

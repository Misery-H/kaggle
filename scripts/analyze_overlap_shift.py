from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SURFACE_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def tvt_from_contacts(horizontal: pd.DataFrame, typewell: pd.DataFrame, ref_col: str = "EGFDU") -> np.ndarray:
    tw_geo = typewell.dropna(subset=["Geology"])
    if tw_geo.empty:
        raise ValueError("typewell has no Geology labels")
    if ref_col not in horizontal.columns:
        ref_col = next(col for col in SURFACE_COLS if col in horizontal.columns)
    ref_tvt = tw_geo.loc[tw_geo["Geology"] == ref_col, "TVT"].min()
    if pd.isna(ref_tvt):
        ref_col = str(tw_geo["Geology"].iloc[0])
        ref_tvt = tw_geo.loc[tw_geo["Geology"] == ref_col, "TVT"].min()
    physical = ref_tvt - (horizontal["Z"].to_numpy(float) - horizontal[ref_col].to_numpy(float))
    offset = float(np.nanmean(horizontal["TVT"].to_numpy(float) - physical))
    return physical + offset


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    submission = pd.read_csv(path)
    if submission.columns.tolist() != ["id", "tvt"]:
        raise ValueError(f"{path} must have columns ['id', 'tvt']")
    if len(submission) != len(sample) or not submission["id"].astype(str).equals(sample["id"].astype(str)):
        raise ValueError(f"{path} does not match sample_submission id order")
    submission = submission.copy()
    submission["well"] = submission["id"].astype(str).str[:8]
    submission["row_idx"] = submission["id"].astype(str).str[9:].astype(int)
    return submission


def summarize_overlap(data_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    train_wells = {p.name.split("__")[0] for p in (data_root / "train").glob("*__horizontal_well.csv")}
    test_wells = sorted(p.name.split("__")[0] for p in (data_root / "test").glob("*__horizontal_well.csv"))
    overlap = [well for well in test_wells if well in train_wells]
    rows: list[dict[str, object]] = []

    for well in overlap:
        train = pd.read_csv(data_root / "train" / f"{well}__horizontal_well.csv")
        test = pd.read_csv(data_root / "test" / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(data_root / "train" / f"{well}__typewell.csv")
        contact = tvt_from_contacts(train, typewell)
        known = test["TVT_input"].notna().to_numpy()
        hidden = ~known
        if len(train) != len(test):
            raise ValueError(f"{well} train/test row count mismatch: {len(train)} vs {len(test)}")

        row: dict[str, object] = {
            "well": well,
            "rows": int(len(test)),
            "known_rows": int(known.sum()),
            "hidden_rows": int(hidden.sum()),
            "known_frac": float(known.mean()),
        }
        for col in ["MD", "X", "Y", "Z", "GR"]:
            diff = test[col].to_numpy(float) - train[col].to_numpy(float)
            row[f"{col.lower()}_max_abs_train_test_diff"] = float(np.nanmax(np.abs(diff)))
            row[f"{col.lower()}_rmse_train_test_diff"] = rmse(diff)

        known_tvt_diff = test.loc[known, "TVT_input"].to_numpy(float) - train.loc[known, "TVT"].to_numpy(float)
        row["known_tvt_vs_train_rmse"] = rmse(known_tvt_diff)
        row["known_tvt_vs_train_max_abs"] = float(np.nanmax(np.abs(known_tvt_diff)))
        contact_diff = test.loc[known, "TVT_input"].to_numpy(float) - contact[known]
        row["known_tvt_vs_contact_rmse"] = rmse(contact_diff)
        row["known_tvt_vs_contact_mean"] = float(np.nanmean(contact_diff))
        row["known_tvt_vs_contact_tail50_mean"] = float(np.nanmean(contact_diff[-50:]))
        train_hidden = train.loc[hidden, "TVT"].to_numpy(float)
        row["train_hidden_tvt_mean"] = float(np.nanmean(train_hidden))
        row["train_hidden_tvt_std"] = float(np.nanstd(train_hidden, ddof=1))
        rows.append(row)

    report = pd.DataFrame(rows)
    summary = {
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "overlap_wells": overlap,
        "all_test_wells_are_train_wells": sorted(test_wells) == sorted(overlap),
    }
    return report, summary


def summarize_submissions(data_root: Path, sample: pd.DataFrame, submissions: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, path in submissions:
        submission = read_submission(path, sample)
        for well, group in submission.groupby("well", sort=False):
            train_path = data_root / "train" / f"{well}__horizontal_well.csv"
            if not train_path.exists():
                continue
            train = pd.read_csv(train_path)
            idx = group["row_idx"].to_numpy(dtype=int)
            pred = group["tvt"].to_numpy(dtype=float)
            truth = train["TVT"].to_numpy(dtype=float)[idx]
            diff = pred - truth
            rows.append(
                {
                    "submission": name,
                    "path": str(path),
                    "well": well,
                    "rows": int(len(group)),
                    "rmse_vs_train_hidden": rmse(diff),
                    "mean_vs_train_hidden": float(np.nanmean(diff)),
                    "median_vs_train_hidden": float(np.nanmedian(diff)),
                    "p95_abs_vs_train_hidden": float(np.nanquantile(np.abs(diff), 0.95)),
                    "min_vs_train_hidden": float(np.nanmin(diff)),
                    "max_vs_train_hidden": float(np.nanmax(diff)),
                }
            )
    return pd.DataFrame(rows)


def parse_submission_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ROGII overlap wells and local submission train-copy drift.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/rogii-wellbore-geology-prediction"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/overlap_shift_analysis"))
    parser.add_argument(
        "--submission",
        action="append",
        default=[
            "anchor=outputs/codex_rogii_lb7201/submission.csv",
            "w060=outputs/codex_rogii_w060/submission.csv",
            "light=outputs/codex_light_u_smoother/submission.csv",
            "anti=outputs/codex_anti_light_u_smoother/submission.csv",
        ],
        help="Submission as name=path. Can be repeated.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(args.data_root / "sample_submission.csv")
    overlap_report, summary = summarize_overlap(args.data_root)
    submissions = [parse_submission_arg(item) for item in args.submission]
    submission_report = summarize_submissions(args.data_root, sample, submissions)

    overlap_report.to_csv(args.out_dir / "overlap_prefix_report.csv", index=False, lineterminator="\n")
    submission_report.to_csv(args.out_dir / "submission_vs_train_hidden.csv", index=False, lineterminator="\n")
    summary["submission_count"] = len(submissions)
    summary["outputs"] = {
        "overlap_prefix_report": str(args.out_dir / "overlap_prefix_report.csv"),
        "submission_vs_train_hidden": str(args.out_dir / "submission_vs_train_hidden.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(overlap_report.to_string(index=False), flush=True)
    print(submission_report.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

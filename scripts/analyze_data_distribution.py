from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_numeric(series: pd.Series) -> dict[str, float]:
    q = series.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    out = {
        "count": float(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
    }
    out.update({f"q{int(k * 100):02d}": float(v) for k, v in q.items()})
    return out


def collect_well_stats(data_root: Path, subset: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizontal_path in sorted((data_root / subset).glob("*__horizontal_well.csv")):
        well = horizontal_path.name.split("__")[0]
        typewell_path = data_root / subset / f"{well}__typewell.csv"
        horizontal = pd.read_csv(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        row: dict[str, object] = {
            "subset": subset,
            "well": well,
            "n_rows": int(len(horizontal)),
            "md_min": float(horizontal["MD"].min()),
            "md_max": float(horizontal["MD"].max()),
            "md_span": float(horizontal["MD"].max() - horizontal["MD"].min()),
            "z_min": float(horizontal["Z"].min()),
            "z_max": float(horizontal["Z"].max()),
            "z_span": float(horizontal["Z"].max() - horizontal["Z"].min()),
            "gr_nonnull": int(horizontal["GR"].notna().sum()),
            "gr_frac": float(horizontal["GR"].notna().mean()),
            "tvt_input_nonnull": int(horizontal["TVT_input"].notna().sum()),
            "tvt_input_frac": float(horizontal["TVT_input"].notna().mean()),
            "type_rows": int(len(typewell)),
            "type_gr_mean": float(typewell["GR"].mean()),
            "type_gr_std": float(typewell["GR"].std()),
            "type_tvt_min": float(typewell["TVT"].min()),
            "type_tvt_max": float(typewell["TVT"].max()),
            "type_tvt_span": float(typewell["TVT"].max() - typewell["TVT"].min()),
        }
        if "TVT" in horizontal.columns:
            row.update(
                {
                    "tvt_min": float(horizontal["TVT"].min()),
                    "tvt_max": float(horizontal["TVT"].max()),
                    "tvt_span": float(horizontal["TVT"].max() - horizontal["TVT"].min()),
                    "hidden_rows": int((horizontal["TVT_input"].isna() & horizontal["TVT"].notna()).sum()),
                    "hidden_tvt_delta_min": float((horizontal["TVT"] - horizontal["TVT_input"].dropna().iloc[-1]).min()),
                    "hidden_tvt_delta_max": float((horizontal["TVT"] - horizontal["TVT_input"].dropna().iloc[-1]).max()),
                }
            )
        if "Geology" in typewell.columns:
            row["type_geology_nonnull"] = int(typewell["Geology"].notna().sum())
            row["type_geology_nunique"] = int(typewell["Geology"].nunique(dropna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ROGII train/test well distributions.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/rogii-wellbore-geology-prediction"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/data_distribution"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = collect_well_stats(args.data_root, "train")
    test = collect_well_stats(args.data_root, "test")
    all_stats = pd.concat([train, test], ignore_index=True)
    all_stats.to_csv(args.out_dir / "well_stats.csv", index=False)

    summary: dict[str, object] = {
        "train_wells": int(len(train)),
        "test_wells": int(len(test)),
        "train_summary": {
            col: summarize_numeric(train[col])
            for col in [
                "n_rows",
                "md_span",
                "tvt_span",
                "z_span",
                "gr_frac",
                "tvt_input_frac",
                "type_rows",
                "type_geology_nunique",
            ]
            if col in train
        },
        "test_wells_detail": test.to_dict(orient="records"),
        "test_wells_in_train": sorted(set(test["well"]) & set(train["well"])),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

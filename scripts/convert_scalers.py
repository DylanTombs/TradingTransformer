#!/usr/bin/env python3
"""
scripts/convert_scalers.py

Converts featureScaler.pkl / targetScaler.pkl (sklearn StandardScaler objects)
to the CSV format expected by the C++ backtester's ScalerParams::loadFromCSV().

Output format:
    feature,mean,scale
    high,102.15,25.30
    ...

Usage:
    python scripts/convert_scalers.py
    python scripts/convert_scalers.py --model-dir models --out-dir models
"""

import argparse
import csv
import os
import sys

try:
    import joblib
except ImportError:
    sys.exit("[ERROR] joblib not installed. Run: pip install joblib")


# Feature column order — must match exportModel.py::load_args() auxilFeatures + target
FEATURE_COLUMNS = [
    "high", "low", "volume", "adj close",
    "P", "R1", "R2", "R3", "S1", "S2", "S3",
    "obv", "volume_zscore",
    "rsi", "macd", "macds", "macdh",
    "sma", "lma", "sema", "lema",
    "overnight_gap",
    "return_lag_1", "return_lag_3", "return_lag_5",
    "volatility",
    "SR_K", "SR_D", "SR_RSI_K", "SR_RSI_D",
    "ATR", "HL_PCT", "PCT_CHG",
]
TARGET_COLUMN = "close"


def write_scaler_csv(scaler, columns: list[str], path: str) -> None:
    if len(scaler.mean_) != len(columns):
        raise ValueError(
            f"Scaler has {len(scaler.mean_)} features but {len(columns)} column names were provided."
        )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "mean", "scale"])
        for name, mean, scale in zip(columns, scaler.mean_, scaler.scale_):
            writer.writerow([name, mean, scale])
    print(f"  Written: {path} ({len(columns)} features)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert sklearn .pkl scalers to CSV for C++ backtester")
    parser.add_argument("--model-dir", default="models", help="Directory containing .pkl scaler files")
    parser.add_argument("--out-dir",   default="models", help="Directory to write .csv scaler files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    feat_pkl  = os.path.join(args.model_dir, "featureScaler.pkl")
    tgt_pkl   = os.path.join(args.model_dir, "targetScaler.pkl")

    for path in (feat_pkl, tgt_pkl):
        if not os.path.exists(path):
            sys.exit(f"[ERROR] File not found: {path}")

    print("Loading scalers...")
    feat_scaler = joblib.load(feat_pkl)
    tgt_scaler  = joblib.load(tgt_pkl)

    print("Writing CSV scalers...")

    # feature_scaler.csv: 34 entries — 33 aux features (featureScaler.pkl)
    # + close appended from targetScaler.pkl.
    # MLStrategy applies this single scaler to the full 34-feature input vector.
    # During training, close in the encoder input was scaled by targetScaler,
    # so its parameters belong at position 33 (last) in the combined CSV.
    combined_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    combined_mean    = list(feat_scaler.mean_)  + [tgt_scaler.mean_[0]]
    combined_scale   = list(feat_scaler.scale_) + [tgt_scaler.scale_[0]]

    with open(os.path.join(args.out_dir, "feature_scaler.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "mean", "scale"])
        for name, mean, scale in zip(combined_columns, combined_mean, combined_scale):
            writer.writerow([name, mean, scale])
    print(f"  Written: feature_scaler.csv ({len(combined_columns)} features)")

    # target_scaler.csv: 1 entry — used only to inverse-transform model output
    write_scaler_csv(
        tgt_scaler,
        [TARGET_COLUMN],
        os.path.join(args.out_dir, "target_scaler.csv"),
    )

    print("Done. Scaler CSVs are ready for the C++ backtester.")


if __name__ == "__main__":
    main()

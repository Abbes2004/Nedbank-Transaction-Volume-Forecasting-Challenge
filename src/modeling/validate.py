"""
validate.py
============
Offline validation utilities:
  1. Score OOF predictions from train.py
  2. Analyse prediction distribution vs training target distribution
  3. Print error breakdown by target quantile (identifies where the model struggles)

Usage
-----
    python src/modeling/validate.py
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OOF_PATH    = "data/processed/oof_predictions.csv"
TRAIN_PATH  = "data/processed/train_features.parquet"
TARGET_COL  = "next_3m_txn_count"
ID_COL      = "UniqueID"


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


# ---------------------------------------------------------------------------
# OOF analysis
# ---------------------------------------------------------------------------

def validate_oof(oof_path: str = OOF_PATH) -> None:
    """Score the OOF predictions file produced by train.py."""
    print(f"[INFO] Loading OOF from: {oof_path}")
    oof = pd.read_csv(oof_path)

    y_true = oof[f"{TARGET_COL}_true"].values.astype(float)
    y_pred = oof[f"{TARGET_COL}_pred"].values.astype(float)

    overall_rmsle = rmsle(y_true, y_pred)
    overall_mae   = mae(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  Overall RMSLE : {overall_rmsle:.6f}")
    print(f"  Overall MAE   : {overall_mae:.2f}")
    print(f"{'='*50}\n")

    # Error by quantile bucket
    print("Error by target quantile bucket:")
    print(f"  {'Bucket':<20} {'N':>6}  {'RMSLE':>8}  {'MAE':>8}  {'Mean True':>10}  {'Mean Pred':>10}")
    print(f"  {'-'*70}")

    quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    thresholds = np.quantile(y_true, quantiles)

    for i in range(len(quantiles) - 1):
        lo = thresholds[i]
        hi = thresholds[i + 1]
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.sum() == 0:
            continue
        bucket_rmsle = rmsle(y_true[mask], y_pred[mask])
        bucket_mae   = mae(y_true[mask], y_pred[mask])
        label = f"P{int(quantiles[i]*100)}-P{int(quantiles[i+1]*100)}"
        print(
            f"  {label:<20} {mask.sum():>6}  {bucket_rmsle:>8.4f}  "
            f"{bucket_mae:>8.1f}  {y_true[mask].mean():>10.1f}  {y_pred[mask].mean():>10.1f}"
        )

    # Distribution summary
    print(f"\nPrediction distribution:")
    print(f"  Min    : {y_pred.min():.1f}")
    print(f"  P25    : {np.percentile(y_pred, 25):.1f}")
    print(f"  Median : {np.median(y_pred):.1f}")
    print(f"  Mean   : {y_pred.mean():.1f}")
    print(f"  P75    : {np.percentile(y_pred, 75):.1f}")
    print(f"  Max    : {y_pred.max():.1f}")

    print(f"\nTrue target distribution:")
    print(f"  Min    : {y_true.min():.1f}")
    print(f"  P25    : {np.percentile(y_true, 25):.1f}")
    print(f"  Median : {np.median(y_true):.1f}")
    print(f"  Mean   : {y_true.mean():.1f}")
    print(f"  P75    : {np.percentile(y_true, 75):.1f}")
    print(f"  Max    : {y_true.max():.1f}")


# ---------------------------------------------------------------------------
# Feature null report
# ---------------------------------------------------------------------------

def validate_features(train_path: str = TRAIN_PATH) -> None:
    """Check for missing values in the processed training set."""
    print(f"\n[INFO] Checking feature quality in: {train_path}")
    df = pd.read_parquet(train_path)
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)

    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]

    if len(null_cols) == 0:
        print("  [OK] No null values in processed features.")
    else:
        print(f"  [WARN] {len(null_cols)} columns with nulls:")
        for col, count in null_cols.items():
            pct = 100 * count / len(df)
            print(f"    {col:<50} {count:>6}  ({pct:.1f}%)")

    print(f"  Total features : {df.shape[1]}")
    print(f"  Total rows     : {len(df):,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    validate_features()
    validate_oof()
    print("\n[DONE] validate.py complete.")


if __name__ == "__main__":
    main()

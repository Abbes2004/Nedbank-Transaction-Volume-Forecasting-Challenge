"""
merge_features.py
==================
Joins all pre-built feature tables (transactions, financials, demographics)
with Train.csv and Test.csv to produce final model-ready datasets.

Outputs:
    data/processed/train_features.parquet   — 8,360 rows × N features + target
    data/processed/test_features.parquet    — 3,584 rows × N features (no target)

Design decisions:
  - All joins are LEFT joins from the train/test ID list → no leakage,
    no rows added.
  - Customers with no data in a feature table receive 0-filled or -1-filled
    values depending on column type (numeric → 0, encoded categoricals already
    handled upstream).
  - The target column `next_3m_txn_count` is included only in train output.
  - A log1p-transformed target `log1p_target` is also appended to the train
    output as a convenience column for RMSLE-optimised modelling.

Usage:
    python merge_features.py
    python merge_features.py \\
        --train  data/raw/Train.csv \\
        --test   data/raw/Test.csv  \\
        --txn    data/processed/transactions_features_agg.parquet \\
        --fin    data/processed/financials_features_agg.parquet \\
        --demo   data/processed/demographics_features_agg.parquet \\
        --outdir data/processed

You can also call build_all() programmatically from another script.
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── default paths ─────────────────────────────────────────────────────────────
TRAIN_PATH = "data/raw/Train.csv"
TEST_PATH = "data/raw/Test.csv"
TXN_FEAT_PATH = "data/processed/transactions_features_agg.parquet"
FIN_FEAT_PATH = "data/processed/financials_features_agg.parquet"
DEMO_FEAT_PATH = "data/processed/demographics_features_agg.parquet"
OUTPUT_DIR = "data/processed"

# ── loaders ──────────────────────────────────────────────────────────────────

def load_labels(train_path: str, test_path: str):
    """Load Train and Test CSVs. Returns (train_df, test_df)."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"[INFO] Train customers : {len(train):,}")
    print(f"[INFO] Test  customers : {len(test):,}")
    assert train["UniqueID"].nunique() == len(train), "Duplicate UniqueIDs in Train!"
    assert test["UniqueID"].nunique() == len(test), "Duplicate UniqueIDs in Test!"
    assert len(set(train["UniqueID"]) & set(test["UniqueID"])) == 0, \
        "Train/Test UniqueID overlap detected!"
    return train, test


def load_feature_table(path: str, label: str) -> pd.DataFrame:
    """Load a pre-built feature parquet and report shape."""
    print(f"[INFO] Loading {label} features from: {path}")
    df = pd.read_parquet(path)
    print(f"[INFO]   → {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


# ── merge logic ───────────────────────────────────────────────────────────────

def _remove_id_duplicates(feat_df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "UniqueID" not in feat_df.columns:
        feat_df = feat_df.reset_index()

    feat_df = feat_df.drop_duplicates(subset="UniqueID", keep="first")
    return feat_df


def merge_features(
    base: pd.DataFrame,
    txn: pd.DataFrame,
    fin: pd.DataFrame,
    demo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join all feature tables onto a base DataFrame keyed by UniqueID.
    base must contain at minimum a UniqueID column (plus target for train).
    """
    txn = _remove_id_duplicates(txn, "transactions")
    fin = _remove_id_duplicates(fin, "financials")
    demo = _remove_id_duplicates(demo, "demographics")

    merged = base.copy()
    for feat_df, label in [(txn, "transactions"), (fin, "financials"), (demo, "demographics")]:
        n_before = len(merged)
        merged = merged.merge(feat_df, on="UniqueID", how="left")
        assert len(merged) == n_before, (
            f"Row count changed after merging {label} features! "
            f"Before: {n_before}, After: {len(merged)}"
        )
        print(f"[INFO] After merging {label}: {merged.shape[1]} columns.")

    return merged


def fill_missing_after_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values introduced by left joins for customers absent from a
    feature source.  Numeric columns → 0.0.  No string columns expected here
    since all categoricals are already encoded upstream.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target columns from fill
    fill_cols = [c for c in numeric_cols if c not in ("next_3m_txn_count", "log1p_target")]
    df[fill_cols] = df[fill_cols].fillna(0.0)
    return df


def validate_output(df: pd.DataFrame, split_name: str, expected_rows: int):
    """Basic sanity checks on the merged output."""
    assert len(df) == expected_rows, (
        f"{split_name}: expected {expected_rows} rows, got {len(df)}."
    )
    n_nan = df.drop(columns=["next_3m_txn_count", "log1p_target"], errors="ignore").isna().sum().sum()
    if n_nan > 0:
        print(f"[WARN] {split_name}: {n_nan} NaN values remain after merge fill.")
    else:
        print(f"[OK]   {split_name}: no NaN values in feature columns.")
    print(f"[OK]   {split_name}: {len(df):,} rows × {df.shape[1]} columns.")


# ── orchestrator ─────────────────────────────────────────────────────────────

def build_all(
    train_path: str,
    test_path: str,
    txn_feat_path: str,
    fin_feat_path: str,
    demo_feat_path: str,
    output_dir: str,
) -> tuple:
    """
    Load labels, load feature tables, merge, validate, and save.
    Returns (train_features, test_features) as DataFrames.
    """
    # 1. Load
    train, test = load_labels(train_path, test_path)
    txn = load_feature_table(txn_feat_path, "transactions")
    fin = load_feature_table(fin_feat_path, "financials")
    demo = load_feature_table(demo_feat_path, "demographics")

    # 2. Merge
    print("\n[INFO] Merging train split ...")
    train_merged = merge_features(train, txn, fin, demo)

    print("\n[INFO] Merging test split ...")
    test_merged = merge_features(test, txn, fin, demo)

    # 3. Fill remaining NaNs
    train_merged = fill_missing_after_merge(train_merged)
    test_merged = fill_missing_after_merge(test_merged)

    # 4. Add log1p target to train (convenience for RMSLE modelling)
    train_merged["log1p_target"] = np.log1p(train_merged["next_3m_txn_count"])

    # 5. Validate
    print()
    validate_output(train_merged, "train_features", expected_rows=len(train))
    validate_output(test_merged, "test_features", expected_rows=len(test))

    # 6. Save
    os.makedirs(output_dir, exist_ok=True)

    train_out = os.path.join(output_dir, "train_features.parquet")
    test_out = os.path.join(output_dir, "test_features.parquet")

    train_merged.to_parquet(train_out, index=False)
    test_merged.to_parquet(test_out, index=False)

    print(f"\n[INFO] Saved: {train_out}")
    print(f"[INFO] Saved: {test_out}")

    return train_merged, test_merged


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge all feature tables into train/test parquet files."
    )
    parser.add_argument("--train", default=TRAIN_PATH)
    parser.add_argument("--test", default=TEST_PATH)
    parser.add_argument("--txn", default=TXN_FEAT_PATH)
    parser.add_argument("--fin", default=FIN_FEAT_PATH)
    parser.add_argument("--demo", default=DEMO_FEAT_PATH)
    parser.add_argument("--outdir", default=OUTPUT_DIR)
    args = parser.parse_args()

    train_df, test_df = build_all(
        train_path=args.train,
        test_path=args.test,
        txn_feat_path=args.txn,
        fin_feat_path=args.fin,
        demo_feat_path=args.demo,
        output_dir=args.outdir,
    )

    print("\n[SUMMARY] Train feature columns:")
    for col in train_df.columns:
        print(f"  {col}")

    print(f"\n[DONE] train: {train_df.shape}, test: {test_df.shape}")


if __name__ == "__main__":
    main()

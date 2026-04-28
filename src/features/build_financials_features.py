"""
build_financials_features.py
==============================
Builds customer-level features from the raw financials parquet file.

Features produced (1 row per UniqueID):
  - Account counts       : total distinct accounts, by product type
  - Snapshot coverage    : number of snapshots (months of data available)
  - NetInterestIncome    : mean, std, min, max, last value, trend (last - first)
  - NetInterestRevenue   : mean, std, min, max, last value
  - Product flags        : binary indicator per product type
  - Income trajectory    : slope of NII over time (simple linear proxy)

Reference date: 2015-10-31 (consistent with transaction features).

Usage:
    python build_financials_features.py
    python build_financials_features.py --input path/to/financials.parquet \\
                                          --output path/to/out.parquet
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
REFERENCE_DATE = pd.Timestamp("2015-10-31")
INPUT_PATH = "data/raw/financials_features.parquet"
OUTPUT_PATH = "data/processed/financials_features_agg.parquet"

REQUIRED_COLS = [
    "UniqueID",
    "AccountID",
    "RunDate",
    "Product",
    "NetInterestIncome",
    "NetInterestRevenue",
]

PRODUCT_TYPES = ["Transactional", "Investments", "Mortgages"]


# ── loaders ──────────────────────────────────────────────────────────────────

def load_financials(path: str) -> pd.DataFrame:
    """Load required columns and parse dates."""
    print(f"[INFO] Loading financials from: {path}")
    df = pd.read_parquet(path, columns=REQUIRED_COLS)
    df["RunDate"] = pd.to_datetime(df["RunDate"])
    print(f"[INFO] Loaded {len(df):,} rows for {df['UniqueID'].nunique():,} customers.")
    return df


# ── feature builders ─────────────────────────────────────────────────────────

def build_account_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count distinct accounts overall and per product type.
    Note: Mortgages have null AccountID — we count UniqueID rows instead for them.
    """
    # Total distinct non-null AccountIDs
    total_accs = (
        df[df["AccountID"].notna()]
        .groupby("UniqueID")["AccountID"]
        .nunique()
        .rename("n_distinct_accounts")
        .reset_index()
    )

    # Rows per product type → proxy for "has this product"
    product_counts = []
    for product in PRODUCT_TYPES:
        col = f"has_{product.lower()}"
        flag = (
            df[df["Product"] == product]
            .groupby("UniqueID")
            .size()
            .gt(0)
            .astype(int)
            .rename(col)
            .reset_index()
        )
        product_counts.append(flag)

    result = total_accs
    for pc in product_counts:
        result = result.merge(pc, on="UniqueID", how="outer")
    result = result.fillna(0)
    result["n_distinct_accounts"] = result["n_distinct_accounts"].astype(int)
    return result


def build_snapshot_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct RunDate snapshots per customer (data richness proxy)."""
    coverage = (
        df.groupby("UniqueID")["RunDate"]
        .nunique()
        .rename("n_financial_snapshots")
        .reset_index()
    )
    return coverage


def build_nii_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NetInterestIncome aggregations: mean, std, min, max, last value,
    and a simple trend (last NII − first NII, ordered by RunDate).
    """
    df_sorted = df.sort_values("RunDate")

    agg = df.groupby("UniqueID")["NetInterestIncome"].agg(
        nii_mean="mean",
        nii_std="std",
        nii_min="min",
        nii_max="max",
    ).reset_index()
    agg["nii_std"] = agg["nii_std"].fillna(0.0)

    first_nii = (
        df_sorted.groupby("UniqueID")["NetInterestIncome"]
        .first()
        .rename("nii_first")
        .reset_index()
    )
    last_nii = (
        df_sorted.groupby("UniqueID")["NetInterestIncome"]
        .last()
        .rename("nii_last")
        .reset_index()
    )

    trend = first_nii.merge(last_nii, on="UniqueID")
    trend["nii_trend"] = trend["nii_last"] - trend["nii_first"]

    result = agg.merge(trend[["UniqueID", "nii_last", "nii_trend"]], on="UniqueID", how="left")
    return result


def build_nir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NetInterestRevenue aggregations: mean, std, min, max, last value.
    """
    df_sorted = df.sort_values("RunDate")

    agg = df.groupby("UniqueID")["NetInterestRevenue"].agg(
        nir_mean="mean",
        nir_std="std",
        nir_min="min",
        nir_max="max",
    ).reset_index()
    agg["nir_std"] = agg["nir_std"].fillna(0.0)

    last_nir = (
        df_sorted.groupby("UniqueID")["NetInterestRevenue"]
        .last()
        .rename("nir_last")
        .reset_index()
    )
    result = agg.merge(last_nir, on="UniqueID", how="left")
    return result


def build_recency_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Days since the most recent financial snapshot relative to REFERENCE_DATE."""
    last_snap = (
        df.groupby("UniqueID")["RunDate"]
        .max()
        .reset_index()
    )
    last_snap["fin_recency_days"] = (REFERENCE_DATE - last_snap["RunDate"]).dt.days
    return last_snap[["UniqueID", "fin_recency_days"]]


# ── orchestrator ─────────────────────────────────────────────────────────────

def build_financials_features(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load raw financials and produce one feature row per customer.
    Writes result to output_path and also returns the DataFrame.
    """
    df = load_financials(input_path)

    print("[INFO] Building account count features ...")
    accounts = build_account_count_features(df)

    print("[INFO] Building snapshot coverage ...")
    coverage = build_snapshot_coverage(df)

    print("[INFO] Building NetInterestIncome features ...")
    nii = build_nii_features(df)

    print("[INFO] Building NetInterestRevenue features ...")
    nir = build_nir_features(df)

    print("[INFO] Building financial recency feature ...")
    recency = build_recency_feature(df)

    # ── merge ──────────────────────────────────────────────────────────────
    print("[INFO] Merging all financial feature tables ...")
    feature_tables = [accounts, coverage, nii, nir, recency]
    features = feature_tables[0]
    for tbl in feature_tables[1:]:
        features = features.merge(tbl, on="UniqueID", how="outer")

    # NaN filling: numeric columns → 0 (customer not present in financials)
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features[numeric_cols] = features[numeric_cols].fillna(0.0)

    print(f"[INFO] Final shape: {features.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_parquet(output_path, index=False)
    print(f"[INFO] Saved to: {output_path}")
    return features


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build financials-level customer features.")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to financials parquet file.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path for output parquet file.")
    args = parser.parse_args()

    features = build_financials_features(args.input, args.output)

    print("\n[SUMMARY] Feature columns:")
    for col in features.columns:
        print(f"  {col}")
    print(f"\n[DONE] {features.shape[0]:,} customers, {features.shape[1]} features.")


if __name__ == "__main__":
    main()

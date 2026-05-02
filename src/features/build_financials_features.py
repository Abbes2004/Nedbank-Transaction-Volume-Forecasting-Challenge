"""
build_financials_features.py
==============================
Builds customer-level features from the financial snapshots data.

Source   : data/raw/financials_features.parquet
Output   : data/interim/financials_features.parquet

Features
--------
- Average / std / last balance (NetInterestIncome, NetInterestRevenue)
- Balance volatility (std / |mean|)
- Number of distinct accounts per customer
- Number of distinct products per customer
- Product-type indicators and counts (Transactional, Investments, Mortgages)
- Loan-to-non-loan account ratio
- Snapshot count (proxy for data richness / tenure)
- Most recent snapshot values (last known state)
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTOFF_DATE = pd.Timestamp("2015-10-31")
RAW_PATH    = "data/raw/financials_features.parquet"
OUT_PATH    = "data/interim/financials_features.parquet"

LOAD_COLS = [
    "UniqueID",
    "AccountID",
    "RunDate",
    "Product",
    "NetInterestIncome",
    "NetInterestRevenue",
]

LOAN_PRODUCTS      = {"Mortgages"}
NON_LOAN_PRODUCTS  = {"Transactional", "Investments"}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_financials(path: str = RAW_PATH) -> pd.DataFrame:
    print(f"[INFO] Loading financials from: {path}")
    fin = pd.read_parquet(path, columns=LOAD_COLS)
    fin["RunDate"] = pd.to_datetime(fin["RunDate"])
    # Keep only snapshots up to cutoff (no leakage)
    fin = fin[fin["RunDate"] <= CUTOFF_DATE].copy()
    print(f"[INFO] {len(fin):,} rows after cutoff filter.")
    return fin


# ---------------------------------------------------------------------------
# Balance / income features
# ---------------------------------------------------------------------------

def build_balance_features(fin: pd.DataFrame) -> pd.DataFrame:
    """
    For NetInterestIncome and NetInterestRevenue compute:
      mean, std, min, max, last value, volatility (std / |mean|).
    """
    frames = []
    for col in ["NetInterestIncome", "NetInterestRevenue"]:
        agg = fin.groupby("UniqueID")[col].agg(
            **{
                f"{col}_mean": "mean",
                f"{col}_std":  "std",
                f"{col}_min":  "min",
                f"{col}_max":  "max",
                f"{col}_sum":  "sum",
            }
        )
        # Last known value (most recent snapshot)
        last = (
            fin.sort_values("RunDate")
            .groupby("UniqueID")[col]
            .last()
            .rename(f"{col}_last")
        )
        frames.append(pd.concat([agg, last], axis=1))

    feat = pd.concat(frames, axis=1)

    # Volatility ratios
    for col in ["NetInterestIncome", "NetInterestRevenue"]:
        feat[f"{col}_volatility"] = (
            feat[f"{col}_std"] / feat[f"{col}_mean"].abs().clip(lower=1e-6)
        )

    return feat.fillna(0)


# ---------------------------------------------------------------------------
# Account-level features
# ---------------------------------------------------------------------------

def build_account_features(fin: pd.DataFrame) -> pd.DataFrame:
    """
    n_accounts        : distinct AccountIDs (excluding null / Mortgage nulls)
    n_products        : distinct Product types
    has_mortgage      : 1/0 flag
    has_investment    : 1/0 flag
    has_transactional : 1/0 flag
    loan_account_ratio: mortgage count / total account count
    snapshot_count    : total snapshot rows (proxy for data richness)
    """
    # Non-null AccountIDs
    non_null_acc = fin.dropna(subset=["AccountID"])
    n_accounts = (
        non_null_acc.groupby("UniqueID")["AccountID"]
        .nunique()
        .rename("n_accounts")
    )

    n_products = (
        fin.groupby("UniqueID")["Product"]
        .nunique()
        .rename("n_products")
    )

    snapshot_count = (
        fin.groupby("UniqueID")
        .size()
        .rename("snapshot_count")
    )

    # Product flags
    product_flags = {}
    for prod, flag in [
        ("Mortgages",      "has_mortgage"),
        ("Investments",    "has_investment"),
        ("Transactional",  "has_transactional"),
    ]:
        mask = fin[fin["Product"] == prod].groupby("UniqueID").size().gt(0).astype(int)
        product_flags[flag] = mask.rename(flag)

    # Loan ratio: number of Mortgage rows / total rows per customer
    mortgage_rows = (
        fin[fin["Product"].isin(LOAN_PRODUCTS)]
        .groupby("UniqueID")
        .size()
        .rename("mortgage_row_count")
    )

    feat = pd.concat(
        [n_accounts, n_products, snapshot_count] +
        list(product_flags.values()) +
        [mortgage_rows],
        axis=1,
    ).fillna(0)

    feat["loan_account_ratio"] = (
        feat["mortgage_row_count"] / feat["snapshot_count"].clip(lower=1)
    )

    return feat


# ---------------------------------------------------------------------------
# Trend in financial data
# ---------------------------------------------------------------------------

def build_financial_trend(fin: pd.DataFrame) -> pd.DataFrame:
    """
    Compare last 6 months vs all-time mean for NetInterestIncome.
    Captures whether the customer's financial engagement is growing.
    """
    last6_start = CUTOFF_DATE - pd.Timedelta(days=180)
    recent  = fin[fin["RunDate"] > last6_start]
    overall = fin

    recent_mean = (
        recent.groupby("UniqueID")["NetInterestIncome"]
        .mean()
        .rename("nii_recent_mean")
    )
    overall_mean = (
        overall.groupby("UniqueID")["NetInterestIncome"]
        .mean()
        .rename("nii_overall_mean")
    )

    trend = pd.concat([recent_mean, overall_mean], axis=1).fillna(0)
    trend["nii_trend_ratio"] = (
        trend["nii_recent_mean"] / trend["nii_overall_mean"].abs().clip(lower=1e-6)
    )

    return trend[["nii_recent_mean", "nii_trend_ratio"]]


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_financials_features(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Full pipeline. Returns DataFrame indexed by UniqueID.
    """
    fin = _load_financials(path)

    print("[INFO] Building balance features ...")
    bal  = build_balance_features(fin)

    print("[INFO] Building account features ...")
    acc  = build_account_features(fin)

    print("[INFO] Building financial trend features ...")
    trnd = build_financial_trend(fin)

    feats = pd.concat([bal, acc, trnd], axis=1).fillna(0)
    feats.index.name = "UniqueID"

    print(f"[INFO] Financial features shape: {feats.shape}")
    return feats


def save_features(feats: pd.DataFrame, out_path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feats.reset_index().to_parquet(out_path, index=False)
    print(f"[INFO] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    feats = build_financials_features()
    save_features(feats)
    print("[DONE] build_financials_features complete.")


if __name__ == "__main__":
    main()

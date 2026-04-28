"""
build_transactions_features.py
================================
Builds customer-level features from the raw transactions parquet file.

Features produced (1 row per UniqueID):
  - Overall stats        : total count, amount stats (mean, std, min, max, median)
  - Debit / Credit split : counts and amount sums
  - Temporal windows     : last-30-day and last-90-day activity counts & amounts
  - Recency              : days since last transaction (relative to REFERENCE_DATE)
  - Frequency trends     : monthly transaction counts for the 3 months before cutoff
  - Account diversity    : number of distinct accounts used
  - Type diversity       : number of distinct TransactionTypeDescriptions
  - Balance features     : last StatementBalance, mean & std StatementBalance
  - Reversal rate        : fraction of transactions with a non-null ReversalTypeDescription

Reference date: 2015-10-31 (last date in dataset, just before prediction window).

Usage:
    python build_transactions_features.py
    python build_transactions_features.py --input path/to/transactions.parquet \\
                                           --output path/to/out.parquet
"""

import argparse
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ── constants ────────────────────────────────────────────────────────────────
REFERENCE_DATE = pd.Timestamp("2015-10-31")
INPUT_PATH = "data/raw/transactions_features.parquet"
OUTPUT_PATH = "data/processed/transactions_features_agg.parquet"

# Only load the columns we actually need — critical for 18 M-row dataset
REQUIRED_COLS = [
    "UniqueID",
    "AccountID",
    "TransactionDate",
    "TransactionAmount",
    "TransactionTypeDescription",
    "IsDebitCredit",
    "StatementBalance",
    "ReversalTypeDescription",
]


# ── loaders ──────────────────────────────────────────────────────────────────

def load_transactions(path):

    table = pq.ParquetFile(path)

    for batch in table.iter_batches(batch_size=500_000):
        yield batch.to_pandas()


# ── feature builders ─────────────────────────────────────────────────────────

def build_overall_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Global transaction count and amount aggregations per customer."""
    grp = df.groupby("UniqueID")
    stats = grp["TransactionAmount"].agg(
        txn_count="count",
        amount_mean="mean",
        amount_std="std",
        amount_min="min",
        amount_max="max",
        amount_median="median",
        amount_sum="sum",
    ).reset_index()
    stats["amount_std"] = stats["amount_std"].fillna(0.0)
    return stats


def build_debit_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Debit vs Credit counts and total amounts."""
    debit = df[df["IsDebitCredit"] == "D"].groupby("UniqueID").agg(
        debit_count=("TransactionAmount", "count"),
        debit_sum=("TransactionAmount", "sum"),
    ).reset_index()

    credit = df[df["IsDebitCredit"] == "C"].groupby("UniqueID").agg(
        credit_count=("TransactionAmount", "count"),
        credit_sum=("TransactionAmount", "sum"),
    ).reset_index()

    combined = debit.merge(credit, on="UniqueID", how="outer").fillna(0.0)

    # Derived ratio — avoid divide-by-zero
    total = combined["debit_count"] + combined["credit_count"]
    combined["debit_ratio"] = np.where(total > 0, combined["debit_count"] / total, 0.0)
    return combined


def build_window_features(df: pd.DataFrame, days: int, prefix: str) -> pd.DataFrame:
    """Transaction count and total amount in the last `days` days."""
    cutoff = REFERENCE_DATE - pd.Timedelta(days=days)
    window = df[df["TransactionDate"] > cutoff]
    feats = window.groupby("UniqueID").agg(
        **{
            f"{prefix}_txn_count": ("TransactionAmount", "count"),
            f"{prefix}_amount_sum": ("TransactionAmount", "sum"),
            f"{prefix}_amount_mean": ("TransactionAmount", "mean"),
        }
    ).reset_index()
    return feats


def build_recency_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Days since the customer's most recent transaction."""
    last_date = df.groupby("UniqueID")["TransactionDate"].max().reset_index()
    last_date.columns = ["UniqueID", "last_txn_date"]
    last_date["recency_days"] = (REFERENCE_DATE - last_date["last_txn_date"]).dt.days
    return last_date[["UniqueID", "recency_days"]]


def build_monthly_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction counts for each of the 3 months immediately before the cutoff:
    Aug, Sep, Oct 2015. Captures recent trend momentum.
    """
    months = {
        "m_aug2015": ("2015-08-01", "2015-08-31"),
        "m_sep2015": ("2015-09-01", "2015-09-30"),
        "m_oct2015": ("2015-10-01", "2015-10-31"),
    }
    frames = []
    for col_name, (start, end) in months.items():
        mask = (df["TransactionDate"] >= start) & (df["TransactionDate"] <= end)
        cnt = (
            df[mask]
            .groupby("UniqueID")["TransactionAmount"]
            .count()
            .rename(col_name)
            .reset_index()
        )
        frames.append(cnt)

    trend = frames[0]
    for f in frames[1:]:
        trend = trend.merge(f, on="UniqueID", how="outer")
    trend = trend.fillna(0.0)

    # Simple slope: oct - aug  (positive = increasing activity)
    trend["trend_slope_3m"] = trend["m_oct2015"] - trend["m_aug2015"]
    return trend


def build_diversity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct accounts and transaction types per customer."""
    account_div = (
        df.groupby("UniqueID")["AccountID"]
        .nunique()
        .rename("n_distinct_accounts")
        .reset_index()
    )
    type_div = (
        df.groupby("UniqueID")["TransactionTypeDescription"]
        .nunique()
        .rename("n_txn_types")
        .reset_index()
    )
    return account_div.merge(type_div, on="UniqueID", how="outer")


def build_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    StatementBalance-based features: last balance, mean, and std.
    Sorted by date to ensure 'last' is the most recent.
    """
    df_sorted = df.sort_values("TransactionDate")
    last_bal = (
        df_sorted.groupby("UniqueID")["StatementBalance"]
        .last()
        .rename("last_statement_balance")
        .reset_index()
    )
    bal_stats = df.groupby("UniqueID")["StatementBalance"].agg(
        balance_mean="mean",
        balance_std="std",
    ).reset_index()
    bal_stats["balance_std"] = bal_stats["balance_std"].fillna(0.0)
    return last_bal.merge(bal_stats, on="UniqueID", how="left")


def build_reversal_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of transactions that are reversals."""
    df = df.copy()
    df["is_reversal"] = df["ReversalTypeDescription"].notna().astype(int)
    rev = df.groupby("UniqueID").agg(
        reversal_count=("is_reversal", "sum"),
        reversal_rate=("is_reversal", "mean"),
    ).reset_index()
    return rev


# ── orchestrator ─────────────────────────────────────────────────────────────

def build_transaction_features(input_path, output_path):
    

    agg_list = []

    for chunk in load_transactions(input_path):
        # conversions légères
        chunk["TransactionDate"] = pd.to_datetime(chunk["TransactionDate"])

        # features simples (exemple)
        chunk["TransactionAmount"] = chunk["TransactionAmount"].astype("float32")
        chunk["UniqueID"] = chunk["UniqueID"].astype("category")

        chunk["is_debit"] = (chunk["TransactionAmount"] < 0).astype(int)
        chunk["is_credit"] = (chunk["TransactionAmount"] > 0).astype(int)

        grp = chunk.groupby("UniqueID").agg(
            txn_count=("TransactionAmount", "count"),
            txn_sum=("TransactionAmount", "sum"),
            debit_count=("is_debit", "sum"),
            credit_count=("is_credit", "sum"),
            txn_mean=("TransactionAmount", "mean"),
        )

        agg_list.append(grp)

    # merge final (safe memory)
    final = pd.concat(agg_list).groupby(level=0).sum().reset_index()
    final.to_parquet(output_path)
    print(final.columns)
    return final


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build transaction-level customer features.")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to transactions parquet file.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path for output parquet file.")
    args = parser.parse_args()

    features = build_transaction_features(args.input, args.output)

    print("\n[SUMMARY] Feature columns:")
    for col in features.columns:
        print(f"  {col}")
    print(f"\n[DONE] {features.shape[0]:,} customers, {features.shape[1]} features.")


if __name__ == "__main__":
    main()

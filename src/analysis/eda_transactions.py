"""
eda_transactions.py
-------------------
Exploratory Data Analysis for transactions_features.parquet.

Responsibility: Analyse transaction history — shape, dates, amounts,
debit/credit split, transaction types, and memory usage.

Run:
    python eda_transactions.py
"""

import pandas as pd
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = "data/raw/transactions_features.parquet"

# Columns we actually need for EDA (avoids loading unused columns into RAM)
REQUIRED_COLS = [
    "UniqueID",
    "AccountID",
    "TransactionDate",
    "TransactionAmount",
    "TransactionTypeDescription",
]


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_transactions(path: str = DATA_PATH) -> pd.DataFrame:
    """Load only the required columns from the parquet file."""
    print(f"[INFO] Loading transactions from: {path}")
    df = pd.read_parquet(path, columns=REQUIRED_COLS, engine="pyarrow")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    print(f"[INFO] Loaded {len(df):,} rows, {df.shape[1]} columns.\n")
    return df


# ── Analysis functions ────────────────────────────────────────────────────────
def show_schema(df: pd.DataFrame) -> None:
    """Print shape, column names, and dtypes."""
    print("=" * 60)
    print("SCHEMA")
    print("=" * 60)
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print()
    print(f"  {'Column':<35} {'Dtype'}")
    print(f"  {'-'*35} {'-'*15}")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<35} {dtype}")
    print()


def show_missing_values(df: pd.DataFrame) -> None:
    """Print missing-value counts and percentages for every column."""
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)
    if report.empty:
        print("  No missing values found.\n")
    else:
        for col, row in report.iterrows():
            print(f"  {col:<35} {row['missing_count']:>10,}  ({row['missing_pct']}%)")
        print()


def show_date_range(df: pd.DataFrame) -> None:
    """Print the min/max TransactionDate."""
    print("=" * 60)
    print("DATE RANGE")
    print("=" * 60)
    min_date = df["TransactionDate"].min()
    max_date = df["TransactionDate"].max()
    n_null_dates = df["TransactionDate"].isnull().sum()
    print(f"  Min date        : {min_date.date()}")
    print(f"  Max date        : {max_date.date()}")
    print(f"  Null dates      : {n_null_dates:,}")
    print(f"  Span (days)     : {(max_date - min_date).days:,}")
    print()


def show_transactions_per_customer(df: pd.DataFrame) -> None:
    """Print distribution statistics for transactions per customer."""
    print("=" * 60)
    print("TRANSACTIONS PER CUSTOMER")
    print("=" * 60)
    counts = df.groupby("UniqueID").size()
    print(f"  Unique customers   : {counts.shape[0]:,}")
    print(f"  Mean txn / customer: {counts.mean():.2f}")
    print(f"  Median             : {counts.median():.0f}")
    print(f"  Std                : {counts.std():.2f}")
    print(f"  Min                : {counts.min():,}")
    print(f"  Max                : {counts.max():,}")
    print()
    # Percentile breakdown
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("  Percentile breakdown:")
    for p in percentiles:
        print(f"    P{p:<3}: {np.percentile(counts, p):,.0f}")
    print()


def show_amount_stats(df: pd.DataFrame) -> None:
    """Print descriptive statistics for TransactionAmount."""
    print("=" * 60)
    print("TRANSACTION AMOUNT STATS")
    print("=" * 60)
    amt = df["TransactionAmount"].dropna()
    print(f"  Count  : {len(amt):,}")
    print(f"  Mean   : {amt.mean():,.4f}")
    print(f"  Std    : {amt.std():,.4f}")
    print(f"  Min    : {amt.min():,.4f}")
    print(f"  Max    : {amt.max():,.4f}")
    print()
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print("  Percentile breakdown:")
    for p in percentiles:
        print(f"    P{p:<3}: {np.percentile(amt, p):>15,.4f}")
    print()


def show_debit_credit_split(df: pd.DataFrame) -> None:
    """Print debit vs credit transaction counts and total amounts."""
    print("=" * 60)
    print("DEBIT VS CREDIT")
    print("=" * 60)
    amt = df["TransactionAmount"].dropna()
    debits = amt[amt < 0]
    credits = amt[amt > 0]
    zeros = amt[amt == 0]
    total = len(amt)

    print(f"  {'Type':<12} {'Count':>12}  {'% of total':>12}  {'Total Amount':>18}")
    print(f"  {'-'*12} {'-'*12}  {'-'*12}  {'-'*18}")
    print(f"  {'Debit (<0)':<12} {len(debits):>12,}  {len(debits)/total*100:>11.2f}%  {debits.sum():>18,.2f}")
    print(f"  {'Credit (>0)':<12} {len(credits):>12,}  {len(credits)/total*100:>11.2f}%  {credits.sum():>18,.2f}")
    print(f"  {'Zero':<12} {len(zeros):>12,}  {len(zeros)/total*100:>11.2f}%  {'0.00':>18}")
    print()


def show_top_transaction_types(df: pd.DataFrame, top_n: int = 15) -> None:
    """Print the most frequent TransactionTypeDescription values."""
    print("=" * 60)
    print(f"TOP {top_n} TRANSACTION TYPE DESCRIPTIONS")
    print("=" * 60)
    if "TransactionTypeDescription" not in df.columns:
        print("  Column not available.\n")
        return
    counts = (
        df["TransactionTypeDescription"]
        .value_counts()
        .head(top_n)
    )
    total = len(df)
    print(f"  {'Type':<45} {'Count':>10}  {'%':>7}")
    print(f"  {'-'*45} {'-'*10}  {'-'*7}")
    for txn_type, cnt in counts.items():
        print(f"  {str(txn_type):<45} {cnt:>10,}  {cnt/total*100:>6.2f}%")
    print()


def show_memory_usage(df: pd.DataFrame) -> None:
    """Print per-column and total memory usage."""
    print("=" * 60)
    print("MEMORY USAGE")
    print("=" * 60)
    mem = df.memory_usage(deep=True)
    total_mb = mem.sum() / 1024 ** 2
    print(f"  {'Column':<35} {'MB':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for col, bytes_used in mem.items():
        if col == "Index":
            continue
        print(f"  {col:<35} {bytes_used / 1024**2:>10.2f}")
    print(f"\n  Total memory usage: {total_mb:.2f} MB\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    df = load_transactions()
    show_schema(df)
    show_missing_values(df)
    show_date_range(df)
    show_transactions_per_customer(df)
    show_amount_stats(df)
    show_debit_credit_split(df)
    show_top_transaction_types(df)
    show_memory_usage(df)
    print("[DONE] eda_transactions.py complete.")


if __name__ == "__main__":
    main()

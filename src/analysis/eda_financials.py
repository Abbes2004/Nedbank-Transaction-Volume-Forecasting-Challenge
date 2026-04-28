"""
eda_financials.py
-----------------
Exploratory Data Analysis for financials_features.parquet.

Responsibility: Analyse financial snapshots — schema, missing values,
accounts per customer, snapshot frequency, product mix, and income metrics.

Run:
    python eda_financials.py
"""

import pandas as pd
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = "data/raw/financials_features.parquet"


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_financials(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the full financials parquet file."""
    print(f"[INFO] Loading financials from: {path}")
    df = pd.read_parquet(path)
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
    print(f"  {'Column':<40} {'Dtype'}")
    print(f"  {'-'*40} {'-'*15}")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<40} {dtype}")
    print()


def show_missing_values(df: pd.DataFrame) -> None:
    """Print missing-value analysis with special note on AccountID."""
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report.sort_values("missing_count", ascending=False)
    report_nonzero = report[report["missing_count"] > 0]

    if report_nonzero.empty:
        print("  No missing values found.\n")
    else:
        print(f"  {'Column':<40} {'Missing':>10}  {'%':>8}")
        print(f"  {'-'*40} {'-'*10}  {'-'*8}")
        for col, row in report_nonzero.iterrows():
            flag = " ← 100% null (Mortgage rows)" if row["missing_pct"] == 100.0 else ""
            print(f"  {col:<40} {row['missing_count']:>10,}  {row['missing_pct']:>7.2f}%{flag}")
        print()

    # Special note about AccountID
    if "AccountID" in df.columns:
        n_null_account = df["AccountID"].isnull().sum()
        pct_null_account = n_null_account / total * 100
        print(f"  [NOTE] AccountID: {n_null_account:,} nulls ({pct_null_account:.1f}%)")
        print("         Mortgage rows have no AccountID — join via UniqueID only.\n")


def show_accounts_per_customer(df: pd.DataFrame) -> None:
    """Print distribution of accounts per customer."""
    print("=" * 60)
    print("ACCOUNTS PER CUSTOMER")
    print("=" * 60)

    if "AccountID" not in df.columns or "UniqueID" not in df.columns:
        print("  Required columns missing.\n")
        return

    # Only count rows where AccountID is not null
    df_with_account = df[df["AccountID"].notna()]
    accounts_per_customer = (
        df_with_account.groupby("UniqueID")["AccountID"].nunique()
    )

    print(f"  Customers with ≥1 account : {accounts_per_customer.shape[0]:,}")
    print(f"  Mean accounts / customer  : {accounts_per_customer.mean():.2f}")
    print(f"  Median                    : {accounts_per_customer.median():.0f}")
    print(f"  Max                       : {accounts_per_customer.max():,}")
    print()

    # Distribution table
    dist = accounts_per_customer.value_counts().sort_index().head(10)
    print("  Distribution (accounts → customers):")
    for n_accounts, n_customers in dist.items():
        print(f"    {n_accounts} account(s): {n_customers:,} customers")
    print()


def show_snapshots_per_account(df: pd.DataFrame) -> None:
    """Print how many financial snapshot rows exist per account."""
    print("=" * 60)
    print("SNAPSHOTS PER ACCOUNT")
    print("=" * 60)

    if "AccountID" not in df.columns:
        print("  AccountID column missing.\n")
        return

    df_with_account = df[df["AccountID"].notna()]
    snaps = df_with_account.groupby("AccountID").size()

    print(f"  Unique accounts     : {snaps.shape[0]:,}")
    print(f"  Mean snapshots      : {snaps.mean():.2f}")
    print(f"  Median snapshots    : {snaps.median():.0f}")
    print(f"  Std                 : {snaps.std():.2f}")
    print(f"  Min                 : {snaps.min()}")
    print(f"  Max                 : {snaps.max()}")
    print()


def show_product_distribution(df: pd.DataFrame, top_n: int = 15) -> None:
    """Print product-type distribution."""
    print("=" * 60)
    print(f"PRODUCT DISTRIBUTION (top {top_n})")
    print("=" * 60)

    # Try common product-column names
    product_col = None
    for candidate in ["ProductName", "ProductType", "Product", "ProductDescription"]:
        if candidate in df.columns:
            product_col = candidate
            break

    if product_col is None:
        # Fall back to any column with 'product' in the name (case-insensitive)
        matches = [c for c in df.columns if "product" in c.lower()]
        if matches:
            product_col = matches[0]

    if product_col is None:
        print("  No product column found. Available columns:\n  ", list(df.columns), "\n")
        return

    print(f"  Column used: {product_col}\n")
    counts = df[product_col].value_counts().head(top_n)
    total = len(df)
    print(f"  {'Product':<40} {'Count':>10}  {'%':>8}")
    print(f"  {'-'*40} {'-'*10}  {'-'*8}")
    for product, cnt in counts.items():
        print(f"  {str(product):<40} {cnt:>10,}  {cnt/total*100:>7.2f}%")
    print()


def show_net_interest_income_stats(df: pd.DataFrame) -> None:
    """Print descriptive statistics for NetInterestIncome (if present)."""
    print("=" * 60)
    print("NET INTEREST INCOME STATS")
    print("=" * 60)

    col = "NetInterestIncome"
    if col not in df.columns:
        # Try any column containing 'interest'
        matches = [c for c in df.columns if "interest" in c.lower()]
        if matches:
            col = matches[0]
            print(f"  [NOTE] '{col}' used as proxy for NetInterestIncome.\n")
        else:
            print("  NetInterestIncome column not found.\n")
            return

    series = df[col].dropna()
    n_null = df[col].isnull().sum()
    print(f"  Non-null count : {len(series):,}")
    print(f"  Null count     : {n_null:,}  ({n_null/len(df)*100:.2f}%)")
    print(f"  Mean           : {series.mean():,.4f}")
    print(f"  Std            : {series.std():,.4f}")
    print(f"  Min            : {series.min():,.4f}")
    print(f"  Max            : {series.max():,.4f}")
    print()
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print("  Percentile breakdown:")
    for p in percentiles:
        print(f"    P{p:<3}: {np.percentile(series, p):>18,.4f}")
    print()


def show_numeric_column_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of all numeric columns."""
    print("=" * 60)
    print("NUMERIC COLUMN SUMMARY")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("  No numeric columns found.\n")
        return
    print(f"  {'Column':<40} {'Non-null':>10}  {'Mean':>14}  {'Std':>14}  {'Min':>14}  {'Max':>14}")
    print(f"  {'-'*40} {'-'*10}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            print(f"  {col:<40} {'0':>10}  {'N/A':>14}  {'N/A':>14}  {'N/A':>14}  {'N/A':>14}")
        else:
            print(
                f"  {col:<40} {len(s):>10,}  {s.mean():>14.4f}  "
                f"{s.std():>14.4f}  {s.min():>14.4f}  {s.max():>14.4f}"
            )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    df = load_financials()
    show_schema(df)
    show_missing_values(df)
    show_accounts_per_customer(df)
    show_snapshots_per_account(df)
    show_product_distribution(df)
    show_net_interest_income_stats(df)
    show_numeric_column_summary(df)
    print("[DONE] eda_financials.py complete.")


if __name__ == "__main__":
    main()

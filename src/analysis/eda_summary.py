"""
eda_summary.py
--------------
Exploratory Data Analysis for Train.csv and Test.csv.

Responsibility: Validate splits, inspect target distribution (next_3m_txn_count),
detect outliers, and print high-level insights for the modelling phase.

Run:
    python eda_summary.py
"""

import pandas as pd
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/raw/Train.csv"
TEST_PATH = "data/raw/Test.csv"
TARGET_COL = "next_3m_txn_count"

# IQR multiplier for outlier detection
IQR_MULTIPLIER = 3.0

# Additional fixed threshold for business-logic outlier check
HIGH_TXN_THRESHOLD = 500


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_splits(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Train.csv and Test.csv."""
    print(f"[INFO] Loading train from : {train_path}")
    train = pd.read_csv(train_path)
    print(f"[INFO] Loading test from  : {test_path}")
    test = pd.read_csv(test_path)
    print()
    return train, test


# ── Analysis functions ────────────────────────────────────────────────────────
def show_split_overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print row counts, column lists, and dtype info for both splits."""
    print("=" * 60)
    print("SPLIT OVERVIEW")
    print("=" * 60)
    print(f"  Train rows      : {len(train):,}")
    print(f"  Train columns   : {list(train.columns)}")
    print(f"  Test rows       : {len(test):,}")
    print(f"  Test columns    : {list(test.columns)}")
    total = len(train) + len(test)
    print(f"  Total customers : {total:,}")
    print(f"  Train share     : {len(train)/total*100:.1f}%")
    print(f"  Test share      : {len(test)/total*100:.1f}%")
    print()


def show_id_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check whether any UniqueID appears in both train and test."""
    print("=" * 60)
    print("TRAIN / TEST ID OVERLAP")
    print("=" * 60)
    train_ids = set(train["UniqueID"].unique())
    test_ids = set(test["UniqueID"].unique())

    overlap = train_ids & test_ids
    train_only = train_ids - test_ids
    test_only = test_ids - train_ids

    print(f"  Unique IDs in train    : {len(train_ids):,}")
    print(f"  Unique IDs in test     : {len(test_ids):,}")
    print(f"  IDs in both sets       : {len(overlap):,}  ← should be 0")
    print(f"  Train-only IDs         : {len(train_only):,}")
    print(f"  Test-only IDs          : {len(test_only):,}")

    if overlap:
        print(f"\n  [WARNING] {len(overlap)} ID(s) appear in both train AND test!")
    else:
        print("\n  [OK] No overlap — train and test are disjoint.")
    print()

    # Duplicate ID check within each split
    train_dupes = train["UniqueID"].duplicated().sum()
    test_dupes = test["UniqueID"].duplicated().sum()
    print(f"  Duplicate UniqueIDs in train : {train_dupes:,}")
    print(f"  Duplicate UniqueIDs in test  : {test_dupes:,}")
    if train_dupes or test_dupes:
        print("  [WARNING] Duplicates found — investigate before modelling.")
    else:
        print("  [OK] No duplicate IDs within each split.")
    print()


def show_target_missing(train: pd.DataFrame) -> None:
    """Check for nulls in the target column."""
    print("=" * 60)
    print("TARGET MISSING VALUES")
    print("=" * 60)
    n_null = train[TARGET_COL].isnull().sum()
    print(f"  Null target rows : {n_null:,}")
    if n_null:
        print("  [WARNING] Target has null values — these rows cannot be used for training.")
    else:
        print("  [OK] No null values in target.")
    print()


def show_target_distribution(train: pd.DataFrame) -> None:
    """Print full descriptive statistics for next_3m_txn_count."""
    print("=" * 60)
    print("TARGET DISTRIBUTION  (next_3m_txn_count)")
    print("=" * 60)
    y = train[TARGET_COL].dropna()

    print(f"  Count   : {len(y):,}")
    print(f"  Mean    : {y.mean():.4f}")
    print(f"  Median  : {y.median():.0f}")
    print(f"  Std     : {y.std():.4f}")
    print(f"  Min     : {y.min():,}")
    print(f"  Max     : {y.max():,}")
    print(f"  Skew    : {y.skew():.4f}")
    print(f"  Kurtosis: {y.kurt():.4f}")
    print()

    # Percentile breakdown
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("  Percentile breakdown:")
    for p in percentiles:
        print(f"    P{p:<3}: {np.percentile(y, p):>10.0f}")
    print()

    # Zero / low transaction customers
    n_zero = (y == 0).sum()
    n_low = (y <= 5).sum()
    print(f"  Customers with 0 transactions : {n_zero:,}  ({n_zero/len(y)*100:.2f}%)")
    print(f"  Customers with ≤5 transactions: {n_low:,}  ({n_low/len(y)*100:.2f}%)")
    print()

    # Value-count table for the most common discrete values
    top_values = y.value_counts().sort_index().head(20)
    print("  Most common target values (first 20 by value):")
    print(f"  {'Value':>8}  {'Count':>8}  {'%':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}")
    for val, cnt in top_values.items():
        print(f"  {int(val):>8}  {cnt:>8,}  {cnt/len(y)*100:>7.2f}%")
    print()


def show_outliers(train: pd.DataFrame) -> None:
    """Detect outliers using IQR method and a hard business threshold."""
    print("=" * 60)
    print("OUTLIER DETECTION  (next_3m_txn_count)")
    print("=" * 60)
    y = train[TARGET_COL].dropna()

    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - IQR_MULTIPLIER * iqr
    upper_fence = q3 + IQR_MULTIPLIER * iqr

    outliers_iqr = y[(y < lower_fence) | (y > upper_fence)]
    outliers_high = y[y > HIGH_TXN_THRESHOLD]

    print(f"  IQR method (×{IQR_MULTIPLIER})")
    print(f"    Q1            : {q1:.0f}")
    print(f"    Q3            : {q3:.0f}")
    print(f"    IQR           : {iqr:.0f}")
    print(f"    Lower fence   : {lower_fence:.0f}")
    print(f"    Upper fence   : {upper_fence:.0f}")
    print(f"    Outlier count : {len(outliers_iqr):,}  ({len(outliers_iqr)/len(y)*100:.2f}%)")
    print()
    print(f"  Hard threshold (>{HIGH_TXN_THRESHOLD} transactions)")
    print(f"    Outlier count : {len(outliers_high):,}  ({len(outliers_high)/len(y)*100:.2f}%)")
    if len(outliers_high) > 0:
        print(f"    Max value     : {outliers_high.max():,}")
    print()


def show_log_target_stats(train: pd.DataFrame) -> None:
    """Print stats on log1p(target) — directly relevant to RMSLE optimisation."""
    print("=" * 60)
    print("LOG1P TARGET STATS  (relevant for RMSLE)")
    print("=" * 60)
    y = train[TARGET_COL].dropna()
    log_y = np.log1p(y)

    print(f"  Mean    : {log_y.mean():.4f}")
    print(f"  Median  : {log_y.median():.4f}")
    print(f"  Std     : {log_y.std():.4f}")
    print(f"  Skew    : {log_y.skew():.4f}")
    print()
    percentiles = [5, 25, 50, 75, 95]
    print("  Percentile breakdown (log1p scale):")
    for p in percentiles:
        print(f"    P{p:<3}: {np.percentile(log_y, p):>10.4f}")
    print()
    print("  [INFO] Training a model on log1p(y) and exponentiating predictions")
    print("         will directly minimise RMSLE.")
    print()


def print_insights_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print a human-readable bullet summary of key findings."""
    y = train[TARGET_COL].dropna()
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + IQR_MULTIPLIER * iqr
    n_outliers = (y > upper_fence).sum()

    print("=" * 60)
    print("KEY INSIGHTS SUMMARY")
    print("=" * 60)
    insights = [
        f"Training set has {len(train):,} labelled customers; "
        f"test set has {len(test):,} customers to predict.",

        f"Target ranges from {int(y.min())} to {int(y.max())} transactions, "
        f"with a mean of {y.mean():.1f} and median of {int(y.median())}.",

        f"Target is right-skewed (skew={y.skew():.2f}); "
        f"log-transforming before modelling is recommended.",

        f"{(y == 0).sum():,} customers ({(y == 0).mean()*100:.1f}%) made "
        f"zero transactions in the target window — consider zero-heavy modelling.",

        f"{n_outliers:,} customers ({n_outliers/len(y)*100:.1f}%) are "
        f"IQR-based outliers (×{IQR_MULTIPLIER}); inspect before capping.",

        "Metric is RMSLE — large absolute errors on high-volume customers "
        "are penalised less on log scale; focus on relative accuracy.",

        "No model training here — EDA complete. Proceed to feature engineering.",
    ]
    for i, insight in enumerate(insights, start=1):
        # Wrap long lines at 56 chars after the number
        print(f"  {i}. {insight}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    train, test = load_splits()
    show_split_overview(train, test)
    show_id_overlap(train, test)
    show_target_missing(train)
    show_target_distribution(train)
    show_outliers(train)
    show_log_target_stats(train)
    print_insights_summary(train, test)
    print("[DONE] eda_summary.py complete.")


if __name__ == "__main__":
    main()

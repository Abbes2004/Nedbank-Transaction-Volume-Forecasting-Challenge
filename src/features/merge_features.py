"""
merge_features.py
==================
Merges all interim feature tables into final train / test feature sets,
then applies post-merge interaction engineering.

Inputs
------
data/interim/transactions_features.parquet
data/interim/financials_features.parquet
data/interim/demographics_features.parquet
data/raw/Train.csv
data/raw/Test.csv

Outputs
-------
data/processed/train_features.parquet
data/processed/test_features.parquet

Design rules
------------
- LEFT JOINs only on UniqueID (no customer ever dropped)
- No leakage: all features pre-computed on data ≤ 2015-10-31
- Consistent NaN fill: median for numerics, -1 for categoricals (_enc cols)
- Interaction features applied AFTER fill so transforms never see raw NaNs
- Final column set is identical for train and test
"""

import os

import numpy as np
import pandas as pd

from features.build_interaction_features import build_interaction_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERIM_TXN   = "data/interim/transactions_features.parquet"
INTERIM_FIN   = "data/interim/financials_features.parquet"
INTERIM_DEMO  = "data/interim/demographics_features.parquet"
TRAIN_LABELS  = "data/raw/Train.csv"
TEST_IDS      = "data/raw/Test.csv"
OUT_TRAIN     = "data/processed/train_features.parquet"
OUT_TEST      = "data/processed/test_features.parquet"

TARGET_COL    = "next_3m_txn_count"
ID_COL        = "UniqueID"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_parquet(path: str, label: str) -> pd.DataFrame:
    print(f"[INFO] Loading {label} from: {path}")
    df = pd.read_parquet(path)
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)
    print(f"[INFO]   → {df.shape[0]:,} rows, {df.shape[1]} features")
    return df


def load_all_features() -> pd.DataFrame:
    """
    Load and LEFT-JOIN all interim feature tables.
    Returns a combined DataFrame indexed by UniqueID.
    """
    txn  = _load_parquet(INTERIM_TXN,  "transactions")
    fin  = _load_parquet(INTERIM_FIN,  "financials")
    demo = _load_parquet(INTERIM_DEMO, "demographics")

    all_ids = txn.index.union(fin.index).union(demo.index)
    base = pd.DataFrame(index=all_ids)
    base.index.name = ID_COL

    merged = base.join(txn,  how="left")
    merged = merged.join(fin,  how="left")
    merged = merged.join(demo, how="left")

    print(f"[INFO] Merged shape (all customers): {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    - _enc suffix columns → fill -1
    - _bin suffix columns → fill -1
    - all other numeric   → fill column median
    """
    special_cols = [c for c in df.columns
                    if c.endswith("_enc") or c.endswith("_bin")]
    numeric_cols = [c for c in df.columns if c not in special_cols]

    df[special_cols] = df[special_cols].fillna(-1)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    remaining = df.isna().sum().sum()
    if remaining > 0:
        print(f"[WARN] {remaining} NaN values remain after fill — forcing to 0.")
        df = df.fillna(0)

    return df


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def check_schema_consistency(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Assert train and test have identical columns; re-order test if needed."""
    only_train = set(train.columns) - set(test.columns)
    only_test  = set(test.columns)  - set(train.columns)

    if only_train or only_test:
        raise ValueError(
            f"Column mismatch!\n"
            f"  Only in train : {sorted(only_train)}\n"
            f"  Only in test  : {sorted(only_test)}"
        )

    test = test[train.columns]
    print(f"[INFO] Schema OK — {len(train.columns)} feature columns.")
    return test


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def merge_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build train_features and test_features DataFrames.
    Both have identical columns; train has the target appended last.

    Returns
    -------
    train_df : pd.DataFrame  (features + target)
    test_df  : pd.DataFrame  (features only)
    """
    print("[INFO] Loading train labels and test IDs ...")
    train_labels = pd.read_csv(TRAIN_LABELS).set_index(ID_COL)
    test_ids     = pd.read_csv(TEST_IDS).set_index(ID_COL)

    # ── Step 1: raw join ─────────────────────────────────────────────────────
    all_features = load_all_features()

    # ── Step 2: first NaN fill (before transforms) ───────────────────────────
    print("[INFO] First NaN fill (pre-transform) ...")
    all_features = fill_missing_values(all_features)

    # ── Step 3: interaction / transform features ─────────────────────────────
    print("[INFO] Building interaction features ...")
    all_features = build_interaction_features(all_features)

    # ── Step 4: second NaN fill (new columns may have NaNs) ──────────────────
    print("[INFO] Second NaN fill (post-transform) ...")
    all_features = fill_missing_values(all_features)

    # ── Step 5: split ────────────────────────────────────────────────────────
    train_features = all_features.loc[
        all_features.index.isin(train_labels.index)
    ].copy()
    test_features  = all_features.loc[
        all_features.index.isin(test_ids.index)
    ].copy()

    # Attach target
    train_features[TARGET_COL] = train_labels[TARGET_COL]

    # Validate schema
    test_features = check_schema_consistency(
        train_features.drop(columns=[TARGET_COL]),
        test_features,
    )

    # Sanity: no ID overlap
    overlap = set(train_features.index) & set(test_features.index)
    assert len(overlap) == 0, f"ID overlap detected: {overlap}"

    print(f"[INFO] Train features : {train_features.shape}")
    print(f"[INFO] Test  features : {test_features.shape}")

    return train_features, test_features


# ---------------------------------------------------------------------------
# Savers
# ---------------------------------------------------------------------------

def save_datasets(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    out_train: str = OUT_TRAIN,
    out_test:  str = OUT_TEST,
) -> None:
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    train.reset_index().to_parquet(out_train, index=False)
    test.reset_index().to_parquet(out_test,   index=False)
    print(f"[INFO] Saved train → {out_train}")
    print(f"[INFO] Saved test  → {out_test}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    train, test = merge_features()
    save_datasets(train, test)
    print("[DONE] merge_features complete.")


if __name__ == "__main__":
    main()
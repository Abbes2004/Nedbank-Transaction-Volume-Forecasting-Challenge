"""
build_interaction_features.py
================================
Post-merge feature engineering applied to the unified feature matrix.
All sub-functions return dicts of new columns — single pd.concat per step
avoids DataFrame fragmentation warnings.
"""

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
CLIP_QUANTILE = 0.99
BIN_COUNT     = 10
EPS           = 1.0

_SKIP_CLIP = [
    "_enc", "_flag", "_bin", "_missing",
    "is_age_valid", "income_is_null",
    "has_mortgage", "has_investment", "has_transactional",
]

_LOG_POSITIVE = [
    "total_txn_count", "total_active_days",
    "txn_count_30d", "txn_count_90d", "txn_count_180d",
    "amount_sum_180d", "credit_sum",
    "nov_jan_txn_count_hist",
    "debit_count", "credit_count",
    "NetInterestIncome_sum", "NetInterestRevenue_sum",
    "txn_per_month", "txn_per_active_day", "avg_txn_per_month_90d",
    "annual_gross_income",
    "snapshot_count",
    "last3_count", "prior3_count",
]

_LOG_ABS = [
    "debit_sum", "net_flow",
    "NetInterestIncome_mean", "NetInterestRevenue_mean",
]

_SQRT_COLS = [
    "txn_count_90d", "txn_count_180d",
    "total_txn_count", "txn_per_month",
    "credit_count", "debit_count",
]

_BIN_COLS = [
    ("txn_count_90d",       "bin_txn_90d"),
    ("txn_per_month",       "bin_txn_per_month"),
    ("age",                 "bin_age"),
    ("total_txn_count",     "bin_total_txn"),
    ("annual_gross_income", "bin_income"),
    ("last3_count",         "bin_last3_count"),
]


# ---------------------------------------------------------------------------
# 1. Missingness flags
# ---------------------------------------------------------------------------

def _build_missingness_flags(df: pd.DataFrame) -> dict:
    candidates = [
        "NetInterestIncome_mean", "NetInterestRevenue_mean",
        "n_accounts", "txn_count_90d", "annual_gross_income", "age",
    ]
    new = {}
    for col in candidates:
        flag = f"{col}_missing"
        if col in df.columns and flag not in df.columns:
            new[flag] = df[col].isna().astype("int8")
    return new


# ---------------------------------------------------------------------------
# 2. Outlier clipping
# ---------------------------------------------------------------------------

def _clip_outliers(df: pd.DataFrame, q: float = CLIP_QUANTILE) -> pd.DataFrame:
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    clip_cols = [c for c in num_cols
                 if not any(p in c for p in _SKIP_CLIP)]
    lo_q = 1.0 - q
    df = df.copy()
    for col in clip_cols:
        lo = df[col].quantile(lo_q)
        hi = df[col].quantile(q)
        if lo < hi:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ---------------------------------------------------------------------------
# 3. Ratio / interaction features
# ---------------------------------------------------------------------------

def _build_ratio_features(df: pd.DataFrame) -> dict:
    eps = EPS
    new = {}

    for metric in ["txn_count", "active_days"]:
        c30  = f"{metric}_30d"
        c90  = f"{metric}_90d"
        c180 = f"{metric}_180d"
        if c30 in df.columns and c180 in df.columns:
            new[f"ratio_{metric}_30d_180d"] = df[c30] / (df[c180] + eps)
        if c90 in df.columns and c180 in df.columns:
            new[f"ratio_{metric}_90d_180d"] = df[c90] / (df[c180] + eps)

    if "amount_sum_30d" in df.columns and "amount_sum_180d" in df.columns:
        new["ratio_amount_30d_180d"] = (
            df["amount_sum_30d"].abs() / (df["amount_sum_180d"].abs() + eps)
        )
    if "txn_per_month" in df.columns and "debit_fraction" in df.columns:
        new["txn_x_debit_frac"] = df["txn_per_month"] * df["debit_fraction"]
    if "age" in df.columns and "txn_per_month" in df.columns:
        new["age_x_txn_per_month"] = df["age"] * df["txn_per_month"]
    if "debit_sum" in df.columns and "credit_sum" in df.columns:
        denom = df["credit_sum"].abs() + df["debit_sum"].abs() + eps
        new["net_flow_direction"] = (df["credit_sum"] + df["debit_sum"]) / denom
    if "last3_count" in df.columns and "prior3_count" in df.columns:
        peak = df[["last3_count", "prior3_count"]].max(axis=1) + eps
        new["count_accel"] = (df["last3_count"] - df["prior3_count"]) / peak
    if "amount_std_90d" in df.columns and "amount_mean_90d" in df.columns:
        new["amount_cv_90d"] = (
            df["amount_std_90d"] / (df["amount_mean_90d"].abs() + eps)
        )
    return new


# ---------------------------------------------------------------------------
# 4 & 5. Log transforms
# ---------------------------------------------------------------------------

def _build_log_transforms(df: pd.DataFrame) -> dict:
    new = {}
    for col in _LOG_POSITIVE:
        if col in df.columns:
            new[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))
    for col in _LOG_ABS:
        if col in df.columns:
            new[f"log1p_abs_{col}"] = np.log1p(df[col].abs())
    return new


# ---------------------------------------------------------------------------
# 6. Sqrt transforms
# ---------------------------------------------------------------------------

def _build_sqrt_transforms(df: pd.DataFrame) -> dict:
    new = {}
    for col in _SQRT_COLS:
        if col in df.columns:
            new[f"sqrt_{col}"] = np.sqrt(df[col].clip(lower=0))
    return new


# ---------------------------------------------------------------------------
# 7. Quantile bins
# ---------------------------------------------------------------------------

def _build_binned_features(df: pd.DataFrame, n: int = BIN_COUNT) -> dict:
    new = {}
    for src, dst in _BIN_COLS:
        if src in df.columns:
            try:
                new[dst] = (
                    pd.qcut(df[src], q=n, labels=False, duplicates="drop")
                    .fillna(-1)
                    .astype("int8")
                )
            except Exception:
                new[dst] = pd.Series(-1, index=df.index, dtype="int8")
    return new


# ---------------------------------------------------------------------------
# 8. Cross-domain features
# ---------------------------------------------------------------------------

def _build_cross_domain(df: pd.DataFrame) -> dict:
    eps = EPS
    new = {}
    if "total_txn_count" in df.columns and "n_accounts" in df.columns:
        new["txn_per_account"] = (
            df["total_txn_count"] / df["n_accounts"].clip(lower=1)
        )
    if "txn_per_month" in df.columns and "NetInterestIncome_mean" in df.columns:
        new["txn_per_month_per_nii"] = df["txn_per_month"] / (
            df["NetInterestIncome_mean"].abs() + eps
        )
    if "annual_gross_income" in df.columns and "txn_per_month" in df.columns:
        raw = df["annual_gross_income"].clip(lower=0) * df["txn_per_month"]
        new["log1p_income_x_txn"] = np.log1p(raw.clip(lower=0))
    if "n_accounts" in df.columns and "txn_per_month" in df.columns:
        new["n_accounts_x_txn"] = df["n_accounts"] * df["txn_per_month"]
    if "nov_jan_txn_frac" in df.columns and "txn_count_90d" in df.columns:
        new["seasonal_x_recent"] = df["nov_jan_txn_frac"] * df["txn_count_90d"]
    if "txn_count_growth" in df.columns and "nov_jan_txn_frac" in df.columns:
        new["trend_x_seasonal"] = df["txn_count_growth"] * df["nov_jan_txn_frac"]
    return new


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all post-merge feature engineering steps in order.
    Each step collects new columns in a dict and concatenates once
    to avoid DataFrame fragmentation warnings.
    """
    n_before = df.shape[1]

    print("[INFO] [1/8] Missingness flags ...")
    new_cols = _build_missingness_flags(df)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print("[INFO] [2/8] Clipping outliers ...")
    df = _clip_outliers(df)

    print("[INFO] [3/8] Ratio / interaction features ...")
    new_cols = _build_ratio_features(df)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print("[INFO] [4/8] Log1p transforms ...")
    new_cols = _build_log_transforms(df)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print("[INFO] [5/8] Sqrt transforms ...")
    new_cols = _build_sqrt_transforms(df)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print("[INFO] [6/8] Quantile binning ...")
    new_cols = _build_binned_features(df)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print("[INFO] [7/8] Cross-domain features ...")
    new_cols = _build_cross_domain(df)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    n_after = df.shape[1]
    print(f"[INFO] [8/8] Done — {n_before} → {n_after} columns "
          f"(+{n_after - n_before} engineered)")
    return df
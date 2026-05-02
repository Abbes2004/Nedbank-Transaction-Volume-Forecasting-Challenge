"""
build_transactions_features.py
================================
Builds customer-level features from the raw transaction history.

Cutoff date      : 2015-10-31  (last date present in transactions data)
Prediction window: Nov-2015 – Jan-2016  (3 months we are forecasting)

Feature groups
--------------
A. Time-window aggregations : last 30 / 90 / 180 days
B. Recency                  : days since last / first txn, activity span
C. Frequency                : txn/month, txn/active-day, 90-day velocity
D. Behavioural              : debit vs credit stats, net flow, volatility
E. Trend                    : last-3M vs prior-3M count / amount growth
F. Type-level               : per-type count + fraction, diversity score
G. Seasonality              : historical Nov-Jan activity fraction

Output
------
data/interim/transactions_features.parquet
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTOFF_DATE = pd.Timestamp("2015-10-31")
RAW_PATH    = "data/raw/transactions_features.parquet"
OUT_PATH    = "data/interim/transactions_features.parquet"

LOAD_COLS = [
    "UniqueID",
    "TransactionDate",
    "TransactionAmount",
    "TransactionTypeDescription",
]

WINDOWS_DAYS = [30, 90, 180]

TOP_TYPES = [
    "Transfers & Payments",
    "Charges & Fees",
    "Interest & Investments",
    "Debit Orders & Standing Orders",
    "Withdrawals",
    "Foreign Exchange",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _type_col_name(t: str) -> str:
    """Convert a type description to a safe column suffix."""
    return (
        t.lower()
        .replace(" & ", "_and_")
        .replace("& ", "_and_")
        .replace(" / ", "_")
        .replace("/ ", "_")
        .replace(" ", "_")
    )


# ---------------------------------------------------------------------------
# A. Time-window aggregations
# ---------------------------------------------------------------------------

def build_window_features(txn: pd.DataFrame) -> pd.DataFrame:
    """
    For each window w in WINDOWS_DAYS compute per-customer:
      txn_count, amount_sum, amount_mean, amount_std, amount_max, amount_min,
      active_days (unique calendar days with at least one transaction).
    """
    frames = []
    for w in WINDOWS_DAYS:
        start  = CUTOFF_DATE - pd.Timedelta(days=w)
        subset = txn[txn["TransactionDate"] > start]

        agg = subset.groupby("UniqueID")["TransactionAmount"].agg(
            **{
                f"txn_count_{w}d":   "count",
                f"amount_sum_{w}d":  "sum",
                f"amount_mean_{w}d": "mean",
                f"amount_std_{w}d":  "std",
                f"amount_max_{w}d":  "max",
                f"amount_min_{w}d":  "min",
            }
        )
        active_days = (
            subset.groupby("UniqueID")["TransactionDate"]
            .nunique()
            .rename(f"active_days_{w}d")
        )
        frames.append(pd.concat([agg, active_days], axis=1))

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# B. Recency features
# ---------------------------------------------------------------------------

def build_recency_features(txn: pd.DataFrame) -> pd.DataFrame:
    """Days since last / first transaction and total activity span in days."""
    grp = txn.groupby("UniqueID")["TransactionDate"].agg(["max", "min"])
    grp.columns = ["last_txn_date", "first_txn_date"]

    grp["days_since_last_txn"]  = (CUTOFF_DATE - grp["last_txn_date"]).dt.days
    grp["days_since_first_txn"] = (CUTOFF_DATE - grp["first_txn_date"]).dt.days
    grp["activity_span_days"]   = (grp["last_txn_date"] - grp["first_txn_date"]).dt.days

    return grp[["days_since_last_txn", "days_since_first_txn", "activity_span_days"]]


# ---------------------------------------------------------------------------
# C. Frequency features
# ---------------------------------------------------------------------------

def build_frequency_features(
    txn:     pd.DataFrame,
    recency: pd.DataFrame,
    window:  pd.DataFrame,
) -> pd.DataFrame:
    """
    total_txn_count       : all-time transaction count
    total_active_days     : unique days with at least one transaction (all-time)
    txn_per_month         : total count / active months (activity_span-based)
    txn_per_active_day    : total count / unique active days
    avg_txn_per_month_90d : txn_count_90d / 3  (most recent velocity)
    """
    total        = txn.groupby("UniqueID").size().rename("total_txn_count")
    unique_days  = (
        txn.groupby("UniqueID")["TransactionDate"]
        .nunique()
        .rename("total_active_days")
    )

    feat = pd.concat([total, unique_days, recency], axis=1)

    months_active             = (feat["activity_span_days"] / 30.0).clip(lower=1)
    feat["txn_per_month"]     = feat["total_txn_count"] / months_active
    feat["txn_per_active_day"] = (
        feat["total_txn_count"] / feat["total_active_days"].clip(lower=1)
    )
    feat["avg_txn_per_month_90d"] = window["txn_count_90d"] / 3.0

    return feat[[
        "total_txn_count",
        "total_active_days",
        "txn_per_month",
        "txn_per_active_day",
        "avg_txn_per_month_90d",
    ]]


# ---------------------------------------------------------------------------
# D. Behavioural features
# ---------------------------------------------------------------------------

def build_behavioral_features(txn: pd.DataFrame) -> pd.DataFrame:
    """
    debit_count / debit_sum    : negative-amount transactions
    credit_count / credit_sum  : positive-amount transactions
    debit_credit_ratio         : |debit_sum| / credit_sum
    debit_fraction             : debit_count / total
    zero_txn_fraction          : zero-amount rows / total
    net_flow                   : credit_sum + debit_sum  (signed)
    amount_volatility          : std / |mean|  (coefficient of variation)
    """
    debits  = txn[txn["TransactionAmount"] < 0].groupby("UniqueID")["TransactionAmount"].agg(
        debit_count="count",
        debit_sum="sum",
    )
    credits = txn[txn["TransactionAmount"] > 0].groupby("UniqueID")["TransactionAmount"].agg(
        credit_count="count",
        credit_sum="sum",
    )
    zeros = (
        (txn["TransactionAmount"] == 0)
        .groupby(txn["UniqueID"])
        .sum()
        .rename("zero_txn_count")
    )
    total_stats = txn.groupby("UniqueID")["TransactionAmount"].agg(
        all_count="count",
        all_mean="mean",
        all_std="std",
    )

    feat = pd.concat([debits, credits, zeros, total_stats], axis=1).fillna(0)

    feat["debit_credit_ratio"] = (
        feat["debit_sum"].abs() / feat["credit_sum"].clip(lower=1)
    )
    feat["debit_fraction"]   = feat["debit_count"]   / feat["all_count"].clip(lower=1)
    feat["zero_txn_fraction"] = feat["zero_txn_count"] / feat["all_count"].clip(lower=1)
    feat["net_flow"]          = feat["credit_sum"] + feat["debit_sum"]
    feat["amount_volatility"] = feat["all_std"] / feat["all_mean"].abs().clip(lower=1e-6)

    return feat[[
        "debit_count", "debit_sum",
        "credit_count", "credit_sum",
        "debit_credit_ratio", "debit_fraction",
        "zero_txn_fraction", "net_flow",
        "amount_volatility",
    ]]


# ---------------------------------------------------------------------------
# E. Trend features (last-3M vs prior-3M)
# ---------------------------------------------------------------------------

def build_trend_features(txn: pd.DataFrame) -> pd.DataFrame:
    """
    last3_count   : transactions in the most recent 90 days
    prior3_count  : transactions in the 90 days before that
    txn_count_growth   : (last3 - prior3) / prior3
    amount_sum_growth  : (last3_sum - prior3_sum) / |prior3_sum|
    """
    last3_start  = CUTOFF_DATE - pd.Timedelta(days=90)
    prior3_start = CUTOFF_DATE - pd.Timedelta(days=180)

    last3  = txn[txn["TransactionDate"] > last3_start].groupby("UniqueID").agg(
        last3_count=("TransactionAmount", "count"),
        last3_sum=("TransactionAmount",   "sum"),
    )
    prior3 = txn[
        (txn["TransactionDate"] >  prior3_start) &
        (txn["TransactionDate"] <= last3_start)
    ].groupby("UniqueID").agg(
        prior3_count=("TransactionAmount", "count"),
        prior3_sum=("TransactionAmount",   "sum"),
    )

    trend = pd.concat([last3, prior3], axis=1).fillna(0)

    trend["txn_count_growth"] = (
        (trend["last3_count"] - trend["prior3_count"]) /
        trend["prior3_count"].clip(lower=1)
    )
    trend["amount_sum_growth"] = (
        (trend["last3_sum"] - trend["prior3_sum"]) /
        trend["prior3_sum"].abs().clip(lower=1e-6)
    )

    return trend[["last3_count", "prior3_count", "txn_count_growth", "amount_sum_growth"]]


# ---------------------------------------------------------------------------
# F. Transaction-type features
# ---------------------------------------------------------------------------

def build_type_features(txn: pd.DataFrame) -> pd.DataFrame:
    """
    For each top transaction type:
      type_<name>_count : raw count
      type_<name>_frac  : fraction of total transactions
    Plus txn_type_diversity : number of unique types used by customer.
    """
    total     = txn.groupby("UniqueID").size().rename("_total")
    diversity = (
        txn.groupby("UniqueID")["TransactionTypeDescription"]
        .nunique()
        .rename("txn_type_diversity")
    )

    frames = [diversity]
    for t in TOP_TYPES:
        col   = _type_col_name(t)
        count = (
            txn[txn["TransactionTypeDescription"] == t]
            .groupby("UniqueID")
            .size()
            .rename(f"type_{col}_count")
        )
        frames.append(count)

    feat = pd.concat(frames, axis=1).fillna(0)

    # Add fractions
    total_aligned = total.reindex(feat.index).clip(lower=1)
    for t in TOP_TYPES:
        col = _type_col_name(t)
        feat[f"type_{col}_frac"] = feat[f"type_{col}_count"] / total_aligned

    return feat


# ---------------------------------------------------------------------------
# G. Holiday seasonality proxy
# ---------------------------------------------------------------------------

def build_seasonality_features(txn: pd.DataFrame) -> pd.DataFrame:
    """
    nov_jan_txn_count_hist : historical transactions in Nov-Jan months
    nov_jan_txn_frac       : fraction of all transactions in Nov-Jan
    These capture whether a customer is seasonally active in the prediction window.
    """
    nov_jan       = txn[txn["TransactionDate"].dt.month.isin([11, 12, 1])]
    nov_jan_count = nov_jan.groupby("UniqueID").size().rename("nov_jan_txn_count_hist")
    total         = txn.groupby("UniqueID").size().rename("_total_s")

    feat = pd.concat([nov_jan_count, total], axis=1).fillna(0)
    feat["nov_jan_txn_frac"] = feat["nov_jan_txn_count_hist"] / feat["_total_s"].clip(lower=1)

    return feat[["nov_jan_txn_count_hist", "nov_jan_txn_frac"]]


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_transactions_features(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.
    Returns a DataFrame with UniqueID as index.
    """
    txn = _load_transactions(path)

    print("[INFO] A. Time-window features ...")
    win = build_window_features(txn)

    print("[INFO] B. Recency features ...")
    rec = build_recency_features(txn)

    print("[INFO] C. Frequency features ...")
    frq = build_frequency_features(txn, rec, win)

    print("[INFO] D. Behavioural features ...")
    beh = build_behavioral_features(txn)

    print("[INFO] E. Trend features ...")
    trn = build_trend_features(txn)

    print("[INFO] F. Type-level features ...")
    typ = build_type_features(txn)

    print("[INFO] G. Seasonality features ...")
    sea = build_seasonality_features(txn)

    feats = pd.concat([win, rec, frq, beh, trn, typ, sea], axis=1)
    feats = feats.fillna(0)
    feats.index.name = "UniqueID"

    print(f"[INFO] Final shape: {feats.shape}  ({feats.shape[1]} features, {len(feats):,} customers)")
    return feats


def _load_transactions(path: str) -> pd.DataFrame:
    """Load transactions parquet with column selection and cutoff filter."""
    print(f"[INFO] Loading: {path}")
    txn = pd.read_parquet(path, columns=LOAD_COLS)
    txn["TransactionDate"] = pd.to_datetime(txn["TransactionDate"])
    # Hard cutoff — no future leakage
    txn = txn[txn["TransactionDate"] <= CUTOFF_DATE].copy()
    print(f"[INFO] {len(txn):,} rows after cutoff filter.")
    return txn


def save_features(feats: pd.DataFrame, out_path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feats.reset_index().to_parquet(out_path, index=False)
    print(f"[INFO] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    feats = build_transactions_features()
    save_features(feats)
    print(f"[DONE] build_transactions_features complete.")


if __name__ == "__main__":
    main()

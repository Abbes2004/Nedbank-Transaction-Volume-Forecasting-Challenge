"""
build_demographics_features.py
================================
Builds customer-level features from the demographics snapshot.
One row per customer.

Source  : data/raw/demographics_clean.parquet
Output  : data/interim/demographics_features.parquet

Features
--------
- age (from BirthDate, validated)
- age_group buckets (ordinal integers)
- is_age_valid flag
- Gender, IncomeCategory, OccupationCategory, etc. → integer-encoded
- AnnualGrossIncome with log transformation
- LowIncomeFlag as binary
- CustomerStatus, ClientType, MaritalStatus → encoded
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_DATE = pd.Timestamp("2015-10-31")
RAW_PATH       = "data/raw/demographics_clean.parquet"
OUT_PATH       = "data/interim/demographics_features.parquet"

# Age validation boundaries
AGE_MIN = 16
AGE_MAX = 110

# Ordered age buckets  → integer codes 0–7
AGE_BINS   = [0, 18, 25, 35, 45, 55, 65, 75, 200]
AGE_LABELS = [0,  1,  2,  3,  4,  5,  6,  7]

# Categorical columns to label-encode (unknown / NaN → -1)
CATEGORICAL_COLS = [
    "Gender",
    "IncomeCategory",
    "CustomerStatus",
    "ClientType",
    "MaritalStatus",
    "OccupationCategory",
    "IndustryCategory",
    "CustomerBankingType",
    "CustomerOnboardingChannel",
    "ResidentialCityName",
    "CountryCodeNationality",
    "LowIncomeFlag",
    "CertificationTypeDescription",
    "ContactPreference",
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_demographics(path: str = RAW_PATH) -> pd.DataFrame:
    print(f"[INFO] Loading demographics from: {path}")
    demo = pd.read_parquet(path)
    print(f"[INFO] {len(demo):,} rows loaded.")
    return demo


# ---------------------------------------------------------------------------
# Age features
# ---------------------------------------------------------------------------

def build_age_features(demo: pd.DataFrame) -> pd.DataFrame:
    """
    age          : float, years from BirthDate to REFERENCE_DATE
    is_age_valid : 1 if 16 ≤ age ≤ 110, else 0
    age_group    : integer bucket (0–7), uses AGE_BINS / AGE_LABELS
    """
    feat = pd.DataFrame(index=demo["UniqueID"])

    # Keep as pandas Series throughout to avoid numpy array attribute errors
    birth = pd.to_datetime(demo["BirthDate"], errors="coerce")
    age   = (REFERENCE_DATE - birth).dt.days / 365.25   # pandas Series

    is_valid = ((age >= AGE_MIN) & (age <= AGE_MAX))

    feat["age"]          = age.where(is_valid, other=np.nan).values
    feat["is_age_valid"] = is_valid.astype(int).values

    # Replace out-of-range ages with NaN before bucketing
    age_clean = age.where(is_valid)
    feat["age_group"] = pd.cut(
        age_clean,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    ).astype("Int64").fillna(-1).astype(int).values

    return feat


# ---------------------------------------------------------------------------
# Income features
# ---------------------------------------------------------------------------

def build_income_features(demo: pd.DataFrame) -> pd.DataFrame:
    """
    annual_gross_income     : raw value (NaN for ~6 % of customers)
    log1p_annual_income     : log1p transform (0 for nulls)
    income_is_null          : 1 if AnnualGrossIncome was missing
    """
    feat = pd.DataFrame(index=demo["UniqueID"])

    inc = demo["AnnualGrossIncome"].values.astype(float)
    feat["annual_gross_income"] = inc

    income_null = pd.isna(inc)
    feat["income_is_null"]     = income_null.astype(int)

    inc_safe = np.where(income_null, 0.0, np.clip(inc, 0, None))
    feat["log1p_annual_income"] = np.log1p(inc_safe)

    return feat


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def build_categorical_features(demo: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode each categorical column.
    Unknown / NaN values are assigned code -1.
    Produces integer columns with suffix _enc.
    """
    feat = pd.DataFrame(index=demo["UniqueID"])

    for col in CATEGORICAL_COLS:
        if col not in demo.columns:
            print(f"[WARN] Column '{col}' not found — skipping.")
            continue

        series = demo[col].astype(str).fillna("__UNKNOWN__")
        # Stable encoding: alphabetical sort → integer map
        categories = sorted(series.unique())
        cat_map    = {c: i for i, c in enumerate(categories)}
        # Map NaN-string and missing to -1
        cat_map["nan"]         = -1
        cat_map["__UNKNOWN__"] = -1
        cat_map["None"]        = -1

        feat[f"{col}_enc"] = series.map(cat_map).fillna(-1).astype(int).values

    return feat


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_demographics_features(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Full pipeline. Returns DataFrame indexed by UniqueID.
    """
    demo = _load_demographics(path)
    demo = demo.set_index("UniqueID") if "UniqueID" in demo.columns else demo

    # Reset index to use UniqueID as a column for sub-builders
    demo_reset = demo.reset_index()

    print("[INFO] Building age features ...")
    age_feat = build_age_features(demo_reset)

    print("[INFO] Building income features ...")
    inc_feat = build_income_features(demo_reset)

    print("[INFO] Building categorical features ...")
    cat_feat = build_categorical_features(demo_reset)

    feats = pd.concat([age_feat, inc_feat, cat_feat], axis=1)
    feats.index.name = "UniqueID"

    # Fill remaining NaN (e.g. age for invalid rows)
    feats["age"] = feats["age"].fillna(feats["age"].median())

    print(f"[INFO] Demographics features shape: {feats.shape}")
    return feats


def save_features(feats: pd.DataFrame, out_path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feats.reset_index().to_parquet(out_path, index=False)
    print(f"[INFO] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    feats = build_demographics_features()
    save_features(feats)
    print("[DONE] build_demographics_features complete.")


if __name__ == "__main__":
    main()
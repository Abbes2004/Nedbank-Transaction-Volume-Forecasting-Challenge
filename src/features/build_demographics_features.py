"""
build_demographics_features.py
================================
Builds customer-level features from the demographics parquet file.
One row per UniqueID (file is already at customer grain).

Features produced:
  - age                  : derived from BirthDate (with quality flag)
  - age_band             : ordinal bucket (0-7)
  - is_valid_age         : flag for records with plausible age (16–110)
  - gender_encoded       : M=1, F=0, unknown=-1
  - annual_gross_income  : raw numeric (NaN → median imputation)
  - income_missing_flag  : 1 if AnnualGrossIncome was null
  - income_category_enc  : ordinal encoding of IncomeCategory (ordered by wealth)
  - is_low_income        : binary from LowIncomeFlag
  - Categorical label encodings (frequency-based or ordinal):
      customer_status_enc, client_type_enc, marital_status_enc,
      occupation_category_enc, industry_category_enc,
      banking_type_enc, onboarding_channel_enc,
      certification_type_enc, contact_preference_enc
  - is_south_african     : CountryCodeNationality == "ZA"
  - top_city_flag        : binary — customer is in one of the top-10 cities

No target leakage: all information is customer-static or pre-dates the
prediction window.

Usage:
    python build_demographics_features.py
    python build_demographics_features.py --input path/to/demographics.parquet \\
                                           --output path/to/out.parquet
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
REFERENCE_DATE = pd.Timestamp("2015-10-31")
INPUT_PATH = "data/raw/demographics_clean.parquet"
OUTPUT_PATH = "data/processed/demographics_features_agg.parquet"

# Ordered income categories (ascending wealth)
INCOME_ORDER = [
    "No Income",
    "Low Income",
    "Lower-Middle Income",
    "Middle Income",
    "Upper-Middle Income",
    "High Income",
    "Very High Income",
    "Not Disclosed / Unknown",
]

# ── loaders ──────────────────────────────────────────────────────────────────

def load_demographics(path: str) -> pd.DataFrame:
    """Load demographics parquet — already one row per customer."""
    print(f"[INFO] Loading demographics from: {path}")
    df = pd.read_parquet(path)
    df["BirthDate"] = pd.to_datetime(df["BirthDate"], errors="coerce")
    print(f"[INFO] Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


# ── feature builders ─────────────────────────────────────────────────────────

def build_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive age from BirthDate relative to REFERENCE_DATE.
    Flag records with implausible ages (< 16 or > 110) as invalid.
    """
    result = df[["UniqueID"]].copy()
    result["age"] = (REFERENCE_DATE - df["BirthDate"]).dt.days / 365.25

    valid_mask = result["age"].between(16, 110)
    result["is_valid_age"] = valid_mask.astype(int)
    result.loc[~valid_mask, "age"] = np.nan

    # Impute invalid/missing ages with the median of valid ages
    median_age = result.loc[valid_mask, "age"].median()
    result["age"] = result["age"].fillna(median_age)

    # Ordinal age band  0=<18, 1=18-24, 2=25-34, 3=35-44, 4=45-54,
    #                   5=55-64, 6=65-74, 7=75+
    bins = [0, 18, 25, 35, 45, 55, 65, 75, 200]
    labels = list(range(8))
    result["age_band"] = pd.cut(
        result["age"], bins=bins, labels=labels, right=False
    ).astype(float)

    return result


def build_gender_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender: M=1, F=0, missing/other=-1."""
    result = df[["UniqueID"]].copy()
    gender_map = {"M": 1, "F": 0}
    result["gender_encoded"] = df["Gender"].map(gender_map).fillna(-1).astype(int)
    return result


def build_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric income (with missingness flag + median imputation) and
    ordinal encoding of IncomeCategory.
    """
    result = df[["UniqueID"]].copy()

    # Raw numeric income
    result["income_missing_flag"] = df["AnnualGrossIncome"].isna().astype(int)
    median_income = df["AnnualGrossIncome"].median()
    result["annual_gross_income"] = df["AnnualGrossIncome"].fillna(median_income)

    # Ordinal income category
    income_order_map = {cat: i for i, cat in enumerate(INCOME_ORDER)}
    result["income_category_enc"] = (
        df["IncomeCategory"]
        .map(income_order_map)
        .fillna(len(INCOME_ORDER))   # unknown → highest index
        .astype(int)
    )

    # Low income binary flag
    low_income_map = {"Y": 1, "N": 0}
    result["is_low_income"] = df["LowIncomeFlag"].map(low_income_map).fillna(-1).astype(int)

    return result


def _frequency_encode(series: pd.Series, col_name: str, top_n: int = 20) -> pd.Series:
    """
    Encode a categorical column by the rank of its frequency (most frequent = 0).
    Categories outside top_n → rank top_n (treated as 'other').
    Missing → -1.
    """
    freq = series.value_counts()
    top_cats = freq.index[:top_n].tolist()
    rank_map = {cat: rank for rank, cat in enumerate(top_cats)}
    encoded = series.map(rank_map)
    # Any category not in top_n gets top_n rank
    encoded = encoded.fillna(top_n)
    # Restore NaN for originally missing values
    encoded = encoded.where(series.notna(), other=-1)
    return encoded.rename(col_name).astype(int)


def build_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency-rank encode all remaining categorical columns."""
    result = df[["UniqueID"]].copy()

    categorical_cols = {
        "CustomerStatus": "customer_status_enc",
        "ClientType": "client_type_enc",
        "MaritalStatus": "marital_status_enc",
        "OccupationCategory": "occupation_category_enc",
        "IndustryCategory": "industry_category_enc",
        "CustomerBankingType": "banking_type_enc",
        "CustomerOnboardingChannel": "onboarding_channel_enc",
        "CertificationTypeDescription": "certification_type_enc",
        "ContactPreference": "contact_preference_enc",
    }

    for raw_col, enc_col in categorical_cols.items():
        result[enc_col] = _frequency_encode(df[raw_col], col_name=enc_col).values

    return result


def build_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_south_african: 1 if CountryCodeNationality == "ZA".
    top_city_flag: 1 if customer lives in one of the 10 most common cities.
    """
    result = df[["UniqueID"]].copy()

    result["is_south_african"] = (
        df["CountryCodeNationality"].str.strip().str.upper() == "ZA"
    ).astype(int)

    top_cities = (
        df["ResidentialCityName"]
        .value_counts()
        .head(10)
        .index.tolist()
    )
    result["top_city_flag"] = df["ResidentialCityName"].isin(top_cities).astype(int)

    return result


# ── orchestrator ─────────────────────────────────────────────────────────────

def build_demographics_features(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load raw demographics and produce one feature row per customer.
    Writes result to output_path and also returns the DataFrame.
    """
    df = load_demographics(input_path)

    print("[INFO] Building age features ...")
    age = build_age_features(df)

    print("[INFO] Building gender features ...")
    gender = build_gender_feature(df)

    print("[INFO] Building income features ...")
    income = build_income_features(df)

    print("[INFO] Building categorical features ...")
    categorical = build_categorical_features(df)

    print("[INFO] Building geo features ...")
    geo = build_geo_features(df)

    # ── merge ──────────────────────────────────────────────────────────────
    print("[INFO] Merging all demographic feature tables ...")
    feature_tables = [age, gender, income, categorical, geo]
    features = feature_tables[0]
    for tbl in feature_tables[1:]:
        features = features.merge(tbl, on="UniqueID", how="left")

    print(f"[INFO] Final shape: {features.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_parquet(output_path, index=False)
    print(f"[INFO] Saved to: {output_path}")
    return features


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build demographic customer features.")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to demographics parquet file.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path for output parquet file.")
    args = parser.parse_args()

    features = build_demographics_features(args.input, args.output)

    print("\n[SUMMARY] Feature columns:")
    for col in features.columns:
        print(f"  {col}")
    print(f"\n[DONE] {features.shape[0]:,} customers, {features.shape[1]} features.")


if __name__ == "__main__":
    main()

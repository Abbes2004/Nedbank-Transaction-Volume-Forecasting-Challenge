"""
eda_demographics.py
-------------------
Exploratory Data Analysis for demographics_clean.parquet.

Responsibility: Analyse customer profiles — schema, missing values,
gender, income category, age (derived from BirthDate), occupation,
and industry distributions.

Run:
    python eda_demographics.py
"""

import pandas as pd
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = "data/raw/demographics_clean.parquet"

# Reference date for age calculation (one month before prediction window starts)
REFERENCE_DATE = pd.Timestamp("2015-10-31")

# Maximum reasonable age — used to flag BirthDate data quality issues
MAX_VALID_AGE = 110
MIN_VALID_AGE = 16


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_demographics(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the demographics parquet (one row per customer)."""
    print(f"[INFO] Loading demographics from: {path}")
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
    """Print missing-value counts and percentages for every column."""
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report.sort_values("missing_count", ascending=False)
    has_missing = report[report["missing_count"] > 0]

    if has_missing.empty:
        print("  No missing values found.\n")
    else:
        print(f"  {'Column':<40} {'Missing':>10}  {'%':>8}")
        print(f"  {'-'*40} {'-'*10}  {'-'*8}")
        for col, row in has_missing.iterrows():
            print(f"  {col:<40} {row['missing_count']:>10,}  {row['missing_pct']:>7.2f}%")
    print()


def show_gender_distribution(df: pd.DataFrame) -> None:
    """Print gender value counts."""
    print("=" * 60)
    print("GENDER DISTRIBUTION")
    print("=" * 60)

    col = _find_column(df, ["Gender", "gender", "Sex", "sex"])
    if col is None:
        print("  Gender column not found.\n")
        return

    counts = df[col].value_counts(dropna=False)
    total = len(df)
    print(f"  Column used: {col}\n")
    print(f"  {'Value':<25} {'Count':>10}  {'%':>8}")
    print(f"  {'-'*25} {'-'*10}  {'-'*8}")
    for val, cnt in counts.items():
        print(f"  {str(val):<25} {cnt:>10,}  {cnt/total*100:>7.2f}%")
    print()


def show_income_category_distribution(df: pd.DataFrame) -> None:
    """Print IncomeCategory (or similar) value counts."""
    print("=" * 60)
    print("INCOME CATEGORY DISTRIBUTION")
    print("=" * 60)

    col = _find_column(df, ["IncomeCategory", "IncomeBand", "IncomeGroup",
                             "income_category", "AnnualGrossIncome"])
    if col is None:
        print("  Income category column not found.\n")
        return

    # For continuous income columns, show stats instead of counts
    if pd.api.types.is_numeric_dtype(df[col]):
        series = df[col].dropna()
        n_null = df[col].isnull().sum()
        print(f"  Column used  : {col}  (numeric)\n")
        print(f"  Non-null     : {len(series):,}  |  Null: {n_null:,} ({n_null/len(df)*100:.2f}%)")
        print(f"  Mean         : {series.mean():,.2f}")
        print(f"  Median       : {series.median():,.2f}")
        print(f"  Std          : {series.std():,.2f}")
        print(f"  Min          : {series.min():,.2f}")
        print(f"  Max          : {series.max():,.2f}")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print("\n  Percentiles:")
        for p in percentiles:
            print(f"    P{p:<3}: {np.percentile(series, p):>18,.2f}")
    else:
        counts = df[col].value_counts(dropna=False)
        total = len(df)
        print(f"  Column used: {col}\n")
        print(f"  {'Category':<35} {'Count':>10}  {'%':>8}")
        print(f"  {'-'*35} {'-'*10}  {'-'*8}")
        for val, cnt in counts.items():
            print(f"  {str(val):<35} {cnt:>10,}  {cnt/total*100:>7.2f}%")
    print()


def show_age_distribution(df: pd.DataFrame) -> None:
    """Derive age from BirthDate and print distribution statistics."""
    print("=" * 60)
    print("AGE DISTRIBUTION  (derived from BirthDate)")
    print("=" * 60)

    col = _find_column(df, ["BirthDate", "DateOfBirth", "DOB", "birthdate"])
    if col is None:
        print("  BirthDate column not found.\n")
        return

    birth = pd.to_datetime(df[col], errors="coerce")
    age = ((REFERENCE_DATE - birth).dt.days / 365.25)

    n_null = birth.isnull().sum()
    print(f"  Reference date : {REFERENCE_DATE.date()}")
    print(f"  Null BirthDate : {n_null:,}  ({n_null/len(df)*100:.2f}%)\n")

    # Flag data quality issues
    invalid = age[(age < MIN_VALID_AGE) | (age > MAX_VALID_AGE)]
    print(f"  Age < {MIN_VALID_AGE} or > {MAX_VALID_AGE} (data quality flag): {len(invalid):,} rows")

    valid_age = age[(age >= MIN_VALID_AGE) & (age <= MAX_VALID_AGE)]
    print(f"  Valid age records : {len(valid_age):,}\n")

    print(f"  Mean   : {valid_age.mean():.1f}")
    print(f"  Median : {valid_age.median():.0f}")
    print(f"  Std    : {valid_age.std():.1f}")
    print(f"  Min    : {valid_age.min():.1f}")
    print(f"  Max    : {valid_age.max():.1f}")
    print()

    # Age-band breakdown
    bins = [0, 18, 25, 35, 45, 55, 65, 75, 200]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    age_band = pd.cut(valid_age, bins=bins, labels=labels, right=False)
    band_counts = age_band.value_counts().sort_index()
    total_valid = len(valid_age)
    print("  Age band distribution:")
    for band, cnt in band_counts.items():
        print(f"    {str(band):<10}: {cnt:>6,}  ({cnt/total_valid*100:.2f}%)")
    print()


def show_top_categories(df: pd.DataFrame, col_name: str, top_n: int = 15) -> None:
    """Generic helper — print top_n value counts for a categorical column."""
    col = _find_column(df, [col_name])
    if col is None:
        # Try a looser match
        matches = [c for c in df.columns if col_name.lower() in c.lower()]
        if matches:
            col = matches[0]
        else:
            print(f"  Column '{col_name}' not found in dataset.\n")
            return

    print("=" * 60)
    print(f"TOP {top_n}: {col.upper()}")
    print("=" * 60)
    counts = df[col].value_counts(dropna=False).head(top_n)
    total = len(df)
    print(f"  Column used: {col}\n")
    print(f"  {'Value':<40} {'Count':>10}  {'%':>8}")
    print(f"  {'-'*40} {'-'*10}  {'-'*8}")
    for val, cnt in counts.items():
        print(f"  {str(val):<40} {cnt:>10,}  {cnt/total*100:>7.2f}%")
    print()


# ── Utilities ────────────────────────────────────────────────────────────────
def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    df = load_demographics()
    show_schema(df)
    show_missing_values(df)
    show_gender_distribution(df)
    show_income_category_distribution(df)
    show_age_distribution(df)
    show_top_categories(df, col_name="Occupation")
    show_top_categories(df, col_name="Industry")
    print("[DONE] eda_demographics.py complete.")


if __name__ == "__main__":
    main()

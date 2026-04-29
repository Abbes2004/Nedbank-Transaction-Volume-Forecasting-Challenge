"""
src/modeling/predict.py
------------------------
Generate test-set predictions by averaging across all K-fold models.

Pipeline
--------
1. Load test_features.parquet
2. Load each fold model from models/
3. Predict log1p(y) for each fold → average ensemble
4. Invert: expm1(avg_log_pred), clip to non-negative
5. Round to nearest integer (target is a count)
6. Write submissions/<timestamp>_submission.csv

Usage (standalone)
------------------
    python src/modeling/predict.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from src.utils.config import (           # noqa: E402
    TEST_FEATURES,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    ID_COL,
    TARGET_COL,
    NON_FEATURE_COLS,
    CV_N_FOLDS,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_test(path: Path = TEST_FEATURES) -> pd.DataFrame:
    """Load processed test parquet. Raises if file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Test features not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded test data: %d rows × %d cols", len(df), df.shape[1])
    return df


def extract_test_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split test dataframe into feature matrix X and ID series.

    Returns
    -------
    X   : Feature matrix.
    ids : UniqueID series.
    """
    # Remove target-related columns if they accidentally leaked into test
    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns and c != ID_COL]
    ids = df[ID_COL].copy()
    X   = df.drop(columns=[ID_COL] + drop_cols, errors="ignore").copy()

    # Align dtypes: object/string → category
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("category")

    logger.info("Test feature matrix: %d features", X.shape[1])
    return X, ids


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_fold_models(
    models_dir: Path = MODELS_DIR,
    n_folds: int = CV_N_FOLDS,
) -> list[lgb.Booster]:
    """
    Load all fold Booster objects saved by train.py.

    Parameters
    ----------
    models_dir : Directory containing lgbm_fold{k}.txt files.
    n_folds    : Expected number of fold models.

    Returns
    -------
    list of lgb.Booster
    """
    boosters: list[lgb.Booster] = []

    for k in range(1, n_folds + 1):
        model_path = models_dir / f"lgbm_fold{k}.txt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run src/modeling/train.py first."
            )
        booster = lgb.Booster(model_file=str(model_path))
        boosters.append(booster)
        logger.info("Loaded model: %s", model_path.name)

    logger.info("Total fold models loaded: %d", len(boosters))
    return boosters


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_ensemble(
    X: pd.DataFrame,
    boosters: list[lgb.Booster],
) -> np.ndarray:
    """
    Average log1p predictions across all fold boosters.

    Parameters
    ----------
    X        : Test feature matrix.
    boosters : Loaded fold Booster objects.

    Returns
    -------
    np.ndarray of shape (n_samples,) — averaged log1p predictions.
    """
    fold_preds = np.zeros((len(X), len(boosters)), dtype=np.float64)

    for i, booster in enumerate(boosters):
        fold_preds[:, i] = booster.predict(X)
        logger.info("Fold %d predictions generated.", i + 1)

    avg_log_preds = fold_preds.mean(axis=1)
    logger.info(
        "Ensemble log1p pred — mean: %.4f | std: %.4f | min: %.4f | max: %.4f",
        avg_log_preds.mean(),
        avg_log_preds.std(),
        avg_log_preds.min(),
        avg_log_preds.max(),
    )
    return avg_log_preds


def postprocess_predictions(log_preds: np.ndarray) -> np.ndarray:
    """
    Convert log1p predictions → integer counts.

    Steps
    -----
    1. expm1  — invert log1p transform.
    2. clip   — enforce non-negativity (RMSLE is undefined for negatives).
    3. round  — target is an integer count.

    Parameters
    ----------
    log_preds : Raw ensemble log1p predictions.

    Returns
    -------
    np.ndarray of non-negative integers.
    """
    raw      = np.expm1(log_preds)
    clipped  = np.clip(raw, a_min=0.0, a_max=None)
    rounded  = np.round(clipped).astype(np.int64)

    n_clipped = int(np.sum(raw < 0))
    if n_clipped > 0:
        logger.warning("%d predictions were clipped from negative to 0.", n_clipped)

    logger.info(
        "Final pred stats — mean: %.1f | median: %.1f | min: %d | max: %d",
        rounded.mean(),
        np.median(rounded),
        rounded.min(),
        rounded.max(),
    )
    return rounded


# ─────────────────────────────────────────────────────────────────────────────
# Submission writer
# ─────────────────────────────────────────────────────────────────────────────

def build_submission(
    ids: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Construct submission DataFrame in the required format.

    Parameters
    ----------
    ids         : UniqueID series.
    predictions : Processed integer count predictions.

    Returns
    -------
    pd.DataFrame with columns [ID_COL, TARGET_COL].
    """
    return pd.DataFrame({
        ID_COL:     ids.values,
        TARGET_COL: predictions,
    })


def save_submission(
    sub: pd.DataFrame,
    submissions_dir: Path = SUBMISSIONS_DIR,
    filename: str | None = None,
) -> Path:
    """
    Save submission CSV with a timestamped filename.

    Parameters
    ----------
    sub             : Submission DataFrame.
    submissions_dir : Output directory.
    filename        : Override filename (optional).

    Returns
    -------
    Path of the saved CSV.
    """
    submissions_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_submission.csv"

    out_path = submissions_dir / filename
    sub.to_csv(out_path, index=False)
    logger.info("Submission saved → %s  (%d rows)", out_path, len(sub))
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("═══ Prediction pipeline started ═══")

    # Load test data
    test_df         = load_test()
    X_test, ids     = extract_test_features(test_df)

    # Load models
    boosters        = load_fold_models()

    # Predict
    log_preds       = predict_ensemble(X_test, boosters)
    final_preds     = postprocess_predictions(log_preds)

    # Build and save submission
    submission      = build_submission(ids, final_preds)
    save_submission(submission)

    logger.info("═══ Prediction pipeline complete ═══")


if __name__ == "__main__":
    main()

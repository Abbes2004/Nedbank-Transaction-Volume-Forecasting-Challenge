"""
src/utils/config.py
-------------------
Central configuration: paths, model hyperparameters, training constants.
"""

from pathlib import Path

# ── Project root (two levels up from this file: src/utils/ → src/ → project/) ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data paths ──────────────────────────────────────────────────────────────────
DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM    = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"

TRAIN_FEATURES  = DATA_PROCESSED / "train_features.parquet"
TEST_FEATURES   = DATA_PROCESSED / "test_features.parquet"

# ── Output paths ────────────────────────────────────────────────────────────────
MODELS_DIR      = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# ── Column constants ────────────────────────────────────────────────────────────
ID_COL          = "UniqueID"
TARGET_COL      = "next_3m_txn_count"
LOG_TARGET_COL  = "log1p_target"

# Columns to always exclude from feature matrix
NON_FEATURE_COLS = {ID_COL, TARGET_COL, LOG_TARGET_COL}

# ── Cross-validation ─────────────────────────────────────────────────────────────
CV_N_FOLDS      = 5
CV_RANDOM_STATE = 42

# ── LightGBM hyperparameters ─────────────────────────────────────────────────────
LGBM_PARAMS: dict = {
    "objective":        "regression",
    "metric":           "rmse",           # rmse on log1p(y) ≡ rmsle on y
    "boosting_type":    "gbdt",
    "n_estimators":     3000,
    "learning_rate":    0.03,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 20,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_jobs":           -1,
    "random_state":     CV_RANDOM_STATE,
    "verbose":          -1,
}

LGBM_FIT_PARAMS: dict = {
    "early_stopping_rounds": 100,
    "log_period":            50,
    "verbose":               False,
}

# ── Submission ───────────────────────────────────────────────────────────────────
SUBMISSION_FILENAME = "submission.csv"
OOF_FILENAME        = "oof_predictions.parquet"

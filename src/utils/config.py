"""
src/utils/config.py
-------------------
Central configuration: paths, model hyperparameters, training constants.
"""

from pathlib import Path

# ── Directory layout ─────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parents[2]

DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM    = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"

TRAIN_FEATURES  = DATA_PROCESSED / "train_features.parquet"
TEST_FEATURES   = DATA_PROCESSED / "test_features.parquet"

MODELS_DIR      = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
METRICS_PATH    = MODELS_DIR / "metrics_log.csv"

# ── Column constants ──────────────────────────────────────────────────────────
ID_COL           = "UniqueID"
TARGET_COL       = "next_3m_txn_count"
LOG_TARGET_COL   = "log1p_target"

NON_FEATURE_COLS = {ID_COL, TARGET_COL, LOG_TARGET_COL}

# ── Cross-validation ──────────────────────────────────────────────────────────
CV_N_FOLDS      = 5
CV_RANDOM_STATE = 42

# ── LightGBM hyperparameters (tuned for RMSLE ≤ 0.34) ───────────────────────
#
#   Key changes vs previous config:
#   - learning_rate  0.03  → 0.01   (slower, more trees, better generalization)
#   - n_estimators   3000  → 5000   (more headroom for early stopping)
#   - num_leaves     127   → 63     (reduced complexity → less overfit)
#   - max_depth      -1    → 7      (explicit depth cap)
#   - min_child_samples 20 → 50     (requires more evidence per leaf)
#   - lambda_l1      0.1   → 2.0    (stronger L1 sparsity)
#   - lambda_l2      1.0   → 5.0    (stronger L2 smoothing)
#   - feature_fraction 0.8 → 0.75   (more aggressive feature subsampling)
#   - bagging_fraction 0.8 → 0.75
#   - bagging_freq    1    → 1      (kept)
#   - early_stopping 100   → 200    (give low-LR models more time)
#
LGBM_PARAMS: dict = {
    "objective":         "regression",
    "metric":            "rmse",
    "boosting_type":     "gbdt",
    "n_estimators":      5000,
    "learning_rate":     0.01,
    "num_leaves":        63,
    "max_depth":         7,
    "min_child_samples": 50,
    "feature_fraction":  0.75,
    "bagging_fraction":  0.75,
    "bagging_freq":      1,
    "lambda_l1":         2.0,
    "lambda_l2":         5.0,
    "min_split_gain":    0.02,
    "verbose":           -1,
    "n_jobs":            -1,
    "seed":              CV_RANDOM_STATE,
}

LGBM_FIT_PARAMS: dict = {
    "early_stopping_rounds": 200,
    "log_period":            100,
    "verbose":               False,
}

# ── Feature selection thresholds ──────────────────────────────────────────────
FEAT_IMPORTANCE_MIN  = 1      # drop features with mean importance < this
FEAT_CORR_THRESHOLD  = 0.98   # drop one of each pair with |corr| > this

# ── Saved artefact names ──────────────────────────────────────────────────────
FOLD_MODEL_TEMPLATE  = "fold_{fold}_model.pkl"   # in MODELS_DIR
FOLD_WEIGHTS_FILE    = "fold_weights.npy"
SELECTED_FEATS_FILE  = "selected_features.pkl"
SUBMISSION_FILENAME  = "submission.csv"
OOF_FILENAME         = "oof_predictions.csv"
IMPORTANCE_FILENAME  = "feature_importance.csv"
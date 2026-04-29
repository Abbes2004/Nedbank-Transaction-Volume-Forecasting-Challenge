"""
src/modeling/train.py
----------------------
LightGBM K-Fold training for RMSLE-optimised regression.

Pipeline
--------
1. Load train_features.parquet
2. Build feature matrix X and log1p target y
3. K-Fold cross-validation → train one LGB model per fold
4. Persist per-fold models + OOF predictions

Usage (standalone)
------------------
    python src/modeling/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

# ── Resolve project root so this script is runnable from any working directory ──
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))          # project root → sys.path

from src.utils.config import (                     # noqa: E402
    TRAIN_FEATURES,
    MODELS_DIR,
    ID_COL,
    LOG_TARGET_COL,
    NON_FEATURE_COLS,
    CV_N_FOLDS,
    CV_RANDOM_STATE,
    LGBM_PARAMS,
    LGBM_FIT_PARAMS,
    OOF_FILENAME,
)
from src.utils.logger import get_logger            # noqa: E402

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_train(path: Path = TRAIN_FEATURES) -> pd.DataFrame:
    """Load processed training parquet. Raises if file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Training features not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded training data: %d rows × %d cols", len(df), df.shape[1])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into (X, y_log, ids).

    Parameters
    ----------
    df : Full training dataframe, must contain LOG_TARGET_COL and ID_COL.

    Returns
    -------
    X      : Feature matrix (all columns except NON_FEATURE_COLS).
    y_log  : log1p-transformed target.
    ids    : UniqueID series for OOF bookkeeping.
    """
    if LOG_TARGET_COL not in df.columns:
        raise KeyError(
            f"Column '{LOG_TARGET_COL}' not found. "
            "Run feature engineering before training."
        )

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X     = df[feature_cols].copy()
    y_log = df[LOG_TARGET_COL].copy()
    ids   = df[ID_COL].copy()

    # Convert object/string columns to category (LightGBM handles them natively)
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("category")

    logger.info("Feature matrix: %d features", X.shape[1])
    return X, y_log, ids


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_kfold(
    X: pd.DataFrame,
    y_log: pd.Series,
    ids: pd.Series,
    n_folds: int = CV_N_FOLDS,
    random_state: int = CV_RANDOM_STATE,
    models_dir: Path = MODELS_DIR,
) -> tuple[list[lgb.LGBMRegressor], pd.DataFrame]:
    """
    K-Fold LightGBM training.

    Parameters
    ----------
    X            : Feature matrix.
    y_log        : log1p target.
    ids          : UniqueID series (same index as X / y_log).
    n_folds      : Number of CV folds.
    random_state : RNG seed.
    models_dir   : Directory where per-fold models are saved.

    Returns
    -------
    models      : List of trained LGBMRegressor objects (one per fold).
    oof_df      : DataFrame with columns [ID_COL, 'oof_log_pred', 'y_log_true'].
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_log_preds = np.zeros(len(X), dtype=np.float64)
    models: list[lgb.LGBMRegressor] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        logger.info("── Fold %d / %d ──────────────────────────────", fold_idx, n_folds)

        X_tr, X_val   = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val   = y_log.iloc[train_idx], y_log.iloc[val_idx]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=LGBM_FIT_PARAMS["early_stopping_rounds"],
                    verbose=LGBM_FIT_PARAMS["verbose"],
                ),
                lgb.log_evaluation(
                    period=LGBM_FIT_PARAMS["log_period"],
                ),
            ],
        )

        val_preds                = model.predict(X_val)
        oof_log_preds[val_idx]   = val_preds

        fold_rmse = np.sqrt(np.mean((val_preds - y_val.values) ** 2))
        logger.info(
            "Fold %d — best iteration: %d | val RMSE(log1p): %.6f",
            fold_idx,
            model.best_iteration_,
            fold_rmse,
        )

        # Persist model
        model_path = models_dir / f"lgbm_fold{fold_idx}.txt"
        model.booster_.save_model(str(model_path))
        logger.info("Model saved → %s", model_path)

        models.append(model)

    # Assemble OOF dataframe
    oof_df = pd.DataFrame({
        ID_COL:          ids.values,
        "oof_log_pred":  oof_log_preds,
        "y_log_true":    y_log.values,
    })

    # Save OOF predictions for post-hoc validation
    oof_path = models_dir / OOF_FILENAME
    oof_df.to_parquet(oof_path, index=False)
    logger.info("OOF predictions saved → %s", oof_path)

    return models, oof_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("═══ Training pipeline started ═══")

    df            = load_train()
    X, y_log, ids = build_feature_matrix(df)
    models, oof_df = train_kfold(X, y_log, ids)

    overall_rmse = np.sqrt(
        np.mean((oof_df["oof_log_pred"].values - oof_df["y_log_true"].values) ** 2)
    )
    logger.info("Overall OOF RMSE (log1p scale): %.6f", overall_rmse)
    logger.info("═══ Training pipeline complete (%d folds, %d models) ═══", CV_N_FOLDS, len(models))


if __name__ == "__main__":
    main()

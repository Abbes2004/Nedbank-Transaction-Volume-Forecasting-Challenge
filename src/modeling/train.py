"""
train.py
=========
Train a LightGBM model to predict next_3m_txn_count.

Key design choices
------------------
- Target is log1p-transformed before training (directly minimises RMSLE)
- 5-fold KFold CV with out-of-fold RMSLE reporting
- All 5 fold models saved individually (for ensemble prediction)
- Fold weights computed as  w_i = (1 / RMSLE_i) / Σ(1 / RMSLE_j)
- Feature selection after CV:
    (a) drop zero-importance features
    (b) drop one of each highly-correlated pair (|corr| > 0.98),
        keeping the higher-importance feature
- Final retrained model on full selected features (backup single model)
- Selected feature list saved so predict.py uses the same subset

Usage
-----
    python src/modeling/train.py
"""

import os
import pickle

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("lightgbm is required.  pip install lightgbm")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_PATH       = "data/processed/train_features.parquet"
MODELS_DIR       = "models"
OOF_PATH         = "data/processed/oof_predictions.csv"
IMPORTANCE_PATH  = "data/processed/feature_importance.csv"
FOLD_MODEL_TMPL  = os.path.join(MODELS_DIR, "fold_{fold}_model.pkl")
FOLD_WEIGHTS_PATH = os.path.join(MODELS_DIR, "fold_weights.npy")
SEL_FEATS_PATH   = os.path.join(MODELS_DIR, "selected_features.pkl")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_model.pkl")

TARGET_COL  = "next_3m_txn_count"
ID_COL      = "UniqueID"
N_FOLDS     = 5
RANDOM_SEED = 42

# Feature-selection thresholds
IMPORTANCE_MIN   = 1       # mean importance must exceed this
CORR_THRESHOLD   = 0.98    # |corr| above this → drop lower-importance feature


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_train_data(
    path: str = TRAIN_PATH,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    print(f"[INFO] Loading train data from: {path}")
    df = pd.read_parquet(path)
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)

    y            = df[TARGET_COL].astype(float)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X            = df[feature_cols]

    print(f"[INFO] Train shape: {X.shape}  |  "
          f"Target [{y.min():.0f}, {y.max():.0f}]  mean={y.mean():.1f}")
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# LightGBM parameters
# ---------------------------------------------------------------------------

def get_lgbm_params() -> dict:
    return {
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
        "seed":              RANDOM_SEED,
    }


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    X:             pd.DataFrame,
    importances:   pd.DataFrame,
    imp_min:       float = IMPORTANCE_MIN,
    corr_thresh:   float = CORR_THRESHOLD,
) -> list[str]:
    """
    Two-stage feature selection:
      1. Drop features whose mean cross-fold importance < imp_min.
      2. For each pair of highly-correlated features, drop the one
         with lower importance.

    Parameters
    ----------
    X           : feature DataFrame (used for correlation)
    importances : DataFrame with columns ['feature', 'importance_mean']
    imp_min     : minimum acceptable mean importance
    corr_thresh : absolute correlation threshold

    Returns
    -------
    List of selected feature names.
    """
    # Stage 1 — importance filter
    imp_map  = importances.set_index("feature")["importance_mean"].to_dict()
    keep     = [f for f in X.columns if imp_map.get(f, 0) >= imp_min]
    dropped_imp = len(X.columns) - len(keep)

    # Stage 2 — correlation filter
    X_keep    = X[keep]
    corr_mat  = X_keep.corr().abs()
    upper     = corr_mat.where(
        np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
    )

    drop_corr: set[str] = set()
    for col in upper.columns:
        high_corr_partners = upper.index[upper[col] > corr_thresh].tolist()
        for other in high_corr_partners:
            if other in drop_corr or col in drop_corr:
                continue
            imp_col   = imp_map.get(col,   0)
            imp_other = imp_map.get(other, 0)
            # Keep higher-importance; drop lower
            if imp_col >= imp_other:
                drop_corr.add(other)
            else:
                drop_corr.add(col)

    selected = [f for f in keep if f not in drop_corr]

    print(f"\n[INFO] Feature selection:")
    print(f"  Total features       : {len(X.columns)}")
    print(f"  Dropped (importance) : {dropped_imp}")
    print(f"  Dropped (correlation): {len(drop_corr)}")
    print(f"  Selected             : {len(selected)}")

    return selected


# ---------------------------------------------------------------------------
# Cross-validation trainer
# ---------------------------------------------------------------------------

def train_with_cv(
    X:            pd.DataFrame,
    y:            pd.Series,
    feature_cols: list[str],
    n_folds:      int = N_FOLDS,
) -> tuple[lgb.LGBMRegressor, np.ndarray, pd.DataFrame, list, np.ndarray]:
    """
    K-fold CV on log1p(y).

    Returns
    -------
    final_model  : LGBMRegressor retrained on full data (after feature selection)
    oof_preds    : out-of-fold predictions (raw scale)
    importances  : DataFrame with feature importances averaged over folds
    fold_models  : list of N_FOLDS LGBMRegressor objects (for ensemble predict)
    fold_weights : 1-D array of normalised weights  (1/RMSLE, normalised)
    """
    from sklearn.model_selection import KFold

    y_log      = np.log1p(y.values)
    oof_preds  = np.zeros(len(X))
    imp_df     = pd.DataFrame({"feature": feature_cols})
    fold_models: list = []
    fold_scores: list[float] = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]

        model = lgb.LGBMRegressor(**get_lgbm_params())
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        val_pred_log = model.predict(X_val)
        val_pred_raw = np.expm1(val_pred_log).clip(0)
        oof_preds[val_idx] = val_pred_raw

        fold_rmsle = rmsle(y.values[val_idx], val_pred_raw)
        fold_scores.append(fold_rmsle)
        fold_models.append(model)

        print(f"  Fold {fold}/{n_folds}  RMSLE={fold_rmsle:.6f}  "
              f"iters={model.best_iteration_}")

        imp_df[f"fold_{fold}"] = model.feature_importances_

    cv_rmsle = rmsle(y.values, oof_preds)
    std_rmsle = float(np.std(fold_scores))
    print(f"\n[RESULT] CV RMSLE : {cv_rmsle:.6f}  ±{std_rmsle:.4f}")
    print(f"         Fold scores: {[f'{s:.4f}' for s in fold_scores]}")

    # Fold weights: w_i = (1/RMSLE_i) / sum(1/RMSLE_j)
    raw_w = np.array([1.0 / s for s in fold_scores])
    fold_weights = raw_w / raw_w.sum()
    print(f"         Fold weights: {[f'{w:.4f}' for w in fold_weights]}")

    # Aggregate importance
    imp_cols = [c for c in imp_df.columns if c.startswith("fold_")]
    imp_df["importance_mean"] = imp_df[imp_cols].mean(axis=1)
    imp_df["importance_std"]  = imp_df[imp_cols].std(axis=1)
    importances = (
        imp_df[["feature", "importance_mean", "importance_std"]]
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    # ── Feature selection ────────────────────────────────────────────────────
    selected = select_features(X, importances)
    X_sel    = X[selected]

    # ── Retrain on full data with selected features ──────────────────────────
    print("\n[INFO] Retraining on full dataset (selected features) ...")
    final_model = lgb.LGBMRegressor(**get_lgbm_params())
    final_model.fit(X_sel, y_log)

    return final_model, oof_preds, importances, fold_models, fold_weights, selected


# ---------------------------------------------------------------------------
# Savers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_fold_models(
    fold_models:  list,
    fold_weights: np.ndarray,
) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for i, m in enumerate(fold_models, 1):
        path = FOLD_MODEL_TMPL.format(fold=i)
        with open(path, "wb") as f:
            pickle.dump(m, f)
        print(f"[INFO] Fold model saved → {path}")
    np.save(FOLD_WEIGHTS_PATH, fold_weights)
    print(f"[INFO] Fold weights saved → {FOLD_WEIGHTS_PATH}")


def save_selected_features(selected: list[str]) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(SEL_FEATS_PATH, "wb") as f:
        pickle.dump(selected, f)
    print(f"[INFO] Selected features saved → {SEL_FEATS_PATH}  "
          f"({len(selected)} features)")


def save_final_model(model: lgb.LGBMRegressor) -> None:
    _ensure_dir(FINAL_MODEL_PATH)
    with open(FINAL_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Final model saved → {FINAL_MODEL_PATH}")


def save_oof(y_true: pd.Series, oof_preds: np.ndarray) -> None:
    _ensure_dir(OOF_PATH)
    oof_df = pd.DataFrame({
        ID_COL:                 y_true.index,
        f"{TARGET_COL}_true":  y_true.values,
        f"{TARGET_COL}_pred":  np.round(oof_preds).astype(int),
    })
    oof_df.to_csv(OOF_PATH, index=False)
    print(f"[INFO] OOF predictions saved → {OOF_PATH}")


def save_feature_importance(imp: pd.DataFrame) -> None:
    _ensure_dir(IMPORTANCE_PATH)
    imp.to_csv(IMPORTANCE_PATH, index=False)
    print(f"[INFO] Feature importance saved → {IMPORTANCE_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    X, y, feature_cols = load_train_data()

    (final_model,
     oof_preds,
     importances,
     fold_models,
     fold_weights,
     selected_features) = train_with_cv(X, y, feature_cols)

    # Persist artefacts
    save_fold_models(fold_models, fold_weights)
    save_selected_features(selected_features)
    save_final_model(final_model)
    save_oof(y, oof_preds)
    save_feature_importance(importances)

    # Summary
    print("\n[INFO] Top 20 features:")
    print(importances.head(20).to_string(index=False))

    cv_rmsle = rmsle(y.values, oof_preds)
    print(f"\n[SUMMARY] Final OOF RMSLE : {cv_rmsle:.6f}")
    print("[DONE] train.py complete.")


if __name__ == "__main__":
    main()
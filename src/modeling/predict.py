"""
predict.py
===========
Generate test-set predictions using the saved fold ensemble.

Strategy
--------
- Load all N fold models + their normalised weights
- Load the selected feature list (from train.py's feature selection)
- Predict with each fold model → weighted average in log1p space
- Apply expm1, clip to PRED_FLOOR, round to integer

Fallback: if fold models are not found, falls back to single lgbm_model.pkl.

Usage
-----
    python src/modeling/predict.py
"""

import os
import pickle

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR        = "models"
TEST_PATH         = "data/processed/test_features.parquet"
SUBMIT_PATH       = "submissions/submission.csv"
SAMPLE_PATH       = "data/raw/SampleSubmission.csv"

FINAL_MODEL_PATH  = os.path.join(MODELS_DIR, "lgbm_model.pkl")
FOLD_MODEL_TMPL   = os.path.join(MODELS_DIR, "fold_{fold}_model.pkl")
FOLD_WEIGHTS_PATH = os.path.join(MODELS_DIR, "fold_weights.npy")
SEL_FEATS_PATH    = os.path.join(MODELS_DIR, "selected_features.pkl")

TARGET_COL  = "next_3m_txn_count"
ID_COL      = "UniqueID"
N_FOLDS     = 5
PRED_FLOOR  = 1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_test_features(path: str = TEST_PATH) -> pd.DataFrame:
    print(f"[INFO] Loading test features from: {path}")
    df = pd.read_parquet(path)
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)
    df = df.drop(columns=[TARGET_COL], errors="ignore")
    print(f"[INFO] Test shape: {df.shape}")
    return df


def load_selected_features() -> list[str] | None:
    if not os.path.exists(SEL_FEATS_PATH):
        print("[WARN] selected_features.pkl not found — using all columns.")
        return None
    with open(SEL_FEATS_PATH, "rb") as f:
        feats = pickle.load(f)
    print(f"[INFO] Loaded {len(feats)} selected features.")
    return feats


def load_fold_models() -> tuple[list, np.ndarray] | tuple[None, None]:
    """
    Load all fold models and weights.
    Returns (None, None) if any fold model file is missing.
    """
    models  = []
    weights = None

    for fold in range(1, N_FOLDS + 1):
        path = FOLD_MODEL_TMPL.format(fold=fold)
        if not os.path.exists(path):
            print(f"[WARN] Fold model not found: {path}")
            return None, None
        with open(path, "rb") as f:
            models.append(pickle.load(f))

    if os.path.exists(FOLD_WEIGHTS_PATH):
        weights = np.load(FOLD_WEIGHTS_PATH)
        print(f"[INFO] Fold weights: {[f'{w:.4f}' for w in weights]}")
    else:
        print("[WARN] fold_weights.npy not found — using equal weights.")
        weights = np.ones(N_FOLDS) / N_FOLDS

    print(f"[INFO] Loaded {len(models)} fold models.")
    return models, weights


def load_fallback_model():
    print(f"[INFO] Loading fallback model: {FINAL_MODEL_PATH}")
    with open(FINAL_MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_ensemble(
    fold_models:  list,
    fold_weights: np.ndarray,
    X:            pd.DataFrame,
) -> np.ndarray:
    """
    Weighted average of fold model predictions in log1p space,
    then expm1 + clip + round.
    """
    log_preds_stack = np.stack(
        [model.predict(X) for model in fold_models],
        axis=0,
    )  # shape (n_folds, n_samples)

    # Weighted average in log1p space (more stable than raw space)
    weighted_log = np.average(log_preds_stack, axis=0, weights=fold_weights)

    preds_raw = np.expm1(weighted_log)
    preds_raw = np.clip(preds_raw, PRED_FLOOR, None)
    return np.round(preds_raw).astype(int)


def predict_single(model, X: pd.DataFrame) -> np.ndarray:
    preds_log = model.predict(X)
    preds_raw = np.expm1(preds_log)
    preds_raw = np.clip(preds_raw, PRED_FLOOR, None)
    return np.round(preds_raw).astype(int)


# ---------------------------------------------------------------------------
# Submission builder
# ---------------------------------------------------------------------------

def build_submission(
    ids:   pd.Index,
    preds: np.ndarray,
) -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_PATH)
    expected  = set(sample[ID_COL].values)
    predicted = set(ids)

    missing = expected - predicted
    extra   = predicted - expected

    if missing:
        raise ValueError(f"Missing {len(missing)} UniqueIDs in predictions.")
    if extra:
        raise ValueError(f"{len(extra)} unexpected UniqueIDs.")

    sub = pd.DataFrame({ID_COL: ids, TARGET_COL: preds})
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")

    assert sub[TARGET_COL].isna().sum() == 0, "NaN predictions detected!"
    assert (sub[TARGET_COL] < 0).sum()  == 0, "Negative predictions detected!"

    return sub


def save_submission(sub: pd.DataFrame, path: str = SUBMIT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"[INFO] Submission saved → {path}  ({len(sub):,} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    X = load_test_features()                          # ← 158 features, untouched

    fold_models, fold_weights = load_fold_models()

    if fold_models is not None:
        print("[INFO] Using fold ensemble for prediction.")
        # Fold models were trained on ALL 158 features — NO selection applied
        preds = predict_ensemble(fold_models, fold_weights, X)
    else:
        print("[INFO] Using single final model for prediction.")
        # Final model was retrained on 127 selected features — apply selection
        selected = load_selected_features()
        if selected is not None:
            selected = [f for f in selected if f in X.columns]
            X = X[selected]
            print(f"[INFO] Test set after feature selection: {X.shape}")
        model = load_fallback_model()
        preds = predict_single(model, X)

    print(f"[INFO] Prediction stats: "
          f"min={preds.min()}  max={preds.max()}  "
          f"mean={preds.mean():.1f}  median={int(np.median(preds))}")

    sub = build_submission(X.index, preds)
    save_submission(sub)
    print("[DONE] predict.py complete.")


if __name__ == "__main__":
    main()
"""
src/pipeline.py
---------------
End-to-end orchestration: train → validate → predict.

This script is the single entry point for a full run. It delegates to the
individual modules in src/modeling/ and reports the final RMSLE.

Usage
-----
    python src/pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))   # src/ on path
sys.path.insert(0, str(_HERE.parents[1]))  # project root on path

import numpy as np                      # noqa: E402

from src.modeling.train import (        # noqa: E402
    load_train,
    build_feature_matrix,
    train_kfold,
)
from src.modeling.validate import (     # noqa: E402
    load_oof,
    load_raw_targets,
    compute_oof_rmsle,
    print_validation_report,
)
from src.modeling.predict import (      # noqa: E402
    load_test,
    extract_test_features,
    load_fold_models,
    predict_ensemble,
    postprocess_predictions,
    build_submission,
    save_submission,
)
from src.utils.config import CV_N_FOLDS  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def run_training() -> None:
    """Execute K-Fold training and persist models + OOF predictions."""
    logger.info("── Phase 1 / 3 : Training ──────────────────────────────────")
    df             = load_train()
    X, y_log, ids  = build_feature_matrix(df)
    models, oof_df = train_kfold(X, y_log, ids)

    overall_rmse = float(
        np.sqrt(np.mean(
            (oof_df["oof_log_pred"].values - oof_df["y_log_true"].values) ** 2
        ))
    )
    logger.info(
        "Training complete — %d folds | OOF RMSE(log1p): %.6f",
        CV_N_FOLDS,
        overall_rmse,
    )


def run_validation() -> dict[str, float]:
    """Load OOF predictions and compute RMSLE on the original scale."""
    logger.info("── Phase 2 / 3 : Validation ────────────────────────────────")
    oof_df      = load_oof()
    raw_targets = load_raw_targets()
    metrics     = compute_oof_rmsle(oof_df, raw_targets)
    print_validation_report(metrics)
    return metrics


def run_prediction() -> Path:
    """Load fold models, ensemble-predict the test set, save submission."""
    logger.info("── Phase 3 / 3 : Prediction ────────────────────────────────")
    test_df       = load_test()
    X_test, ids   = extract_test_features(test_df)
    boosters      = load_fold_models()
    log_preds     = predict_ensemble(X_test, boosters)
    final_preds   = postprocess_predictions(log_preds)
    submission    = build_submission(ids, final_preds)
    out_path      = save_submission(submission)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("═══════════════════════════════════════════════════════════")
    logger.info("  Nedbank Transaction Forecasting — Full Pipeline")
    logger.info("═══════════════════════════════════════════════════════════")

    run_training()
    metrics    = run_validation()
    out_path   = run_prediction()

    logger.info("═══════════════════════════════════════════════════════════")
    logger.info("  Pipeline complete")
    logger.info("  OOF RMSLE        : %.6f", metrics["rmsle"])
    logger.info("  Submission saved : %s", out_path)
    logger.info("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()

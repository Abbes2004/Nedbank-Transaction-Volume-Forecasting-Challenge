"""
src/modeling/validate.py
-------------------------
Offline validation using OOF predictions saved by train.py.

Computes RMSLE correctly by inverting the log1p transform before
comparing to the raw integer target.

Usage (standalone)
------------------
    python src/modeling/validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from src.utils.config import (          # noqa: E402
    MODELS_DIR,
    TRAIN_FEATURES,
    ID_COL,
    TARGET_COL,
    LOG_TARGET_COL,
    OOF_FILENAME,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error.

    Parameters
    ----------
    y_true : Ground-truth counts (non-negative).
    y_pred : Predicted counts (non-negative).

    Returns
    -------
    float : RMSLE score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if np.any(y_pred < 0):
        raise ValueError("Predictions must be non-negative for RMSLE.")
    if np.any(y_true < 0):
        raise ValueError("Ground-truth values must be non-negative for RMSLE.")

    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_oof(models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Load OOF parquet written by train.py."""
    oof_path = models_dir / OOF_FILENAME
    if not oof_path.exists():
        raise FileNotFoundError(
            f"OOF file not found: {oof_path}\n"
            "Run src/modeling/train.py first."
        )
    df = pd.read_parquet(oof_path)
    logger.info("Loaded OOF predictions: %d rows", len(df))
    return df


def load_raw_targets(train_path: Path = TRAIN_FEATURES) -> pd.DataFrame:
    """
    Load original integer targets from the processed training parquet.
    Returns a DataFrame with [ID_COL, TARGET_COL].
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found: {train_path}")
    cols = [ID_COL, TARGET_COL]
    df   = pd.read_parquet(train_path, columns=cols)
    logger.info("Loaded raw targets: %d rows", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Validation logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_oof_rmsle(
    oof_df: pd.DataFrame,
    raw_targets: pd.DataFrame,
) -> dict[str, float]:
    """
    Merge OOF predictions with raw integer targets and compute RMSLE.

    Parameters
    ----------
    oof_df      : Output of load_oof() — contains ID_COL and 'oof_log_pred'.
    raw_targets : Output of load_raw_targets() — contains ID_COL and TARGET_COL.

    Returns
    -------
    dict with keys:
        'rmsle'      : Full RMSLE on original count scale.
        'rmse_log1p' : RMSE on log1p scale (directly comparable to training logs).
        'n_rows'     : Number of OOF observations evaluated.
    """
    merged = oof_df.merge(raw_targets, on=ID_COL, how="inner")

    if len(merged) != len(oof_df):
        logger.warning(
            "ID mismatch: OOF has %d rows, matched %d after merge.",
            len(oof_df),
            len(merged),
        )

    # Invert log1p → original count scale, clip negatives as a safety net
    y_pred_raw = np.expm1(merged["oof_log_pred"].values)
    y_pred_raw = np.clip(y_pred_raw, a_min=0.0, a_max=None)

    y_true_raw = merged[TARGET_COL].values.astype(np.float64)

    score_rmsle     = rmsle(y_true_raw, y_pred_raw)
    score_rmse_log  = float(
        np.sqrt(np.mean((merged["oof_log_pred"].values - merged["y_log_true"].values) ** 2))
    )

    return {
        "rmsle":      score_rmsle,
        "rmse_log1p": score_rmse_log,
        "n_rows":     len(merged),
    }


def print_validation_report(metrics: dict[str, float]) -> None:
    """Print a human-readable validation summary to stdout."""
    sep = "═" * 50
    print(sep)
    print("  OOF VALIDATION REPORT")
    print(sep)
    print(f"  Rows evaluated  : {metrics['n_rows']:,}")
    print(f"  RMSLE (count)   : {metrics['rmsle']:.6f}   ← Zindi metric")
    print(f"  RMSE (log1p)    : {metrics['rmse_log1p']:.6f}   ← training proxy")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("═══ Validation pipeline started ═══")

    oof_df      = load_oof()
    raw_targets = load_raw_targets()
    metrics     = compute_oof_rmsle(oof_df, raw_targets)

    print_validation_report(metrics)
    logger.info("RMSLE: %.6f", metrics["rmsle"])
    logger.info("═══ Validation pipeline complete ═══")


if __name__ == "__main__":
    main()

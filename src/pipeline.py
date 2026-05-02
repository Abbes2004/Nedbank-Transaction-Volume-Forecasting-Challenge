"""
pipeline.py
============
End-to-end pipeline orchestrator for the Nedbank Transaction Forecasting Challenge.

Steps
-----
1. transactions  → data/interim/transactions_features.parquet
2. financials    → data/interim/financials_features.parquet
3. demographics  → data/interim/demographics_features.parquet
4. merge         → data/processed/train_features.parquet
                   data/processed/test_features.parquet
                   (includes interaction engineering)
5. train         → models/fold_*_model.pkl + fold_weights.npy + selected_features.pkl
6. predict       → submissions/submission.csv
7. validate      → prints OOF RMSLE report

Usage
-----
Full pipeline:
    python src/pipeline.py

Specific steps:
    python src/pipeline.py --steps merge train predict
"""

import argparse
import sys
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_transactions():
    from features.build_transactions_features import main
    print("\n" + "=" * 60)
    print("STEP 1 — Build transaction features")
    print("=" * 60)
    main()


def run_financials():
    from features.build_financials_features import main
    print("\n" + "=" * 60)
    print("STEP 2 — Build financial features")
    print("=" * 60)
    main()


def run_demographics():
    from features.build_demographics_features import main
    print("\n" + "=" * 60)
    print("STEP 3 — Build demographic features")
    print("=" * 60)
    main()


def run_merge():
    from features.merge_features import main
    print("\n" + "=" * 60)
    print("STEP 4 — Merge + interaction features")
    print("=" * 60)
    main()


def run_train():
    from modeling.train import main
    print("\n" + "=" * 60)
    print("STEP 5 — Train model (CV + fold ensemble)")
    print("=" * 60)
    main()


def run_predict():
    from modeling.predict import main
    print("\n" + "=" * 60)
    print("STEP 6 — Generate predictions (fold ensemble)")
    print("=" * 60)
    main()


def run_validate():
    from modeling.validate import main
    print("\n" + "=" * 60)
    print("STEP 7 — Validate OOF predictions")
    print("=" * 60)
    main()


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEP_REGISTRY = {
    "transactions": run_transactions,
    "financials":   run_financials,
    "demographics": run_demographics,
    "features":     lambda: [run_transactions(), run_financials(), run_demographics()],
    "merge":        run_merge,
    "train":        run_train,
    "predict":      run_predict,
    "validate":     run_validate,
}

ALL_STEPS = [
    "transactions",
    "financials",
    "demographics",
    "merge",
    "train",
    "predict",
    "validate",
]


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

def log_metrics(rmsle: float, path: str) -> None:
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rmsle":     round(rmsle, 6),
    }])
    os.makedirs(Path(path).parent, exist_ok=True)
    mode   = "a" if os.path.exists(path) else "w"
    header = not os.path.exists(path)
    row.to_csv(path, mode=mode, header=header, index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nedbank pipeline orchestrator")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEP_REGISTRY.keys()),
        default=ALL_STEPS,
        help="Which steps to run (default: all)",
    )
    args = parser.parse_args()

    total_start = time.time()

    for step in args.steps:
        step_start = time.time()
        STEP_REGISTRY[step]()
        elapsed = time.time() - step_start
        print(f"  ✓ {step} done in {elapsed:.1f}s")

    # Log RMSLE after validate
    if "validate" in args.steps:
        metrics_path = "models/metrics_log.csv"
        oof_path     = "data/processed/oof_predictions.csv"
        if os.path.exists(oof_path):
            oof    = pd.read_csv(oof_path)
            y_true = oof["next_3m_txn_count_true"].values
            y_pred = oof["next_3m_txn_count_pred"].values.clip(0)
            score  = float(np.sqrt(
                np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)
            ))
            log_metrics(score, metrics_path)
            print(f"[INFO] RMSLE {score:.6f} logged → {metrics_path}")

    total_elapsed = time.time() - total_start
    print(f"\n[DONE] Pipeline complete in {total_elapsed:.1f}s")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    main()
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = ROOT / "data/processed/test.parquet"
MODEL_FILE = ROOT / "models/selected_house_price_model.joblib"

REPORT_DIR = ROOT / "reports"
METRICS_FILE = REPORT_DIR / "metrics.json"
PREDICTIONS_FILE = REPORT_DIR / "test_predictions.csv"


def main():
    # This must be the held-out labeled test split,
    # not Kaggle's unlabeled data/raw/test.csv.
    for path in [TEST_FILE, MODEL_FILE]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    # Load only model artifacts you trust.
    # Expected: a fitted pipeline that includes preprocessing
    # and returns SalePrice predictions in dollars.
    model = joblib.load(MODEL_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    if len(test_df) < 2:
        raise ValueError("At least two test rows are required.")

    required = {"Id", "SalePrice"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if test_df["Id"].isna().any():
        raise ValueError("Missing house IDs.")

    if test_df["Id"].duplicated().any():
        raise ValueError("Duplicate house IDs.")

    y_test = pd.to_numeric(
        test_df["SalePrice"], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(y_test).all() or (y_test <= 0).any():
        raise ValueError("SalePrice must be finite and positive.")

    # Match the exact input columns used during training.
    # Fit your training pipeline with a pandas DataFrame
    # so scikit-learn records feature_names_in_.
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The saved model must record its training columns. "
            "Train and save a pipeline fitted on a pandas DataFrame."
        )

    features = list(model.feature_names_in_)

    if {"Id", "SalePrice"} & set(features):
        raise ValueError("Model inputs must exclude Id and SalePrice.")

    missing_features = set(features) - set(test_df.columns)
    if missing_features:
        raise ValueError(
            f"Missing model features: {sorted(missing_features)}"
        )

    X_test = test_df[features]

    # Evaluate the frozen model: do not call fit() here.
    predictions = np.asarray(
        model.predict(X_test), dtype=float
    )

    if predictions.shape != y_test.shape:
        raise ValueError("Unexpected prediction shape.")

    if not np.isfinite(predictions).all():
        raise ValueError("Model returned non-finite predictions.")

    if (predictions <= 0).any():
        raise ValueError(
            "Model returned nonpositive prices. "
            "Review the model before computing log-price error."
        )

    # Error in dollars.
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))

    # Competition-style error: RMSE of natural-log prices.
    log_rmse = float(
        np.sqrt(
            mean_squared_error(
                np.log(y_test),
                np.log(predictions),
            )
        )
    )

    # R² is undefined when all actual target values are identical.
    r2 = (
        float(r2_score(y_test, predictions))
        if np.unique(y_test).size > 1
        else None
    )

    metrics = {
        "test_rows": len(test_df),
        "rmse_dollars": rmse,
        "mae_dollars": mae,
        "rmse_log_price": log_rmse,
        "r2": r2,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, allow_nan=False)

    # Positive residual means the model underpredicted.
    residuals = y_test - predictions

    pd.DataFrame(
        {
            "Id": test_df["Id"].to_numpy(),
            "actual_price": y_test,
            "predicted_price": predictions,
            "residual": residuals,
        }
    ).to_csv(PREDICTIONS_FILE, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, predictions, alpha=0.6)

    lower = float(min(y_test.min(), predictions.min()))
    upper = float(max(y_test.max(), predictions.max()))
    ax.plot(
        [lower, upper],
        [lower, upper],
        "--",
        color="black",
        label="Perfect prediction",
    )
    ax.set(
        xlabel="Actual price ($)",
        ylabel="Predicted price ($)",
        title="House Prices: Actual vs. Predicted",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predictions, residuals, alpha=0.6)
    ax.axhline(0, linestyle="--", color="black")
    ax.set(
        xlabel="Predicted price ($)",
        ylabel="Actual minus predicted price ($)",
        title="House Prices: Test Residuals",
    )
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "residuals.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed"
MODEL_FILE = ROOT / "models/selected_house_price_model.joblib"
METRICS_FILE = ROOT / "artifacts/model_selection.json"


def load_data(path):
    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Empty dataset: {path}")

    if not {"Id", "SalePrice"}.issubset(df.columns):
        raise ValueError(f"Missing Id or SalePrice: {path}")

    if df["Id"].isna().any() or df["Id"].duplicated().any():
        raise ValueError(f"Missing or duplicate IDs: {path}")

    y = pd.to_numeric(df["SalePrice"], errors="raise")

    if not np.isfinite(y.to_numpy(dtype=float)).all() or (y <= 0).any():
        raise ValueError(f"Prices must be finite and positive: {path}")

    return df, y


def make_preprocessor(numeric, categorical):
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                keep_empty_features=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Missing",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main():
    train, y_train = load_data(DATA_DIR / "train.parquet")
    validation, y_validation = load_data(
        DATA_DIR / "validation.parquet"
    )

    if set(train["Id"]) & set(validation["Id"]):
        raise ValueError("Training and validation IDs overlap")

    features = [
        column for column in train.columns
        if column not in {"Id", "SalePrice"}
    ]

    missing = set(features) - set(validation.columns)
    if missing:
        raise ValueError(f"Missing validation features: {sorted(missing)}")

    X_train = train[features]
    X_validation = validation[features]

    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [
        column for column in features if column not in numeric
    ]

    # MSSubClass is a category represented by a numeric code.
    if "MSSubClass" in numeric:
        numeric.remove("MSSubClass")
        categorical.append("MSSubClass")

    candidates = {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_model = None
    best_name = None
    best_score = float("inf")

    for name, estimator in candidates.items():
        # Preprocessing is fitted only on training rows.
        # The target wrapper learns log prices and automatically
        # returns dollar prices during predict().
        model = Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(numeric, categorical),
                ),
                (
                    "regressor",
                    TransformedTargetRegressor(
                        regressor=estimator,
                        func=np.log,
                        inverse_func=np.exp,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)

        if (
            not np.isfinite(predictions).all()
            or (predictions <= 0).any()
        ):
            raise ValueError(f"{name} returned invalid prices")

        scores = {
            "rmse_log_price": float(
                np.sqrt(
                    mean_squared_error(
                        np.log(y_validation),
                        np.log(predictions),
                    )
                )
            ),
            "rmse_dollars": float(
                np.sqrt(
                    mean_squared_error(y_validation, predictions)
                )
            ),
            "mae_dollars": float(
                mean_absolute_error(y_validation, predictions)
            ),
        }

        results[name] = scores
        print(f"{name}: {json.dumps(scores)}")

        if scores["rmse_log_price"] < best_score:
            best_score = scores["rmse_log_price"]
            best_model = model
            best_name = name

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_FILE)

    report = {
        "selected_model": best_name,
        "selection_metric": "rmse_log_price",
        "validation_rmse_log_price": best_score,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "models": results,
    }

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print(f"Selected model: {best_name}")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Selection metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
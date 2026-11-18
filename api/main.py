"""Local API for the selected house-price pipeline."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas.request import HousePriceRequest

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models/selected_house_price_model.joblib"


def make_features(request, feature_names):
    values = request.model_dump(by_alias=True)

    values["house_age_at_sale"] = (
        request.YrSold - request.YearBuilt
    )
    values["years_since_remodel"] = (
        request.YrSold - request.YearRemodAdd
    )

    # Match SQL NULL behavior: an unknown component makes
    # the combined feature unknown.
    if request.TotalBsmtSF is not None:
        values["total_living_area_sf"] = (
            request.TotalBsmtSF + request.GrLivArea
        )

    if (
        request.BsmtFullBath is not None
        and request.BsmtHalfBath is not None
    ):
        values["total_bathrooms"] = (
            request.FullBath
            + 0.5 * request.HalfBath
            + request.BsmtFullBath
            + 0.5 * request.BsmtHalfBath
        )

    outdoor_fields = [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
    ]

    if all(values.get(name) is not None for name in outdoor_fields):
        values["total_outdoor_area_sf"] = sum(
            values[name] for name in outdoor_fields
        )

    for source, target in [
        ("GarageArea", "has_garage"),
        ("TotalBsmtSF", "has_basement"),
        ("PoolArea", "has_pool"),
    ]:
        value = values.get(source)
        if value is not None:
            values[target] = int(value > 0)

    # Supply every column expected by the saved pipeline.
    # Omitted inputs are handled by its fitted imputers.
    row = {
        name: (
            values[name]
            if values.get(name) is not None
            else np.nan
        )
        for name in feature_names
    }

    return pd.DataFrame([row], columns=feature_names)


@asynccontextmanager
async def lifespan(app):
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_FILE}")

    # Load only the trusted model created by your training code.
    model = joblib.load(MODEL_FILE)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model must record its input feature names")

    features = list(model.feature_names_in_)

    if {"Id", "SalePrice"} & set(features):
        raise ValueError("Model inputs must exclude Id and SalePrice")

    app.state.model = model
    app.state.features = features

    yield


app = FastAPI(
    title="House Price Prediction API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": "house-price-model",
        "task": "regression",
        "target": "SalePrice",
        "feature_count": len(app.state.features),
        "feature_names": app.state.features,
    }


@app.post("/predict")
def predict(request: HousePriceRequest):
    features = make_features(request, app.state.features)
    prediction = float(app.state.model.predict(features)[0])

    if not np.isfinite(prediction) or prediction <= 0:
        raise HTTPException(
            status_code=500,
            detail="Model returned an invalid price",
        )

    return {"predicted_price": prediction}
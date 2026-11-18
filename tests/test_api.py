"""Integration tests for the house-price FastAPI endpoints."""

import math

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    # Run application startup and shutdown, including model loading.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_house():
    return {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotFrontage": 65.0,
        "LotArea": 8450,
        "Neighborhood": "CollgCr",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "GrLivArea": 1710,
        "TotalBsmtSF": 856,
        "FullBath": 2,
        "HalfBath": 1,
        "GarageCars": 2,
        "GarageArea": 548,
        "MoSold": 2,
        "YrSold": 2008,
    }


def assert_valid_price(data):
    price = data["predicted_price"]

    assert isinstance(price, (int, float))
    assert not isinstance(price, bool)
    assert math.isfinite(price)
    assert price > 0


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "house-price-model"


def test_predict(client, valid_house):
    response = client.post("/predict", json=valid_house)

    assert response.status_code == 200, response.text
    assert_valid_price(response.json())


def test_predict_missing_lot_frontage(client, valid_house):
    # Assumes LotFrontage is optional and defaults to None.
    valid_house.pop("LotFrontage")

    response = client.post("/predict", json=valid_house)

    assert response.status_code == 200, response.text
    assert_valid_price(response.json())


@pytest.mark.parametrize("quality", [0, 11])
def test_predict_invalid_quality(client, valid_house, quality):
    valid_house["OverallQual"] = quality

    response = client.post("/predict", json=valid_house)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field",
    ["LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea"],
)
def test_predict_negative_area(client, valid_house, field):
    valid_house[field] = -1

    response = client.post("/predict", json=valid_house)

    assert response.status_code == 422


def test_predict_invalid_category(client, valid_house):
    # Assumes MSZoning uses an enum or Literal of valid zoning codes.
    valid_house["MSZoning"] = "Unknown"

    response = client.post("/predict", json=valid_house)

    assert response.status_code == 422


def test_predict_missing_required_field(client, valid_house):
    # Assumes GrLivArea is required by the request schema.
    valid_house.pop("GrLivArea")

    response = client.post("/predict", json=valid_house)

    assert response.status_code == 422


def test_predict_extra_field_rejected(client, valid_house):
    # Assumes the request schema uses extra="forbid".
    valid_house["unexpected_field"] = "test"

    response = client.post("/predict", json=valid_house)

    assert response.status_code == 422


def test_predict_sale_price_rejected(client, valid_house):
    # SalePrice is the prediction target, not a request feature.
    valid_house["SalePrice"] = 208500

    response = client.post("/predict", json=valid_house)

    assert response.status_code == 422
    
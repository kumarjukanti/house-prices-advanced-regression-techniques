import pytest
from pydantic import ValidationError

from api.schemas.request import HousePriceRequest


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


def test_valid_house(valid_house):
    request = HousePriceRequest(**valid_house)

    assert request.OverallQual == 7
    assert request.GrLivArea == 1710


def test_missing_lot_frontage_allowed(valid_house):
    valid_house.pop("LotFrontage")

    request = HousePriceRequest(**valid_house)

    assert request.LotFrontage is None


@pytest.mark.parametrize("quality", [0, 11])
def test_invalid_quality_rejected(valid_house, quality):
    valid_house["OverallQual"] = quality

    with pytest.raises(ValidationError):
        HousePriceRequest(**valid_house)


@pytest.mark.parametrize(
    "field",
    ["LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea"],
)
def test_negative_area_rejected(valid_house, field):
    valid_house[field] = -1

    with pytest.raises(ValidationError):
        HousePriceRequest(**valid_house)


@pytest.mark.parametrize(
    "field",
    ["FullBath", "HalfBath", "GarageCars"],
)
def test_negative_count_rejected(valid_house, field):
    valid_house[field] = -1

    with pytest.raises(ValidationError):
        HousePriceRequest(**valid_house)


@pytest.mark.parametrize("month", [0, 13])
def test_invalid_sale_month_rejected(valid_house, month):
    valid_house["MoSold"] = month

    with pytest.raises(ValidationError):
        HousePriceRequest(**valid_house)


def test_invalid_zoning_rejected(valid_house):
    valid_house["MSZoning"] = "Unknown"

    with pytest.raises(ValidationError):
        HousePriceRequest(**valid_house)


def test_extra_field_rejected(valid_house):
    valid_house["unexpected_field"] = "test"

    with pytest.raises(ValidationError):
        HousePriceRequest(**valid_house)


def test_sale_price_rejected_as_input(valid_house):
    valid_house["SalePrice"] = 208500

    with pytest.raises(ValidationError):
        HousePriceRequest(**valid_house)
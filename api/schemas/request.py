"""Validation for house-price prediction requests."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

NonNegativeFloat = Annotated[float, Field(ge=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class HousePriceRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    MSSubClass: Literal[
        20, 30, 40, 45, 50, 60, 70, 75,
        80, 85, 90, 120, 150, 160, 180, 190,
    ]
    MSZoning: Literal["A", "C (all)", "FV", "I", "RH", "RL", "RP", "RM"]
    LotFrontage: NonNegativeFloat | None = None
    LotArea: Annotated[int, Field(gt=0)]
    Neighborhood: str = Field(min_length=1)

    OverallQual: Annotated[int, Field(ge=1, le=10)]
    OverallCond: Annotated[int, Field(ge=1, le=10)]

    YearBuilt: Annotated[int, Field(ge=1800, le=2100)]
    YearRemodAdd: Annotated[int, Field(ge=1800, le=2100)]

    GrLivArea: Annotated[int, Field(gt=0)]
    TotalBsmtSF: NonNegativeInt | None = None

    FullBath: NonNegativeInt
    HalfBath: NonNegativeInt
    BsmtFullBath: NonNegativeInt | None = None
    BsmtHalfBath: NonNegativeInt | None = None

    GarageCars: NonNegativeInt | None = None
    GarageArea: NonNegativeInt | None = None
    PoolArea: NonNegativeInt | None = None

    WoodDeckSF: NonNegativeInt | None = None
    OpenPorchSF: NonNegativeInt | None = None
    EnclosedPorch: NonNegativeInt | None = None
    ScreenPorch: NonNegativeInt | None = None
    ThreeSeasonPorchSF: NonNegativeInt | None = Field(
        default=None,
        alias="3SsnPorch",
    )

    MoSold: Annotated[int, Field(ge=1, le=12)]
    YrSold: Annotated[int, Field(ge=1800, le=2100)]
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"

EXPECTED_COLUMNS = {
    "Id",
    "SalePrice",
    "MSSubClass",
    "Neighborhood",
    "OverallQual",
    "YearBuilt",
    "GrLivArea",
    "house_age_at_sale",
    "years_since_remodel",
    "total_living_area_sf",
    "total_bathrooms",
    "total_outdoor_area_sf",
    "has_garage",
    "has_basement",
    "has_pool",
}


@pytest.mark.parametrize(
    "name,expected_rows",
    [
        ("train", 876),
        ("validation", 292),
        ("test", 292),
    ],
)
def test_dataset_schema(name, expected_rows):
    path = DATA_DIR / f"{name}.parquet"
    assert path.is_file(), f"Missing dataset: {path}"

    df = pd.read_parquet(path)

    assert len(df) == expected_rows

    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"

    assert df["Id"].is_unique
    assert df["Id"].notna().all()

    prices = pd.to_numeric(df["SalePrice"], errors="raise")
    assert np.isfinite(prices.to_numpy(dtype=float)).all()
    assert (prices > 0).all()

    # Unknown feature values may remain missing until preprocessing.
    for column in ["has_garage", "has_basement", "has_pool"]:
        assert df[column].dropna().isin([0, 1]).all()


def test_no_house_overlap():
    ids = {}

    for name in ["train", "validation", "test"]:
        df = pd.read_parquet(DATA_DIR / f"{name}.parquet")
        ids[name] = set(df["Id"])

    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])


def test_splits_cover_source_data():
    source = pd.read_parquet(
        ROOT / "data/interim/model_source.parquet"
    )

    splits = [
        pd.read_parquet(DATA_DIR / f"{name}.parquet")
        for name in ["train", "validation", "test"]
    ]

    for split in splits:
        assert set(split.columns) == set(source.columns)

    combined = pd.concat(splits, ignore_index=True)

    assert len(combined) == len(source)
    assert combined["Id"].is_unique
    assert set(combined["Id"]) == set(source["Id"])

    # Splitting should preserve all original values.
    pd.testing.assert_frame_equal(
        combined.sort_values("Id").reset_index(drop=True),
        source.sort_values("Id").reset_index(drop=True),
    )
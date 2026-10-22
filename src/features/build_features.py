import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/interim/model_source.parquet"
OUTPUT_DIR = ROOT / "data/processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(SOURCE)

    if df.empty:
        raise ValueError("Source dataset is empty")

    missing = {"Id", "SalePrice"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["Id"].isna().any() or df["Id"].duplicated().any():
        raise ValueError("House IDs must be present and unique")

    prices = pd.to_numeric(df["SalePrice"], errors="raise")
    if (
        not np.isfinite(prices.to_numpy(dtype=float)).all()
        or (prices <= 0).any()
    ):
        raise ValueError("SalePrice must be finite and positive")

    df["SalePrice"] = prices

    # Split labeled data into 60% train, 20% validation, 20% test.
    train_validation, test = train_test_split(
        df,
        test_size=0.20,
        random_state=args.random_state,
    )

    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=args.random_state,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        output = OUTPUT_DIR / f"{name}.parquet"
        split.to_parquet(output, index=False)
        print(f"{name}: {len(split)} rows saved to {output}")


if __name__ == "__main__":
    main()
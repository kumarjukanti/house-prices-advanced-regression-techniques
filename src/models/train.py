import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    source = ROOT / "data/interim/model_source.parquet"
    output_dir = ROOT / "data/processed"

    df = pd.read_parquet(source)

    if df.empty:
        raise ValueError("The source dataset is empty.")

    if not {"Id", "SalePrice"}.issubset(df.columns):
        raise ValueError("The dataset must contain Id and SalePrice.")

    if df["Id"].isna().any() or df["Id"].duplicated().any():
        raise ValueError("House IDs must be present and unique.")

    if df["SalePrice"].isna().any() or (df["SalePrice"] <= 0).any():
        raise ValueError("SalePrice must be present and positive.")

    # Reserve 20% for final evaluation.
    train_validation, test = train_test_split(
        df,
        test_size=0.20,
        random_state=args.random_state,
    )

    # Reserve 25% of the remaining 80%: 20% of the full dataset.
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=args.random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        path = output_dir / f"{name}.parquet"
        split.to_parquet(path, index=False)
        print(f"{name}: {len(split)} rows -> {path}")


if __name__ == "__main__":
    main()
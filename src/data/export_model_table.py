import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DB = ROOT / "house_prices.db"
OUTPUT = ROOT / "data" / "interim" / "model_source.parquet"


def main():
    print("Exporting house-price model source from SQLite...")

    if not DB.is_file():
        raise FileNotFoundError(f"Database not found: {DB}")

    # Read-only connection prevents accidental database changes.
    with sqlite3.connect(
        f"{DB.as_uri()}?mode=ro", uri=True
    ) as con:
        df = pd.read_sql_query(
            "SELECT * FROM model_source ORDER BY Id",
            con,
        )

    if df.empty:
        raise ValueError(
            "model_source is empty. Load the house data and "
            "price labels before exporting."
        )

    required = {"Id", "SalePrice"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["Id"].isna().any():
        raise ValueError("Missing house IDs found")

    if df["Id"].duplicated().any():
        raise ValueError("Duplicate house IDs found")

    prices = pd.to_numeric(df["SalePrice"], errors="raise")

    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("SalePrice contains missing or infinite values")

    if (prices <= 0).any():
        raise ValueError("SalePrice must be positive")

    df["SalePrice"] = prices

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print(f"Rows exported: {len(df)}")
    print(f"Columns exported: {len(df.columns)}")
    print(f"File saved: {OUTPUT}")
    print("Export completed successfully!")


if __name__ == "__main__":
    main()
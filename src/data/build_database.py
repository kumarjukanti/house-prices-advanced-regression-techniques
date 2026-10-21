from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "house_prices.db"
SCHEMA_PATH = ROOT / "sql" / "01_schema.sql"
RAW_DIR = ROOT / "data" / "raw"


def main():
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")

    labels = train[["Id", "SalePrice"]]
    houses = pd.concat(
        [train.drop(columns=["SalePrice"]), test],
        ignore_index=True,
    )

    if houses["Id"].duplicated().any():
        raise ValueError("Duplicate house IDs found in the CSV files.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_PATH.read_text())

        # Verify that the CSV columns match the database schema.
        expected_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(houses)")
        ]
        if set(houses.columns) != set(expected_columns):
            raise ValueError("CSV columns do not match the houses table.")

        houses = houses[expected_columns]

        # Convert pandas missing values to SQL NULL.
        houses = houses.astype(object).where(houses.notna(), None)
        labels = labels.astype(object).where(labels.notna(), None)

        quoted_columns = ", ".join(f'"{col}"' for col in expected_columns)
        placeholders = ", ".join("?" for _ in expected_columns)

        # Replace existing data atomically, retaining tables and views.
        conn.execute("BEGIN")
        conn.execute("DELETE FROM house_price_labels")
        conn.execute("DELETE FROM houses")

        conn.executemany(
            f"INSERT INTO houses ({quoted_columns}) "
            f"VALUES ({placeholders})",
            houses.itertuples(index=False, name=None),
        )
        conn.executemany(
            "INSERT INTO house_price_labels (Id, SalePrice) VALUES (?, ?)",
            labels.itertuples(index=False, name=None),
        )

        print(f"Loaded {len(houses)} houses")
        print(f"Loaded {len(labels)} price labels")

    print(f"Database saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
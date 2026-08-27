"""Muat seluruh tabel Parquet di data/gold ke PostgreSQL."""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "gold"
DATABASE_URL = (
    "postgresql+psycopg://banking:changeme@localhost:5433/banking_dw"
)


def main() -> None:
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Tidak ada file Parquet di {DATA_DIR}")

    engine = create_engine(DATABASE_URL)

    for parquet_file in parquet_files:
        table_name = parquet_file.stem.lower()
        print(f"Membaca {parquet_file.name}...")
        dataframe = pd.read_parquet(parquet_file)
        print(
            f"Memuat {len(dataframe):,} baris dan "
            f"{len(dataframe.columns)} kolom ke {table_name}..."
        )
        dataframe.to_sql(
            name=table_name,
            con=engine,
            schema="public",
            if_exists="replace",
            index=False,
            # Hindari batas jumlah parameter PostgreSQL saat tabel punya banyak kolom.
            chunksize=500,
            method="multi",
        )

    with engine.connect() as connection:
        tables = connection.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
            )
        )
        print("\nTabel yang berhasil tersedia:")
        for table in tables:
            print(f"- {table.table_name}")


if __name__ == "__main__":
    main()

"""Muat tabel gold dari parquet ke Postgres (skema `gold`).

Parquet tetap menjadi sumber kebenaran pipeline; Postgres adalah lapisan sajian
untuk Streamlit, dbt, dan query ad-hoc analis.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, text

from pipelines.config import GOLD_DIR, settings

LOG = logging.getLogger("pipelines.loaders")

SKEMA = "gold"
UKURAN_BATCH = 20_000

# Indeks yang dibuat setelah pemuatan - kolom yang paling sering difilter UI.
INDEKS = {
    "dim_debitur": ["cif_sk", "grup_id", "is_current"],
    "fact_laporan_keuangan": ["cif_sk", "tahun_buku"],
    "fact_pengajuan": ["cif_sk", "tanggal_pengajuan"],
    "fact_fasilitas": ["cif_sk", "application_id"],
    "fact_kolektibilitas": ["facility_id", "snapshot_date"],
    "fact_covenant": ["facility_id", "snapshot_date"],
    "fact_eksposur_grup": ["grup_id", "snapshot_date"],
    "feat_graf_pit": ["application_id", "snapshot_date"],
    "graph_snapshot_bulanan": ["node_id", "snapshot_date"],
    "gold_graph_edges": ["src_node_id", "dst_node_id", "valid_from"],
    "fact_transfer_giro": ["waktu"],
}


def _engine():
    return create_engine(settings.postgres_dsn, future=True)


def muat_gold(hanya: list[str] | None = None) -> dict[str, int]:
    """Tulis semua parquet di data/gold ke skema gold di Postgres."""
    engine = _engine()
    with engine.begin() as kon:
        kon.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SKEMA}"))

    hasil: dict[str, int] = {}
    for path in sorted(GOLD_DIR.glob("*.parquet")):
        nama = path.stem
        if hanya and nama not in hanya:
            continue
        df = pd.read_parquet(path)
        # Kolom Int64/boolean bernilai NA tidak dikenal driver; turunkan ke object.
        for kolom in df.columns:
            if str(df[kolom].dtype) in ("Int64", "boolean", "string"):
                df[kolom] = df[kolom].astype("object").where(df[kolom].notna(), None)
        df.to_sql(
            nama,
            engine,
            schema=SKEMA,
            if_exists="replace",
            index=False,
            chunksize=UKURAN_BATCH,
            method="multi",
        )
        hasil[nama] = len(df)
        LOG.info("muat %s.%s -> %s baris", SKEMA, nama, len(df))

    with engine.begin() as kon:
        for tabel, kolom in INDEKS.items():
            if tabel not in hasil:
                continue
            for k in kolom:
                kon.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS idx_{tabel}_{k} '
                        f'ON {SKEMA}."{tabel}" ("{k}")'
                    )
                )
    return hasil


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(muat_gold())

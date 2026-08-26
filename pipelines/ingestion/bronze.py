"""Layer BRONZE: baca CSV mentah apa adanya, hanya seleksi kolom + provenance.

Aturan layer ini: tidak ada rekayasa nilai, tidak ada join. Yang dilakukan hanya
memilih kolom yang dipakai rencana data, memberi id baris sebagai provenance,
dan mengubah file besar menjadi parquet supaya layer berikutnya murah dibaca.
"""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from pipelines.config import (
    RAW_FILES,
    TAIWAN_COLUMN_MAP,
    US_PANEL_COLUMN_MAP,
    settings,
)
from pipelines.utils import bersihkan_uang, iter_csv_chunks, write_table

LOG = logging.getLogger("pipelines.bronze")

# Jumlah rekening AML yang dipertahankan. Subgraf terinduksi dari rekening
# berderajat tertinggi -> struktur fan-in/fan-out dan siklusnya tetap utuh,
# tapi ukuran filenya masuk akal untuk laptop.
AML_TOP_ACCOUNTS = 20_000


# --------------------------------------------------------------------- Taiwan
def ingest_taiwan() -> int:
    """data.csv -> bronze.br_taiwan_ratio (rasio keuangan + label, NYATA)."""
    df = pd.read_csv(RAW_FILES["taiwan"])
    tersedia = {k: v for k, v in TAIWAN_COLUMN_MAP.items() if k in df.columns}
    hilang = set(TAIWAN_COLUMN_MAP) - set(tersedia)
    if hilang:
        LOG.warning("kolom Taiwan tidak ditemukan dan dilewati: %s", sorted(hilang))
    out = df[list(tersedia)].rename(columns=tersedia)
    out.insert(0, "taiwan_row_id", range(1, len(out) + 1))
    out["sumber"] = "taiwan_tej"
    write_table(out, "bronze", "br_taiwan_ratio")
    return len(out)


# ------------------------------------------------------------------ US panel
def ingest_us_panel() -> int:
    """american_bankruptcy.csv -> bronze.br_us_panel (pos akuntansi mentah)."""
    df = pd.read_csv(RAW_FILES["us_panel"], encoding="utf-8-sig")
    df = df.rename(columns=US_PANEL_COLUMN_MAP)
    df.insert(0, "us_row_id", range(1, len(df) + 1))
    df["status_label"] = df["status_label"].str.strip().str.lower()
    df["year"] = df["year"].astype("int16")
    write_table(df, "bronze", "br_us_panel")
    return len(df)


# --------------------------------------------------------------------- rating
def ingest_rating() -> int:
    """corporate_rating.csv -> bronze.br_rating (skala rating + rasio per sektor)."""
    df = pd.read_csv(RAW_FILES["rating"])
    df = df.rename(
        columns={
            "Rating": "rating",
            "Name": "nama_perusahaan",
            "Symbol": "simbol",
            "Rating Agency Name": "lembaga_rating",
            "Date": "tanggal_rating",
            "Sector": "sektor_sumber",
        }
    )
    df["tanggal_rating"] = pd.to_datetime(df["tanggal_rating"], errors="coerce")
    df.insert(0, "rating_row_id", range(1, len(df) + 1))
    write_table(df, "bronze", "br_rating")
    return len(df)


# ------------------------------------------------------------------------ SBA
SBA_COLUMNS = [
    "LoanNr_ChkDgt",
    "State",
    "BankState",
    "NAICS",
    "ApprovalDate",
    "ApprovalFY",
    "Term",
    "NoEmp",
    "NewExist",
    "CreateJob",
    "RetainedJob",
    "UrbanRural",
    "RevLineCr",
    "LowDoc",
    "ChgOffDate",
    "DisbursementDate",
    "DisbursementGross",
    "MIS_Status",
    "ChgOffPrinGr",
    "GrAppv",
    "SBA_Appv",
]


def ingest_sba() -> int:
    """SBAnational.csv (~180 MB, kolom uang berformat string) -> bronze.br_sba."""
    potongan = []
    for chunk in iter_csv_chunks(
        RAW_FILES["sba"],
        chunksize=settings.csv_chunksize,
        usecols=SBA_COLUMNS,
        max_rows=200_000 if settings.sample_mode else None,
    ):
        for kolom in ("DisbursementGross", "ChgOffPrinGr", "GrAppv", "SBA_Appv"):
            chunk[kolom] = bersihkan_uang(chunk[kolom])
        for kolom in ("ApprovalDate", "ChgOffDate", "DisbursementDate"):
            chunk[kolom] = pd.to_datetime(chunk[kolom], format="%d-%b-%y", errors="coerce")
        chunk["MIS_Status"] = chunk["MIS_Status"].astype("string").str.strip()
        chunk["RevLineCr"] = chunk["RevLineCr"].astype("string").str.strip().str.upper()
        chunk["LowDoc"] = chunk["LowDoc"].astype("string").str.strip().str.upper()
        chunk["NAICS"] = chunk["NAICS"].astype("string").str.zfill(6)
        potongan.append(chunk)

    df = pd.concat(potongan, ignore_index=True)
    df = df.rename(columns={"LoanNr_ChkDgt": "sba_loan_nr"})
    df["sba_loan_nr"] = df["sba_loan_nr"].astype("string")
    write_table(df, "bronze", "br_sba")
    return len(df)


# ------------------------------------------------------------------------ AML
AML_RENAME = {
    "Timestamp": "waktu_asli",
    "From Bank": "bank_pengirim",
    "Account": "rekening_pengirim",
    "To Bank": "bank_penerima",
    "Account.1": "rekening_penerima",
    "Amount Received": "nominal_diterima",
    "Receiving Currency": "mata_uang_diterima",
    "Amount Paid": "nominal_dibayar",
    "Payment Currency": "mata_uang_dibayar",
    "Payment Format": "format_pembayaran",
    "Is Laundering": "src_is_laundering",
}


def _aml_reader(usecols: list[str]):
    return iter_csv_chunks(
        RAW_FILES["aml"],
        chunksize=settings.csv_chunksize,
        usecols=usecols,
        max_rows=settings.aml_max_rows,
        # Header LI-Small_Trans.csv memakai nama "Account" dua kali; pandas
        # otomatis menjadikan yang kedua "Account.1".
        dtype={"Account": "string", "Account.1": "string"},
    )


def ingest_aml() -> int:
    """LI-Small_Trans.csv (~650 MB) -> bronze.br_aml_transfer.

    Dua lintasan: lintasan pertama menghitung derajat rekening, lintasan kedua
    menyimpan subgraf terinduksi dari rekening paling aktif. Yang diambil adalah
    topologinya (fan-in/fan-out, siklus), bukan ceritanya.
    """
    derajat: Counter[str] = Counter()
    for chunk in _aml_reader(["Account", "Account.1"]):
        derajat.update(chunk["Account"].dropna().tolist())
        derajat.update(chunk["Account.1"].dropna().tolist())
    LOG.info("AML: %s rekening unik terbaca", len(derajat))

    top_n = 2_000 if settings.sample_mode else AML_TOP_ACCOUNTS
    terpilih = {akun for akun, _ in derajat.most_common(top_n)}

    potongan = []
    for chunk in _aml_reader(list(AML_RENAME)):
        chunk = chunk.rename(columns=AML_RENAME)
        chunk = chunk[
            chunk["rekening_pengirim"].isin(terpilih)
            & chunk["rekening_penerima"].isin(terpilih)
            & (chunk["rekening_pengirim"] != chunk["rekening_penerima"])
        ]
        if len(chunk):
            potongan.append(chunk)

    df = (
        pd.concat(potongan, ignore_index=True)
        if potongan
        else pd.DataFrame(columns=list(AML_RENAME.values()))
    )
    df["waktu_asli"] = pd.to_datetime(df["waktu_asli"], format="%Y/%m/%d %H:%M", errors="coerce")
    df.insert(0, "aml_row_id", range(1, len(df) + 1))

    write_table(df, "bronze", "br_aml_transfer")

    profil = (
        pd.DataFrame({"rekening": list(derajat.keys()), "derajat": list(derajat.values())})
        .sort_values("derajat", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    write_table(profil, "bronze", "br_aml_rekening")
    return len(df)


# ----------------------------------------------------------------------- ICIJ
ICIJ_REL_DIPAKAI = [
    "officer_of",
    "registered_address",
    "intermediary_of",
    "same_name_as",
    "same_address_as",
    "same_id_as",
    "similar",
    "similar_company_as",
    "probably_same_officer_as",
    "same_company_as",
    "underlying",
]


def ingest_icij() -> dict[str, int]:
    """Node & relasi ICIJ Offshore Leaks -> beberapa tabel bronze."""
    jumlah: dict[str, int] = {}

    entities = pd.read_csv(
        RAW_FILES["icij_entities"],
        usecols=[
            "node_id",
            "name",
            "jurisdiction",
            "jurisdiction_description",
            "company_type",
            "address",
            "incorporation_date",
            "inactivation_date",
            "status",
            "countries",
            "sourceID",
        ],
        low_memory=False,
    )
    for kolom in ("incorporation_date", "inactivation_date"):
        entities[kolom] = pd.to_datetime(entities[kolom], format="%d-%b-%Y", errors="coerce")
    write_table(entities, "bronze", "br_icij_entity")
    jumlah["entity"] = len(entities)

    officers = pd.read_csv(
        RAW_FILES["icij_officers"],
        usecols=["node_id", "name", "countries", "country_codes", "sourceID"],
        low_memory=False,
    )
    write_table(officers, "bronze", "br_icij_officer")
    jumlah["officer"] = len(officers)

    addresses = pd.read_csv(
        RAW_FILES["icij_addresses"],
        usecols=["node_id", "address", "countries", "country_codes", "sourceID"],
        low_memory=False,
    )
    write_table(addresses, "bronze", "br_icij_address")
    jumlah["address"] = len(addresses)

    intermediaries = pd.read_csv(
        RAW_FILES["icij_intermediaries"],
        usecols=["node_id", "name", "status", "countries", "country_codes", "sourceID"],
        low_memory=False,
    )
    write_table(intermediaries, "bronze", "br_icij_intermediary")
    jumlah["intermediary"] = len(intermediaries)

    potongan = []
    for chunk in iter_csv_chunks(
        RAW_FILES["icij_relationships"],
        chunksize=settings.csv_chunksize,
        usecols=[
            "node_id_start",
            "node_id_end",
            "rel_type",
            "link",
            "status",
            "start_date",
            "end_date",
            "sourceID",
        ],
    ):
        potongan.append(chunk[chunk["rel_type"].isin(ICIJ_REL_DIPAKAI)])
    rel = pd.concat(potongan, ignore_index=True)
    for kolom in ("start_date", "end_date"):
        rel[kolom] = pd.to_datetime(rel[kolom], format="%d-%b-%Y", errors="coerce")
    write_table(rel, "bronze", "br_icij_relationship")
    jumlah["relationship"] = len(rel)

    return jumlah

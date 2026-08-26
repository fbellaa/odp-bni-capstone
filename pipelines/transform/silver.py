"""Layer SILVER: bersihkan, standarkan, dan turunkan rasio dari layer bronze.

Masih satu tabel per sumber - belum ada join antar dataset. Semua definisi rasio
di sini harus sama persis dengan yang dipakai korpus kebijakan kredit sintetis,
karena gerbang kepatuhan (proposal §5.3) membandingkan keduanya.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipelines.config import NAICS_TO_KBLI, RATING_ORDER, settings
from pipelines.utils import kuantil_bucket, rasio_aman, read_table, write_table

LOG = logging.getLogger("pipelines.silver")

# Asumsi yang dipakai untuk menurunkan beban bunga dari pos akuntansi mentah.
# X1..X18 tidak memuat beban bunga, jadi diturunkan dari identitas
# NI = (EBIT - bunga) * (1 - tarif pajak). Asumsi ini WAJIB ikut didokumentasikan
# bersama angka ICR - lihat docs/data-lineage.md.
TARIF_PAJAK = 0.25
BUNGA_MINIMUM_ATAS_LIABILITAS = 0.01

# Kurs statis untuk menyamakan satuan bobot edge AML. Nilai transfer di dataset
# sumber bukan rupiah; konversi ini sintesis dan hanya dipakai sebagai bobot.
KURS_KE_USD = {
    "US Dollar": 1.0,
    "Euro": 1.08,
    "Yuan": 0.14,
    "Yen": 0.0067,
    "Australian Dollar": 0.66,
    "UK Pound": 1.27,
    "Canadian Dollar": 0.74,
    "Swiss Franc": 1.12,
    "Rupee": 0.012,
    "Ruble": 0.011,
    "Brazil Real": 0.20,
    "Mexican Peso": 0.058,
    "Saudi Riyal": 0.27,
    "Shekel": 0.27,
    "Bitcoin": 60000.0,
}
KURS_USD_IDR = 16_000.0


# --------------------------------------------------------------- panel US
def build_silver_us_panel() -> int:
    """Turunkan DER, ICR, debt/EBITDA, siklus modal kerja dari X1..X18."""
    df = read_table("bronze", "br_us_panel")

    ekuitas = df["total_assets"] - df["total_liabilities"]
    df["ekuitas"] = ekuitas
    df["ekuitas_negatif"] = ekuitas <= 0

    df["der"] = rasio_aman(df["total_liabilities"], ekuitas.where(ekuitas > 0))
    total_utang = df["total_long_term_debt"] + df["total_current_liabilities"]
    df["total_utang"] = total_utang
    df["debt_to_ebitda"] = rasio_aman(total_utang, df["ebitda"].where(df["ebitda"] > 0))

    bunga = df["ebit"] - df["net_income"] / (1 - TARIF_PAJAK)
    lantai_bunga = BUNGA_MINIMUM_ATAS_LIABILITAS * df["total_liabilities"].clip(lower=0)
    df["beban_bunga_estimasi"] = np.maximum(bunga, lantai_bunga).replace(0, np.nan)
    df["icr"] = rasio_aman(df["ebit"], df["beban_bunga_estimasi"])

    df["roa"] = rasio_aman(df["net_income"], df["total_assets"])
    df["operating_margin"] = rasio_aman(df["ebit"], df["total_revenue"])
    df["gross_margin"] = rasio_aman(df["gross_profit"], df["total_revenue"])
    df["current_ratio"] = rasio_aman(df["current_assets"], df["total_current_liabilities"])
    df["quick_ratio"] = rasio_aman(
        df["current_assets"] - df["inventory"], df["total_current_liabilities"]
    )
    df["asset_turnover"] = rasio_aman(df["total_revenue"], df["total_assets"])
    df["wc_to_ta"] = rasio_aman(
        df["current_assets"] - df["total_current_liabilities"], df["total_assets"]
    )
    df["re_to_ta"] = rasio_aman(df["retained_earnings"], df["total_assets"])

    # Arus kas operasi tidak ada di X1..X18; proxy = laba bersih + penyusutan.
    df["cfo_proxy"] = df["net_income"] + df["depreciation_amortization"]
    df["cfo_to_ebitda"] = rasio_aman(df["cfo_proxy"], df["ebitda"].where(df["ebitda"] > 0))
    df["cfo_to_liability"] = rasio_aman(df["cfo_proxy"], df["total_liabilities"])

    # Siklus modal kerja: DSO + DIO. DPO tidak bisa dihitung karena utang usaha
    # tidak tersedia pada X1..X18 - keterbatasan ini dicatat di dokumentasi.
    df["dso_hari"] = rasio_aman(df["total_receivables"], df["total_revenue"]) * 365
    df["dio_hari"] = rasio_aman(df["inventory"], df["cogs"]) * 365
    df["siklus_modal_kerja_hari"] = (df["dso_hari"] + df["dio_hari"]).clip(0, 720)

    df = df.sort_values(["company_name", "year"])
    grup = df.groupby("company_name", sort=False)
    df["growth_penjualan"] = grup["total_revenue"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["label_default"] = (df["status_label"] == "failed").astype("int8")

    kolom = [
        "us_row_id",
        "company_name",
        "year",
        "status_label",
        "label_default",
        "total_assets",
        "total_revenue",
        "total_liabilities",
        "total_utang",
        "ekuitas",
        "ekuitas_negatif",
        "ebit",
        "ebitda",
        "net_income",
        "der",
        "debt_to_ebitda",
        "icr",
        "beban_bunga_estimasi",
        "roa",
        "operating_margin",
        "gross_margin",
        "current_ratio",
        "quick_ratio",
        "asset_turnover",
        "wc_to_ta",
        "re_to_ta",
        "cfo_to_ebitda",
        "cfo_to_liability",
        "dso_hari",
        "dio_hari",
        "siklus_modal_kerja_hari",
        "growth_penjualan",
    ]
    out = df[kolom].reset_index(drop=True)
    write_table(out, "silver", "sl_us_panel")
    return len(out)


# ------------------------------------------------------------------- Taiwan
def build_silver_taiwan() -> int:
    """Rasio Taiwan + bucket kuantil untuk pencocokan longgar (bukan nilai persis)."""
    df = read_table("bronze", "br_taiwan_ratio")

    # 'Debt ratio %' pada sumber sudah dinormalisasi 0-1, bukan persen.
    df["der"] = rasio_aman(df["debt_ratio"], (1 - df["debt_ratio"]).where(lambda s: s > 0))
    df["roa"] = df["roa_taiwan"]
    df["icr_dasar"] = df["interest_expense_ratio"]
    df["dso_dasar"] = df["ar_turnover"]
    df["dio_dasar"] = df["inventory_turnover"]

    # Kunci pencocokan sengaja longgar: label + kuintil DER + kuintil ROA.
    # Pencocokan ketat akan menciptakan korelasi palsu yang meniup AUC.
    df["bucket_der"] = kuantil_bucket(df["der"], 5)
    df["bucket_roa"] = kuantil_bucket(df["roa"], 5)
    df["kunci_match"] = (
        df["label_default_taiwan"].astype(str)
        + "-"
        + df["bucket_der"].astype(str)
        + "-"
        + df["bucket_roa"].astype(str)
    )
    write_table(df, "silver", "sl_taiwan_ratio")
    return len(df)


# ------------------------------------------------------------------- rating
def build_silver_rating() -> int:
    """Skala rating + tabel rujukan rasio per kelas rating dan per sektor."""
    df = read_table("bronze", "br_rating")
    df["rating"] = df["rating"].astype("string").str.strip().str.upper()
    df = df[df["rating"].isin(RATING_ORDER)].copy()
    df["urutan_rating"] = df["rating"].map({r: i for i, r in enumerate(RATING_ORDER)})
    write_table(df, "silver", "sl_rating")

    rasio = [
        "currentRatio",
        "quickRatio",
        "debtRatio",
        "debtEquityRatio",
        "operatingProfitMargin",
        "netProfitMargin",
        "returnOnAssets",
        "returnOnEquity",
        "assetTurnover",
        "operatingCashFlowPerShare",
    ]
    rasio = [c for c in rasio if c in df.columns]
    rujukan = (
        df.groupby("rating")[rasio]
        .median()
        .reindex([r for r in RATING_ORDER if r in set(df["rating"])])
        .reset_index()
    )
    rujukan["jumlah_observasi"] = (
        df.groupby("rating").size().reindex(rujukan["rating"]).to_numpy()
    )
    write_table(rujukan, "silver", "sl_rating_rujukan")

    peer = df.groupby(["sektor_sumber", "rating"])[rasio].median().reset_index()
    write_table(peer, "silver", "sl_rating_peer_sektor")
    return len(df)


# ---------------------------------------------------------------------- SBA
def build_silver_sba() -> int:
    """Bersihkan SBA dan hitung lgd_realisasi - satu-satunya LGD nyata di proyek."""
    df = read_table("bronze", "br_sba")
    df = df[df["MIS_Status"].isin(["P I F", "CHGOFF"])].copy()
    df = df[(df["DisbursementGross"] > 0) & (df["Term"] > 0)]

    df["is_default"] = (df["MIS_Status"] == "CHGOFF").astype("int8")
    df["ChgOffPrinGr"] = df["ChgOffPrinGr"].fillna(0.0)
    df["lgd_realisasi"] = (df["ChgOffPrinGr"] / df["DisbursementGross"]).clip(0, 1)
    df.loc[df["is_default"] == 0, "lgd_realisasi"] = np.nan

    df["revolving"] = df["RevLineCr"].eq("Y").fillna(False).astype(bool)
    df["jenis_fasilitas"] = np.where(df["revolving"], "modal_kerja", "investasi")
    df["dokumen_ringkas"] = df["LowDoc"].eq("Y").fillna(False).astype(bool)
    df["perusahaan_baru"] = df["NewExist"].eq(1).fillna(False).astype(bool)

    naics2 = df["NAICS"].astype("string").str[:2]
    kbli = naics2.map(lambda x: NAICS_TO_KBLI.get(x, ("G", "Perdagangan Besar dan Eceran")))
    df["kbli_kategori"] = [k[0] for k in kbli]
    df["kbli_deskripsi"] = [k[1] for k in kbli]

    df["hari_ke_default"] = (df["ChgOffDate"] - df["DisbursementDate"]).dt.days
    df["porsi_penjaminan"] = (df["SBA_Appv"] / df["GrAppv"]).clip(0, 1)
    df["skala_pegawai"] = pd.cut(
        df["NoEmp"].fillna(0),
        bins=[-1, 5, 20, 100, 500, np.inf],
        labels=["mikro", "kecil", "menengah", "besar", "korporasi"],
    ).astype("string")

    kolom = [
        "sba_loan_nr",
        "State",
        "BankState",
        "NAICS",
        "kbli_kategori",
        "kbli_deskripsi",
        "ApprovalDate",
        "ApprovalFY",
        "DisbursementDate",
        "ChgOffDate",
        "Term",
        "NoEmp",
        "skala_pegawai",
        "perusahaan_baru",
        "CreateJob",
        "RetainedJob",
        "revolving",
        "jenis_fasilitas",
        "dokumen_ringkas",
        "DisbursementGross",
        "GrAppv",
        "SBA_Appv",
        "porsi_penjaminan",
        "ChgOffPrinGr",
        "MIS_Status",
        "is_default",
        "lgd_realisasi",
        "hari_ke_default",
    ]
    out = df[kolom].reset_index(drop=True)
    write_table(out, "silver", "sl_sba")

    lgd = out[out["is_default"] == 1]["lgd_realisasi"]
    LOG.info(
        "SBA: %s pinjaman, default %.2f%%, LGD rata-rata %.3f",
        len(out),
        100 * out["is_default"].mean(),
        lgd.mean(),
    )
    return len(out)


# ---------------------------------------------------------------------- AML
def build_silver_aml() -> int:
    """Normalisasi nilai transfer dan petakan waktunya ke jendela model.

    Timestamp asli LI-Small_Trans hanya mencakup 1-17 September 2022. Urutan dan
    jarak relatif antar transfer dipertahankan, lalu diregangkan linier ke
    settings.aml_window supaya snapshot bulanan graf punya isi. Transformasi ini
    SINTESIS dan wajib disebut di dokumentasi model.
    """
    df = read_table("bronze", "br_aml_transfer")
    if df.empty:
        write_table(df, "silver", "sl_aml_transfer")
        return 0

    t = df["waktu_asli"]
    t_min, t_maks = t.min(), t.max()
    awal, akhir = (pd.Timestamp(d) for d in settings.aml_window)
    rentang_asli = (t_maks - t_min).total_seconds() or 1.0
    porsi = (t - t_min).dt.total_seconds() / rentang_asli
    df["waktu"] = awal + pd.to_timedelta(porsi * (akhir - awal).total_seconds(), unit="s")
    df["snapshot_bulan"] = df["waktu"].dt.to_period("M").dt.to_timestamp("M").dt.normalize()

    kurs = df["mata_uang_dibayar"].map(KURS_KE_USD).fillna(1.0)
    df["nominal_usd"] = df["nominal_dibayar"] * kurs
    df["nominal_rp"] = df["nominal_usd"] * KURS_USD_IDR
    df["selisih_rekonsiliasi"] = (
        df["nominal_dibayar"] - df["nominal_diterima"]
    ).abs() / df["nominal_dibayar"].replace(0, np.nan)

    write_table(df, "silver", "sl_aml_transfer")
    return len(df)


# --------------------------------------------------------------------- ICIJ
KATEGORI_LINK = {
    "kepemilikan": ("shareholder", "owner", "beneficial", "beneficiary", "share holder"),
    "kepengurusan": (
        "director",
        "secretary",
        "president",
        "manager",
        "auditor",
        "liquidator",
        "representative",
        "signatory",
        "records & registers",
        "protector",
        "trustee",
        "nominee",
        "officer",
        "partner",
        "member",
        "resident agent",
    ),
}


def _kategori_relasi(rel_type: str, link: str) -> str:
    link = (link or "").lower()
    if rel_type == "officer_of":
        for kategori, kata in KATEGORI_LINK.items():
            if any(k in link for k in kata):
                return kategori
        return "kepengurusan"
    if rel_type == "registered_address":
        return "berbagi_atribut"
    if rel_type == "intermediary_of":
        return "perantara"
    return "kandidat_duplikat"


def build_silver_icij() -> dict[str, int]:
    """Node ICIJ + relasi yang sudah dikategorikan menjadi jenis edge graf."""
    hasil: dict[str, int] = {}

    rel = read_table("bronze", "br_icij_relationship")
    rel["kategori"] = [
        _kategori_relasi(r, l)
        for r, l in zip(rel["rel_type"].astype(str), rel["link"].astype(str))
    ]
    rel["valid_from"] = rel["start_date"].fillna(pd.Timestamp(settings.tanggal_default_edge))
    rel["valid_to"] = rel["end_date"]
    rel["node_id_start"] = pd.to_numeric(rel["node_id_start"], errors="coerce").astype("Int64")
    rel["node_id_end"] = pd.to_numeric(rel["node_id_end"], errors="coerce").astype("Int64")
    rel = rel.dropna(subset=["node_id_start", "node_id_end"])
    write_table(rel, "silver", "sl_icij_relationship")
    hasil["relationship"] = len(rel)

    entity = read_table("bronze", "br_icij_entity")
    entity["node_id"] = pd.to_numeric(entity["node_id"], errors="coerce").astype("Int64")
    entity = entity.dropna(subset=["node_id"])
    derajat = (
        pd.concat([rel["node_id_start"], rel["node_id_end"]])
        .value_counts()
        .rename("derajat_icij")
    )
    entity = entity.merge(derajat, left_on="node_id", right_index=True, how="left")
    entity["derajat_icij"] = entity["derajat_icij"].fillna(0).astype("int32")
    entity["punya_relasi"] = entity["derajat_icij"] > 0
    write_table(entity, "silver", "sl_icij_entity")
    hasil["entity"] = len(entity)

    for nama in ("officer", "address", "intermediary"):
        node = read_table("bronze", f"br_icij_{nama}")
        node["node_id"] = pd.to_numeric(node["node_id"], errors="coerce").astype("Int64")
        node = node.dropna(subset=["node_id"])
        write_table(node, "silver", f"sl_icij_{nama}")
        hasil[nama] = len(node)

    return hasil

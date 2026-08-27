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
from pipelines.utils import kuantil_bucket, rasio_aman, read_table, winsorize, write_table

LOG = logging.getLogger("pipelines.silver")

# Rasio turunan yang dipotong p1/p99 sebelum masuk gold. Hanya RASIO - pos
# absolut (total_assets, ebitda, ...) dibiarkan apa adanya karena itu angka
# akuntansi yang nanti diskalakan ke rupiah, bukan hasil pembagian yang bisa
# meledak oleh penyebut kecil.
RASIO_DIWINSORISASI = [
    "der",
    "debt_to_ebitda",
    "icr",
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
    "growth_penjualan",
]

# Asumsi yang dipakai untuk menurunkan beban bunga dari pos akuntansi mentah.
# X1..X18 tidak memuat beban bunga, jadi diturunkan dari identitas
# NI = (EBIT - bunga) * (1 - tarif pajak). Asumsi ini WAJIB ikut didokumentasikan
# bersama angka ICR - lihat docs/data-lineage.md.
TARIF_PAJAK = 0.22
BUNGA_MINIMUM_ATAS_LIABILITAS = 0.01

# Kurs statis untuk menyamakan satuan bobot edge AML. Nilai transfer di dataset
KURS_KE_USD = {
    "US Dollar": 1.00,
    "Euro": 1.17,
    "Yuan": 0.15,
    "Yen": 0.0063,
    "Australian Dollar": 0.72,
    "UK Pound": 1.36,
    "Canadian Dollar": 0.72,
    "Swiss Franc": 1.25,
    "Rupee": 0.0105,
    "Ruble": 0.012,
    "Brazil Real": 0.19,
    "Mexican Peso": 0.059,
    "Saudi Riyal": 0.27,
    "Shekel": 0.34,
    "Bitcoin": 112000.0,
}

KURS_USD_IDR = 17_700.0

# Pelokalan kanal pembayaran ke padanan sistem pembayaran Indonesia. Seluruh
# konteks Indonesia di pipeline ini sintesis - lihat docs/data-lineage.md.
# Catatan penting: "Bitcoin" -> "Transfer Valas" MENGUBAH MAKNA kanalnya, bukan
# sekadar terjemahan. Kripto tidak sah sebagai alat pembayaran di Indonesia
# (UU 7/2011), sedangkan fact table membingkai transfer ini sebagai giro rupiah
# antar entitas Indonesia. Padanan yang dipilih mempertahankan peran aslinya
# sebagai kanal lintas yurisdiksi berisiko tinggi, dalam bentuk yang sah.
FORMAT_PEMBAYARAN_ID = {
    "Cheque": "Cek",
    "Wire": "RTGS",
    "ACH": "Kliring",
    "Cash": "Tunai",
    "Credit Card": "Kartu Kredit",
    "Bitcoin": "Transfer Valas",
}


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

    df = df.sort_values(["company_name", "year"])
    grup = df.groupby("company_name", sort=False)
    df["growth_penjualan"] = grup["total_revenue"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["label_default"] = (df["status_label"] == "failed").astype("int8")

    # Winsorisasi p1/p99 atas seluruh rasio turunan.
    #
    # rasio_aman() sudah menahan pembagian nol, tapi tidak menahan penyebut yang
    # kecil-tapi-sah. Sisanya lolos apa adanya ke ABT, dan hasilnya ekor yang
    # tidak berarti apa-apa: growth_penjualan mencapai 12.739 (1,27 juta persen),
    # ICR 43.456, current_ratio 4.759. Rasio max/p99 sampai 4.649x, jadi satu
    # baris tunggal bisa menggeser skala seluruh kolom.
    #
    # Dipotong DI SINI, di panel sumber, bukan di ABT - kuantilnya dihitung atas
    # seluruh panel firm-year dan tidak menyentuh populasi pemodelan, sehingga
    # tidak ada informasi test yang merembes ke train lewat ambang potongnya.
    # Pemotongan juga bebas label: murni kuantil kolomnya sendiri.
    for kolom in RASIO_DIWINSORISASI:
        df[kolom] = winsorize(df[kolom])

    # Dihitung setelah dso/dio dipotong, kalau tidak siklusnya menumpuk di
    # batas 720 hari hanya karena ekor yang belum dibersihkan.
    df["siklus_modal_kerja_hari"] = (df["dso_hari"] + df["dio_hari"]).clip(0, 720)

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
    """Normalisasi nilai transfer, waktu, dan kanal pembayarannya.

    Timestamp asli LI-Small_Trans hanya mencakup 1-17 September 2022. Urutan dan
    jarak relatif antar transfer dipertahankan, lalu diregangkan linier ke
    settings.aml_window supaya snapshot bulanan graf punya isi. Transformasi ini
    SINTESIS dan wajib disebut di dokumentasi model.

    `format_pembayaran` juga dipetakan ke padanan Indonesia lewat
    FORMAT_PEMBAYARAN_ID. Label asli disimpan di `src_format_pembayaran` supaya
    pemetaannya tetap bisa ditelusuri.
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

    df["src_format_pembayaran"] = df["format_pembayaran"]
    df["format_pembayaran"] = (
        df["src_format_pembayaran"].map(FORMAT_PEMBAYARAN_ID).fillna(df["src_format_pembayaran"])
    )
    tak_terpetakan = sorted(
        set(df.loc[df["format_pembayaran"] == df["src_format_pembayaran"], "src_format_pembayaran"])
        - set(FORMAT_PEMBAYARAN_ID.values())
    )
    if tak_terpetakan:
        LOG.warning("format pembayaran tanpa padanan Indonesia: %s", tak_terpetakan)

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

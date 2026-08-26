"""Konfigurasi terpusat untuk seluruh data pipeline.

Semua path, skala sintesis, dan parameter timeline dikumpulkan di sini supaya
flow Prefect, generator, dan uji kualitas memakai angka yang sama persis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv opsional
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
INTERIM_DIR = DATA_DIR / "interim"
QUALITY_DIR = DATA_DIR / "quality"

# ---------------------------------------------------------------- file mentah
RAW_FILES = {
    "taiwan": RAW_DIR / "data.csv",
    "us_panel": RAW_DIR / "american_bankruptcy.csv",
    "rating": RAW_DIR / "corporate_rating.csv",
    "sba": RAW_DIR / "SBAnational.csv",
    "aml": RAW_DIR / "LI-Small_Trans.csv",
    "icij_entities": RAW_DIR / "nodes-entities.csv",
    "icij_officers": RAW_DIR / "nodes-officers.csv",
    "icij_addresses": RAW_DIR / "nodes-addresses.csv",
    "icij_intermediaries": RAW_DIR / "nodes-intermediaries.csv",
    "icij_relationships": RAW_DIR / "relationships.csv",
}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """Parameter yang boleh digeser lewat environment variable."""

    seed: int = _int_env("PIPELINE_SEED", 42)

    # Skala populasi. Rencana data meminta 8.000-12.000 firm-year.
    n_debitur: int = _int_env("N_DEBITUR", 3000)
    panel_years: int = _int_env("PANEL_YEARS", 3)

    # Mode cepat untuk smoke test / CI: batasi baris yang dibaca dari file besar.
    sample_mode: bool = os.getenv("SAMPLE_MODE", "0") == "1"
    csv_chunksize: int = _int_env("CSV_CHUNKSIZE", 1_000_000)

    # ------------------------------------------------------------- timeline
    # Tahun buku terakhir pada panel sintetis; pengajuan terjadi setelahnya.
    tahun_buku_terakhir: int = 2024
    tanggal_awal_pengajuan: date = date(2025, 1, 1)
    tanggal_akhir_pengajuan: date = date(2025, 12, 31)
    # Rentang snapshot bulanan graf, covenant, dan kolektibilitas.
    snapshot_awal: date = date(2024, 1, 31)
    # Perilaku fasilitas (kolektibilitas, covenant) diamati sampai akhir 2026.
    snapshot_akhir: date = date(2026, 12, 31)
    # Timestamp AML asli hanya mencakup 1-17 September 2022. Topologinya dipakai
    # apa adanya, tapi waktunya dipetakan monoton ke jendela di bawah supaya
    # sejalan dengan timeline sintetis. WAJIB didokumentasikan (lihat README).
    aml_window: tuple[date, date] = (date(2023, 1, 1), date(2025, 12, 31))
    # Edge ICIJ tanpa tanggal memakai default ini sebagai valid_from.
    tanggal_default_edge: date = date(2019, 1, 1)

    # -------------------------------------------------- skala rupiah (§3.5)
    penjualan_min_rp: float = 30e9
    penjualan_max_rp: float = 300e9
    plafon_min_rp: float = 10e9
    plafon_max_rp: float = 150e9
    saldo_min_rp: float = 10e9
    saldo_max_rp: float = 50e9

    # -------------------------------------------------------------- output
    tulis_ke_postgres: bool = os.getenv("LOAD_TO_POSTGRES", "0") == "1"

    @property
    def aml_max_rows(self) -> int | None:
        raw = os.getenv("AML_MAX_ROWS")
        if raw:
            return int(raw) or None
        return 500_000 if self.sample_mode else None

    @property
    def postgres_dsn(self) -> str:
        user = os.getenv("POSTGRES_USER", "banking")
        pwd = os.getenv("POSTGRES_PASSWORD", "changeme")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "banking_dw")
        return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

    @property
    def snapshot_dates(self) -> list[date]:
        """Semua akhir bulan antara snapshot_awal dan snapshot_akhir."""
        import pandas as pd

        rng = pd.date_range(self.snapshot_awal, self.snapshot_akhir, freq="ME")
        return [d.date() for d in rng]


settings = Settings()

# ------------------------------------------------- kamus kolom rasio Taiwan
# Kolom asli pada data.csv diawali satu spasi (kecuali "Bankrupt?").
TAIWAN_COLUMN_MAP = {
    "Bankrupt?": "label_default_taiwan",
    " Debt ratio %": "debt_ratio",
    " Liability to Equity": "der_taiwan",
    " Interest Expense Ratio": "interest_expense_ratio",
    " Operating Profit Rate": "operating_margin",
    " Operating Gross Margin": "gross_margin",
    " Net Income to Total Assets": "roa_taiwan",
    " Operating profit/Paid-in capital": "op_profit_to_paid_in_capital",
    " Cash Flow to Liability": "cfo_to_liability",
    " Cash Flow to Sales": "cfo_to_sales",
    " Current Ratio": "current_ratio",
    " Quick Ratio": "quick_ratio",
    " Working Capital to Total Assets": "wc_to_ta",
    " Total Asset Turnover": "asset_turnover",
    " Accounts Receivable Turnover": "ar_turnover",
    " Inventory Turnover Rate (times)": "inventory_turnover",
    " Cash/Total Assets": "cash_to_ta",
    " Retained Earnings to Total Assets": "re_to_ta",
    " Total debt/Total net worth": "debt_to_net_worth",
    " Net Value Growth Rate": "growth",
}

# ------------------------------------------ kamus X1..X18 american_bankruptcy
US_PANEL_COLUMN_MAP = {
    "X1": "current_assets",
    "X2": "cogs",
    "X3": "depreciation_amortization",
    "X4": "ebitda",
    "X5": "inventory",
    "X6": "net_income",
    "X7": "total_receivables",
    "X8": "market_value",
    "X9": "net_sales",
    "X10": "total_assets",
    "X11": "total_long_term_debt",
    "X12": "ebit",
    "X13": "gross_profit",
    "X14": "total_current_liabilities",
    "X15": "retained_earnings",
    "X16": "total_revenue",
    "X17": "total_liabilities",
    "X18": "total_operating_expenses",
}

# ------------------------------------------- pemetaan NAICS 2 digit -> KBLI
NAICS_TO_KBLI = {
    "11": ("A", "Pertanian, Kehutanan dan Perikanan"),
    "21": ("B", "Pertambangan dan Penggalian"),
    "22": ("D", "Pengadaan Listrik dan Gas"),
    "23": ("F", "Konstruksi"),
    "31": ("C", "Industri Pengolahan"),
    "32": ("C", "Industri Pengolahan"),
    "33": ("C", "Industri Pengolahan"),
    "42": ("G", "Perdagangan Besar dan Eceran"),
    "44": ("G", "Perdagangan Besar dan Eceran"),
    "45": ("G", "Perdagangan Besar dan Eceran"),
    "48": ("H", "Pengangkutan dan Pergudangan"),
    "49": ("H", "Pengangkutan dan Pergudangan"),
    "51": ("J", "Informasi dan Komunikasi"),
    "52": ("K", "Jasa Keuangan dan Asuransi"),
    "53": ("L", "Real Estat"),
    "54": ("M", "Jasa Profesional, Ilmiah dan Teknis"),
    "55": ("M", "Jasa Profesional, Ilmiah dan Teknis"),
    "56": ("N", "Jasa Persewaan dan Penunjang Usaha"),
    "61": ("P", "Jasa Pendidikan"),
    "62": ("Q", "Jasa Kesehatan dan Kegiatan Sosial"),
    "71": ("R", "Kesenian, Hiburan dan Rekreasi"),
    "72": ("I", "Penyediaan Akomodasi dan Makan Minum"),
    "81": ("S", "Kegiatan Jasa Lainnya"),
    "92": ("O", "Administrasi Pemerintahan"),
}

RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]


def ensure_dirs() -> None:
    for d in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, INTERIM_DIR, QUALITY_DIR):
        d.mkdir(parents=True, exist_ok=True)

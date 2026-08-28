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
except ImportError:
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

    # Skala populasi. Dinaikkan ke 6.000 (dari 6.251 perusahaan layak) supaya
    # setelah dibagi dua angkatan, tiap sisi masih punya cukup kejadian gagal
    # bayar untuk diukur. Pada 3.000, uji out-of-time hanya menyisakan 3
    # kejadian - AUC di atasnya tidak berarti apa-apa.
    n_debitur: int = _int_env("N_DEBITUR", 6000)
    panel_years: int = _int_env("PANEL_YEARS", 3)

    # Mode cepat untuk smoke test / CI: batasi baris yang dibaca dari file besar.
    sample_mode: bool = os.getenv("SAMPLE_MODE", "0") == "1"
    csv_chunksize: int = _int_env("CSV_CHUNKSIZE", 1_000_000)

    # ------------------------------------------------------------- timeline
    #
    # DUA ANGKATAN. Versi pertama proyek ini memakai satu angkatan: semua
    # debitur mengajukan di 2025 dan gagal bayar sesudahnya. Akibatnya pada
    # snapshot penilaian belum ada satu pun default yang terjadi, sehingga
    # neighbor_default_rate_1hop nyaris selalu nol dan seluruh cerita penularan
    # tidak punya tempat untuk muncul (terukur: hanya 6 dari 3.000 pengajuan
    # punya tetangga yang pernah default).
    #
    # Buku lama menghasilkan riwayat gagal bayar; buku baru mengajukan setelah
    # riwayat itu ada dan karena itu bisa "melihat" afiliasinya kolaps.
    porsi_buku_lama: float = 0.5

    buku_lama_tahun_buku_terakhir: int = 2021
    buku_lama_awal_pengajuan: date = date(2022, 1, 1)
    buku_lama_akhir_pengajuan: date = date(2023, 12, 31)

    buku_baru_tahun_buku_terakhir: int = 2024
    buku_baru_awal_pengajuan: date = date(2025, 1, 1)
    buku_baru_akhir_pengajuan: date = date(2025, 12, 31)

    # Rentang snapshot bulanan graf, covenant, dan kolektibilitas. Harus mundur
    # sampai sebelum pengajuan buku lama yang paling awal.
    snapshot_awal: date = date(2021, 1, 31)
    snapshot_akhir: date = date(2026, 12, 31)
    # Timestamp AML asli hanya mencakup 1-17 September 2022. Topologinya dipakai
    # apa adanya, tapi waktunya dipetakan monoton ke jendela di bawah supaya
    # sejalan dengan timeline sintetis. WAJIB didokumentasikan (lihat README).
    aml_window: tuple[date, date] = (date(2021, 1, 1), date(2026, 6, 30))
    # Edge ICIJ tanpa tanggal memakai default ini sebagai valid_from.
    tanggal_default_edge: date = date(2019, 1, 1)

    # -------------------------------------------------- skala rupiah (§3.5)
    penjualan_min_rp: float = 30e9
    penjualan_max_rp: float = 300e9
    plafon_min_rp: float = 10e9
    plafon_max_rp: float = 150e9
    saldo_min_rp: float = 10e9
    saldo_max_rp: float = 50e9

    # Batas kewajaran neraca terhadap penjualan pada ABT.
    #
    # Penjualan dibatasi ke band segmen komersial (Rp 30-300 M), tapi neraca dan
    # laba-rugi ditarik terpisah, sehingga ~5% debitur berakhir dengan aset
    # puluhan sampai ribuan kali omzetnya. Profil itu koheren ke dalam
    # (aset = liabilitas + ekuitas tetap eksak, |EBITDA|/aset tetap wajar) tapi
    # tidak punya padanan di segmen ini, dan yang menentukan: bad rate-nya sama
    # dengan populasi (3,4% vs 3,2%) padahal skor kreditnya jatuh (median 38 vs
    # 61). Jadi baris itu memberi sinyal distress ekstrem ke seluruh blok fin_
    # tanpa outcome yang mengikutinya - derau label persis di fitur terkuat.
    #
    # Ambang 10x = asset turnover < 0,1. Tidak ada celah alami di distribusinya
    # (ekornya meluruh mulus), jadi angka ini adalah keputusan: cukup longgar
    # untuk melewatkan usaha padat aset yang sah, cukup ketat untuk menyingkirkan
    # ekor yang tak bisa ditafsirkan.
    aset_thd_penjualan_maks: float = 10.0

    # ------------------------------------------------ workout LGD (agunan)
    # LGD dulunya disalin bulat-bulat dari SBA dan tidak pernah menyentuh
    # FACT_AGUNAN, sehingga korelasinya terhadap coverage ratio nol (0,05) -
    # model LGD tidak bisa menjawab "berapa kerugian kalau agunan ditambah",
    # pertanyaan yang justru paling sering datang dari bisnis.
    #
    # Parameter di bawah membentuk LGD STRUKTURAL: berapa yang benar-benar
    # tertagih dari eksekusi agunan. Yang menentukan bukan ada/tidaknya agunan,
    # melainkan agunan yang bisa dieksekusi - lihat _lgd_dari_agunan().
    #
    # Nilai agunan turun antara taksasi (15-90 hari sebelum cair) dan eksekusi
    # (bertahun kemudian, saat debitur kolaps dan pasar sedang buruk).
    workout_penurunan_nilai: tuple[float, float] = (0.60, 0.90)
    # Biaya lelang, kuasa hukum, dan pengurusan - dipotong dari hasil eksekusi.
    workout_biaya: float = 0.12
    # Sisa tagihan di atas nilai agunan praktis tidak berjaminan.
    workout_pemulihan_tanpa_jaminan: tuple[float, float] = (0.05, 0.15)
    # Eksekusi agunan di Indonesia jarang selesai di bawah setahun.
    workout_tahun: tuple[float, float] = (1.0, 3.5)
    workout_diskonto: float = 0.10
    # Sebaran nilai LGD tetap diambil dari SBA; yang dibentuk di sini URUTANNYA.
    #
    # Urutan itu dibentuk DUA penggerak, dan porsinya penting. Versi pertama
    # memakai agunan saja (bobot_sba = 0), dan itu keliru: ia mengacak ulang
    # urutan LGD SBA sampai hubungan fitur-SBA terhadap target hilang sama
    # sekali. Terukur - model yang dilatih di 156.610 pinjaman SBA lalu
    # diterapkan ke portofolio jatuh dari R2 +0,42 ke -0,43, yaitu lebih buruk
    # daripada sekadar menebak rata-rata. Padahal tenor, porsi penjaminan, dan
    # sektor memang memengaruhi LGD di pasar mana pun.
    #
    # 0,55 menyeimbangkan keduanya: korelasi LGD-coverage sekitar -0,26 (sejajar
    # data workout bank riil) sementara transfer SBA -> portofolio bertahan di
    # R2 ~0,28, hampir setara R2 dalam-domain SBA sendiri (0,33). Menaikkannya
    # mengembalikan transfer tapi mematikan sinyal agunan; menurunkannya
    # sebaliknya. Dua gerbang di quality/checks.py menjaga keduanya.
    workout_bobot_sba: float = 0.55
    # Derau pengacak di atas campuran peringkat. Kecil saja - dua penggerak
    # sudah menyediakan variasinya sendiri.
    workout_sigma_peringkat: float = 0.05

    # -------------------------------- injeksi afiliasi tersembunyi (langkah 7)
    # Komposisi klaster: 2 debitur yang benar-benar gagal bayar + 4 yang tidak.
    # Keanggotaan klaster otomatis berkorelasi dengan label - dilusi 2:4 menjaga
    # P(default | anggota) di kisaran 33% terhadap base rate 7%, dan angka itu
    # WAJIB dilaporkan bersama hasil ablasi.
    injeksi_afiliasi: bool = os.getenv("INJEKSI_AFILIASI", "1") == "1"
    afiliasi_default_per_klaster: int = 2
    afiliasi_sehat_per_klaster: int = 4
    # Batas berapa banyak debitur gagal bayar yang boleh diserap klaster.
    # Percobaan pertama tanpa batas ini menyerap 224 dari 229 debitur gagal bayar
    # buku baru, sehingga yang BUKAN anggota klaster tinggal bad rate 0,3% dan
    # keanggotaan klaster praktis menjadi label itu sendiri. Injeksi harus
    # menambah struktur, bukan memonopoli populasi kejadiannya.
    afiliasi_porsi_default_terpakai: float = 0.40
    # Jeda antar gagal bayar dalam satu klaster - inilah yang membuat penularan
    # hanya terlihat ke belakang, bukan bersamaan.
    afiliasi_jeda_bulan: tuple[int, int] = (2, 8)

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
    def angkatan(self) -> dict[str, dict]:
        """Parameter tiap angkatan, dipakai joins dan generators."""
        return {
            "buku_lama": {
                "tahun_buku_terakhir": self.buku_lama_tahun_buku_terakhir,
                "awal_pengajuan": self.buku_lama_awal_pengajuan,
                "akhir_pengajuan": self.buku_lama_akhir_pengajuan,
            },
            "buku_baru": {
                "tahun_buku_terakhir": self.buku_baru_tahun_buku_terakhir,
                "awal_pengajuan": self.buku_baru_awal_pengajuan,
                "akhir_pengajuan": self.buku_baru_akhir_pengajuan,
            },
        }

    @property
    def tahun_buku_terakhir(self) -> int:
        """Tahun buku paling akhir di seluruh populasi (dipakai uji kualitas)."""
        return self.buku_baru_tahun_buku_terakhir

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

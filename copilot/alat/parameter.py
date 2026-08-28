"""Konstanta kebijakan yang dipakai tool perhitungan.

Angka-angka di sini sengaja disalin dari `app/ui/lib/mock_engine.py`, bukan
diimpor darinya. Dua alasan:

1. Arah ketergantungan. `app/ui/lib` adalah kode demo antarmuka yang akan
   diganti pemanggilan FastAPI begitu model asli siap (lihat app/ui/README.md).
   Lapisan tool tidak boleh ikut mati saat modul itu dibongkar.
2. Sifat angkanya berbeda. Nilai di mock_engine adalah *placeholder skoring*;
   yang di sini adalah *parameter kebijakan* - ambang covenant, matriks
   kewenangan, batas segmen. Yang kedua berasal dari dokumen kredit, bukan dari
   model, dan berumur lebih panjang.

Konsekuensinya jelas dan disengaja: bila ambang kebijakan berubah, kedua berkas
harus diperbarui. Yang tidak boleh terjadi adalah lapisan tool diam-diam ikut
berubah ketika seseorang menyetel ulang mesin skoring tiruan.
"""

from __future__ import annotations

# Batas segmen komersial (proposal 3.5).
SEGMEN = {
    "penjualan_min": 30e9,
    "penjualan_maks": 300e9,
    "plafon_min": 10e9,
    "plafon_maks": 150e9,
    "saldo_min": 10e9,
    "saldo_maks": 50e9,
}

# Komponen pricing berbasis risiko.
BIAYA_DANA = 0.0475
BIAYA_OPERASIONAL = 0.0060
MARGIN_TARGET = 0.0225
PRICING_MIN = 0.0700
PRICING_MAX = 0.1600

# Batas maksimum pemberian kredit satu grup debitur.
BATAS_BMPK_GRUP = 750e9

# Tingkat pemulihan per jenis agunan komersial. LGD = 1 - pemulihan efektif.
RECOVERY_AGUNAN = {
    "Tanpa agunan (clean basis)": 0.10,
    "Piutang dagang (fidusia)": 0.35,
    "Persediaan (fidusia)": 0.40,
    "Mesin dan peralatan": 0.48,
    "Penjaminan korporasi grup": 0.45,
    "Tanah dan bangunan pabrik (SHM/SHGB)": 0.78,
    "Deposito / cash collateral": 0.95,
}

# Fasilitas revolving tidak beramortisasi: kewajiban berjalannya hanya bunga,
# sehingga kapasitas arus kas tidak boleh diuji memakai angsuran anuitas.
FASILITAS_REVOLVING = {
    "Modal kerja - rekening koran",
    "Trade finance - LC impor",
    "Bank garansi proyek",
}

# Skala rating internal: batas atas PD 12 bulan per kelas.
BATAS_GRADE = [
    (0.008, "AAA"),
    (0.016, "AA"),
    (0.030, "A"),
    (0.055, "BBB"),
    (0.095, "BB"),
    (0.185, "B"),
    (1.000, "CCC"),
]

# Covenant keuangan wajib per kelas rating.
COVENANT_PER_RATING = {
    "AAA": {"der_maks": 2.50, "icr_min": 2.00, "dscr_min": 1.25, "uji": "Semesteran"},
    "AA": {"der_maks": 2.50, "icr_min": 2.00, "dscr_min": 1.25, "uji": "Semesteran"},
    "A": {"der_maks": 2.25, "icr_min": 2.25, "dscr_min": 1.25, "uji": "Semesteran"},
    "BBB": {"der_maks": 2.25, "icr_min": 2.25, "dscr_min": 1.25, "uji": "Triwulanan"},
    "BB": {"der_maks": 2.00, "icr_min": 2.50, "dscr_min": 1.35, "uji": "Triwulanan"},
    "B": {"der_maks": 1.75, "icr_min": 3.00, "dscr_min": 1.50, "uji": "Bulanan"},
    "CCC": {"der_maks": 1.50, "icr_min": 3.50, "dscr_min": 1.75, "uji": "Bulanan"},
}

# Tingkat pertanggungan agunan minimum per kelas rating.
COVERAGE_MIN = {
    "AAA": 1.00, "AA": 1.00, "A": 1.10, "BBB": 1.25,
    "BB": 1.25, "B": 1.50, "CCC": 1.50,
}

# Matriks kewenangan komite komersial berdasarkan besaran limit.
MATRIKS_KEWENANGAN = [
    (25e9, "Komite Kredit Wilayah"),
    (75e9, "Komite Kredit Komersial"),
    (150e9, "Komite Kredit Komersial Pusat"),
]

DSCR_MIN_KEBIJAKAN = 1.25

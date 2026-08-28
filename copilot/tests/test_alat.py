"""Uji tool perhitungan.

Seluruh uji di sini berjalan tanpa Ollama. Itu memang inti pemisahannya: kalau
angka pada memo bergantung pada model, angka itu tidak akan pernah bisa diuji
seperti ini.
"""

from __future__ import annotations

import pytest

from copilot.alat import keuangan
from copilot.alat.registrasi import jalankan


# ------------------------------------------------------------ rasio keuangan
def test_der_dan_marjin():
    h = keuangan.hitung_rasio_keuangan(
        penjualan=240e9, ebitda=26.4e9, utang_berbunga=45e9, ekuitas=25e9, beban_bunga=4.2e9
    )
    assert h["der"] == pytest.approx(1.8)
    assert h["ebitda_margin"] == pytest.approx(0.11)
    # Nilai dibulatkan 4 desimal di tool supaya angka di memo tidak berekor panjang.
    assert h["interest_coverage"] == pytest.approx(26.4 / 4.2, abs=1e-4)


def test_icr_tanpa_beban_bunga_tidak_dianggap_lolos():
    """Beban bunga nol berarti ICR tak terdefinisi, bukan tak hingga.

    Kalau dikembalikan sebagai angka besar, uji covenant akan meloloskannya
    seolah-olah rasio itu terukur.
    """
    h = keuangan.hitung_rasio_keuangan(
        penjualan=100e9, ebitda=10e9, utang_berbunga=0, ekuitas=50e9, beban_bunga=0
    )
    assert h["interest_coverage"] is None
    assert "catatan" in h

    cov = keuangan.periksa_covenant(grade="BBB", der=0.0, dscr=2.0, interest_coverage=None)
    assert not cov["lolos"]
    assert "Interest coverage minimum" in cov["dilanggar"]


def test_ekuitas_nol_ditolak():
    with pytest.raises(keuangan.GalatMasukan):
        keuangan.hitung_rasio_keuangan(
            penjualan=100e9, ebitda=10e9, utang_berbunga=10e9, ekuitas=0
        )


# ------------------------------------------------------------------ angsuran
def test_revolving_hanya_bunga():
    """Fasilitas revolving tidak beramortisasi.

    Menguji kapasitas arus kasnya dengan angsuran anuitas akan menolak nasabah
    yang sebenarnya sanggup.
    """
    revolving = keuangan.hitung_angsuran(
        pokok=80e9, tenor_bulan=12, bunga_tahunan=0.10,
        jenis_fasilitas="Modal kerja - rekening koran",
    )
    assert revolving["revolving"]
    assert revolving["kewajiban_tahunan"] == pytest.approx(8e9)

    term = keuangan.hitung_angsuran(
        pokok=80e9, tenor_bulan=12, bunga_tahunan=0.10,
        jenis_fasilitas="Investasi - term loan",
    )
    assert not term["revolving"]
    assert term["kewajiban_tahunan"] > revolving["kewajiban_tahunan"]


def test_bunga_dalam_persen_ditolak():
    """10.5 hampir pasti maksudnya 10,5 persen, bukan 1050 persen."""
    with pytest.raises(keuangan.GalatMasukan):
        keuangan.hitung_angsuran(pokok=80e9, tenor_bulan=36, bunga_tahunan=10.5)


# ---------------------------------------------------------------------- LGD
def test_lgd_dibatasi_porsi_yang_tertutup():
    """Agunan senilai setengah plafon tidak memberi pemulihan penuh."""
    penuh = keuangan.estimasi_lgd("Deposito / cash collateral", 100e9, 100e9)
    separuh = keuangan.estimasi_lgd("Deposito / cash collateral", 50e9, 100e9)
    assert penuh["lgd"] == pytest.approx(0.05)
    assert separuh["lgd"] == pytest.approx(0.525)


def test_coverage_berlebih_tidak_menambah_pemulihan():
    banyak = keuangan.estimasi_lgd("Mesin dan peralatan", 500e9, 100e9)
    pas = keuangan.estimasi_lgd("Mesin dan peralatan", 100e9, 100e9)
    assert banyak["lgd"] == pas["lgd"]


def test_agunan_tidak_dikenal_ditolak():
    with pytest.raises(keuangan.GalatMasukan):
        keuangan.estimasi_lgd("Rumah tinggal", 1e9, 1e9)


# ------------------------------------------------------------ rating, BMPK
def test_grade_pada_batas_kelas():
    assert keuangan.grade_dari_pd(0.008)["grade"] == "AAA"
    assert keuangan.grade_dari_pd(0.0081)["grade"] == "AA"
    assert keuangan.grade_dari_pd(0.99)["grade"] == "CCC"


def test_bmpk_terlampaui():
    h = keuangan.periksa_bmpk(eksposur_grup_berjalan=700e9, limit_usulan=80e9)
    assert not h["lolos"]
    assert h["sisa_ruang"] < 0
    assert h["limit_maksimum_yang_masih_muat"] == pytest.approx(50e9)


def test_kewenangan_naik_untuk_rating_rendah():
    """Limit sama, rating berbeda, komite berbeda."""
    bbb = keuangan.kewenangan_komite(limit=20e9, grade="BBB")
    b = keuangan.kewenangan_komite(limit=20e9, grade="B")
    assert bbb["komite_pemutus"] == "Komite Kredit Wilayah"
    assert b["komite_pemutus"] == "Komite Kredit Komersial Pusat"
    assert b["dinaikkan_karena_rating"]


# ------------------------------------------------------- pelaksana dan jejak
def test_galat_menjadi_jejak_bukan_exception():
    """Galat harus bisa dikembalikan ke model sebagai umpan balik.

    Melempar exception ke atas memutus kesempatan model memperbaiki argumennya
    dan membatalkan seluruh analisis.
    """
    j = jalankan("estimasi_lgd", {"jenis_agunan": "Rumah", "nilai_agunan": 1, "plafon": 1})
    assert not j.berhasil
    assert "tidak dikenal" in j.galat
    assert "galat" in j.untuk_model()


def test_tool_tak_dikenal_menyebutkan_yang_tersedia():
    j = jalankan("hitung_apa_saja", {})
    assert not j.berhasil
    assert "hitung_dscr" in j.galat


def test_argumen_wajib_hilang_dilaporkan():
    j = jalankan("hitung_dscr", {"ebitda": 10e9})
    assert not j.berhasil
    assert "Argumen tidak sesuai" in j.galat


@pytest.mark.parametrize(
    "masukan, harapan",
    [
        ("1.500.000.000", 1_500_000_000.0),   # pemisah ribuan Indonesia
        ("2.50", 2.5),                         # titik desimal gaya JSON
        ("1.250,75", 1250.75),                 # koma desimal Indonesia
        ("Rp 80.000.000.000", 80_000_000_000.0),
        (" 0.105 ", 0.105),
    ],
)
def test_string_angka_diurai_benar(masukan, harapan):
    """Titik itu ambigu; aturannya mengikuti jumlah tanda, bukan tebakan lokal."""
    from copilot.alat.registrasi import _ke_angka

    assert _ke_angka(masukan) == pytest.approx(harapan)


def test_string_bukan_angka_dibiarkan_utuh():
    j = jalankan(
        "estimasi_lgd",
        {"jenis_agunan": "Deposito / cash collateral", "nilai_agunan": "1.000.000", "plafon": "1.000.000"},
    )
    assert j.berhasil
    assert j.argumen["jenis_agunan"] == "Deposito / cash collateral"


def test_definisi_dan_implementasi_sepadan():
    """Tool yang dideklarasikan tapi tak ada implementasinya baru ketahuan saat
    model memanggilnya di tengah demo - terlalu terlambat."""
    from copilot.alat.definisi import DEFINISI, PETA

    assert {d["function"]["name"] for d in DEFINISI} == set(PETA)

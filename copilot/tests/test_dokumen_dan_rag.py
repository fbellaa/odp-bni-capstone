"""Uji pembacaan dokumen dan pemotongan kebijakan - tanpa memanggil model.

Yang diuji di sini adalah bagian yang deterministik: kontrak skema, penebakan
jenis dokumen dari kata kunci, penggabungan hasil, dan pemotongan per pasal.
Ekstraksi yang sesungguhnya butuh Ollama dan diuji `copilot/scripts/uji_asap.py`.
"""

from __future__ import annotations

import pytest

from copilot.dokumen import ekstraksi
from copilot.dokumen.pdf import HalamanPDF, kelompokkan, tebak_jenis
from copilot.dokumen.skema import (
    Akta,
    BarisMutasi,
    BerkasPengajuan,
    DokumenTerstruktur,
    LaporanKeuangan,
    PemegangSaham,
    Pengurus,
    RekeningKoran,
    Sumber,
)
from copilot.konfigurasi import POLICY_DIR

POJK = POLICY_DIR / "pojk-40-pojk.03-2019.pdf"
DEMO_DIR = POLICY_DIR.parent / "demo-data"


# ------------------------------------------------------- kontrak ke lapisan graf
def test_argumen_resolusi_sesuai_tanda_tangan_telusuri_afiliasi():
    """Bentuknya harus persis kunci yang diterima telusuri_afiliasi()."""
    from inspect import signature

    from pipelines.graph.resolusi import telusuri_afiliasi

    berkas = BerkasPengajuan(
        dokumen=[
            DokumenTerstruktur(
                jenis="akta",
                sumber=Sumber(berkas="akta.pdf"),
                akta=Akta(
                    alamat_operasional="Jl. Industri 5, Karawang",
                    pengurus=[Pengurus(nama="Budi Santoso", jabatan="Direktur Utama")],
                ),
            ),
            DokumenTerstruktur(
                jenis="rekening_koran",
                sumber=Sumber(berkas="rk.pdf"),
                rekening_koran=RekeningKoran(
                    mutasi=[
                        BarisMutasi(kredit=100, rekening_lawan="0011"),
                        BarisMutasi(debit=50, rekening_lawan="0022"),
                        BarisMutasi(debit=25, rekening_lawan="0011"),  # duplikat
                        BarisMutasi(debit=10),  # tanpa lawan
                    ]
                ),
            ),
        ]
    )

    argumen = berkas.argumen_resolusi()
    parameter = set(signature(telusuri_afiliasi).parameters) - {"tanggal"}
    assert set(argumen) == parameter

    assert argumen["nama_pengurus"] == ["Budi Santoso"]
    assert argumen["rekening_lawan"] == ["0011", "0022"]  # unik dan terurut


def test_dokumen_tak_disertakan_menghasilkan_nilai_kosong():
    """telusuri_afiliasi membedakan 'tidak disertakan' dari 'tidak cocok'.

    Pembedaan itu ikut muncul di memo sebagai batas penelaahan, jadi nilai
    kosong tidak boleh diganti string kosong atau nol.
    """
    argumen = BerkasPengajuan().argumen_resolusi()
    assert argumen["alamat_operasional"] is None
    assert argumen["nama_pengurus"] == []
    assert argumen["rekening_lawan"] == []


def test_kelengkapan_menandai_dokumen_yang_kurang():
    berkas = BerkasPengajuan(
        dokumen=[
            DokumenTerstruktur(
                jenis="laporan_keuangan",
                sumber=Sumber(berkas="lk.pdf"),
                laporan_keuangan=LaporanKeuangan(periode="2025", penjualan=1e9),
            )
        ]
    )
    assert berkas.kelengkapan() == {
        "rekening_koran": False,
        "laporan_keuangan": True,
        "akta_dan_kepemilikan": False,
    }


# ---------------------------------------------------------------- penggabungan
def test_lapkeu_digabung_tanpa_menimpa_pos_yang_sudah_terisi():
    """Potongan belakangan biasanya catatan atas laporan - rincian, bukan total."""
    hasil = ekstraksi._gabung_lapkeu(
        [
            LaporanKeuangan(periode="2025", penjualan=240e9),
            LaporanKeuangan(penjualan=12e9, ekuitas=25e9),  # rincian segmen
        ]
    )
    assert hasil.penjualan == 240e9
    assert hasil.ekuitas == 25e9


def test_akta_digabung_tanpa_pengurus_ganda():
    hasil = ekstraksi._gabung_akta(
        [
            Akta(nama_perusahaan="PT A", pengurus=[Pengurus(nama="Budi")]),
            Akta(
                pengurus=[Pengurus(nama="budi"), Pengurus(nama="Siti")],
                pemegang_saham=[PemegangSaham(nama="PT Induk", persentase=70)],
            ),
        ]
    )
    assert [p.nama for p in hasil.pengurus] == ["Budi", "Siti"]
    assert hasil.nama_perusahaan == "PT A"
    assert len(hasil.pemegang_saham) == 1


def test_rekening_koran_menggabung_seluruh_mutasi():
    hasil = ekstraksi._gabung_rekening(
        [
            RekeningKoran(nomor_rekening="123", mutasi=[BarisMutasi(kredit=10)]),
            RekeningKoran(mutasi=[BarisMutasi(debit=5)]),
        ]
    )
    assert hasil.nomor_rekening == "123"
    assert len(hasil.mutasi) == 2
    assert hasil.total_kredit == 10
    assert hasil.total_debit == 5


# ------------------------------------------------------------- penebakan jenis
@pytest.mark.parametrize(
    "teks, harapan",
    [
        ("MUTASI REKENING\nNo. Rekening 001\nSaldo awal ... Saldo akhir", "rekening_koran"),
        ("LAPORAN POSISI KEUANGAN\nAset lancar\nLiabilitas\nEkuitas", "laporan_keuangan"),
        ("AKTA PENDIRIAN\nNotaris\nAnggaran dasar\nPemegang saham", "akta"),
        ("Undangan rapat tahunan pemegang polis asuransi", "tidak_dikenali"),
    ],
)
def test_tebak_jenis(teks, harapan):
    jenis, _ = tebak_jenis([HalamanPDF(nomor=1, teks=teks)])
    assert jenis == harapan


@pytest.mark.skipif(not DEMO_DIR.exists(), reason="docs/demo-data tidak ada")
@pytest.mark.parametrize(
    "pola, harapan",
    [
        ("Data_Kepemilikan_*.pdf", "akta"),
        ("Laporan_Keuangan_*.pdf", "laporan_keuangan"),
        ("Rekening_Koran_*.pdf", "rekening_koran"),
    ],
)
def test_dokumen_demo_terklasifikasi_benar(pola, harapan):
    """Berkas yang akan dipakai saat demo harus tertebak benar tanpa koreksi manual.

    Penebakan jenis berjalan tanpa model, jadi kegagalannya bisa ditangkap di
    sini alih-alih di depan penonton.
    """
    from copilot.dokumen.pdf import baca_halaman

    berkas = sorted(DEMO_DIR.glob(pola))
    assert berkas, f"tidak ada berkas demo yang cocok {pola}"
    for path in berkas:
        jenis, skor = tebak_jenis(baca_halaman(path))
        assert jenis == harapan, f"{path.name} tertebak {jenis} (skor {skor})"


def test_kelompokkan_melewati_halaman_kosong():
    halaman = [
        HalamanPDF(nomor=1, teks="a" * 4000),
        HalamanPDF(nomor=2, teks=""),
        HalamanPDF(nomor=3, teks="b" * 4000),
    ]
    kelompok = kelompokkan(halaman, maks_karakter=5000)
    assert [[h.nomor for h in k] for k in kelompok] == [[1], [3]]


# ---------------------------------------------------------- pemotongan pasal
@pytest.mark.skipif(not POJK.exists(), reason="PDF kebijakan tidak ada")
def test_potongan_membawa_nomor_pasal_dan_halaman():
    """Syarat dari docs/policies/README.md: sitasi harus bisa menunjuk pasalnya."""
    from copilot.rag.potong import potong_pdf

    potongan = potong_pdf(POJK)
    assert len(potongan) > 50

    berpasal = [p for p in potongan if p.pasal]
    assert len(berpasal) > 50
    assert len({p.pasal for p in berpasal}) > 50

    for p in berpasal:
        assert p.halaman, f"potongan {p.id} tidak punya nomor halaman"
        assert f"Pasal {p.pasal}" in p.rujukan

    # Potongan mengikuti urutan dokumen, dan pasal yang terbagi tetap berdekatan
    # - kalau tidak, sitasi menunjuk pasal yang benar tetapi konteks yang salah.
    assert berpasal[0].pasal == "1"

    urut = [p.pasal for p in berpasal]
    jumlah_blok = sum(1 for i, p in enumerate(urut) if i == 0 or p != urut[i - 1])
    assert jumlah_blok == len(set(urut)), (
        "potongan dari satu pasal harus berurutan, tidak terselang pasal lain"
    )


# ------------------------------------------------- paragraf afiliasi di memo
def test_paragraf_nihil_menyebut_cakupan_dan_batas_metode():
    """Nihil tanpa cakupan adalah klaim yang lebih besar dari datanya.

    Paragraf ini masuk credit memo yang dibaca komite. Ia harus menyatakan
    berapa besar semesta yang dicari DAN bahwa nihil berarti tidak diketahui -
    bukan terbukti tidak berafiliasi.
    """
    from copilot.dokumen.jembatan import ringkas_untuk_memo

    paragraf = ringkas_untuk_memo(
        {
            "tersedia": True,
            "tanggal": "2026-05-31",
            "jumlah_kandidat": 0,
            "perlu_telaah": False,
            "jalur_terpakai": [],
            "jalur_kosong": ["pengurus (tidak ada kecocokan)"],
            "cakupan": {"pihak": 18056, "rekening_lawan": 17853},
            "kandidat": [],
            "ada_afiliasi_gagal_bayar": False,
        }
    )
    assert "18.056" in paragraf and "17.853" in paragraf
    assert "TIDAK DIKETAHUI" in paragraf
    assert "2026-05-31" in paragraf


def test_paragraf_nihil_jujur_saat_cakupan_tidak_tercatat():
    """Jangan mengarang cakupan kalau resolusi tidak memberikannya."""
    from copilot.dokumen.jembatan import ringkas_untuk_memo

    paragraf = ringkas_untuk_memo(
        {
            "tersedia": True,
            "tanggal": "2026-05-31",
            "jumlah_kandidat": 0,
            "perlu_telaah": False,
            "jalur_terpakai": [],
            "jalur_kosong": ["pengurus (tidak ada kecocokan)"],
            "cakupan": {},
            "kandidat": [],
            "ada_afiliasi_gagal_bayar": False,
        }
    )
    assert "tidak tercatat" in paragraf


def test_paragraf_temuan_juga_menyebut_cakupan():
    from copilot.dokumen.jembatan import ringkas_untuk_memo

    paragraf = ringkas_untuk_memo(
        {
            "tersedia": True,
            "tanggal": "2026-05-31",
            "jumlah_kandidat": 8,
            "perlu_telaah": True,
            "jalur_terpakai": ["alamat"],
            "jalur_kosong": [],
            "cakupan": {"alamat": 1543},
            "kandidat": [{"dasar": "alamat_persis"}],
            "ada_afiliasi_gagal_bayar": True,
        }
    )
    assert "1.543" in paragraf
    assert "KKK-13.6" in paragraf

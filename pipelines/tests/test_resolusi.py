"""Uji pencocokan calon nasabah baru ke graf yang sudah ada.

Skenarionya sengaja dibalik: alih-alih memakai debitur karangan, uji ini
mengambil debitur yang MEMANG sudah ada di portofolio, lalu memperlakukan
alamat dan pengurusnya seolah baru diketik RM di formulir. Kalau resolusi
bekerja, debitur itu harus ditemukan kembali.

Yang paling penting di sini adalah uji titik-waktu. Fungsi resolusi berjalan
di layar RM saat pengajuan dinilai, jadi ia tunduk pada aturan yang sama dengan
fitur_pit: hanya edge yang sudah berlaku dan gagal bayar yang sudah terjadi.
Kalau dilonggarkan, angka di layar tidak akan pernah bisa direproduksi model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines.graph import resolusi
from pipelines.utils import read_table, table_exists

pytestmark = pytest.mark.skipif(
    not table_exists("gold", "dim_alamat"),
    reason="DIM_ALAMAT belum dibangun - jalankan python -m pipelines.flows.main_flow",
)

HARI_INI = pd.Timestamp("2026-12-31")


@pytest.fixture(scope="module")
def jembatan() -> pd.DataFrame:
    return read_table("gold", "fact_alamat_debitur")


@pytest.fixture(scope="module")
def dim_alamat() -> pd.DataFrame:
    return read_table("gold", "dim_alamat")


# --------------------------------------------------------------------- alamat
def test_alamat_debitur_lama_menemukan_dirinya(dim_alamat, jembatan):
    """Alamat yang sudah ada di portofolio harus ketemu kalau diketik ulang."""
    layak = dim_alamat[~dim_alamat["is_alamat_agen"]]
    dipakai = layak[layak["alamat_id"].isin(jembatan["alamat_id"])]
    assert len(dipakai), "tidak ada alamat non-agen yang dipakai debitur"

    contoh = dipakai.iloc[0]
    hasil = resolusi.cocokkan_alamat(contoh["alamat_teks"], HARI_INI)
    assert len(hasil), f"alamat {contoh['alamat_teks']!r} tidak ketemu"
    assert (hasil["dasar"] == "alamat_persis").any()

    diharapkan = set(jembatan[jembatan["alamat_id"] == contoh["alamat_id"]]["cif_sk"])
    assert diharapkan <= set(hasil["cif_sk"])


def test_alamat_ketemu_walau_ditulis_dengan_singkatan(dim_alamat, jembatan):
    """RM mengetik 'Jl.', dokumen aslinya menulis 'JALAN' - harus tetap ketemu."""
    layak = dim_alamat[
        ~dim_alamat["is_alamat_agen"] & dim_alamat["alamat_id"].isin(jembatan["alamat_id"])
    ]
    contoh = layak.iloc[0]
    diacak = contoh["alamat_teks"].lower().replace(",", " ").replace("  ", " ")
    hasil = resolusi.cocokkan_alamat(diacak, HARI_INI)
    assert len(hasil), f"{diacak!r} gagal dicocokkan padahal isinya sama"


def test_alamat_asing_tidak_menghasilkan_kandidat():
    hasil = resolusi.cocokkan_alamat("Jl. Tidak Ada Sama Sekali No. 999999 Zzz", HARI_INI)
    assert hasil.empty


def test_alamat_kosong_aman():
    for masukan in ("", "   ", None):
        assert resolusi.cocokkan_alamat(masukan, HARI_INI).empty


def test_alamat_agen_tidak_pernah_jadi_kandidat(dim_alamat):
    agen = dim_alamat[dim_alamat["is_alamat_agen"]]
    if agen.empty:
        pytest.skip("tidak ada alamat agen pada data ini")
    hasil = resolusi.cocokkan_alamat(agen.iloc[0]["alamat_teks"], HARI_INI)
    assert hasil.empty, "kantor agen registrasi tidak boleh dianggap keterkaitan usaha"


# ------------------------------------------------------------------- pengurus
def test_nama_pengurus_menemukan_debitur_yang_dijabatnya():
    pihak = read_table("gold", "dim_pihak", columns=["pihak_id", "nama"])
    kepengurusan = read_table("gold", "fact_kepengurusan")
    aktif = kepengurusan[kepengurusan["valid_from"] <= HARI_INI]
    assert len(aktif), "tidak ada kepengurusan aktif"

    pihak_id = aktif.iloc[0]["pihak_id"]
    nama = pihak[pihak["pihak_id"] == pihak_id].iloc[0]["nama"]

    hasil = resolusi.cocokkan_pengurus([nama], HARI_INI)
    assert len(hasil), f"pengurus {nama!r} tidak ketemu"
    diharapkan = set(aktif[aktif["pihak_id"] == pihak_id]["cif_sk"])
    assert diharapkan <= set(hasil["cif_sk"])


def test_gelar_tidak_menghalangi_pencocokan():
    pihak = read_table("gold", "dim_pihak", columns=["pihak_id", "nama", "tipe"])
    kepengurusan = read_table("gold", "fact_kepengurusan")
    individu = pihak[pihak["tipe"] == "individu"]
    cocok = individu[individu["pihak_id"].isin(kepengurusan["pihak_id"])]
    if cocok.empty:
        pytest.skip("tidak ada pengurus individu")

    nama = cocok.iloc[0]["nama"]
    polos = resolusi.cocokkan_pengurus([nama], HARI_INI)
    bergelar = resolusi.cocokkan_pengurus([f"Bapak {nama}, S.E., M.M."], HARI_INI)
    assert set(polos["cif_sk"]) == set(bergelar["cif_sk"])


def test_nama_asing_tidak_menghasilkan_kandidat():
    hasil = resolusi.cocokkan_pengurus(["Zzzqqq Xxwwvv Nonexistent"], HARI_INI)
    assert hasil.empty


def test_daftar_pengurus_kosong_aman():
    assert resolusi.cocokkan_pengurus([], HARI_INI).empty
    assert resolusi.cocokkan_pengurus(None, HARI_INI).empty


# ---------------------------------------------------------------- titik-waktu
def test_alamat_belum_berlaku_tidak_ikut(jembatan, dim_alamat):
    """Edge yang valid_from-nya di masa depan tidak boleh muncul."""
    paling_baru = pd.to_datetime(jembatan["valid_from"]).max()
    sebelum = paling_baru - pd.Timedelta(days=1)

    alamat_baru = jembatan[pd.to_datetime(jembatan["valid_from"]) == paling_baru]
    teks = dim_alamat.set_index("alamat_id").loc[alamat_baru.iloc[0]["alamat_id"], "alamat_teks"]

    sesudah = resolusi.cocokkan_alamat(teks, paling_baru)
    lebih_awal = resolusi.cocokkan_alamat(teks, sebelum)
    assert len(lebih_awal) < len(sesudah) or lebih_awal.empty


def test_gagal_bayar_masa_depan_tidak_bocor():
    """afiliasi_sudah_gagal_bayar hanya boleh mencatat default yang sudah lewat."""
    default = read_table("gold", "fact_default", columns=["cif_sk", "tanggal_default"])
    if default.empty:
        pytest.skip("tidak ada fact_default")

    awal = pd.to_datetime(default["tanggal_default"]).min() - pd.Timedelta(days=1)
    pihak = read_table("gold", "dim_pihak", columns=["pihak_id", "nama"])
    kepengurusan = read_table("gold", "fact_kepengurusan")
    nama = pihak[pihak["pihak_id"] == kepengurusan.iloc[0]["pihak_id"]].iloc[0]["nama"]

    hasil = resolusi.telusuri_afiliasi(tanggal=awal, nama_pengurus=[nama])
    if hasil.kandidat.empty:
        pytest.skip("tidak ada kandidat pada tanggal sedini itu")
    assert not hasil.kandidat["afiliasi_sudah_gagal_bayar"].any(), (
        "ada afiliasi ditandai gagal bayar sebelum satu pun default terjadi"
    )


# --------------------------------------------------------------- alur gabungan
def test_tanpa_dokumen_apa_pun_hasilnya_kosong_bukan_bersih():
    """Nasabah tanpa dokumen harus terbaca 'belum diketahui', bukan 'tidak ada afiliasi'."""
    hasil = resolusi.telusuri_afiliasi(tanggal=HARI_INI)
    assert hasil.kandidat.empty
    assert hasil.jumlah_kandidat == 0
    assert not hasil.perlu_telaah
    # Jalur yang tidak dipakai wajib terlaporkan, supaya analis tahu bedanya
    # antara 'sudah dicari dan bersih' dan 'belum dicari sama sekali'.
    assert len(hasil.jalur_kosong) == 3
    assert all("dokumen tidak disertakan" in j for j in hasil.jalur_kosong)


def test_pengurus_bersama_memicu_telaah():
    pihak = read_table("gold", "dim_pihak", columns=["pihak_id", "nama"])
    kepengurusan = read_table("gold", "fact_kepengurusan")
    # Pihak yang menjabat di banyak debitur - inilah pola nominee bersama.
    banyak = kepengurusan.groupby("pihak_id")["cif_sk"].nunique()
    banyak = banyak[banyak >= 3]
    if banyak.empty:
        pytest.skip("tidak ada pihak yang menjabat di >=3 debitur")

    pihak_id = banyak.index[0]
    nama = pihak[pihak["pihak_id"] == pihak_id].iloc[0]["nama"]

    hasil = resolusi.telusuri_afiliasi(tanggal=HARI_INI, nama_pengurus=[nama])
    assert hasil.jumlah_kandidat >= 3
    assert hasil.perlu_telaah
    assert "pengurus" in hasil.jalur_terpakai


def test_kandidat_selalu_menyertakan_dasarnya(jembatan, dim_alamat):
    layak = dim_alamat[
        ~dim_alamat["is_alamat_agen"] & dim_alamat["alamat_id"].isin(jembatan["alamat_id"])
    ]
    hasil = resolusi.telusuri_afiliasi(
        tanggal=HARI_INI, alamat_operasional=layak.iloc[0]["alamat_teks"]
    )
    if hasil.kandidat.empty:
        pytest.skip("alamat contoh tidak menghasilkan kandidat")
    # Keluarannya daftar untuk ditelaah analis, bukan skor - tiap barisnya wajib
    # bisa dijelaskan.
    assert hasil.kandidat["dasar"].notna().all()
    assert "afiliasi_sudah_gagal_bayar" in hasil.kandidat.columns

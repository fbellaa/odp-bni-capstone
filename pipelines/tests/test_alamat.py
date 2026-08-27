"""Uji DIM_ALAMAT dan pencocokan calon nasabah baru.

Dua hal yang dijaga di sini:

1. Alamat benar-benar menjadi entitas yang bisa dicari - kalau normalisasinya
   tidak konsisten, seluruh premis "alamat calon nasabah bisa dicocokkan"
   runtuh tanpa ada uji lain yang gagal.
2. Klaster afiliasi bermekanisme alamat tidak boleh bisa dibedakan dari alamat
   ICIJ biasa lewat bentuk datanya. Kalau bisa, injeksinya tidak tersembunyi.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines.graph.alamat import (
    MAKS_DEBITUR_PER_ALAMAT,
    normalisasi_alamat,
    pasangan_seralamat,
)
from pipelines.utils import read_table, table_exists

pytestmark = pytest.mark.skipif(
    not table_exists("gold", "dim_alamat"),
    reason="DIM_ALAMAT belum dibangun - jalankan python -m pipelines.flows.main_flow",
)


@pytest.fixture(scope="module")
def dim_alamat() -> pd.DataFrame:
    return read_table("gold", "dim_alamat")


@pytest.fixture(scope="module")
def jembatan() -> pd.DataFrame:
    return read_table("gold", "fact_alamat_debitur")


# ------------------------------------------------------------------ normalisasi
@pytest.mark.parametrize(
    "varian",
    [
        ["Jl. Merdeka No. 12", "JALAN MERDEKA NOMOR 12", "  jln merdeka no 12  "],
        ["Gd. Menara A Lt. 5", "GEDUNG MENARA A LANTAI 5"],
        ["12 High Street", "12 HIGH ST"],
        ["Komp. Ruko Indah Blk. B", "KOMPLEKS RUKO INDAH BLOK B"],
        ["Gg. Melati No. 3", "GANG MELATI NOMOR 3"],
        ["Perum Griya Asri", "PERUMAHAN GRIYA ASRI"],
        ["Ds. Sukamaju, Dsn. Kidul", "DESA SUKAMAJU DUSUN KIDUL"],
    ],
)
def test_varian_penulisan_runtuh_jadi_satu_kunci(varian):
    hasil = normalisasi_alamat(pd.Series(varian)).unique()
    assert len(hasil) == 1, f"{varian} menghasilkan {list(hasil)}"


def test_alamat_berbeda_tidak_ikut_runtuh():
    hasil = normalisasi_alamat(pd.Series(["Jl. Merdeka No. 12", "Jl. Merdeka No. 21"]))
    assert hasil.nunique() == 2


def test_normalisasi_tahan_nilai_kosong():
    hasil = normalisasi_alamat(pd.Series([None, "", "   ", pd.NA]))
    assert (hasil == "").all()


# ---------------------------------------------------------------- bentuk tabel
def test_alamat_id_unik_dan_teksnya_ikut(dim_alamat):
    assert dim_alamat["alamat_id"].is_unique
    assert dim_alamat["alamat_normal"].is_unique
    # Inti perubahan ini: teks alamat sampai ke gold, tidak berhenti di silver.
    assert dim_alamat["alamat_teks"].notna().all()
    assert (dim_alamat["alamat_teks"].astype(str).str.strip() != "").all()


def test_jembatan_menunjuk_alamat_yang_ada(dim_alamat, jembatan):
    assert set(jembatan["alamat_id"]) <= set(dim_alamat["alamat_id"])
    assert not jembatan.duplicated(subset=["cif_sk", "alamat_id"]).any()


def test_debitur_menunjuk_cif_yang_ada(jembatan):
    debitur = read_table("gold", "dim_debitur", columns=["cif_sk"])
    assert set(jembatan["cif_sk"]) <= set(debitur["cif_sk"])


def test_alamat_agen_ditandai_dan_tidak_dipasangkan(dim_alamat, jembatan):
    agen = dim_alamat[dim_alamat["is_alamat_agen"]]
    assert (agen["jumlah_debitur"] > MAKS_DEBITUR_PER_ALAMAT).all()

    # Kantor agen registrasi dengan ratusan badan hukum bukan grup usaha. Kalau
    # ikut dipasangkan ia menghasilkan klik raksasa dan seluruh metrik graf
    # kehilangan arti.
    pasangan = pasangan_seralamat(dim_alamat, jembatan)
    assert not pasangan["alamat_id"].isin(set(agen["alamat_id"])).any()


# -------------------------------------------------------------------- privasi
def test_alamat_asli_icij_tidak_ikut_ke_gold(dim_alamat):
    """Teks alamat ICIJ adalah data nyata dari dokumen bocoran.

    Kebijakan proyek sudah menolak membawa nama asli ICIJ ke gold; alamat tunduk
    pada aturan yang sama. Yang disimpan alamat sintetis, dengan struktur berbagi
    alamatnya dipertahankan.
    """
    asli = set(
        read_table("silver", "sl_icij_address", columns=["address"])["address"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    di_gold = set(dim_alamat["alamat_teks"].dropna().astype(str).str.strip())
    bocor = di_gold & asli
    assert not bocor, f"{len(bocor)} alamat asli ICIJ ikut ke gold, contoh: {list(bocor)[:3]}"


def test_alamat_gold_berbentuk_indonesia(dim_alamat):
    """Debitur di sini badan hukum Indonesia sintetis - alamatnya harus sejalan."""
    assert (dim_alamat["negara"] == "Indonesia").all()


def test_struktur_berbagi_alamat_bertahan_setelah_disamarkan():
    """Penyamaran tidak boleh memecah atau menggabungkan kelompok se-alamat.

    Ini yang membuat penyamaran aman: kalau dua entitas berbagi alamat di ICIJ,
    sesudah disamarkan mereka harus tetap berbagi alamat - dan kalau tidak, tetap
    tidak.
    """
    from pipelines.graph.alamat import normalisasi_alamat as norm

    rel = read_table("silver", "sl_icij_alamat_terpilih")
    node = read_table("silver", "sl_icij_address", columns=["node_id", "address"])
    peta_cif = read_table("silver", "sl_peta_cif", columns=["cif_sk", "node_id"])

    df = rel.copy()
    df["cif_sk"] = df["node_id_start"].map(peta_cif.set_index("node_id")["cif_sk"])
    df = df.dropna(subset=["cif_sk"])
    df["asli"] = df["node_id_end"].map(node.set_index("node_id")["address"])
    df = df[df["asli"].notna()]
    df["kunci_asli"] = norm(df["asli"])
    df = df[df["kunci_asli"].str.len() > 0]

    kelompok_asli = df.groupby("kunci_asli")["cif_sk"].apply(lambda s: frozenset(s.astype(int)))

    jembatan = read_table("gold", "fact_alamat_debitur")
    kelompok_gold = set(
        jembatan.groupby("alamat_id")["cif_sk"].apply(lambda s: frozenset(s.astype(int)))
    )
    hilang = [k for k in kelompok_asli if len(k) > 1 and k not in kelompok_gold]
    assert not hilang, f"{len(hilang)} kelompok se-alamat pecah setelah penyamaran"


# --------------------------------------------------------------- lapisan graf
def test_alamat_hadir_sebagai_simpul_graf(dim_alamat):
    nodes = read_table("gold", "gold_graph_nodes")
    alamat = nodes[nodes["node_type"] == "alamat"]
    assert len(alamat) == len(dim_alamat)
    assert set(alamat["ref_id"]) == set(dim_alamat["alamat_id"])


def test_edge_beralamat_di_menyambung_debitur_ke_alamat():
    nodes = read_table("gold", "gold_graph_nodes")
    edges = read_table("gold", "gold_graph_edges")
    beralamat = edges[edges["rel_type"] == "beralamat_di"]
    assert len(beralamat), "tidak ada edge beralamat_di"

    tipe = nodes.set_index("node_id")["node_type"]
    assert (beralamat["src_node_id"].map(tipe) == "badan_hukum").all()
    assert (beralamat["dst_node_id"].map(tipe) == "alamat").all()


def test_edge_alamat_tidak_mengubah_fitur_berbagi_atribut():
    """beralamat_di adalah rel_type baru, bukan berbagi_atribut yang dilebarkan.

    shared_attribute_degree di fitur_pit.py menghitung berbagi_atribut. Kalau
    edge debitur->alamat ikut masuk ke sana, derajat tiap debitur naik satu
    tanpa ada keterkaitan baru dan fitur lama diam-diam bergeser artinya.
    """
    edges = read_table("gold", "gold_graph_edges")
    nodes = read_table("gold", "gold_graph_nodes")
    tipe = nodes.set_index("node_id")["node_type"]
    atribut = edges[edges["rel_type"] == "berbagi_atribut"]
    assert (atribut["src_node_id"].map(tipe) == "badan_hukum").all()
    assert (atribut["dst_node_id"].map(tipe) == "badan_hukum").all()


# ------------------------------------------- injeksi tetap tidak bisa dibedakan
@pytest.mark.skipif(
    not table_exists("gold", "fact_afiliasi_tersembunyi"),
    reason="injeksi afiliasi dimatikan (INJEKSI_AFILIASI=0)",
)
class TestAfiliasiLewatAlamat:
    @pytest.fixture(scope="class")
    def klaster_alamat(self) -> pd.DataFrame:
        klaster = read_table("gold", "fact_afiliasi_tersembunyi")
        return klaster[klaster["mekanisme"] == "alamat_operasional_bersama"]

    def test_anggota_klaster_punya_alamat_bersama(self, klaster_alamat, jembatan):
        """Seluruh anggota satu klaster harus berbagi setidaknya satu alamat_id.

        Satu cif boleh punya lebih dari satu alamat (alamat ICIJ lamanya tetap
        ada), jadi yang diuji irisannya, bukan alamat tunggalnya.
        """
        if klaster_alamat.empty:
            pytest.skip("tidak ada klaster bermekanisme alamat")
        peta = jembatan.groupby("cif_sk")["alamat_id"].apply(set).to_dict()
        for afiliasi_id, sub in klaster_alamat.groupby("afiliasi_id"):
            irisan: set | None = None
            for cif in sub["cif_sk"]:
                punya = peta.get(int(cif), set())
                irisan = punya if irisan is None else (irisan & punya)
            assert irisan, f"klaster {afiliasi_id} tidak punya alamat bersama"

    def test_alamat_klaster_tidak_bertanda(self, klaster_alamat, dim_alamat, jembatan):
        """DIM_ALAMAT tidak boleh punya kolom yang menyebut afiliasi.

        Ground truth-nya hidup di FACT_AFILIASI_TERSEMBUNYI. Kalau alamat hasil
        injeksi bisa dikenali dari kolomnya sendiri, deteksi cuma perlu membaca
        kolom itu dan seluruh evaluasi menjadi hampa.
        """
        jejak = [
            c
            for c in dim_alamat.columns
            if any(k in c for k in ("afiliasi", "klaster", "peran", "mekanisme"))
        ]
        assert not jejak, f"kolom berjejak afiliasi di dim_alamat: {jejak}"
        assert not [c for c in jembatan.columns if "afiliasi" in c]

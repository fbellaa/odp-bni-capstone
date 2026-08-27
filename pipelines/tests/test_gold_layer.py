"""Uji integrasi atas layer gold yang sudah dibangun.

Otomatis di-skip bila data/gold belum diisi, supaya `pytest` tetap hijau di mesin
yang belum menjalankan pipeline.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from pipelines.config import settings
from pipelines.utils import read_table, table_exists

pytestmark = pytest.mark.skipif(
    not table_exists("gold", "feat_graf_pit"),
    reason="layer gold belum dibangun - jalankan python -m pipelines.flows.main_flow",
)


@pytest.fixture(scope="module")
def pengajuan() -> pd.DataFrame:
    return read_table("gold", "fact_pengajuan")


@pytest.fixture(scope="module")
def feat() -> pd.DataFrame:
    return read_table("gold", "feat_graf_pit")


@pytest.fixture(scope="module")
def edges() -> pd.DataFrame:
    return read_table("gold", "gold_graph_edges")


# ------------------------------------------------------------ anti-kebocoran
def test_fitur_graf_memakai_snapshot_bulan_sebelumnya(feat, pengajuan):
    gabung = feat.merge(pengajuan[["application_id", "tanggal_pengajuan"]], on="application_id")
    awal_bulan_t = gabung["tanggal_pengajuan"].values.astype("datetime64[M]")
    assert (gabung["snapshot_date"].to_numpy().astype("datetime64[ns]") < awal_bulan_t).all()


def test_tidak_ada_edge_dengan_valid_from_setelah_snapshot(feat, edges):
    """Uji wajib §7.4: edge masa depan tidak boleh ikut ke snapshot mana pun."""
    snapshot = pd.to_datetime(read_table("gold", "graph_snapshot_bulanan", columns=["snapshot_date"])["snapshot_date"])
    for tanggal in sorted(snapshot.unique())[:3]:
        tanggal = pd.Timestamp(tanggal)
        aktif = edges[
            (edges["valid_from"] <= tanggal)
            & (edges["valid_to"].isna() | (edges["valid_to"] > tanggal))
        ]
        assert (aktif["valid_from"] <= tanggal).all()

        g = nx.Graph()
        g.add_edges_from(zip(aktif["src_node_id"], aktif["dst_node_id"]))
        tersimpan = read_table("gold", "graph_snapshot_bulanan")
        tersimpan = tersimpan[tersimpan["snapshot_date"] == tanggal].set_index("node_id")["degree"]
        contoh = tersimpan[tersimpan > 0].head(200)
        dihitung_ulang = pd.Series(dict(g.degree()), dtype="float64").reindex(contoh.index).fillna(0)
        pd.testing.assert_series_equal(
            contoh.astype("float64"), dihitung_ulang, check_names=False
        )


def test_feat_graf_pit_tidak_memuat_label(feat):
    assert "label_default" not in feat.columns
    assert "src_is_laundering" not in feat.columns


def test_blok_graf_bisa_didrop_utuh(feat):
    """Uji ablasi §7.3 hanya bermakna kalau FEAT_GRAF_PIT terpisah fisik."""
    lk = read_table("gold", "fact_laporan_keuangan")
    tumpang_tindih = set(feat.columns) & set(lk.columns) - {"cif_sk", "application_id"}
    assert not tumpang_tindih


# ------------------------------------------------------------ bentuk data
def test_panel_tiga_tahun_per_debitur():
    lk = read_table("gold", "fact_laporan_keuangan", columns=["cif_sk", "tahun_buku"])
    assert (lk.groupby("cif_sk")["tahun_buku"].nunique() == settings.panel_years).all()
    assert lk["tahun_buku"].max() == settings.tahun_buku_terakhir


def test_lgd_realisasi_berasal_dari_sba_dan_bervariasi():
    default = read_table("gold", "fact_default")
    if default.empty:
        pytest.skip("belum ada fasilitas default pada jendela observasi")
    assert default["lgd_realisasi"].between(0, 1).all()
    # LGD tetap harus punya sebaran - kalau konstan berarti kembali ke asumsi.
    assert default["lgd_realisasi"].nunique() > 5
    assert default["src_sba_loannr"].notna().any()


def test_skala_rupiah_segmen_komersial():
    debitur = read_table("gold", "dim_debitur")
    kini = debitur[debitur["is_current"]]
    assert kini["penjualan_rp"].between(
        settings.penjualan_min_rp * 0.9, settings.penjualan_max_rp * 1.1
    ).all()


def test_scd2_tidak_tumpang_tindih():
    dim = read_table("gold", "dim_debitur").sort_values(["cif_sk", "valid_from"])
    berlapis = dim[dim.duplicated("cif_sk", keep=False)]
    for _, sub in berlapis.groupby("cif_sk"):
        assert len(sub) == 2
        lama, baru = sub.iloc[0], sub.iloc[1]
        assert lama["valid_to"] < baru["valid_from"]
        assert not lama["is_current"] and baru["is_current"]


def test_setiap_debitur_punya_provenance():
    dim = read_table("gold", "dim_debitur")
    for kolom in ("src_us_company", "src_icij_node_id", "src_taiwan_row_id"):
        assert dim[kolom].notna().all(), f"{kolom} kosong - provenance join wajib tercatat"


def test_embedding_graf_terkanonikkan():
    """Regresi: 16 kolom embedding sempat berubah tiap pipeline dijalankan ulang.

    ARPACK (scipy svds) memberi tanda sembarang dan urutan nilai singular menaik,
    jadi embedding bergeser tanpa datanya berubah sedikit pun. Tanpa
    kanonikalisasi, data scientist melatih di fitur yang diam-diam berbeda dari
    yang dipakai kemarin.

    Diuji di GRAPH_SNAPSHOT_BULANAN, bukan FEAT_GRAF_PIT: kanonikalisasi berlaku
    atas seluruh simpul graf, sedangkan FEAT_GRAF_PIT hanya memuat baris debitur.
    """
    snap = read_table("gold", "graph_snapshot_bulanan")
    kolom = [c for c in snap.columns if c.startswith("emb_")]
    assert len(kolom) == 16

    satu = snap[snap["snapshot_date"] == snap["snapshot_date"].min()]
    for c in kolom:
        v = satu[c].dropna()
        if len(v):
            assert v.loc[v.abs().idxmax()] > 0, f"tanda {c} belum dikanonikkan"

    # Nilai singular menurun: ragam kolom pertama >= kolom terakhir.
    ragam = satu[kolom].var()
    assert ragam.iloc[0] >= ragam.iloc[-1], "urutan embedding tidak menurun"


def test_parameter_build_tercatat_dan_cocok():
    """Regresi: `.env` yang tertinggal sempat menimpa N_DEBITUR tanpa terdeteksi.

    Kode di 6000, `.env` di 3000 -> populasi separuh, kejadian di uji OOT turun
    dari 14 ke 3, seluruh angka ablasi bergeser. Gejalanya menyamar sebagai
    pipeline yang tidak deterministik, padahal pipeline-nya reproducible dan
    yang berubah parameternya.
    """
    par = read_table("gold", "parameter_build")
    nilai = par.set_index("parameter")["nilai"]

    assert str(nilai["n_debitur"]) == str(settings.n_debitur)
    assert str(nilai["seed"]) == str(settings.seed)

    debitur = read_table("gold", "dim_debitur", columns=["cif_sk", "is_current"])
    assert int(debitur["is_current"].sum()) == settings.n_debitur

    # Kolom `sumber` adalah inti tabel ini - ia menandai penimpaan dari .env.
    assert set(par["sumber"].str.split(":").str[0]) <= {
        "default_kode",
        "env",
        "turunan",
        "lingkungan",
    }
    assert "git_commit" in par["parameter"].values

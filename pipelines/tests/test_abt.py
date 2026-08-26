"""Uji paket serah terima ke data scientist.

Yang dijaga di sini adalah hal-hal yang kalau lolos diam-diam akan menghasilkan
AUC bagus tapi palsu: kolom masa depan, target pada baris tersensor, dan split
yang beririsan.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines.config import settings
from pipelines.exports.abt import HORIZON_EWS_BULAN, HORIZON_PD_BULAN, KOLOM_TERLARANG
from pipelines.utils import read_table, table_exists

pytestmark = pytest.mark.skipif(
    not table_exists("gold", "abt_pd"),
    reason="ABT belum dibangun - jalankan python -m pipelines.flows.main_flow",
)


@pytest.fixture(scope="module")
def abt_pd() -> pd.DataFrame:
    return read_table("gold", "abt_pd")


@pytest.fixture(scope="module")
def abt_ews() -> pd.DataFrame:
    return read_table("gold", "abt_ews")


# --------------------------------------------------------------- kolom bocor
@pytest.mark.parametrize("nama", ["abt_pd", "abt_ews", "abt_lgd"])
def test_tidak_ada_kolom_terlarang(nama):
    df = read_table("gold", nama)
    bocor = {c for c in df.columns if c in KOLOM_TERLARANG}
    bocor |= {c for c in df.columns if c.endswith(tuple(KOLOM_TERLARANG))}
    assert not bocor, f"{nama} membocorkan {sorted(bocor)}"


def test_abt_pd_tidak_memuat_perilaku_pasca_pencairan(abt_pd):
    """outstanding, pemakaian plafon, dan kolektibilitas adalah masa depan bagi PD."""
    terlarang = ("outstanding", "pemakaian_plafon", "kolektibilitas", "dpd", "tanggal_default")
    bocor = [c for c in abt_pd.columns if any(t in c for t in terlarang)]
    assert not bocor, f"kolom pasca-pencairan bocor ke abt_pd: {bocor}"


def test_abt_pd_snapshot_graf_mendahului_pengajuan(abt_pd):
    awal_bulan_t = abt_pd["tanggal_pengajuan"].values.astype("datetime64[M]")
    assert (abt_pd["snapshot_date"].to_numpy().astype("datetime64[ns]") < awal_bulan_t).all()


# ---------------------------------------------------------------- target
def test_target_pd_konsisten_dengan_sensor(abt_pd):
    tersensor = abt_pd["y_tersensor"]
    assert abt_pd.loc[tersensor, "y_default_12bln"].isna().all()
    assert abt_pd.loc[~tersensor, "y_default_12bln"].notna().all()
    assert abt_pd["y_default_12bln"].dropna().isin([0.0, 1.0]).all()


def test_target_pd_sesuai_horizon(abt_pd):
    """y=1 hanya kalau default benar-benar terjadi dalam 12 bulan sejak pencairan."""
    positif = abt_pd[abt_pd["y_default_12bln"] == 1]
    batas = positif["tanggal_pencairan"] + pd.DateOffset(months=HORIZON_PD_BULAN)
    umur = positif["y_umur_hari"]
    assert (umur > 0).all()
    assert (positif["tanggal_pencairan"] + pd.to_timedelta(umur, unit="D") <= batas).all()


def test_bad_rate_cukup_untuk_dilatih(abt_pd):
    """Regresi: pemangkasan hari_ke_default sempat membuat bad rate 0,15%."""
    dapat_dilatih = abt_pd["y_default_12bln"].dropna()
    assert len(dapat_dilatih) >= 1000
    assert dapat_dilatih.sum() >= 30, "kejadian positif terlalu sedikit untuk melatih PD"
    assert 0.01 <= dapat_dilatih.mean() <= 0.20


def test_umur_default_tersebar_bukan_menumpuk():
    """Kalau default menumpuk di ujung jendela, artinya kembali ke pemangkasan."""
    fasilitas = read_table("gold", "fact_fasilitas", columns=["facility_id", "tanggal_pencairan"])
    default = read_table("gold", "fact_default", columns=["facility_id", "tanggal_default"])
    umur = default.merge(fasilitas, on="facility_id")
    hari = (umur["tanggal_default"] - umur["tanggal_pencairan"]).dt.days
    assert hari.min() >= 60
    # Sebaran nyata SBA punya ekor panjang; setelah diskala harus tetap lebar.
    assert hari.quantile(0.9) - hari.quantile(0.1) > 180


# ------------------------------------------------------------------- split
def test_split_out_of_time_tidak_beririsan(abt_pd):
    latih = set(abt_pd[abt_pd["split"] == "latih"]["cif_sk"])
    uji = set(abt_pd[abt_pd["split"] == "uji_oot"]["cif_sk"])
    assert not (latih & uji)
    assert abt_pd[abt_pd["split"] == "latih"]["tanggal_pengajuan"].max() < abt_pd[
        abt_pd["split"] == "uji_oot"
    ]["tanggal_pengajuan"].min()


# --------------------------------------------------------------------- EWS
def test_ews_tidak_memuat_baris_setelah_default(abt_ews):
    default = read_table("gold", "fact_default", columns=["facility_id", "tanggal_default"])
    gabung = abt_ews.merge(default, on="facility_id", how="left")
    sudah_default = gabung["tanggal_default"].notna()
    assert (gabung.loc[sudah_default, "snapshot_date"] < gabung.loc[sudah_default, "tanggal_default"]).all()


def test_ews_target_forward_looking(abt_ews):
    """y=0 hanya boleh kalau 6 bulan ke depan benar-benar teramati penuh.

    Kalau defaultnya sudah terjadi, y=1 tetap sah walau horizonnya melewati akhir
    observasi - kejadiannya sudah terlihat, tidak ada yang perlu ditunggu.
    """
    batas = pd.Timestamp(settings.snapshot_akhir)
    akhir_horizon = abt_ews["snapshot_date"] + pd.DateOffset(months=HORIZON_EWS_BULAN)

    negatif = abt_ews["y_default_6bln"] == 0
    assert (akhir_horizon[negatif] <= batas).all()
    assert abt_ews.loc[abt_ews["y_tersensor"], "y_default_6bln"].isna().all()


# ------------------------------------------------------------------- kamus
def test_kamus_menutup_semua_kolom(abt_pd, abt_ews):
    kamus = read_table("gold", "kamus_data_abt")
    for nama, df in (("abt_pd", abt_pd), ("abt_ews", abt_ews)):
        terdaftar = set(kamus[kamus["abt"] == nama]["kolom"])
        assert terdaftar == set(df.columns), f"kamus {nama} tidak sinkron dengan tabelnya"


# ------------------------------------------------------- audit sinyal (regresi)
def test_blok_taiwan_tidak_kembali_ke_abt_pd(abt_pd):
    """Kolom tw_* dicocokkan MEMAKAI label, jadi membawa target ke dalam fitur.

    Diukur saat audit: 4 kolom tw_ saja memberi AUC out-of-time 0,826 - lebih
    tinggi dari 29 kolom rasio nyata (0,662). Itu kebocoran lewat kunci
    pencocokan langkah 2, bukan sinyal keuangan.
    """
    bocor = [c for c in abt_pd.columns if "tw_" in c]
    assert not bocor, f"blok Taiwan kembali masuk abt_pd: {bocor}"


def test_kolektibilitas_bukan_prediktor_deterministik(abt_ews):
    """Kalau P(default | kol=3) = 1,000 maka model EWS cuma menghafal generator."""
    dilatih = abt_ews[abt_ews["y_default_6bln"].notna()]
    kol3 = dilatih[dilatih["perilaku_kolektibilitas"] == 3]["y_default_6bln"]
    assert len(kol3) >= 20, "terlalu sedikit observasi kol-3 untuk menilai"
    assert kol3.mean() < 0.9, (
        f"kol-3 memprediksi default dengan probabilitas {kol3.mean():.3f} - "
        "episode tekanan yang pulih (cured) hilang dari generator"
    )


def test_ada_fasilitas_kol3_yang_pulih():
    """Debitur yang jatuh ke kol 3 lalu pulih harus ada - itu kasus paling menarik."""
    kolek = read_table("gold", "fact_kolektibilitas")
    default = read_table("gold", "fact_default", columns=["facility_id"])
    pernah_buruk = set(kolek[kolek["kolektibilitas"] >= 3]["facility_id"])
    pulih = pernah_buruk - set(default["facility_id"])
    assert pulih, "tidak ada satu pun fasilitas yang memburuk lalu pulih"


def test_tabel_ditolak_seruang_fitur_dengan_abt_pd(abt_pd):
    """Reject inference mustahil kalau kedua populasi tidak bisa diskor model sama."""
    ditolak = read_table("gold", "abt_pengajuan_ditolak")
    fitur_pd = {c for c in abt_pd.columns if c.startswith(("fin_", "app_", "graf_"))}
    fitur_ditolak = {c for c in ditolak.columns if c.startswith(("fin_", "app_", "graf_"))}
    assert fitur_pd == fitur_ditolak, (
        f"selisih fitur: hanya di abt_pd {sorted(fitur_pd - fitur_ditolak)}, "
        f"hanya di ditolak {sorted(fitur_ditolak - fitur_pd)}"
    )
    assert "y_default_12bln" not in ditolak.columns


def test_abt_lgd_sumber_populasi_nyata():
    """Data latih LGD harus populasi SBA penuh, bukan 75 baris sintetis."""
    sumber = read_table("gold", "abt_lgd_sumber")
    assert len(sumber) > 100_000
    assert sumber["y_lgd_realisasi"].between(0, 1).all()
    assert set(sumber["split"]) == {"latih", "uji_oot"}
    assert sumber["y_lgd_realisasi"].std() > 0.1


def test_kamus_menandai_fitur_derau_pada_abt_lgd():
    """Feature importance abt_lgd gampang disalahtafsirkan - kamus wajib menandai."""
    kamus = read_table("gold", "kamus_data_abt")
    lgd = kamus[kamus["abt"] == "abt_lgd"]
    fitur = lgd[lgd["blok"].isin(["app", "fin"])]
    assert len(fitur) > 0
    assert fitur["catatan"].str.contains("DERAU").any()
    assert fitur["catatan"].str.contains("sinyal nyata").any()

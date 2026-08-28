"""Uji paket serah terima ke data scientist.

Yang dijaga di sini adalah hal-hal yang kalau lolos diam-diam akan menghasilkan
AUC bagus tapi palsu: kolom masa depan, target pada baris tersensor, dan split
yang beririsan.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines.config import settings
from pipelines.exports.abt import (
    FITUR_LGD_TERAPAN,
    HORIZON_EWS_BULAN,
    HORIZON_PD_BULAN,
    KOLOM_TERLARANG,
)
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


@pytest.mark.parametrize("nama", ["abt_pd", "abt_pengajuan_ditolak"])
def test_neraca_sepadan_dengan_penjualan(nama):
    """Tidak ada baris beraset puluhan kali omzetnya.

    Bukan sekadar soal angka jelek: rasio yang dipakai model dihitung langsung
    dari kolom level ini, jadi baris seperti itu menyuntik profil distress
    ekstrem ke seluruh blok fin_ tanpa outcome yang mengikutinya.
    """
    df = read_table("gold", nama)
    penjualan = df["fin_penjualan_rp"]
    rasio = df["fin_total_aset_rp"] / penjualan.where(penjualan > 0)
    tak_wajar = rasio > settings.aset_thd_penjualan_maks
    assert not tak_wajar.any(), (
        f"{nama}: {int(tak_wajar.sum())} baris beraset > "
        f"{settings.aset_thd_penjualan_maks}x penjualan (maks {rasio.max():.1f}x)"
    )


def test_abt_lgd_sumber_populasi_nyata():
    """Data latih LGD harus populasi SBA penuh, bukan 75 baris sintetis."""
    sumber = read_table("gold", "abt_lgd_sumber")
    assert len(sumber) > 100_000
    assert sumber["y_lgd_realisasi"].between(0, 1).all()
    assert set(sumber["split"]) == {"latih", "uji_oot"}
    assert sumber["y_lgd_realisasi"].std() > 0.1


def test_lgd_bergerak_mengikuti_agunan():
    """LGD harus sensitif terhadap agunan - tapi bukan tulisan ulang coverage.

    Dulu korelasinya -0,05: model LGD tidak bisa menjawab "berapa kerugian
    kalau coverage dinaikkan", padahal itu pertanyaan bisnis yang paling
    sering datang. Batas bawah -0,70 menjaga agar perbaikannya tidak berlebihan
    sampai LGD jadi fungsi deterministik agunan.
    """
    lgd = read_table("gold", "abt_lgd")
    cov_ead = lgd["app_coverage_ratio"] / lgd["app_ead_thd_plafon"]
    korelasi = lgd["y_lgd_realisasi"].corr(cov_ead)
    assert -0.70 <= korelasi <= -0.20, (
        f"corr(LGD, coverage terhadap EAD) = {korelasi:+.3f}, di luar [-0,70; -0,20]"
    )


def test_fitur_sba_masih_memprediksi_lgd_di_portofolio():
    """Pemetaan peringkat tidak boleh memutus transfer SBA -> portofolio.

    Ini penjaga terhadap kesalahan yang pernah terjadi: membentuk urutan LGD
    dari agunan SAJA mengacak urutan SBA sampai model yang dilatih di 156.610
    pinjaman SBA jatuh ke R2 -0,43 saat diterapkan ke portofolio - lebih buruk
    daripada menebak rata-rata. Data latih LGD tidak punya kolom agunan, jadi
    kalau korelasi ini hilang, tabel penerapan menjadi tidak bisa diskor sama
    sekali oleh model yang bisa dilatih.

    Yang diuji versi murahnya: peringkat LGD portofolio harus masih berkorelasi
    dengan peringkat LGD SBA asalnya, tanpa perlu melatih model apa pun.
    """
    lgd = read_table("gold", "abt_lgd")
    sba = read_table("silver", "sl_peta_sba", columns=["src_sba_loannr", "lgd_realisasi"])
    gabung = lgd.merge(
        sba.rename(columns={"lgd_realisasi": "lgd_sba"}), on="src_sba_loannr", how="inner"
    )
    korelasi = gabung["y_lgd_realisasi"].corr(gabung["lgd_sba"], method="spearman")
    assert korelasi > 0.35, (
        f"korelasi peringkat LGD portofolio vs SBA = {korelasi:+.3f}; "
        "di bawah ini fitur SBA tidak lagi bisa memprediksi LGD portofolio"
    )


def test_lgd_marginal_tidak_bergeser_dari_sba():
    """Agunan menentukan URUTAN, SBA tetap menentukan sebaran nilainya."""
    default = read_table("gold", "fact_default")
    sba = read_table("silver", "sl_peta_sba", columns=["cif_sk", "lgd_realisasi"])
    sumber = sba[sba["cif_sk"].isin(default["cif_sk"])]["lgd_realisasi"].mean()
    assert abs(default["lgd_realisasi"].mean() - sumber) < 0.02, (
        f"rerata LGD {default['lgd_realisasi'].mean():.4f} vs sumber SBA {sumber:.4f}"
    )


def test_abt_lgd_tanpa_kolom_split():
    """Seluruh abt_lgd adalah data uji; kolom split memancing latih di test set."""
    lgd = read_table("gold", "abt_lgd")
    assert "split" not in lgd.columns
    sumber = read_table("gold", "abt_lgd_sumber")
    tumpang = set(lgd["src_sba_loannr"]) & set(sumber["sba_loan_nr"])
    assert len(tumpang) == len(lgd), (
        "setiap baris abt_lgd harus bisa dikeluarkan dari data latih lewat "
        f"src_sba_loannr; yang cocok {len(tumpang)} dari {len(lgd)}"
    )


def test_kamus_menandai_fitur_derau_pada_abt_lgd():
    """Feature importance abt_lgd gampang disalahtafsirkan - kamus wajib menandai."""
    kamus = read_table("gold", "kamus_data_abt")
    lgd = kamus[kamus["abt"] == "abt_lgd"]
    fitur = lgd[lgd["blok"].isin(["app", "fin"])]
    assert len(fitur) > 0
    assert fitur["catatan"].str.contains("DERAU").any()
    assert fitur["catatan"].str.contains("sinyal nyata").any()


# ------------------------------------------ injeksi afiliasi tersembunyi (§7)
@pytest.mark.skipif(
    not table_exists("gold", "fact_afiliasi_tersembunyi"),
    reason="injeksi afiliasi dimatikan (INJEKSI_AFILIASI=0)",
)
class TestAfiliasiTersembunyi:
    """Penjaga langkah 7. Ground truth-nya hanya untuk evaluasi deteksi."""

    def test_ground_truth_tidak_bocor_ke_abt(self, abt_pd):
        jejak = [
            c
            for c in abt_pd.columns
            if any(k in c for k in ("afiliasi", "klaster", "peran", "mekanisme"))
        ]
        assert not jejak, f"ground truth afiliasi bocor ke abt_pd: {jejak}"

    def test_edge_injeksi_tidak_ditandai(self):
        """Kalau edge injeksi punya penanda sendiri, ia tidak lagi tersembunyi."""
        edges = read_table("gold", "gold_graph_edges")
        assert set(edges["sumber"].unique()) <= {"icij", "aml"}
        assert not [c for c in edges.columns if "afiliasi" in c]

    def test_klaster_melintasi_grup_usaha(self):
        """Afiliasi tersembunyi harus melintasi grup - kalau tidak, ia cuma grup."""
        klaster = read_table("gold", "fact_afiliasi_tersembunyi")
        debitur = read_table("gold", "dim_debitur", columns=["cif_sk", "grup_id", "is_current"])
        grup = debitur[debitur["is_current"]].set_index("cif_sk")["grup_id"]
        per_klaster = klaster.assign(g=klaster["cif_sk"].map(grup)).groupby("afiliasi_id")["g"]
        assert (per_klaster.nunique() > 1).all()

    def test_penularan_hanya_terlihat_ke_belakang(self):
        """Sumber wajib jatuh sebelum anggota terinfeksi mengajukan kredit."""
        klaster = read_table("gold", "fact_afiliasi_tersembunyi")
        default = read_table("gold", "fact_default", columns=["cif_sk", "tanggal_default"])
        pengajuan = read_table("gold", "fact_pengajuan", columns=["cif_sk", "tanggal_pengajuan"])

        sumber = (
            klaster[klaster["peran"] == "sumber"]
            .merge(default, on="cif_sk")
            .groupby("afiliasi_id")["tanggal_default"]
            .min()
        )
        infeksi = (
            klaster[klaster["peran"] == "terinfeksi"]
            .merge(pengajuan, on="cif_sk")
            .groupby("afiliasi_id")["tanggal_pengajuan"]
            .min()
        )
        bersama = sumber.index.intersection(infeksi.index)
        assert len(bersama) > 0
        assert (sumber[bersama] < infeksi[bersama]).all()

    def test_komposisi_klaster_sesuai_spesifikasi(self):
        klaster = read_table("gold", "fact_afiliasi_tersembunyi")
        per_peran = klaster.groupby(["afiliasi_id", "peran"]).size().unstack(fill_value=0)
        assert (per_peran["sumber"] == settings.afiliasi_default_per_klaster).all()
        assert (per_peran["terinfeksi"] == settings.afiliasi_default_per_klaster).all()
        assert (per_peran["sehat"] == settings.afiliasi_sehat_per_klaster).all()


def test_kamus_menandai_perlakuan_nan():
    """NaN pada debt_to_ebitda bermakna (EBITDA <= 0) - kamus wajib memperingatkan.

    Tanpa peringatan ini, fillna(median) menghapus sinyal sekaligus memberi
    debitur ber-EBITDA negatif angka rasio yang tampak sehat.
    """
    kamus = read_table("gold", "kamus_data_abt")
    pd_kamus = kamus[kamus["abt"] == "abt_pd"].set_index("kolom")["catatan"]

    assert "NaN BERMAKNA" in pd_kamus.get("fin_debt_to_ebitda", "")
    assert "tidak ada relasi" in pd_kamus.get("graf_supplier_concentration_hhi", "")
    assert "tersensor kanan" in pd_kamus.get("y_default_12bln", "")
    assert "y_umur_teramati_hari" in pd_kamus.get("y_umur_hari", "")


def test_nan_debt_to_ebitda_memang_ebitda_nonpositif(abt_pd):
    """Kalau NaN-nya berasal dari sebab lain, catatan di kamus jadi menyesatkan."""
    lk = read_table("gold", "fact_laporan_keuangan")
    akhir = lk[lk["is_tahun_terakhir"]].set_index("cif_sk")["ebitda_rp"]
    kosong = abt_pd.set_index("cif_sk")["fin_debt_to_ebitda"].isna()
    ebitda = akhir.reindex(kosong.index)
    assert (ebitda[kosong] <= 0).all(), "ada NaN debt_to_ebitda dengan EBITDA positif"


def test_ruang_fitur_lgd_sejajar():
    """Model LGD dilatih di satu tabel dan dipanggil di tabel lain.

    Regresi: abt_lgd_sumber sempat memuat 6 fitur yang tidak ada di abt_lgd
    (nilai USD, jumlah pegawai, negara bagian Amerika), sehingga model yang
    dilatih dengan seluruh fitur gagal saat predict() dipanggil pada portofolio.
    """
    sumber = read_table("gold", "abt_lgd_sumber")
    terap = read_table("gold", "abt_lgd")

    fitur_latih = {c for c in sumber.columns if c.startswith("app_")}
    fitur_terap = {c for c in terap.columns if c.startswith("app_")}

    assert not (fitur_latih - fitur_terap), (
        "abt_lgd_sumber memuat fitur yang tidak ada saat menerapkan: "
        f"{sorted(fitur_latih - fitur_terap)}"
    )
    for kolom in FITUR_LGD_TERAPAN:
        assert kolom in sumber.columns, f"{kolom} hilang dari data latih"
        assert kolom in terap.columns, f"{kolom} hilang dari data terapan"


def test_model_lgd_bisa_dilatih_lalu_diterapkan():
    """Uji ujung ke ujung: latih di sumber, panggil di portofolio."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    sumber = read_table("gold", "abt_lgd_sumber")
    terap = read_table("gold", "abt_lgd")

    kat = [c for c in FITUR_LGD_TERAPAN if sumber[c].dtype == object or str(sumber[c].dtype) in ("string", "bool", "category")]
    num = [c for c in FITUR_LGD_TERAPAN if c not in kat]
    prep = ColumnTransformer(
        [
            ("k", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), kat),
            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ]
    )
    model = make_pipeline(prep, Ridge(alpha=1.0))
    latih = sumber[sumber["split"] == "latih"].head(20_000)
    model.fit(latih[FITUR_LGD_TERAPAN], latih["y_lgd_realisasi"])

    prediksi = model.predict(terap[FITUR_LGD_TERAPAN])
    assert len(prediksi) == len(terap)
    assert ((prediksi >= 0) & (prediksi <= 1.5)).all(), "prediksi LGD di luar rentang wajar"

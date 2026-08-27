"""Analytic Base Table - paket serah terima untuk data scientist.

Tiga ABT dibangun di sini:

| ABT          | Grain                      | Target                          |
|--------------|----------------------------|---------------------------------|
| `abt_pd`     | `application_id`           | `y_default_12bln`               |
| `abt_ews`    | `facility_id` x `snapshot` | `y_default_6bln` (ke depan)     |
| `abt_lgd`    | `facility_id` (yang default)| `y_lgd_realisasi`              |

Aturan yang dijaga modul ini:

1. **Prefiks blok**. Setiap kolom fitur diberi prefiks `fin_`, `app_`, `perilaku_`,
   atau `graf_`. Uji ablasi §7.3 karena itu cukup satu baris:
   `X.drop(columns=X.filter(like="graf_").columns)`.
2. **Tidak ada kolom masa depan**. `outstanding_rp`, `pemakaian_plafon_pct`, dan
   kolektibilitas adalah perilaku SETELAH pencairan - dilarang masuk `abt_pd`,
   dipakai di `abt_ews` hanya pada snapshot yang sedang dinilai.
3. **Sensor kanan jujur**. Fasilitas yang jendela observasinya belum genap 12
   bulan tidak ditandai `y=0`; targetnya `NA` dan `y_tersensor=True`.
4. **Split out-of-time**, bukan acak. Satu grup usaha bisa punya beberapa
   debitur yang berbagi fitur graf, jadi split acak bocor lintas grup.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipelines.config import GOLD_DIR, settings
from pipelines.utils import read_table, write_table

LOG = logging.getLogger("pipelines.exports")

HORIZON_PD_BULAN = 12
HORIZON_EWS_BULAN = 6
# Pengajuan sampai akhir kuartal ini dipakai melatih; sesudahnya untuk uji.
BATAS_SPLIT_OOT = pd.Timestamp("2025-09-30")

# Kolom yang tidak boleh pernah muncul di ABT mana pun sebagai fitur.
KOLOM_TERLARANG = {
    "label_default",
    "label_default_debitur",
    "status_label",
    "src_is_laundering",
    "skala_rupiah",
}


def _tandai_split(tanggal: pd.Series) -> pd.Series:
    return np.where(tanggal <= BATAS_SPLIT_OOT, "latih", "uji_oot")


def _blok(df: pd.DataFrame, prefiks: str, kecuali: set[str]) -> pd.DataFrame:
    """Beri prefiks blok pada semua kolom kecuali kunci."""
    return df.rename(
        columns={c: f"{prefiks}{c}" for c in df.columns if c not in kecuali}
    )


# --------------------------------------------------------------------- ABT PD
def build_abt_pd() -> pd.DataFrame:
    """Satu baris per pengajuan yang cair, dengan target PD 12 bulan."""
    pengajuan = read_table("gold", "fact_pengajuan")
    fasilitas = read_table("gold", "fact_fasilitas")
    lk = read_table("gold", "fact_laporan_keuangan")
    agunan = read_table("gold", "fact_agunan")
    feat = read_table("gold", "feat_graf_pit")
    debitur = read_table("gold", "dim_debitur")
    default = read_table("gold", "fact_default")
    produk = read_table("gold", "dim_produk_fasilitas")

    batas_observasi = pd.Timestamp(settings.snapshot_akhir)

    # ---- kerangka: hanya pengajuan yang benar-benar cair (ada outcome).
    # Pengajuan yang ditolak tidak punya outcome teramati; menandainya y=0 adalah
    # reject inference yang salah. Baris ditolak tetap diekspor terpisah.
    inti = fasilitas[
        ["facility_id", "application_id", "cif_sk", "tanggal_pencairan", "tenor_bulan"]
    ].merge(
        pengajuan[
            [
                "application_id",
                "tanggal_pengajuan",
                "produk_id",
                "plafon_diminta_rp",
                "keputusan",
                "pricing_bps",
                "komite_level",
                "dokumen_ringkas",
            ]
        ],
        on="application_id",
        how="left",
    )
    inti = inti.merge(
        fasilitas[["facility_id", "plafon_rp"]], on="facility_id", how="left"
    )

    # ---- target
    inti["akhir_horizon"] = inti["tanggal_pencairan"] + pd.DateOffset(
        months=HORIZON_PD_BULAN
    )
    inti = inti.merge(
        default[["facility_id", "tanggal_default", "lgd_realisasi", "ead_rp"]],
        on="facility_id",
        how="left",
    )
    default_dalam_horizon = inti["tanggal_default"].notna() & (
        inti["tanggal_default"] <= inti["akhir_horizon"]
    )
    horizon_belum_genap = inti["akhir_horizon"] > batas_observasi
    tersensor = horizon_belum_genap & ~default_dalam_horizon

    inti["y_default_12bln"] = np.where(
        tersensor, np.nan, default_dalam_horizon.astype(float)
    )
    inti["y_tersensor"] = tersensor
    inti["y_umur_hari"] = (inti["tanggal_default"] - inti["tanggal_pencairan"]).dt.days
    inti["y_umur_teramati_hari"] = np.where(
        inti["tanggal_default"].notna(),
        inti["y_umur_hari"],
        (batas_observasi - inti["tanggal_pencairan"]).dt.days,
    )
    inti["split"] = _tandai_split(inti["tanggal_pengajuan"])

    # ---- blok fin_: laporan keuangan tahun buku terakhir sebelum pengajuan
    lk_akhir = lk[lk["is_tahun_terakhir"]].copy()
    buang = {
        "lk_id",
        "tahun_buku",
        "is_tahun_terakhir",
        "angkatan",
        "sumber",
        "src_us_row_id",
        "src_taiwan_row_id",
    }
    buang |= KOLOM_TERLARANG
    # Kolom tw_* datang dari baris Taiwan yang dicocokkan MEMAKAI label gagal
    # bayar (langkah 2: kunci = label + kuintil DER + kuintil ROA). Jadi kolom
    # itu membawa informasi label yang disuntikkan oleh prosedur pencocokan,
    # bukan sinyal keuangan. Diukur: 4 kolom tw_ saja menghasilkan AUC out-of-time
    # 0,826 - lebih tinggi dari 29 kolom rasio nyata (0,662). Itu kebocoran.
    buang |= {c for c in lk_akhir.columns if c.startswith("tw_")}
    lk_akhir = lk_akhir[[c for c in lk_akhir.columns if c not in buang]]
    lk_akhir = _blok(lk_akhir, "fin_", {"cif_sk"})

    # ---- blok app_: karakteristik pengajuan + agunan
    cov = (
        agunan.groupby("facility_id")
        .agg(
            jumlah_agunan=("agunan_id", "size"),
            nilai_likuidasi_rp=("nilai_likuidasi_rp", "sum"),
            coverage_ratio=("coverage_ratio", "first"),
            ada_agunan_likuid=("jenis", lambda s: bool((s == "deposito").any())),
            ada_jaminan_silang=("dijaminkan_silang", "any"),
        )
        .reset_index()
    )
    cov = _blok(cov, "app_", {"facility_id"})

    profil = debitur[debitur["is_current"]][
        ["cif_sk", "sektor_kbli", "kelas_penjualan", "penjualan_rp", "tahun_berdiri",
         "rating_internal", "skor_kredit", "grup_id", "angkatan"]
    ].copy()
    tahun_acuan = profil["angkatan"].map(
        {k: v["tahun_buku_terakhir"] for k, v in settings.angkatan.items()}
    )
    profil["umur_perusahaan_tahun"] = tahun_acuan - profil["tahun_berdiri"]
    profil = _blok(profil, "app_", {"cif_sk", "grup_id", "angkatan"})

    inti = inti.merge(produk, on="produk_id", how="left")
    inti = inti.rename(
        columns={
            "plafon_diminta_rp": "app_plafon_diminta_rp",
            "plafon_rp": "app_plafon_rp",
            "tenor_bulan": "app_tenor_bulan",
            "keputusan": "app_keputusan",
            "pricing_bps": "app_pricing_bps",
            "komite_level": "app_komite_level",
            "dokumen_ringkas": "app_dokumen_ringkas",
            "nama_produk": "app_nama_produk",
            "jenis_fasilitas": "app_jenis_fasilitas",
            "revolving": "app_revolving",
        }
    )

    # ---- blok graf_: seluruh FEAT_GRAF_PIT, siap di-drop utuh
    graf = _blok(
        feat.drop(columns=["cif_sk"]), "graf_", {"application_id", "snapshot_date"}
    )

    abt = (
        inti.merge(profil, on="cif_sk", how="left")
        .merge(lk_akhir, on="cif_sk", how="left")
        .merge(cov, on="facility_id", how="left")
        .merge(graf, on="application_id", how="left")
    )

    abt["app_plafon_thd_penjualan"] = abt["app_plafon_rp"] / abt["app_penjualan_rp"]

    kunci = [
        "application_id",
        "facility_id",
        "cif_sk",
        "grup_id",
        "angkatan",
        "tanggal_pengajuan",
        "tanggal_pencairan",
        "snapshot_date",
        "split",
    ]
    target = [
        "y_default_12bln",
        "y_tersensor",
        "y_umur_hari",
        "y_umur_teramati_hari",
    ]
    fitur = [c for c in abt.columns if c.startswith(("fin_", "app_", "graf_"))]
    abt = abt[kunci + sorted(fitur) + target]

    _pastikan_bersih(abt, "abt_pd")
    return abt.sort_values("tanggal_pengajuan").reset_index(drop=True)


def build_abt_pengajuan_ditolak(abt_pd: pd.DataFrame) -> pd.DataFrame:
    """Pengajuan yang ditolak - tanpa target, untuk analisis reject inference.

    Ruang fiturnya sengaja dibuat SAMA dengan `abt_pd`. Reject inference hanya
    bisa dikerjakan kalau baris ditolak dan baris cair bisa diskor oleh model
    yang sama; kolom yang lahir dari fasilitas (agunan, plafon final) memang
    tidak ada dan diisi NA.
    """
    pengajuan = read_table("gold", "fact_pengajuan")
    fasilitas = read_table("gold", "fact_fasilitas", columns=["application_id"])
    lk = read_table("gold", "fact_laporan_keuangan")
    feat = read_table("gold", "feat_graf_pit")
    debitur = read_table("gold", "dim_debitur")
    produk = read_table("gold", "dim_produk_fasilitas")

    ditolak = pengajuan[
        ~pengajuan["application_id"].isin(fasilitas["application_id"])
    ].copy()

    lk_akhir = lk[lk["is_tahun_terakhir"]].copy()
    buang = {
        "lk_id",
        "tahun_buku",
        "is_tahun_terakhir",
        "angkatan",
        "sumber",
        "src_us_row_id",
        "src_taiwan_row_id",
    }
    buang |= KOLOM_TERLARANG
    buang |= {c for c in lk_akhir.columns if c.startswith("tw_")}
    lk_akhir = _blok(
        lk_akhir[[c for c in lk_akhir.columns if c not in buang]], "fin_", {"cif_sk"}
    )

    profil = debitur[debitur["is_current"]][
        ["cif_sk", "sektor_kbli", "kelas_penjualan", "penjualan_rp", "tahun_berdiri",
         "rating_internal", "skor_kredit", "grup_id", "angkatan"]
    ].copy()
    tahun_acuan = profil["angkatan"].map(
        {k: v["tahun_buku_terakhir"] for k, v in settings.angkatan.items()}
    )
    profil["umur_perusahaan_tahun"] = tahun_acuan - profil["tahun_berdiri"]
    profil = _blok(profil, "app_", {"cif_sk", "grup_id", "angkatan"})

    graf = _blok(
        feat.drop(columns=["cif_sk"]), "graf_", {"application_id", "snapshot_date"}
    )

    ditolak = (
        ditolak.merge(produk, on="produk_id", how="left")
        .merge(profil, on="cif_sk", how="left")
        .merge(lk_akhir, on="cif_sk", how="left")
        .merge(graf, on="application_id", how="left")
    )
    ditolak = ditolak.rename(
        columns={
            "plafon_diminta_rp": "app_plafon_diminta_rp",
            "tenor_bulan": "app_tenor_bulan",
            "keputusan": "app_keputusan",
            "pricing_bps": "app_pricing_bps",
            "komite_level": "app_komite_level",
            "dokumen_ringkas": "app_dokumen_ringkas",
            "nama_produk": "app_nama_produk",
            "jenis_fasilitas": "app_jenis_fasilitas",
            "revolving": "app_revolving",
        }
    )
    ditolak["split"] = _tandai_split(ditolak["tanggal_pengajuan"])

    # Samakan kolom dengan abt_pd; yang hanya ada pada fasilitas cair diisi NA.
    for kolom in abt_pd.columns:
        if kolom not in ditolak.columns and kolom.startswith(("fin_", "app_", "graf_")):
            ditolak[kolom] = np.nan
    # facility_id dan tanggal_pencairan hanya ada pada pengajuan yang cair.
    tanpa = {"facility_id", "tanggal_pencairan"}
    kolom_akhir = [
        c for c in abt_pd.columns if not c.startswith("y_") and c not in tanpa
    ]
    return ditolak[kolom_akhir].reset_index(drop=True)


# -------------------------------------------------------------------- ABT EWS
def build_abt_ews() -> pd.DataFrame:
    """Panel bulanan fasilitas dengan target default 6 bulan ke depan."""
    kolek = read_table("gold", "fact_kolektibilitas")
    covenant = read_table("gold", "fact_covenant")
    fasilitas = read_table("gold", "fact_fasilitas")
    default = read_table("gold", "fact_default", columns=["facility_id", "tanggal_default"])
    debitur = read_table("gold", "dim_debitur")
    eksposur = read_table("gold", "fact_eksposur_grup")

    batas_observasi = pd.Timestamp(settings.snapshot_akhir)

    panel = kolek.merge(
        fasilitas[["facility_id", "plafon_rp", "tanggal_pencairan", "produk_id"]],
        on="facility_id",
        how="left",
    ).merge(default, on="facility_id", how="left")

    # Baris SETELAH default tidak boleh ikut - fasilitasnya sudah keluar populasi.
    panel = panel[
        panel["tanggal_default"].isna() | (panel["snapshot_date"] < panel["tanggal_default"])
    ].copy()

    panel["akhir_horizon"] = panel["snapshot_date"] + pd.DateOffset(months=HORIZON_EWS_BULAN)
    default_dalam_horizon = panel["tanggal_default"].notna() & (
        panel["tanggal_default"] <= panel["akhir_horizon"]
    )
    tersensor = (panel["akhir_horizon"] > batas_observasi) & ~default_dalam_horizon
    panel["y_default_6bln"] = np.where(tersensor, np.nan, default_dalam_horizon.astype(float))
    panel["y_tersensor"] = tersensor

    # ---- fitur perilaku pada snapshot (bukan sesudahnya)
    panel = panel.sort_values(["facility_id", "snapshot_date"])
    g = panel.groupby("facility_id", sort=False)
    panel["perilaku_kolektibilitas"] = panel["kolektibilitas"]
    panel["perilaku_dpd"] = panel["dpd"]
    panel["perilaku_restrukturisasi"] = panel["flag_restrukturisasi"]
    panel["perilaku_pemakaian_plafon"] = panel["outstanding_rp"] / panel["plafon_rp"]
    panel["perilaku_umur_fasilitas_hari"] = (
        panel["snapshot_date"] - panel["tanggal_pencairan"]
    ).dt.days
    panel["perilaku_kol_maks_3bln"] = g["kolektibilitas"].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    panel["perilaku_dpd_maks_3bln"] = g["dpd"].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    panel["perilaku_kol_memburuk"] = (g["kolektibilitas"].diff().fillna(0) > 0)

    pelanggaran = (
        covenant.assign(langgar=covenant["status"].eq("langgar"))
        .groupby(["facility_id", "snapshot_date"])["langgar"]
        .sum()
        .rename("perilaku_covenant_dilanggar")
        .reset_index()
    )
    panel = panel.merge(pelanggaran, on=["facility_id", "snapshot_date"], how="left")
    panel["perilaku_covenant_dilanggar"] = panel["perilaku_covenant_dilanggar"].fillna(0).astype(int)

    profil = debitur[debitur["is_current"]][
        ["cif_sk", "sektor_kbli", "rating_internal", "skor_kredit", "grup_id"]
    ]
    profil = _blok(profil, "app_", {"cif_sk", "grup_id", "angkatan"})
    panel = panel.merge(profil, on="cif_sk", how="left")

    panel = panel.merge(
        eksposur[["grup_id", "snapshot_date", "group_exposure_share"]].rename(
            columns={"group_exposure_share": "graf_group_exposure_share"}
        ),
        on=["grup_id", "snapshot_date"],
        how="left",
    )

    panel["split"] = _tandai_split(panel["tanggal_pencairan"])

    kunci = ["facility_id", "cif_sk", "grup_id", "snapshot_date", "split"]
    fitur = [c for c in panel.columns if c.startswith(("perilaku_", "app_", "graf_"))]
    target = ["y_default_6bln", "y_tersensor"]
    panel = panel[kunci + sorted(fitur) + target]

    _pastikan_bersih(panel, "abt_ews")
    return panel.reset_index(drop=True)


# -------------------------------------------------------------------- ABT LGD
# Ruang fitur model LGD - SATU sumber kebenaran untuk data latih dan data
# terapan. Aturannya: model tidak boleh belajar dari fitur yang tidak akan ada
# saat ia dipanggil. Kolom SBA seperti nilai pencairan USD, jumlah pegawai, dan
# negara bagian Amerika memang tidak punya padanan di portofolio sintetis, jadi
# dikeluarkan dari data latih - bukan dipaksakan ada di data terapan.
FITUR_LGD_TERAPAN = [
    "app_tenor_bulan",
    "app_jenis_fasilitas",
    "app_revolving",
    "app_sektor_kbli",
    "app_skala_pegawai",
    "app_perusahaan_baru",
    "app_dokumen_ringkas",
    "app_porsi_penjaminan",
]

FITUR_LGD_BERSINYAL = {*FITUR_LGD_TERAPAN, "app_produk_id"}


def build_abt_lgd() -> pd.DataFrame:
    """Fasilitas yang default, dengan LGD realisasi dari SBA sebagai target.

    PERINGATAN: 75 baris, dan hanya kolom pada FITUR_LGD_BERSINYAL yang punya
    hubungan nyata dengan target - karena berasal dari baris SBA yang sama.
    Sisanya disintesis terpisah dari target, jadi derau.

    Tabel ini untuk MENERAPKAN model LGD dan menghitung expected loss, bukan
    untuk melatihnya. Data latihnya ada di `abt_lgd_sumber`.
    """
    default = read_table("gold", "fact_default")
    fasilitas = read_table("gold", "fact_fasilitas")
    agunan = read_table("gold", "fact_agunan")
    debitur = read_table("gold", "dim_debitur")

    cov = (
        agunan.groupby("facility_id")
        .agg(
            app_nilai_likuidasi_rp=("nilai_likuidasi_rp", "sum"),
            app_coverage_ratio=("coverage_ratio", "first"),
            app_jumlah_agunan=("agunan_id", "size"),
        )
        .reset_index()
    )

    # Kolom SBA yang tertarik bersama lgd_realisasi pada langkah 5 - inilah
    # satu-satunya fitur yang hubungannya dengan LGD tidak dibuat-buat.
    sba = read_table("silver", "sl_peta_sba")[
        [
            "cif_sk",
            "Term",
            "porsi_penjaminan",
            "skala_pegawai",
            "dokumen_ringkas",
            "perusahaan_baru",
            "jenis_fasilitas",
            "kbli_kategori",
        ]
    ].rename(
        columns={
            "Term": "app_tenor_bulan",
            "porsi_penjaminan": "app_porsi_penjaminan",
            "skala_pegawai": "app_skala_pegawai",
            "dokumen_ringkas": "app_dokumen_ringkas",
            "perusahaan_baru": "app_perusahaan_baru",
            "jenis_fasilitas": "app_jenis_fasilitas",
            # kbli_kategori sudah dibawa dim_debitur sebagai app_sektor_kbli
            # dengan nilai yang identik, jadi tidak diduplikasi di sini.
        }
    )
    sba["app_revolving"] = sba["app_jenis_fasilitas"].eq("modal_kerja")

    abt = (
        default.merge(
            # cif_sk sudah ada di FACT_DEFAULT - jangan ikut supaya tidak jadi
            # cif_sk_x / cif_sk_y.
            fasilitas[["facility_id", "plafon_rp", "tanggal_pencairan", "produk_id"]],
            on="facility_id",
            how="left",
        )
        .merge(sba, on="cif_sk", how="left")
        .merge(cov, on="facility_id", how="left")
        .merge(
            _blok(
                debitur[debitur["is_current"]][
                    ["cif_sk", "sektor_kbli", "rating_internal", "grup_id"]
                ],
                "app_",
                {"cif_sk", "grup_id"},
            ),
            on="cif_sk",
            how="left",
        )
    )
    abt = abt.rename(
        columns={
            "plafon_rp": "app_plafon_rp",
            "ead_rp": "app_ead_rp",
            "lgd_realisasi": "y_lgd_realisasi",
            "produk_id": "app_produk_id",
        }
    )
    abt["app_ead_thd_plafon"] = abt["app_ead_rp"] / abt["app_plafon_rp"]
    abt["app_lgd_ditutup_agunan"] = (
        abt["app_nilai_likuidasi_rp"] / abt["app_ead_rp"]
    ).clip(0, 5)
    abt["split"] = _tandai_split(abt["tanggal_pencairan"])

    _pastikan_bersih(abt, "abt_lgd")
    return abt.drop(columns=["jumlah_pemulihan_rp"]).reset_index(drop=True)


def build_abt_lgd_sumber() -> pd.DataFrame:
    """Populasi SBA CHGOFF penuh - INI yang dipakai melatih model LGD.

    Seratus lima puluh enam ribu pinjaman nyata dengan LGD nyata dan fiturnya
    sendiri. Tidak ada join sintetis antara fitur dan target, jadi apa pun
    yang dipelajari model di sini adalah hubungan yang memang ada di SBA.

    Split out-of-time memakai tahun persetujuan (`ApprovalFY`), bukan tanggal
    pengajuan sintetis - populasi ini hidup di lini masa aslinya sendiri.
    """
    sba = read_table("silver", "sl_sba")
    lgd = sba[(sba["is_default"] == 1) & sba["lgd_realisasi"].notna()].copy()

    # HANYA fitur yang juga tersedia di abt_lgd. Menambah kolom di sini akan
    # menghasilkan model yang tidak bisa dipanggil pada portofolio - dijaga
    # test_ruang_fitur_lgd_sejajar dan gerbang kualitas.
    fitur = {
        "Term": "app_tenor_bulan",
        "jenis_fasilitas": "app_jenis_fasilitas",
        "revolving": "app_revolving",
        "kbli_kategori": "app_sektor_kbli",
        "skala_pegawai": "app_skala_pegawai",
        "perusahaan_baru": "app_perusahaan_baru",
        "dokumen_ringkas": "app_dokumen_ringkas",
        "porsi_penjaminan": "app_porsi_penjaminan",
    }
    fitur = {k: v for k, v in fitur.items() if k in lgd.columns}
    hilang = set(FITUR_LGD_TERAPAN) - set(fitur.values())
    if hilang:
        raise ValueError(f"fitur LGD tidak tersedia di sl_sba: {sorted(hilang)}")
    out = lgd[["sba_loan_nr", "ApprovalFY", *fitur]].rename(columns=fitur)
    out["y_lgd_realisasi"] = lgd["lgd_realisasi"].to_numpy()

    tahun = pd.to_numeric(out["ApprovalFY"], errors="coerce")
    batas = tahun.quantile(0.8)
    out["split"] = np.where(tahun <= batas, "latih", "uji_oot")

    LOG.info(
        "abt_lgd_sumber: %s pinjaman CHGOFF, LGD rata-rata %.3f",
        len(out),
        float(out["y_lgd_realisasi"].mean()),
    )
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ penjagaan
def _pastikan_ruang_fitur_lgd_sejajar(
    abt_lgd: pd.DataFrame, abt_lgd_sumber: pd.DataFrame
) -> None:
    """Model dilatih di satu tabel dan dipanggil di tabel lain - kolomnya wajib sama.

    Tanpa penjagaan ini, ketidakcocokan baru ketahuan saat data scientist
    memanggil predict() dan mendapat KeyError.
    """
    latih = {c for c in abt_lgd_sumber.columns if c.startswith("app_")}
    terap = {c for c in abt_lgd.columns if c.startswith("app_")}
    hanya_latih = sorted(latih - terap)
    if hanya_latih:
        raise AssertionError(
            "abt_lgd_sumber memuat fitur yang tidak ada di abt_lgd, sehingga "
            f"model tidak bisa diterapkan: {hanya_latih}"
        )
    kurang = sorted(set(FITUR_LGD_TERAPAN) - terap)
    if kurang:
        raise AssertionError(f"abt_lgd kehilangan fitur terapan: {kurang}")


def _pastikan_bersih(df: pd.DataFrame, nama: str) -> None:
    """Gagal keras kalau ada kolom terlarang lolos ke ABT."""
    bocor = sorted(set(df.columns) & KOLOM_TERLARANG)
    bocor += [c for c in df.columns if c.endswith(tuple(KOLOM_TERLARANG))]
    if bocor:
        raise AssertionError(f"{nama}: kolom terlarang lolos ke ABT -> {sorted(set(bocor))}")


# -------------------------------------------------------------- kamus & paket
# NaN pada kolom ini BUKAN data hilang, melainkan keadaan yang punya arti.
# EBITDA nol atau negatif membuat rasionya tak terdefinisi - dan debitur seperti
# itu tiga kali lebih sering gagal bayar (6,3% vs 2,1%). Mengimputasi median
# menghapus sinyalnya sekaligus memberi mereka angka yang tampak sehat.
KOLOM_KOSONG_BERMAKNA = {
    "fin_debt_to_ebitda",
    "fin_debt_to_ebitda_yoy",
    "fin_debt_to_ebitda_delta_3thn",
    "fin_cfo_to_ebitda",
}

# NaN pada kolom ini berarti "tidak ada relasi", bukan "tidak diketahui".
# fillna(0) benar secara semantik; fillna(median) salah.
KOLOM_KOSONG_BERARTI_NOL = {
    "graf_supplier_concentration_hhi",
    "graf_buyer_concentration_hhi",
    "graf_neighbor_default_rate_1hop",
    "graf_community_default_rate",
    "graf_group_exposure_share",
}


def _catatan_kolom(nama_abt: str, kolom: str, kategori: str) -> str:
    """Tandai kolom yang gampang disalahtafsirkan saat dipakai memodelkan."""
    catatan: list[str] = []

    if nama_abt == "abt_lgd" and kategori in ("app", "fin"):
        catatan.append(
            "sinyal nyata - dari baris SBA yang sama dengan target"
            if kolom in FITUR_LGD_BERSINYAL
            else "DERAU terhadap LGD - disintesis terpisah dari target"
        )
    if kolom in ("app_keputusan", "app_pricing_bps", "app_komite_level"):
        catatan.append(
            "hasil keputusan, bukan input - drop bila model untuk mendukung keputusan"
        )
    if kategori == "graf" and not catatan:
        catatan.append("blok ablasi - drop sekaligus untuk model baseline")

    # ---- perlakuan NaN
    if kolom in KOLOM_KOSONG_BERMAKNA:
        catatan.append(
            "NaN BERMAKNA (EBITDA <= 0, bad rate 3x lipat) - "
            "tambahkan indikator kosong, JANGAN fillna(median) polos"
        )
    elif kolom in KOLOM_KOSONG_BERARTI_NOL:
        catatan.append("NaN berarti 'tidak ada relasi' - fillna(0), bukan median")
    elif kolom == "y_umur_hari":
        catatan.append(
            "hanya terisi bila benar-benar gagal bayar - "
            "untuk model survival pakai y_umur_teramati_hari"
        )
    elif kolom.startswith("y_default"):
        catatan.append("NaN = tersensor kanan - buang, JANGAN diisi 0")

    return " | ".join(catatan)


def _kamus(abt: pd.DataFrame, nama_abt: str) -> pd.DataFrame:
    blok = {
        "key": "kunci / audit - jangan dipakai sebagai fitur",
        "fin": "rasio keuangan tahun buku terakhir sebelum pengajuan (NYATA)",
        "app": "karakteristik pengajuan, agunan, profil debitur pada saat T",
        "perilaku": "perilaku fasilitas PADA snapshot (hanya ABT EWS)",
        "graf": "fitur graf titik-waktu - blok ini yang di-drop saat uji ablasi",
        "y": "target",
        "split": "penanda train / uji out-of-time",
    }
    baris = []
    for kolom in abt.columns:
        prefiks = kolom.split("_")[0]
        kategori = prefiks if prefiks in blok else "key"
        baris.append(
            {
                "abt": nama_abt,
                "kolom": kolom,
                "blok": kategori,
                "keterangan_blok": blok[kategori],
                "dtype": str(abt[kolom].dtype),
                "persen_kosong": round(float(abt[kolom].isna().mean()) * 100, 2),
                "contoh_nilai": str(abt[kolom].dropna().iloc[0]) if abt[kolom].notna().any() else "",
                "catatan": _catatan_kolom(nama_abt, kolom, kategori),
            }
        )
    return pd.DataFrame(baris)


def build_abt() -> dict[str, int]:
    """Bangun ketiga ABT, kamus data, dan salinan CSV untuk serah terima."""
    abt_pd = build_abt_pd()
    abt_ews = build_abt_ews()
    abt_lgd = build_abt_lgd()
    abt_lgd_sumber = build_abt_lgd_sumber()
    ditolak = build_abt_pengajuan_ditolak(abt_pd)

    tabel = {
        "abt_pd": abt_pd,
        "abt_ews": abt_ews,
        "abt_lgd": abt_lgd,
        "abt_lgd_sumber": abt_lgd_sumber,
        "abt_pengajuan_ditolak": ditolak,
    }
    _pastikan_ruang_fitur_lgd_sejajar(abt_lgd, abt_lgd_sumber)

    for nama, df in tabel.items():
        write_table(df, "gold", nama)

    kamus = pd.concat(
        [
            _kamus(abt_pd, "abt_pd"),
            _kamus(abt_ews, "abt_ews"),
            _kamus(abt_lgd, "abt_lgd"),
            _kamus(abt_lgd_sumber, "abt_lgd_sumber"),
            _kamus(ditolak, "abt_pengajuan_ditolak"),
        ],
        ignore_index=True,
    )
    write_table(kamus, "gold", "kamus_data_abt")
    kamus.to_csv(GOLD_DIR / "kamus_data_abt.csv", index=False)

    penuh = abt_pd[abt_pd["y_default_12bln"].notna()]
    LOG.info(
        "abt_pd %s baris (%s dapat dilatih, bad rate %.2f%%), abt_ews %s, abt_lgd %s",
        len(abt_pd),
        len(penuh),
        100 * penuh["y_default_12bln"].mean(),
        len(abt_ews),
        len(abt_lgd),
    )
    return {nama: len(df) for nama, df in tabel.items()} | {"kamus_data_abt": len(kamus)}

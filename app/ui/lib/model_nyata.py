"""Lapisan model sungguhan: artefak `ml/artifacts` di atas data emas `data/gold`.

Berbeda dari `lib/mock_engine.py` yang berisi rumus tiruan deterministik, modul
ini benar-benar memuat model yang sudah dilatih:

    ml/artifacts/pd/pd_champion_new.joblib          XGBoost      -> skor default 12 bulan
    ml/artifacts/ews/ews_xgboost_champion.joblib    XGBoost      -> skor default 6 bulan
    ml/artifacts/lgd/final_lgd_xgboost_new.pkl      XGBoost      -> LGD
    ml/artifacts/pd_cluster/pd_cluster_champion.joblib  KMeans   -> ruang klaster portofolio

Tiap artefak datang bersama `*_decision_policy.json` yang menyatakan ambang
peringatan dan pita risikonya. Ambang itu dibaca dari berkas kebijakan, bukan
ditulis ulang di sini — dan bukan pula diambil dari `risk_cutoffs` yang
tertinggal di dalam joblib (lihat `muat_pd`).

Semua fungsi berat dibungkus cache Streamlit supaya halaman tetap responsif;
kalau artefak atau dependensi tidak ada, fungsi mengembalikan `None` dan halaman
memberi tahu apa yang kurang, bukan melempar traceback.

Catatan kompatibilitas artefak
------------------------------
1. Model PD versi ini TIDAK terkalibrasi. Keluarannya skor peringkat, dan layar
   menyebutnya begitu — bukan "PD terkalibrasi" seperti pada versi sebelumnya.
2. Pipeline LGD menyimpan transformer `"passthrough"` sebagai string. scikit-learn
   versi baru menolaknya di `ColumnTransformer.transform`, jadi ada jalur cadangan
   yang menyusun matriksnya manual bila `predict` langsung gagal.
3. Artefak dipickle dengan scikit-learn 1.3.2. Pada lingkungan dengan versi lebih
   baru, pemuatannya memunculkan `InconsistentVersionWarning` — ditelan di sini
   supaya tidak membanjiri layar, tetapi versinya tetap perlu dijaga.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

AKAR = Path(__file__).resolve().parents[3]
DIR_ARTEFAK = AKAR / "ml" / "artifacts"
DIR_GOLD = AKAR / "data" / "gold"

# Galat pemuatan artefak disimpan di sini, bukan ditelan diam-diam: halaman
# perlu bisa menyebut alasannya saat sebuah model tidak muncul.
GALAT_MUAT: dict[str, str] = {}

BERKAS_PD = DIR_ARTEFAK / "pd" / "pd_champion_new.joblib"
KEBIJAKAN_PD = DIR_ARTEFAK / "pd" / "pd_decision_policy.json"
BERKAS_EWS = DIR_ARTEFAK / "ews" / "ews_xgboost_champion.joblib"
BERKAS_LGD = DIR_ARTEFAK / "lgd" / "final_lgd_xgboost_new.pkl"
BERKAS_KLASTER = DIR_ARTEFAK / "pd_cluster" / "pd_cluster_champion.joblib"

# Berkas LGD berekstensi `.pkl` tetapi ditulis joblib, bukan `pickle` polos.
# Dibaca dengan `pickle.load` ia menghasilkan array nama fitur lalu gagal di
# tengah jalan; seluruh pemuatan artefak di modul ini karena itu lewat joblib.

TAHUN_PENILAIAN = 2026

# Urutan rating internal. Dipakai sebagai fitur ordinal pada EWS dan sebagai
# label pada peta klaster. Antarmuka sendiri tidak lagi menerjemahkan PD menjadi
# kelas rating - yang ditampilkan adalah pita risiko dari berkas kebijakan PD.
URUTAN_RATING = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
ORD_RATING = {r: i + 1 for i, r in enumerate(URUTAN_RATING)}

# Sektor pada narasi relationship manager ditulis dengan bahasa sehari-hari,
# sedangkan ABT memakai kategori KBLI satu huruf.
SEKTOR_KE_KBLI = {
    "Manufaktur komponen otomotif": "C",
    "Manufaktur kemasan": "C",
    "Tekstil dan garmen": "C",
    "Pengolahan hasil perkebunan": "C",
    "Distribusi bahan bangunan": "G",
    "Perdagangan besar farmasi": "G",
    "Kontraktor infrastruktur": "F",
    "Logistik dan pergudangan": "H",
}

# Fitur klaster dipilih yang terbaca analis kredit: struktur permodalan,
# kemampuan bayar, efisiensi, dan posisi pada graf grup.
FITUR_KLASTER = [
    "fin_der",
    "fin_icr",
    "fin_debt_to_ebitda",
    "fin_current_ratio",
    "fin_roa",
    "fin_operating_margin",
    "fin_growth_penjualan",
    "fin_cfo_to_liability",
    "app_plafon_thd_penjualan",
    "app_skor_kredit",
    "graf_group_exposure_share",
    "graf_neighbor_default_rate_1hop",
]

NAMA_FITUR_KLASTER = {
    "fin_der": "Debt to equity",
    "fin_icr": "Interest coverage",
    "fin_debt_to_ebitda": "Debt to EBITDA",
    "fin_current_ratio": "Current ratio",
    "fin_roa": "Return on asset",
    "fin_operating_margin": "Marjin operasi",
    "fin_growth_penjualan": "Pertumbuhan penjualan",
    "fin_cfo_to_liability": "Arus kas operasi / liabilitas",
    "app_plafon_thd_penjualan": "Plafon terhadap penjualan",
    "app_skor_kredit": "Skor kredit internal",
    "graf_group_exposure_share": "Porsi eksposur grup",
    "graf_neighbor_default_rate_1hop": "Tingkat default tetangga 1 hop",
}

# Label fitur PD supaya reason code terbaca komite, bukan nama kolom warehouse.
NAMA_FITUR = {
    "fin_der": "Debt to equity",
    "fin_icr": "Interest coverage ratio",
    "fin_debt_to_ebitda": "Debt to EBITDA",
    "fin_current_ratio": "Current ratio",
    "fin_quick_ratio": "Quick ratio",
    "fin_roa": "Return on asset",
    "fin_gross_margin": "Marjin kotor",
    "fin_operating_margin": "Marjin operasi",
    "fin_growth_penjualan": "Pertumbuhan penjualan",
    "fin_cfo_to_ebitda": "Konversi EBITDA ke kas",
    "fin_cfo_to_liability": "Arus kas operasi / liabilitas",
    "fin_asset_turnover": "Perputaran aset",
    "fin_dso_hari": "Days sales outstanding",
    "fin_dio_hari": "Days inventory outstanding",
    "fin_siklus_modal_kerja_hari": "Siklus modal kerja",
    "fin_re_to_ta": "Laba ditahan / total aset",
    "fin_wc_to_ta": "Modal kerja / total aset",
    "fin_ebitda_rp": "EBITDA",
    "fin_ekuitas_rp": "Ekuitas",
    "fin_penjualan_rp": "Penjualan tahunan",
    "fin_laba_bersih_rp": "Laba bersih",
    "fin_total_aset_rp": "Total aset",
    "fin_total_liabilitas_rp": "Total liabilitas",
    "fin_ebitda_nonpositif": "EBITDA tidak positif",
    "fin_der_yoy": "Perubahan DER setahun",
    "fin_icr_yoy": "Perubahan ICR setahun",
    "fin_debt_to_ebitda_yoy": "Perubahan debt/EBITDA setahun",
    "fin_der_delta_3thn": "Tren DER tiga tahun",
    "fin_icr_delta_3thn": "Tren ICR tiga tahun",
    "fin_debt_to_ebitda_delta_3thn": "Tren debt/EBITDA tiga tahun",
    "app_plafon_diminta_rp": "Plafon diminta",
    "app_penjualan_rp": "Penjualan pada berkas pengajuan",
    "app_tenor_bulan": "Tenor",
    "app_skor_kredit": "Skor kredit internal",
    "app_rating_internal": "Rating internal awal",
    "app_sektor_kbli": "Sektor KBLI",
    "app_jenis_fasilitas": "Jenis fasilitas",
    "app_nama_produk": "Produk kredit",
    "app_kelas_penjualan": "Kelas penjualan",
    "app_jumlah_agunan": "Jumlah agunan",
    "app_nilai_likuidasi_rp": "Nilai likuidasi agunan",
    "app_ada_agunan_likuid": "Ada agunan likuid",
    "app_ada_jaminan_silang": "Ada jaminan silang",
    "app_dokumen_ringkas": "Dokumen ringkas",
    "app_revolving": "Fasilitas revolving",
    "app_umur_perusahaan_tahun": "Umur perusahaan",
    "app_tahun_berdiri": "Tahun berdiri",
    "graf_group_exposure_share": "Porsi eksposur grup",
    "graf_neighbor_default_rate_1hop": "Default tetangga 1 hop",
    "graf_community_default_rate": "Default klaster ekosistem",
    "graf_buyer_concentration_hhi": "Konsentrasi pembeli (HHI)",
    "graf_supplier_concentration_hhi": "Konsentrasi pemasok (HHI)",
    "graf_shared_attribute_degree": "Derajat atribut berbagi",
    "graf_circular_payment_flag": "Indikasi transaksi melingkar",
    "graf_pagerank": "PageRank pada graf",
    "graf_betweenness": "Betweenness pada graf",
    "graf_degree": "Derajat simpul",
    "graf_weighted_degree": "Derajat berbobot",
}


def label_fitur(kolom: str) -> str:
    if kolom in NAMA_FITUR:
        return NAMA_FITUR[kolom]
    bersih = kolom.replace("num__", "").replace("cat__", "")
    if bersih in NAMA_FITUR:
        return NAMA_FITUR[bersih]
    # Kolom one-hot: cat__app_sektor_kbli_C -> "Sektor KBLI = C"
    for asal, label in NAMA_FITUR.items():
        if bersih.startswith(asal + "_"):
            return f"{label} = {bersih[len(asal) + 1:]}"
    return bersih.replace("_", " ")


# --------------------------------------------------------------------------
# Pemuatan artefak
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def muat_pd() -> dict | None:
    """Bundel PD: pipeline XGBoost, daftar fitur, dan ambang pita risiko.

    Dua hal yang berubah sejak artefak pindah ke `ml/artifacts`:

    1. Model ini TIDAK terkalibrasi (`pd_decision_policy.json` menyebutnya
       `raw_xgboost_probability`), dan bundelnya memang tidak membawa
       kalibrator. Keluarannya skor peringkat, bukan probabilitas yang boleh
       dibaca sebagai PD regulatoris.
    2. `risk_cutoffs` di dalam joblib tertinggal dari versi sebelumnya. Kuantil
       skor model ini sendiri atas `abt_pd` adalah 0,079 / 0,273 / 0,643,
       sedangkan joblib menyimpan 0,008 / 0,037 / 0,168 — memakainya membuat
       pita "rendah" kosong sama sekali dan sepertiga portofolio jatuh ke pita
       tertinggi. Yang dipakai karena itu ambang pada berkas kebijakan.
    """
    if not BERKAS_PD.exists():
        GALAT_MUAT["pd"] = f"berkas tidak ada: {BERKAS_PD}"
        return None
    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundel = dict(joblib.load(BERKAS_PD))
    except Exception as exc:
        GALAT_MUAT["pd"] = f"{type(exc).__name__}: {exc}"
        return None

    bundel["risk_cutoffs_bundel"] = bundel.get("risk_cutoffs")
    kebijakan = _baca_kebijakan(KEBIJAKAN_PD)
    if kebijakan and kebijakan.get("risk_cutoffs"):
        bundel["risk_cutoffs"] = kebijakan["risk_cutoffs"]
        bundel["asal_cutoffs"] = KEBIJAKAN_PD.name
    else:
        bundel["asal_cutoffs"] = "bundel joblib (berkas kebijakan tidak terbaca)"
    bundel["kebijakan"] = kebijakan or {}
    bundel["terkalibrasi"] = bool((kebijakan or {}).get("calibrated", False))
    return bundel


def _baca_kebijakan(berkas: Path) -> dict | None:
    """Berkas `*_decision_policy.json` yang menyertai satu artefak model."""
    if not berkas.exists():
        return None
    try:
        return json.loads(berkas.read_text(encoding="utf-8"))
    except Exception as exc:
        GALAT_MUAT[berkas.stem] = f"{type(exc).__name__}: {exc}"
        return None


@st.cache_resource(show_spinner=False)
def muat_ews() -> dict | None:
    """Bundel EWS: pipeline XGBoost, ambang peringatan, dan tiga pita risiko."""
    return _muat_bundel("ews", BERKAS_EWS)


@st.cache_resource(show_spinner=False)
def muat_lgd() -> dict | None:
    return _muat_bundel("lgd", BERKAS_LGD)


@st.cache_resource(show_spinner=False)
def muat_klaster() -> dict | None:
    """Artefak klaster portofolio: imputer, scaler, PCA, dan KMeans terlatih."""
    return _muat_bundel("klaster", BERKAS_KLASTER)


def _muat_bundel(nama: str, berkas: Path) -> dict | None:
    """Pemuatan artefak yang seragam; kegagalannya dicatat, bukan dilempar."""
    if not berkas.exists():
        GALAT_MUAT[nama] = f"berkas tidak ada: {berkas}"
        return None
    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(berkas)
    except Exception as exc:
        GALAT_MUAT[nama] = f"{type(exc).__name__}: {exc}"
        return None


@st.cache_data(show_spinner=False)
def gold(nama: str) -> pd.DataFrame | None:
    """Baca satu tabel lapisan emas. `None` bila belum dibangun."""
    berkas = DIR_GOLD / f"{nama}.parquet"
    if not berkas.exists():
        return None
    try:
        return pd.read_parquet(berkas)
    except Exception:
        return None


def status_lapisan_model() -> dict:
    """Ringkasan kesiapan untuk ditampilkan pada sidebar."""
    abt = gold("abt_pd")
    return {
        "galat_muat": dict(GALAT_MUAT),
        "pd": muat_pd() is not None,
        "ews": muat_ews() is not None,
        "lgd": muat_lgd() is not None,
        "klaster": muat_klaster() is not None,
        "gold": abt is not None,
        "baris_abt": 0 if abt is None else len(abt),
    }


# --------------------------------------------------------------------------
# Nilai rujukan portofolio: pengisi fitur yang tidak ada pada narasi
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def rujukan_portofolio() -> dict:
    """Median (numerik) dan modus (kategorikal) dari data latih PD.

    Fitur yang tidak muncul pada narasi maupun dokumen diisi dari sini, dan
    daftar isiannya ditampilkan apa adanya di halaman supaya analis tahu angka
    mana yang bukan berasal dari berkas nasabah.
    """
    bundel = muat_pd()
    abt = gold("abt_pd")
    if bundel is None or abt is None:
        return {}
    latih = abt[abt["split"] == "latih"] if "split" in abt else abt
    nilai: dict[str, object] = {}
    for kolom in bundel["features"]:
        if kolom == "fin_ebitda_nonpositif":
            nilai[kolom] = False
            continue
        if kolom not in latih.columns:
            nilai[kolom] = 0.0
            continue
        seri = latih[kolom]
        if seri.dtype.kind in "biufc":
            nilai[kolom] = float(pd.to_numeric(seri, errors="coerce").median())
        else:
            modus = seri.dropna().astype(str).mode()
            nilai[kolom] = str(modus.iat[0]) if len(modus) else ""
    # Rasio penjualan terhadap ekuitas dipakai untuk menaksir permodalan bila
    # narasi hanya menyebut penjualan dan DER.
    nilai["_penjualan_per_ekuitas"] = float(
        (latih["fin_penjualan_rp"] / latih["fin_ekuitas_rp"].replace(0, np.nan)).median()
    )
    return nilai


def _kelas_penjualan(penjualan: float) -> str:
    miliar = penjualan / 1e9
    if miliar < 50:
        return "Rp 30-50 M"
    if miliar < 100:
        return "Rp 50-100 M"
    if miliar < 200:
        return "Rp 100-200 M"
    return "Rp 200-300 M"


def _produk(jenis_fasilitas: str, agunan: str) -> tuple[str, str, bool]:
    """(app_jenis_fasilitas, app_nama_produk, revolving) dari istilah tampilan."""
    j = (jenis_fasilitas or "").lower()
    if "investasi" in j or "term loan" in j:
        produk = "Kredit Investasi Mesin" if "mesin" in (agunan or "").lower() \
            else "Kredit Investasi Ekspansi"
        return "investasi", produk, False
    if "rekening koran" in j:
        return "modal_kerja", "Kredit Modal Kerja Revolving", True
    return "modal_kerja", "Kredit Modal Kerja Transaksional", "koran" in j


def _angka(entitas: dict, kunci: str) -> float | None:
    """Nilai numerik dari entitas, atau `None` kalau pos itu tidak terbaca."""
    nilai = entitas.get(kunci)
    if nilai is None or nilai == "":
        return None
    try:
        return float(nilai)
    except (TypeError, ValueError):
        return None


def ebitda_bangun(laba_bersih, pajak, bunga, penyusutan) -> float | None:
    """EBITDA disusun ulang dari laba bersih + pajak + bunga + penyusutan.

    Laporan in-house hampir tidak pernah memuat baris EBITDA, sementara tiga
    fitur model bergantung padanya (debt to EBITDA, ICR, marjin operasi). Tanpa
    penyusunan ini ketiganya berdiri di atas taksiran marjin 10% - angka yang
    tidak pernah dilihat siapa pun di dokumen nasabah.

    Butuh laba bersih dan sedikitnya satu komponen tambahan; kalau tidak,
    hasilnya `None` dan pemanggil kembali ke taksiran marjin.
    """
    if laba_bersih is None:
        return None
    tambahan = [x for x in (pajak, bunga, penyusutan) if x is not None]
    if not tambahan:
        return None
    return float(laba_bersih) + float(sum(tambahan))


def _rasio_tahun(fakta: dict) -> dict:
    """DER, ICR, debt to EBITDA, dan penjualan untuk satu periode laporan.

    Rumusnya dibuat sama persis dengan jalur utama `bangun_fitur_pd` supaya
    perubahan antar tahun benar-benar perubahan angka nasabah, bukan efek dua
    definisi yang berbeda.
    """
    penjualan = fakta.get("penjualan")
    liabilitas = fakta.get("total_liabilitas")
    ekuitas = fakta.get("ekuitas")
    bunga = fakta.get("beban_bunga")
    ebitda = ebitda_bangun(
        fakta.get("laba_bersih"), fakta.get("pajak"), bunga, fakta.get("penyusutan")
    )
    rasio: dict[str, float] = {}
    if penjualan:
        rasio["penjualan_rp"] = float(penjualan)
    if liabilitas and ekuitas:
        rasio["der"] = float(liabilitas) / float(ekuitas)
    if liabilitas and ebitda and ebitda > 0:
        rasio["debt_to_ebitda"] = float(liabilitas) / ebitda
    if ebitda and bunga:
        rasio["icr"] = ebitda / float(bunga)
    return rasio


def fitur_tren(riwayat: dict) -> dict[str, float]:
    """Fitur tren dari laporan beberapa periode.

    Mengikuti `pipelines/generators/sintesis.py`: `_yoy` adalah perubahan
    relatif terhadap tahun sebelumnya, `_delta_3thn` adalah SELISIH ABSOLUT
    terhadap tahun paling awal - bukan rasio, dan bukan rata-rata.

    Tahun yang tersedia kurang dari dua berarti tidak ada tren untuk dihitung,
    dan `_delta_3thn` hanya diisi kalau tiga periode benar-benar terbaca.
    """
    tahun = sorted(int(x) for x in riwayat)
    if len(tahun) < 2:
        return {}
    seri = {y: _rasio_tahun(riwayat[y]) for y in tahun}
    kini, lalu, awal = seri[tahun[-1]], seri[tahun[-2]], seri[tahun[0]]

    hasil: dict[str, float] = {}
    for kolom, nama in (("der", "fin_der"), ("icr", "fin_icr"),
                        ("debt_to_ebitda", "fin_debt_to_ebitda")):
        if kolom in kini and kolom in lalu and lalu[kolom]:
            hasil[f"{nama}_yoy"] = (kini[kolom] - lalu[kolom]) / abs(lalu[kolom])
        if len(tahun) >= 3 and kolom in kini and kolom in awal:
            hasil[f"{nama}_delta_3thn"] = kini[kolom] - awal[kolom]
    if "penjualan_rp" in kini and lalu.get("penjualan_rp"):
        hasil["fin_growth_penjualan"] = (
            kini["penjualan_rp"] - lalu["penjualan_rp"]) / lalu["penjualan_rp"]
    return hasil


def bangun_fitur_pd(entitas: dict) -> tuple[pd.DataFrame, list[str]]:
    """Susun satu baris fitur PD dari entitas hasil ekstraksi narasi/dokumen.

    Mengembalikan (baris, daftar_fitur_yang_diisi_rujukan). Fitur graf memang
    tidak bisa dibaca dari dokumen — nilainya datang dari lapisan graf; pada
    demo ini indikasi pada narasi dipakai untuk menggesernya dari median.
    """
    rujukan = rujukan_portofolio()
    bundel = muat_pd()
    if bundel is None or not rujukan:
        return pd.DataFrame(), []

    penjualan = float(entitas.get("penjualan_tahunan") or rujukan["fin_penjualan_rp"])
    margin = float(entitas.get("ebitda_margin") or 0.10)
    ebitda = float(entitas.get("ebitda_rp") or penjualan * margin)
    # Baris EBITDA hampir tidak pernah ada di laporan in-house. Kalau dokumen
    # tidak memuatnya, susun ulang dari komponennya sebelum jatuh ke taksiran
    # marjin - dan setel marjin operasi mengikuti EBITDA hasil susunan itu,
    # supaya keduanya tidak saling bertentangan di baris fitur yang sama.
    if not entitas.get("ebitda_rp"):
        disusun = ebitda_bangun(
            _angka(entitas, "laba_bersih_rp"), _angka(entitas, "pajak_rp"),
            _angka(entitas, "beban_bunga_rp"), _angka(entitas, "penyusutan_rp"),
        )
        if disusun and disusun > 0 and penjualan > 0:
            ebitda = disusun
            margin = ebitda / penjualan
    der = float(entitas.get("der") or rujukan["fin_der"])
    ekuitas = float(entitas.get("ekuitas_rp") or penjualan / max(rujukan["_penjualan_per_ekuitas"], 0.1))
    liabilitas = float(entitas.get("total_liabilitas_rp") or der * ekuitas)
    aset = float(entitas.get("total_aset_rp") or (ekuitas + liabilitas))
    bunga = float(entitas.get("beban_bunga_rp") or liabilitas * 0.62 * 0.105)
    plafon = float(entitas.get("plafon") or rujukan["app_plafon_diminta_rp"])
    agunan_nilai = float(entitas.get("nilai_agunan") or plafon * 1.3)
    umur = float(entitas.get("umur_usaha_thn") or rujukan["app_umur_perusahaan_tahun"])
    jenis, produk, revolving = _produk(
        entitas.get("jenis_fasilitas", ""), entitas.get("jenis_agunan", "")
    )

    terhitung = {
        "fin_penjualan_rp": penjualan,
        "fin_ebitda_rp": ebitda,
        "fin_ekuitas_rp": ekuitas,
        "fin_total_liabilitas_rp": liabilitas,
        "fin_total_aset_rp": aset,
        "fin_laba_bersih_rp": float(entitas.get("laba_bersih_rp") or max(ebitda - bunga, 0.0) * 0.75),
        "fin_der": der,
        "fin_debt_to_ebitda": liabilitas / ebitda if ebitda > 0 else 12.0,
        "fin_icr": ebitda / bunga if bunga > 0 else 8.0,
        "fin_operating_margin": margin,
        "fin_gross_margin": min(margin * 2.4, 0.65),
        "fin_roa": (max(ebitda - bunga, 0.0) * 0.75) / aset if aset > 0 else 0.0,
        "fin_asset_turnover": penjualan / aset if aset > 0 else 1.0,
        "fin_ebitda_nonpositif": ebitda <= 0,
        "app_penjualan_rp": penjualan,
        "app_plafon_diminta_rp": plafon,
        "app_tenor_bulan": int(entitas.get("tenor_bulan") or 36),
        "app_umur_perusahaan_tahun": umur,
        "app_tahun_berdiri": int(TAHUN_PENILAIAN - umur),
        "app_kelas_penjualan": _kelas_penjualan(penjualan),
        "app_jenis_fasilitas": jenis,
        "app_nama_produk": produk,
        "app_revolving": bool(revolving),
        "app_sektor_kbli": SEKTOR_KE_KBLI.get(entitas.get("sektor", ""), "C"),
        # Nilai likuidasi dikutip dari hasil appraisal pada nota analisa bila ada;
        # potongan 0,75 hanya taksiran cadangan saat nota tidak memuatnya.
        "app_nilai_likuidasi_rp": float(
            _angka(entitas, "nilai_likuidasi") or agunan_nilai * 0.75),
        "app_jumlah_agunan": int(
            _angka(entitas, "jumlah_agunan")
            or (1 if "Tanpa agunan" in str(entitas.get("jenis_agunan", "")) else 2)),
        "app_ada_agunan_likuid": "Deposito" in str(entitas.get("jenis_agunan", "")),
        # Jaminan silang dinyatakan di nota analisa. Rangkap jabatan dipakai
        # sebagai penanda cadangan saja - keduanya memang berkorelasi, tetapi
        # yang satu pernyataan pengusul dan yang satu temuan graf.
        "app_ada_jaminan_silang": bool(
            entitas.get("ada_jaminan_silang", entitas.get("indikasi_rangkap_jabatan"))),
        "app_dokumen_ringkas": bool(entitas.get("dokumen_ringkas", False)),
    }

    # Penilaian unit risiko. Keduanya menyumbang porsi gain terbesar di model PD,
    # jadi membiarkannya jatuh ke median bukan pilihan netral: itu menarik setiap
    # pengajuan ke arah rata-rata portofolio dan menumpulkan sinyal fitur lain.
    rating = str(entitas.get("rating_internal") or "").strip().upper()
    if rating in ORD_RATING:
        terhitung["app_rating_internal"] = rating
    skor = _angka(entitas, "skor_kredit")
    if skor is not None:
        terhitung["app_skor_kredit"] = skor

    # Rasio neraca lancar hanya dihitung kalau pos penyusunnya benar-benar
    # terbaca dari dokumen. Kalau tidak, kolomnya sengaja dibiarkan jatuh ke
    # median portofolio di bawah dan muncul pada `diisi_rujukan` - lebih baik
    # halaman menyebut "ini angka median" daripada menampilkan rasio yang
    # sebetulnya karangan.
    #
    # Rumusnya mengikuti persis `pipelines/transform/silver.py`, tempat fitur
    # yang sama dibentuk untuk data latih. Menghitungnya dengan definisi lain
    # berarti memberi model angka yang tidak sebanding dengan yang dipelajarinya.
    lancar = _angka(entitas, "aset_lancar_rp")
    lancar_lb = _angka(entitas, "liabilitas_lancar_rp")
    persediaan = _angka(entitas, "persediaan_rp")
    piutang = _angka(entitas, "piutang_rp")
    laba_ditahan = _angka(entitas, "laba_ditahan_rp")
    hpp = _angka(entitas, "hpp_rp")

    if lancar and lancar_lb:
        terhitung["fin_current_ratio"] = lancar / lancar_lb
        # Perusahaan jasa yang memang tidak punya persediaan membuat quick ratio
        # sama dengan current ratio, dan itu betul - bukan pos yang hilang.
        terhitung["fin_quick_ratio"] = (lancar - (persediaan or 0.0)) / lancar_lb
        if aset > 0:
            terhitung["fin_wc_to_ta"] = (lancar - lancar_lb) / aset
    if laba_ditahan is not None and aset > 0:
        terhitung["fin_re_to_ta"] = laba_ditahan / aset
    if piutang and penjualan > 0:
        terhitung["fin_dso_hari"] = piutang / penjualan * 365.0
    if persediaan and hpp:
        terhitung["fin_dio_hari"] = persediaan / hpp * 365.0
    # Laba bruto dikutip kalau ada barisnya; kalau tidak, selisih penjualan dan
    # harga pokok memberi angka yang sama tanpa mengarang apa pun.
    laba_kotor = _angka(entitas, "laba_kotor_rp")
    if laba_kotor is None and hpp:
        laba_kotor = penjualan - hpp
    if laba_kotor is not None and penjualan > 0:
        terhitung["fin_gross_margin"] = laba_kotor / penjualan

    # Arus kas operasi: baris sungguhan kalau laporannya memuatnya, kalau tidak
    # proxy `laba bersih + penyusutan` - definisi yang sama persis dipakai saat
    # model dilatih, karena panel sumbernya pun tidak punya laporan arus kas.
    cfo = _angka(entitas, "arus_kas_operasi_rp")
    penyusutan = _angka(entitas, "penyusutan_rp")
    if cfo is None and penyusutan is not None:
        cfo = float(terhitung["fin_laba_bersih_rp"]) + penyusutan
    if cfo is not None:
        if ebitda > 0:
            terhitung["fin_cfo_to_ebitda"] = cfo / ebitda
        if liabilitas > 0:
            terhitung["fin_cfo_to_liability"] = cfo / liabilitas
    # Siklus modal kerja pada data latih adalah DSO + DIO tanpa DPO, dipotong di
    # 720 hari. Ia hanya bermakna kalau kedua komponennya nyata; satu komponen
    # nyata ditambah satu komponen median menghasilkan angka yang bukan keduanya.
    if "fin_dso_hari" in terhitung and "fin_dio_hari" in terhitung:
        terhitung["fin_siklus_modal_kerja_hari"] = float(np.clip(
            terhitung["fin_dso_hari"] + terhitung["fin_dio_hari"], 0.0, 720.0
        ))

    # Tren hanya ada kalau laporannya memuat lebih dari satu periode. Fitur yang
    # tidak terisi tetap jatuh ke median dan tetap disebut di `diisi_rujukan`.
    terhitung.update(fitur_tren(entitas.get("riwayat_tahun") or {}))

    baris: dict[str, object] = {}
    diisi_rujukan: list[str] = []
    for kolom in bundel["features"]:
        if kolom in terhitung:
            baris[kolom] = terhitung[kolom]
        else:
            baris[kolom] = rujukan.get(kolom, 0.0)
            diisi_rujukan.append(kolom)

    df = pd.DataFrame([baris])
    # Hanya lima kolom yang masuk blok kategorikal pipeline; sisanya numerik,
    # termasuk penanda boolean seperti `app_revolving`.
    for kolom in ("app_jenis_fasilitas", "app_kelas_penjualan", "app_nama_produk",
                  "app_rating_internal", "app_sektor_kbli"):
        if kolom in df.columns:
            df[kolom] = df[kolom].astype(str)
    return df, diisi_rujukan


# --------------------------------------------------------------------------
# Skoring PD
# --------------------------------------------------------------------------
@dataclass
class KontribusiFitur:
    fitur: str
    nilai: str
    dampak: float


@dataclass
class HasilPD:
    """Keluaran model PD untuk satu pengajuan.

    `skor` adalah probabilitas mentah XGBoost. Versi sebelumnya membawa dua
    angka - mentah dan terkalibrasi - karena bundelnya menyertakan kalibrator
    logistik. Artefak sekarang tidak, dan menyebut satu angka sebagai
    "terkalibrasi" hanya karena namanya begitu di kode lama akan membuat layar
    menjanjikan ketelitian yang tidak dimiliki model ini.
    """

    skor: float
    band: str
    warna: str
    cutoffs: dict
    terkalibrasi: bool
    kontribusi: list[KontribusiFitur]
    fitur_rujukan: list[str]
    baris: pd.DataFrame


# Warna pita risiko mengikuti palet aplikasi: tosca untuk sisi baik, jingga
# yang makin gelap untuk sisi buruk (lihat lib/tampilan.py).
BAND_WARNA = {
    "Risiko rendah": "#2A8080",
    "Risiko sedang": "#40C0C0",
    "Risiko tinggi": "#FF8000",
    "Risiko sangat tinggi": "#7A3C00",
}

# Urutan pita mengikuti `risk_band_definition` pada pd_decision_policy.json:
# Low, Medium, High, Very High - empat, bukan tiga. Ambangnya kuantil skor atas
# portofolio pengembangan, jadi pita ini peringkat relatif terhadap portofolio,
# bukan tingkat kerugian absolut.
URUTAN_BAND = ["Risiko rendah", "Risiko sedang", "Risiko tinggi", "Risiko sangat tinggi"]
BAND_KEBIJAKAN = {
    "Low": "Risiko rendah",
    "Medium": "Risiko sedang",
    "High": "Risiko tinggi",
    "Very High": "Risiko sangat tinggi",
}


def _band(nilai: float, cutoffs: dict) -> str:
    if nilai <= cutoffs["q50"]:
        return "Risiko rendah"
    if nilai <= cutoffs["q80"]:
        return "Risiko sedang"
    if nilai <= cutoffs["q95"]:
        return "Risiko tinggi"
    return "Risiko sangat tinggi"


def _kontribusi_pd(bundel: dict, baris: pd.DataFrame, jumlah: int = 10) -> list[KontribusiFitur]:
    """Reason code dari `pred_contribs` booster XGBoost (nilai SHAP tepat).

    Kolom one-hot dilipat kembali ke fitur asalnya supaya komite membaca
    "Sektor KBLI", bukan dua puluh kolom biner.
    """
    try:
        import xgboost as xgb

        pipa = bundel["model"]
        pra = pipa[:-1]
        est = pipa.steps[-1][1]
        Z = pra.transform(baris)
        Z = np.asarray(Z.todense()) if hasattr(Z, "todense") else np.asarray(Z)
        nama = list(pra.get_feature_names_out())
        dmat = xgb.DMatrix(Z, feature_names=[f"f{i}" for i in range(Z.shape[1])])
        kontrib = est.get_booster().predict(dmat, pred_contribs=True)[0][:-1]
    except Exception:
        return []

    lipat: dict[str, float] = {}
    for kolom, dampak in zip(nama, kontrib):
        lipat[label_fitur(kolom)] = lipat.get(label_fitur(kolom), 0.0) + float(dampak)

    urut = sorted(lipat.items(), key=lambda kv: abs(kv[1]), reverse=True)[:jumlah]
    hasil = []
    for label, dampak in urut:
        asal = next((k for k, v in NAMA_FITUR.items() if v == label), None)
        nilai = "-"
        if asal and asal in baris.columns:
            mentah = baris[asal].iat[0]
            nilai = f"{mentah:,.4g}".replace(",", ".") if isinstance(mentah, (int, float, np.number)) \
                else str(mentah)
        hasil.append(KontribusiFitur(label, nilai, float(dampak)))
    return hasil


def skor_pd(entitas: dict, dengan_kontribusi: bool = True) -> HasilPD | None:
    """Skor default 12 bulan untuk satu pengajuan.

    `dengan_kontribusi=False` melewati perhitungan SHAP — dipakai halaman
    simulasi yang menyapu puluhan skenario sekaligus dan hanya butuh angkanya.
    """
    bundel = muat_pd()
    if bundel is None:
        return None
    baris, rujukan_dipakai = bangun_fitur_pd(entitas)
    if baris.empty:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skor = float(bundel["model"].predict_proba(baris[bundel["features"]])[:, 1][0])
    cutoffs = bundel["risk_cutoffs"]
    band = _band(skor, cutoffs)
    return HasilPD(
        skor=skor,
        band=band,
        warna=BAND_WARNA[band],
        cutoffs=cutoffs,
        terkalibrasi=bool(bundel.get("terkalibrasi", False)),
        kontribusi=_kontribusi_pd(bundel, baris) if dengan_kontribusi else [],
        fitur_rujukan=rujukan_dipakai,
        baris=baris,
    )


# --------------------------------------------------------------------------
# LGD
# --------------------------------------------------------------------------
def _matriks_lgd(bundel: dict, df: pd.DataFrame) -> np.ndarray:
    ct = bundel["model"].steps[0][1]
    ohe = ct.transformers_[0][1]
    kat = df[bundel["categorical_features"]].astype(str)
    A = ohe.transform(kat)
    A = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    N = df[bundel["numeric_features"]].astype(float).values
    return np.hstack([A, N])


def skor_lgd(entitas: dict) -> float | None:
    """LGD dari model XGBoost; `None` bila artefak tidak tersedia."""
    bundel = muat_lgd()
    if bundel is None:
        return None
    jenis, _, _ = _produk(entitas.get("jenis_fasilitas", ""), entitas.get("jenis_agunan", ""))
    penjualan = float(entitas.get("penjualan_tahunan") or 100e9)
    # Skala pegawai adalah variabelnya sendiri, bukan turunan penjualan. Nota
    # analisa memuat jumlah karyawan; ambang mikro/kecil/menengah mengikuti
    # kategori usaha yang dipakai data latih. Tanpa jumlah karyawan, penjualan
    # dipakai sebagai penaksir - dan itu memang hanya penaksir.
    karyawan = entitas.get("jumlah_karyawan")
    if karyawan:
        jumlah = int(karyawan)
        skala = "menengah" if jumlah > 100 else "kecil" if jumlah > 20 else "mikro"
    else:
        skala = "menengah" if penjualan >= 50e9 else "kecil"
    nilai_agunan = float(entitas.get("nilai_agunan") or 0.0)
    plafon = float(entitas.get("plafon") or 1.0)
    baris = pd.DataFrame([{
        "app_tenor_bulan": int(entitas.get("tenor_bulan") or 36),
        "app_porsi_penjaminan": float(min(nilai_agunan / plafon, 2.0)) if plafon else 0.0,
        "app_jenis_fasilitas": jenis,
        "app_sektor_kbli": SEKTOR_KE_KBLI.get(entitas.get("sektor", ""), "C"),
        "app_skala_pegawai": skala,
        "app_perusahaan_baru": str(float(entitas.get("umur_usaha_thn") or 10) < 3),
        "app_dokumen_ringkas": str(bool(entitas.get("dokumen_ringkas", False))),
    }])
    # Jalur utama memakai pipeline apa adanya. Jalur cadangan menyusun matriks
    # sendiri, untuk artefak lama yang menyimpan transformer "passthrough"
    # sebagai string dan ditolak scikit-learn versi baru.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            nilai = float(bundel["model"].predict(baris[bundel["features"]])[0])
        except Exception:
            try:
                Z = _matriks_lgd(bundel, baris)
                nilai = float(bundel["model"].steps[-1][1].predict(Z)[0])
            except Exception as exc:
                GALAT_MUAT["lgd_prediksi"] = f"{type(exc).__name__}: {exc}"
                return None
    # `lgd_decision_policy.json`: keluaran dijepit ke [0, 1], tanpa kalibrasi.
    return float(np.clip(nilai, 0.0, 1.0))


# --------------------------------------------------------------------------
# EWS
# --------------------------------------------------------------------------
def siapkan_fitur_ews(df: pd.DataFrame) -> pd.DataFrame:
    """Bangun ulang fitur turunan EWS dari panel bulanan `abt_ews`."""
    d = df.sort_values(["facility_id", "snapshot_date"]).copy()
    g = d.groupby("facility_id")
    d["app_rating_ord"] = d["app_rating_internal"].map(ORD_RATING)
    d["dpd_gap_vs_max3m"] = d["perilaku_dpd_maks_3bln"] - d["perilaku_dpd"]
    d["kol_gap_vs_max3m"] = d["perilaku_kol_maks_3bln"] - d["perilaku_kolektibilitas"]
    pasangan = [
        ("dpd", "perilaku_dpd"),
        ("util", "perilaku_pemakaian_plafon"),
        ("covenant", "perilaku_covenant_dilanggar"),
        ("kol", "perilaku_kolektibilitas"),
    ]
    for nama, asal in pasangan:
        d[f"{nama}_delta_1m"] = g[asal].diff(1).fillna(0)
        d[f"{nama}_delta_3m"] = g[asal].diff(3).fillna(0)
    d["util_avg_3m"] = g["perilaku_pemakaian_plafon"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    d["util_vs_avg3m"] = d["perilaku_pemakaian_plafon"] - d["util_avg_3m"]
    return d


def skor_ews(df_siap: pd.DataFrame) -> np.ndarray | None:
    """Skor peringatan dini 6 bulan untuk panel yang sudah disiapkan.

    Ditulis ulang untuk artefak XGBoost pada `ml/artifacts/ews`. Versi
    sebelumnya menghitung sendiri hasil kali koefisien regresi logistik dan
    menyisipkan kolom one-hot rating `D` yang hilang dari preprocessor lama —
    seluruhnya tidak berlaku lagi, dan kalau dibiarkan ia gagal diam-diam
    menjadi `None` karena terbungkus `try/except`.
    """
    bundel = muat_ews()
    if bundel is None:
        return None
    kurang = [f for f in bundel["features"] if f not in df_siap.columns]
    if kurang:
        GALAT_MUAT["ews_prediksi"] = f"kolom panel tidak ada: {', '.join(kurang[:5])}"
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bundel["model"].predict_proba(df_siap[bundel["features"]])[:, 1]
    except Exception as exc:
        GALAT_MUAT["ews_prediksi"] = f"{type(exc).__name__}: {exc}"
        return None


# Pita EWS hanya tiga - LOW, MEDIUM, HIGH - dan itu memang kontrak artefaknya,
# berbeda dari empat pita PD. Keduanya sengaja tidak diseragamkan: ambangnya
# datang dari sebaran skor yang berbeda, atas target yang berbeda pula.
PITA_EWS = {"LOW": "Pantauan biasa", "MEDIUM": "Perlu diperhatikan", "HIGH": "Peringatan dini"}
WARNA_EWS = {"LOW": "#2A8080", "MEDIUM": "#FF8000", "HIGH": "#7A3C00"}


def _pita_ews(nilai: float, cutoffs: dict) -> str:
    if nilai < cutoffs["q80"]:
        return "LOW"
    if nilai < cutoffs["q95"]:
        return "MEDIUM"
    return "HIGH"


@st.cache_data(show_spinner=False)
def _panel_ews() -> pd.DataFrame | None:
    """Panel bulanan lengkap dengan fitur turunannya, disiapkan sekali."""
    abt = gold("abt_ews")
    if abt is None:
        return None
    try:
        return siapkan_fitur_ews(abt)
    except Exception as exc:
        GALAT_MUAT["ews_panel"] = f"{type(exc).__name__}: {exc}"
        return None


@dataclass
class PantauanEWS:
    """Status peringatan dini fasilitas milik debitur yang tercocok."""

    tanggal: pd.Timestamp
    tabel: pd.DataFrame           # cif_sk, facility_id, snapshot, skor, pita, alarm
    ambang: float
    cutoffs: dict
    cif_tanpa_fasilitas: int      # tercocok, tetapi tidak punya baris pada panel

    @property
    def jumlah_alarm(self) -> int:
        return int(self.tabel["alarm"].sum()) if not self.tabel.empty else 0

    def cacah_pita(self) -> dict:
        if self.tabel.empty:
            return {}
        return self.tabel["pita"].value_counts().to_dict()


def ews_afiliasi(cif: tuple[int, ...], tanggal) -> PantauanEWS | None:
    """Status peringatan dini debitur eksisting yang tercocok dari dokumen.

    Ini satu-satunya cara EWS masuk ke halaman pengajuan, dan alasannya penting:
    pemohon baru TIDAK punya perilaku fasilitas — tidak ada tunggakan,
    pemakaian plafon, atau covenant yang bisa dilanggar — sehingga tidak ada
    satu pun dari 26 fitur model ini yang terisi untuknya. Yang punya perilaku
    adalah afiliasinya, dan memburuknya mereka justru pertanyaan kredit yang
    sah: apakah grup ini sedang menuju masalah selagi salah satu anggotanya
    meminta fasilitas baru.

    Skor dibaca pada snapshot terakhir yang tidak melewati `tanggal`, supaya
    penelaahan tidak memakai perilaku yang belum terjadi pada tanggal telaah.
    """
    if not cif:
        return None
    bundel = muat_ews()
    panel = _panel_ews()
    if bundel is None or panel is None:
        return None

    tanggal = pd.Timestamp(tanggal)
    bagian = panel[panel["cif_sk"].isin(cif)].copy()
    bagian["snapshot_date"] = pd.to_datetime(bagian["snapshot_date"])
    bagian = bagian[bagian["snapshot_date"] <= tanggal]
    if bagian.empty:
        return PantauanEWS(tanggal, pd.DataFrame(), float(bundel["threshold"]),
                           dict(bundel["risk_cutoffs"]), len(set(cif)))

    akhir = (bagian.sort_values("snapshot_date")
             .groupby("facility_id", as_index=False).last())
    skor = skor_ews(akhir)
    if skor is None:
        return None

    cutoffs = dict(bundel["risk_cutoffs"])
    ambang = float(bundel["threshold"])
    tabel = pd.DataFrame({
        "cif_sk": akhir["cif_sk"].astype(int).values,
        "facility_id": akhir["facility_id"].values,
        "snapshot": akhir["snapshot_date"].values,
        "skor": skor,
        "pita": [_pita_ews(float(v), cutoffs) for v in skor],
        "alarm": skor >= ambang,
        "dpd": akhir["perilaku_dpd"].values if "perilaku_dpd" in akhir else np.nan,
        "kolektibilitas": (akhir["perilaku_kolektibilitas"].values
                           if "perilaku_kolektibilitas" in akhir else np.nan),
        "pemakaian_plafon": (akhir["perilaku_pemakaian_plafon"].values
                             if "perilaku_pemakaian_plafon" in akhir else np.nan),
    }).sort_values("skor", ascending=False).reset_index(drop=True)

    tanpa = len(set(cif) - set(tabel["cif_sk"].tolist()))
    return PantauanEWS(tanggal, tabel, ambang, cutoffs, tanpa)


# --------------------------------------------------------------------------
# Ruang klaster portofolio
# --------------------------------------------------------------------------
@dataclass
class RuangKlaster:
    titik: pd.DataFrame          # proyeksi 2D seluruh portofolio latih
    ringkas: pd.DataFrame        # profil tiap klaster
    pusat: pd.DataFrame          # pusat klaster pada bidang proyeksi
    varians: tuple[float, float]
    _pra: object
    _km: object
    _pca: object


@st.cache_resource(show_spinner=False)
def ruang_klaster(k: int = 4) -> RuangKlaster | None:
    """Ruang klaster portofolio dari artefak `pd_cluster`.

    Sebelumnya K-Means dilatih ulang setiap kali halaman dibuka, atas dua belas
    fitur pilihan modul ini. Sekarang imputer, scaler, PCA, dan KMeans datang
    terlatih dari `ml/artifacts/pd_cluster` bersama daftar fiturnya sendiri —
    jadi peta yang dilihat analis adalah peta yang sama dengan yang dievaluasi
    di notebook, bukan hasil pelatihan ulang yang kebetulan mirip.

    Argumen `k` dipertahankan demi pemanggil lama tetapi tidak lagi dipakai:
    jumlah klaster ditentukan artefak.
    """
    bundel = muat_klaster()
    abt = gold("abt_pd")
    if bundel is None or abt is None:
        return None

    fitur = list(bundel["features"])
    data = abt[abt["y_default_12bln"].notna()].copy()
    if "fin_ebitda_nonpositif" in fitur and "fin_ebitda_nonpositif" not in data:
        data["fin_ebitda_nonpositif"] = (data["fin_ebitda_rp"] <= 0).astype(int)
    kurang = [f for f in fitur if f not in data.columns]
    if kurang:
        GALAT_MUAT["klaster"] = f"kolom emas tidak ada: {', '.join(kurang[:5])}"
        return None

    X = data[fitur].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    X = _jepit_klaster(X, bundel.get("clip_bounds") or {})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Urutan artefak: imputer -> scaler -> PCA(22 komponen) -> KMeans.
            # KMeans dilatih di ruang PCA, jadi ia diberi P, bukan Z; dua
            # komponen pertama P itu juga yang menjadi sumbu peta.
            Z = bundel["scaler"].transform(bundel["imputer"].transform(X))
            P = bundel["pca"].transform(Z)
            label = bundel["kmeans"].predict(P)
    except Exception as exc:
        GALAT_MUAT["klaster"] = f"{type(exc).__name__}: {exc}"
        return None

    titik = pd.DataFrame({
        "x": P[:, 0],
        "y": P[:, 1],
        "klaster": label,
        "default": data["y_default_12bln"].astype(int).values,
        "rating": data["app_rating_internal"].astype(str).values,
        "penjualan": data["fin_penjualan_rp"].values,
        "plafon": data["app_plafon_diminta_rp"].values,
    })

    ringkas = (
        titik.groupby("klaster")
        .agg(jumlah=("default", "size"), tingkat_default=("default", "mean"))
        .reset_index()
    )
    # Profil klaster tetap ditampilkan pada dua belas rasio yang terbaca analis,
    # bukan seluruh fitur artefak - tabel 37 kolom tidak menolong siapa pun.
    profil = [f for f in FITUR_KLASTER if f in data.columns]
    for kolom in profil:
        ringkas[kolom] = [
            float(pd.to_numeric(data.loc[(titik["klaster"] == c).values, kolom],
                                errors="coerce").median())
            for c in ringkas["klaster"]
        ]
    ringkas = ringkas.sort_values("tingkat_default", ascending=False).reset_index(drop=True)
    ringkas["nama"] = _nama_klaster_artefak(ringkas, bundel)

    # Pusat klaster sudah berada di ruang PCA; dua komponen pertamanya langsung
    # menjadi koordinat peta tanpa transformasi lagi.
    pusat = pd.DataFrame(bundel["kmeans"].cluster_centers_[:, :2], columns=["x", "y"])
    pusat["klaster"] = range(len(pusat))
    pusat = pusat.merge(ringkas[["klaster", "nama", "tingkat_default"]], on="klaster")

    varians = getattr(bundel["pca"], "explained_variance_ratio_", [0.0, 0.0])
    return RuangKlaster(
        titik=titik, ringkas=ringkas, pusat=pusat,
        varians=(float(varians[0]), float(varians[1])),
        _pra=bundel, _km=bundel["kmeans"], _pca=bundel["pca"],
    )


def _jepit_klaster(X: pd.DataFrame, batas: dict) -> pd.DataFrame:
    """Terapkan `clip_bounds` artefak; rasio ekstrem dijepit seperti saat latih."""
    if not batas:
        return X
    Y = X.copy()
    for kolom, rentang in batas.items():
        if kolom not in Y.columns or not isinstance(rentang, (list, tuple)) or len(rentang) != 2:
            continue
        Y[kolom] = pd.to_numeric(Y[kolom], errors="coerce").clip(rentang[0], rentang[1])
    return Y


def _nama_klaster_artefak(ringkas: pd.DataFrame, bundel: dict) -> list[str]:
    """Nama klaster dari `cluster_segments` artefak, dengan cadangan lokal."""
    segmen = {int(k): str(v) for k, v in (bundel.get("cluster_segments") or {}).items()}
    terjemah = {
        "High Risk / Default-like": "Kantong default",
        "Low Risk / Non-default-like": "Inti sehat",
    }
    nama = []
    for i, c in enumerate(ringkas["klaster"]):
        asli = segmen.get(int(c))
        nama.append(terjemah.get(asli, asli) if asli else _nama_klaster(ringkas)[i])
    return nama


def _nama_klaster(ringkas: pd.DataFrame) -> list[str]:
    """Nama klaster mengikuti peringkat tingkat default, bukan nomor K-Means."""
    n = len(ringkas)
    label = ["Kantong default", "Rawan memburuk", "Menengah stabil", "Inti sehat",
             "Inti sehat (2)", "Inti sehat (3)"]
    return [label[i] if i < len(label) else f"Klaster {i}" for i in range(n)]


@dataclass
class PosisiKlaster:
    x: float
    y: float
    klaster: int
    nama: str
    tingkat_default_klaster: float
    jarak: pd.DataFrame          # jarak ke tiap pusat klaster (ruang fitur baku)
    condong_default: float       # 0..1, kedekatan relatif ke klaster paling berisiko


def posisi_klaster(entitas: dict, hasil_pd: HasilPD | None = None) -> PosisiKlaster | None:
    """Petakan satu pengajuan baru ke ruang klaster portofolio."""
    ruang = ruang_klaster()
    if ruang is None:
        return None
    baris = hasil_pd.baris if hasil_pd is not None else bangun_fitur_pd(entitas)[0]
    if baris is None or baris.empty:
        return None

    bundel = ruang._pra
    fitur = list(bundel["features"])
    nilai = {}
    for kolom in fitur:
        if kolom in baris.columns:
            nilai[kolom] = pd.to_numeric(baris[kolom].iat[0], errors="coerce")
        elif kolom == "app_plafon_thd_penjualan":
            nilai[kolom] = float(entitas.get("plafon", 0)) / max(
                float(entitas.get("penjualan_tahunan", 1)), 1.0
            )
        else:
            # Fitur yang tidak ada pada baris pengajuan dibiarkan kosong dan
            # diisi imputer artefak - nilai yang sama dengan saat pelatihan.
            nilai[kolom] = np.nan
    X = _jepit_klaster(pd.DataFrame([nilai])[fitur], bundel.get("clip_bounds") or {})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = bundel["scaler"].transform(bundel["imputer"].transform(X))
    P = ruang._pca.transform(Z)
    # Jarak diukur di ruang PCA, ruang yang sama tempat KMeans dilatih.
    jarak = np.linalg.norm(ruang._km.cluster_centers_ - P[0], axis=1)
    idx = int(np.argmin(jarak))

    tabel = ruang.ringkas.copy()
    tabel["jarak"] = [float(jarak[int(c)]) for c in tabel["klaster"]]
    tabel = tabel.sort_values("jarak")

    paling_berisiko = int(ruang.ringkas.iloc[0]["klaster"])
    paling_sehat = int(ruang.ringkas.iloc[-1]["klaster"])
    d_risk, d_sehat = float(jarak[paling_berisiko]), float(jarak[paling_sehat])
    condong = d_sehat / (d_risk + d_sehat) if (d_risk + d_sehat) > 0 else 0.5

    baris_klaster = ruang.ringkas[ruang.ringkas["klaster"] == idx].iloc[0]
    return PosisiKlaster(
        x=float(P[0, 0]), y=float(P[0, 1]), klaster=idx,
        nama=str(baris_klaster["nama"]),
        tingkat_default_klaster=float(baris_klaster["tingkat_default"]),
        jarak=tabel[["nama", "jarak", "tingkat_default", "jumlah"]],
        condong_default=float(condong),
    )


# --------------------------------------------------------------------------
# Metrik model dari data emas
# --------------------------------------------------------------------------
def _metrik_biner(y: np.ndarray, p: np.ndarray, ambang: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prediksi = (p >= ambang).astype(int)
    fpr_urut = np.argsort(p)
    y_urut = y[fpr_urut]
    n_pos, n_neg = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    tpr = np.cumsum(y_urut[::-1]) / n_pos
    fpr = np.cumsum((1 - y_urut)[::-1]) / n_neg
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": float(np.max(np.abs(tpr - fpr))),
        "recall": float(recall_score(y, prediksi, zero_division=0)),
        "presisi": float(precision_score(y, prediksi, zero_division=0)),
        "porsi_alarm": float(prediksi.mean()),
        "n": int(len(y)),
        "tingkat_kejadian": float(y.mean()),
    }


@st.cache_data(show_spinner=False)
def evaluasi_pd() -> dict | None:
    """Metrik PD pada split out-of-time, plus kurva recall terhadap ambang."""
    bundel, abt = muat_pd(), gold("abt_pd")
    if bundel is None or abt is None:
        return None
    try:
        hasil = {}
        for split in ["latih", "uji_oot"]:
            bagian = abt[(abt["split"] == split) & abt["y_default_12bln"].notna()].copy()
            if bagian.empty:
                continue
            bagian["fin_ebitda_nonpositif"] = bagian["fin_ebitda_rp"] <= 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = bundel["model"].predict_proba(bagian[bundel["features"]])[:, 1]
            y = bagian["y_default_12bln"].astype(int).values
            # Artefak ini tidak punya kalibrator, jadi seluruh metrik dihitung
            # atas skor mentah. Kunci `skor_kalibrasi` dipertahankan sebagai
            # alias supaya pembaca lama tidak pecah.
            hasil[split] = {
                **_metrik_biner(y, p, float(bundel["risk_cutoffs"]["q80"])),
                "skor": p, "skor_kalibrasi": p, "y": y,
            }
        oot = hasil.get("uji_oot")
        if oot is not None:
            kurva = []
            for nama, amb in [
                ("q50 · saring luas", bundel["risk_cutoffs"]["q50"]),
                ("q80 · operasional", bundel["risk_cutoffs"]["q80"]),
                ("q95 · eskalasi", bundel["risk_cutoffs"]["q95"]),
            ]:
                m = _metrik_biner(oot["y"], oot["skor_kalibrasi"], float(amb))
                kurva.append({"ambang": nama, "nilai_ambang": float(amb), **{
                    k: m[k] for k in ("recall", "presisi", "porsi_alarm")}})
            hasil["kurva_ambang"] = pd.DataFrame(kurva)
        return hasil
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def evaluasi_ews() -> dict | None:
    abt = gold("abt_ews")
    if abt is None or muat_ews() is None:
        return None
    try:
        siap = siapkan_fitur_ews(abt)
        oot = siap[(siap["split"] == "uji_oot") & siap["y_default_6bln"].notna()]
        p = skor_ews(oot)
        if p is None:
            return None
        y = oot["y_default_6bln"].astype(int).values
        kurva = pd.DataFrame([
            {"ambang": f"{amb:.0%}", "nilai_ambang": amb,
             **{k: _metrik_biner(y, p, amb)[k] for k in ("recall", "presisi", "porsi_alarm")}}
            for amb in (0.01, 0.02, 0.05, 0.10)
        ])
        return {"uji_oot": {**_metrik_biner(y, p, 0.02), "skor": p, "y": y},
                "kurva_ambang": kurva}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def evaluasi_lgd() -> dict | None:
    bundel, abt = muat_lgd(), gold("abt_lgd")
    if bundel is None or abt is None:
        return None
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = _matriks_lgd(bundel, abt)
            p = bundel["model"].steps[-1][1].predict(Z)
        y = abt[bundel["target"]].astype(float).values
        return {
            "n": int(len(y)),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(mean_squared_error(y, p) ** 0.5),
            "r2": float(r2_score(y, p)),
            "rata_realisasi": float(np.mean(y)),
            "rata_prediksi": float(np.mean(p)),
            "prediksi": p,
            "realisasi": y,
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def psi_fitur(jumlah_bin: int = 10) -> pd.DataFrame | None:
    """Population stability index fitur PD: data latih vs out-of-time."""
    bundel, abt = muat_pd(), gold("abt_pd")
    if bundel is None or abt is None:
        return None
    latih = abt[abt["split"] == "latih"]
    uji = abt[abt["split"] == "uji_oot"]
    if latih.empty or uji.empty:
        return None

    numerik = [
        k for k in bundel["features"]
        if k in abt.columns and abt[k].dtype.kind in "if" and abt[k].nunique() > 5
    ]
    baris = []
    for kolom in numerik:
        a = pd.to_numeric(latih[kolom], errors="coerce").dropna()
        b = pd.to_numeric(uji[kolom], errors="coerce").dropna()
        if len(a) < 50 or len(b) < 20:
            continue
        tepi = np.unique(np.quantile(a, np.linspace(0, 1, jumlah_bin + 1)))
        if len(tepi) < 3:
            continue
        tepi[0], tepi[-1] = -np.inf, np.inf
        pa = np.histogram(a, bins=tepi)[0] / len(a)
        pb = np.histogram(b, bins=tepi)[0] / len(b)
        pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
        baris.append({"fitur": label_fitur(kolom), "kolom": kolom,
                      "psi": float(np.sum((pb - pa) * np.log(pb / pa)))})
    if not baris:
        return None
    return pd.DataFrame(baris).sort_values("psi", ascending=False).reset_index(drop=True)

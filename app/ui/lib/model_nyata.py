"""Lapisan model sungguhan: artefak `ml/models` di atas data emas `data/gold`.

Berbeda dari `lib/mock_engine.py` yang berisi rumus tiruan deterministik, modul
ini benar-benar memuat model yang sudah dilatih:

    ml/models/pd_champion.joblib          XGBoost + kalibrator logistik   -> PD 12 bulan
    ml/models/ews_logistic_champion.joblib  Regresi logistik              -> EWS 6 bulan
    ml/models/final_lgd_xgboost.pkl       XGBoost regresi                 -> LGD

serta membangun ruang klaster portofolio dari `data/gold/abt_pd.parquet` untuk
memetakan pengajuan baru terhadap klaster default dan non-default.

Semua fungsi berat dibungkus cache Streamlit supaya halaman tetap responsif;
kalau artefak atau dependensi tidak ada, fungsi mengembalikan `None` dan halaman
memberi tahu apa yang kurang, bukan melempar traceback.

Catatan kompatibilitas artefak
------------------------------
1. Pipeline LGD menyimpan transformer `"passthrough"` sebagai string. scikit-learn
   versi baru menolaknya di `ColumnTransformer.transform`, jadi transformasinya
   dilakukan manual: one-hot untuk kolom kategorikal, kolom numerik apa adanya.
2. Preprocessor EWS dilatih tanpa rating `D` pada kamus one-hot, sementara
   regresi logistiknya berdimensi satu kolom lebih besar. Kolom rating `D`
   karena itu disisipkan kembali pada posisi blok rating sebelum skor dihitung.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

AKAR = Path(__file__).resolve().parents[3]
DIR_MODEL = AKAR / "ml" / "models"
DIR_GOLD = AKAR / "data" / "gold"

# Galat pemuatan artefak disimpan di sini, bukan ditelan diam-diam: halaman
# perlu bisa menyebut alasannya saat sebuah model tidak muncul.
GALAT_MUAT: dict[str, str] = {}

BERKAS_PD = DIR_MODEL / "pd_champion.joblib"
BERKAS_EWS = DIR_MODEL / "ews_logistic_champion.joblib"
BERKAS_LGD = DIR_MODEL / "final_lgd_xgboost.pkl"

TAHUN_PENILAIAN = 2026

# Urutan rating internal, dipakai sebagai fitur ordinal pada EWS dan untuk
# menerjemahkan PD menjadi kelas rating pada tampilan.
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
    """Bundel PD: pipeline XGBoost, kalibrator, daftar fitur, ambang risiko."""
    if not BERKAS_PD.exists():
        GALAT_MUAT["pd"] = f"berkas tidak ada: {BERKAS_PD}"
        return None
    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(BERKAS_PD)
    except Exception as exc:
        GALAT_MUAT["pd"] = f"{type(exc).__name__}: {exc}"
        return None


@st.cache_resource(show_spinner=False)
def muat_ews() -> object | None:
    if not BERKAS_EWS.exists():
        GALAT_MUAT["ews"] = f"berkas tidak ada: {BERKAS_EWS}"
        return None
    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(BERKAS_EWS)
    except Exception as exc:
        GALAT_MUAT["ews"] = f"{type(exc).__name__}: {exc}"
        return None


@st.cache_resource(show_spinner=False)
def muat_lgd() -> dict | None:
    if not BERKAS_LGD.exists():
        GALAT_MUAT["lgd"] = f"berkas tidak ada: {BERKAS_LGD}"
        return None
    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(BERKAS_LGD)
    except Exception as exc:
        GALAT_MUAT["lgd"] = f"{type(exc).__name__}: {exc}"
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
        "app_nilai_likuidasi_rp": agunan_nilai * 0.75,
        "app_jumlah_agunan": 1 if "Tanpa agunan" in str(entitas.get("jenis_agunan", "")) else 2,
        "app_ada_agunan_likuid": "Deposito" in str(entitas.get("jenis_agunan", "")),
        "app_ada_jaminan_silang": bool(entitas.get("indikasi_rangkap_jabatan")),
        "app_dokumen_ringkas": bool(entitas.get("dokumen_ringkas", False)),
    }

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
    pd_kalibrasi: float
    pd_mentah: float
    band: str
    warna: str
    cutoffs: dict
    kontribusi: list[KontribusiFitur]
    fitur_rujukan: list[str]
    baris: pd.DataFrame


BAND_WARNA = {
    "Risiko rendah": "#1f8a5f",
    "Risiko sedang": "#c58a17",
    "Risiko tinggi": "#d4703a",
    "Risiko sangat tinggi": "#c0392b",
}


def _band(pd_nilai: float, cutoffs: dict) -> str:
    if pd_nilai < cutoffs["q50"]:
        return "Risiko rendah"
    if pd_nilai < cutoffs["q80"]:
        return "Risiko sedang"
    if pd_nilai < cutoffs["q95"]:
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
    """PD 12 bulan terkalibrasi untuk satu pengajuan.

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
        mentah = float(bundel["model"].predict_proba(baris[bundel["features"]])[:, 1][0])
        z = np.log(np.clip(mentah, 1e-6, 1 - 1e-6) / (1 - np.clip(mentah, 1e-6, 1 - 1e-6)))
        kalibrasi = float(bundel["calibrator"].predict_proba(np.array([[z]]))[:, 1][0])
    cutoffs = bundel["risk_cutoffs"]
    band = _band(kalibrasi, cutoffs)
    return HasilPD(
        pd_kalibrasi=kalibrasi,
        pd_mentah=mentah,
        band=band,
        warna=BAND_WARNA[band],
        cutoffs=cutoffs,
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
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = _matriks_lgd(bundel, baris)
            nilai = float(bundel["model"].steps[-1][1].predict(Z)[0])
        return float(np.clip(nilai, 0.0, 1.0))
    except Exception:
        return None


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
    """Probabilitas default 6 bulan untuk panel yang sudah disiapkan.

    Kolom one-hot rating `D` disisipkan kembali (lihat catatan modul).
    """
    pipa = muat_ews()
    if pipa is None:
        return None
    try:
        ct, lr = pipa.steps[0][1], pipa.steps[-1][1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = ct.transform(df_siap[list(pipa.feature_names_in_)])
        Z = np.asarray(Z.todense()) if hasattr(Z, "todense") else np.asarray(Z)
        if Z.shape[1] == lr.coef_.shape[1] - 1:
            kamus = list(ct.transformers_[1][1].steps[-1][1].categories_[0])
            sisip = len(ct.transformers_[0][2]) + len(kamus)
            kolom_d = (df_siap["app_rating_internal"].astype(str) == "D").astype(float).values
            Z = np.hstack([Z[:, :sisip], kolom_d.reshape(-1, 1), Z[:, sisip:]])
        skor = Z @ lr.coef_.ravel() + float(lr.intercept_[0])
        return 1.0 / (1.0 + np.exp(-skor))
    except Exception:
        return None


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
    """K-Means di ruang fitur baku, ditampilkan lewat proyeksi PCA dua sumbu.

    Klaster tidak dilatih memakai label default. Label hanya dipakai sesudahnya
    untuk memberi nama tiap klaster ("kantong default", "inti sehat"), sehingga
    posisi pengajuan baru terbaca sebagai kemiripan struktur keuangan, bukan
    sebagai prediksi kedua.
    """
    abt = gold("abt_pd")
    if abt is None:
        return None
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    data = abt[abt["y_default_12bln"].notna()].copy()
    X = data[FITUR_KLASTER].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    pra = Pipeline([("imputer", SimpleImputer(strategy="median")), ("skala", StandardScaler())])
    Z = pra.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(Z)
    pca = PCA(n_components=2, random_state=7).fit(Z)
    P = pca.transform(Z)

    titik = pd.DataFrame({
        "x": P[:, 0],
        "y": P[:, 1],
        "klaster": km.labels_,
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
    for kolom in FITUR_KLASTER:
        ringkas[kolom] = [
            float(X.loc[titik["klaster"] == c, kolom].median()) for c in ringkas["klaster"]
        ]
    ringkas = ringkas.sort_values("tingkat_default", ascending=False).reset_index(drop=True)
    ringkas["nama"] = _nama_klaster(ringkas)

    pusat = pd.DataFrame(pca.transform(km.cluster_centers_), columns=["x", "y"])
    pusat["klaster"] = range(len(pusat))
    pusat = pusat.merge(ringkas[["klaster", "nama", "tingkat_default"]], on="klaster")

    return RuangKlaster(
        titik=titik, ringkas=ringkas, pusat=pusat,
        varians=(float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])),
        _pra=pra, _km=km, _pca=pca,
    )


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

    nilai = {}
    for kolom in FITUR_KLASTER:
        if kolom in baris.columns:
            nilai[kolom] = pd.to_numeric(baris[kolom].iat[0], errors="coerce")
        elif kolom == "app_plafon_thd_penjualan":
            nilai[kolom] = float(entitas.get("plafon", 0)) / max(
                float(entitas.get("penjualan_tahunan", 1)), 1.0
            )
        else:
            nilai[kolom] = np.nan
    X = pd.DataFrame([nilai])[FITUR_KLASTER]

    Z = ruang._pra.transform(X)
    P = ruang._pca.transform(Z)
    jarak = np.linalg.norm(ruang._km.cluster_centers_ - Z[0], axis=1)
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
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pk = bundel["calibrator"].predict_proba(z.reshape(-1, 1))[:, 1]
            hasil[split] = {
                **_metrik_biner(y, p, float(bundel["risk_cutoffs"]["q80"])),
                "skor": p, "skor_kalibrasi": pk, "y": y,
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

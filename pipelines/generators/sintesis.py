"""Langkah 6 rencana data: generator sintesis untuk konteks debitur Indonesia.

Semua yang dihasilkan modul ini SINTESIS dan sengaja dibuat sintesis (UU 27/2022):
nama badan hukum, CIF, NPWP, sektor KBLI, skala rupiah, pengajuan, keputusan
komite, agunan, covenant, dan snapshot kolektibilitas bulanan.

Yang TIDAK boleh lahir dari sini: label gagal bayar, rasio keuangan, LGD, dan
topologi relasi - semuanya datang dari dataset nyata lewat pipelines.transform.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from faker import Faker

from pipelines.config import RATING_ORDER, settings
from pipelines.utils import read_table, winsorize

LOG = logging.getLogger("pipelines.generators")

BENTUK_BADAN = ["PT", "PT", "PT", "CV"]
SUFIKS_USAHA = [
    "Sejahtera",
    "Makmur",
    "Abadi",
    "Nusantara",
    "Jaya",
    "Mandiri",
    "Perkasa",
    "Lestari",
    "Bersama",
    "Utama",
    "Sentosa",
    "Karya",
]

JENIS_AGUNAN = [
    ("tanah_bangunan", "APHT", 0.70),
    ("mesin_peralatan", "fidusia", 0.50),
    ("persediaan", "fidusia", 0.40),
    ("piutang_usaha", "cesie", 0.45),
    ("deposito", "gadai", 1.00),
]

KOMITE_AMBANG = [
    (25e9, "Kepala Cabang"),
    (75e9, "Komite Kredit Wilayah"),
    (float("inf"), "Komite Kredit Pusat"),
]


def _faker() -> Faker:
    fk = Faker("id_ID")
    Faker.seed(settings.seed)
    return fk


def _kelas_penjualan(penjualan_rp: pd.Series) -> pd.Series:
    return pd.cut(
        penjualan_rp,
        bins=[0, 50e9, 100e9, 200e9, np.inf],
        labels=["Rp 30-50 M", "Rp 50-100 M", "Rp 100-200 M", "Rp 200-300 M"],
    ).astype("string")


def _skala_rupiah(debitur: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Faktor pengali per cif supaya penjualan tahun terakhir masuk Rp 30-300 M.

    Peringkat penjualan asli dipertahankan (log-uniform pada rentang target),
    jadi urutan besar-kecil antar debitur tetap mengikuti data nyata.
    """
    peringkat = debitur["total_revenue"].rank(pct=True, method="first")
    lo, hi = np.log(settings.penjualan_min_rp), np.log(settings.penjualan_max_rp)
    goyangan = rng.normal(0, 0.03, size=len(debitur))
    target = np.exp(np.clip(lo + peringkat.to_numpy() * (hi - lo) + goyangan, lo, hi))
    return pd.Series(target / debitur["total_revenue"].to_numpy(), index=debitur.index)


def _skor_kredit(df: pd.DataFrame) -> pd.Series:
    """Skor 0-100 dari rasio nyata; dipakai untuk rating internal dan keputusan."""
    komponen = pd.DataFrame(
        {
            "der": -df["der"].rank(pct=True),
            "debt_to_ebitda": -df["debt_to_ebitda"].fillna(df["debt_to_ebitda"].median()).rank(pct=True),
            "icr": df["icr"].rank(pct=True),
            "roa": df["roa"].rank(pct=True),
        }
    )
    skor = komponen.mean(axis=1)
    return (skor.rank(pct=True) * 100).round(2)


def _rating_internal(skor: pd.Series) -> pd.Series:
    """Kalibrasi skor ke distribusi rating nyata dari corporate_rating.csv."""
    rating_nyata = read_table("silver", "sl_rating", columns=["rating"])
    frekuensi = rating_nyata["rating"].value_counts(normalize=True)
    frekuensi = frekuensi.reindex([r for r in RATING_ORDER if r in frekuensi.index]).dropna()
    batas = frekuensi.iloc[::-1].cumsum()  # dari rating terburuk ke terbaik

    persentil = skor.rank(pct=True)
    hasil = pd.Series(index=skor.index, dtype="object")
    sisa = pd.Series(True, index=skor.index)
    for rating, ambang in batas.items():
        pilih = sisa & (persentil <= ambang)
        hasil[pilih] = rating
        sisa &= ~pilih
    hasil[sisa] = batas.index[-1]
    return hasil.astype("string")


# ------------------------------------------------------------- DIM_DEBITUR
def buat_dim_debitur(rng: np.random.Generator) -> pd.DataFrame:
    peta = read_table("silver", "sl_peta_cif")
    sba = read_table("silver", "sl_peta_sba")
    entity = read_table(
        "silver", "sl_icij_entity", columns=["node_id", "incorporation_date", "jurisdiction"]
    )
    fk = _faker()

    df = peta.merge(
        sba[["cif_sk", "kbli_kategori", "kbli_deskripsi", "NAICS", "perusahaan_baru", "NoEmp"]],
        on="cif_sk",
        how="left",
    ).merge(entity, on="node_id", how="left")

    df["skala_rupiah"] = _skala_rupiah(df, rng)
    df["penjualan_rp"] = df["total_revenue"] * df["skala_rupiah"]
    df["kelas_penjualan"] = _kelas_penjualan(df["penjualan_rp"])

    bentuk = rng.choice(BENTUK_BADAN, size=len(df))
    inti = [fk.last_name() for _ in range(len(df))]
    sufiks = rng.choice(SUFIKS_USAHA, size=len(df))
    df["nama_badan_hukum"] = [f"{b} {i} {s}" for b, i, s in zip(bentuk, inti, sufiks)]
    df["npwp"] = [fk.numerify("##.###.###.#-###.###") for _ in range(len(df))]
    df["alamat_domisili"] = [fk.address().replace("\n", ", ") for _ in range(len(df))]
    df["kota"] = [fk.city() for _ in range(len(df))]

    # tahun_berdiri: pakai tanggal pendirian ICIJ bila masuk akal, kalau tidak
    # diturunkan dari flag NewExist SBA.
    #
    # BATAS ATASNYA WAJIB IKUT ANGKATAN. Tanpa itu perusahaan bisa "berdiri"
    # 2023 padahal mengajukan Februari 2022, dan app_umur_perusahaan_tahun di
    # abt.py (tahun_buku_terakhir - tahun_berdiri) keluar negatif. Terukur pada
    # build sebelumnya: 22 baris abt_pd berumur -1 sampai -2 tahun.
    #
    # Batasnya tahun buku terakhir, bukan tahun pengajuan: debitur harus sudah
    # ada saat laporan keuangan yang dipakai menilainya disusun.
    tahun_maks = df["angkatan"].map(
        {k: v["tahun_buku_terakhir"] for k, v in settings.angkatan.items()}
    )
    tahun_icij = pd.to_datetime(df["incorporation_date"], errors="coerce").dt.year
    tahun_acak = np.where(
        df["perusahaan_baru"].fillna(False),
        rng.integers(2018, 2025, size=len(df)),
        rng.integers(1985, 2018, size=len(df)),
    )
    tahun_berdiri = (
        tahun_icij.where(tahun_icij.between(1970, 2024)).fillna(pd.Series(tahun_acak, index=df.index))
    )
    df["tahun_berdiri"] = np.minimum(tahun_berdiri, tahun_maks).astype(int)

    df["skor_kredit"] = _skor_kredit(df)
    df["rating_internal"] = _rating_internal(df["skor_kredit"])

    dim = pd.DataFrame(
        {
            "cif_sk": df["cif_sk"],
            "cif": df["cif"],
            "nama_badan_hukum": df["nama_badan_hukum"],
            "npwp": df["npwp"],
            "alamat_domisili": df["alamat_domisili"],
            "kota": df["kota"],
            "sektor_kbli": df["kbli_kategori"],
            "sektor_deskripsi": df["kbli_deskripsi"],
            "kelas_penjualan": df["kelas_penjualan"],
            "penjualan_rp": df["penjualan_rp"].round(0),
            "tahun_berdiri": df["tahun_berdiri"],
            "grup_id": df["grup_id"],
            "angkatan": df["angkatan"],
            "rating_internal": df["rating_internal"],
            "skor_kredit": df["skor_kredit"],
            "valid_from": pd.Timestamp(settings.snapshot_awal),
            "valid_to": pd.NaT,
            "is_current": True,
            "src_us_company": df["company_name"],
            "src_icij_node_id": df["node_id"],
            "src_taiwan_row_id": df["taiwan_row_id"],
            "src_naics": df["NAICS"],
            "skala_rupiah": df["skala_rupiah"],
            "label_default_debitur": df["label_default_debitur"],
        }
    )

    dim["valid_to"] = dim["valid_to"].astype("datetime64[ns]")

    # SCD-2: sebagian debitur mengalami migrasi rating di tengah jendela.
    pindah = rng.random(len(dim)) < 0.15
    lama = dim[pindah].copy()
    if len(lama):
        tanggal_pindah = pd.Timestamp("2025-06-30")
        urutan = {r: i for i, r in enumerate(RATING_ORDER)}
        arah = rng.choice([-1, 1], size=len(lama))
        rating_lama = [
            RATING_ORDER[int(np.clip(urutan[r] + a, 0, len(RATING_ORDER) - 1))]
            for r, a in zip(lama["rating_internal"], arah)
        ]
        lama["rating_internal"] = rating_lama
        lama["valid_to"] = tanggal_pindah
        lama["is_current"] = False
        dim.loc[pindah, "valid_from"] = tanggal_pindah + pd.Timedelta(days=1)
        # valid_to pada baris versi kini memang seluruhnya NaT (belum berakhir).
        # pandas memperingatkan kolom all-NA saat concat; dtype kedua frame sudah
        # datetime64[ns] eksplisit di atas, jadi perilaku barunya tidak mengubah
        # apa pun di sini.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            dim = pd.concat([lama, dim], ignore_index=True)

    dim = dim.sort_values(["cif_sk", "valid_from"]).reset_index(drop=True)
    dim.insert(0, "debitur_sk", np.arange(1, len(dim) + 1))
    return dim


# ------------------------------------------------------- FACT_LAPORAN_KEUANGAN
def buat_fact_laporan_keuangan(dim_debitur: pd.DataFrame) -> pd.DataFrame:
    """Panel 3 tahun: rasio dari data nyata, nominal diskala ke rupiah."""
    panel = read_table("silver", "sl_panel_terpilih")
    taiwan = read_table("silver", "sl_taiwan_ratio")
    peta = read_table("silver", "sl_peta_cif", columns=["cif_sk", "taiwan_row_id"])
    skala = dim_debitur[dim_debitur["is_current"]][["cif_sk", "skala_rupiah"]]

    df = panel.merge(skala, on="cif_sk", how="left")
    for kolom, hasil in (
        ("total_revenue", "penjualan_rp"),
        ("total_assets", "total_aset_rp"),
        ("total_liabilities", "total_liabilitas_rp"),
        ("ekuitas", "ekuitas_rp"),
        ("ebitda", "ebitda_rp"),
        ("net_income", "laba_bersih_rp"),
    ):
        df[hasil] = (df[kolom] * df["skala_rupiah"]).round(0)

    # Blok rasio tambahan dari baris Taiwan yang ditempelkan (langkah 2).
    kolom_taiwan = [
        "taiwan_row_id",
        "current_ratio",
        "quick_ratio",
        "cash_to_ta",
        "wc_to_ta",
        "re_to_ta",
        "gross_margin",
        "cfo_to_sales",
        "op_profit_to_paid_in_capital",
    ]
    tw = taiwan[kolom_taiwan].add_prefix("tw_").rename(columns={"tw_taiwan_row_id": "taiwan_row_id"})
    df = df.merge(peta, on="cif_sk", how="left").merge(tw, on="taiwan_row_id", how="left")

    # Tren 3 tahun (proposal §6: DER, ICR, debt/EBITDA dan trennya).
    df = df.sort_values(["cif_sk", "tahun_buku"])
    g = df.groupby("cif_sk", sort=False)
    turunan = []
    for kolom in ("der", "icr", "debt_to_ebitda", "penjualan_rp"):
        df[f"{kolom}_yoy"] = (
            g[kolom].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        )
        df[f"{kolom}_delta_3thn"] = g[kolom].transform(lambda s: s - s.iloc[0])
        turunan += [f"{kolom}_yoy", f"{kolom}_delta_3thn"]

    # Winsorisasi ulang untuk kolom turunan.
    #
    # Winsorisasi di silver memotong rasio SUMBER-nya, tapi tren dihitung ulang
    # di sini, dan pct_change bisa meledak lagi walau pembilang sudah dipotong:
    # cukup tahun sebelumnya mendekati nol. penjualan_rp_yoy paling parah karena
    # penjualan_rp adalah pos rupiah yang tidak ikut dipotong di silver sama
    # sekali - terukur mencapai 12.739 (1,27 juta persen) sebelum ini ada.
    for kolom in turunan:
        df[kolom] = winsorize(df[kolom])

    fact = pd.DataFrame(
        {
            "cif_sk": df["cif_sk"],
            "tahun_buku": df["tahun_buku"],
            "angkatan": df["angkatan"],
            "is_tahun_terakhir": df["is_tahun_terakhir"],
            "penjualan_rp": df["penjualan_rp"],
            "total_aset_rp": df["total_aset_rp"],
            "total_liabilitas_rp": df["total_liabilitas_rp"],
            "ekuitas_rp": df["ekuitas_rp"],
            "ebitda_rp": df["ebitda_rp"],
            "laba_bersih_rp": df["laba_bersih_rp"],
            "der": df["der"],
            "debt_to_ebitda": df["debt_to_ebitda"],
            "icr": df["icr"],
            "roa": df["roa"],
            "cfo_to_ebitda": df["cfo_to_ebitda"],
            "cfo_to_liability": df["cfo_to_liability"],
            "operating_margin": df["operating_margin"],
            "gross_margin": df["gross_margin"],
            "current_ratio": df["current_ratio"],
            "quick_ratio": df["quick_ratio"],
            "asset_turnover": df["asset_turnover"],
            "wc_to_ta": df["wc_to_ta"],
            "re_to_ta": df["re_to_ta"],
            "dso_hari": df["dso_hari"],
            "dio_hari": df["dio_hari"],
            "siklus_modal_kerja_hari": df["siklus_modal_kerja_hari"],
            "der_yoy": df["der_yoy"],
            "icr_yoy": df["icr_yoy"],
            "debt_to_ebitda_yoy": df["debt_to_ebitda_yoy"],
            "der_delta_3thn": df["der_delta_3thn"],
            "icr_delta_3thn": df["icr_delta_3thn"],
            "debt_to_ebitda_delta_3thn": df["debt_to_ebitda_delta_3thn"],
            "growth_penjualan": df["penjualan_rp_yoy"],
            "tw_current_ratio": df["tw_current_ratio"],
            "tw_quick_ratio": df["tw_quick_ratio"],
            "tw_cash_to_ta": df["tw_cash_to_ta"],
            "tw_cfo_to_sales": df["tw_cfo_to_sales"],
            "label_default": df["label_default"],
            "sumber": "us_panel+taiwan",
            "src_us_row_id": df["us_row_id"],
            "src_taiwan_row_id": df["taiwan_row_id"],
        }
    ).reset_index(drop=True)
    fact.insert(0, "lk_id", np.arange(1, len(fact) + 1))
    return fact


# --------------------------------------------------- DIM_PRODUK_FASILITAS
def buat_dim_produk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, "Kredit Modal Kerja Revolving", "modal_kerja", True, 12),
            (2, "Kredit Modal Kerja Transaksional", "modal_kerja", True, 24),
            (3, "Kredit Investasi Mesin", "investasi", False, 60),
            (4, "Kredit Investasi Bangunan", "investasi", False, 120),
            (5, "Kredit Investasi Ekspansi", "investasi", False, 84),
        ],
        columns=["produk_id", "nama_produk", "jenis_fasilitas", "revolving", "tenor_maks_bulan"],
    )


# ------------------------------------------------------------ FACT_PENGAJUAN
def buat_fact_pengajuan(
    dim_debitur: pd.DataFrame, produk: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    sba = read_table("silver", "sl_peta_sba")
    kini = dim_debitur[dim_debitur["is_current"]].copy()
    df = kini.merge(
        sba[["cif_sk", "Term", "revolving", "jenis_fasilitas", "src_sba_loannr", "dokumen_ringkas"]],
        on="cif_sk",
        how="left",
    )

    # Tiap angkatan punya jendela pengajuannya sendiri. Buku lama mengajukan
    # 2022-2023 dan menghasilkan riwayat gagal bayar; buku baru mengajukan 2025
    # dan karena itu bisa melihat riwayat tersebut lewat fitur graf.
    tanggal = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for nama, par in settings.angkatan.items():
        pilih = df["angkatan"] == nama
        if not pilih.any():
            continue
        awal = pd.Timestamp(par["awal_pengajuan"])
        rentang = (pd.Timestamp(par["akhir_pengajuan"]) - awal).days
        tanggal[pilih] = awal + pd.to_timedelta(
            rng.integers(0, rentang + 1, int(pilih.sum())), unit="D"
        )
    df["tanggal_pengajuan"] = tanggal

    # Plafon: 15-45% penjualan tahunan, dipangkas ke rentang Rp 10-150 M.
    porsi = rng.uniform(0.15, 0.45, len(df))
    df["plafon_diminta_rp"] = (
        (df["penjualan_rp"] * porsi)
        .clip(settings.plafon_min_rp, settings.plafon_max_rp)
        .round(-6)
    )

    # Tenor dari SBA Term (NYATA), dipangkas ke tenor maksimum produk.
    pilih = []
    for jenis in df["jenis_fasilitas"].fillna("modal_kerja"):
        kandidat = produk[produk["jenis_fasilitas"] == jenis]
        pilih.append(int(kandidat["produk_id"].iloc[rng.integers(0, len(kandidat))]))
    df["produk_id"] = pilih
    tenor_maks = df["produk_id"].map(produk.set_index("produk_id")["tenor_maks_bulan"])
    df["tenor_bulan"] = np.minimum(df["Term"].fillna(12), tenor_maks).astype(int)

    # Keputusan komite: skor kredit + rasio plafon terhadap penjualan.
    beban = (df["plafon_diminta_rp"] / df["penjualan_rp"]).clip(0, 1)
    peluang_setuju = (0.55 + 0.006 * df["skor_kredit"] - 0.35 * beban).clip(0.05, 0.97)
    undian = rng.random(len(df))
    df["keputusan"] = np.select(
        [undian < peluang_setuju * 0.8, undian < peluang_setuju],
        ["setuju", "setuju_dengan_syarat"],
        default="tolak",
    )

    urutan = {r: i for i, r in enumerate(RATING_ORDER)}
    tingkat_rating = df["rating_internal"].map(urutan).fillna(5)
    df["pricing_bps"] = (
        650 + 55 * tingkat_rating + rng.normal(0, 25, len(df)) - 0.6 * df["skor_kredit"]
    ).clip(600, 1800).round().astype(int)

    df["komite_level"] = pd.cut(
        df["plafon_diminta_rp"],
        bins=[0] + [a for a, _ in KOMITE_AMBANG],
        labels=[n for _, n in KOMITE_AMBANG],
    ).astype("string")

    fact = df[
        [
            "cif_sk",
            "produk_id",
            "tanggal_pengajuan",
            "plafon_diminta_rp",
            "tenor_bulan",
            "keputusan",
            "pricing_bps",
            "komite_level",
            "dokumen_ringkas",
            "src_sba_loannr",
        ]
    ].copy()
    fact = fact.sort_values(["tanggal_pengajuan", "cif_sk"]).reset_index(drop=True)
    fact.insert(0, "application_id", np.arange(1, len(fact) + 1))
    return fact


# ------------------------------------------------------------ FACT_FASILITAS
def buat_fact_fasilitas(
    pengajuan: pd.DataFrame, dim_debitur: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    disetujui = pengajuan[pengajuan["keputusan"] != "tolak"].copy()
    kini = dim_debitur[dim_debitur["is_current"]][["cif_sk", "skor_kredit", "penjualan_rp"]]
    df = disetujui.merge(kini, on="cif_sk", how="left")

    potongan = np.where(df["keputusan"] == "setuju_dengan_syarat", rng.uniform(0.6, 0.9, len(df)), 1.0)
    df["plafon_rp"] = (df["plafon_diminta_rp"] * potongan).round(-6)
    df["tanggal_pencairan"] = df["tanggal_pengajuan"] + pd.to_timedelta(
        rng.integers(7, 45, len(df)), unit="D"
    )
    df["tanggal_jatuh_tempo"] = df["tanggal_pencairan"] + pd.to_timedelta(
        df["tenor_bulan"] * 30, unit="D"
    )
    # Fasilitas revolving diperpanjang otomatis; perilaku bayarnya tetap diamati
    # minimal 24 bulan supaya jendela EWS punya isi.
    df["tanggal_akhir_observasi"] = np.maximum(
        df["tanggal_jatuh_tempo"],
        df["tanggal_pencairan"] + pd.DateOffset(months=24),
    )

    pemakaian = np.clip(rng.beta(5, 2, len(df)) + (60 - df["skor_kredit"]) / 400, 0.15, 1.0)
    df["pemakaian_plafon_pct"] = pemakaian.round(4)
    df["outstanding_rp"] = (df["plafon_rp"] * pemakaian).round(-6)
    df["frekuensi_overdraft_12bln"] = rng.poisson(
        np.clip(4 - df["skor_kredit"] / 30, 0.2, 6), len(df)
    )

    fasilitas = df[
        [
            "application_id",
            "cif_sk",
            "produk_id",
            "plafon_rp",
            "outstanding_rp",
            "pemakaian_plafon_pct",
            "frekuensi_overdraft_12bln",
            "tenor_bulan",
            "tanggal_pencairan",
            "tanggal_jatuh_tempo",
            "tanggal_akhir_observasi",
            "pricing_bps",
            "src_sba_loannr",
        ]
    ].reset_index(drop=True)
    fasilitas.insert(0, "facility_id", np.arange(1, len(fasilitas) + 1))
    return fasilitas


# --------------------------------------------------------------- FACT_AGUNAN
def buat_fact_agunan(
    fasilitas: pd.DataFrame, dim_debitur: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    grup = dim_debitur[dim_debitur["is_current"]][["cif_sk", "grup_id"]]
    df = fasilitas.merge(grup, on="cif_sk", how="left")

    baris = []
    jumlah_agunan = rng.integers(1, 4, len(df))
    for (_, fas), n in zip(df.iterrows(), jumlah_agunan):
        sisa_target = fas["plafon_rp"] * rng.uniform(1.0, 1.8)
        for i in range(int(n)):
            jenis, pengikatan, haircut = JENIS_AGUNAN[int(rng.integers(0, len(JENIS_AGUNAN)))]
            porsi = 1.0 / n
            nilai = sisa_target * porsi / haircut
            baris.append(
                {
                    "facility_id": int(fas["facility_id"]),
                    "jenis": jenis,
                    "nilai_taksasi_rp": round(nilai, -6),
                    "haircut": haircut,
                    "nilai_likuidasi_rp": round(nilai * haircut, -6),
                    "status_pengikatan": pengikatan,
                    "dijaminkan_silang": bool(rng.random() < 0.12),
                    "tanggal_taksasi": fas["tanggal_pencairan"] - pd.Timedelta(days=int(rng.integers(15, 90))),
                }
            )
    agunan = pd.DataFrame(baris)
    total = agunan.groupby("facility_id")["nilai_likuidasi_rp"].transform("sum")
    plafon = agunan["facility_id"].map(fasilitas.set_index("facility_id")["plafon_rp"])
    agunan["coverage_ratio"] = (total / plafon).round(4)
    agunan.insert(0, "agunan_id", np.arange(1, len(agunan) + 1))
    return agunan


# ------------------------------------------------------------- FACT_COVENANT
AMBANG_COVENANT = {
    "der_maks": {"AAA": 1.5, "AA": 1.75, "A": 2.0, "BBB": 2.5, "BB": 3.0, "B": 3.5},
    "icr_min": {"AAA": 4.0, "AA": 3.5, "A": 3.0, "BBB": 2.5, "BB": 2.0, "B": 1.5},
    "debt_to_ebitda_maks": {"AAA": 2.0, "AA": 2.5, "A": 3.0, "BBB": 3.5, "BB": 4.0, "B": 4.5},
}


def buat_fact_covenant(
    fasilitas: pd.DataFrame,
    laporan: pd.DataFrame,
    dim_debitur: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Posisi covenant bulanan: ambang dari kelas rating, nilai aktual dari rasio nyata."""
    rasio_akhir = (
        laporan[laporan["is_tahun_terakhir"]]
        .set_index("cif_sk")[["der", "icr", "debt_to_ebitda"]]
    )
    rating = dim_debitur[dim_debitur["is_current"]].set_index("cif_sk")["rating_internal"]

    snapshot = pd.to_datetime(pd.Series(settings.snapshot_dates))
    baris = []
    for _, fas in fasilitas.iterrows():
        cif = fas["cif_sk"]
        kelas = rating.get(cif, "BB")
        bulan = snapshot[
            (snapshot >= fas["tanggal_pencairan"])
            & (snapshot <= fas["tanggal_akhir_observasi"])
        ]
        if bulan.empty:
            continue
        dasar = rasio_akhir.loc[cif] if cif in rasio_akhir.index else None
        for jenis, tabel in AMBANG_COVENANT.items():
            ambang = tabel.get(str(kelas), list(tabel.values())[-1])
            kolom = jenis.replace("_maks", "").replace("_min", "")
            nilai_awal = float(dasar[kolom]) if dasar is not None and pd.notna(dasar[kolom]) else ambang
            jalan = nilai_awal * np.cumprod(1 + rng.normal(0, 0.03, len(bulan)))
            for tanggal, nilai in zip(bulan, jalan):
                melanggar = nilai > ambang if jenis.endswith("maks") else nilai < ambang
                baris.append(
                    {
                        "facility_id": int(fas["facility_id"]),
                        "snapshot_date": tanggal,
                        "jenis": jenis,
                        "ambang": ambang,
                        "nilai_aktual": round(float(nilai), 4),
                        "status": "langgar" if melanggar else "patuh",
                    }
                )
    covenant = pd.DataFrame(baris)
    covenant.insert(0, "covenant_id", np.arange(1, len(covenant) + 1))
    return covenant


# ------------------------------------------- FACT_KOLEKTIBILITAS & FACT_DEFAULT
# Jendela umur fasilitas saat default bisa teramati (hari sejak pencairan).
UMUR_DEFAULT_MIN_HARI = 60
UMUR_DEFAULT_MAKS_HARI = 730


def _umur_ke_default(
    fasilitas: pd.DataFrame, sba: pd.DataFrame, label: pd.Series
) -> pd.Series:
    """Petakan hari_ke_default SBA ke jendela observasi tanpa merusak urutannya.

    hari_ke_default di SBA bermedian 1.314 hari (P10 694, P90 2.344) - jauh di
    luar jendela 24 bulan yang teramati di sini. Memangkasnya begitu saja
    (min(hari, sisa_jendela)) membuat seluruh default menumpuk di ujung jendela:
    bad rate 12 bulan jatuh ke 0,15% dan PD 12 bulan tidak bisa dilatih sama
    sekali.

    Yang dipakai sekarang adalah penskalaan monoton berbasis peringkat. Debitur
    yang di data nyata gagal lebih cepat tetap gagal lebih cepat relatif terhadap
    yang lain, tapi sebarannya mengisi seluruh jendela. Perlakuannya sama dengan
    penskalaan timestamp AML dan tercatat di docs/data-lineage.md sebagai
    transformasi SINTESIS.

    Fasilitas yang umur hasil pemetaannya melewati batas observasinya sendiri
    TIDAK dipaksa default - ia menjadi observasi tersensor kanan, dan itu memang
    keadaan yang sebenarnya.
    """
    calon = fasilitas[fasilitas["cif_sk"].map(label).fillna(0).astype(bool)]
    if calon.empty:
        return pd.Series(dtype="float64")

    hari_sba = calon["cif_sk"].map(sba["hari_ke_default"])
    hari_sba = hari_sba.fillna(hari_sba.median())
    peringkat = hari_sba.rank(pct=True, method="first")
    umur = UMUR_DEFAULT_MIN_HARI + peringkat * (UMUR_DEFAULT_MAKS_HARI - UMUR_DEFAULT_MIN_HARI)
    hasil = pd.Series(umur.round().to_numpy(), index=calon["facility_id"].to_numpy())

    # Sumber afiliasi tersembunyi dipaksa jatuh lebih awal, supaya kolapsnya
    # sempat terlihat sebelum anggota buku baru mengajukan (langkah 7).
    from pipelines.utils import table_exists

    if settings.injeksi_afiliasi and table_exists("silver", "sl_afiliasi_tersembunyi"):
        from pipelines.generators.afiliasi import umur_default_paksa

        klaster = read_table("silver", "sl_afiliasi_tersembunyi")
        paksa = umur_default_paksa(klaster)
        if len(paksa):
            per_cif = calon.set_index("facility_id")["cif_sk"]
            cocok = per_cif.map(paksa).dropna()
            hasil.loc[cocok.index] = cocok.to_numpy()
    return hasil


def buat_kolektibilitas_dan_default(
    fasilitas: pd.DataFrame,
    dim_debitur: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Migrasi kolektibilitas bulanan; yang default memakai LGD nyata dari SBA."""
    sba = read_table("silver", "sl_peta_sba").set_index("cif_sk")
    label = dim_debitur[dim_debitur["is_current"]].set_index("cif_sk")["label_default_debitur"]

    snapshot = pd.to_datetime(pd.Series(settings.snapshot_dates))
    batas_akhir = pd.Timestamp(settings.snapshot_akhir)
    umur_default = _umur_ke_default(fasilitas, sba, label)

    baris_kol, baris_def = [], []
    for _, fas in fasilitas.iterrows():
        cif = int(fas["cif_sk"])
        bulan = snapshot[snapshot >= fas["tanggal_pencairan"]]
        bulan = bulan[bulan <= min(fas["tanggal_akhir_observasi"], batas_akhir)]
        if bulan.empty:
            continue

        akan_default = bool(label.get(cif, 0))
        tanggal_default = pd.NaT
        if akan_default:
            umur = umur_default.get(int(fas["facility_id"]), np.nan)
            horizon = (bulan.iloc[-1] - fas["tanggal_pencairan"]).days
            # Umur yang melewati horizon fasilitas ini dibiarkan tersensor kanan,
            # BUKAN dipangkas ke ujung jendela.
            if pd.notna(umur) and UMUR_DEFAULT_MIN_HARI <= umur <= horizon:
                tanggal_default = fas["tanggal_pencairan"] + pd.Timedelta(days=int(umur))

        # Episode tekanan pada fasilitas yang TIDAK berakhir default. Tanpa ini,
        # kolektibilitas >= 3 hanya pernah muncul menjelang default, sehingga
        # P(default | kol=3) = 1,000 dan model EWS cuma menghafal aturan
        # generator - bukan belajar. Di bank sungguhan ada debitur yang jatuh ke
        # kol 3 lalu pulih (cured), dan itu justru kasus yang paling menarik.
        episode_tekanan: set[int] = set()
        if pd.isna(tanggal_default) and rng.random() < 0.07 and len(bulan) >= 4:
            mulai = int(rng.integers(0, max(1, len(bulan) - 3)))
            episode_tekanan = set(range(mulai, min(mulai + int(rng.integers(2, 5)), len(bulan))))

        for i, tanggal in enumerate(bulan):
            if pd.notna(tanggal_default):
                bulan_ke_default = (tanggal_default - tanggal).days / 30.0
                if bulan_ke_default <= 0:
                    kol = 5
                elif bulan_ke_default <= 1:
                    kol = 4
                elif bulan_ke_default <= 2:
                    kol = 3
                elif bulan_ke_default <= 4:
                    kol = 2
                else:
                    kol = 1
                # Sebagian debitur terlihat masih sehat sampai dekat sekali ke
                # gagal bayar. Tanpa derau ini, kol adalah fungsi deterministik
                # dari jarak ke tanggal default.
                if kol in (2, 3, 4) and rng.random() < 0.30:
                    kol -= 1
            elif i in episode_tekanan:
                kol = 3 if rng.random() < 0.6 else 2
            else:
                kol = 1 if rng.random() > 0.06 else 2
            dpd = {1: 0, 2: int(rng.integers(1, 90)), 3: int(rng.integers(91, 120)),
                   4: int(rng.integers(121, 180)), 5: int(rng.integers(181, 365))}[kol]
            baris_kol.append(
                {
                    "facility_id": int(fas["facility_id"]),
                    "cif_sk": cif,
                    "snapshot_date": tanggal,
                    "kolektibilitas": kol,
                    "dpd": dpd,
                    "flag_restrukturisasi": bool(kol in (3, 4) and rng.random() < 0.3),
                    "outstanding_rp": float(fas["outstanding_rp"]),
                }
            )

        if pd.notna(tanggal_default):
            lgd = sba.loc[cif, "lgd_realisasi"] if cif in sba.index else np.nan
            lgd = float(lgd) if pd.notna(lgd) else 0.45
            ead = float(fas["outstanding_rp"])
            baris_def.append(
                {
                    "facility_id": int(fas["facility_id"]),
                    "cif_sk": cif,
                    "tanggal_default": tanggal_default,
                    "ead_rp": ead,
                    "lgd_realisasi": round(lgd, 4),
                    "jumlah_pemulihan_rp": round(ead * (1 - lgd), -6),
                    "src_sba_loannr": sba.loc[cif, "src_sba_loannr"] if cif in sba.index else None,
                }
            )

    kolektibilitas = pd.DataFrame(baris_kol)
    default = pd.DataFrame(baris_def)
    LOG.info("kolektibilitas %s baris, default %s fasilitas", len(kolektibilitas), len(default))
    return kolektibilitas, default


# ------------------------------------------- DIM_GRUP_USAHA & FACT_EKSPOSUR_GRUP
def buat_grup_dan_eksposur(
    dim_debitur: pd.DataFrame,
    fasilitas: pd.DataFrame,
    kolektibilitas: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fk = _faker()
    kini = dim_debitur[dim_debitur["is_current"]]
    kedalaman = read_table("silver", "sl_kedalaman_grup") if _ada_kedalaman() else None

    grup = (
        kini.groupby("grup_id")
        .agg(jumlah_entitas=("cif_sk", "nunique"), penjualan_grup_rp=("penjualan_rp", "sum"))
        .reset_index()
    )
    grup["nama_grup"] = [f"Grup {fk.last_name()}" for _ in range(len(grup))]
    if kedalaman is not None:
        grup = grup.merge(kedalaman, on="grup_id", how="left")
    grup["kedalaman_kepemilikan"] = grup.get(
        "kedalaman_kepemilikan", pd.Series(1, index=grup.index)
    ).fillna(1).astype(int)
    # BMPK: 25% modal bank sintetis (Rp 12 T) untuk satu grup peminjam.
    modal_bank_rp = 12e12
    grup["batas_bmpk_rp"] = 0.25 * modal_bank_rp

    fas_grup = fasilitas.merge(kini[["cif_sk", "grup_id"]], on="cif_sk", how="left")
    kol = kolektibilitas.merge(
        fas_grup[["facility_id", "grup_id"]], on="facility_id", how="left"
    )
    eksposur = (
        kol.groupby(["grup_id", "snapshot_date"])["outstanding_rp"].sum().reset_index()
    )
    eksposur = eksposur.rename(columns={"outstanding_rp": "total_eksposur_rp"})
    eksposur = eksposur.merge(grup[["grup_id", "batas_bmpk_rp"]], on="grup_id", how="left")
    eksposur["group_exposure_share"] = (
        eksposur["total_eksposur_rp"] / eksposur["batas_bmpk_rp"]
    ).round(6)
    eksposur["sisa_ruang_rp"] = (
        eksposur["batas_bmpk_rp"] - eksposur["total_eksposur_rp"]
    ).round(0)
    return grup, eksposur


def _ada_kedalaman() -> bool:
    from pipelines.utils import table_exists

    return table_exists("silver", "sl_kedalaman_grup")

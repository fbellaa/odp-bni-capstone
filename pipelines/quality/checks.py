"""Gerbang kualitas data untuk layer gold.

Uji paling penting di berkas ini adalah uji kebocoran waktu (§7.4): pipeline
HARUS gagal kalau ada edge dengan valid_from lebih baru dari snapshot yang
memakainya, atau kalau fitur graf sebuah pengajuan dihitung pada bulan pengajuan
itu sendiri. Uji ini dibuat sebelum fiturnya ada, bukan sesudah.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd

from pipelines.config import QUALITY_DIR, settings
from pipelines.utils import read_table, table_exists, write_table

LOG = logging.getLogger("pipelines.quality")


@dataclass
class Hasil:
    baris: list[dict] = field(default_factory=list)

    def catat(self, nama: str, lolos: bool, detail: str, kritis: bool = True) -> None:
        self.baris.append(
            {
                "uji": nama,
                "lolos": bool(lolos),
                "kritis": kritis,
                "detail": detail,
                "dijalankan_pada": pd.Timestamp.now(),
            }
        )
        LOG.log(
            logging.INFO if lolos else logging.ERROR,
            "[%s] %s - %s",
            "LOLOS" if lolos else "GAGAL",
            nama,
            detail,
        )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.baris)


# ------------------------------------------------------------ uji anti-bocor
def uji_kebocoran_waktu(h: Hasil) -> None:
    """Tiga lapis uji kebocoran pada lapisan graf."""
    feat = read_table("gold", "feat_graf_pit")
    pengajuan = read_table("gold", "fact_pengajuan", columns=["application_id", "tanggal_pengajuan"])
    edges = read_table("gold", "gold_graph_edges")
    snap = read_table("gold", "graph_snapshot_bulanan", columns=["node_id", "snapshot_date", "degree"])

    # 1. snapshot fitur harus berada di akhir bulan SEBELUM tanggal pengajuan
    gabung = feat.merge(pengajuan, on="application_id", how="left")
    awal_bulan_pengajuan = gabung["tanggal_pengajuan"].values.astype("datetime64[M]")
    melanggar = (gabung["snapshot_date"].to_numpy().astype("datetime64[ns]") >= awal_bulan_pengajuan)
    h.catat(
        "pit_snapshot_sebelum_bulan_pengajuan",
        not melanggar.any(),
        f"{int(melanggar.sum())} dari {len(gabung)} pengajuan memakai snapshot di bulan T atau sesudahnya",
    )

    # 2. derajat pada snapshot harus sama dengan derajat yang dihitung ulang
    #    hanya dari edge yang sudah valid pada tanggal itu
    tanggal_uji = pd.to_datetime(pd.Series(snap["snapshot_date"].unique()))
    dipilih = tanggal_uji.sample(min(3, len(tanggal_uji)), random_state=settings.seed)
    selisih_maks = 0
    for tanggal in dipilih:
        aktif = edges[
            (edges["valid_from"] <= tanggal)
            & (edges["valid_to"].isna() | (edges["valid_to"] > tanggal))
        ]
        g = nx.Graph()
        g.add_edges_from(zip(aktif["src_node_id"], aktif["dst_node_id"]))
        derajat_ulang = pd.Series(dict(g.degree()), dtype="float64")
        tersimpan = snap[snap["snapshot_date"] == tanggal].set_index("node_id")["degree"]
        contoh = tersimpan[tersimpan > 0].sample(
            min(500, int((tersimpan > 0).sum())), random_state=settings.seed
        )
        selisih = (contoh - derajat_ulang.reindex(contoh.index).fillna(0)).abs().max()
        selisih_maks = max(selisih_maks, float(selisih))
    h.catat(
        "snapshot_hanya_pakai_edge_valid",
        selisih_maks == 0,
        f"selisih derajat maksimum saat dihitung ulang = {selisih_maks}",
    )

    # 3. filter valid_from memang menggigit: harus ada edge yang tersaring keluar
    snapshot_paling_awal = pd.to_datetime(snap["snapshot_date"]).min()
    tersaring = int((edges["valid_from"] > snapshot_paling_awal).sum())
    h.catat(
        "filter_valid_from_aktif",
        tersaring > 0,
        f"{tersaring} edge punya valid_from setelah snapshot paling awal dan tersaring keluar",
    )


def uji_pemisahan_label(h: Hasil) -> None:
    """src_is_laundering dan label_default tidak boleh berada di satu tabel fitur."""
    feat = read_table("gold", "feat_graf_pit")
    dilarang = [c for c in feat.columns if c in ("src_is_laundering", "label_default")]
    h.catat(
        "feat_graf_pit_bebas_label",
        not dilarang,
        f"kolom terlarang di FEAT_GRAF_PIT: {dilarang or 'tidak ada'}",
    )

    transfer = read_table("gold", "fact_transfer_giro")
    h.catat(
        "is_laundering_hanya_di_transfer",
        "label_default" not in transfer.columns,
        "FACT_TRANSFER_GIRO tidak memuat label_default",
    )

    lk = read_table("gold", "fact_laporan_keuangan")
    h.catat(
        "blok_graf_bisa_didrop_utuh",
        not any(c.startswith(("supplier_", "buyer_", "neighbor_", "community_", "node_emb")) for c in lk.columns),
        "FACT_LAPORAN_KEUANGAN tidak tercampur fitur graf (uji ablasi §7.3 tetap mungkin)",
    )


# -------------------------------------------------------- uji struktur tabel
KUNCI_PRIMER = {
    "dim_debitur": ["debitur_sk"],
    "dim_grup_usaha": ["grup_id"],
    "dim_produk_fasilitas": ["produk_id"],
    "dim_pihak": ["pihak_id"],
    "dim_counterparty": ["cp_id"],
    "fact_laporan_keuangan": ["lk_id"],
    "fact_pengajuan": ["application_id"],
    "fact_fasilitas": ["facility_id"],
    "fact_agunan": ["agunan_id"],
    "fact_covenant": ["covenant_id"],
    "fact_default": ["facility_id"],
    "fact_kolektibilitas": ["facility_id", "snapshot_date"],
    "fact_eksposur_grup": ["grup_id", "snapshot_date"],
    "feat_graf_pit": ["application_id"],
    "gold_graph_nodes": ["node_id"],
    "gold_graph_edges": ["edge_id"],
    "graph_snapshot_bulanan": ["node_id", "snapshot_date"],
    "bridge_rekening": ["rekening_id"],
    "fact_transfer_giro": ["transfer_id"],
}

RELASI_FK = [
    ("fact_laporan_keuangan", "cif_sk", "dim_debitur", "cif_sk"),
    ("fact_pengajuan", "cif_sk", "dim_debitur", "cif_sk"),
    ("fact_pengajuan", "produk_id", "dim_produk_fasilitas", "produk_id"),
    ("fact_fasilitas", "application_id", "fact_pengajuan", "application_id"),
    ("fact_agunan", "facility_id", "fact_fasilitas", "facility_id"),
    ("fact_covenant", "facility_id", "fact_fasilitas", "facility_id"),
    ("fact_kolektibilitas", "facility_id", "fact_fasilitas", "facility_id"),
    ("fact_default", "facility_id", "fact_fasilitas", "facility_id"),
    ("fact_eksposur_grup", "grup_id", "dim_grup_usaha", "grup_id"),
    ("feat_graf_pit", "application_id", "fact_pengajuan", "application_id"),
    ("fact_kepemilikan", "pihak_id", "dim_pihak", "pihak_id"),
    ("fact_kepengurusan", "pihak_id", "dim_pihak", "pihak_id"),
    ("gold_graph_edges", "src_node_id", "gold_graph_nodes", "node_id"),
    ("gold_graph_edges", "dst_node_id", "gold_graph_nodes", "node_id"),
]


def uji_kunci(h: Hasil) -> None:
    for tabel, kunci in KUNCI_PRIMER.items():
        df = read_table("gold", tabel, columns=kunci)
        duplikat = int(df.duplicated(subset=kunci).sum())
        h.catat(f"pk_unik::{tabel}", duplikat == 0, f"{duplikat} baris duplikat pada {kunci}")


def uji_integritas_referensial(h: Hasil) -> None:
    for tabel_anak, kolom_anak, tabel_induk, kolom_induk in RELASI_FK:
        anak = read_table("gold", tabel_anak, columns=[kolom_anak])[kolom_anak].dropna()
        induk = set(read_table("gold", tabel_induk, columns=[kolom_induk])[kolom_induk].dropna())
        yatim = int((~anak.isin(induk)).sum())
        h.catat(
            f"fk::{tabel_anak}.{kolom_anak}->{tabel_induk}",
            yatim == 0,
            f"{yatim} nilai tidak punya induk",
        )


# ------------------------------------------------------------ uji nilai bisnis
def uji_rentang_nilai(h: Hasil) -> None:
    lk = read_table("gold", "fact_laporan_keuangan")
    debitur = read_table("gold", "dim_debitur")
    fasilitas = read_table("gold", "fact_fasilitas")
    kolek = read_table("gold", "fact_kolektibilitas")
    default = read_table("gold", "fact_default")
    agunan = read_table("gold", "fact_agunan")

    kini = debitur[debitur["is_current"]]
    di_luar = int(
        (~kini["penjualan_rp"].between(settings.penjualan_min_rp * 0.9, settings.penjualan_max_rp * 1.1)).sum()
    )
    h.catat(
        "penjualan_dalam_rentang_segmen_komersial",
        di_luar == 0,
        f"{di_luar} debitur di luar Rp 30-300 M",
    )

    di_luar = int(
        (~fasilitas["plafon_rp"].between(settings.plafon_min_rp * 0.5, settings.plafon_max_rp * 1.1)).sum()
    )
    h.catat("plafon_dalam_rentang", di_luar == 0, f"{di_luar} fasilitas di luar rentang plafon")

    salah = int((~kolek["kolektibilitas"].between(1, 5)).sum())
    h.catat("kolektibilitas_1_sampai_5", salah == 0, f"{salah} baris di luar 1-5")

    salah = int((~default["lgd_realisasi"].between(0, 1)).sum())
    h.catat("lgd_realisasi_0_sampai_1", salah == 0, f"{salah} baris LGD di luar [0,1]")

    salah = int((agunan["coverage_ratio"] <= 0).sum())
    h.catat("coverage_ratio_positif", salah == 0, f"{salah} agunan dengan coverage <= 0")

    tahun_per_cif = lk.groupby("cif_sk")["tahun_buku"].nunique()
    kurang = int((tahun_per_cif != settings.panel_years).sum())
    h.catat(
        "panel_lengkap_3_tahun",
        kurang == 0,
        f"{kurang} debitur tidak punya {settings.panel_years} tahun buku",
    )

    urut = lk.sort_values(["cif_sk", "tahun_buku"])
    h.catat(
        "tahun_buku_sebelum_pengajuan",
        int(urut["tahun_buku"].max()) <= settings.tahun_buku_terakhir,
        f"tahun buku maksimum {int(urut['tahun_buku'].max())}",
    )


def uji_konsistensi_waktu(h: Hasil) -> None:
    fasilitas = read_table("gold", "fact_fasilitas")
    pengajuan = read_table("gold", "fact_pengajuan", columns=["application_id", "tanggal_pengajuan"])
    default = read_table("gold", "fact_default")

    gabung = fasilitas.merge(pengajuan, on="application_id", how="left")
    salah = int((gabung["tanggal_pencairan"] < gabung["tanggal_pengajuan"]).sum())
    h.catat("pencairan_setelah_pengajuan", salah == 0, f"{salah} fasilitas cair sebelum diajukan")

    gabung_def = default.merge(
        fasilitas[["facility_id", "tanggal_pencairan"]], on="facility_id", how="left"
    )
    salah = int((gabung_def["tanggal_default"] <= gabung_def["tanggal_pencairan"]).sum())
    h.catat("default_setelah_pencairan", salah == 0, f"{salah} default terjadi sebelum pencairan")

    kolek = read_table("gold", "fact_kolektibilitas")
    puncak = kolek[kolek["kolektibilitas"] == 5].groupby("facility_id")["snapshot_date"].min()
    cocok = default.set_index("facility_id")["tanggal_default"]
    bersama = puncak.index.intersection(cocok.index)
    h.catat(
        "kolektibilitas_5_konsisten_dengan_default",
        len(bersama) == len(puncak),
        f"{len(puncak) - len(bersama)} fasilitas kol-5 tanpa baris FACT_DEFAULT",
    )


def uji_parameter_build(h: Hasil) -> None:
    """Parameter efektif harus tercatat dan cocok dengan konfigurasi saat ini.

    Regresi: `.env` yang tertinggal memuat N_DEBITUR=3000 sementara kode sudah
    di 6000, sehingga dua build menghasilkan populasi berbeda dengan seed sama.
    Gejalanya menyamar sebagai nondeterminisme.
    """
    if not table_exists("gold", "parameter_build"):
        h.catat("parameter_build_tercatat", False, "gold.parameter_build tidak ada")
        return

    par = read_table("gold", "parameter_build").set_index("parameter")["nilai"]
    h.catat("parameter_build_tercatat", True, f"{len(par)} parameter tercatat")

    cocok = str(par.get("n_debitur")) == str(settings.n_debitur)
    h.catat(
        "parameter_build_cocok_dengan_konfigurasi",
        cocok,
        f"n_debitur tercatat {par.get('n_debitur')} vs konfigurasi {settings.n_debitur}",
    )

    debitur = read_table("gold", "dim_debitur", columns=["cif_sk", "is_current"])
    jumlah = int(debitur["is_current"].sum())
    h.catat(
        "jumlah_debitur_sesuai_parameter",
        jumlah == settings.n_debitur,
        f"{jumlah} debitur aktif vs n_debitur {settings.n_debitur}",
    )

    # Yang berbahaya bukan "parameter datang dari .env", melainkan "nilainya BEDA
    # dari default kode". Berkas .env tidak masuk git, jadi selisih seperti itu
    # membuat dua orang dengan kode identik menghasilkan data berbeda.
    par = read_table("gold", "parameter_build")
    dari_env = par[par["sumber"].str.startswith("env:")]
    menyimpang = dari_env[
        (dari_env["default_kode"] != "") & (dari_env["nilai"] != dari_env["default_kode"])
    ]
    rincian = [
        f"{b.parameter}={b.nilai} (default {b.default_kode})" for b in menyimpang.itertuples()
    ]
    h.catat(
        "parameter_env_tidak_menyimpang_dari_default",
        not rincian,
        f"menyimpang dari default kode: {rincian or 'tidak ada'}",
    )


def uji_kewajaran_statistik(h: Hasil) -> None:
    debitur = read_table("gold", "dim_debitur")
    kini = debitur[debitur["is_current"]]
    tingkat = float(kini["label_default_debitur"].mean())
    h.catat(
        "tingkat_default_masuk_akal",
        0.01 <= tingkat <= 0.25,
        f"tingkat default debitur {tingkat:.2%}",
        kritis=False,
    )

    transfer = read_table("gold", "fact_transfer_giro", columns=["src_is_laundering"])
    rasio = float(transfer["src_is_laundering"].mean())
    h.catat(
        "base_rate_aml_mendekati_sumber",
        rasio < 0.01,
        f"rasio transfer ilisit {rasio:.4%} (sumber LI-Small ~0,05%)",
        kritis=False,
    )

    feat = read_table("gold", "feat_graf_pit")
    kosong = float(feat["supplier_concentration_hhi"].isna().mean())
    h.catat(
        "fitur_graf_terisi",
        kosong < 0.9,
        f"{kosong:.1%} pengajuan tanpa relasi pemasok pada snapshot-nya",
        kritis=False,
    )


def uji_abt(h: Hasil) -> None:
    """ABT adalah yang benar-benar dipegang data scientist - jaga paling ketat."""
    from pipelines.exports.abt import KOLOM_TERLARANG

    for nama in ("abt_pd", "abt_ews", "abt_lgd"):
        df = read_table("gold", nama)
        bocor = sorted(
            {c for c in df.columns if c in KOLOM_TERLARANG}
            | {c for c in df.columns if c.endswith(tuple(KOLOM_TERLARANG))}
        )
        h.catat(f"abt_bebas_kolom_terlarang::{nama}", not bocor, f"kolom bocor: {bocor or 'tidak ada'}")

    abt_pd = read_table("gold", "abt_pd")

    # Fitur graf harus bisa dibuang sekaligus untuk uji ablasi.
    blok_graf = [c for c in abt_pd.columns if c.startswith("graf_")]
    h.catat(
        "abt_pd_blok_graf_bisa_didrop",
        len(blok_graf) >= 10,
        f"{len(blok_graf)} kolom berprefiks graf_",
    )

    # Baris tersensor tidak boleh punya target 0/1 - itu memalsukan bad rate.
    salah = int((abt_pd["y_tersensor"] & abt_pd["y_default_12bln"].notna()).sum())
    h.catat("abt_pd_sensor_konsisten", salah == 0, f"{salah} baris tersensor tapi punya target")

    dapat_dilatih = abt_pd[abt_pd["y_default_12bln"].notna()]
    bad = float(dapat_dilatih["y_default_12bln"].mean())
    h.catat(
        "abt_pd_bad_rate_bisa_dimodelkan",
        0.01 <= bad <= 0.20 and int(dapat_dilatih["y_default_12bln"].sum()) >= 30,
        f"bad rate 12 bulan {bad:.2%} dengan {int(dapat_dilatih['y_default_12bln'].sum())} kejadian",
    )

    # Split OOT: tidak boleh ada cif yang muncul di kedua sisi.
    beririsan = set(abt_pd[abt_pd["split"] == "latih"]["cif_sk"]) & set(
        abt_pd[abt_pd["split"] == "uji_oot"]["cif_sk"]
    )
    h.catat("abt_pd_split_oot_tidak_beririsan", not beririsan, f"{len(beririsan)} cif ada di dua split")

    # Semua fitur harus punya prefiks blok yang dikenal.
    kunci = {
        "application_id",
        "facility_id",
        "cif_sk",
        "grup_id",
        "angkatan",
        "snapshot_date",
        "split",
    }
    liar = [
        c
        for c in abt_pd.columns
        if c not in kunci and not c.startswith(("fin_", "app_", "graf_", "y_", "tanggal_"))
    ]
    h.catat("abt_pd_semua_fitur_berprefiks", not liar, f"kolom tanpa prefiks blok: {liar or 'tidak ada'}")

    ews = read_table("gold", "abt_ews")
    salah = int((ews["y_tersensor"] & ews["y_default_6bln"].notna()).sum())
    h.catat("abt_ews_sensor_konsisten", salah == 0, f"{salah} baris tersensor tapi punya target")

    # Kolom tw_* dicocokkan memakai label gagal bayar - membawa target ke fitur.
    tw = [c for c in abt_pd.columns if "tw_" in c]
    h.catat("abt_pd_bebas_blok_taiwan", not tw, f"kolom tw_ di abt_pd: {tw or 'tidak ada'}")

    # Kolektibilitas tidak boleh memprediksi default secara deterministik.
    dilatih = ews[ews["y_default_6bln"].notna()]
    kol3 = dilatih[dilatih["perilaku_kolektibilitas"] == 3]["y_default_6bln"]
    rasio = float(kol3.mean()) if len(kol3) else 0.0
    h.catat(
        "abt_ews_kolektibilitas_tidak_deterministik",
        len(kol3) >= 20 and rasio < 0.9,
        f"P(default | kol=3) = {rasio:.3f} pada {len(kol3)} observasi",
    )

    # Reject inference butuh ruang fitur yang identik.
    ditolak = read_table("gold", "abt_pengajuan_ditolak")
    fitur_pd = {c for c in abt_pd.columns if c.startswith(("fin_", "app_", "graf_"))}
    fitur_tolak = {c for c in ditolak.columns if c.startswith(("fin_", "app_", "graf_"))}
    h.catat(
        "abt_ditolak_seruang_fitur",
        fitur_pd == fitur_tolak,
        f"selisih {len(fitur_pd ^ fitur_tolak)} kolom",
    )

    # ---- injeksi afiliasi tersembunyi (langkah 7)
    if table_exists("gold", "fact_afiliasi_tersembunyi"):
        klaster = read_table("gold", "fact_afiliasi_tersembunyi")

        # Ground truth tidak boleh bocor ke ABT dalam bentuk apa pun.
        jejak = [
            c
            for c in abt_pd.columns
            if any(k in c for k in ("afiliasi", "klaster", "peran", "mekanisme"))
        ]
        h.catat(
            "afiliasi_ground_truth_tidak_bocor",
            not jejak,
            f"kolom berjejak afiliasi di abt_pd: {jejak or 'tidak ada'}",
        )

        # Klaster wajib melintasi grup usaha yang kasat mata - kalau tidak,
        # afiliasinya bukan tersembunyi, cuma duplikat DIM_GRUP_USAHA.
        peta_grup = (
            read_table("gold", "dim_debitur", columns=["cif_sk", "grup_id", "is_current"])
            .query("is_current")
            .set_index("cif_sk")["grup_id"]
        )
        grup_per_klaster = klaster.assign(grup=klaster["cif_sk"].map(peta_grup)).groupby(
            "afiliasi_id"
        )["grup"].nunique()
        h.catat(
            "afiliasi_melintasi_grup",
            bool((grup_per_klaster > 1).all()),
            f"{int((grup_per_klaster <= 1).sum())} klaster tidak melintasi grup",
        )

        # Sumber harus benar-benar jatuh SEBELUM anggota terinfeksi mengajukan.
        default = read_table("gold", "fact_default", columns=["cif_sk", "tanggal_default"])
        pengajuan = read_table("gold", "fact_pengajuan", columns=["cif_sk", "tanggal_pengajuan"])
        sumber_tgl = (
            klaster[klaster["peran"] == "sumber"]
            .merge(default, on="cif_sk")
            .groupby("afiliasi_id")["tanggal_default"]
            .min()
        )
        infeksi_tgl = (
            klaster[klaster["peran"] == "terinfeksi"]
            .merge(pengajuan, on="cif_sk")
            .groupby("afiliasi_id")["tanggal_pengajuan"]
            .min()
        )
        bersama = sumber_tgl.index.intersection(infeksi_tgl.index)
        melanggar = int((sumber_tgl[bersama] >= infeksi_tgl[bersama]).sum())
        h.catat(
            "afiliasi_penularan_hanya_ke_belakang",
            melanggar == 0,
            f"{melanggar} dari {len(bersama)} klaster: sumber jatuh setelah terinfeksi mengajukan",
        )

        # Kadar kebocoran residual - dilaporkan, bukan digagalkan.
        label = (
            read_table("gold", "dim_debitur", columns=["cif_sk", "label_default_debitur", "is_current"])
            .query("is_current")
            .set_index("cif_sk")["label_default_debitur"]
        )
        anggota_baru = klaster[klaster["peran"].isin(["terinfeksi", "sehat"])]
        kadar = float(anggota_baru["cif_sk"].map(label).mean())
        h.catat(
            "afiliasi_dilusi_keanggotaan",
            0.15 <= kadar <= 0.50,
            f"P(gagal bayar | anggota klaster buku baru) = {kadar:.1%}",
            kritis=False,
        )

    # ---- nilai yang tidak mungkin secara akuntansi / definisi
    # Ketiganya pernah lolos ke ABT dan baru ketahuan saat profiling manual.
    h.catat(
        "abt_umur_perusahaan_tidak_negatif",
        bool((abt_pd["app_umur_perusahaan_tahun"] >= 0).all()),
        f"{int((abt_pd['app_umur_perusahaan_tahun'] < 0).sum())} baris berumur negatif",
    )

    # Winsorisasi p1/p99 di silver membatasi ekor rasio. Ambang 200x longgar -
    # yang dijaga bukan bentuk distribusinya, tapi supaya satu baris tunggal
    # tidak lagi menggeser skala seluruh kolom.
    ekor = {}
    for kolom in (
        "fin_growth_penjualan",
        "fin_icr",
        "fin_current_ratio",
        "fin_der",
        "fin_icr_yoy",
        "fin_der_yoy",
    ):
        if kolom not in abt_pd.columns:
            continue
        x = abt_pd[kolom].dropna()
        p99 = abs(x.quantile(0.99))
        if p99 > 0 and abs(x.max()) / p99 > 200:
            ekor[kolom] = round(float(abs(x.max()) / p99), 1)
    h.catat(
        "abt_rasio_keuangan_terwinsorisasi",
        not ekor,
        f"rasio max/p99 di atas 200x: {ekor or 'tidak ada'}",
    )

    # Embedding graf hasil SVD atas adjacency berbobot log. Kalau bobot rupiah
    # mentah bocor kembali ke sana, nilainya langsung meledak ke 1e13.
    emb = [c for c in abt_pd.columns if "node_emb" in c]
    if emb:
        maks = float(np.nanmax(np.abs(abt_pd[emb].to_numpy())))
        h.catat(
            "abt_embedding_graf_berskala_wajar",
            maks < 1e6,
            f"|emb| maksimum = {maks:.3g}",
        )

    # ---- DIM_ALAMAT: alamat sebagai entitas yang bisa dicocokkan
    if table_exists("gold", "dim_alamat"):
        dim_alamat = read_table("gold", "dim_alamat")
        jembatan_alamat = read_table("gold", "fact_alamat_debitur")

        # Kunci pencocokan harus unik, kalau tidak satu alamat terpecah dua dan
        # debitur yang sekantor tidak pernah bertemu.
        h.catat(
            "alamat_kunci_normal_unik",
            bool(dim_alamat["alamat_normal"].is_unique),
            f"{len(dim_alamat)} alamat, {dim_alamat['alamat_normal'].nunique()} kunci unik",
        )

        # Teks alamat asli ICIJ adalah data nyata dari dokumen bocoran dan tidak
        # boleh ikut ke gold - sama seperti nama asli ICIJ.
        asli = set(
            read_table("silver", "sl_icij_address", columns=["address"])["address"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        bocor = set(dim_alamat["alamat_teks"].dropna().astype(str).str.strip()) & asli
        h.catat(
            "alamat_asli_icij_tidak_ikut_ke_gold",
            not bocor,
            f"{len(bocor)} alamat asli ICIJ ikut ke gold",
        )

        # Alamat agen registrasi wajib ditandai. Tanpa ini satu kantor notaris
        # berisi ratusan badan hukum terbaca sebagai satu grup usaha raksasa.
        agen = dim_alamat[dim_alamat["is_alamat_agen"]]
        h.catat(
            "alamat_agen_ditandai",
            bool((agen["jumlah_debitur"] > 20).all()) if len(agen) else True,
            f"{len(agen)} alamat ditandai agen registrasi",
            kritis=False,
        )

        # Inti nilai tabel ini: menemukan keterkaitan yang TIDAK terlihat di
        # grup usaha. Kalau nol, alamat tidak menambah apa pun di atas grup_id.
        peta_grup_alamat = (
            read_table("gold", "dim_debitur", columns=["cif_sk", "grup_id", "is_current"])
            .query("is_current")
            .set_index("cif_sk")["grup_id"]
        )
        layak = jembatan_alamat[
            jembatan_alamat["alamat_id"].isin(dim_alamat[~dim_alamat["is_alamat_agen"]]["alamat_id"])
        ]
        grup_per_alamat = layak.assign(grup=layak["cif_sk"].map(peta_grup_alamat)).groupby(
            "alamat_id"
        )["grup"].nunique()
        lintas = int((grup_per_alamat > 1).sum())
        h.catat(
            "alamat_menemukan_keterkaitan_lintas_grup",
            lintas > 0,
            f"{lintas} alamat dipakai debitur dari lebih dari satu grup usaha",
        )

    # Ruang fitur LGD harus sejajar: model dilatih di satu tabel, dipanggil di lain.
    fitur_latih = {c for c in read_table("gold", "abt_lgd_sumber").columns if c.startswith("app_")}
    fitur_terap = {c for c in read_table("gold", "abt_lgd").columns if c.startswith("app_")}
    tidak_terpakai = sorted(fitur_latih - fitur_terap)
    h.catat(
        "lgd_ruang_fitur_sejajar",
        not tidak_terpakai,
        f"fitur latih yang tak ada saat menerapkan: {tidak_terpakai or 'tidak ada'}",
    )

    sumber = read_table("gold", "abt_lgd_sumber", columns=["y_lgd_realisasi", "split"])
    h.catat(
        "abt_lgd_sumber_cukup_besar",
        len(sumber) > 100_000,
        f"{len(sumber)} pinjaman SBA CHGOFF untuk melatih LGD",
    )


# ------------------------------------------------------------------ orkestra
def jalankan_semua(strict: bool = True) -> pd.DataFrame:
    h = Hasil()
    uji_kunci(h)
    uji_integritas_referensial(h)
    uji_rentang_nilai(h)
    uji_konsistensi_waktu(h)
    uji_pemisahan_label(h)
    uji_kebocoran_waktu(h)
    uji_kewajaran_statistik(h)
    uji_parameter_build(h)
    uji_abt(h)

    df = h.dataframe()
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    cap = datetime.now().strftime("%Y%m%dT%H%M%S")
    df.to_parquet(QUALITY_DIR / f"hasil_uji_{cap}.parquet", index=False)
    write_table(df, "gold", "data_quality_report")

    gagal_kritis = df[(~df["lolos"]) & df["kritis"]]
    LOG.info("uji kualitas: %s lolos, %s gagal", int(df["lolos"].sum()), int((~df["lolos"]).sum()))
    if strict and len(gagal_kritis):
        raise AssertionError(
            "uji kualitas kritis gagal:\n"
            + "\n".join(f"  - {r.uji}: {r.detail}" for r in gagal_kritis.itertuples())
        )
    return df

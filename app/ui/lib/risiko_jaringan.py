"""Indikator risiko jaringan atas data graf nyata.

Menggantikan skor jaringan tiruan yang dulu ada di `lib/dummy_data.py` - ia
mengarang angka dari RNG dan menempelkan pola anomali acak beserta bukti yang
tidak menunjuk apa pun.

APA YANG NYATA DAN APA YANG TIDAK
---------------------------------
Seluruh *komponen* dibaca dari `data/gold`: `feat_graf_pit`, `fact_default`
lewat tabel resolusi, `fact_afiliasi_tersembunyi`, `fact_agunan`, dan metrik
snapshot bulanan. Tidak ada satu pun angka yang dibangkitkan acak.

*Pembobotannya* karangan. Tidak ada model terlatih yang memetakan enam
komponen ini ke satu angka 0-100; bobot di `BOBOT` adalah keputusan kebijakan
yang ditulis terbuka supaya bisa diperdebatkan dan diganti, bukan hasil
kalibrasi. Karena itu keluarannya disebut **indikator**, dan `HasilJaringan`
membawa rincian komponen supaya angka akhirnya bisa dibongkar, bukan dipercaya
begitu saja.

Docstring `graf_nyata.resolusi_calon` menegaskan keluarannya bukan skor risiko.
Modul ini tidak membantah itu: resolusi tetap dipakai apa adanya sebagai daftar
afiliasi yang layak diperiksa, dan skor di sini adalah lapisan terpisah di
atasnya yang dilabeli sebagai indikator beraturan.

BATAS PENAFSIRAN
----------------
Pemohon baru belum punya simpul di graf. Yang diukur adalah lingkungan dari
debitur eksisting yang berhasil dicocokkan lewat alamat operasional, nama
pengurus, dan rekening lawan - jadi bacalah sebagai "risiko jaringan dari
afiliasi yang terdeteksi", bukan sebagai properti pemohon itu sendiri. Bila
tidak ada dokumen yang bisa dicocokkan, modul mengembalikan `tersedia=False`
dan skor `None`; itu berbeda dari skor nol dan halaman wajib membedakannya.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import streamlit as st

from lib import graf_nyata as gn

# Bobot penggabungan komponen. Jumlahnya 1,0. Ini ambang kebijakan yang
# ditetapkan tim, bukan koefisien hasil pelatihan - ubah di sini, dan angka di
# layar ikut berubah tanpa menyentuh kode lain.
BOBOT: dict[str, float] = {
    "afiliasi_gagal_bayar": 0.30,
    "neighbor_default_rate_1hop": 0.20,
    "community_default_rate": 0.15,
    "afiliasi_tersembunyi": 0.15,
    "konsentrasi_mitra": 0.10,
    "group_exposure_share": 0.10,
}

# Titik jenuh normalisasi: nilai di atas ini dianggap 1,0. Tingkat gagal bayar
# tetangga 15% sudah sangat tinggi untuk portofolio komersial, jadi menaikkan
# langit-langit hanya membuat seluruh kasus terlihat rendah.
JENUH_DEFAULT_RATE = 0.15

# HHI di bawah 0,25 dianggap terdiversifikasi; 0,75 ke atas dianggar terkonsentrasi.
HHI_BAWAH, HHI_ATAS = 0.25, 0.75

LABEL_POLA = {
    "shared_attribute": "Beberapa badan hukum berbagi alamat domisili, pengurus, atau rekening pencairan",
    "circular_payment": "Siklus transaksi melingkar antar pihak berelasi yang menaikkan pendapatan secara artifisial",
    "cross_guarantee_chain": "Penjaminan silang berantai antar afiliasi sehingga nilai agunan berpotensi dihitung ganda",
    "layered_ownership": "Struktur kepemilikan berlapis melalui entitas tanpa aktivitas usaha",
    "undisclosed_affiliation": "Afiliasi tidak dinyatakan - dua debitur secara topologi berada dalam satu kendali",
}

# Mekanisme pada fact_afiliasi_tersembunyi -> kode pola yang dipakai antarmuka.
MEKANISME_KE_POLA = {
    "siklus_pembayaran": "circular_payment",
    "nominee_bersama": "layered_ownership",
    "alamat_operasional_bersama": "shared_attribute",
}

# Pola yang ada di kosakata antarmuka tetapi TIDAK diperiksa modul ini.
# Dilaporkan apa adanya supaya "tidak diperiksa" tidak terbaca sebagai "aman".
POLA_TAK_DIPERIKSA = {
    "transfer_spike": (
        "Lonjakan transfer antar entitas satu grup sesaat sebelum tanggal "
        "laporan keuangan - butuh analisis deret waktu atas fact_transfer_giro "
        "yang belum dibangun."
    ),
}

# Dasar bukti resolusi yang menandakan atribut dipakai bersama.
DASAR_ATRIBUT_BERSAMA = (
    "Alamat sama persis", "Alamat mirip", "Pengurus bersama", "Pemilik bersama",
)

# `telusuri_afiliasi` mengosongkan sebuah jalur karena dua sebab yang sangat
# berbeda maknanya (pipelines/graph/resolusi.py:310-313). Jalur yang berjalan
# tanpa hasil adalah temuan - "tidak ada afiliasi". Jalur yang dokumennya tidak
# ada adalah ketidaktahuan. Keduanya tidak boleh dilebur.
SEBAB_TIDAK_DIJALANKAN = "dokumen tidak disertakan"


@dataclass(frozen=True)
class Komponen:
    """Satu penyusun indikator, lengkap dengan nilai mentahnya."""

    kunci: str
    label: str
    mentah: str          # nilai apa adanya, untuk ditampilkan
    nilai: float         # ternormalisasi 0..1
    bobot: float
    sumber: str          # tabel gold asalnya

    @property
    def sumbangan(self) -> float:
        return self.nilai * self.bobot * 100


@dataclass
class HasilJaringan:
    application_id: str
    tersedia: bool
    skor: float | None
    pola: list[dict] = field(default_factory=list)
    komponen: list[Komponen] = field(default_factory=list)
    jumlah_afiliasi: int = 0
    afiliasi_gagal_bayar: int = 0
    # Debitur eksisting yang tercocok. Dibawa keluar karena eksposur grup BMPK
    # dibaca dari cif ini - tanpa cocokan, tidak ada grup yang bisa ditunjuk.
    cif_tercocok: tuple = ()
    perlu_telaah: bool = False
    snapshot: pd.Timestamp | None = None
    catatan: list[str] = field(default_factory=list)
    tak_diperiksa: dict[str, str] = field(default_factory=lambda: dict(POLA_TAK_DIPERIKSA))

    def sebagai_dict(self) -> dict[str, Any]:
        """Bentuk yang dibaca halaman dan penyusun memo.

        Kunci `skor` dan `pola` dipertahankan dari skor jaringan tiruan yang
        digantikan modul ini, supaya penggantiannya tidak menuntut perubahan di
        sisi pembaca. `skor` bisa `None` - itu justru informasi, dan pembaca
        harus menanganinya.
        """
        return {
            "application_id": self.application_id,
            "skor": self.skor,
            "pola": self.pola,
            "tersedia": self.tersedia,
            "catatan": self.catatan,
        }


def _normal(nilai: float, jenuh: float) -> float:
    return float(min(max(nilai / jenuh, 0.0), 1.0)) if jenuh else 0.0


def _normal_hhi(nilai: float) -> float:
    return float(min(max((nilai - HHI_BAWAH) / (HHI_ATAS - HHI_BAWAH), 0.0), 1.0))


@st.cache_data(show_spinner=False)
def _fitur_pit(cif: tuple[int, ...]) -> pd.DataFrame | None:
    """Baris feat_graf_pit untuk debitur tercocok.

    Satu baris per cif_sk, bertanggal pada pengajuan debitur itu sendiri - jadi
    ini point-in-time terhadap riwayat mereka, bukan terhadap tanggal telaah
    yang sedang berjalan. Perbedaan itu dicatat sebagai peringatan pemakai.
    """
    kolom = (
        "cif_sk", "snapshot_date", "buyer_concentration_hhi",
        "supplier_concentration_hhi", "neighbor_default_rate_1hop",
        "community_default_rate", "shared_attribute_degree",
        "circular_payment_flag", "group_exposure_share",
    )
    df = gn._baca("feat_graf_pit", kolom)
    if df is None or not cif:
        return None
    return df[df["cif_sk"].isin(cif)].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _afiliasi_tersembunyi(cif: tuple[int, ...]) -> pd.DataFrame | None:
    df = gn._baca("fact_afiliasi_tersembunyi", ("afiliasi_id", "cif_sk", "peran", "mekanisme"))
    if df is None or not cif:
        return None
    # Peran 'sehat' ikut tercatat pada klaster yang sama tetapi bukan temuan;
    # yang menjadi bukti hanyalah sumber penularan dan yang tertular.
    df = df[df["cif_sk"].isin(cif) & df["peran"].isin(("sumber", "terinfeksi"))]
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _agunan_silang(cif: tuple[int, ...]) -> int:
    """Jumlah agunan berstatus dijaminkan silang pada fasilitas debitur tercocok."""
    fas = gn._baca("fact_fasilitas", ("facility_id", "cif_sk"))
    ag = gn._baca("fact_agunan", ("agunan_id", "facility_id", "dijaminkan_silang"))
    if fas is None or ag is None or not cif:
        return 0
    milik = fas[fas["cif_sk"].isin(cif)]["facility_id"]
    return int(ag[ag["facility_id"].isin(milik) & ag["dijaminkan_silang"].fillna(False)].shape[0])


@st.cache_data(show_spinner=False)
def _community_default(cif: tuple[int, ...], tanggal: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    """Rerata community_default_rate debitur tercocok pada snapshot <= tanggal.

    Inilah satu-satunya komponen yang benar-benar point-in-time terhadap tanggal
    telaah: `graph_snapshot_bulanan` menyimpan metrik per bulan, jadi snapshot
    terakhir yang tidak melewati tanggal bisa dipilih tanpa membocorkan masa depan.
    """
    tersedia = [t for t in gn.snapshot_tersedia() if t <= pd.Timestamp(tanggal)]
    if not tersedia or not cif:
        return None, None
    dipakai = max(tersedia)

    simpul = gn._baca("gold_graph_nodes", ("node_id", "node_type", "ref_id"))
    metrik = gn.metrik_snapshot(dipakai)
    if simpul is None or metrik is None:
        return None, dipakai

    badan = simpul[(simpul["node_type"] == "badan_hukum") & simpul["ref_id"].isin(cif)]
    sel = metrik[metrik["node_id"].isin(badan["node_id"])]
    if sel.empty:
        return None, dipakai
    return float(sel["community_default_rate"].mean()), dipakai


def _pola_terdeteksi(
    tabel: pd.DataFrame,
    pit: pd.DataFrame | None,
    tersembunyi: pd.DataFrame | None,
    agunan_silang: int,
) -> list[dict]:
    """Pola yang benar-benar punya bukti di tabel gold, dengan buktinya."""
    pola: list[dict] = []

    def tambah(kode: str, bukti: str) -> None:
        if any(p["kode"] == kode for p in pola):
            return
        pola.append({"kode": kode, "deskripsi": LABEL_POLA[kode], "bukti": bukti})

    # 1. Atribut bersama - dari dasar pencocokan resolusi itu sendiri.
    berbagi = tabel[tabel["dasar_utama"].isin(DASAR_ATRIBUT_BERSAMA)]
    if not berbagi.empty:
        rincian = ", ".join(
            f"{n} ({d})" for n, d in
            berbagi[["nama_debitur", "dasar_utama"]].head(3).itertuples(index=False)
        )
        tambah("shared_attribute", f"{len(berbagi)} debitur tercocok lewat atribut bersama: {rincian}")

    if pit is not None and not pit.empty:
        derajat = pd.to_numeric(pit["shared_attribute_degree"], errors="coerce").fillna(0)
        if (derajat > 0).any():
            tambah(
                "shared_attribute",
                f"shared_attribute_degree tertinggi {int(derajat.max())} pada "
                f"{int((derajat > 0).sum())} dari {len(pit)} debitur tercocok",
            )
        siklus = pit["circular_payment_flag"].fillna(False)
        if siklus.any():
            tambah(
                "circular_payment",
                f"circular_payment_flag aktif pada {int(siklus.sum())} dari "
                f"{len(pit)} debitur tercocok (feat_graf_pit)",
            )

    # 2. Afiliasi tersembunyi - tabel ini memang berisi temuan, bukan dugaan.
    if tersembunyi is not None and not tersembunyi.empty:
        per_mekanisme = tersembunyi["mekanisme"].value_counts()
        klaster = tersembunyi["afiliasi_id"].nunique()
        terinfeksi = int((tersembunyi["peran"] == "terinfeksi").sum())
        tambah(
            "undisclosed_affiliation",
            f"{len(tersembunyi)} debitur tercocok muncul di {klaster} klaster afiliasi "
            f"tersembunyi ({terinfeksi} berperan terinfeksi)",
        )
        for mekanisme, jumlah in per_mekanisme.items():
            kode = MEKANISME_KE_POLA.get(str(mekanisme))
            if kode:
                tambah(kode, f"mekanisme {mekanisme} pada {int(jumlah)} debitur tercocok "
                             f"(fact_afiliasi_tersembunyi)")

    # 3. Penjaminan silang.
    if agunan_silang:
        tambah(
            "cross_guarantee_chain",
            f"{agunan_silang} agunan berstatus dijaminkan silang pada fasilitas "
            "debitur tercocok (fact_agunan)",
        )

    return pola


def skor_jaringan(
    application_id: str,
    hasil_resolusi: gn.HasilResolusiUI | None,
    tanggal: pd.Timestamp | str,
) -> HasilJaringan:
    """Indikator risiko jaringan untuk satu pengajuan.

    `hasil_resolusi` datang dari `graf_nyata.resolusi_calon()` - pemohon baru
    tidak punya simpul sendiri, jadi seluruh pengukuran berpijak pada debitur
    eksisting yang berhasil dicocokkan darinya.
    """
    tanggal = pd.Timestamp(tanggal)
    catatan: list[str] = []

    if hasil_resolusi is None:
        return HasilJaringan(
            application_id, False, None,
            catatan=["Resolusi afiliasi belum dijalankan - dokumen belum dibaca jalur LLM."],
        )
    if hasil_resolusi.galat:
        return HasilJaringan(
            application_id, False, None,
            catatan=[f"Resolusi afiliasi gagal: {hasil_resolusi.galat}"],
        )

    # Yang menentukan bisa-tidaknya diperiksa adalah jalur yang BERJALAN, bukan
    # jalur yang menghasilkan kecocokan. Alamat yang diberikan lalu tidak cocok
    # tetap merupakan pemeriksaan yang sah.
    dijalankan = [
        j["nama"] for j in hasil_resolusi.jalur
        if j["dipakai"] or SEBAB_TIDAK_DIJALANKAN not in j["keterangan"]
    ]
    tanpa_dokumen = [
        j["nama"] for j in hasil_resolusi.jalur
        if not j["dipakai"] and SEBAB_TIDAK_DIJALANKAN in j["keterangan"]
    ]
    if not dijalankan:
        return HasilJaringan(
            application_id, False, None,
            catatan=[
                "Tidak satu pun jalur pencocokan bisa dijalankan - seluruh dokumen "
                "sumbernya tidak disertakan: " + ", ".join(tanpa_dokumen)
            ],
        )
    if tanpa_dokumen:
        catatan.append(
            "Jalur yang tidak bisa diperiksa karena dokumennya tidak ada: "
            + ", ".join(tanpa_dokumen)
        )

    tabel = hasil_resolusi.tabel
    cif = tuple(int(x) for x in tabel["cif_sk"].dropna().unique()) if not tabel.empty else ()

    if not cif:
        # Jalur berjalan tetapi tidak menemukan siapa pun. Itu temuan yang sah,
        # dan berbeda dari "tidak bisa diperiksa" di cabang-cabang di atas.
        return HasilJaringan(
            application_id, True, 0.0,
            jumlah_afiliasi=0, perlu_telaah=hasil_resolusi.perlu_telaah,
            catatan=catatan + [
                f"{len(dijalankan)} jalur pencocokan berjalan dan tidak menemukan "
                "satu pun afiliasi."
            ],
        )

    pit = _fitur_pit(cif)
    tersembunyi = _afiliasi_tersembunyi(cif)
    silang = _agunan_silang(cif)
    komunitas, snapshot = _community_default(cif, tanggal)

    komponen: list[Komponen] = []

    # 1. Afiliasi yang sudah gagal bayar - fakta, bukan proksi.
    n_gagal = int(tabel["sudah_gagal_bayar"].sum()) if "sudah_gagal_bayar" in tabel else 0
    porsi_gagal = n_gagal / len(cif)
    komponen.append(Komponen(
        "afiliasi_gagal_bayar", "Afiliasi yang sudah gagal bayar",
        f"{n_gagal} dari {len(cif)} debitur", porsi_gagal,
        BOBOT["afiliasi_gagal_bayar"], "resolusi afiliasi + fact_default",
    ))

    if pit is not None and not pit.empty:
        tetangga = float(pd.to_numeric(pit["neighbor_default_rate_1hop"], errors="coerce").mean())
        komponen.append(Komponen(
            "neighbor_default_rate_1hop", "Gagal bayar tetangga 1-hop",
            f"{tetangga * 100:.1f}%", _normal(tetangga, JENUH_DEFAULT_RATE),
            BOBOT["neighbor_default_rate_1hop"], "feat_graf_pit",
        ))

        hhi = pd.concat([
            pd.to_numeric(pit["buyer_concentration_hhi"], errors="coerce"),
            pd.to_numeric(pit["supplier_concentration_hhi"], errors="coerce"),
        ], axis=1).max(axis=1).mean()
        if pd.notna(hhi):
            komponen.append(Komponen(
                "konsentrasi_mitra", "Konsentrasi pembeli/pemasok (HHI)",
                f"{hhi:.2f}", _normal_hhi(float(hhi)),
                BOBOT["konsentrasi_mitra"], "feat_graf_pit",
            ))

        porsi = float(pd.to_numeric(pit["group_exposure_share"], errors="coerce").fillna(0).max())
        komponen.append(Komponen(
            "group_exposure_share", "Porsi eksposur grup terhadap BMPK",
            f"{porsi * 100:.0f}%", float(min(max(porsi, 0.0), 1.0)),
            BOBOT["group_exposure_share"], "feat_graf_pit",
        ))

        tanggal_pit = pd.to_datetime(pit["snapshot_date"])
        catatan.append(
            "Fitur feat_graf_pit bertanggal pada pengajuan masing-masing debitur "
            f"({tanggal_pit.min():%b %Y} - {tanggal_pit.max():%b %Y}), bukan pada "
            f"tanggal telaah {tanggal:%d %b %Y}."
        )
    else:
        catatan.append(
            "feat_graf_pit tidak memuat satu pun debitur tercocok; komponen "
            "tetangga, konsentrasi, dan eksposur grup tidak ikut dihitung."
        )

    if komunitas is not None:
        komponen.append(Komponen(
            "community_default_rate", "Gagal bayar komunitas graf",
            f"{komunitas * 100:.1f}%", _normal(komunitas, JENUH_DEFAULT_RATE),
            BOBOT["community_default_rate"], f"graph_snapshot_bulanan {snapshot:%b %Y}",
        ))
    else:
        catatan.append("Metrik komunitas tidak tersedia pada snapshot mana pun sebelum tanggal telaah.")

    n_sembunyi = 0 if tersembunyi is None else int(tersembunyi["cif_sk"].nunique())
    komponen.append(Komponen(
        "afiliasi_tersembunyi", "Debitur pada klaster afiliasi tersembunyi",
        f"{n_sembunyi} dari {len(cif)} debitur", n_sembunyi / len(cif),
        BOBOT["afiliasi_tersembunyi"], "fact_afiliasi_tersembunyi",
    ))

    # Bobot dinormalisasi ulang atas komponen yang benar-benar terhitung, supaya
    # komponen yang datanya hilang tidak diam-diam terbaca sebagai nilai nol.
    total_bobot = sum(k.bobot for k in komponen)
    skor = 100 * sum(k.nilai * k.bobot for k in komponen) / total_bobot if total_bobot else None
    if total_bobot < 0.999:
        catatan.append(
            f"Bobot dinormalisasi ulang atas {len(komponen)} komponen yang terhitung "
            f"({total_bobot:.2f} dari 1,00); komponen yang datanya tidak ada tidak dianggap nol."
        )

    return HasilJaringan(
        application_id=application_id,
        tersedia=True,
        skor=None if skor is None else float(round(skor, 1)),
        pola=_pola_terdeteksi(tabel, pit, tersembunyi, silang),
        komponen=komponen,
        jumlah_afiliasi=len(cif),
        afiliasi_gagal_bayar=n_gagal,
        cif_tercocok=cif,
        perlu_telaah=hasil_resolusi.perlu_telaah,
        snapshot=snapshot,
        catatan=catatan,
    )

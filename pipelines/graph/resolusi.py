"""Pencocokan calon nasabah baru ke graf yang sudah ada (entity resolution).

Inilah pasangan sisi-baca dari graph/alamat.py. Modul itu membuat alamat bisa
dicari; modul ini yang mencarinya.

Masalah yang diselesaikan: saat RM memasukkan pengajuan calon nasabah baru,
debitur itu belum punya satu pun edge di GOLD_GRAPH_EDGES. Seluruh fitur graf
(neighbor_default_rate_1hop, community_default_rate, supplier_concentration_hhi)
kosong, dan pertanyaan "apakah calon ini terafiliasi dengan debitur lain di
portofolio" tidak bisa dijawab dari tabel mana pun.

Yang dipakai untuk menjawabnya adalah tiga berkas yang memang sudah wajib
dikumpulkan saat CDD, bukan data baru:

    akta / data kepemilikan  -> nama pengurus dan pemegang saham -> DIM_PIHAK
    dokumen domisili usaha   -> alamat operasional               -> DIM_ALAMAT
    rekening koran           -> rekening lawan transaksi         -> BRIDGE_REKENING

Dua yang pertama bekerja sejak hari pertama karena sifatnya identitas, bukan
perilaku. Yang ketiga baru berguna kalau calon sudah punya riwayat mutasi -
untuk perusahaan yang benar-benar baru, hasilnya memang kosong, dan itu harus
dilaporkan sebagai "tidak diketahui", bukan sebagai "tidak ada afiliasi".

ATURAN TITIK-WAKTU. Setiap fungsi di sini menerima `tanggal` dan hanya melihat
edge yang sudah berlaku serta gagal bayar yang sudah terjadi pada tanggal itu.
Aturannya sama dengan graph/fitur_pit.py - kalau dilonggarkan di sini, angka
yang muncul di layar RM tidak akan pernah bisa direproduksi oleh model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from pipelines.graph.alamat import normalisasi_alamat
from pipelines.utils import read_table, table_exists

LOG = logging.getLogger("pipelines.graph.resolusi")

# Ambang kemiripan. Sengaja tinggi: keluaran modul ini memicu penelaahan manual
# (KKK-13.6), jadi positif palsu memakan waktu analis. Lebih baik melewatkan
# kandidat lemah daripada mengubur yang kuat di dalam daftar panjang.
AMBANG_ALAMAT = 0.80
AMBANG_NAMA = 0.85

# Gelar dan bentuk badan hukum tidak membawa informasi identitas.
STOPWORD_NAMA = {
    "PT", "CV", "UD", "PD", "TBK", "PERSERO", "HOLDING",
    "BAPAK", "IBU", "BP", "SDR", "H", "HJ",
    "DR", "IR", "DRS", "SE", "SH", "MM", "MBA", "ST", "MSI",
    "MR", "MRS", "MS", "LTD", "LIMITED", "INC", "CORP", "LLC",
}

_BUKAN_HURUF = re.compile(r"[^0-9A-Z]+")
_TITIK = re.compile(r"\.")


def _token_nama(nama: str) -> set[str]:
    """Token identitas sebuah nama, tanpa gelar dan bentuk badan hukum.

    Titik dibuang lebih dulu, bukan diubah menjadi spasi: 'S.E., M.M.' harus
    menjadi {SE, MM} supaya kena daftar gelar. Kalau titiknya jadi spasi, yang
    tersisa {S, E, M, M} - tidak ada di daftar mana pun, dan nama yang sama
    persis gagal cocok hanya karena RM menuliskan gelarnya.

    Inisial satu huruf ikut dibuang; ia tidak membedakan siapa pun dan hanya
    menggerus skor kemiripan.
    """
    teks = _TITIK.sub("", str(nama).upper())
    bersih = _BUKAN_HURUF.sub(" ", teks)
    return {t for t in bersih.split() if len(t) > 1 and t not in STOPWORD_NAMA}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _aktif_pada(df: pd.DataFrame, tanggal: pd.Timestamp) -> pd.DataFrame:
    """Baris yang sudah berlaku dan belum berakhir pada tanggal penilaian."""
    if df.empty:
        return df
    mulai = pd.to_datetime(df["valid_from"]) <= tanggal
    selesai = df["valid_to"].isna() | (pd.to_datetime(df["valid_to"]) > tanggal)
    return df[mulai & selesai]


# --------------------------------------------------------------------- alamat
def cocokkan_alamat(
    teks: str, tanggal: pd.Timestamp, ambang: float = AMBANG_ALAMAT
) -> pd.DataFrame:
    """Cari debitur yang beralamat operasional sama dengan `teks`.

    Dua tahap: kecocokan persis atas bentuk ternormalisasi, lalu kemiripan token
    untuk alamat yang ditulis dengan urutan atau ejaan berbeda.
    """
    kolom = ["cif_sk", "alamat_id", "alamat_teks", "skor", "dasar"]
    if not teks or not str(teks).strip():
        return pd.DataFrame(columns=kolom)

    dim = read_table("gold", "dim_alamat")
    jembatan = read_table("gold", "fact_alamat_debitur")
    if dim.empty or jembatan.empty:
        return pd.DataFrame(columns=kolom)

    kunci = normalisasi_alamat(pd.Series([teks])).iloc[0]
    if not kunci:
        return pd.DataFrame(columns=kolom)

    # Alamat agen registrasi dibuang - ratusan badan hukum di satu kantor notaris
    # bukan tanda keterkaitan usaha, dan kalau ikut ia menenggelamkan sisanya.
    layak = dim[~dim["is_alamat_agen"]].copy()

    persis = layak[layak["alamat_normal"] == kunci].copy()
    persis["skor"] = 1.0
    persis["dasar"] = "alamat_persis"

    sisa = layak[layak["alamat_normal"] != kunci].copy()
    token_kunci = set(kunci.split())
    sisa["skor"] = [_jaccard(token_kunci, set(str(a).split())) for a in sisa["alamat_normal"]]
    mirip = sisa[sisa["skor"] >= ambang].copy()
    mirip["dasar"] = "alamat_mirip"

    cocok = pd.concat([persis, mirip], ignore_index=True)
    if cocok.empty:
        return pd.DataFrame(columns=kolom)

    hasil = _aktif_pada(jembatan, tanggal).merge(
        cocok[["alamat_id", "alamat_teks", "skor", "dasar"]], on="alamat_id", how="inner"
    )
    return hasil[kolom].sort_values("skor", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------------- pengurus
def cocokkan_pengurus(
    nama_pengurus: list[str], tanggal: pd.Timestamp, ambang: float = AMBANG_NAMA
) -> pd.DataFrame:
    """Cari debitur yang pengurus/pemiliknya beririsan dengan daftar `nama_pengurus`.

    Ini jalur yang paling kuat untuk calon nasabah baru: sifatnya identitas, jadi
    sudah bisa dipakai pada hari pengajuan tanpa menunggu riwayat transaksi apa
    pun terbentuk.
    """
    kolom = ["cif_sk", "pihak_id", "nama_pihak", "nama_input", "skor", "dasar"]
    nama_bersih = [n for n in (nama_pengurus or []) if n and str(n).strip()]
    if not nama_bersih:
        return pd.DataFrame(columns=kolom)

    pihak = read_table("gold", "dim_pihak", columns=["pihak_id", "nama"])
    if pihak.empty:
        return pd.DataFrame(columns=kolom)

    token_pihak = {int(p): _token_nama(n) for p, n in zip(pihak["pihak_id"], pihak["nama"])}

    baris = []
    for masukan in nama_bersih:
        token_masukan = _token_nama(masukan)
        if not token_masukan:
            continue
        for pihak_id, token in token_pihak.items():
            skor = _jaccard(token_masukan, token)
            if skor >= ambang:
                baris.append({"pihak_id": pihak_id, "nama_input": masukan, "skor": skor})
    if not baris:
        return pd.DataFrame(columns=kolom)

    cocok = pd.DataFrame(baris).sort_values("skor", ascending=False)
    cocok = cocok.drop_duplicates(subset=["pihak_id"], keep="first")

    # Satu pihak yang sama menjabat / memiliki di beberapa debitur - itulah
    # nominee bersama, mekanisme afiliasi tersembunyi yang paling sering dipakai.
    relasi = []
    for tabel, dasar in (
        ("fact_kepengurusan", "pengurus_bersama"),
        ("fact_kepemilikan", "pemilik_bersama"),
    ):
        if not table_exists("gold", tabel):
            continue
        df = _aktif_pada(read_table("gold", tabel), tanggal)
        if df.empty:
            continue
        relasi.append(df[["pihak_id", "cif_sk"]].assign(dasar=dasar))
    if not relasi:
        return pd.DataFrame(columns=kolom)

    hasil = pd.concat(relasi, ignore_index=True).merge(cocok, on="pihak_id", how="inner")
    hasil = hasil.merge(pihak.rename(columns={"nama": "nama_pihak"}), on="pihak_id", how="left")
    return hasil[kolom].sort_values("skor", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- rekening koran
def cocokkan_counterparty(rekening_lawan: list[str], tanggal: pd.Timestamp) -> pd.DataFrame:
    """Cari debitur yang bertransaksi dengan rekening lawan yang sama.

    Berbeda dari dua fungsi di atas, ini berbasis PERILAKU dan karena itu hanya
    berguna kalau calon sudah punya riwayat mutasi. Untuk perusahaan yang baru
    berdiri hasilnya kosong - dan kosong di sini berarti belum diketahui, bukan
    terbukti bersih.
    """
    kolom = ["cif_sk", "rekening_lawan", "jumlah_transfer", "dasar"]
    akun = [str(a).strip() for a in (rekening_lawan or []) if a and str(a).strip()]
    if not akun:
        return pd.DataFrame(columns=kolom)

    bridge = read_table("gold", "bridge_rekening")
    transfer = read_table("gold", "fact_transfer_giro")
    if bridge.empty or transfer.empty:
        return pd.DataFrame(columns=kolom)

    sasaran = bridge[bridge["src_aml_account"].astype("string").isin(akun)]
    if sasaran.empty:
        return pd.DataFrame(columns=kolom)

    tr = transfer[pd.to_datetime(transfer["waktu"]) <= tanggal]
    if tr.empty:
        return pd.DataFrame(columns=kolom)

    id_sasaran = set(sasaran["rekening_id"])
    pemilik = bridge.set_index("rekening_id")["cif_sk"]
    akun_asal = bridge.set_index("rekening_id")["src_aml_account"]

    sisi = []
    for kolom_lawan, kolom_debitur in (
        ("rekening_id_pengirim", "rekening_id_penerima"),
        ("rekening_id_penerima", "rekening_id_pengirim"),
    ):
        sub = tr[tr[kolom_lawan].isin(id_sasaran)]
        if sub.empty:
            continue
        sisi.append(
            pd.DataFrame(
                {
                    "cif_sk": sub[kolom_debitur].map(pemilik),
                    "rekening_lawan": sub[kolom_lawan].map(akun_asal),
                }
            )
        )
    if not sisi:
        return pd.DataFrame(columns=kolom)

    gabung = pd.concat(sisi, ignore_index=True).dropna(subset=["cif_sk"])
    if gabung.empty:
        return pd.DataFrame(columns=kolom)

    hasil = (
        gabung.groupby(["cif_sk", "rekening_lawan"], as_index=False)
        .size()
        .rename(columns={"size": "jumlah_transfer"})
    )
    hasil["cif_sk"] = hasil["cif_sk"].astype("int64")
    hasil["dasar"] = "counterparty_bersama"
    return hasil[kolom].sort_values("jumlah_transfer", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------- gabungan
@dataclass
class HasilResolusi:
    """Kandidat afiliasi calon nasabah, beserta buktinya."""

    tanggal: pd.Timestamp
    kandidat: pd.DataFrame = field(default_factory=pd.DataFrame)
    jalur_terpakai: list[str] = field(default_factory=list)
    jalur_kosong: list[str] = field(default_factory=list)
    # Semesta yang benar-benar tercari pada tanggal penilaian, per jalur.
    # Ada di sini supaya lapisan memo tidak bisa menuliskan klaim yang lebih
    # besar dari datanya: "tidak ditemukan" hanya bermakna bersama "dicari di
    # antara berapa". Lihat cakupan_pencarian().
    cakupan: dict[str, int] = field(default_factory=dict)

    @property
    def jumlah_kandidat(self) -> int:
        return 0 if self.kandidat.empty else int(self.kandidat["cif_sk"].nunique())

    @property
    def perlu_telaah(self) -> bool:
        """Pemicu KKK-13.6 - ambangnya aturan kebijakan, bukan skor model."""
        if self.kandidat.empty:
            return False
        return self.jumlah_kandidat >= 3 or bool(
            (self.kandidat["dasar"] == "pengurus_bersama").any()
        )


def cakupan_pencarian(tanggal: pd.Timestamp | str) -> dict[str, int]:
    """Besar semesta yang bisa dicari pada tanggal penilaian, per jalur.

    Hasil nihil dari telusuri_afiliasi() tidak berarti apa-apa tanpa angka ini.
    "Tidak ditemukan afiliasi" adalah klaim tentang seluruh dunia; yang bisa
    dibuktikan datanya hanya "tidak ditemukan di antara sekian debitur yang
    kami kenal per tanggal sekian". Selisih dua pernyataan itu persis yang
    memisahkan catatan CDD yang bisa dipertanggungjawabkan dari yang tidak.

    Angkanya dihitung dengan filter titik-waktu yang SAMA dengan pencocoknya -
    kalau berbeda, memo akan menyebut cakupan yang tidak pernah benar-benar
    dicari.
    """
    tanggal = pd.Timestamp(tanggal)
    hasil: dict[str, int] = {}

    if table_exists("gold", "dim_alamat") and table_exists("gold", "fact_alamat_debitur"):
        dim = read_table("gold", "dim_alamat")
        # Alamat agen registrasi dibuang di cocokkan_alamat, jadi tidak boleh
        # ikut dihitung sebagai cakupan.
        layak = dim[~dim["is_alamat_agen"]]
        jembatan = _aktif_pada(read_table("gold", "fact_alamat_debitur"), tanggal)
        hasil["alamat"] = int(layak["alamat_id"].nunique())
        hasil["debitur_beralamat"] = int(
            jembatan[jembatan["alamat_id"].isin(layak["alamat_id"])]["cif_sk"].nunique()
        )

    pihak_aktif: set[int] = set()
    for tabel in ("fact_kepengurusan", "fact_kepemilikan"):
        if table_exists("gold", tabel):
            df = _aktif_pada(read_table("gold", tabel), tanggal)
            if not df.empty:
                pihak_aktif |= set(df["pihak_id"].astype("int64"))
    if pihak_aktif:
        hasil["pihak"] = len(pihak_aktif)

    if table_exists("gold", "bridge_rekening") and table_exists("gold", "fact_transfer_giro"):
        bridge = read_table("gold", "bridge_rekening", columns=["rekening_id", "src_aml_account"])
        transfer = read_table("gold", "fact_transfer_giro", columns=["waktu"])
        hasil["rekening_lawan"] = int(bridge["src_aml_account"].nunique())
        hasil["transfer_sampai_tanggal"] = int(
            (pd.to_datetime(transfer["waktu"]) <= tanggal).sum()
        )

    return hasil


def telusuri_afiliasi(
    tanggal: pd.Timestamp | str,
    alamat_operasional: str | None = None,
    nama_pengurus: list[str] | None = None,
    rekening_lawan: list[str] | None = None,
) -> HasilResolusi:
    """Satukan tiga jalur pencocokan menjadi satu daftar kandidat afiliasi.

    Keluarannya BUKAN skor risiko dan tidak boleh diperlakukan begitu. Ia daftar
    debitur eksisting yang layak diperiksa analis, lengkap dengan alasan kenapa
    masing-masing muncul - persis bentuk yang dibutuhkan penelaahan BMPK dan
    pihak terafiliasi.
    """
    tanggal = pd.Timestamp(tanggal)
    bagian: list[pd.DataFrame] = []
    terpakai: list[str] = []
    kosong: list[str] = []

    for nama_jalur, hasil in (
        ("alamat", cocokkan_alamat(alamat_operasional, tanggal) if alamat_operasional else None),
        ("pengurus", cocokkan_pengurus(nama_pengurus, tanggal) if nama_pengurus else None),
        (
            "rekening_koran",
            cocokkan_counterparty(rekening_lawan, tanggal) if rekening_lawan else None,
        ),
    ):
        if hasil is None:
            kosong.append(f"{nama_jalur} (dokumen tidak disertakan)")
            continue
        if hasil.empty:
            kosong.append(f"{nama_jalur} (tidak ada kecocokan)")
            continue
        terpakai.append(nama_jalur)
        bagian.append(hasil)

    cakupan = cakupan_pencarian(tanggal)

    if not bagian:
        # Nihil, TAPI tercatat berapa besar yang dicari. Tanpa cakupan, keadaan
        # ini tidak bisa dibedakan dari "pencarian tidak pernah jalan".
        LOG.info(
            "resolusi %s: nihil, jalur kosong %s, cakupan %s",
            tanggal.date(), kosong, cakupan,
        )
        return HasilResolusi(
            tanggal=tanggal, jalur_terpakai=terpakai, jalur_kosong=kosong, cakupan=cakupan
        )

    kandidat = pd.concat(bagian, ignore_index=True)
    kandidat["cif_sk"] = kandidat["cif_sk"].astype("int64")

    # Lampirkan gagal bayar yang SUDAH terjadi pada tanggal penilaian. Inilah
    # bahan neighbor_default_rate_1hop versi calon nasabah - dan alasan kenapa
    # filter titik-waktu di atas tidak boleh dilonggarkan.
    if table_exists("gold", "fact_default"):
        default = read_table("gold", "fact_default", columns=["cif_sk", "tanggal_default"])
        sudah = set(default[pd.to_datetime(default["tanggal_default"]) <= tanggal]["cif_sk"])
        kandidat["afiliasi_sudah_gagal_bayar"] = kandidat["cif_sk"].isin(sudah)
    else:
        kandidat["afiliasi_sudah_gagal_bayar"] = False

    LOG.info(
        "resolusi %s: %s kandidat lewat jalur %s",
        tanggal.date(),
        kandidat["cif_sk"].nunique(),
        terpakai or "-",
    )
    return HasilResolusi(
        tanggal=tanggal,
        kandidat=kandidat.reset_index(drop=True),
        jalur_terpakai=terpakai,
        jalur_kosong=kosong,
        cakupan=cakupan,
    )

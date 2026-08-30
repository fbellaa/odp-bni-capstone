"""Rantai copilot pengajuan: dokumen PDF -> fakta -> model -> keputusan.

Modul ini yang menyambung tiga lapisan yang sebelumnya terpisah di dua halaman:

    unggahan PDF ──> pembacaan dokumen ──┐
                                          ├──> entitas gabungan ──> model PD/LGD
    chat relationship manager ───────────┘                          + klaster
                                                                   + agen tool

Pembacaan dokumen punya dua jalur. Jalur penuh memakai model bahasa lokal lewat
paket `copilot` (hasilnya terstruktur dan rapi). Jalur cadangan hanya membaca
teks PDF dan menyapu angka dengan pola — dipakai saat Ollama tidak hidup, supaya
demo tetap berjalan dan halaman tetap jujur menyebut jalur mana yang dipakai.

Yang datang dari dokumen selalu ditandai sumbernya, sehingga analis bisa
membedakan angka hasil pembacaan berkas, angka dari narasi, dan angka isian
median portofolio.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from lib import copilot_lokal as ck
from lib import dummy_data, mock_engine, model_nyata as mn
from lib import parameter_kebijakan as pk
from lib import risiko_jaringan as rj

# Jenis dokumen yang diminta pada pengajuan komersial.
JENIS_DOKUMEN = {
    "laporan_keuangan": "Laporan keuangan / home statement",
    "akta": "Data kepemilikan (pemegang saham, pengurus)",
    "rekening_koran": "Rekening koran perusahaan",
    "pengajuan": "Form pengajuan (nota analisa kredit)",
}

# Dua di antaranya wajib: laporan keuangan mengisi rasio yang dipakai model PD,
# nota analisa mengisi struktur fasilitas yang diminta. Tanpa keduanya yang
# keluar adalah skor atas asumsi segmen - tampilannya sama persis dengan skor
# atas berkas nasabah, dan itu yang membuatnya menyesatkan.
DOKUMEN_WAJIB = ("laporan_keuangan", "pengajuan")

# Hijau untuk angka yang bisa dibuka ulang di berkas nasabah, biru untuk isian
# nota analisa bank, abu untuk yang diasumsikan sistem karena tidak tertulis.
SUMBER_WARNA = {
    "dokumen": "#1f8a5f", "turunan dokumen": "#1f8a5f",
    "pengajuan": "#1d5fa8", "bawaan": "#8b97a6", "rujukan": "#8b97a6",
}


@dataclass
class HasilDokumen:
    """Keluaran pembacaan berkas, apa pun jalur yang dipakai."""

    jalur: str                      # "llm" atau "pola"
    berkas: object | None = None    # BerkasPengajuan bila jalur llm
    per_berkas: list[dict] = field(default_factory=list)
    fakta: dict = field(default_factory=dict)
    sumber_fakta: dict = field(default_factory=dict)
    # {tahun_buku: pos} untuk laporan yang memuat lebih dari satu periode.
    # Kosong kalau laporannya hanya satu tahun - tren tidak dikarang dari satu titik.
    fakta_tahun: dict = field(default_factory=dict)
    pemegang_saham: list[dict] = field(default_factory=list)
    pengurus: list[str] = field(default_factory=list)
    # Isian nota analisa: struktur fasilitas, agunan, dan penilaian internal.
    pengajuan: dict = field(default_factory=dict)
    catatan: list[str] = field(default_factory=list)

    def kelengkapan(self) -> dict[str, bool]:
        ada = {j: False for j in JENIS_DOKUMEN}
        for d in self.per_berkas:
            if d.get("jenis") in ada:
                ada[d["jenis"]] = True
        return ada


# --------------------------------------------------------------------------
# Kesiapan lingkungan
# --------------------------------------------------------------------------
def status_lengkap() -> dict:
    """Gabungan kesiapan lapisan model dan lapisan copilot lokal."""
    status = {"copilot": ck.TERSEDIA, "ollama": False, "index": False,
              "galat_impor": ck.GALAT_IMPOR, "host": "-", "profil": "-"}
    if ck.TERSEDIA:
        try:
            lingkungan = ck.status_lingkungan()
            status.update(
                ollama=bool(lingkungan.get("ollama")),
                index=bool(lingkungan.get("index")),
                host=lingkungan.get("host", "-"),
                profil=lingkungan.get("profil", "-"),
                model_kurang=lingkungan.get("model_kurang", []),
            )
        except Exception as exc:
            status["galat_impor"] = f"{type(exc).__name__}: {exc}"
    # Ollama yang menjawab belum berarti modelnya sudah ditarik. Membedakan
    # keduanya menentukan jalur pembacaan: jalur LLM dengan model yang tidak ada
    # mengembalikan dokumen kosong tanpa satu pun galat yang terlihat di layar,
    # dan seluruh isian diam-diam jatuh ke nilai bawaan.
    status["llm_siap"] = bool(status["ollama"]) and not status.get("model_kurang")
    status.update(mn.status_lapisan_model())
    return status


# --------------------------------------------------------------------------
# Jalur cadangan: baca teks PDF dan sapu angka dengan pola
# --------------------------------------------------------------------------
# Keterangan skala pada kepala laporan. Bentuk "dalam jutaan rupiah" bukan
# satu-satunya yang dipakai di lapangan: laporan keuangan UKM Indonesia sering
# menulis "Dalam Rp Juta" atau "(Rp juta)", dan versi pertama pola ini hanya
# mengenali bentuk pertama - akibatnya seluruh angka terbaca 1.000.000 kali
# lebih kecil tanpa satu pun tanda di layar.
_SKALA = [
    (re.compile(r"\bdalam\s+(?:rp\.?\s*)?(?:jutaan|juta)(?:\s+rupiah)?\b", re.I), 1e6),
    (re.compile(r"\bdalam\s+(?:rp\.?\s*)?(?:ribuan|ribu)(?:\s+rupiah)?\b", re.I), 1e3),
    (re.compile(r"\bdalam\s+(?:rp\.?\s*)?(?:miliar|milyar)(?:\s+rupiah)?\b", re.I), 1e9),
    (re.compile(r"\(\s*rp\.?\s*(?:jutaan|juta)\s*\)", re.I), 1e6),
    (re.compile(r"\(\s*rp\.?\s*(?:ribuan|ribu)\s*\)", re.I), 1e3),
    (re.compile(r"\(\s*rp\.?\s*(?:miliar|milyar)\s*\)", re.I), 1e9),
]

NAMA_SKALA = {1.0: "rupiah penuh", 1e3: "ribuan", 1e6: "jutaan", 1e9: "miliar"}

# Pos yang punya konsumen di perhitungan; menambah pos berarti menambah
# kesempatan salah baca, jadi daftarnya sengaja pendek.
_POS = {
    # "hasil termijn bersih" adalah nama total pendapatan pada laporan jasa
    # konstruksi. Letaknya SESUDAH "total pendapatan": laporan yang memuat
    # keduanya (PT Arunika pada data demo) memakai hasil termijn hanya untuk
    # tagihan proyek, tanpa penjualan rumah - jadi ia total yang salah di sana,
    # dan total yang benar di laporan yang tidak punya baris "Total Pendapatan".
    "penjualan": ["penjualan bersih", "pendapatan usaha", "penjualan neto",
                  "total pendapatan", "hasil termijn bersih", "penjualan"],
    "ebitda": ["ebitda"],
    "laba_bersih": ["laba bersih", "laba tahun berjalan", "laba periode berjalan"],
    # "Biaya Bunga Bank" lazim dipakai laporan UKM; kunci umum tanpa "bank"
    # tidak cocok karena angkanya tidak menempel langsung sesudahnya.
    "beban_bunga": ["beban bunga bank", "biaya bunga bank", "beban bunga", "biaya bunga"],
    "total_aset": ["jumlah aset", "total aset", "jumlah aktiva", "total aktiva"],
    "total_liabilitas": ["jumlah liabilitas", "total liabilitas",
                         "jumlah kewajiban", "total kewajiban"],
    "ekuitas": ["jumlah ekuitas", "total ekuitas", "total modal dan laba",
                "jumlah modal"],
    "utang_berbunga": ["utang bank", "pinjaman bank", "utang berbunga"],
    "arus_kas_operasi": ["arus kas dari aktivitas operasi", "kas bersih dari operasi"],
    "saldo_rata_rata": ["saldo rata-rata", "rata-rata saldo", "saldo akhir"],
    # Pos neraca lancar. Angkanya sudah ada di laporan yang dipakai demo, hanya
    # belum punya konsumen sampai `bangun_fitur_pd` menghitung current ratio,
    # quick ratio, dan modal kerja - tujuh fitur yang sebelumnya terisi median
    # portofolio padahal bahannya tergeletak di halaman pertama dokumen.
    "aset_lancar": ["total aktiva lancar", "jumlah aktiva lancar",
                    "total aset lancar", "jumlah aset lancar"],
    "liabilitas_lancar": ["total hutang lancar", "jumlah hutang lancar",
                          "total kewajiban lancar", "jumlah kewajiban lancar",
                          "total liabilitas jangka pendek"],
    "persediaan": ["persediaan"],
    "laba_ditahan": ["laba ditahan", "saldo laba"],
    # Harga pokok pada laporan jasa konstruksi bernama "Total Biaya Proyek
    # Langsung"; nama bakunya tetap dicoba lebih dulu supaya laporan manufaktur
    # juga terbaca. "HPP Activa" pada blok pengeluaran lain-lain SENGAJA tidak
    # dijadikan kunci - itu pelepasan aset tetap, bukan harga pokok penjualan.
    "hpp": ["harga pokok penjualan", "beban pokok penjualan",
            "total biaya proyek langsung", "harga pokok"],
    "laba_kotor": ["laba bruto", "laba kotor"],
    # Pajak melengkapi bahan bangun-ulang EBITDA: laba bersih + pajak + bunga +
    # penyusutan. Tanpanya EBITDA hanya taksiran 10% dari penjualan, dan tiga
    # fitur yang bergantung padanya (debt/EBITDA, ICR, marjin operasi) ikut jadi
    # taksiran.
    "pajak": ["pajak penghasilan badan", "pajak penghasilan", "beban pajak"],
}


# Piutang dijumlahkan, bukan diambil satu baris. Pada laporan jasa konstruksi
# piutang termijn justru lebih besar daripada piutang usaha (PT Sagara Prima
# pada data demo: 62.000 lawan 38.000), sehingga mengambil "piutang usaha" saja
# membuat DSO terbaca sekitar sepertiga dari yang sebenarnya.
#
# Tiap label hanya diambil kemunculan PERTAMA - sama seperti pos lain - supaya
# rincian yang diulang di halaman penjelasan ayat tidak terhitung dua kali.
_KOMPONEN_PIUTANG = ("piutang usaha", "piutang termijn", "piutang lain-lain",
                     "piutang dagang")

# Baris penyusutan tersebar di tiga blok biaya (proyek tidak langsung, adm dan
# umum), jadi tidak ada satu baris "total penyusutan" yang bisa dikutip.
# Penjumlahannya perlu karena arus kas operasi pada data latih adalah proxy
# `laba bersih + penyusutan` (lihat `pipelines/transform/silver.py`) - laporan
# in-house memang jarang memuat laporan arus kas sungguhan.
# Angkanya ditangkap di sini, bukan lewat `_angka_setelah`: yang menempel pada
# kata "penyusutan" adalah nama aktivanya ("Mesin & Peralatan"), bukan nilainya.
# Pola juga menolak baris "Akumulasi Penyusutan ..." pada neraca - itu saldo
# akumulasi, bukan beban tahun berjalan.
_PENYUSUTAN = re.compile(
    r"^\s*(biaya\s+penyusutan\s+[\w&/ .-]{3,40}?)\s+(\(|-)?\s*([\d.,]+)\)?\s*$", re.I
)

_ANGKA = re.compile(r"\(?\s*(?:rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*\)?", re.I)


# Baris yang isinya hanya sebuah angka - bentuk yang muncul ketika laporan
# disusun sebagai tabel dan tiap sel jatuh ke barisnya sendiri saat diekstrak.
_HANYA_ANGKA = re.compile(r"^\s*\(?\s*-?\s*(?:rp\.?\s*)?\d[\d.,]*\s*\)?\s*$", re.I)


def rapatkan_tabel(teks: str) -> str:
    """Satukan baris label dengan baris angka yang mengikutinya.

    Laporan yang dicetak dari tabel - Word, LibreOffice, atau ReportLab Table -
    kehilangan hubungan kolomnya saat teks diekstrak: "Setara Kas" dan "18.500"
    berakhir sebagai dua baris. Seluruh pembacaan pos di modul ini mensyaratkan
    angka menempel pada labelnya, jadi bentuk itu dirapatkan lebih dulu.

    Penyatuan hanya terjadi bila baris berikutnya BENAR-BENAR hanya berisi angka
    dan baris sekarang belum punya angka di ujungnya. Laporan dua kolom seperti
    neraca Arunika - yang barisnya sudah lengkap - karena itu tidak tersentuh.
    """
    baris = teks.splitlines()
    keluar: list[str] = []
    lewati = False
    for i, sekarang in enumerate(baris):
        if lewati:
            lewati = False
            continue
        isi = sekarang.strip()
        berikut = baris[i + 1].strip() if i + 1 < len(baris) else ""
        if (isi and berikut and not _HANYA_ANGKA.match(isi)
                and _HANYA_ANGKA.match(berikut) and not re.search(r"\d\s*$", isi)):
            keluar.append(f"{isi} {berikut}")
            lewati = True
        else:
            keluar.append(sekarang)
    return "\n".join(keluar)


def _ke_float(teks: str, negatif: bool) -> float | None:
    """Angka Indonesia: titik ribuan, koma desimal. Kurung berarti negatif."""
    bersih = teks.strip()
    if bersih.count(",") == 1 and len(bersih.split(",")[-1]) <= 2:
        bersih = bersih.replace(".", "").replace(",", ".")
    else:
        bersih = bersih.replace(".", "").replace(",", "")
    try:
        nilai = float(bersih)
    except ValueError:
        return None
    return -nilai if negatif else nilai


def _angka_pada_baris(baris: str) -> float | None:
    cocok = _ANGKA.search(baris)
    if not cocok:
        return None
    return _ke_float(cocok.group(1), negatif=cocok.group(0).strip().startswith("("))


# Angka yang menempel LANGSUNG sesudah kata kunci - hanya spasi, titik dua, atau
# titik pemandu yang boleh menyelanya.
# Tanda minus TIDAK boleh ikut kelas pemisah: "Imbalan Pasti -2.000" akan
# terbaca +2.000 kalau minusnya dianggap garis pemandu. Kurung dan minus
# sama-sama dipakai sebagai penanda negatif di laporan Indonesia.
_ANGKA_MENEMPEL = re.compile(
    r"^[\s:.…]*(\(|-)?\s*(?:rp\.?\s*)?"
    r"(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)",
    re.I,
)


def _angka_setelah(baris: str, kata: str) -> float | None:
    """Angka yang mengikuti `kata` pada baris, bukan angka pertama baris itu.

    Dua masalah nyata yang diselesaikan aturan "menempel" ini:

    1. NERACA DUA KOLOM. Baris "Tanah 45.000 Laba Tahun Berjalan 24.500" berisi
       pos kiri dan pos kanan sekaligus. Mengambil angka pertama baris membuat
       laba tahun berjalan terbaca 45.000 - nilai tanah.

    2. PRESEDENSI KATA KUNCI. "Hasil Penjualan Rumah 72.000" mengandung kata
       "penjualan", dan kalau angka pertama baris yang diambil, baris itu menang
       lebih dulu daripada "TOTAL PENDAPATAN 387.000" yang sebenarnya dicari.
       Karena "penjualan" di situ diikuti kata "Rumah" dan bukan angka, baris
       itu kini tidak dianggap cocok sama sekali.

    Nilai None berarti "kata kunci ada tapi bukan angkanya" - dan itu memang
    lebih baik dibiarkan tidak terbaca daripada diisi angka tetangganya.
    """
    rendah = baris.lower()
    mulai = rendah.find(kata)
    while mulai != -1:
        # Kata kunci harus berdiri sendiri, bukan penggalan kata lain.
        sebelum = rendah[mulai - 1] if mulai else " "
        if not sebelum.isalpha():
            cocok = _ANGKA_MENEMPEL.match(baris[mulai + len(kata):])
            if cocok:
                return _ke_float(cocok.group(2), negatif=cocok.group(1) is not None)
        mulai = rendah.find(kata, mulai + 1)
    return None


def fakta_dari_teks(teks: str) -> dict:
    """Sapu pos keuangan dari teks PDF apa adanya.

    Nilai yang terlalu kecil untuk segmen komersial dinaikkan skalanya menurut
    keterangan "dalam jutaan/ribuan rupiah" pada kepala laporan; kalau tidak ada
    keterangan itu, angka dipakai apa adanya dan halaman menandainya.
    """
    teks = rapatkan_tabel(teks)
    faktor = 1.0
    for pola, skala in _SKALA:
        if pola.search(teks):
            faktor = skala
            break

    baris_isi = [b for b in teks.splitlines() if b.strip()]

    # Penelusuran per KATA KUNCI, bukan per baris. Daftar kunci tiap pos disusun
    # dari yang paling spesifik ke yang paling umum ("total pendapatan" sebelum
    # "penjualan"), dan urutan itu hanya berarti kalau seluruh dokumen dicoba
    # dengan kunci pertama dulu - bukan baris pertama yang memuat kunci mana pun.
    hasil: dict[str, float] = {}
    for pos, kunci in _POS.items():
        for kata in kunci:
            nilai = next(
                (v for v in (_angka_setelah(b, kata) for b in baris_isi)
                 if v is not None and v != 0),
                None,
            )
            if nilai is not None:
                hasil[pos] = nilai * faktor
                break
    return hasil


# Komponen yang boleh dijumlahkan menjadi total liabilitas kalau laporan tidak
# memuat barisnya. Sengaja hanya dua, dan hanya kalau keduanya ketemu.
_KOMPONEN_LIABILITAS = (
    ["total hutang lancar", "jumlah hutang lancar",
     "total kewajiban lancar", "jumlah kewajiban lancar"],
    ["total hutang jangka panjang", "jumlah hutang jangka panjang",
     "total kewajiban jangka panjang", "jumlah kewajiban jangka panjang"],
)


def liabilitas_turunan(teks: str) -> float | None:
    """Total liabilitas dari penjumlahan hutang lancar + jangka panjang.

    Sebagian laporan - PT Sagara Prima pada data demo salah satunya - hanya
    memuat kedua subtotal itu tanpa baris "Total Liabilitas". Tanpa penjumlahan
    ini, seluruh rasio yang memakai liabilitas (DER, debt to EBITDA) kosong
    padahal angkanya ada di dokumen.

    Hanya dipakai kalau `fakta_dari_teks` TIDAK menemukan total langsungnya, dan
    hasilnya ditandai "turunan" di `sumber_fakta` supaya analis tahu angka itu
    dihitung, bukan dikutip.
    """
    faktor = 1.0
    for pola, skala in _SKALA:
        if pola.search(teks):
            faktor = skala
            break

    baris_isi = [b for b in rapatkan_tabel(teks).splitlines() if b.strip()]
    bagian: list[float] = []
    for kunci in _KOMPONEN_LIABILITAS:
        nilai = None
        for kata in kunci:
            nilai = next(
                (v for v in (_angka_setelah(b, kata) for b in baris_isi)
                 if v is not None and v != 0),
                None,
            )
            if nilai is not None:
                break
        if nilai is None:
            return None
        bagian.append(nilai)
    return sum(bagian) * faktor


def piutang_total(teks: str) -> float | None:
    """Jumlah seluruh komponen piutang pada neraca; `None` kalau tak satu pun ada.

    Dipakai sebagai `total_receivables` pada rumus DSO, mengikuti definisi yang
    dipakai saat model dilatih (`pipelines/transform/silver.py`).
    """
    faktor = 1.0
    for pola, skala in _SKALA:
        if pola.search(teks):
            faktor = skala
            break

    baris_isi = [b for b in rapatkan_tabel(teks).splitlines() if b.strip()]
    bagian = [
        v for v in (
            next((x for x in (_angka_setelah(b, kata) for b in baris_isi)
                  if x is not None and x != 0), None)
            for kata in _KOMPONEN_PIUTANG
        )
        if v is not None
    ]
    return sum(bagian) * faktor if bagian else None


def penyusutan_total(teks: str) -> float | None:
    """Jumlah seluruh baris beban penyusutan; `None` kalau tak satu pun ada.

    Label dipakai sebagai kunci dedup supaya rincian yang diulang di halaman
    penjelasan ayat tidak menggandakan angkanya.
    """
    faktor = 1.0
    for pola, skala in _SKALA:
        if pola.search(teks):
            faktor = skala
            break

    per_label: dict[str, float] = {}
    for baris in rapatkan_tabel(teks).splitlines():
        cocok = _PENYUSUTAN.match(baris)
        if not cocok:
            continue
        label = " ".join(cocok.group(1).lower().split())
        if label in per_label:
            continue
        nilai = _ke_float(cocok.group(3), negatif=cocok.group(2) is not None)
        if nilai:
            per_label[label] = nilai
    return sum(per_label.values()) * faktor if per_label else None


_SAHAM = re.compile(
    r"(?:pt|cv|tuan|nyonya|ny\.|tn\.)?\s*([A-Z][\w.\- ]{2,60}?)\s*[-:•]?\s*"
    r"(\d{1,3}(?:[.,]\d+)?)\s*%",
    re.M,
)


# Jabatan yang menandai kolom "Posisi" pada daftar pemegang saham.
_JABATAN = re.compile(
    r"direktur|komisaris|pemegang\s+saham|presiden|pengurus|kuasa|manaj", re.I
)

# Jumlah lembar saham: selalu berpemisah ribuan, jadi nomor urut satu-dua digit
# pada kolom pertama tidak ikut tertangkap.
_JUMLAH_SAHAM = re.compile(r"^\(?\s*(\d{1,3}(?:[.,]\d{3})+)\s*\)?$")


def pemegang_saham_dari_tabel(teks: str) -> list[dict]:
    """Daftar pemegang saham berbentuk TABEL, satu sel per baris.

    Ekstraksi PDF menurunkan tabel berkolom menjadi urutan sel:

        " 1" / "Adrian Wicaksana" / " 25.000.000" / "Komisaris Utama"

    Dua hal membuat `pemegang_saham_dari_teks` buta terhadap bentuk ini: nama dan
    angkanya tidak sebaris, dan dokumennya sama sekali tidak memuat persen -
    kepemilikan dinyatakan sebagai jumlah lembar saham. Porsi karena itu dihitung
    terhadap baris TOTAL, atau terhadap jumlah seluruh baris kalau TOTAL tidak ada.
    """
    baris = [b.strip() for b in teks.splitlines() if b.strip()]

    total = None
    for i, b in enumerate(baris):
        if b.upper().startswith("TOTAL"):
            angka = _angka_pada_baris(b)
            if angka is None and i + 1 < len(baris):
                angka = _angka_pada_baris(baris[i + 1])
            if angka:
                total = angka
                break

    temuan: list[dict] = []
    for i in range(len(baris) - 2):
        nama, jumlah, posisi = baris[i], baris[i + 1], baris[i + 2]
        cocok = _JUMLAH_SAHAM.match(jumlah)
        if not cocok or not _JABATAN.search(posisi):
            continue
        if len(nama) < 3 or not any(c.isalpha() for c in nama):
            continue
        if _JUMLAH_SAHAM.match(nama) or nama.upper().startswith("TOTAL"):
            continue
        lembar = _ke_float(cocok.group(1), negatif=False)
        if not lembar:
            continue
        temuan.append({"nama": " ".join(nama.split()), "lembar": lembar,
                       "jabatan": " ".join(posisi.split())})

    if not temuan:
        return []
    pembagi = total or sum(t["lembar"] for t in temuan)
    if not pembagi:
        return []
    return sorted(
        [
            {"nama": t["nama"], "porsi": t["lembar"] / pembagi, "jabatan": t["jabatan"]}
            for t in temuan
        ],
        key=lambda p: p["porsi"],
        reverse=True,
    )[:12]


_BARIS_ANGKA = re.compile(
    r"^\(?\s*(?:rp\.?\s*)?-?\s*\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\s*\)?$", re.I
)


def nilai_blok_label(teks: str, label: str) -> list[float]:
    """Nilai untuk `label` pada tabel berbentuk blok label lalu blok angka.

    Ringkasan rekening koran keluar dari ekstraksi PDF seperti ini - empat nama
    kolom dulu, empat nilainya kemudian:

        "Saldo Awal" / "Total Debit" / "Total Kredit" / "Saldo Akhir"
        "Rp 6.800.000.000" / "Rp 6.515.000.000" / "Rp 5.815.000.000" / "Rp 6.100.000.000"

    Pasangannya ditentukan oleh POSISI: panjang blok angka menentukan berapa
    baris label tepat di atasnya yang menjadi pasangannya, lalu label dicari di
    dalam blok itu. Mencari "angka pertama sesudah label" akan salah di sini -
    yang muncul justru nilai kolom pertama, bukan kolom labelnya.
    """
    baris = [b.strip() for b in teks.splitlines() if b.strip()]
    sasaran = label.strip().lower()
    hasil: list[float] = []

    i = 0
    while i < len(baris):
        if not _BARIS_ANGKA.match(baris[i]):
            i += 1
            continue
        # Panjang blok angka yang mulai di sini.
        j = i
        while j < len(baris) and _BARIS_ANGKA.match(baris[j]):
            j += 1
        panjang = j - i
        awal_label = i - panjang
        if awal_label >= 0:
            blok = [b.lower() for b in baris[awal_label:i]]
            if not any(_BARIS_ANGKA.match(b) for b in baris[awal_label:i]):
                for k, nama in enumerate(blok):
                    if nama == sasaran:
                        nilai = _angka_pada_baris(baris[i + k])
                        if nilai is not None:
                            hasil.append(nilai)
        i = j
    return hasil


def saldo_dari_rekening_koran(teks: str) -> float | None:
    """Saldo rata-rata dari ringkasan bulanan rekening koran.

    Rekening koran lima bulan memuat lima baris "Saldo Akhir". Yang dipakai
    rata-ratanya, bukan yang pertama ditemukan: satu bulan saja tidak mewakili
    perputaran giro, dan `saldo_giro_rata` memang bermakna rata-rata.
    """
    for label in ("Saldo Rata-Rata", "Rata-Rata Saldo", "Saldo Akhir"):
        nilai = [v for v in nilai_blok_label(teks, label) if v]
        if nilai:
            return sum(nilai) / len(nilai)
    return None


def pemegang_saham_dari_teks(teks: str) -> list[dict]:
    """Baris "Nama ... 60%" pada akta atau daftar pemegang saham."""
    hasil, terlihat = [], set()
    for nama, porsi in _SAHAM.findall(teks):
        bersih = " ".join(nama.split())
        if len(bersih) < 3 or bersih.lower() in terlihat:
            continue
        try:
            nilai = float(porsi.replace(",", "."))
        except ValueError:
            continue
        if not 0 < nilai <= 100:
            continue
        terlihat.add(bersih.lower())
        hasil.append({"nama": bersih, "porsi": nilai / 100})
    return sorted(hasil, key=lambda p: p["porsi"], reverse=True)[:12]


# --------------------------------------------------------------------------
# Form pengajuan (nota analisa kredit)
# --------------------------------------------------------------------------
# Nota analisa dicetak dari sistem yang berbeda-beda antar unit, dan bentuknya
# ikut berbeda saat teksnya diekstrak:
#
#   bentuk baris   "Plafon diminta        Rp 45.000.000.000"
#   bentuk tabel   "Plafon diminta"  lalu  "Rp 45.000.000.000" di baris berikutnya
#
# Bentuk kedua muncul pada nota yang disusun sebagai tabel - dan itu bentuk yang
# paling lazim, karena nota memang formulir. Pembaca menerima keduanya: baris
# yang sama dicoba lebih dulu, baru baris berikutnya.
#
# Label juga tidak seragam. Satu unit menulis "Relationship manager", unit lain
# cukup "RM"; alias diurutkan dari yang paling spesifik supaya "Nama debitur"
# tidak kalah oleh "Nama".
ALIAS_FIELD: dict[str, tuple[str, ...]] = {
    "nomor_pengajuan": ("Nomor pengajuan", "No. Pengajuan"),
    "unit_kerja": ("Unit kerja", "Unit"),
    # "RM" dicoba sebelum "Relationship manager": nota kerap memuat blok tanda
    # tangan berjudul "Relationship Manager" di kaki halaman, dan pada bentuk
    # tabel judul kolom itu akan mengambil nama di bawahnya - yaitu nama analis
    # kredit, bukan nama RM-nya.
    "nama_rm": ("Nama RM", "RM", "Relationship manager"),
    "tanggal_pengajuan": ("Tanggal pengajuan", "Tanggal"),
    "nama_debitur": ("Nama debitur", "Nama"),
    "cif": ("CIF / NPWP", "CIF"),
    "npwp": ("NPWP",),
    "alamat_usaha": ("Alamat usaha", "Alamat"),
    "sektor": ("Sektor / KBLI", "Sektor usaha", "Sektor"),
    "jenis_fasilitas": ("Jenis fasilitas", "Jenis"),
    "tujuan_penggunaan": ("Tujuan penggunaan", "Tujuan"),
    "rating_internal": ("Rating internal awal", "Rating internal", "Rating"),
}

# Label yang menggabungkan dua isian dengan garis miring, misalnya
# "Tahun berdiri / Karyawan" -> "2011 / 214 orang".
GABUNGAN = {
    "Tahun berdiri / Karyawan": ("tahun_berdiri", "jumlah_karyawan"),
    "CIF / NPWP": ("cif", "npwp"),
}

_BARIS_RP = re.compile(r"^\s*rp\.?\s*[\d.,]+", re.I)
# Kepala seksi ("C. AGUNAN") dan label formulir lain yang menghentikan
# penyambungan baris lanjutan.
_TAMPAK_LABEL = re.compile(
    r"^\s*(?:[A-Z]\.\s|catatan|jumlah|kelengkapan|jaminan|rating|skor)",
    re.I,
)
_SKALA_NOMINAL = (("miliar", 1e9), ("milyar", 1e9), ("juta", 1e6), ("ribu", 1e3))


def _baris_bersih(teks: str) -> list[str]:
    return [b.strip() for b in teks.splitlines() if b.strip()]


def _teks_setelah(teks: str, label: str, sambung: bool = False) -> str | None:
    """Nilai sebuah label, pada baris yang sama maupun pada baris berikutnya.

    Label harus berdiri sebagai baris utuh pada bentuk tabel; pencocokan
    sebagian ditolak supaya "Jenis" tidak menyerobot "Jenis Agunan".

    `sambung` hanya dinyalakan untuk field berisi kalimat bebas yang boleh
    membungkus ke baris berikutnya. Nilai berkode pendek seperti rating tidak
    boleh ikut disambung - "BBB" tidak berakhir dengan titik, dan tanpa batas ini
    ia akan menelan seluruh sisa halaman.
    """
    baris = _baris_bersih(teks)
    rendah_label = label.lower()
    for i, b in enumerate(baris):
        rendah = b.lower()
        if not rendah.startswith(rendah_label):
            continue
        sisa = b[len(label):].lstrip(" :\t")
        if sisa:
            return sisa.strip()
        # Baris hanya berisi labelnya: nilainya ada di baris berikutnya, kecuali
        # baris itu ternyata label lain (formulir dengan isian kosong).
        if i + 1 < len(baris):
            calon = baris[i + 1]
            if not calon.endswith(":"):
                return _sambung_lanjutan(baris, i + 1, calon) if sambung else calon
        return None
    return None


def _sambung_lanjutan(baris: list[str], i: int, nilai: str) -> str:
    """Sambung baris lanjutan dari nilai yang membungkus.

    Kalimat panjang seperti tujuan penggunaan kredit terpotong di tepi kolom dan
    jatuh ke baris berikutnya, kadang berakhir dengan kata yang diawali huruf
    besar ("... kawasan industri di" / "Bekasi."). Karena itu penyambungan
    berhenti pada tanda baca akhir kalimat atau pada baris yang tampak sebagai
    label - bukan pada besar-kecilnya huruf pertama.
    """
    hasil = nilai
    j = i + 1
    while j < len(baris) and not hasil.rstrip().endswith((".", ":", ";")):
        calon = baris[j]
        if _TAMPAK_LABEL.match(calon) or _BARIS_RP.match(calon):
            break
        if any(calon.lower().startswith(a.lower())
               for alias in ALIAS_FIELD.values() for a in alias):
            break
        hasil += " " + calon
        j += 1
    return hasil


def _nominal(nilai: str | None) -> float | None:
    """Nominal rupiah dari isian formulir, menghormati satuan yang tertulis.

    "Rp 45.000.000.000" dan "Rp 45.000 juta" bernilai sama; nota yang menulis
    angkanya dalam juta tanpa penanda satuan akan terbaca 1.000.000 kali lebih
    kecil, dan itu memang harus terlihat, bukan ditebak diam-diam.
    """
    if not nilai:
        return None
    cocok = re.search(r"(\d[\d.,]*)", nilai.replace("Rp", " ").replace("rp", " "))
    if not cocok:
        return None
    angka = _ke_float(cocok.group(1), negatif=False)
    if angka is None:
        return None
    ekor = nilai[cocok.end():].lower()
    for kata, faktor in _SKALA_NOMINAL:
        if kata in ekor:
            return angka * faktor
    return angka


def _bilangan(nilai: str | None) -> int | None:
    if not nilai:
        return None
    cocok = re.search(r"(\d+)", nilai.replace(".", ""))
    return int(cocok.group(1)) if cocok else None


def _ya(nilai: str | None) -> bool:
    return bool(nilai) and nilai.strip().lower().startswith(("ya", "ada", "true"))


def _seksi(teks: str, huruf: str) -> str:
    """Isi satu seksi bernomor huruf ("C. AGUNAN" sampai "D. ...").

    Seluruh teks dikembalikan kalau penandanya tidak ada, supaya nota tanpa
    penomoran seksi tetap terbaca - hanya dengan risiko salah ambil yang lebih
    besar, dan itu pilihan yang lebih baik daripada tidak membaca sama sekali.
    """
    mulai = re.search(rf"^\s*{huruf}\.\s+\S", teks, re.M)
    if not mulai:
        return teks
    berikut = re.search(r"^\s*[A-Z]\.\s+\S", teks[mulai.end():], re.M)
    return teks[mulai.start(): mulai.end() + berikut.start()] if berikut else teks[mulai.start():]


# Bentuk baris: "1. Tanah dan bangunan ... Taksasi Rp ... Likuidasi Rp ..."
#
# Pemisahnya `\s+`, bukan `\s{2,}`: perataan kolom pada PDF hilang saat teks
# diekstrak karena pypdf merapatkan deretan spasi menjadi satu.
_AGUNAN_BARIS = re.compile(
    r"^\s*\d+\.\s+(?P<jenis>.+?)\s+taksasi\s+rp\.?\s*(?P<taksasi>[\d.,]+\s*\w*)"
    r"\s+likuidasi\s+rp\.?\s*(?P<likuidasi>[\d.,]+\s*\w*)\s*$",
    re.I | re.M,
)

# Kata yang muncul sebagai kepala kolom atau label di seksi agunan, jadi tidak
# boleh ikut terbaca sebagai nama agunan pada bentuk tabel.
_BUKAN_AGUNAN = (
    "jenis agunan", "nilai taksasi", "nilai likuidasi", "jumlah item",
    "jaminan silang", "agunan", "catatan", "uraian",
)


def _agunan_dari_tabel(teks: str) -> list[dict]:
    """Agunan pada nota berbentuk tabel: nama, lalu taksasi, lalu likuidasi.

    Tiap kolom jatuh ke barisnya sendiri saat teks diekstrak, sehingga satu item
    agunan tampil sebagai tiga baris berurutan. Baris kepala kolom disaring
    supaya "Jenis Agunan / Nilai Taksasi / Nilai Likuidasi" tidak terbaca sebagai
    item pertama.
    """
    baris = _baris_bersih(teks)
    hasil = []
    for i in range(len(baris) - 2):
        nama = baris[i]
        if _BARIS_RP.match(nama) or any(k in nama.lower() for k in _BUKAN_AGUNAN):
            continue
        if _BARIS_RP.match(baris[i + 1]) and _BARIS_RP.match(baris[i + 2]):
            hasil.append({
                "jenis": nama,
                "nilai_taksasi": _nominal(baris[i + 1]),
                "nilai_likuidasi": _nominal(baris[i + 2]),
            })
    return hasil


def pengajuan_dari_teks(teks: str) -> dict:
    """Isian nota analisa kredit. Field yang tidak tertulis TIDAK diisi.

    Yang absen sengaja dibiarkan hilang, bukan diberi nilai bawaan di sini:
    pemberian nilai bawaan terjadi satu lapis di atas, di `entitas_dari_dokumen`,
    supaya halaman bisa membedakan "tertulis di nota" dari "diasumsikan sistem".
    """
    isi: dict[str, object] = {}

    # Label gabungan lebih dulu; kalau tidak, "CIF / NPWP" akan terbaca sebagai
    # nilai CIF seluruhnya, berikut NPWP-nya.
    for label, (kiri, kanan) in GABUNGAN.items():
        nilai = _teks_setelah(teks, label)
        if nilai and "/" in nilai:
            a, b = (x.strip() for x in nilai.split("/", 1))
            isi[kiri], isi[kanan] = a, b

    # Hanya dua field yang isinya kalimat bebas dan boleh membungkus baris.
    KALIMAT = ("alamat_usaha", "tujuan_penggunaan")
    for kunci, alias in ALIAS_FIELD.items():
        if isi.get(kunci):
            continue
        for label in alias:
            nilai = _teks_setelah(teks, label, sambung=kunci in KALIMAT)
            if nilai:
                isi[kunci] = nilai
                break

    # Nomor pengajuan kerap ditulis "No. APP-2026-0451" pada kepala dokumen,
    # tanpa label bergaya formulir sama sekali.
    if not isi.get("nomor_pengajuan"):
        cocok = re.search(r"\bAPP-[\w-]+", teks)
        if cocok:
            isi["nomor_pengajuan"] = cocok.group(0)

    for kunci, label in (("tahun_berdiri", "Tahun berdiri"),
                         ("jumlah_karyawan", "Jumlah karyawan"),
                         ("tenor_bulan", "Tenor"),
                         ("jumlah_entitas_grup", "Jumlah entitas grup")):
        nilai = _bilangan(isi.get(kunci) if isinstance(isi.get(kunci), str)
                          else _teks_setelah(teks, label))
        if nilai is not None:
            isi[kunci] = nilai

    plafon = _nominal(_teks_setelah(teks, "Plafon diminta"))
    if plafon:
        isi["plafon_diminta"] = plafon

    for label in ("Skor kredit internal", "Skor kredit"):
        skor = _teks_setelah(teks, label)
        cocok = re.search(r"(\d+(?:[.,]\d+)?)", skor or "")
        if cocok:
            isi["skor_kredit"] = float(cocok.group(1).replace(",", "."))
            break

    seksi_agunan = _seksi(teks, "C")
    agunan = [
        {"jenis": m.group("jenis").strip(),
         "nilai_taksasi": _nominal(m.group("taksasi")),
         "nilai_likuidasi": _nominal(m.group("likuidasi"))}
        for m in _AGUNAN_BARIS.finditer(seksi_agunan)
    ] or _agunan_dari_tabel(seksi_agunan)
    if agunan:
        isi["agunan"] = agunan

    for kunci, label in (("ada_jaminan_silang", "Jaminan silang grup"),
                         ("indikasi_konsentrasi_pembeli", "Konsentrasi pembeli"),
                         ("indikasi_konsentrasi_pemasok", "Konsentrasi pemasok")):
        nilai = _teks_setelah(teks, label) or _teks_setelah(teks, label.split(" grup")[0])
        isi[kunci] = _ya(nilai)

    lengkap = _teks_setelah(teks, "Kelengkapan berkas")
    if lengkap:
        isi["dokumen_lengkap"] = lengkap.strip().lower().startswith("lengkap")
    return isi


# Kepala seksi yang menyebut periode: "PER 31 MEI 2026" pada neraca dan
# "Untuk Tahun Yang Berakhir 31 Mei 2026" pada laba rugi.
_TAHUN_SEKSI = re.compile(r"(?:per|berakhir)\s+\d{1,2}\s+[a-z]+\s+(20\d{2})", re.I)


def _sapu_lengkap(teks: str) -> dict:
    """Seluruh pos yang bisa dibaca dari satu blok teks, termasuk yang turunan."""
    fakta = fakta_dari_teks(teks)
    if "total_liabilitas" not in fakta:
        nilai = liabilitas_turunan(teks)
        if nilai:
            fakta["total_liabilitas"] = nilai
    for kunci, fungsi in (("piutang", piutang_total), ("penyusutan", penyusutan_total)):
        if kunci not in fakta:
            nilai = fungsi(teks)
            if nilai:
                fakta[kunci] = nilai
    return fakta


def fakta_per_tahun(halaman: list) -> dict[int, dict]:
    """Pos keuangan per periode untuk laporan yang memuat beberapa tahun.

    Tahun diambil dari kepala tiap halaman dan dibawa turun ke halaman-halaman
    berikutnya sampai kepala berikutnya muncul, sehingga lampiran penjelasan
    ayat ikut ke periode yang benar. Halaman sebelum kepala pertama diabaikan:
    menebak periodenya berarti menaruh angka pada tahun yang salah.

    Laporan satu periode mengembalikan dict berisi satu tahun, dan pemanggilnya
    yang memutuskan bahwa satu titik tidak cukup untuk menghitung tren.
    """
    teks_tahun: dict[int, list[str]] = {}
    berjalan: int | None = None
    for h in halaman:
        if not h.teks.strip():
            continue
        cocok = _TAHUN_SEKSI.search(h.teks)
        if cocok:
            berjalan = int(cocok.group(1))
        if berjalan is None:
            continue
        teks_tahun.setdefault(berjalan, []).append(h.teks)

    hasil = {}
    for tahun, bagian in teks_tahun.items():
        fakta = _sapu_lengkap("\n".join(bagian))
        if fakta.get("penjualan"):
            hasil[tahun] = fakta
    return hasil


def baca_dengan_pola(path_list: list[Path], jenis_manual: dict[str, str]) -> HasilDokumen:
    """Jalur cadangan tanpa model bahasa."""
    from copilot.dokumen import pdf as pdf_util

    hasil = HasilDokumen(jalur="pola")
    fakta: dict[str, float] = {}
    turunan: set[str] = set()
    skala_terbaca: set[str] = set()
    for path in path_list:
        try:
            halaman = pdf_util.baca_halaman(path)
        except Exception as exc:
            hasil.catatan.append(f"{Path(path).name}: {exc}")
            continue
        teks = pdf_util.gabung_teks(halaman)
        jenis = jenis_manual.get(Path(path).name) or pdf_util.tebak_jenis(halaman)[0]
        temuan = fakta_dari_teks(teks)
        for kunci, nilai in temuan.items():
            fakta.setdefault(kunci, nilai)

        # Skala dicatat supaya halaman bisa menyebutnya. Laporan yang tidak
        # menuliskan satuannya sama sekali terbaca sebagai rupiah penuh, dan
        # itu perlu terlihat - bukan diam-diam meleset seribu kali lipat.
        skala_terbaca.add(NAMA_SKALA[next(
            (s for pola, s in _SKALA if pola.search(teks)), 1.0
        )])

        if jenis == "rekening_koran" and "saldo_rata_rata" not in fakta:
            saldo = saldo_dari_rekening_koran(teks)
            if saldo:
                fakta["saldo_rata_rata"] = saldo
                temuan = {**temuan, "saldo_rata_rata": saldo}

        if "total_liabilitas" not in fakta:
            gabungan = liabilitas_turunan(teks)
            if gabungan:
                fakta["total_liabilitas"] = gabungan
                turunan.add("total_liabilitas")
                temuan = {**temuan, "total_liabilitas (turunan)": gabungan}

        if "piutang" not in fakta:
            piutang = piutang_total(teks)
            if piutang:
                fakta["piutang"] = piutang
                turunan.add("piutang")
                temuan = {**temuan, "piutang (turunan)": piutang}

        if "penyusutan" not in fakta:
            susut = penyusutan_total(teks)
            if susut:
                fakta["penyusutan"] = susut
                turunan.add("penyusutan")
                temuan = {**temuan, "penyusutan (turunan)": susut}

        if jenis == "laporan_keuangan" and not hasil.fakta_tahun:
            per_tahun = fakta_per_tahun(halaman)
            if len(per_tahun) > 1:
                hasil.fakta_tahun = per_tahun
                temuan = {**temuan, "periode": ", ".join(
                    str(x) for x in sorted(per_tahun))}

        if jenis == "pengajuan" and not hasil.pengajuan:
            hasil.pengajuan = pengajuan_dari_teks(teks)
            temuan = {**temuan, "isian nota": len(hasil.pengajuan)}

        if jenis == "akta":
            # Bentuk "Nama ... 60%" dicoba dulu; kalau dokumennya tabel berisi
            # jumlah lembar saham (tanpa persen sama sekali), pembaca tabel yang
            # mengambil alih.
            saham = pemegang_saham_dari_teks(teks) or pemegang_saham_dari_tabel(teks)
            hasil.pemegang_saham.extend(saham)
            # Nama pengurus dibutuhkan pencocokan afiliasi di halaman 3. Jalur
            # LLM sudah mengisinya; tanpa baris ini jalur pola tidak pernah.
            for orang in saham:
                jabatan = orang.get("jabatan")
                if jabatan and re.search(r"direktur|komisaris", jabatan, re.I):
                    label = f"{orang['nama']} — {jabatan}"
                    if label not in hasil.pengurus:
                        hasil.pengurus.append(label)
        hasil.per_berkas.append({
            "berkas": Path(path).name,
            "jenis": jenis,
            "halaman": len([h for h in halaman if h.teks.strip()]),
            "total_halaman": len(halaman),
            "pos_terbaca": ", ".join(temuan) or "-",
        })
    hasil.fakta = fakta
    hasil.sumber_fakta = {
        k: ("turunan dokumen" if k in turunan else "dokumen") for k in fakta
    }
    if fakta:
        hasil.catatan.append(
            "Angka disapu dengan pola dari teks PDF, tanpa model bahasa. "
            "Periksa ulang sebelum dipakai untuk keputusan."
        )
        hasil.catatan.append(
            "Satuan angka terbaca: " + ", ".join(sorted(skala_terbaca))
            + ". Kalau ini tidak sesuai kepala laporan, seluruh nominal meleset "
            "berkelipatan seribu."
        )
    if turunan:
        hasil.catatan.append(
            "Total liabilitas tidak ada sebagai satu baris; dihitung dari hutang "
            "lancar + hutang jangka panjang."
        )
    return hasil


def baca_dengan_llm(path_list: list[Path], jenis_manual: dict[str, str]) -> HasilDokumen:
    """Jalur penuh: `copilot.dokumen.ekstraksi` dengan model bahasa lokal."""
    berkas = ck.baca_dokumen(list(path_list), jenis_manual)
    hasil = HasilDokumen(jalur="llm", berkas=berkas)
    for d in berkas.dokumen:
        hasil.per_berkas.append({
            "berkas": d.sumber.berkas,
            "jenis": d.jenis,
            "halaman": len(d.sumber.halaman),
            "total_halaman": d.sumber.jumlah_halaman,
            "pos_terbaca": "; ".join(d.catatan) or "terbaca",
        })

    form = berkas.pengajuan_utama
    if form is not None:
        # Bentuk dict-nya sengaja dibuat sama dengan keluaran jalur pola supaya
        # `entitas_dari_dokumen` tidak perlu tahu jalur mana yang dipakai.
        hasil.pengajuan = {
            k: v for k, v in form.model_dump(exclude_none=True).items() if v != []
        }
        if form.agunan:
            hasil.pengajuan["agunan"] = [a.model_dump() for a in form.agunan]

    lapkeu = berkas.lapkeu_terbaru
    fakta: dict[str, float] = {}
    if lapkeu is not None:
        for pos in ("penjualan", "ebitda", "laba_bersih", "beban_bunga", "total_aset",
                    "total_liabilitas", "ekuitas", "utang_berbunga", "arus_kas_operasi"):
            nilai = getattr(lapkeu, pos, None)
            if nilai:
                fakta[pos] = float(nilai)
    rekening = berkas.semua_rekening_koran
    if rekening:
        saldo = [r.saldo_rata_rata for r in rekening if r.saldo_rata_rata]
        if saldo:
            fakta["saldo_rata_rata"] = float(sum(saldo) / len(saldo))
        else:
            # Tanpa baris saldo rata-rata, mutasi kredit dipakai sebagai
            # perkiraan kasar perputaran rekening dan ditandai demikian.
            masuk = sum(r.total_kredit for r in rekening)
            if masuk:
                fakta["mutasi_kredit"] = float(masuk)

    akta = berkas.akta_utama
    if akta is not None:
        hasil.pemegang_saham = [
            {"nama": p.nama, "porsi": (p.persentase or 0) / 100 if (p.persentase or 0) > 1
             else (p.persentase or 0), "jenis": p.jenis}
            for p in akta.pemegang_saham
        ]
        hasil.pengurus = [f"{p.nama} — {p.jabatan or 'pengurus'}" for p in akta.pengurus]
    hasil.fakta = fakta
    hasil.sumber_fakta = {k: "dokumen" for k in fakta}
    if berkas.nama_debitur:
        hasil.fakta["nama_debitur"] = berkas.nama_debitur
        hasil.sumber_fakta["nama_debitur"] = "dokumen"
    return _tambal_dengan_pola(hasil, path_list, jenis_manual)


def _tambal_dengan_pola(
    hasil: HasilDokumen, path_list: list[Path], jenis_manual: dict[str, str]
) -> HasilDokumen:
    """Isi bagian yang tidak dihasilkan model bahasa dengan sapuan pola.

    Model bahasa lokal bisa gagal diam-diam - modelnya belum ditarik, potongan
    terlalu panjang, JSON-nya tidak valid - dan hasilnya dokumen kosong tanpa
    galat yang terlihat. Halaman lalu memakai nilai bawaan untuk plafon dan
    agunan, dan angka bawaan itu tampil sebagai hasil analisa yang meyakinkan.
    Itu kegagalan paling berbahaya di antara semua yang mungkin terjadi di sini.

    Sapuan pola karena itu dijalankan sebagai jaring, bukan sebagai pengganti:
    hanya bagian yang KOSONG yang diisi, sehingga hasil model tetap menang di
    mana ia berhasil. Nota analisa adalah formulir dengan label tetap, jadi pola
    justru lebih tepat di sana daripada model 3B.
    """
    kurang = [
        nama for nama, ada in (
            ("pos keuangan", bool(hasil.fakta)),
            ("periode laporan", bool(hasil.fakta_tahun)),
            ("isian nota analisa", bool(hasil.pengajuan)),
            ("pemegang saham", bool(hasil.pemegang_saham)),
        ) if not ada
    ]
    if not kurang:
        return hasil

    try:
        pola = baca_dengan_pola(path_list, jenis_manual)
    except Exception as exc:
        hasil.catatan.append(f"Sapuan pola cadangan gagal: {exc}")
        return hasil

    if not hasil.fakta and pola.fakta:
        hasil.fakta = pola.fakta
        hasil.sumber_fakta = pola.sumber_fakta
    if not hasil.fakta_tahun:
        hasil.fakta_tahun = pola.fakta_tahun
    if not hasil.pengajuan:
        hasil.pengajuan = pola.pengajuan
    if not hasil.pemegang_saham:
        hasil.pemegang_saham = pola.pemegang_saham
        hasil.pengurus = hasil.pengurus or pola.pengurus

    hasil.jalur = "llm+pola"
    hasil.catatan.append(
        "Model bahasa tidak menghasilkan " + ", ".join(kurang)
        + ". Bagian itu diisi sapuan pola; periksa ulang sebelum dipakai untuk keputusan."
    )
    return hasil


def jenis_unggahan(unggahan, jenis_manual: dict[str, str]) -> set[str]:
    """Jenis tiap berkas yang sedang dipilih, tanpa menuliskannya ke disk dulu.

    Dipakai daftar kelengkapan yang tampil SEBELUM tombol ditekan, jadi berkas
    memang belum boleh disimpan. Kegagalan menebak dibiarkan diam: daftar ini
    alat bantu, dan pembacaan sebenarnya tetap terjadi di `baca_dengan_pola`.
    """
    from copilot.dokumen import pdf as pdf_util

    terbaca: set[str] = set()
    for berkas in unggahan or []:
        manual = jenis_manual.get(berkas.name)
        if manual:
            terbaca.add(manual)
            continue
        try:
            from pypdf import PdfReader

            pembaca = PdfReader(io.BytesIO(berkas.getvalue()))
            halaman = [
                pdf_util.HalamanPDF(nomor=i, teks=hal.extract_text() or "")
                for i, hal in enumerate(pembaca.pages[:3], start=1)
            ]
            terbaca.add(pdf_util.tebak_jenis(halaman)[0])
        except Exception:
            continue
    return terbaca


def simpan_unggahan(unggahan) -> Path:
    """Tulis unggahan Streamlit ke cakram; pembaca PDF butuh path."""
    if ck.TERSEDIA:
        return ck.simpan_unggahan(unggahan)
    tujuan = Path(st.session_state.get("_dir_unggahan", ".")) / unggahan.name
    tujuan.write_bytes(unggahan.getbuffer())
    return tujuan


# --------------------------------------------------------------------------
# Penggabungan narasi dan dokumen
# --------------------------------------------------------------------------
# Nilai bawaan untuk field yang tidak tertulis di nota analisa. Dipisahkan ke
# konstanta supaya jelas mana yang datang dari berkas dan mana yang asumsi
# sistem - halaman menandai keduanya dengan warna berbeda.
BAWAAN_ENTITAS = {
    "jenis_fasilitas": "Modal kerja - rekening koran",
    "sektor": "Kontraktor infrastruktur",
    "wilayah": "Karawang",
    "umur_usaha_thn": 14.0,
    "penjualan_tahunan": 200e9,
    "ebitda_margin": 0.11,
    "der": 1.8,
    "plafon": 80e9,
    "tenor_bulan": 36,
    "jenis_agunan": "Tanpa agunan (clean basis)",
    "nilai_agunan": 0.0,
    "saldo_giro_rata": 15e9,
    "jumlah_entitas_grup": 1,
    "indikasi_rangkap_jabatan": False,
    "indikasi_konsentrasi_pembeli": False,
    "indikasi_konsentrasi_pemasok": False,
}

TAHUN_PENILAIAN = 2026


def _padankan_agunan(teks: str) -> str:
    """Nama agunan pada nota dipetakan ke kosakata `mock_engine.RECOVERY_AGUNAN`."""
    rendah = (teks or "").lower()
    for nama in dummy_data.JENIS_AGUNAN:
        if nama.lower() in rendah:
            return nama
    for nama, kunci in dummy_data.KUNCI_AGUNAN:
        if any(k in rendah for k in kunci):
            return nama
    return "Tanpa agunan (clean basis)"


def _entitas_dari_form(form: dict) -> tuple[dict, dict]:
    """Bagian entitas yang berasal dari nota analisa kredit."""
    entitas: dict[str, object] = {}
    asal: dict[str, str] = {}

    def pakai(kunci: str, nilai) -> None:
        if nilai is not None:
            entitas[kunci] = nilai
            asal[kunci] = "pengajuan"

    pakai("nama_debitur", form.get("nama_debitur"))
    pakai("alamat_usaha", form.get("alamat_usaha"))
    pakai("jenis_fasilitas", form.get("jenis_fasilitas"))
    pakai("tujuan_penggunaan", form.get("tujuan_penggunaan"))
    pakai("plafon", form.get("plafon_diminta"))
    pakai("tenor_bulan", form.get("tenor_bulan"))
    pakai("jumlah_karyawan", form.get("jumlah_karyawan"))
    pakai("rating_internal", form.get("rating_internal"))
    pakai("skor_kredit", form.get("skor_kredit"))
    pakai("jumlah_entitas_grup", form.get("jumlah_entitas_grup"))
    pakai("indikasi_konsentrasi_pembeli", form.get("indikasi_konsentrasi_pembeli"))
    pakai("indikasi_konsentrasi_pemasok", form.get("indikasi_konsentrasi_pemasok"))
    # Jaminan silang dan rangkap jabatan dua hal berbeda, dan hanya yang pertama
    # dinyatakan di nota. Rangkap jabatan adalah TEMUAN lapisan graf; nota tidak
    # dipakai sebagai sumbernya, supaya temuan tidak dikonfirmasi oleh dokumen
    # yang ditulis pihak pengusulnya sendiri.
    pakai("ada_jaminan_silang", form.get("ada_jaminan_silang"))
    if "dokumen_lengkap" in form:
        entitas["dokumen_ringkas"] = not form["dokumen_lengkap"]
        asal["dokumen_ringkas"] = "pengajuan"

    # Sektor ditulis "Kontraktor infrastruktur (F)"; kode KBLI dalam kurung
    # dibuang karena dimensi warehouse memakai nama sektornya.
    sektor = form.get("sektor")
    if sektor:
        pakai("sektor", re.sub(r"\s*\([A-Z]\)\s*$", "", str(sektor)).strip())

    # Wilayah tidak punya barisnya sendiri di nota, tetapi alamat usaha memuatnya.
    # Menurunkannya dari alamat lebih baik daripada meminta satu field lagi, dan
    # kalau kotanya di luar daftar segmen, wilayah memang dibiarkan bawaan
    # ketimbang dipaksa ke kota terdekat yang kebetulan ada di daftar.
    alamat = str(form.get("alamat_usaha") or "").lower()
    if alamat:
        kota = next((w for w in dummy_data.WILAYAH if w.lower() in alamat), None)
        if kota:
            pakai("wilayah", kota)

    tahun = form.get("tahun_berdiri")
    if tahun:
        pakai("umur_usaha_thn", float(TAHUN_PENILAIAN - int(tahun)))

    agunan = form.get("agunan") or []
    if agunan:
        # Kalau lebih dari satu, yang menentukan LGD adalah agunan dengan tingkat
        # pemulihan tertinggi - sama seperti pengikatan berjenjang di lapangan.
        nama = [_padankan_agunan(a.get("jenis", "")) for a in agunan]
        entitas["jenis_agunan"] = max(
            nama, key=lambda a: mock_engine.RECOVERY_AGUNAN.get(a, 0.0))
        entitas["nilai_agunan"] = float(sum(a.get("nilai_taksasi") or 0.0 for a in agunan))
        entitas["nilai_likuidasi"] = float(
            sum(a.get("nilai_likuidasi") or 0.0 for a in agunan))
        entitas["jumlah_agunan"] = len(agunan)
        for kunci in ("jenis_agunan", "nilai_agunan", "nilai_likuidasi", "jumlah_agunan"):
            asal[kunci] = "pengajuan"
    return entitas, asal


def entitas_dari_dokumen(dokumen: HasilDokumen | None) -> tuple[dict, dict]:
    """Entitas final untuk model, seluruhnya dari berkas yang diunggah.

    Sebelumnya bagian ini menggabungkan narasi chat relationship manager dengan
    dokumen, dan angka dokumen menang atas angka narasi. Narasi sudah tidak ada:
    empat berkas - laporan keuangan tiga periode, data kepemilikan, rekening
    koran, dan nota analisa pengajuan - menutup seluruh masukan yang dipakai
    model.

    Yang tetap ada adalah pembedaan asal-usulnya. Tiap kunci dipetakan ke
    "pengajuan", "dokumen", "turunan dokumen", atau "bawaan", supaya halaman bisa
    menunjukkan angka mana yang bisa dibuka ulang saat komite bertanya dan angka
    mana yang diasumsikan sistem karena berkasnya tidak memuatnya.
    """
    entitas = dict(BAWAAN_ENTITAS)
    asal = {k: "bawaan" for k in entitas}

    if dokumen is None:
        return entitas, asal

    if dokumen.pengajuan:
        dari_form, asal_form = _entitas_dari_form(dokumen.pengajuan)
        entitas.update(dari_form)
        asal.update(asal_form)

    f = dokumen.fakta
    if f.get("nama_debitur"):
        entitas["nama_debitur"] = f["nama_debitur"]
        asal["nama_debitur"] = "dokumen"
    if f.get("penjualan"):
        entitas["penjualan_tahunan"] = float(f["penjualan"])
        asal["penjualan_tahunan"] = "dokumen"
    if f.get("ebitda"):
        entitas["ebitda_rp"] = float(f["ebitda"])
        asal["ebitda_rp"] = "dokumen"
        if entitas.get("penjualan_tahunan"):
            entitas["ebitda_margin"] = float(f["ebitda"]) / entitas["penjualan_tahunan"]
            asal["ebitda_margin"] = "dokumen"
    for pos, kunci in [
        ("ekuitas", "ekuitas_rp"), ("total_aset", "total_aset_rp"),
        ("total_liabilitas", "total_liabilitas_rp"), ("beban_bunga", "beban_bunga_rp"),
        ("laba_bersih", "laba_bersih_rp"),
        # Pos neraca lancar: bahan current ratio, quick ratio, modal kerja, DSO,
        # dan DIO. Tanpa baris ini kelimanya tetap terisi median portofolio.
        ("aset_lancar", "aset_lancar_rp"), ("liabilitas_lancar", "liabilitas_lancar_rp"),
        ("persediaan", "persediaan_rp"), ("piutang", "piutang_rp"),
        ("laba_ditahan", "laba_ditahan_rp"), ("hpp", "hpp_rp"),
        ("laba_kotor", "laba_kotor_rp"), ("penyusutan", "penyusutan_rp"),
        ("arus_kas_operasi", "arus_kas_operasi_rp"), ("pajak", "pajak_rp"),
    ]:
        if f.get(pos):
            entitas[kunci] = float(f[pos])
            asal[kunci] = dokumen.sumber_fakta.get(pos, "dokumen")
    if f.get("ekuitas") and f.get("total_liabilitas"):
        entitas["der"] = float(f["total_liabilitas"]) / max(float(f["ekuitas"]), 1.0)
        asal["der"] = "dokumen"
    if f.get("saldo_rata_rata"):
        entitas["saldo_giro_rata"] = float(f["saldo_rata_rata"])
        asal["saldo_giro_rata"] = "dokumen"

    # Laporan in-house tidak memuat baris EBITDA, tetapi memuat seluruh
    # komponennya. Menyusunnya di sini - bukan hanya di dalam model - membuat
    # `ebitda_margin` berhenti ditandai asumsi sistem padahal angkanya turun
    # dari berkas.
    if not entitas.get("ebitda_rp") and entitas.get("penjualan_tahunan"):
        disusun = mn.ebitda_bangun(
            f.get("laba_bersih"), f.get("pajak"), f.get("beban_bunga"), f.get("penyusutan"))
        if disusun and disusun > 0:
            entitas["ebitda_margin"] = disusun / float(entitas["penjualan_tahunan"])
            asal["ebitda_margin"] = "turunan dokumen"

    if dokumen.fakta_tahun:
        entitas["riwayat_tahun"] = dict(dokumen.fakta_tahun)
        asal["riwayat_tahun"] = "dokumen"

    if dokumen.pemegang_saham:
        entitas["jumlah_pemegang_saham"] = len(dokumen.pemegang_saham)
        entitas["porsi_pengendali"] = float(dokumen.pemegang_saham[0]["porsi"])
        asal["jumlah_pemegang_saham"] = "dokumen"
        asal["porsi_pengendali"] = "dokumen"
    return entitas, asal


def lengkapi_fitur_graf(
    entitas: dict,
    application_id: str,
    hasil_jaringan=None,
    tanggal_telaah=None,
) -> dict:
    """Fitur yang datangnya dari lapisan graf dan riwayat, bukan dari berkas.

    Nilai awalnya median portofolio dari `parameter_kebijakan.asumsi_portofolio()`,
    bukan lagi konstanta yang ditulis tangan. Bila `hasil_jaringan` (lihat
    `lib/risiko_jaringan.py`) membawa komponen hasil pencocokan afiliasi ke
    `data/gold`, nilai nyata itulah yang menang.

    Tiap fitur membawa asalnya pada `fitur["asal_fitur"]`: median portofolio,
    turunan dokumen, atau pengukuran atas afiliasi tercocok. Tanpa itu, angka
    yang diukur dan angka yang dipinjam dari portofolio tampil sama saja di
    layar - dan hanya salah satunya yang boleh dibela di depan komite.
    """
    rujukan, asal_rujukan = pk.asumsi_portofolio()
    fitur = dict(entitas)
    asal_fitur: dict[str, str] = {}

    # Utang berbunga eksisting: porsi dari total liabilitas yang benar-benar
    # berbunga, bukan setengahnya. Taksiran lama 0,50 membuat ICR dan DSCR
    # pemohon terbaca lebih sehat daripada portofolio yang melahirkannya.
    liabilitas = entitas.get("total_liabilitas_rp")
    if liabilitas:
        fitur["utang_berbunga_eksisting"] = float(liabilitas) * rujukan["porsi_utang_berbunga"]
        asal_fitur["utang_berbunga_eksisting"] = (
            f"total liabilitas dokumen x {rujukan['porsi_utang_berbunga']:.3f} "
            "(porsi berbunga, median portofolio)")
    else:
        fitur["utang_berbunga_eksisting"] = float(entitas["plafon"]) * 0.25
        asal_fitur["utang_berbunga_eksisting"] = "taksiran seperempat plafon (liabilitas tak terbaca)"

    for kunci in ("konversi_ebitda_kas", "utilisasi_plafon", "buyer_concentration_hhi",
                  "supplier_concentration_hhi", "neighbor_default_rate_1hop",
                  "group_exposure_share"):
        fitur[kunci] = rujukan[kunci]
        asal_fitur[kunci] = asal_rujukan[kunci]

    # Konversi EBITDA ke kas bisa dihitung langsung bila laporan memuat arus kas
    # operasi - angka pemohon sendiri, bukan median siapa-siapa.
    arus = entitas.get("arus_kas_operasi_rp")
    ebitda = entitas.get("ebitda_rp")
    if arus and ebitda and float(ebitda) > 0:
        fitur["konversi_ebitda_kas"] = float(min(max(float(arus) / float(ebitda), 0.05), 2.0))
        asal_fitur["konversi_ebitda_kas"] = "dokumen: arus kas operasi / EBITDA"

    # Lama menjadi nasabah tidak ada sumbernya di mana pun - ia turunan umur
    # usaha, dan disebut turunan supaya tidak terbaca sebagai riwayat relasi.
    fitur["tenure_nasabah_thn"] = max(float(entitas.get("umur_usaha_thn", 10.0)) - 6.0, 0.0)
    asal_fitur["tenure_nasabah_thn"] = "turunan umur usaha, bukan riwayat relasi"
    fitur["asal_fitur"] = asal_fitur

    if hasil_jaringan is None or not getattr(hasil_jaringan, "tersedia", False):
        # Tanpa pencocokan afiliasi tidak ada fitur graf yang bisa diukur.
        # Skor jaringan sengaja TIDAK diisi angka apa pun di sini.
        fitur["asal_fitur_graf"] = "median_portofolio"
        return fitur

    nyata = {k.kunci: k for k in hasil_jaringan.komponen}
    # Komponen indikator menyimpan nilai ternormalisasi; yang dibutuhkan model
    # adalah besaran aslinya, jadi hanya yang skalanya memang 0..1 yang dipakai
    # langsung. Sisanya dibiarkan pada asumsi daripada dikonversi balik.
    if "neighbor_default_rate_1hop" in nyata:
        fitur["neighbor_default_rate_1hop"] = (
            nyata["neighbor_default_rate_1hop"].nilai * rj.JENUH_DEFAULT_RATE
        )
        asal_fitur["neighbor_default_rate_1hop"] = "feat_graf_pit atas afiliasi tercocok"
    if "group_exposure_share" in nyata:
        fitur["group_exposure_share"] = nyata["group_exposure_share"].nilai
        asal_fitur["group_exposure_share"] = "feat_graf_pit atas afiliasi tercocok"
    if hasil_jaringan.skor is not None:
        fitur["network_risk_score"] = hasil_jaringan.skor

    # Eksposur BMPK grup dalam rupiah, bukan porsi. Batas kredit dihitung atas
    # sisa ruang grup, jadi angkanya harus datang dari grup yang benar-benar
    # ditunjuk cocokan - bukan dari porsi asumsi dikali batas rata-rata.
    cif = tuple(hasil_jaringan.cif_tercocok)
    utang = pk.utang_berbunga(cif)
    if utang:
        fitur["utang_berbunga_eksisting"] = utang["nilai"]
        asal_fitur["utang_berbunga_eksisting"] = (
            f"{utang['sumber']} atas {utang['jumlah_fasilitas']} fasilitas tercocok")

    bmpk = pk.eksposur_grup(
        cif,
        tanggal_telaah or hasil_jaringan.snapshot or pd.Timestamp.today(),
    )
    if bmpk:
        fitur["eksposur_grup_rp"] = bmpk["eksposur_rp"]
        fitur["batas_bmpk_rp"] = bmpk["batas_bmpk_rp"]
        fitur["group_exposure_share"] = bmpk["share"]
        fitur["bmpk_grup_id"] = bmpk["grup_id"]
        fitur["bmpk_snapshot"] = bmpk["snapshot"]
        fitur["asal_bmpk"] = bmpk["sumber"]
        fitur["catatan_bmpk"] = bmpk["catatan"]
    else:
        fitur["asal_bmpk"] = "tidak ada grup debitur yang bisa ditunjuk"

    fitur["asal_fitur_graf"] = "data_gold"
    return fitur

"""Pembacaan teks PDF dan penebakan jenis dokumen.

Lapisan ini sengaja bodoh: ia hanya mengeluarkan teks per halaman dan menebak
jenis dokumen lewat kata kunci. Seluruh penafsiran diserahkan ke
`copilot.dokumen.ekstraksi`.

Pemisahan itu ada alasannya. Penebakan jenis dokumen dengan kata kunci bisa
diuji tanpa model dan berjalan dalam milidetik, sedangkan tiap pemanggilan LLM
di CPU Kaggle berbiaya puluhan detik. Semakin banyak yang bisa diputuskan
sebelum model dipanggil, semakin pendek demonya.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from copilot.dokumen.skema import JenisDokumen, Sumber

LOG = logging.getLogger(__name__)

# pypdf mencatat tiap baris CMap yang tidak bisa diurai sebagai peringatan
# tersendiri. Satu PDF dengan font tertanam bisa menghasilkan ratusan baris
# semacam "Skipping broken line ...: Odd-length string" - tidak satu pun bisa
# ditindaklanjuti, dan semuanya menenggelamkan log yang benar-benar penting di
# terminal saat demo. Teks halamannya sendiri tetap terbaca utuh.
logging.getLogger("pypdf").setLevel(logging.ERROR)


class GalatPDF(RuntimeError):
    """PDF tidak bisa dibaca."""


@dataclass
class HalamanPDF:
    nomor: int  # 1-indeks, sama dengan yang dilihat orang di pembaca PDF
    teks: str


# Kata kunci penanda jenis dokumen. Bobot memisahkan penanda kuat ("mutasi
# rekening") dari kata yang bisa muncul di dokumen mana pun ("saldo").
PENANDA: dict[JenisDokumen, list[tuple[str, int]]] = {
    "rekening_koran": [
        ("mutasi rekening", 5), ("rekening koran", 5), ("account statement", 4),
        ("saldo awal", 3), ("saldo akhir", 3), ("debet", 2), ("debit", 1),
        ("kredit", 1), ("no. rekening", 3), ("nomor rekening", 3),
    ],
    "laporan_keuangan": [
        ("laporan posisi keuangan", 5), ("laporan laba rugi", 5), ("neraca", 4),
        ("catatan atas laporan keuangan", 5), ("ekuitas", 3), ("liabilitas", 3),
        ("arus kas", 3), ("penjualan bersih", 3), ("pendapatan usaha", 3),
        ("ebitda", 2), ("aset lancar", 3),
    ],
    "akta": [
        ("akta pendirian", 5), ("notaris", 4), ("anggaran dasar", 5),
        ("perseroan terbatas", 3), ("pemegang saham", 4), ("direktur utama", 3),
        ("komisaris", 3), ("kementerian hukum", 4), ("modal disetor", 3),
    ],
}

AMBANG_PENANDA = 5


def baca_halaman(path: str | Path) -> list[HalamanPDF]:
    """Teks per halaman. Halaman kosong tetap dikembalikan agar nomor tidak geser."""
    path = Path(path)
    if not path.exists():
        raise GalatPDF(f"Berkas tidak ditemukan: {path}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - bergantung lingkungan
        raise GalatPDF(
            "pypdf belum terpasang. Jalankan: pip install -r copilot/requirements.txt"
        ) from exc

    try:
        pembaca = PdfReader(str(path))
    except Exception as exc:
        raise GalatPDF(f"Gagal membuka {path.name}: {exc}") from exc

    halaman = []
    for i, hal in enumerate(pembaca.pages, start=1):
        try:
            teks = hal.extract_text() or ""
        except Exception as exc:
            LOG.warning("halaman %s pada %s gagal dibaca: %s", i, path.name, exc)
            teks = ""
        halaman.append(HalamanPDF(nomor=i, teks=_rapikan(teks)))

    if not any(h.teks.strip() for h in halaman):
        raise GalatPDF(
            f"{path.name} tidak menghasilkan teks sama sekali. Kemungkinan hasil pindaian; "
            "dokumen seperti ini butuh OCR dan belum ditangani lapisan ini."
        )
    return halaman


def _rapikan(teks: str) -> str:
    """Rapikan spasi tanpa menghapus pergantian baris.

    Baris adalah satu-satunya sisa struktur tabel yang selamat dari ekstraksi
    teks PDF. Menggabungkannya jadi satu paragraf akan membuat model kehilangan
    batas antar baris mutasi.
    """
    teks = teks.replace("\xa0", " ")
    teks = re.sub(r"[ \t]+", " ", teks)
    teks = re.sub(r"\n{3,}", "\n\n", teks)
    return teks.strip()


def tebak_jenis(halaman: list[HalamanPDF]) -> tuple[JenisDokumen, dict[str, int]]:
    """Tebak jenis dari beberapa halaman pertama.

    Halaman pertama sudah cukup untuk akta dan lapkeu; rekening koran kadang
    baru menampilkan judul tabel di halaman kedua.
    """
    contoh = " ".join(h.teks for h in halaman[:3]).lower()
    skor = {
        jenis: sum(bobot for kata, bobot in penanda if kata in contoh)
        for jenis, penanda in PENANDA.items()
    }
    terbaik = max(skor, key=skor.get)
    if skor[terbaik] < AMBANG_PENANDA:
        return "tidak_dikenali", skor
    return terbaik, skor


def sumber_dari(path: str | Path, halaman: list[HalamanPDF]) -> Sumber:
    return Sumber(
        berkas=Path(path).name,
        halaman=[h.nomor for h in halaman if h.teks.strip()],
        jumlah_halaman=len(halaman),
    )


def kelompokkan(
    halaman: list[HalamanPDF], maks_karakter: int = 6000
) -> list[list[HalamanPDF]]:
    """Bagi halaman jadi kelompok yang muat di jendela konteks model kecil.

    Model 3B efektif hanya pada beberapa ribu karakter sekali baca meski jendela
    nominalnya jauh lebih besar. Batas ini soal ketelitian, bukan soal muat.
    """
    kelompok: list[list[HalamanPDF]] = []
    berjalan: list[HalamanPDF] = []
    panjang = 0

    for h in halaman:
        if not h.teks.strip():
            continue
        if berjalan and panjang + len(h.teks) > maks_karakter:
            kelompok.append(berjalan)
            berjalan, panjang = [], 0
        berjalan.append(h)
        panjang += len(h.teks)

    if berjalan:
        kelompok.append(berjalan)
    return kelompok


def gabung_teks(halaman: list[HalamanPDF]) -> str:
    """Teks berlabel nomor halaman - label ikut masuk prompt sebagai jejak sitasi."""
    return "\n\n".join(f"[halaman {h.nomor}]\n{h.teks}" for h in halaman if h.teks.strip())

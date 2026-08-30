"""Pemotongan dokumen kebijakan, dengan nomor pasal ikut menempel.

Catatan pada `docs/policies/README.md` menetapkan syaratnya: pertahankan
referensi nomor pasal per potongan agar jawaban copilot bisa disitasi balik ke
sumbernya. Karena itu batas potongan di sini mengikuti batas pasal, bukan
jumlah karakter.

Pasal yang lebih panjang dari batas dipecah menjadi beberapa potongan yang
semuanya membawa nomor pasal yang sama - sitasi tetap menunjuk unit yang benar
meski isinya terbagi.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from copilot.dokumen.pdf import baca_halaman
from copilot.konfigurasi import PENGATURAN

# "Pasal 12", "Pasal 12A", juga bentuk pada awal baris setelah nomor halaman.
POLA_PASAL = re.compile(r"^\s*Pasal\s+(\d+[A-Za-z]?)\s*$", re.MULTILINE)
POLA_BAB = re.compile(r"^\s*BAB\s+([IVXLC]+)\s*$", re.MULTILINE)


@dataclass
class Potongan:
    """Satu unit yang di-embed dan disitasi."""

    id: str
    teks: str
    berkas: str
    pasal: str | None = None
    bab: str | None = None
    halaman: list[int] = field(default_factory=list)

    @property
    def rujukan(self) -> str:
        """Label sitasi yang muncul di memo dan jawaban chat."""
        bagian = [self.berkas]
        if self.pasal:
            bagian.append(f"Pasal {self.pasal}")
        if self.halaman:
            bagian.append(f"hal. {self.halaman[0]}")
        return " · ".join(bagian)

    def untuk_prompt(self) -> str:
        return f"[{self.rujukan}]\n{self.teks}"


def potong_pdf(
    path: str | Path,
    *,
    ukuran: int | None = None,
    tumpang_tindih: int | None = None,
) -> list[Potongan]:
    """Baca satu PDF kebijakan menjadi daftar potongan bersitasi."""
    ukuran = ukuran or PENGATURAN.ukuran_potongan
    tumpang_tindih = tumpang_tindih or PENGATURAN.tumpang_tindih
    path = Path(path)

    halaman = baca_halaman(path)
    # Peta posisi karakter -> nomor halaman, supaya potongan tahu halaman
    # asalnya setelah teks digabung.
    penuh_bagian, batas_halaman = [], []
    jalan = 0
    for h in halaman:
        penuh_bagian.append(h.teks)
        jalan += len(h.teks) + 1
        batas_halaman.append((jalan, h.nomor))
    penuh = "\n".join(penuh_bagian)

    potongan: list[Potongan] = []
    for bagian_teks, awal, pasal, bab in _bagi_per_pasal(penuh):
        for sub, geser in _bagi_panjang(bagian_teks, ukuran, tumpang_tindih):
            if len(sub.strip()) < 40:  # sisa pemotongan, bukan isi
                continue
            potongan.append(
                Potongan(
                    id=f"{path.stem}#{len(potongan):04d}",
                    teks=sub.strip(),
                    berkas=path.name,
                    pasal=pasal,
                    bab=bab,
                    halaman=_halaman_pada(awal + geser, len(sub), batas_halaman),
                )
            )
    return potongan


def _bagi_per_pasal(teks: str) -> list[tuple[str, int, str | None, str | None]]:
    """Pisah teks pada tiap penanda "Pasal N", sambil melacak BAB berjalan."""
    penanda = sorted(
        [(m.start(), "pasal", m.group(1)) for m in POLA_PASAL.finditer(teks)]
        + [(m.start(), "bab", m.group(1)) for m in POLA_BAB.finditer(teks)]
    )
    if not penanda:
        return [(teks, 0, None, None)]

    bagian: list[tuple[str, int, str | None, str | None]] = []
    bab_berjalan: str | None = None

    # Bagian pembuka sebelum penanda pertama (judul, menimbang, mengingat).
    if penanda[0][0] > 0:
        bagian.append((teks[: penanda[0][0]], 0, None, None))

    for i, (posisi, jenis, nilai) in enumerate(penanda):
        akhir = penanda[i + 1][0] if i + 1 < len(penanda) else len(teks)
        if jenis == "bab":
            # BAB tanpa isi sendiri; potongannya menyusul pada pasal berikutnya.
            bab_berjalan = nilai
            continue
        bagian.append((teks[posisi:akhir], posisi, nilai, bab_berjalan))

    return bagian


def _bagi_panjang(teks: str, ukuran: int, tumpang_tindih: int) -> list[tuple[str, int]]:
    """Pecah teks yang melewati batas, mengembalikan (potongan, geser awal)."""
    if len(teks) <= ukuran:
        return [(teks, 0)]

    hasil, mulai = [], 0
    langkah = max(1, ukuran - tumpang_tindih)
    while mulai < len(teks):
        potong = teks[mulai : mulai + ukuran]
        # Rapikan ke batas baris terakhir supaya kalimat tidak terpenggal.
        if mulai + ukuran < len(teks):
            pisah = potong.rfind("\n")
            if pisah > ukuran // 2:
                potong = potong[:pisah]
        hasil.append((potong, mulai))
        mulai += max(langkah, len(potong) - tumpang_tindih)
    return hasil


def _halaman_pada(awal: int, panjang: int, batas: list[tuple[int, int]]) -> list[int]:
    """Nomor halaman yang beririsan dengan rentang karakter [awal, awal+panjang).

    `batas` berisi (posisi_akhir_kumulatif, nomor_halaman) terurut, sehingga
    halaman ke-i menempati rentang (posisi_akhir sebelumnya, posisi_akhir].
    """
    akhir = awal + panjang
    halaman, mulai_halaman = [], 0
    for posisi_akhir, nomor in batas:
        if mulai_halaman < akhir and posisi_akhir > awal:
            halaman.append(nomor)
        mulai_halaman = posisi_akhir
    return halaman or [batas[-1][1]] if batas else []

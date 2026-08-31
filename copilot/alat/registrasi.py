"""Pelaksana tool: jalankan satu panggilan dan catat jejaknya.

Setiap panggilan menghasilkan satu `JejakAlat`, dan seluruh jejak itulah yang
ditampilkan di antarmuka dan dilampirkan ke draft memo. Tanpa jejak, angka di
memo tidak bisa dibedakan dari angka yang dikarang model - dan itu justru yang
harus dicegah oleh pemisahan model/tool.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from copilot.alat.definisi import DEFINISI, PETA
from copilot.alat.keuangan import GalatMasukan

LOG = logging.getLogger(__name__)


@dataclass
class JejakAlat:
    nama: str
    argumen: dict[str, Any]
    hasil: dict[str, Any] | None = None
    galat: str | None = None
    durasi_ms: int = 0

    @property
    def berhasil(self) -> bool:
        return self.galat is None

    def untuk_model(self) -> str:
        """Isi pesan role=tool yang dikembalikan ke model."""
        if self.galat:
            return json.dumps({"galat": self.galat}, ensure_ascii=False)
        return json.dumps(self.hasil, ensure_ascii=False, default=str)

    def ringkas(self) -> str:
        argumen = ", ".join(f"{k}={v}" for k, v in self.argumen.items())
        if self.galat:
            return f"{self.nama}({argumen}) -> GAGAL: {self.galat}"
        # Tool pemeriksa mengembalikan `lolos` terpisah dari `rumus`, dan
        # `rumus`-nya berisi ambang - bukan putusan. Tanpa baris ini, sebuah
        # pemeriksaan yang gagal tampil dengan teks ambang yang sama persis
        # dengan yang lolos, dan pembaca jejak menyimpulkan sebaliknya.
        putusan = self.hasil.get("lolos")
        awalan = "" if putusan is None else ("LOLOS: " if putusan else "TIDAK LOLOS: ")
        return f"{self.nama}({argumen}) -> {awalan}{self.hasil.get('rumus', self.hasil)}"


@dataclass
class Rekaman:
    """Kumpulan jejak satu sesi agen."""

    jejak: list[JejakAlat] = field(default_factory=list)

    def tambah(self, j: JejakAlat) -> JejakAlat:
        self.jejak.append(j)
        return j

    @property
    def hasil_terakhir(self) -> dict[str, dict[str, Any]]:
        """Hasil terakhir per nama tool - bahan angka untuk draft memo."""
        keluaran: dict[str, dict[str, Any]] = {}
        for j in self.jejak:
            if j.berhasil:
                keluaran[j.nama] = j.hasil
        return keluaran

    def gagal(self) -> list[JejakAlat]:
        return [j for j in self.jejak if not j.berhasil]


def jalankan(nama: str, argumen: dict[str, Any]) -> JejakAlat:
    """Jalankan satu tool. Tidak pernah melempar - galat jadi bagian jejak.

    Model yang menerima pesan galat sebagai balasan tool biasanya memperbaiki
    argumennya pada putaran berikutnya. Melempar exception ke atas justru
    memutus kesempatan itu dan membatalkan seluruh analisis.
    """
    mulai = time.perf_counter()

    fungsi = PETA.get(nama)
    if fungsi is None:
        return JejakAlat(
            nama=nama,
            argumen=argumen,
            galat=f"Tool {nama!r} tidak ada. Yang tersedia: {', '.join(sorted(PETA))}.",
        )

    argumen_bersih = _bersihkan(argumen)
    try:
        hasil = fungsi(**argumen_bersih)
        galat = None
    except GalatMasukan as exc:
        hasil, galat = None, str(exc)
    except TypeError as exc:
        # Argumen wajib hilang atau nama field salah - keduanya bisa diperbaiki
        # model bila pesannya dikembalikan apa adanya.
        hasil, galat = None, f"Argumen tidak sesuai: {exc}"
    except Exception as exc:  # pragma: no cover - jaring pengaman
        LOG.exception("tool %s gagal tak terduga", nama)
        hasil, galat = None, f"Galat tak terduga: {exc}"

    return JejakAlat(
        nama=nama,
        argumen=argumen_bersih,
        hasil=hasil,
        galat=galat,
        durasi_ms=int((time.perf_counter() - mulai) * 1000),
    )


def _bersihkan(argumen: dict[str, Any]) -> dict[str, Any]:
    """Rapikan bentuk argumen yang lazim salah dari model kecil.

    Yang ditangani hanya kesalahan bentuk - string berisi angka, null eksplisit.
    Kesalahan nilai (satuan keliru, pos tertukar) sengaja dibiarkan lolos ke
    fungsi perhitungan supaya tertangkap validasinya dan terlihat di jejak.
    """
    bersih: dict[str, Any] = {}
    for kunci, nilai in (argumen or {}).items():
        if nilai is None:
            continue
        if isinstance(nilai, str):
            teks = nilai.strip().replace("_", "")
            if teks.lower() in {"", "null", "none", "n/a"}:
                continue
            angka = _ke_angka(teks)
            bersih[kunci] = nilai if angka is None else angka
            continue
        bersih[kunci] = nilai
    return bersih


def _ke_angka(teks: str) -> float | None:
    """Urai string angka, atau None bila memang bukan angka.

    Titik itu ambigu: "1.500.000.000" memakainya sebagai pemisah ribuan,
    "2.50" sebagai koma desimal. Aturannya mengikuti jumlah tanda, bukan
    tebakan lokal - satu titik dibaca desimal (bentuk yang dipakai JSON), lebih
    dari satu titik dibaca pemisah ribuan.
    """
    calon = teks.replace("Rp", "").replace(" ", "")
    if not calon or not any(c.isdigit() for c in calon):
        return None

    if "," in calon:
        # Bentuk Indonesia: titik ribuan, koma desimal.
        calon = calon.replace(".", "").replace(",", ".")
    elif calon.count(".") > 1:
        calon = calon.replace(".", "")

    try:
        return float(calon)
    except ValueError:
        return None


def daftar_tool() -> list[dict[str, Any]]:
    """Definisi tool untuk dikirim ke model."""
    return DEFINISI


def ringkas_katalog() -> list[dict[str, str]]:
    """Katalog ringkas untuk ditampilkan di antarmuka."""
    return [
        {
            "nama": d["function"]["name"],
            "deskripsi": d["function"]["description"],
            "wajib": ", ".join(d["function"]["parameters"]["required"]),
        }
        for d in DEFINISI
    ]

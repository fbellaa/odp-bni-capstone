from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Halaman:
    nomor: int
    teks: str


def baca_halaman(path: str | Path):
    path = Path(path)
    reader = PdfReader(str(path))

    hasil = []
    for nomor, page in enumerate(
        reader.pages,
        start=1,
    ):
        hasil.append(
            Halaman(
                nomor=nomor,
                teks=(page.extract_text() or "").strip(),
            )
        )
    return hasil

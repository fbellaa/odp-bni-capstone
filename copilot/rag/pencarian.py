"""Pencarian kebijakan dengan sitasi per pasal.

Keluaran `kutipan()` berkunci `pasal`, `isi`, `skor`, `versi`, `halaman`.
Bentuk itu dipakai langsung oleh `app/ui/lib/kebijakan.py` untuk mengisi
rujukan pada gerbang kepatuhan dan bagian rujukan draft credit memo -
menggantikan kutipan tiruan yang dulu ditulis tangan di lapisan antarmuka.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from copilot.konfigurasi import PENGATURAN
from copilot.llm.klien import KlienOllama, klien
from copilot.rag.indeks import (
    IndexKebijakan,
    index_tersedia,
    muat_index,
    pasal_dalam_kueri,
)

LOG = logging.getLogger(__name__)

PERINTAH_JAWAB = """\
Kamu asisten kepatuhan kredit komersial. Jawab HANYA berdasarkan kutipan
peraturan yang diberikan.

- Sebut rujukan di dalam kalimat, contoh: "(POJK 40/2019 Pasal 12)".
- Bila kutipan tidak memuat jawabannya, katakan terus terang bahwa aturannya
  tidak ditemukan pada korpus. Jangan mengarang pasal atau nomor.
- Jawab ringkas, dalam bahasa Indonesia.
"""


@lru_cache(maxsize=1)
def _index() -> IndexKebijakan:
    """Index dimuat sekali per proses; Streamlit menjalankan ulang skrip terus."""
    return muat_index()


def cari(
    kueri: str,
    *,
    top_k: int | None = None,
    kl: KlienOllama | None = None,
    bobot_leksikal: float | None = None,
) -> list[dict[str, Any]]:
    """Potongan kebijakan paling relevan, terurut menurun.

    Kueri ikut dikirim apa adanya ke index karena peringkatnya hibrida: selain
    kosinus, ada kecocokan istilah dan jalur khusus untuk kueri yang menyebut
    nomor pasal. Rinciannya di `IndexKebijakan.cari`.
    """
    kl = kl or klien()
    top_k = top_k or PENGATURAN.top_k
    if bobot_leksikal is None:
        bobot_leksikal = PENGATURAN.bobot_leksikal
    vektor = np.asarray(kl.embed([kueri])[0], dtype=np.float32)
    norma = np.linalg.norm(vektor)
    if norma:
        vektor = vektor / norma
    hasil = _index().cari(
        vektor, top_k, kueri=kueri, bobot_leksikal=bobot_leksikal
    )
    for h in hasil:
        h["halaman"] = json.loads(h.get("halaman") or "[]")
    return hasil


def kutipan(kueri: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
    """Hasil pencarian dalam bentuk yang dipakai tampilan dan penyusun memo."""
    if not index_tersedia():
        return []
    try:
        potongan = cari(kueri, top_k=top_k)
    except Exception as exc:  # index rusak atau Ollama mati
        LOG.warning("pencarian kebijakan gagal: %s", exc)
        return []

    keluaran = []
    for p in potongan:
        isi = p["teks"].replace("\n", " ").strip()
        keluaran.append(
            {
                "pasal": p["rujukan"],
                "isi": isi if len(isi) <= 600 else isi[:600].rsplit(" ", 1)[0] + "…",
                "skor": round(p["skor"], 3),
                "versi": p["berkas"],
                "halaman": p["halaman"],
            }
        )
    return keluaran


def jawab(
    pertanyaan: str, *, top_k: int | None = None, kl: KlienOllama | None = None
) -> dict[str, Any]:
    """Jawaban chat atas pertanyaan kebijakan, beserta sitasi yang dipakai."""
    kl = kl or klien()
    potongan = cari(pertanyaan, top_k=top_k, kl=kl)
    if not potongan:
        return {
            "jawaban": "Korpus kebijakan kosong atau index belum dibangun.",
            "sitasi": [],
        }

    konteks = "\n\n---\n\n".join(
        f"[{p['rujukan']}]\n{p['teks']}" for p in potongan
    )

    # Kueri boleh menyebut pasal yang tidak ada di korpus - POJK 40/2019 bukan
    # satu-satunya peraturan yang dipakai analis. Ketidakhadirannya disebut
    # terang-terangan supaya model tidak menambalnya dengan pasal termirip.
    ada = {str(p.get("pasal")).upper() for p in potongan}
    hilang = [n for n in pasal_dalam_kueri(pertanyaan) if n not in ada]
    catatan = (
        f"\n\nCatatan: Pasal {', '.join(hilang)} tidak ada pada korpus. "
        "Katakan itu apa adanya dan jangan menggantinya dengan pasal lain."
        if hilang
        else ""
    )

    balasan = kl.chat(
        [
            {"role": "system", "content": PERINTAH_JAWAB},
            {
                "role": "user",
                "content": (
                    f"Kutipan peraturan:\n\n{konteks}\n\nPertanyaan: {pertanyaan}{catatan}"
                ),
            },
        ],
        peran="chat",
    )
    return {
        "jawaban": balasan.get("content", "").strip(),
        # `cari()` sudah mengurai kolom halaman menjadi list.
        "sitasi": [
            {"rujukan": p["rujukan"], "skor": round(p["skor"], 3), "halaman": p["halaman"]}
            for p in potongan
        ],
    }

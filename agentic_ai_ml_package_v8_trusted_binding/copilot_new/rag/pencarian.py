"""Pencarian kebijakan dengan sitasi per pasal.

Bentuk keluaran `kutipan()` sengaja dibuat sama dengan
`app/ui/lib/dummy_data.kutipan_kebijakan()` - kunci `pasal`, `isi`, `skor`,
`versi`. Halaman Streamlit yang sudah ada bisa beralih dari kutipan tiruan ke
hasil RAG asli tanpa mengubah kode tampilan.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from copilot.konfigurasi import PENGATURAN
from copilot.llm.klien import KlienOllama, klien
from copilot.rag.indeks import IndexKebijakan, index_tersedia, muat_index

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
    kueri: str, *, top_k: int | None = None, kl: KlienOllama | None = None
) -> list[dict[str, Any]]:
    """Potongan kebijakan paling relevan, terurut menurun."""
    kl = kl or klien()
    top_k = top_k or PENGATURAN.top_k
    vektor = np.asarray(kl.embed([kueri])[0], dtype=np.float32)
    norma = np.linalg.norm(vektor)
    if norma:
        vektor = vektor / norma
    hasil = _index().cari(vektor, top_k)
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
    balasan = kl.chat(
        [
            {"role": "system", "content": PERINTAH_JAWAB},
            {
                "role": "user",
                "content": f"Kutipan peraturan:\n\n{konteks}\n\nPertanyaan: {pertanyaan}",
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

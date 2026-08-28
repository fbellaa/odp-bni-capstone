"""Pembangunan dan pemuatan index vektor kebijakan.

Index disimpan sebagai satu `.npz` (matriks vektor) plus satu `.parquet`
(metadata potongan) di `data/index/` - bukan di `data/gold/`. Gold adalah
warehouse keluaran pipeline dan tunduk pada gerbang kualitasnya; vektor tidak
punya hubungan apa pun dengan uji-uji itu dan tidak boleh ikut dinilai di sana.

Tidak ada FAISS di sini. Korpus kebijakan berukuran ratusan potongan, dan
perkalian matriks NumPy atas 500 x 768 selesai dalam hitungan milidetik. Satu
dependensi berbobot yang gagal dipasang di Kaggle berbiaya jauh lebih mahal
daripada pencarian brute force yang selalu jalan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from copilot.konfigurasi import INDEX_DIR, PENGATURAN, POLICY_DIR
from copilot.llm.klien import KlienOllama, klien
from copilot.rag.potong import Potongan, potong_pdf

LOG = logging.getLogger(__name__)

BERKAS_VEKTOR = INDEX_DIR / "kebijakan_vektor.npz"
BERKAS_META = INDEX_DIR / "kebijakan_potongan.parquet"
BERKAS_INFO = INDEX_DIR / "kebijakan_info.json"

# Ukuran kelompok saat memanggil /api/embed. Terlalu besar membuat Ollama di
# CPU Kaggle kehabisan memori di tengah pembangunan index.
UKURAN_BATCH = 16


@dataclass
class IndexKebijakan:
    vektor: np.ndarray  # (n, dim), sudah dinormalisasi L2
    potongan: pd.DataFrame
    model_embedding: str

    def __len__(self) -> int:
        return len(self.potongan)

    def cari(self, vektor_kueri: np.ndarray, top_k: int) -> list[dict]:
        """Kemiripan kosinus. Vektor sudah dinormalisasi, jadi cukup dot product."""
        skor = self.vektor @ vektor_kueri
        urutan = np.argsort(-skor)[:top_k]
        hasil = []
        for i in urutan:
            baris = self.potongan.iloc[int(i)].to_dict()
            baris["skor"] = float(skor[int(i)])
            hasil.append(baris)
        return hasil


def bangun_index(
    *,
    direktori: Path | None = None,
    kl: KlienOllama | None = None,
    paksa: bool = False,
) -> IndexKebijakan:
    """Potong seluruh PDF kebijakan, embed, lalu simpan.

    Hanya perlu dijalankan sekali per korpus. Di Kaggle, panggil sekali di sel
    persiapan - bukan di dalam antarmuka, karena Streamlit menjalankan ulang
    skripnya pada tiap interaksi.
    """
    direktori = direktori or POLICY_DIR
    kl = kl or klien()

    if not paksa and BERKAS_VEKTOR.exists():
        LOG.info("index sudah ada, memuat dari cakram")
        return muat_index()

    berkas_pdf = sorted(Path(direktori).glob("*.pdf"))
    if not berkas_pdf:
        raise FileNotFoundError(
            f"Tidak ada PDF kebijakan di {direktori}. "
            "Lihat docs/policies/README.md untuk daftar dokumen sumber."
        )

    potongan: list[Potongan] = []
    for path in berkas_pdf:
        bagian = potong_pdf(path)
        LOG.info("%s -> %s potongan", path.name, len(bagian))
        potongan.extend(bagian)

    if not potongan:
        raise RuntimeError("Seluruh PDF kebijakan menghasilkan nol potongan.")

    model_embedding = PENGATURAN.model_untuk("embedding")
    kl.pastikan_model(model_embedding)

    vektor = []
    for i in range(0, len(potongan), UKURAN_BATCH):
        kelompok = potongan[i : i + UKURAN_BATCH]
        LOG.info("embedding %s-%s dari %s", i, i + len(kelompok), len(potongan))
        vektor.extend(kl.embed([p.teks for p in kelompok]))

    matriks = _normalisasi(np.asarray(vektor, dtype=np.float32))
    meta = pd.DataFrame(
        [
            {
                "id": p.id,
                "teks": p.teks,
                "berkas": p.berkas,
                "pasal": p.pasal,
                "bab": p.bab,
                "halaman": json.dumps(p.halaman),
                "rujukan": p.rujukan,
            }
            for p in potongan
        ]
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BERKAS_VEKTOR, vektor=matriks)
    meta.to_parquet(BERKAS_META, index=False)
    BERKAS_INFO.write_text(
        json.dumps(
            {
                "model_embedding": model_embedding,
                "dimensi": int(matriks.shape[1]),
                "jumlah_potongan": len(potongan),
                "berkas": [p.name for p in berkas_pdf],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info("index tersimpan: %s potongan, dimensi %s", len(potongan), matriks.shape[1])
    return IndexKebijakan(matriks, meta, model_embedding)


def muat_index() -> IndexKebijakan:
    if not BERKAS_VEKTOR.exists():
        raise FileNotFoundError(
            "Index kebijakan belum dibangun. Jalankan:\n"
            "    python -m copilot.rag.indeks"
        )
    info = json.loads(BERKAS_INFO.read_text(encoding="utf-8"))
    model_tersimpan = info["model_embedding"]
    model_sekarang = PENGATURAN.model_untuk("embedding")
    if model_tersimpan != model_sekarang:
        # Vektor dari dua model berbeda tidak sebanding. Membiarkannya lewat
        # akan menghasilkan sitasi yang terlihat wajar tapi salah pasal.
        raise RuntimeError(
            f"Index dibangun dengan {model_tersimpan!r}, sedangkan konfigurasi "
            f"sekarang memakai {model_sekarang!r}. Bangun ulang:\n"
            "    python -m copilot.rag.indeks --paksa"
        )
    return IndexKebijakan(
        vektor=np.load(BERKAS_VEKTOR)["vektor"],
        potongan=pd.read_parquet(BERKAS_META),
        model_embedding=model_tersimpan,
    )


def index_tersedia() -> bool:
    return BERKAS_VEKTOR.exists() and BERKAS_META.exists() and BERKAS_INFO.exists()


def _normalisasi(m: np.ndarray) -> np.ndarray:
    norma = np.linalg.norm(m, axis=1, keepdims=True)
    norma[norma == 0] = 1.0
    return m / norma


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Bangun index RAG kebijakan.")
    ap.add_argument("--paksa", action="store_true", help="bangun ulang meski index sudah ada")
    argumen = ap.parse_args()

    idx = bangun_index(paksa=argumen.paksa)
    print(f"Index siap: {len(idx)} potongan, model {idx.model_embedding}")

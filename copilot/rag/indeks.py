"""Pembangunan dan pemuatan index vektor kebijakan.

Index disimpan sebagai satu `.npz` (matriks vektor) plus satu `.parquet`
(metadata potongan) di `data/index/` - bukan di `data/gold/`. Gold adalah
warehouse keluaran pipeline dan tunduk pada gerbang kualitasnya; vektor tidak
punya hubungan apa pun dengan uji-uji itu dan tidak boleh ikut dinilai di sana.

Tidak ada FAISS di sini. Korpus kebijakan berukuran ratusan potongan, dan
perkalian matriks NumPy atas 500 x 768 selesai dalam hitungan milidetik. Satu
dependensi berbobot yang gagal dipasang di Kaggle berbiaya jauh lebih mahal
daripada pencarian brute force yang selalu jalan.

Peringkatnya hibrida: kosinus atas vektor, ditambah kecocokan kata berbobot
IDF, ditambah satu jalur khusus untuk kueri yang menyebut nomor pasal. Alasan
tiap bagian ada di `skor_leksikal` dan `cari`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
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

# "Pasal 12", "pasal 12A", "POJK 40/2019 Pasal 5 ayat (2)".
POLA_RUJUKAN_PASAL = re.compile(r"pasal\s+(\d+[A-Za-z]?)", re.IGNORECASE)

POLA_TOKEN = re.compile(r"[a-z0-9]+")

# Saturasi tf ala BM25. Kemunculan pertama sebuah istilah bernilai penuh;
# kemunculan berikutnya menambah makin sedikit.
K_SATURASI = 1.2

# Kata yang muncul di hampir setiap pasal. Dibuang supaya kueri panjang tidak
# tenggelam oleh kata sambung; istilah teknisnya yang harus menentukan.
STOPWORD = frozenset(
    """
    yang dan atau dengan untuk pada dalam dari dimaksud sebagaimana ayat huruf
    bagi serta oleh ini itu tersebut adalah dapat wajib harus tidak akan sebagai
    telah bank umum jika maka agar antara lain paling atas juga bila
    """.split()
)


def tokenisasi(teks: str) -> list[str]:
    """Token leksikal: huruf/angka, huruf kecil, tanpa stopword dan token satu huruf."""
    return [
        t
        for t in POLA_TOKEN.findall(teks.lower())
        if len(t) > 2 and t not in STOPWORD
    ]


def pasal_dalam_kueri(kueri: str) -> list[str]:
    """Nomor pasal yang disebut eksplisit di kueri, urut kemunculan, tanpa duplikat."""
    urut: list[str] = []
    for nomor in POLA_RUJUKAN_PASAL.findall(kueri):
        nomor = nomor.upper()
        if nomor not in urut:
            urut.append(nomor)
    return urut


def _bangun_leksikal(teks: list[str]) -> dict:
    """Hitungan token per potongan + IDF korpus. Sekali per proses."""
    hitungan = [Counter(tokenisasi(t)) for t in teks]
    df: Counter[str] = Counter()
    for tf in hitungan:
        df.update(tf.keys())
    n = max(len(hitungan), 1)
    idf = {t: math.log(1.0 + n / (1.0 + jumlah)) for t, jumlah in df.items()}
    return {"hitungan": hitungan, "idf": idf}


@dataclass
class IndexKebijakan:
    vektor: np.ndarray  # (n, dim), sudah dinormalisasi L2
    potongan: pd.DataFrame
    model_embedding: str
    # Struktur leksikal dibangun saat pencarian pertama, bukan saat memuat:
    # pemakai yang hanya memeriksa ketersediaan index tidak perlu membayarnya.
    _leksikal: dict | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.potongan)

    # ------------------------------------------------------------------ skor
    def skor_kosinus(self, vektor_kueri: np.ndarray) -> np.ndarray:
        """Vektor sudah dinormalisasi L2, jadi kosinus = dot product."""
        return self.vektor @ vektor_kueri

    def skor_leksikal(self, kueri: str) -> np.ndarray:
        """Kecocokan kata berbobot IDF, dinormalisasi ke [0,1].

        Kosinus atas model embedding kecil menilai kemiripan makna, dan itu
        justru merugikan pada korpus peraturan: "hapus buku", "CKPN", dan
        "agunan yang diambil alih" adalah istilah berdefinisi, bukan parafrase.
        Skor ini mengembalikan bobot pada kecocokan istilah persis, dengan IDF
        supaya kata yang ada di setiap pasal tidak ikut menentukan.

        Bentuknya: porsi bobot IDF kueri yang benar-benar muncul di potongan,
        dengan saturasi tf ala BM25 (K=1,2) supaya pengulangan istilah memberi
        tambahan yang mengecil, lalu dipangkas ke 1,0. Potongan yang memuat
        seluruh istilah kueri karena itu bisa mencapai 1,0 - setara skala
        kosinus, syarat supaya pembobotan di `cari` bermakna.
        """
        if self._leksikal is None:
            self._leksikal = _bangun_leksikal(self.potongan["teks"].tolist())
        idf = self._leksikal["idf"]
        hitungan = self._leksikal["hitungan"]

        token = {t for t in tokenisasi(kueri) if t in idf}
        skor = np.zeros(len(self.potongan), dtype=np.float32)
        if not token:
            return skor

        pembagi = sum(idf[t] for t in token) or 1.0
        for i, tf in enumerate(hitungan):
            nilai = sum(
                idf[t] * (tf[t] * (K_SATURASI + 1.0)) / (tf[t] + K_SATURASI)
                for t in token
                if t in tf
            )
            skor[i] = min(nilai / pembagi, 1.0)
        return skor

    def indeks_pasal(self, nomor: str) -> list[int]:
        """Posisi baris seluruh potongan milik satu pasal, urut seperti di dokumen."""
        kolom = self.potongan["pasal"].astype("object")
        cocok = np.array(
            [str(nilai).upper() == nomor.upper() for nilai in kolom], dtype=bool
        )
        return [int(i) for i in np.flatnonzero(cocok)]

    # --------------------------------------------------------------- peringkat
    def cari(
        self,
        vektor_kueri: np.ndarray,
        top_k: int,
        *,
        kueri: str | None = None,
        bobot_leksikal: float = 0.0,
    ) -> list[dict]:
        """Peringkat hibrida: kosinus + leksikal, dengan jalur khusus nomor pasal.

        Kueri yang menyebut pasal secara eksplisit ("apa isi Pasal 12?") tidak
        boleh bergantung pada kemiripan vektor sama sekali: nomor pasal adalah
        kunci pencarian yang tepat, dan metadatanya sudah dibawa tiap potongan
        sejak pemotongan. Potongan pasal itu karena itu diangkat ke atas apa
        adanya - kalau nomornya tidak ada di korpus, peringkat kembali ke
        skor biasa dan model bahasa yang menyatakan aturannya tidak ditemukan.
        """
        skor = self.skor_kosinus(vektor_kueri).astype(np.float32)
        if kueri and bobot_leksikal > 0:
            skor = (1.0 - bobot_leksikal) * skor + bobot_leksikal * self.skor_leksikal(kueri)

        urutan = [int(i) for i in np.argsort(-skor)]
        diminta: list[int] = []
        if kueri:
            for nomor in pasal_dalam_kueri(kueri):
                diminta.extend(i for i in self.indeks_pasal(nomor) if i not in diminta)
        if diminta:
            # Urutan dokumen dipertahankan di sini, bukan urutan skor: yang
            # dibaca adalah satu pasal utuh, dan ayat (1) harus datang lebih
            # dulu dari ayat (3).
            urutan = diminta + [i for i in urutan if i not in set(diminta)]

        # Pasal yang diminta eksplisit dibawa utuh: memotongnya di tengah
        # membuat jawaban mengutip separuh aturan tanpa memberi tahu.
        batas = max(top_k, len(diminta))

        hasil = []
        for i in urutan[:batas]:
            baris = self.potongan.iloc[i].to_dict()
            baris["skor"] = float(skor[i])
            baris["rujukan_eksplisit"] = i in diminta
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

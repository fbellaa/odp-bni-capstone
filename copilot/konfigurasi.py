"""Konfigurasi terpusat lapisan copilot (LLM lokal, RAG, agen).

Mengikuti pola `pipelines/config.py`: seluruh path dan parameter yang boleh
digeser lewat environment variable dikumpulkan di satu berkas, supaya klien
LLM, pembangun index, dan agen memakai angka yang sama persis.

Anggaran memori adalah batasan utama di sini. Target jalannya adalah Kaggle /
Colab free tier (T4 16 GB VRAM, RAM host 13-16 GB), jadi model dipilih pada
kuantisasi 4-bit dan dipisah per peran supaya yang besar tidak ikut termuat
saat tidak dipakai.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS_DIR = PROJECT_ROOT / "docs"
POLICY_DIR = DOCS_DIR / "policies"

# Artefak index RAG (potongan + vektor) TIDAK masuk data/gold. Gold adalah
# keluaran pipeline dan tunduk pada gerbang kualitas; vektor tidak punya
# hubungan apa pun dengan uji-uji itu.
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Unggahan PDF pengajuan dari antarmuka. Sementara dan tidak di-commit.
UNGGAHAN_DIR = PROJECT_ROOT / "data" / "unggahan"


# --------------------------------------------------------------------- model
# Dua peran yang sengaja dipisah:
#
#   ekstraksi/chat  bahasa Indonesia, membaca dokumen, menyusun narasi memo
#   agen            tool calling terhadap perhitungan - butuh model yang
#                   patuh pada skema fungsi, bukan yang fasih berbahasa
#
# Pemisahan ini juga yang membuat profil "hemat" mungkin: satu model kecil
# memegang dua peran sekaligus tanpa mengubah kode pemanggil.
PROFIL_MODEL = {
    # Bawaan sejak lapisan agentic AI (ml/agentic_ai) masuk repo. Lapisan itu
    # memakai qwen2.5:7b-instruct untuk agent, narrator, dan extractor-nya;
    # menyamakan peran di sini berarti Ollama cukup memuat satu bobot Qwen,
    # bukan dua generasi sekaligus. Yang tersisa khas copilot tinggal embedding
    # RAG kebijakan dan sintesis jawabannya.
    "terpadu": {
        "ekstraksi": "qwen2.5:7b-instruct",
        "chat": "qwen2.5:7b-instruct",
        "agen": "qwen2.5:7b-instruct",
        "embedding": "nomic-embed-text",
    },
    # ~2,5 GB total. Aman di CPU-only sekalipun. Jalur mundur bila 7B tidak
    # muat - mis. sesi tanpa GPU sama sekali.
    "hemat": {
        "ekstraksi": "qwen2.5:3b-instruct",
        "chat": "qwen2.5:3b-instruct",
        "agen": "qwen2.5:3b-instruct",
        "embedding": "nomic-embed-text",
    },
    # ~8 GB total, muat di satu T4. Peran bahasa Indonesia pindah ke SahabatAI;
    # tool calling naik ke Qwen 7B yang jauh lebih jarang salah bentuk argumen.
    "seimbang": {
        "ekstraksi": "",  # wajib diisi COPILOT_MODEL_SAHABAT - lihat catatan
        "chat": "",
        "agen": "qwen2.5:7b-instruct",
        "embedding": "nomic-embed-text",
    },
}

PROFIL_BAWAAN = "terpadu"

# Perkiraan jejak memori per model pada kuantisasi bawaan Ollama (Q4_K_M),
# dipakai `copilot.llm.klien.ringkas_anggaran()` untuk memperingatkan sebelum
# sesi Kaggle kehabisan VRAM di tengah demo.
PERKIRAAN_GB = {
    "qwen2.5:3b-instruct": 2.0,
    "qwen2.5:7b-instruct": 4.7,
    "nomic-embed-text": 0.3,
    # Bukan peran copilot - VLM dipanggil ml/agentic_ai untuk transkripsi
    # halaman hasil scan. Dicatat di sini supaya angkanya benar bila seseorang
    # menimpa salah satu peran ke model ini, dan supaya jejaknya tidak jatuh ke
    # PERKIRAAN_GB_TIDAK_DIKENAL yang menebak 6 GB.
    "qwen3-vl:4b-instruct": 3.0,
}
PERKIRAAN_GB_TIDAK_DIKENAL = 6.0

# Batas VRAM yang dianggap tersedia. T4 di Kaggle/Colab free tier punya 15-16 GB,
# tetapi sebagian sudah dipakai runtime notebook.
ANGGARAN_GB = float(os.getenv("COPILOT_ANGGARAN_GB", "13"))


def _model(peran: str, profil: str) -> str:
    """Nilai environment menang atas profil; profil menang atas kekosongan."""
    langsung = os.getenv(f"COPILOT_MODEL_{peran.upper()}")
    if langsung:
        return langsung
    if peran in ("ekstraksi", "chat"):
        sahabat = os.getenv("COPILOT_MODEL_SAHABAT")
        if sahabat:
            return sahabat
    return PROFIL_MODEL[profil][peran]


def _url_ollama(nilai: str | None) -> str:
    """Ubah isi OLLAMA_HOST menjadi URL yang benar-benar bisa dituju.

    `OLLAMA_HOST` menanggung dua arti yang berbeda. Bagi server Ollama ia
    alamat BIND; bagi klien mana pun - termasuk kode ini - ia alamat TUJUAN.
    Siapa pun yang menyetel `OLLAMA_HOST=0.0.0.0` supaya server mau menerima
    koneksi dari container lalu mendapati klien di mesin yang sama berhenti
    bekerja, karena 0.0.0.0 berarti "dengarkan semua antarmuka" dan bukan
    alamat yang bisa dihubungi. Gejalanya menyesatkan: `hidup()` mengembalikan
    False, dan antarmuka menyarankan menjalankan `ollama serve` justru ketika
    server sudah jalan.

    Bentuk singkat seperti "localhost:11434" atau "127.0.0.1" juga diterima -
    keduanya lazim ditulis orang, dan menolaknya tidak ada gunanya.
    """
    teks = (nilai or "").strip().rstrip("/")
    if not teks:
        return "http://127.0.0.1:11434"

    skema, _, sisa = teks.partition("://")
    if not sisa:
        skema, sisa = "http", teks

    # Alamat bind "semua antarmuka" tidak bisa dituju; yang dimaksud penyetelnya
    # selalu mesin ini sendiri. Diperiksa atas `sisa` yang utuh, sebelum port
    # dipisah: "::" yang dibelah pada titik dua menyisakan inang kosong.
    if sisa in ("0.0.0.0", "::", "[::]", "*"):
        return f"{skema}://127.0.0.1:11434"

    if sisa.startswith("["):                      # IPv6 berkurung: [::1]:11434
        inang, _, ekor = sisa.partition("]")
        inang += "]"
        porta = ekor.lstrip(":")
    else:
        inang, _, porta = sisa.partition(":")
    if inang in ("0.0.0.0", "[::]", ""):
        inang = "127.0.0.1"
    return f"{skema}://{inang}:{porta or '11434'}"


@dataclass(frozen=True)
class Pengaturan:
    """Parameter yang boleh digeser lewat environment variable."""

    profil: str = os.getenv("COPILOT_PROFIL", PROFIL_BAWAAN)
    host_ollama: str = _url_ollama(os.getenv("OLLAMA_HOST"))

    # Detik. Ekstraksi satu lapkeu 20 halaman di CPU bisa lewat dari semenit.
    timeout: int = int(os.getenv("COPILOT_TIMEOUT", "300"))

    # Suhu 0 untuk ekstraksi dan tool calling: dua-duanya harus bisa diulang.
    # Narasi memo diberi sedikit ruang supaya tidak kaku.
    suhu_ekstraksi: float = 0.0
    suhu_agen: float = 0.0
    suhu_chat: float = 0.3

    # RAG. Potongan sengaja besar (satu pasal utuh kalau muat) supaya sitasi
    # menunjuk ke unit yang benar-benar dibaca analis, bukan penggalan kalimat.
    ukuran_potongan: int = int(os.getenv("COPILOT_UKURAN_POTONGAN", "1200"))
    tumpang_tindih: int = int(os.getenv("COPILOT_TUMPANG_TINDIH", "150"))
    top_k: int = int(os.getenv("COPILOT_TOP_K", "5"))

    # Bobot skor leksikal pada peringkat akhir (sisanya kosinus). Embedding
    # kecil seperti nomic-embed-text menempatkan parafrase umum di atas istilah
    # persis peraturan ("hapus buku", "CKPN", "agunan yang diambil alih"), dan
    # analis kredit mengetik istilah persis itu. Nol mengembalikan perilaku
    # dense murni.
    bobot_leksikal: float = float(os.getenv("COPILOT_BOBOT_LEKSIKAL", "0.35"))

    # Batas putaran tool calling. Agen yang tidak berhenti akan menghabiskan
    # sesi notebook, bukan sekadar melambat.
    maks_putaran_agen: int = int(os.getenv("COPILOT_MAKS_PUTARAN", "8"))

    model: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.profil not in PROFIL_MODEL:
            raise ValueError(
                f"COPILOT_PROFIL={self.profil!r} tidak dikenal. "
                f"Pilihan: {', '.join(PROFIL_MODEL)}"
            )
        object.__setattr__(
            self,
            "model",
            {peran: _model(peran, self.profil) for peran in PROFIL_MODEL[self.profil]},
        )

    def model_untuk(self, peran: str) -> str:
        nama = self.model.get(peran, "")
        if not nama:
            raise RuntimeError(
                f"Model untuk peran {peran!r} pada profil {self.profil!r} belum disetel.\n"
                "Profil 'seimbang' memakai SahabatAI untuk peran bahasa Indonesia, dan tag\n"
                "Ollama-nya tidak dibawakan repo ini karena bergantung pada berkas GGUF\n"
                "yang Anda pilih sendiri. Setel salah satu:\n"
                "    export COPILOT_MODEL_SAHABAT=hf.co/<repo-gguf-sahabatai>:Q4_K_M\n"
                "    export COPILOT_PROFIL=hemat      # jalan tanpa SahabatAI"
            )
        return nama

    def anggaran_terpakai(self) -> tuple[float, dict[str, float]]:
        """Perkiraan total GB bila seluruh peran termuat bersamaan."""
        per_model = {
            nama: PERKIRAAN_GB.get(nama, PERKIRAAN_GB_TIDAK_DIKENAL)
            for nama in dict.fromkeys(self.model.values())
            if nama
        }
        return sum(per_model.values()), per_model


PENGATURAN = Pengaturan()

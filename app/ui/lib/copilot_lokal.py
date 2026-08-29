"""Jembatan antarmuka ke paket `copilot`.

Halaman Streamlit tidak mengimpor `copilot.*` langsung. Seluruh sentuhan ke
lapisan itu lewat sini, dengan dua alasan:

1. `app/ui` dijalankan dengan cwd `app/ui`, sehingga akar proyek perlu
   dimasukkan ke sys.path lebih dulu. Menaruh urusan itu di satu berkas lebih
   baik daripada mengulangnya di tiap halaman.
2. Lapisan copilot boleh saja tidak tersedia - sesi demo tanpa Ollama, atau
   dependensi yang belum terpasang. Halaman harus tetap terbuka dan menjelaskan
   apa yang kurang, bukan menampilkan traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parents[3]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))

TERSEDIA = True
GALAT_IMPOR: str | None = None

try:
    from copilot import memo as memo_copilot
    from copilot.agen.perhitungan import AgenPerhitungan, HasilAgen
    from copilot.alat.registrasi import JejakAlat, ringkas_katalog
    from copilot.dokumen import ekstraksi, jembatan
    from copilot.dokumen.skema import BerkasPengajuan
    from copilot.konfigurasi import PENGATURAN, UNGGAHAN_DIR
    from copilot.llm.klien import klien, ringkas_anggaran
    from copilot.rag import indeks as rag_indeks
    from copilot.rag import pencarian as rag_cari
except Exception as exc:  # dependensi belum lengkap
    TERSEDIA = False
    GALAT_IMPOR = f"{type(exc).__name__}: {exc}"

# Sebagian nama di atas tidak dipakai berkas ini, melainkan diteruskan ke
# halaman sebagai `ck.<nama>` - itulah gunanya modul ini sebagai satu-satunya
# pintu ke lapisan copilot.
__all__ = [
    "AgenPerhitungan",
    "BerkasPengajuan",
    "GALAT_IMPOR",
    "HasilAgen",
    "JejakAlat",
    "TERSEDIA",
    "baca_dokumen",
    "cari_kebijakan",
    "jawab_kebijakan",
    "memo_copilot",
    "ringkas_katalog",
    "simpan_unggahan",
    "status_lingkungan",
    "telusuri_afiliasi",
]


def status_lingkungan() -> dict:
    """Ringkasan kesiapan untuk ditampilkan di sidebar sebelum demo dimulai."""
    if not TERSEDIA:
        return {"siap": False, "alasan": GALAT_IMPOR, "ollama": False, "index": False}

    kl = klien()
    hidup = kl.hidup()
    terpasang = kl.daftar_model() if hidup else []
    anggaran = ringkas_anggaran()
    diminta = [m for m in anggaran["model"].values() if m]
    kurang = [
        m for m in dict.fromkeys(diminta)
        if m not in terpasang and f"{m}:latest" not in terpasang
    ]
    return {
        "siap": hidup and not kurang,
        "alasan": None,
        "ollama": hidup,
        "host": PENGATURAN.host_ollama,
        "index": rag_indeks.index_tersedia(),
        "model_kurang": kurang,
        "model_terpasang": terpasang,
        **anggaran,
    }


def simpan_unggahan(berkas_unggah) -> Path:
    """Tulis satu unggahan Streamlit ke cakram; pypdf butuh path, bukan buffer."""
    UNGGAHAN_DIR.mkdir(parents=True, exist_ok=True)
    tujuan = UNGGAHAN_DIR / berkas_unggah.name
    tujuan.write_bytes(berkas_unggah.getbuffer())
    return tujuan


def baca_dokumen(
    path_list: list[Path], jenis_per_berkas: dict[str, str] | None = None
) -> BerkasPengajuan:
    return ekstraksi.baca_berkas_pengajuan(
        list(path_list), jenis_per_berkas=jenis_per_berkas or {}
    )


def telusuri_afiliasi(berkas: BerkasPengajuan, tanggal) -> dict:
    return jembatan.telusuri(berkas, tanggal)


def cari_kebijakan(kueri: str, top_k: int = 5) -> list[dict]:
    return rag_cari.kutipan(kueri, top_k=top_k)


def jawab_kebijakan(pertanyaan: str) -> dict:
    return rag_cari.jawab(pertanyaan)

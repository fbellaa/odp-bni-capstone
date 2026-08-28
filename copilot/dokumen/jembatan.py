"""Sambungan dari dokumen yang sudah dibaca ke lapisan graf pipeline.

`pipelines.graph.resolusi.telusuri_afiliasi()` sudah menerima daftar hasil
ekstraksi - alamat operasional, nama pengurus, rekening lawan - bukan PDF.
Kontrak itu memang dirancang untuk diisi parser. Modul ini yang mengisinya, dan
tidak melakukan apa pun selain itu.

Impor `pipelines` sengaja ditunda sampai fungsi dipanggil. Antarmuka Streamlit
harus tetap bisa membaca dokumen dan menyusun draft memo di lingkungan yang
tidak punya data gold sama sekali - misalnya sesi Kaggle yang baru dibuat.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from copilot.dokumen.skema import BerkasPengajuan

LOG = logging.getLogger(__name__)


def telusuri(
    berkas: BerkasPengajuan, tanggal: pd.Timestamp | str
) -> dict[str, Any]:
    """Jalankan resolusi afiliasi atas dokumen yang sudah dibaca.

    Keluarannya dict biasa - bukan `HasilResolusi` - supaya antarmuka dan
    lapisan agen tidak ikut bergantung pada tipe internal pipeline.

    Ini BUKAN skor risiko. Ia daftar debitur eksisting yang layak diperiksa
    analis, lengkap dengan alasan kemunculan masing-masing.
    """
    argumen = berkas.argumen_resolusi()

    try:
        from pipelines.graph.resolusi import telusuri_afiliasi
    except ImportError as exc:
        LOG.warning("lapisan graf tidak tersedia: %s", exc)
        return _kosong(argumen, f"Modul pipeline tidak bisa diimpor: {exc}")

    try:
        hasil = telusuri_afiliasi(tanggal, **argumen)
    except FileNotFoundError as exc:
        # Tabel gold belum dibangun. Ini kondisi wajar di lingkungan demo,
        # bukan galat - dan memo harus mengatakannya, bukan diam.
        LOG.warning("tabel gold belum tersedia: %s", exc)
        return _kosong(argumen, f"Tabel gold belum dibangun: {exc}")

    kandidat = hasil.kandidat
    return {
        "tersedia": True,
        "argumen": argumen,
        "tanggal": str(hasil.tanggal.date()),
        "jumlah_kandidat": hasil.jumlah_kandidat,
        "perlu_telaah": hasil.perlu_telaah,
        "jalur_terpakai": hasil.jalur_terpakai,
        "jalur_kosong": hasil.jalur_kosong,
        "cakupan": hasil.cakupan,
        "kandidat": [] if kandidat.empty else kandidat.to_dict("records"),
        "ada_afiliasi_gagal_bayar": (
            False if kandidat.empty
            else bool(kandidat.get("afiliasi_sudah_gagal_bayar", pd.Series(dtype=bool)).any())
        ),
        "catatan": None,
    }


def _kosong(argumen: dict, catatan: str) -> dict[str, Any]:
    return {
        "tersedia": False,
        "argumen": argumen,
        "tanggal": None,
        "jumlah_kandidat": 0,
        "perlu_telaah": False,
        "jalur_terpakai": [],
        "jalur_kosong": [
            f"{nama} (dokumen tidak disertakan)"
            for nama, nilai in (
                ("alamat", argumen["alamat_operasional"]),
                ("pengurus", argumen["nama_pengurus"]),
                ("rekening_koran", argumen["rekening_lawan"]),
            )
            if not nilai
        ],
        "cakupan": {},
        "kandidat": [],
        "ada_afiliasi_gagal_bayar": False,
        "catatan": catatan,
    }


# Label yang dibaca manusia untuk tiap semesta pencarian.
_LABEL_CAKUPAN = {
    "debitur_beralamat": "debitur beralamat terdaftar",
    "alamat": "alamat (di luar alamat agen registrasi)",
    "pihak": "pengurus/pemilik aktif",
    "rekening_lawan": "rekening lawan",
    "transfer_sampai_tanggal": "transfer sampai tanggal penilaian",
}


def _kalimat_cakupan(cakupan: dict[str, int] | None) -> str:
    """Nyatakan besar semesta yang dicari - tanpa ini, nihil tidak bisa ditafsirkan."""
    if not cakupan:
        return "Cakupan penelusuran tidak tercatat pada sesi ini."
    bagian = [
        f"{jumlah:,} {_LABEL_CAKUPAN.get(kunci, kunci)}".replace(",", ".")
        for kunci, jumlah in cakupan.items()
        if jumlah
    ]
    if not bagian:
        return "Cakupan penelusuran tidak tercatat pada sesi ini."
    return "Cakupan pencarian: " + "; ".join(bagian) + "."


def ringkas_untuk_memo(hasil: dict[str, Any]) -> str:
    """Satu paragraf siap tempel ke bagian afiliasi pada draft memo."""
    if not hasil["tersedia"]:
        return (
            "Penelusuran afiliasi tidak dapat dijalankan pada sesi ini "
            f"({hasil['catatan']}). Bagian ini wajib diisi manual sebelum memo naik ke komite."
        )
    if hasil["jumlah_kandidat"] == 0:
        jalur = ", ".join(hasil["jalur_kosong"]) or "-"
        return (
            "Tidak ditemukan kandidat afiliasi pada tanggal penilaian "
            f"({hasil['tanggal']}). Jalur yang tidak menghasilkan kecocokan: {jalur}. "
            f"{_kalimat_cakupan(hasil.get('cakupan'))} Hasil nihil ini berarti "
            "TIDAK DIKETAHUI, bukan terbukti tidak berafiliasi: penelusuran hanya "
            "menjangkau pihak yang sudah dikenal bank, sehingga afiliasi dengan "
            "pihak di luar basis tersebut tidak terdeteksi metode ini dan tetap "
            "harus diuji lewat wawancara serta dokumen CDD."
        )

    dasar = sorted({k.get("dasar", "-") for k in hasil["kandidat"]})
    kalimat = (
        f"Ditemukan {hasil['jumlah_kandidat']} kandidat afiliasi lewat jalur "
        f"{', '.join(hasil['jalur_terpakai'])} (dasar pencocokan: {', '.join(dasar)}). "
        f"{_kalimat_cakupan(hasil.get('cakupan'))}"
    )
    if hasil["ada_afiliasi_gagal_bayar"]:
        kalimat += (
            " Sebagian kandidat sudah tercatat gagal bayar pada tanggal penilaian, "
            "sehingga eksposur grup wajib ditelaah sebelum akad."
        )
    if hasil["perlu_telaah"]:
        kalimat += " Ambang penelaahan lanjutan KKK-13.6 terpicu."
    return kalimat

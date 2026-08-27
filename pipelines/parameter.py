"""Catat parameter efektif yang menghasilkan sebuah build gold.

Alasan modul ini ada, dan wajib dibaca sebelum menghapusnya:

`pipelines/config.py` memanggil `load_dotenv()`, sehingga berkas `.env` MENIMPA
nilai default di kode. Sepanjang pembangunan proyek ini, `.env` yang tertinggal
memuat `N_DEBITUR=3000` sementara kode sudah berpindah ke 6000. Akibatnya dua
pembangunan menghasilkan populasi berbeda dengan kode dan seed yang sama persis,
dan gejalanya menyamar sebagai nondeterminisme - butuh penelusuran panjang
sebelum ketahuan bahwa yang berubah adalah parameternya, bukan pipeline-nya.

Dengan tabel ini, tiap tabel gold bisa ditelusuri ke konfigurasi yang
menghasilkannya, dan selisih parameter ketahuan langsung dari datanya sendiri.
Kolom `sumber` menandai parameter mana yang datang dari environment - itulah
kolom yang akan menjawab pertanyaan "kenapa angkanya beda dari kemarin".
"""

from __future__ import annotations

import dataclasses
import logging
import os
import platform
import subprocess
from datetime import datetime

import pandas as pd

from pipelines.config import Settings, settings
from pipelines.utils import write_table

LOG = logging.getLogger("pipelines.parameter")

# Nama environment variable per field Settings, mengikuti config.py.
ENV_PER_FIELD = {
    "seed": "PIPELINE_SEED",
    "n_debitur": "N_DEBITUR",
    "panel_years": "PANEL_YEARS",
    "sample_mode": "SAMPLE_MODE",
    "csv_chunksize": "CSV_CHUNKSIZE",
    "injeksi_afiliasi": "INJEKSI_AFILIASI",
    "tulis_ke_postgres": "LOAD_TO_POSTGRES",
}


def _git_commit() -> str:
    try:
        hasil = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if hasil.returncode == 0:
            return hasil.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "(tidak diketahui)"


def _git_kotor() -> bool:
    """True kalau ada perubahan belum dikomit - build tidak bisa direproduksi persis."""
    try:
        hasil = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        return hasil.returncode == 0 and bool(hasil.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def kumpulkan_parameter() -> pd.DataFrame:
    """Susun parameter efektif beserta asalnya (default kode atau environment)."""
    baris = []
    for field in dataclasses.fields(Settings):
        nama = field.name
        nilai = getattr(settings, nama)
        env = ENV_PER_FIELD.get(nama)
        dari_env = bool(env and os.getenv(env) is not None)
        baris.append(
            {
                "parameter": nama,
                "nilai": str(nilai),
                "sumber": f"env:{env}" if dari_env else "default_kode",
                "default_kode": str(field.default) if field.default is not dataclasses.MISSING else "",
                "kategori": "settings",
            }
        )

    # Properti turunan yang ikut menentukan bentuk data.
    for nama, nilai in (
        ("tahun_buku_terakhir", settings.tahun_buku_terakhir),
        ("jumlah_snapshot", len(settings.snapshot_dates)),
        ("snapshot_pertama", settings.snapshot_dates[0]),
        ("snapshot_terakhir", settings.snapshot_dates[-1]),
    ):
        baris.append(
            {
                "parameter": nama,
                "nilai": str(nilai),
                "sumber": "turunan",
                "default_kode": "",
                "kategori": "turunan",
            }
        )

    for nama, nilai in (
        ("dibangun_pada", datetime.now().isoformat(timespec="seconds")),
        ("git_commit", _git_commit()),
        ("git_ada_perubahan_belum_dikomit", _git_kotor()),
        ("python", platform.python_version()),
        ("pandas", pd.__version__),
        ("platform", platform.platform()),
    ):
        baris.append(
            {
                "parameter": nama,
                "nilai": str(nilai),
                "sumber": "lingkungan",
                "default_kode": "",
                "kategori": "lingkungan",
            }
        )

    return pd.DataFrame(baris)


def tulis_parameter_build() -> pd.DataFrame:
    """Tulis gold.parameter_build dan peringatkan bila ada penimpaan dari .env."""
    df = kumpulkan_parameter()

    ditimpa = df[df["sumber"].str.startswith("env:")]
    for baris in ditimpa.itertuples():
        LOG.warning(
            "parameter '%s' ditimpa dari environment: %s (default kode: %s)",
            baris.parameter,
            baris.nilai,
            baris.default_kode,
        )

    write_table(df, "gold", "parameter_build")
    return df

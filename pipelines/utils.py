"""Helper I/O dan pembersihan yang dipakai lintas layer bronze/silver/gold."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from pipelines.config import BRONZE_DIR, GOLD_DIR, SILVER_DIR

LOG = logging.getLogger("pipelines")

_LAYER_DIR = {"bronze": BRONZE_DIR, "silver": SILVER_DIR, "gold": GOLD_DIR}


def layer_path(layer: str, name: str) -> Path:
    directory = _LAYER_DIR[layer]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.parquet"


def write_table(df: pd.DataFrame, layer: str, name: str) -> Path:
    """Tulis dataframe ke parquet pada layer tertentu dan kembalikan path-nya."""
    path = layer_path(layer, name)
    df.to_parquet(path, index=False)
    LOG.info("tulis %s.%s -> %s baris=%s kolom=%s", layer, name, path, len(df), df.shape[1])
    return path


def read_table(layer: str, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = layer_path(layer, name)
    if not path.exists():
        raise FileNotFoundError(
            f"Tabel {layer}.{name} belum ada di {path}. Jalankan flow layer sebelumnya."
        )
    return pd.read_parquet(path, columns=columns)


def table_exists(layer: str, name: str) -> bool:
    return layer_path(layer, name).exists()


def iter_csv_chunks(
    path: Path,
    chunksize: int,
    usecols: list[str] | None = None,
    max_rows: int | None = None,
    **kwargs,
) -> Iterator[pd.DataFrame]:
    """Baca CSV besar per potongan, berhenti setelah max_rows baris."""
    terbaca = 0
    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=usecols,
        low_memory=False,
        **kwargs,
    )
    for chunk in reader:
        if max_rows is not None and terbaca + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - terbaca]
        terbaca += len(chunk)
        if len(chunk):
            yield chunk
        if max_rows is not None and terbaca >= max_rows:
            break


_MONEY_RE = re.compile(r"[^0-9.\-]")


def bersihkan_uang(seri: pd.Series) -> pd.Series:
    """Ubah string berformat '$1,234,567.00 ' menjadi float."""
    if pd.api.types.is_numeric_dtype(seri):
        return seri.astype("float64")
    dibersihkan = seri.astype("string").str.strip().str.replace(_MONEY_RE, "", regex=True)
    return pd.to_numeric(dibersihkan, errors="coerce")


def rasio_aman(pembilang: pd.Series, penyebut: pd.Series, batas: float = 1e6) -> pd.Series:
    """Pembagian yang tahan penyebut nol/negatif-kecil, hasil dipangkas ke +/- batas."""
    penyebut = penyebut.replace(0, np.nan)
    hasil = pembilang / penyebut
    return hasil.replace([np.inf, -np.inf], np.nan).clip(-batas, batas)


def winsorize(seri: pd.Series, bawah: float = 0.01, atas: float = 0.99) -> pd.Series:
    lo, hi = seri.quantile(bawah), seri.quantile(atas)
    return seri.clip(lo, hi)


def kuantil_bucket(seri: pd.Series, n: int = 5) -> pd.Series:
    """Bucket kuantil yang tidak meledak saat nilainya banyak yang sama."""
    peringkat = seri.rank(method="first", pct=True)
    return np.ceil(peringkat * n).clip(1, n).fillna(1).astype("int8")


def akhir_bulan(seri: pd.Series) -> pd.Series:
    return pd.to_datetime(seri) + pd.offsets.MonthEnd(0)


def akhir_bulan_sebelum(tanggal: pd.Series) -> pd.Series:
    """Akhir bulan SEBELUM tanggal tersebut - kunci anti-bocor pada FEAT_GRAF_PIT.

    Untuk pengajuan 2025-03-15 hasilnya 2025-02-28, bukan 2025-03-31. Perbedaan
    satu bulan ini persis yang membedakan fitur graf yang sah dari yang bocor.
    """
    ts = pd.to_datetime(tanggal)
    return ts.dt.to_period("M").dt.to_timestamp() - pd.Timedelta(days=1)


def kolom_wajib(df: pd.DataFrame, kolom: Iterable[str], nama_tabel: str) -> None:
    hilang = [c for c in kolom if c not in df.columns]
    if hilang:
        raise ValueError(f"{nama_tabel}: kolom wajib hilang {hilang}")

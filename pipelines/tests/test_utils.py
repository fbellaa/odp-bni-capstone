"""Uji unit untuk helper yang dipakai lintas layer."""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines.utils import (
    akhir_bulan,
    akhir_bulan_sebelum,
    bersihkan_uang,
    kolom_wajib,
    kuantil_bucket,
    rasio_aman,
)


def test_akhir_bulan_sebelum_mundur_satu_bulan():
    """Regresi: sempat mengembalikan akhir bulan yang SAMA - itu kebocoran waktu."""
    tanggal = pd.Series(pd.to_datetime(["2025-03-15", "2025-01-01", "2025-12-31", "2025-03-01"]))
    hasil = akhir_bulan_sebelum(tanggal)
    assert list(hasil.dt.strftime("%Y-%m-%d")) == [
        "2025-02-28",
        "2024-12-31",
        "2025-11-30",
        "2025-02-28",
    ]


def test_akhir_bulan_sebelum_selalu_lebih_awal_dari_bulan_pengajuan():
    tanggal = pd.Series(pd.date_range("2025-01-01", "2025-12-31", freq="D"))
    hasil = akhir_bulan_sebelum(tanggal)
    assert (hasil < tanggal.values.astype("datetime64[M]")).all()


def test_akhir_bulan():
    tanggal = pd.Series(pd.to_datetime(["2025-02-10", "2025-02-28"]))
    assert list(akhir_bulan(tanggal).dt.day) == [28, 28]


def test_bersihkan_uang_format_sba():
    seri = pd.Series(["$60,000.00 ", "$0.00 ", None, '"$1,234,567.00 "'])
    hasil = bersihkan_uang(seri)
    assert hasil.iloc[0] == 60000.0
    assert hasil.iloc[1] == 0.0
    assert pd.isna(hasil.iloc[2])
    assert hasil.iloc[3] == 1234567.0


def test_rasio_aman_menolak_pembagian_nol():
    hasil = rasio_aman(pd.Series([10.0, 5.0]), pd.Series([0.0, 2.5]))
    assert pd.isna(hasil.iloc[0])
    assert hasil.iloc[1] == 2.0


def test_kuantil_bucket_stabil_saat_banyak_nilai_sama():
    seri = pd.Series([1.0] * 50 + [2.0] * 50)
    bucket = kuantil_bucket(seri, 5)
    assert bucket.between(1, 5).all()
    assert bucket.nunique() == 5


def test_kolom_wajib_menggagalkan_kolom_hilang():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="kolom wajib hilang"):
        kolom_wajib(df, ["a", "b"], "tabel_uji")

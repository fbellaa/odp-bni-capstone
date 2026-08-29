"""Helper format angka untuk tampilan demo.

Segmen komersial bergerak pada satuan miliar sampai triliun, jadi bentuk
singkatnya dimulai dari miliar dan tidak lagi memakai satuan ribuan.
"""
from __future__ import annotations


def _angka_id(nilai: float, desimal: int) -> str:
    """Angka gaya Indonesia: titik pemisah ribuan, koma pemisah desimal."""
    return f"{nilai:,.{desimal}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def rupiah(nilai: float, singkat: bool = False) -> str:
    if nilai is None:
        return "-"
    if singkat:
        if abs(nilai) >= 1_000_000_000_000:
            return f"Rp {_angka_id(nilai / 1_000_000_000_000, 2)} T"
        if abs(nilai) >= 1_000_000_000:
            return f"Rp {_angka_id(nilai / 1_000_000_000, 1)} M"
        if abs(nilai) >= 1_000_000:
            return f"Rp {_angka_id(nilai / 1_000_000, 0)} jt"
    return "Rp " + _angka_id(nilai, 0)


def miliar(nilai: float, desimal: int = 1) -> str:
    """Bentuk baku pada halaman komersial: seluruh nominal dibaca dalam miliar."""
    if nilai is None:
        return "-"
    return f"Rp {_angka_id(nilai / 1_000_000_000, desimal)} M"


def persen(nilai: float, desimal: int = 2) -> str:
    return f"{nilai * 100:.{desimal}f}%".replace(".", ",")


def kali(nilai: float, desimal: int = 2) -> str:
    """Rasio keuangan komersial ditulis dengan satuan x, koma desimal Indonesia."""
    return f"{nilai:.{desimal}f}x".replace(".", ",")


def bps(nilai: float) -> str:
    return f"{nilai * 10_000:,.0f} bps".replace(",", ".")


def cacah(nilai: int) -> str:
    """Bilangan bulat dengan titik pemisah ribuan."""
    return f"{int(nilai):,}".replace(",", ".")

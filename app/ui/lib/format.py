"""Helper format angka untuk tampilan demo."""
from __future__ import annotations


def rupiah(nilai: float, singkat: bool = False) -> str:
    if nilai is None:
        return "-"
    if singkat:
        if abs(nilai) >= 1_000_000_000:
            return f"Rp {nilai / 1_000_000_000:,.2f} M".replace(",", ".")
        if abs(nilai) >= 1_000_000:
            return f"Rp {nilai / 1_000_000:,.1f} jt".replace(",", ".")
        if abs(nilai) >= 1_000:
            return f"Rp {nilai / 1_000:,.0f} rb".replace(",", ".")
    return "Rp " + f"{nilai:,.0f}".replace(",", ".")


def persen(nilai: float, desimal: int = 2) -> str:
    return f"{nilai * 100:.{desimal}f}%".replace(".", ",")


def bps(nilai: float) -> str:
    return f"{nilai * 10_000:,.0f} bps".replace(",", ".")

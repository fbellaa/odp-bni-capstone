"""Mesin skoring tiruan untuk live demo.

CATATAN PENTING
---------------
Seluruh perhitungan di sini adalah *placeholder deterministik*, bukan model.
Tujuannya hanya agar halaman what-if benar-benar responsif terhadap masukan
selama model asli (PD / LGD / network risk) belum tersedia.

Ketika layanan FastAPI sudah jalan, ganti isi fungsi di modul ini dengan
pemanggilan endpoint `/api/score_pd`, `/api/estimate_lgd`,
`/api/recommend_limit_pricing`, dan `/api/score_network_risk`.
Antarmuka fungsi (nama argumen dan bentuk keluaran) sengaja dibuat sama dengan
tool agen pada proposal bagian 5.1 supaya halaman tidak perlu diubah.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Asumsi biaya yang dipakai pada rantai perhitungan keputusan (proposal 3.2)
BIAYA_DANA = 0.0550
BIAYA_OPERASIONAL = 0.0125
MARGIN_TARGET = 0.0350
PRICING_MIN = 0.0900
PRICING_MAX = 0.2400

# Recovery rate per jenis agunan -> LGD = 1 - recovery, disesuaikan coverage
RECOVERY_AGUNAN = {
    "Tanpa agunan": 0.10,
    "BPKB motor": 0.35,
    "BPKB mobil": 0.50,
    "Deposito": 0.90,
    "SHM / SHGB": 0.75,
    "Kios / los pasar": 0.55,
    "Penjaminan pihak ketiga": 0.45,
}

BATAS_GRADE = [
    (0.010, "AAA"),
    (0.020, "AA"),
    (0.035, "A"),
    (0.060, "BBB"),
    (0.100, "BB"),
    (0.160, "B"),
    (1.000, "CCC"),
]

WARNA_GRADE = {
    "AAA": "#1b7f4b", "AA": "#1b7f4b", "A": "#3d8f3d",
    "BBB": "#b58900", "BB": "#c9721c", "B": "#c0392b", "CCC": "#8e1b1b",
}


@dataclass
class Kontribusi:
    fitur: str
    nilai: str
    dampak: float  # dalam satuan log-odds, positif = menaikkan risiko


@dataclass
class HasilSkor:
    pd: float
    grade: str
    lgd: float
    ead: float
    expected_loss: float
    pricing: float
    limit_usulan: float
    tenor_usulan: int
    dscr: float
    angsuran: float
    kontribusi: list = field(default_factory=list)
    catatan: list = field(default_factory=list)


def angsuran_anuitas(pokok: float, tenor_bulan: int, rate_tahunan: float) -> float:
    i = rate_tahunan / 12
    if tenor_bulan <= 0:
        return 0.0
    if i == 0:
        return pokok / tenor_bulan
    return pokok * i / (1 - (1 + i) ** (-tenor_bulan))


def estimate_lgd(jenis_agunan: str, nilai_agunan: float, plafon: float) -> float:
    """LGD = 1 - recovery efektif, dibatasi 5% sampai 90%."""
    recovery_dasar = RECOVERY_AGUNAN.get(jenis_agunan, 0.20)
    coverage = 0.0 if plafon <= 0 else min(nilai_agunan / plafon, 1.5)
    recovery_efektif = recovery_dasar * min(coverage / 1.0, 1.0)
    return float(min(max(1 - recovery_efektif, 0.05), 0.90))


def score_pd(pengajuan: dict) -> tuple[float, list[Kontribusi]]:
    """Model PD tiruan: logistik atas fitur keuangan, relasi, dan graf."""
    omzet = max(pengajuan["omzet_bulanan"], 1.0)
    plafon = max(pengajuan["plafon"], 1.0)
    tenor = max(int(pengajuan["tenor_bulan"]), 1)
    lama_usaha = pengajuan["lama_usaha_thn"]
    stabilitas = pengajuan.get("stabilitas_arus_kas", 0.25)   # 0 stabil, 1 bergejolak
    supplier_hhi = pengajuan.get("supplier_concentration_hhi", 0.30)
    neighbor_default = pengajuan.get("neighbor_default_rate_1hop", 0.05)
    network_risk = pengajuan.get("network_risk_score", 20) / 100
    kedalaman_relasi = pengajuan.get("tenure_nasabah_thn", 2.0)

    arus_kas = omzet * pengajuan.get("margin_usaha", 0.14)
    angsuran = angsuran_anuitas(plafon, tenor, 0.16)
    dscr = arus_kas / max(angsuran, 1.0)
    rasio_plafon_omzet = plafon / (omzet * 12)

    komponen = [
        Kontribusi("Debt service coverage ratio", f"{dscr:.2f}x", -0.85 * (dscr - 1.50)),
        Kontribusi("Rasio plafon terhadap omzet tahunan", f"{rasio_plafon_omzet:.2f}x", 1.60 * (rasio_plafon_omzet - 0.30)),
        Kontribusi("Lama usaha", f"{lama_usaha:.1f} tahun", -0.22 * (lama_usaha - 4.0)),
        Kontribusi("Stabilitas arus kas", f"{stabilitas:.2f}", 1.90 * (stabilitas - 0.25)),
        Kontribusi("Konsentrasi pemasok (HHI)", f"{supplier_hhi:.2f}", 1.10 * (supplier_hhi - 0.30)),
        Kontribusi("Gagal bayar tetangga 1-hop", f"{neighbor_default * 100:.1f}%", 3.20 * (neighbor_default - 0.05)),
        Kontribusi("Skor risiko jaringan", f"{network_risk * 100:.0f}/100", 1.40 * (network_risk - 0.20)),
        Kontribusi("Kedalaman relasi dengan bank", f"{kedalaman_relasi:.1f} tahun", -0.18 * (kedalaman_relasi - 2.0)),
        Kontribusi("Tenor", f"{tenor} bulan", 0.012 * (tenor - 24)),
    ]

    z = -3.05 + sum(k.dampak for k in komponen)
    pd = 1 / (1 + math.exp(-z))
    pd = float(min(max(pd, 0.002), 0.65))
    komponen.sort(key=lambda k: abs(k.dampak), reverse=True)
    return pd, komponen


def grade_dari_pd(pd: float) -> str:
    for batas, grade in BATAS_GRADE:
        if pd <= batas:
            return grade
    return "CCC"


def recommend_limit_pricing(pengajuan: dict) -> HasilSkor:
    """Rantai perhitungan keputusan sesuai proposal bagian 3.2."""
    pd, kontribusi = score_pd(pengajuan)
    plafon = float(pengajuan["plafon"])
    tenor = int(pengajuan["tenor_bulan"])
    lgd = estimate_lgd(pengajuan["jenis_agunan"], pengajuan.get("nilai_agunan", 0.0), plafon)
    ead = plafon
    el = pd * lgd * ead

    pricing = BIAYA_DANA + BIAYA_OPERASIONAL + MARGIN_TARGET + (el / max(plafon, 1.0))
    pricing = float(min(max(pricing, PRICING_MIN), PRICING_MAX))

    omzet = max(pengajuan["omzet_bulanan"], 1.0)
    arus_kas = omzet * pengajuan.get("margin_usaha", 0.14)

    # Limit dibatasi tiga hal: permintaan, kapasitas arus kas (DSCR >= 1.35),
    # dan pagu kewenangan berdasarkan grade.
    angsuran_maks = arus_kas / 1.35
    i = pricing / 12
    plafon_kapasitas = angsuran_maks * (1 - (1 + i) ** (-tenor)) / i if i > 0 else angsuran_maks * tenor
    pagu_grade = {"AAA": 500e6, "AA": 500e6, "A": 400e6, "BBB": 300e6, "BB": 200e6, "B": 100e6, "CCC": 0.0}
    grade = grade_dari_pd(pd)
    limit = float(min(plafon, plafon_kapasitas, pagu_grade[grade]))
    limit = math.floor(limit / 5_000_000) * 5_000_000

    angsuran = angsuran_anuitas(max(limit, 1.0), tenor, pricing)
    dscr = arus_kas / max(angsuran, 1.0)

    catatan = []
    if limit < plafon:
        catatan.append(
            f"Limit diturunkan dari permintaan karena batas {'kapasitas arus kas' if plafon_kapasitas < pagu_grade[grade] else 'kewenangan grade ' + grade}."
        )
    if pengajuan.get("network_risk_score", 0) >= 60:
        catatan.append("Skor risiko jaringan tinggi — wajib verifikasi penjamin dan rekening penerima pencairan.")
    if pd > 0.10:
        catatan.append("PD di atas ambang 10% — keputusan naik ke pejabat pemutus satu tingkat di atas.")
    if lgd > 0.70:
        catatan.append("Pertanggungan agunan tipis; pertimbangkan tambahan agunan atau penjaminan.")

    return HasilSkor(
        pd=pd, grade=grade, lgd=lgd, ead=ead, expected_loss=el, pricing=pricing,
        limit_usulan=limit, tenor_usulan=tenor, dscr=dscr, angsuran=angsuran,
        kontribusi=kontribusi, catatan=catatan,
    )


def keputusan_dari_hasil(hasil: HasilSkor) -> str:
    if hasil.limit_usulan <= 0 or hasil.pd > 0.16:
        return "TOLAK"
    if hasil.pd > 0.10 or hasil.dscr < 1.35:
        return "SETUJU DENGAN SYARAT"
    return "SETUJU"

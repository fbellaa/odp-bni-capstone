"""Definisi tool dalam format function calling Ollama, plus tabel dispatch.

Deskripsi di sini adalah prompt, bukan dokumentasi. Model 3-7B memilih tool
berdasarkan kalimat pertama deskripsi, jadi kalimat itu ditulis sebagai
perintah ("Hitung ...", "Periksa ..."), bukan sebagai penjelasan konsep.

Satuan disebut ulang di hampir setiap field karena itulah kesalahan yang paling
sering terjadi: model mengirim 1.8 untuk "1,8 miliar" atau 10.5 untuk "10,5
persen". Menyebut satuan di deskripsi jauh lebih murah daripada menangkapnya
sebagai galat validasi sesudahnya.
"""

from __future__ import annotations

from typing import Any, Callable

from copilot.alat import keuangan
from copilot.alat.parameter import RECOVERY_AGUNAN

_RUPIAH = "Nilai dalam rupiah penuh (bukan miliar, bukan juta). Rp 1,5 miliar ditulis 1500000000."
_PECAHAN = "Ditulis sebagai pecahan, bukan persen: 10,5 persen ditulis 0.105."


def _fungsi(nama: str, deskripsi: str, properti: dict, wajib: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": nama,
            "description": deskripsi,
            "parameters": {
                "type": "object",
                "properties": properti,
                "required": wajib,
            },
        },
    }


def _angka(deskripsi: str) -> dict[str, str]:
    return {"type": "number", "description": deskripsi}


DEFINISI: list[dict[str, Any]] = [
    _fungsi(
        "hitung_rasio_keuangan",
        "Hitung DER, utang terhadap EBITDA, interest coverage, dan marjin dari pos laporan "
        "keuangan. Panggil ini lebih dulu sebelum menilai covenant.",
        {
            "penjualan": _angka(f"Penjualan/pendapatan tahunan. {_RUPIAH}"),
            "ebitda": _angka(f"EBITDA tahunan. {_RUPIAH}"),
            "utang_berbunga": _angka(
                f"Total utang berbunga (pinjaman bank, obligasi, sewa pembiayaan). "
                f"BUKAN total liabilitas. {_RUPIAH}"
            ),
            "ekuitas": _angka(f"Total ekuitas. {_RUPIAH}"),
            "beban_bunga": _angka(f"Beban bunga tahunan. {_RUPIAH}"),
            "laba_bersih": _angka(f"Laba bersih tahunan, bila tersedia. {_RUPIAH}"),
            "total_aset": _angka(f"Total aset, bila tersedia. {_RUPIAH}"),
            "total_liabilitas": _angka(f"Total liabilitas, bila tersedia. {_RUPIAH}"),
        },
        ["penjualan", "ebitda", "utang_berbunga", "ekuitas"],
    ),
    _fungsi(
        "hitung_angsuran",
        "Hitung angsuran bulanan dan kewajiban tahunan satu fasilitas kredit. Sebutkan "
        "jenis_fasilitas supaya fasilitas revolving tidak dihitung sebagai anuitas.",
        {
            "pokok": _angka(f"Pokok/plafon fasilitas. {_RUPIAH}"),
            "tenor_bulan": {"type": "integer", "description": "Tenor dalam bulan."},
            "bunga_tahunan": _angka(f"Suku bunga tahunan. {_PECAHAN}"),
            "jenis_fasilitas": {
                "type": "string",
                "description": "Jenis fasilitas, contoh 'Modal kerja - rekening koran' atau "
                               "'Investasi - term loan'.",
            },
        },
        ["pokok", "tenor_bulan", "bunga_tahunan"],
    ),
    _fungsi(
        "hitung_dscr",
        "Hitung DSCR dari EBITDA terhadap seluruh kewajiban tahunan. Sertakan kewajiban "
        "fasilitas eksisting, bukan hanya fasilitas baru.",
        {
            "ebitda": _angka(f"EBITDA tahunan. {_RUPIAH}"),
            "kewajiban_tahunan_baru": _angka(
                f"Kewajiban tahunan fasilitas yang diajukan, dari hitung_angsuran. {_RUPIAH}"
            ),
            "kewajiban_tahunan_eksisting": _angka(
                f"Kewajiban tahunan fasilitas yang sudah berjalan. {_RUPIAH}"
            ),
        },
        ["ebitda", "kewajiban_tahunan_baru"],
    ),
    _fungsi(
        "estimasi_lgd",
        "Hitung loss given default dari jenis dan nilai agunan terhadap plafon.",
        {
            "jenis_agunan": {
                "type": "string",
                "enum": list(RECOVERY_AGUNAN),
                "description": "Jenis agunan. Pilih persis salah satu dari daftar.",
            },
            "nilai_agunan": _angka(f"Nilai taksasi agunan. {_RUPIAH}"),
            "plafon": _angka(f"Plafon fasilitas. {_RUPIAH}"),
        },
        ["jenis_agunan", "nilai_agunan", "plafon"],
    ),
    _fungsi(
        "periksa_agunan",
        "Periksa tingkat pertanggungan agunan terhadap minimum kelas rating.",
        {
            "grade": {"type": "string", "description": "Kelas rating internal, misal 'BBB'."},
            "coverage": _angka("Rasio nilai agunan terhadap plafon, dari estimasi_lgd."),
        },
        ["grade", "coverage"],
    ),
    _fungsi(
        "grade_dari_pd",
        "Tentukan kelas rating internal dari probability of default 12 bulan.",
        {"pd_12bulan": _angka(f"PD 12 bulan. {_PECAHAN}")},
        ["pd_12bulan"],
    ),
    _fungsi(
        "hitung_expected_loss",
        "Hitung expected loss dari PD, LGD, dan exposure at default.",
        {
            "pd_12bulan": _angka(f"PD 12 bulan. {_PECAHAN}"),
            "lgd": _angka(f"LGD, dari estimasi_lgd. {_PECAHAN}"),
            "ead": _angka(f"Exposure at default, umumnya sebesar plafon. {_RUPIAH}"),
        },
        ["pd_12bulan", "lgd", "ead"],
    ),
    _fungsi(
        "usulkan_pricing",
        "Hitung suku bunga usulan berbasis risiko dari PD dan LGD.",
        {
            "pd_12bulan": _angka(f"PD 12 bulan. {_PECAHAN}"),
            "lgd": _angka(f"LGD. {_PECAHAN}"),
        },
        ["pd_12bulan", "lgd"],
    ),
    _fungsi(
        "periksa_batas_segmen",
        "Periksa apakah pengajuan masuk definisi segmen komersial.",
        {
            "penjualan": _angka(f"Penjualan tahunan. {_RUPIAH}"),
            "plafon": _angka(f"Plafon yang diajukan. {_RUPIAH}"),
            "saldo_rata_rata": _angka(f"Saldo giro rata-rata, bila tersedia. {_RUPIAH}"),
        },
        ["penjualan", "plafon"],
    ),
    _fungsi(
        "periksa_bmpk",
        "Hitung sisa ruang batas maksimum pemberian kredit satu grup debitur.",
        {
            "eksposur_grup_berjalan": _angka(
                f"Total eksposur grup yang sudah berjalan. {_RUPIAH}"
            ),
            "limit_usulan": _angka(f"Limit yang diusulkan. {_RUPIAH}"),
        },
        ["eksposur_grup_berjalan", "limit_usulan"],
    ),
    _fungsi(
        "periksa_covenant",
        "Periksa DER, DSCR, dan interest coverage terhadap ambang covenant kelas rating. "
        "Pakai der_total, bukan der: ambangnya diturunkan atas basis total liabilitas.",
        {
            "grade": {"type": "string", "description": "Kelas rating internal, misal 'BBB'."},
            "der_total": _angka(
                "DER total (total liabilitas / ekuitas), field `der_total` dari "
                "hitung_rasio_keuangan. Inilah basis ambang covenant."
            ),
            "dscr": _angka("DSCR, dari hitung_dscr."),
            "der": _angka(
                "Cadangan saja: DER berbunga (utang berbunga / ekuitas). Kirim ini "
                "HANYA bila laporan tidak memuat total liabilitas."
            ),
            "interest_coverage": _angka("Interest coverage, dari hitung_rasio_keuangan."),
        },
        ["grade", "der_total", "dscr"],
    ),
    _fungsi(
        "kewenangan_komite",
        "Tentukan komite pemutus dari besaran limit dan kelas rating.",
        {
            "limit": _angka(f"Limit yang diusulkan. {_RUPIAH}"),
            "grade": {"type": "string", "description": "Kelas rating internal, misal 'BBB'."},
        },
        ["limit", "grade"],
    ),
]

# Nama tool -> fungsi Python. Satu-satunya jalan agen memanggil perhitungan.
PETA: dict[str, Callable[..., dict[str, Any]]] = {
    "hitung_rasio_keuangan": keuangan.hitung_rasio_keuangan,
    "hitung_angsuran": keuangan.hitung_angsuran,
    "hitung_dscr": keuangan.hitung_dscr,
    "estimasi_lgd": keuangan.estimasi_lgd,
    "periksa_agunan": keuangan.periksa_agunan,
    "grade_dari_pd": keuangan.grade_dari_pd,
    "hitung_expected_loss": keuangan.hitung_expected_loss,
    "usulkan_pricing": keuangan.usulkan_pricing,
    "periksa_batas_segmen": keuangan.periksa_batas_segmen,
    "periksa_bmpk": keuangan.periksa_bmpk,
    "periksa_covenant": keuangan.periksa_covenant,
    "kewenangan_komite": keuangan.kewenangan_komite,
}


def _periksa_konsistensi() -> None:
    """Definisi dan implementasi harus sepadan.

    Tool yang dideklarasikan tapi tidak punya implementasi baru ketahuan saat
    model memanggilnya di tengah demo - terlalu terlambat.
    """
    dideklarasikan = {d["function"]["name"] for d in DEFINISI}
    if dideklarasikan != set(PETA):
        raise RuntimeError(
            "Definisi tool dan tabel dispatch tidak sepadan: "
            f"hanya di definisi {sorted(dideklarasikan - set(PETA))}, "
            f"hanya di peta {sorted(set(PETA) - dideklarasikan)}"
        )


_periksa_konsistensi()

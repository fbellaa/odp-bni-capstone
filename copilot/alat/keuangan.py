"""Perhitungan kredit - deterministik, tanpa model, tanpa keacakan.

Inilah yang dipanggil agen. Pembagian tugasnya tegas:

    model bahasa   membaca dokumen, memilih tool, menyusun narasi
    modul ini      seluruh angka

Tidak ada satu pun angka di draft memo yang boleh berasal dari model bahasa.
Karena itu setiap fungsi di sini mengembalikan `rumus` - string yang menuliskan
perhitungannya apa adanya - supaya angka di memo bisa diperiksa ulang analis
tanpa membaca kode.

Model PD/LGD/EWS belum dilatih pada tahap ini, jadi tidak ada tool ML di sini.
Yang ada hanya aritmetika kebijakan kredit, dan itu memang tidak butuh model.
"""

from __future__ import annotations

from typing import Any

from copilot.alat import parameter as par


class GalatMasukan(ValueError):
    """Argumen tool tidak masuk akal secara ekonomi.

    Dipisahkan dari ValueError biasa supaya lapisan agen bisa mengembalikan
    pesannya ke model sebagai umpan balik yang bisa diperbaiki, alih-alih
    menghentikan seluruh putaran.
    """


def _positif(nama: str, nilai: float) -> float:
    if nilai is None:
        raise GalatMasukan(f"{nama} wajib diisi.")
    nilai = float(nilai)
    if nilai <= 0:
        raise GalatMasukan(f"{nama} harus lebih besar dari nol, diterima {nilai}.")
    return nilai


def _bagi(pembilang: float, penyebut: float) -> float | None:
    """Pembagian yang mengembalikan None, bukan inf, saat penyebut nol.

    Perusahaan tanpa beban bunga benar-benar punya ICR tak hingga. Menuliskannya
    sebagai angka besar akan membuatnya lolos uji covenant seolah-olah terukur.
    """
    if not penyebut:
        return None
    return pembilang / penyebut


# ------------------------------------------------------------ rasio keuangan
def hitung_rasio_keuangan(
    penjualan: float,
    ebitda: float,
    utang_berbunga: float,
    ekuitas: float,
    beban_bunga: float = 0.0,
    laba_bersih: float | None = None,
    total_aset: float | None = None,
    total_liabilitas: float | None = None,
) -> dict[str, Any]:
    """Rasio inti yang dipakai penilaian kredit komersial."""
    penjualan = _positif("penjualan", penjualan)
    ekuitas = _positif("ekuitas", ekuitas)
    ebitda = float(ebitda)
    utang_berbunga = max(float(utang_berbunga), 0.0)
    beban_bunga = max(float(beban_bunga), 0.0)

    der = utang_berbunga / ekuitas
    utang_ebitda = _bagi(utang_berbunga, ebitda) if ebitda > 0 else None
    icr = _bagi(ebitda, beban_bunga)

    hasil = {
        "der": round(der, 4),
        "utang_terhadap_ebitda": None if utang_ebitda is None else round(utang_ebitda, 4),
        "interest_coverage": None if icr is None else round(icr, 4),
        "ebitda_margin": round(ebitda / penjualan, 4),
        "rumus": (
            f"DER berbunga = utang berbunga {utang_berbunga:,.0f} / ekuitas {ekuitas:,.0f} = {der:.2f}x; "
            f"EBITDA margin = {ebitda:,.0f} / {penjualan:,.0f} = {ebitda / penjualan:.2%}"
        ),
    }
    if total_liabilitas is not None:
        # Basis kedua, dan bukan sekadar pelengkap: ambang covenant serta fitur
        # DER pada data latih model PD diturunkan dari `total_liabilities /
        # ekuitas` (lihat pipelines/transform/silver.py). Menguji ambang itu
        # dengan DER berbunga - yang pembilangnya bagian dari total liabilitas -
        # selalu bias ke arah lolos.
        hasil["der_total"] = round(float(total_liabilitas) / ekuitas, 4)
    if laba_bersih is not None:
        hasil["marjin_laba_bersih"] = round(float(laba_bersih) / penjualan, 4)
        hasil["roe"] = round(float(laba_bersih) / ekuitas, 4)
        if total_aset:
            hasil["roa"] = round(float(laba_bersih) / float(total_aset), 4)
    if total_liabilitas is not None and total_aset:
        hasil["rasio_liabilitas_aset"] = round(float(total_liabilitas) / float(total_aset), 4)
    if icr is None:
        hasil["catatan"] = (
            "Beban bunga nol atau tidak tercantum, sehingga interest coverage tidak terdefinisi "
            "dan tidak boleh dianggap lolos covenant."
        )
    return hasil


# ------------------------------------------------------- angsuran dan kapasitas
def hitung_angsuran(
    pokok: float,
    tenor_bulan: int,
    bunga_tahunan: float,
    jenis_fasilitas: str | None = None,
) -> dict[str, Any]:
    """Kewajiban tahunan atas satu fasilitas.

    Fasilitas revolving tidak beramortisasi. Menguji kapasitas arus kasnya
    dengan angsuran anuitas akan menolak nasabah yang sebenarnya sanggup - itu
    sebabnya jenis fasilitas ikut jadi argumen, bukan diserahkan ke pemanggil.
    """
    pokok = _positif("pokok", pokok)
    bunga_tahunan = float(bunga_tahunan)
    if not 0 < bunga_tahunan < 1:
        raise GalatMasukan(
            f"bunga_tahunan ditulis sebagai pecahan, misal 0.105 untuk 10,5 persen. "
            f"Diterima {bunga_tahunan}."
        )

    revolving = bool(jenis_fasilitas and jenis_fasilitas in par.FASILITAS_REVOLVING)
    if revolving:
        kewajiban = pokok * bunga_tahunan
        return {
            "revolving": True,
            "angsuran_bulanan": round(kewajiban / 12, 2),
            "kewajiban_tahunan": round(kewajiban, 2),
            "rumus": (
                f"Fasilitas revolving: kewajiban tahunan = pokok {pokok:,.0f} x "
                f"bunga {bunga_tahunan:.2%} = {kewajiban:,.0f} (bunga saja, tanpa amortisasi)"
            ),
        }

    tenor_bulan = int(tenor_bulan)
    if tenor_bulan <= 0:
        raise GalatMasukan(f"tenor_bulan harus positif, diterima {tenor_bulan}.")

    i = bunga_tahunan / 12
    angsuran = pokok * i / (1 - (1 + i) ** -tenor_bulan)
    return {
        "revolving": False,
        "angsuran_bulanan": round(angsuran, 2),
        "kewajiban_tahunan": round(angsuran * 12, 2),
        "total_bayar": round(angsuran * tenor_bulan, 2),
        "rumus": (
            f"Anuitas: {pokok:,.0f} x i / (1 - (1+i)^-{tenor_bulan}), i = "
            f"{bunga_tahunan:.2%}/12 -> angsuran {angsuran:,.0f} per bulan"
        ),
    }


def hitung_dscr(
    ebitda: float,
    kewajiban_tahunan_baru: float,
    kewajiban_tahunan_eksisting: float = 0.0,
) -> dict[str, Any]:
    """Debt service coverage ratio atas seluruh kewajiban, bukan fasilitas baru saja."""
    total = float(kewajiban_tahunan_baru) + float(kewajiban_tahunan_eksisting)
    if total <= 0:
        raise GalatMasukan("Total kewajiban tahunan harus lebih besar dari nol.")
    dscr = float(ebitda) / total
    return {
        "dscr": round(dscr, 4),
        "total_kewajiban_tahunan": round(total, 2),
        "ambang_kebijakan": par.DSCR_MIN_KEBIJAKAN,
        "lolos": dscr >= par.DSCR_MIN_KEBIJAKAN,
        "rumus": (
            f"DSCR = EBITDA {float(ebitda):,.0f} / (kewajiban baru "
            f"{float(kewajiban_tahunan_baru):,.0f} + eksisting "
            f"{float(kewajiban_tahunan_eksisting):,.0f}) = {dscr:.2f}x"
        ),
    }


# ---------------------------------------------------------------- agunan, LGD
def estimasi_lgd(jenis_agunan: str, nilai_agunan: float, plafon: float) -> dict[str, Any]:
    """LGD dari jenis dan tingkat pertanggungan agunan.

    Pemulihan dibatasi pada porsi plafon yang benar-benar tertutup agunan.
    Agunan senilai setengah plafon tidak memberi pemulihan penuh atas seluruh
    eksposur, betapapun likuidnya.
    """
    if jenis_agunan not in par.RECOVERY_AGUNAN:
        raise GalatMasukan(
            f"jenis_agunan {jenis_agunan!r} tidak dikenal. Pilihan: "
            + "; ".join(par.RECOVERY_AGUNAN)
        )
    plafon = _positif("plafon", plafon)
    nilai_agunan = max(float(nilai_agunan), 0.0)

    pemulihan_maks = par.RECOVERY_AGUNAN[jenis_agunan]
    coverage = nilai_agunan / plafon
    pemulihan = pemulihan_maks * min(coverage, 1.0)
    lgd = 1 - pemulihan
    return {
        "lgd": round(lgd, 4),
        "coverage": round(coverage, 4),
        "pemulihan_efektif": round(pemulihan, 4),
        "pemulihan_maksimum_jenis": pemulihan_maks,
        "rumus": (
            f"LGD = 1 - ({pemulihan_maks:.0%} x min(coverage {coverage:.2f}, 1)) = {lgd:.2%}"
        ),
    }


def periksa_agunan(grade: str, coverage: float) -> dict[str, Any]:
    """Bandingkan tingkat pertanggungan terhadap minimum kelas rating."""
    grade = _grade_valid(grade)
    minimum = par.COVERAGE_MIN[grade]
    return {
        "grade": grade,
        "coverage": round(float(coverage), 4),
        "coverage_minimum": minimum,
        "lolos": float(coverage) >= minimum,
        "kekurangan": round(max(minimum - float(coverage), 0.0), 4),
        "rumus": f"Coverage {float(coverage):.2f}x terhadap minimum kelas {grade} {minimum:.2f}x",
    }


# ------------------------------------------------------- kerugian dan pricing
def hitung_expected_loss(pd_12bulan: float, lgd: float, ead: float) -> dict[str, Any]:
    """Expected loss dan komponennya."""
    for nama, nilai in (("pd_12bulan", pd_12bulan), ("lgd", lgd)):
        if not 0 <= float(nilai) <= 1:
            raise GalatMasukan(f"{nama} harus pecahan antara 0 dan 1, diterima {nilai}.")
    ead = _positif("ead", ead)
    el = float(pd_12bulan) * float(lgd) * ead
    return {
        "expected_loss": round(el, 2),
        "el_rate": round(el / ead, 6),
        "rumus": (
            f"EL = PD {float(pd_12bulan):.2%} x LGD {float(lgd):.2%} x EAD {ead:,.0f} "
            f"= {el:,.0f}"
        ),
    }


def usulkan_pricing(pd_12bulan: float, lgd: float) -> dict[str, Any]:
    """Suku bunga usulan = biaya dana + operasional + premi risiko + margin."""
    premi = float(pd_12bulan) * float(lgd)
    mentah = par.BIAYA_DANA + par.BIAYA_OPERASIONAL + premi + par.MARGIN_TARGET
    pricing = min(max(mentah, par.PRICING_MIN), par.PRICING_MAX)
    return {
        "pricing": round(pricing, 6),
        "pricing_sebelum_batas": round(mentah, 6),
        "premi_risiko": round(premi, 6),
        "terkena_batas": not (par.PRICING_MIN <= mentah <= par.PRICING_MAX),
        "rumus": (
            f"pricing = biaya dana {par.BIAYA_DANA:.2%} + operasional "
            f"{par.BIAYA_OPERASIONAL:.2%} + premi risiko {premi:.2%} + margin "
            f"{par.MARGIN_TARGET:.2%} = {mentah:.2%}, dibatasi ke {pricing:.2%}"
        ),
    }


def grade_dari_pd(pd_12bulan: float) -> dict[str, Any]:
    """Kelas rating internal dari PD 12 bulan."""
    nilai = float(pd_12bulan)
    if not 0 <= nilai <= 1:
        raise GalatMasukan(f"pd_12bulan harus pecahan antara 0 dan 1, diterima {nilai}.")
    grade = next(g for batas, g in par.BATAS_GRADE if nilai <= batas)
    return {
        "grade": grade,
        "pd_12bulan": round(nilai, 6),
        "rumus": f"PD {nilai:.2%} jatuh pada kelas {grade} menurut skala rating internal",
    }


# ----------------------------------------------------------- gerbang kebijakan
def periksa_batas_segmen(
    penjualan: float, plafon: float, saldo_rata_rata: float | None = None
) -> dict[str, Any]:
    """Uji apakah pengajuan berada di dalam definisi segmen komersial."""
    uji = [
        ("Penjualan tahunan", float(penjualan), par.SEGMEN["penjualan_min"], par.SEGMEN["penjualan_maks"]),
        ("Plafon", float(plafon), par.SEGMEN["plafon_min"], par.SEGMEN["plafon_maks"]),
    ]
    if saldo_rata_rata is not None:
        uji.append(
            ("Saldo rata-rata", float(saldo_rata_rata), par.SEGMEN["saldo_min"], par.SEGMEN["saldo_maks"])
        )

    rincian = [
        {
            "dimensi": nama,
            "nilai": nilai,
            "batas_bawah": bawah,
            "batas_atas": atas,
            "lolos": bawah <= nilai <= atas,
        }
        for nama, nilai, bawah, atas in uji
    ]
    di_luar = [r["dimensi"] for r in rincian if not r["lolos"]]
    return {
        "lolos": not di_luar,
        "di_luar_batas": di_luar,
        "rincian": rincian,
        "rumus": "Batas segmen komersial: " + "; ".join(
            f"{r['dimensi']} {r['batas_bawah']:,.0f}-{r['batas_atas']:,.0f}" for r in rincian
        ),
    }


def periksa_bmpk(eksposur_grup_berjalan: float, limit_usulan: float) -> dict[str, Any]:
    """Sisa ruang batas maksimum pemberian kredit satu grup debitur."""
    berjalan = max(float(eksposur_grup_berjalan), 0.0)
    usulan = _positif("limit_usulan", limit_usulan)
    setelah = berjalan + usulan
    sisa = par.BATAS_BMPK_GRUP - setelah
    return {
        "batas_bmpk": par.BATAS_BMPK_GRUP,
        "eksposur_setelah_pencairan": round(setelah, 2),
        "sisa_ruang": round(sisa, 2),
        "lolos": sisa >= 0,
        "limit_maksimum_yang_masih_muat": round(max(par.BATAS_BMPK_GRUP - berjalan, 0.0), 2),
        "rumus": (
            f"Eksposur grup {berjalan:,.0f} + usulan {usulan:,.0f} = {setelah:,.0f} "
            f"terhadap batas {par.BATAS_BMPK_GRUP:,.0f}; sisa {sisa:,.0f}"
        ),
    }


def periksa_covenant(
    grade: str,
    dscr: float,
    der_total: float | None = None,
    der: float | None = None,
    interest_coverage: float | None = None,
) -> dict[str, Any]:
    """Bandingkan rasio terhadap ambang covenant kelas rating.

    `der_total` yang dipakai, bukan `der`. Ambang `der_maks` diturunkan dari
    `fact_covenant` di lapisan emas, tempat DER berarti total liabilitas dibagi
    ekuitas; menguji ambang itu dengan DER berbunga membandingkan dua besaran
    yang berbeda dan hasilnya selalu condong lolos. `der` tetap diterima
    sebagai cadangan - laporan tanpa baris total liabilitas memang ada - tetapi
    basis yang dipakai selalu ikut dicatat pada hasil.
    """
    grade = _grade_valid(grade)
    cov = par.COVENANT_PER_RATING[grade]

    if der_total is not None:
        der_uji, basis, label = float(der_total), "total_liabilitas", "DER total maksimum"
    elif der is not None:
        der_uji, basis, label = float(der), "utang_berbunga", "DER berbunga maksimum"
    else:
        raise GalatMasukan(
            "der_total wajib diisi (total liabilitas / ekuitas, dari "
            "hitung_rasio_keuangan). Bila laporan tidak memuat total "
            "liabilitas, kirim `der` sebagai gantinya."
        )

    butir = [
        {
            "covenant": label,
            "nilai": round(der_uji, 4),
            "ambang": cov["der_maks"],
            "arah": "maksimum",
            "lolos": der_uji <= cov["der_maks"],
        },
        {
            "covenant": "DSCR minimum",
            "nilai": round(float(dscr), 4),
            "ambang": cov["dscr_min"],
            "arah": "minimum",
            "lolos": float(dscr) >= cov["dscr_min"],
        },
    ]
    if interest_coverage is None:
        butir.append(
            {
                "covenant": "Interest coverage minimum",
                "nilai": None,
                "ambang": cov["icr_min"],
                "arah": "minimum",
                "lolos": False,
                "catatan": "Tidak terdefinisi (beban bunga nol atau tidak tersedia).",
            }
        )
    else:
        butir.append(
            {
                "covenant": "Interest coverage minimum",
                "nilai": round(float(interest_coverage), 4),
                "ambang": cov["icr_min"],
                "arah": "minimum",
                "lolos": float(interest_coverage) >= cov["icr_min"],
            }
        )

    langgar = [b["covenant"] for b in butir if not b["lolos"]]
    return {
        "grade": grade,
        "frekuensi_uji": cov["uji"],
        "butir": butir,
        "basis_der": basis,
        "lolos": not langgar,
        "dilanggar": langgar,
        "rumus": (
            f"Ambang covenant kelas {grade} atas DER basis {basis}: {cov}"
            + ("" if basis == "total_liabilitas"
               else " (cadangan: total liabilitas tidak tersedia)")
        ),
    }


def kewenangan_komite(limit: float, grade: str) -> dict[str, Any]:
    """Komite pemutus dari besaran limit, dinaikkan satu tingkat untuk rating rendah."""
    grade = _grade_valid(grade)
    limit = _positif("limit", limit)
    komite = next(
        (nama for batas, nama in par.MATRIKS_KEWENANGAN if limit <= batas),
        par.MATRIKS_KEWENANGAN[-1][1],
    )
    dinaikkan = False
    if grade in ("B", "CCC") and komite != par.MATRIKS_KEWENANGAN[-1][1]:
        komite = par.MATRIKS_KEWENANGAN[-1][1]
        dinaikkan = True
    return {
        "komite_pemutus": komite,
        "dinaikkan_karena_rating": dinaikkan,
        "rumus": (
            f"Limit {limit:,.0f} pada kelas {grade} -> {komite}"
            + (" (dinaikkan karena rating di bawah BB)" if dinaikkan else "")
        ),
    }


def _grade_valid(grade: str) -> str:
    g = str(grade).strip().upper()
    if g not in par.COVENANT_PER_RATING:
        raise GalatMasukan(
            f"grade {grade!r} tidak dikenal. Pilihan: {', '.join(par.COVENANT_PER_RATING)}"
        )
    return g

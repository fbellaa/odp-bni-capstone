"""Mesin skoring tiruan untuk live demo — segmen kredit komersial.

CATATAN PENTING
---------------
Seluruh perhitungan di sini adalah *placeholder deterministik*, bukan model.
Tujuannya hanya agar halaman what-if benar-benar responsif terhadap masukan
selama model asli (PD / LGD / network risk) belum tersedia.

Sebagian angka kebijakan di bawah sudah tidak lagi dipakai apa adanya:
`lib/parameter_kebijakan.py` menurunkannya dari lapisan emas (recovery agunan
dari `fact_agunan`, ambang covenant dari `fact_covenant`, matriks kewenangan
dari `fact_pengajuan`) lalu menaruhnya di `KALIBRASI`. Konstanta yang tertulis
di sini tinggal cadangan untuk lingkungan tanpa data emas, dan pembacaannya
selalu lewat fungsi pengakses supaya jelas mana yang terpakai.

Ketika layanan FastAPI sudah jalan, ganti isi fungsi di modul ini dengan
pemanggilan endpoint `/api/score_pd`, `/api/estimate_lgd`,
`/api/recommend_limit_pricing`, `/api/score_network_risk`, dan
`/api/check_credit_policy`. Antarmuka fungsi (nama argumen dan bentuk keluaran)
sengaja dibuat sama dengan tool agen pada proposal bagian 5.1 supaya halaman
tidak perlu diubah.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Batas segmen komersial (proposal 3.5)
# --------------------------------------------------------------------------
SEGMEN = {
    "penjualan_min": 30e9,
    "penjualan_maks": 300e9,
    "plafon_min": 10e9,
    "plafon_maks": 150e9,
    "saldo_min": 10e9,
    "saldo_maks": 50e9,
}

# Batas bawah dan atas pricing tetap kebijakan internal. Tumpukan biaya dana,
# biaya operasional, dan margin target yang dulu menyusun suku bunga sudah tidak
# dipakai: pricing sekarang dibaca dari grid per pita pada `PRICING_PER_PITA`,
# yang dikalibrasi ke `pricing_bps` historis. Tumpukan itu memberi harga 116
# sampai 618 bps di atas praktik buku ini.
PRICING_MIN = 0.0600
PRICING_MAX = 0.1600

# Parameter kebijakan hasil kalibrasi atas data emas. Diisi sekali per proses
# oleh `parameter_kebijakan.terapkan()`; selama kosong, seluruh pengakses jatuh
# ke konstanta di bawah. Modul ini sengaja TIDAK mengimpor kalibratornya -
# ketergantungannya satu arah, dan mesin tetap jalan tanpa data emas.
KALIBRASI: dict = {}

# Batas maksimum pemberian kredit satu grup debitur, cadangan untuk lingkungan
# tanpa `fact_eksposur_grup`. Angka nyata datang per grup lewat pengajuan.
BATAS_BMPK_GRUP = 750e9

# Recovery rate per jenis agunan komersial -> LGD = 1 - recovery, disesuaikan
# tingkat pertanggungan.
RECOVERY_AGUNAN = {
    "Tanpa agunan (clean basis)": 0.10,
    "Piutang dagang (fidusia)": 0.35,
    "Persediaan (fidusia)": 0.40,
    "Mesin dan peralatan": 0.48,
    "Penjaminan korporasi grup": 0.45,
    "Tanah dan bangunan pabrik (SHM/SHGB)": 0.78,
    "Deposito / cash collateral": 0.95,
}

JENIS_FASILITAS = [
    "Modal kerja - rekening koran",
    "Modal kerja - demand loan",
    "Investasi - term loan",
    "Trade finance - LC impor",
    "Bank garansi proyek",
]

# Fasilitas revolving tidak beramortisasi: kewajiban berjalannya hanya bunga,
# sehingga kapasitas arus kas tidak boleh diuji memakai angsuran anuitas.
FASILITAS_REVOLVING = {
    "Modal kerja - rekening koran",
    "Trade finance - LC impor",
    "Bank garansi proyek",
}

# Pita risiko model PD. Menggantikan kelas rating huruf yang dulu diturunkan
# dari PD lewat ambang buatan modul ini sendiri: hurufnya tidak pernah datang
# dari model mana pun, sementara pita ini adalah kontrak artefak PD
# (`pd_decision_policy.json`, empat pita atas kuantil skor portofolio).
#
# Empat, bukan tiga. Melebur "sangat tinggi" ke "tinggi" akan menyatukan 142
# pengajuan bertingkat default 47% dengan 639 pengajuan bertingkat default 7%.
URUTAN_PITA = [
    "Risiko rendah",
    "Risiko sedang",
    "Risiko tinggi",
    "Risiko sangat tinggi",
]

# Ambang cadangan bila kalibrasi belum dipasang: kuantil skor model atas
# portofolio pengembangan, sama dengan isi berkas kebijakan PD.
CUTOFF_PITA_BAWAAN = {"q50": 0.0762, "q80": 0.2921, "q95": 0.7060}

# Pita dibaca sebagai satu tanjakan warna, kembar dengan palet lib/tampilan.py.
WARNA_PITA = {
    "Risiko rendah": "#2A8080",
    "Risiko sedang": "#40C0C0",
    "Risiko tinggi": "#FF8000",
    "Risiko sangat tinggi": "#7A3C00",
}

# Pagu limit per pita. Cadangan; angka sebenarnya datang dari persentil 95
# plafon yang pernah disetujui pada tiap pita (lihat parameter_kebijakan).
PAGU_PER_PITA = {
    "Risiko rendah": 90e9,
    "Risiko sedang": 90e9,
    "Risiko tinggi": 80e9,
    "Risiko sangat tinggi": 70e9,
}

# Suku bunga dasar per pita. Cadangan; angka sebenarnya median `pricing_bps`
# yang pernah ditagih pada tiap pita.
PRICING_PER_PITA = {
    "Risiko rendah": 0.0726,
    "Risiko sedang": 0.0801,
    "Risiko tinggi": 0.0872,
    "Risiko sangat tinggi": 0.0906,
}

# Covenant keuangan wajib per pita. Cadangan; ambang sebenarnya dari
# `fact_covenant` yang dijoin ke pita.
COVENANT_PER_PITA = {
    "Risiko rendah": {"der_maks": 2.00, "icr_min": 3.00, "dscr_min": 1.25, "uji": "Semesteran"},
    "Risiko sedang": {"der_maks": 2.50, "icr_min": 2.50, "dscr_min": 1.25, "uji": "Triwulanan"},
    "Risiko tinggi": {"der_maks": 3.00, "icr_min": 2.00, "dscr_min": 1.35, "uji": "Triwulanan"},
    "Risiko sangat tinggi": {"der_maks": 3.50, "icr_min": 1.50, "dscr_min": 1.50, "uji": "Bulanan"},
}

# Tingkat pertanggungan agunan minimum per pita. Ini kebijakan internal dan
# TIDAK diturunkan dari data: lapisan emas hanya memuat pertanggungan yang
# menyertai fasilitas (median 1,33 dan praktis rata di seluruh pita), bukan
# ambang minimum yang disyaratkan.
COVERAGE_MIN = {
    "Risiko rendah": 1.10,
    "Risiko sedang": 1.25,
    "Risiko tinggi": 1.35,
    "Risiko sangat tinggi": 1.50,
}

# Matriks kewenangan komite komersial berdasarkan besaran limit.
MATRIKS_KEWENANGAN = [
    (25e9, "Komite Kredit Wilayah"),
    (75e9, "Komite Kredit Komersial"),
    (150e9, "Komite Kredit Komersial Pusat"),
]

DSCR_MIN_KEBIJAKAN = 1.25


# --------------------------------------------------------------------------
# Pengakses parameter kebijakan
# --------------------------------------------------------------------------
# Seluruh perhitungan di bawah membaca kebijakan lewat empat fungsi ini, bukan
# langsung ke konstanta. Dengan begitu satu tempat saja yang tahu apakah sebuah
# angka datang dari tabel atau dari asumsi bawaan.
def recovery_agunan(jenis: str) -> float:
    peta = KALIBRASI.get("recovery_agunan") or RECOVERY_AGUNAN
    return float(peta.get(jenis, RECOVERY_AGUNAN.get(jenis, 0.20)))


def covenant_untuk(pita: str) -> dict:
    peta = KALIBRASI.get("covenant_per_pita") or COVENANT_PER_PITA
    return dict(peta.get(pita, COVENANT_PER_PITA[pita]))


def pricing_dasar(pita: str) -> float:
    """Suku bunga dasar pita, sebelum batas bawah dan atas kebijakan."""
    peta = KALIBRASI.get("pricing_per_pita") or PRICING_PER_PITA
    return float(peta.get(pita, PRICING_PER_PITA[pita]))


def pagu_untuk(pita: str) -> float:
    peta = KALIBRASI.get("pagu_per_pita") or PAGU_PER_PITA
    return float(peta.get(pita, PAGU_PER_PITA[pita]))


def cutoff_pita() -> dict:
    return dict(KALIBRASI.get("cutoffs_pd") or CUTOFF_PITA_BAWAAN)


def matriks_kewenangan() -> list:
    return list(KALIBRASI.get("matriks_kewenangan") or MATRIKS_KEWENANGAN)


def batas_bmpk(pengajuan: dict | None = None) -> float:
    """Batas BMPK grup: dari pengajuan bila grupnya sudah teridentifikasi."""
    if pengajuan:
        nilai = pengajuan.get("batas_bmpk_rp")
        if nilai:
            return float(nilai)
    return float(BATAS_BMPK_GRUP)

# Batas sumbangan satu fitur terhadap log-odds (meniru binning scorecard WOE).
BATAS_DAMPAK = 0.60


@dataclass
class Kontribusi:
    fitur: str
    nilai: str
    dampak: float  # dalam satuan log-odds, positif = menaikkan risiko


@dataclass
class HasilSkor:
    pd: float
    # Nama `grade` dipertahankan supaya pembaca lama tidak pecah, tetapi isinya
    # kini nama pita risiko model PD, bukan huruf rating.
    grade: str
    lgd: float
    ead: float
    expected_loss: float
    pricing: float
    limit_usulan: float
    tenor_usulan: int
    dscr: float
    icr: float
    der: float
    debt_to_ebitda: float
    ebitda: float
    coverage_agunan: float
    angsuran: float
    komite_pemutus: str
    eksposur_grup: float
    ruang_bmpk: float
    # Batas BMPK yang benar-benar dipakai perhitungan ini: per grup dari
    # `fact_eksposur_grup` bila grupnya teridentifikasi, kalau tidak angka
    # cadangan. Ikut dibawa supaya tampilan dan memo tidak menebak sendiri.
    batas_bmpk: float = BATAS_BMPK_GRUP
    sumber_bmpk: str = "asumsi porsi segmen"
    kontribusi: list = field(default_factory=list)
    catatan: list = field(default_factory=list)
    covenant: dict = field(default_factory=dict)


def angsuran_anuitas(pokok: float, tenor_bulan: int, rate_tahunan: float) -> float:
    i = rate_tahunan / 12
    if tenor_bulan <= 0:
        return 0.0
    if i == 0:
        return pokok / tenor_bulan
    return pokok * i / (1 - (1 + i) ** (-tenor_bulan))


def estimate_lgd(jenis_agunan: str, nilai_agunan: float, plafon: float) -> float:
    """LGD = 1 - recovery efektif, dibatasi 5% sampai 90%."""
    recovery_dasar = recovery_agunan(jenis_agunan)
    coverage = 0.0 if plafon <= 0 else min(nilai_agunan / plafon, 1.5)
    recovery_efektif = recovery_dasar * min(coverage / 1.0, 1.0)
    return float(min(max(1 - recovery_efektif, 0.05), 0.90))


def komite_pemutus(limit: float, grade: str) -> str:
    """Tingkat komite yang berwenang memutus (proposal 5.3 — matriks kewenangan)."""
    matriks = matriks_kewenangan()
    tingkat = len(matriks) - 1
    for i, (batas, _) in enumerate(matriks):
        if limit <= batas:
            tingkat = i
            break
    # Dua pita teratas menaikkan kewenangan satu tingkat.
    if grade in (URUTAN_PITA[2], URUTAN_PITA[3]):
        tingkat = min(tingkat + 1, len(matriks) - 1)
    return matriks[tingkat][1]


def _turunan_keuangan(pengajuan: dict, plafon: float, pricing: float = 0.105) -> dict:
    """Rasio keuangan komersial yang dipakai bersama oleh skoring dan covenant."""
    penjualan = max(float(pengajuan["penjualan_tahunan"]), 1.0)
    ebitda_margin = float(pengajuan.get("ebitda_margin", 0.11))
    ebitda = penjualan * ebitda_margin

    utang_eksisting = float(pengajuan.get("utang_berbunga_eksisting", plafon * 0.6))
    revolving = pengajuan.get("jenis_fasilitas", JENIS_FASILITAS[0]) in FASILITAS_REVOLVING
    # Fasilitas revolving jarang tertarik penuh; baki debet yang diperhitungkan
    # adalah plafon dikali tingkat pemakaian, bukan plafon utuh.
    terpakai = plafon * float(pengajuan.get("utilisasi_plafon", 0.68)) if revolving else plafon
    utang_total = utang_eksisting + terpakai
    debt_to_ebitda = utang_total / max(ebitda, 1.0)

    beban_bunga = utang_total * pricing
    icr = ebitda / max(beban_bunga, 1.0)

    tenor = max(int(pengajuan["tenor_bulan"]), 1)
    angsuran = (plafon * pricing / 12) if revolving else angsuran_anuitas(plafon, tenor, pricing)
    konversi = float(pengajuan.get("konversi_ebitda_kas", 0.75))
    dscr = (ebitda * konversi) / max(angsuran * 12, 1.0)

    return {
        "revolving": revolving,
        "penjualan": penjualan,
        "ebitda": ebitda,
        "utang_total": utang_total,
        "debt_to_ebitda": debt_to_ebitda,
        "icr": icr,
        "dscr": dscr,
        "angsuran": angsuran,
        "rasio_plafon_penjualan": plafon / penjualan,
    }


def score_pd(pengajuan: dict) -> tuple[float, list[Kontribusi]]:
    """Model PD tiruan: logistik atas tiga blok fitur (proposal bagian 6).

    Blok 1 rasio laporan keuangan, blok 2 perilaku fasilitas dan relasi
    perbankan, blok 3 struktur jaringan grup dan rantai pasok.
    """
    plafon = max(float(pengajuan["plafon"]), 1.0)
    t = _turunan_keuangan(pengajuan, plafon)

    der = float(pengajuan.get("der", 1.8))
    utilisasi = float(pengajuan.get("utilisasi_plafon", 0.68))
    saldo_giro = float(pengajuan.get("saldo_giro_rata", plafon * 0.18))
    tahun_berdiri = float(pengajuan.get("umur_usaha_thn", 12.0))
    buyer_hhi = float(pengajuan.get("buyer_concentration_hhi", 0.30))
    supplier_hhi = float(pengajuan.get("supplier_concentration_hhi", 0.30))
    neighbor_default = float(pengajuan.get("neighbor_default_rate_1hop", 0.05))
    group_share = float(pengajuan.get("group_exposure_share", 0.35))
    network_risk = float(pengajuan.get("network_risk_score", 20)) / 100
    tenure = float(pengajuan.get("tenure_nasabah_thn", 4.0))

    kedalaman_giro = saldo_giro / plafon

    komponen = [
        # Blok 1 — rasio laporan keuangan
        Kontribusi("Interest coverage ratio", f"{t['icr']:.2f}x", -0.40 * (t["icr"] - 3.00)),
        Kontribusi("Debt to EBITDA", f"{t['debt_to_ebitda']:.2f}x", 0.30 * (t["debt_to_ebitda"] - 3.50)),
        Kontribusi("Debt to equity ratio", f"{der:.2f}x", 0.60 * (der - 1.80)),
        Kontribusi("Debt service coverage ratio", f"{t['dscr']:.2f}x", -0.70 * (t["dscr"] - 1.35)),
        Kontribusi("EBITDA margin", f"{pengajuan.get('ebitda_margin', 0.11) * 100:.1f}%",
                   -6.50 * (float(pengajuan.get("ebitda_margin", 0.11)) - 0.11)),
        Kontribusi("Konversi EBITDA ke kas",
                   f"{float(pengajuan.get('konversi_ebitda_kas', 0.75)) * 100:.0f}%",
                   -1.60 * (float(pengajuan.get("konversi_ebitda_kas", 0.75)) - 0.75)),
        Kontribusi("Rasio plafon terhadap penjualan tahunan",
                   f"{t['rasio_plafon_penjualan']:.2f}x", 1.90 * (t["rasio_plafon_penjualan"] - 0.35)),
        Kontribusi("Umur badan usaha", f"{tahun_berdiri:.0f} tahun", -0.045 * (tahun_berdiri - 12.0)),
        # Blok 2 — perilaku fasilitas dan kedalaman relasi
        Kontribusi("Tingkat pemakaian plafon", f"{utilisasi * 100:.0f}%", 1.35 * (utilisasi - 0.68)),
        Kontribusi("Saldo giro rata-rata terhadap plafon", f"{kedalaman_giro * 100:.0f}%",
                   -1.20 * (kedalaman_giro - 0.18)),
        Kontribusi("Lama menjadi nasabah", f"{tenure:.1f} tahun", -0.12 * (tenure - 4.0)),
        # Blok 3 — struktur jaringan (proposal 7.3)
        Kontribusi("Konsentrasi pembeli (HHI)", f"{buyer_hhi:.2f}", 1.45 * (buyer_hhi - 0.30)),
        Kontribusi("Konsentrasi pemasok (HHI)", f"{supplier_hhi:.2f}", 1.05 * (supplier_hhi - 0.30)),
        Kontribusi("Gagal bayar entitas 1-hop", f"{neighbor_default * 100:.1f}%",
                   3.10 * (neighbor_default - 0.05)),
        Kontribusi("Porsi eksposur grup terhadap BMPK", f"{group_share * 100:.0f}%",
                   1.15 * (group_share - 0.35)),
        Kontribusi("Skor risiko jaringan", f"{network_risk * 100:.0f}/100", 1.30 * (network_risk - 0.20)),
    ]

    # Setiap fitur dibatasi sumbangannya supaya satu rasio ekstrem tidak
    # mendominasi skor — meniru perilaku binning pada scorecard WOE.
    for k in komponen:
        k.dampak = float(min(max(k.dampak, -BATAS_DAMPAK), BATAS_DAMPAK))

    z = -3.05 + sum(k.dampak for k in komponen)
    pd = 1 / (1 + math.exp(-z))
    pd = float(min(max(pd, 0.0015), 0.60))
    komponen.sort(key=lambda k: abs(k.dampak), reverse=True)

    # Bila model PD sungguhan sudah menghitung angkanya (lihat
    # lib/model_nyata.py), angka itulah yang dipakai. Komponen di atas tetap
    # dikembalikan sebagai pembanding naratif, sedangkan reason code yang
    # ditampilkan halaman berasal dari SHAP model.
    if pengajuan.get("pd_model") is not None:
        pd = float(min(max(float(pengajuan["pd_model"]), 1e-4), 0.99))
    return pd, komponen


def pita_dari_skor(skor: float) -> str:
    """Pita risiko untuk satu skor model, memakai ambang berkas kebijakan PD."""
    c = cutoff_pita()
    if skor <= c["q50"]:
        return URUTAN_PITA[0]
    if skor <= c["q80"]:
        return URUTAN_PITA[1]
    if skor <= c["q95"]:
        return URUTAN_PITA[2]
    return URUTAN_PITA[3]


def recommend_limit_pricing(pengajuan: dict) -> HasilSkor:
    """Rantai perhitungan keputusan sesuai proposal bagian 3.2.

    Skor PD -> pita risiko -> pricing dari grid pita; lalu limit dibatasi oleh
    kapasitas arus kas, pagu limit pita, sisa ruang BMPK grup, dan tingkat
    pertanggungan agunan.

    `expected_loss` tetap dihitung dan dilaporkan, tetapi sebagai angka
    indikatif: PD di sini skor peringkat yang tidak terkalibrasi, jadi hasil
    kalinya dengan LGD dan EAD tidak boleh dibaca sebagai rupiah kerugian yang
    diharapkan. Ia tidak lagi masuk ke pricing.
    """
    pd, kontribusi = score_pd(pengajuan)
    plafon = float(pengajuan["plafon"])
    tenor = int(pengajuan["tenor_bulan"])
    lgd = pengajuan.get("lgd_model")
    lgd = float(lgd) if lgd is not None else estimate_lgd(
        pengajuan["jenis_agunan"], pengajuan.get("nilai_agunan", 0.0), plafon
    )
    ead = plafon
    el = pd * lgd * ead

    grade = pita_dari_skor(pd)
    cov = covenant_untuk(grade)

    # Pricing datang dari grid per pita - median suku bunga yang benar-benar
    # pernah ditagih pada pita itu - bukan dari tumpukan biaya dana + margin +
    # expected loss. Premi expected loss per pengajuan sengaja TIDAK
    # ditambahkan: skor PD artefak ini tidak terkalibrasi, jadi PD x LGD x EAD
    # tidak punya satuan rupiah yang bisa dipertanggungjawabkan.
    pricing = float(min(max(pricing_dasar(grade), PRICING_MIN), PRICING_MAX))

    t = _turunan_keuangan(pengajuan, plafon, pricing)

    # Batas 1 — kapasitas arus kas pada DSCR minimum kebijakan.
    angsuran_maks = (t["ebitda"] * float(pengajuan.get("konversi_ebitda_kas", 0.75))) / (
        cov["dscr_min"] * 12
    )
    if t["revolving"]:
        plafon_kapasitas = angsuran_maks * 12 / max(pricing, 1e-6)
    else:
        i = pricing / 12
        plafon_kapasitas = (
            angsuran_maks * (1 - (1 + i) ** (-tenor)) / i if i > 0 else angsuran_maks * tenor
        )

    # Batas 2 — pagu limit menurut pita risiko.
    pagu = pagu_untuk(grade)

    # Batas 3 — sisa ruang BMPK grup. Bila grup pemohon sudah teridentifikasi,
    # eksposur dan batasnya datang dari `fact_eksposur_grup` lewat pengajuan;
    # porsi asumsi hanya dipakai kalau tidak ada grup yang bisa ditunjuk.
    batas_grup = batas_bmpk(pengajuan)
    eksposur_nyata = pengajuan.get("eksposur_grup_rp")
    if eksposur_nyata is not None:
        eksposur_grup = float(eksposur_nyata)
        group_share = eksposur_grup / batas_grup if batas_grup else 0.0
    else:
        group_share = float(pengajuan.get("group_exposure_share", 0.35))
        eksposur_grup = batas_grup * group_share
    ruang_bmpk = max(batas_grup - eksposur_grup, 0.0)

    # Batas 4 — tingkat pertanggungan agunan minimum per rating.
    nilai_agunan = float(pengajuan.get("nilai_agunan", 0.0))
    coverage_min = COVERAGE_MIN[grade]
    plafon_agunan = (
        nilai_agunan / coverage_min if nilai_agunan > 0 else min(plafon, SEGMEN["plafon_min"])
    )

    limit = float(min(plafon, plafon_kapasitas, pagu, ruang_bmpk, plafon_agunan))
    limit = max(math.floor(limit / 1e9) * 1e9, 0.0)

    angsuran = (
        max(limit, 1.0) * pricing / 12 if t["revolving"]
        else angsuran_anuitas(max(limit, 1.0), tenor, pricing)
    )
    coverage = 0.0 if limit <= 0 else nilai_agunan / limit

    # Rasio yang dilaporkan dihitung pada limit usulan. Bila limit nol (usulan
    # ditolak), rasio tetap dilaporkan pada plafon yang diminta supaya angkanya
    # bermakna dan tidak meledak karena pembagi mendekati nol.
    tt = _turunan_keuangan(pengajuan, limit if limit > 0 else plafon, pricing)
    der = float(pengajuan.get("der", 1.8))

    catatan = []
    if limit < plafon:
        pembatas = min(
            [(plafon_kapasitas, "kapasitas arus kas"), (pagu, f"pagu limit pita {grade.lower()}"),
             (ruang_bmpk, "sisa ruang BMPK grup"), (plafon_agunan, "pertanggungan agunan")],
            key=lambda x: x[0],
        )[1]
        catatan.append(f"Limit diturunkan dari permintaan karena batas {pembatas}.")
    if float(pengajuan.get("network_risk_score", 0)) >= 60:
        catatan.append(
            "Skor risiko jaringan tinggi — wajib penelusuran pemilik manfaat dan verifikasi "
            "rekening penerima pencairan sebelum akad."
        )
    if group_share >= 0.75:
        catatan.append(
            "Eksposur grup sudah melewati 75% batas BMPK — setiap tambahan fasilitas "
            "wajib disertai perhitungan gabungan satu grup debitur."
        )
    if pd > 0.095:
        catatan.append(
            f"Pengajuan berada pada pita {grade.lower()} — keputusan naik satu tingkat "
            "kewenangan dan covenant diuji dengan frekuensi lebih rapat."
        )
    if lgd > 0.70:
        catatan.append("Pertanggungan agunan tipis; pertimbangkan tambahan agunan atau penjaminan korporasi.")
    if der > cov["der_maks"]:
        catatan.append(
            f"DER {_x(der)} sudah di atas ambang covenant pita {grade.lower()} "
            f"({_x(cov['der_maks'])})."
        )

    return HasilSkor(
        pd=pd, grade=grade, lgd=lgd, ead=ead, expected_loss=el, pricing=pricing,
        limit_usulan=limit, tenor_usulan=tenor, dscr=tt["dscr"], icr=tt["icr"], der=der,
        debt_to_ebitda=tt["debt_to_ebitda"], ebitda=t["ebitda"], coverage_agunan=coverage,
        angsuran=angsuran, komite_pemutus=komite_pemutus(limit, grade),
        eksposur_grup=eksposur_grup, ruang_bmpk=ruang_bmpk,
        batas_bmpk=batas_grup,
        sumber_bmpk=("fact_eksposur_grup" if eksposur_nyata is not None
                     else "asumsi porsi segmen"),
        kontribusi=kontribusi, catatan=catatan, covenant=cov,
    )


# --------------------------------------------------------------------------
# Gerbang kepatuhan (proposal 5.3)
# --------------------------------------------------------------------------
def _x(nilai: float) -> str:
    """Rasio dengan satuan x dan koma desimal Indonesia, untuk teks kepatuhan."""
    return f"{nilai:.2f}x".replace(".", ",")


LOLOS = "LOLOS"
PENYESUAIAN = "PERLU PENYESUAIAN"
TELAAH = "PENELAAHAN LANJUTAN"


def check_credit_policy(hasil: HasilSkor, pengajuan: dict) -> list[dict]:
    """Tiruan `check_credit_policy(aspek, konteks)` — satu panggilan per aspek.

    Rekomendasi yang gagal pada gerbang ini tidak pernah ditampilkan sebagai
    usulan setuju; sistem menampilkannya sebagai *perlu penyesuaian* beserta
    dasar aturannya.

    Tiap butir membawa `aturan`, yaitu kalimat kebijakan internal yang benar-benar
    dipakai kode di bawah — ambang segmen, matriks kewenangan, batas BMPK,
    pertanggungan agunan, dan covenant per rating. Yang TIDAK ada di sini adalah
    nomor pasalnya: butir keluar dari fungsi ini dengan `pasal = None`, dan
    `lib/kebijakan.lampirkan_rujukan()` yang mengisinya dari korpus terindeks
    bila memang ada padanannya. Sebelumnya tiap butir membawa nomor karangan
    seperti "KKK-08.2" — terbaca seperti pasal yang bisa dibuka, padahal tidak
    ada berkasnya.
    """
    plafon = float(pengajuan["plafon"])
    penjualan = float(pengajuan["penjualan_tahunan"])
    cov = hasil.covenant
    coverage_min = COVERAGE_MIN[hasil.grade]
    aspek = []

    # 1 — batas segmen komersial
    dalam_segmen = (
        SEGMEN["penjualan_min"] <= penjualan <= SEGMEN["penjualan_maks"]
        and SEGMEN["plafon_min"] <= plafon <= SEGMEN["plafon_maks"]
    )
    aspek.append({
        "aspek": "Batas segmen",
        "status": LOLOS if dalam_segmen else PENYESUAIAN,
        "temuan": f"Penjualan Rp {penjualan / 1e9:.0f} M, plafon Rp {plafon / 1e9:.0f} M",
        "aturan": "Segmen komersial mencakup debitur dengan penjualan tahunan Rp 30 miliar "
                  "sampai Rp 300 miliar dan plafon Rp 10 miliar sampai Rp 150 miliar.",
        "tindakan": "Sesuai lingkup segmen." if dalam_segmen
                    else "Alihkan pengajuan ke segmen UMKM atau korporasi.",
    })

    # 2 — kewenangan komite
    aspek.append({
        "aspek": "Kewenangan",
        "status": LOLOS if hasil.limit_usulan > 0 else PENYESUAIAN,
        "temuan": f"Limit Rp {hasil.limit_usulan / 1e9:.0f} M, pita {hasil.grade.lower()} "
                  f"-> {hasil.komite_pemutus}",
        "aturan": "Fasilitas di atas Rp 75 miliar atau berating di bawah BBB diputus oleh "
                  "Komite Kredit Komersial Pusat.",
        "tindakan": "Ajukan ke tingkat komite di atas." if hasil.limit_usulan <= 0
                    else "Diputus pada tingkat komite tersebut.",
    })

    # 3 — BMPK grup
    porsi = float(pengajuan.get("group_exposure_share", 0.35))
    status_bmpk = LOLOS if porsi < 0.85 else PENYESUAIAN
    aspek.append({
        "aspek": "BMPK grup",
        "status": status_bmpk,
        "temuan": f"Eksposur grup {porsi * 100:.0f}% batas · sisa ruang "
                  f"Rp {hasil.ruang_bmpk / 1e9:.0f} M",
        "aturan": "Eksposur gabungan satu grup debitur tidak melampaui batas maksimum "
                  "pemberian kredit; sisa ruang dicatat pada setiap usulan.",
        "tindakan": "Lolos, sisa ruang dicatat." if status_bmpk == LOLOS
                    else "Turunkan limit atau lunasi fasilitas grup lain lebih dulu.",
    })

    # 4 — agunan
    status_agunan = LOLOS if hasil.coverage_agunan >= coverage_min else PENYESUAIAN
    limit_patuh = (
        float(pengajuan.get("nilai_agunan", 0.0)) / coverage_min if coverage_min else hasil.limit_usulan
    )
    aspek.append({
        "aspek": "Agunan",
        "status": status_agunan,
        "temuan": f"Coverage {hasil.coverage_agunan * 100:.0f}% vs minimum "
                  f"{coverage_min * 100:.0f}%",
        "aturan": "Rasio pertanggungan agunan minimum ditetapkan per pita risiko; "
                  "penjaminan silang tidak boleh dihitung ganda.",
        "tindakan": "Pertanggungan memadai." if status_agunan == LOLOS
                    else f"Tambah agunan atau turunkan limit ke Rp {limit_patuh / 1e9:.0f} M "
                         f"agar coverage {coverage_min * 100:.0f}%.",
    })

    # 5 — covenant
    pelanggaran = []
    if hasil.der > cov["der_maks"]:
        pelanggaran.append(f"DER {_x(hasil.der)} > maks {_x(cov['der_maks'])}")
    if hasil.icr < cov["icr_min"]:
        pelanggaran.append(f"ICR {_x(hasil.icr)} < min {_x(cov['icr_min'])}")
    if hasil.dscr < cov["dscr_min"]:
        pelanggaran.append(f"DSCR {_x(hasil.dscr)} < min {_x(cov['dscr_min'])}")
    aspek.append({
        "aspek": "Covenant",
        "status": LOLOS if not pelanggaran else PENYESUAIAN,
        "temuan": "; ".join(pelanggaran) if pelanggaran
                  else f"DER maks {_x(cov['der_maks'])} · ICR min {_x(cov['icr_min'])} · "
                       f"DSCR min {_x(cov['dscr_min'])}",
        "aturan": f"Pita {hasil.grade.lower()} wajib memuat covenant DER maksimum "
                  f"{_x(cov['der_maks'])} dan ICR minimum {_x(cov['icr_min'])}, "
                  f"diuji {cov['uji'].lower()}.",
        "tindakan": f"Cantumkan sebagai covenant wajib, uji {cov['uji'].lower()}."
                    if not pelanggaran
                    else "Rasio berjalan sudah melanggar ambang — perlu penyesuaian struktur fasilitas.",
    })

    # 6 — pihak terafiliasi
    afiliasi = int(pengajuan.get("jumlah_entitas_grup", 1))
    rangkap = bool(pengajuan.get("indikasi_rangkap_jabatan", False))
    status_afiliasi = TELAAH if (afiliasi >= 3 or rangkap) else LOLOS
    aspek.append({
        "aspek": "Afiliasi",
        "status": status_afiliasi,
        "temuan": f"{afiliasi} entitas satu grup"
                  + (", terdapat rangkap jabatan pengurus" if rangkap else ""),
        "aturan": "Penelusuran pemilik manfaat wajib dilakukan bila ditemukan atribut "
                  "identitas atau pengurus yang dipakai bersama antar badan hukum.",
        "tindakan": "Tidak ada pemicu penelaahan." if status_afiliasi == LOLOS
                    else "Penelaahan lanjutan wajib: telusuri pemilik manfaat akhir.",
    })

    # Kunci rujukan sengaja ada dan kosong: tampilan dan memo membedakan
    # "belum ditelusuri" dari "tidak ada padanan", dan keduanya bukan "patuh".
    for a in aspek:
        a.setdefault("pasal", None)
        a.setdefault("kutipan", None)
        a.setdefault("sumber", "internal")
    return aspek


def status_kepatuhan(gerbang: list[dict]) -> str:
    if any(a["status"] == PENYESUAIAN for a in gerbang):
        return PENYESUAIAN
    if any(a["status"] == TELAAH for a in gerbang):
        return TELAAH
    return LOLOS


def keputusan_dari_hasil(hasil: HasilSkor, gerbang: list[dict] | None = None) -> str:
    """Usulan keputusan setelah melewati gerbang kepatuhan.

    Ambangnya pita, bukan angka PD. Skor model ini tidak terkalibrasi, jadi
    "PD di atas 18,5%" tidak berarti apa-apa sebagai probabilitas - sementara
    "berada pada pita teratas portofolio" berarti sesuatu yang bisa diperiksa.
    """
    if hasil.limit_usulan <= 0 or hasil.grade == URUTAN_PITA[3]:
        return "TOLAK"
    if gerbang is not None and status_kepatuhan(gerbang) == PENYESUAIAN:
        return "PERLU PENYESUAIAN"
    if hasil.grade == URUTAN_PITA[2] or hasil.dscr < hasil.covenant.get(
            "dscr_min", DSCR_MIN_KEBIJAKAN):
        return "SETUJU DENGAN SYARAT"
    if gerbang is not None and status_kepatuhan(gerbang) == TELAAH:
        return "SETUJU DENGAN SYARAT"
    return "SETUJU"

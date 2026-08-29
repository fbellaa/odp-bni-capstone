"""Data dummy untuk live demo — segmen kredit komersial.

Seluruh isi modul ini sintetis dan dibangkitkan dengan seed tetap supaya
tampilan demo konsisten antar sesi. Tidak ada data nasabah sebenarnya.

Rentang penjualan, plafon, dan saldo rata-rata mengikuti batas segmen komersial
pada proposal bagian 3.5 (penjualan Rp 30-300 M, plafon Rp 10-150 M, saldo
rata-rata Rp 10-50 M).

Cara mengganti dengan data asli nanti: setiap fungsi di bawah ini adalah satu
titik sambung ke FastAPI. Ganti isinya dengan `requests.get(f"{API_URL}/...")`
dan pertahankan bentuk keluarannya (DataFrame / dict) agar halaman tidak
perlu disentuh.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from lib import mock_engine

SEED = 20260824

SEKTOR = [
    "Manufaktur komponen otomotif",
    "Distribusi bahan bangunan",
    "Kontraktor infrastruktur",
    "Pengolahan hasil perkebunan",
    "Manufaktur kemasan",
    "Perdagangan besar farmasi",
    "Logistik dan pergudangan",
    "Tekstil dan garmen",
]
WILAYAH = [
    "Karawang", "Cikarang", "Gresik", "Semarang",
    "Surabaya", "Medan", "Batam", "Makassar",
]
JENIS_FASILITAS = list(mock_engine.JENIS_FASILITAS)
JENIS_AGUNAN = list(mock_engine.RECOVERY_AGUNAN.keys())
KOLEKTIBILITAS = [
    "1 - Lancar", "2 - Dalam perhatian khusus", "3 - Kurang lancar",
    "4 - Diragukan", "5 - Macet",
]
POSISI_COVENANT = ["Patuh", "Perlu perhatian", "Terlanggar"]

# Kata kunci yang muncul pada narasi relationship manager untuk setiap sektor.
# Narasi jarang memakai nama sektor persis seperti pada dimensi warehouse
# ("distributor bahan bangunan" vs "Distribusi bahan bangunan").
ALIAS_SEKTOR = {
    "Manufaktur komponen otomotif": ["komponen otomotif", "otomotif", "manufaktur komponen"],
    "Distribusi bahan bangunan": ["bahan bangunan", "distribusi bangunan", "material bangunan"],
    "Kontraktor infrastruktur": ["kontraktor", "infrastruktur", "konstruksi"],
    "Pengolahan hasil perkebunan": ["perkebunan", "hasil perkebunan", "sawit", "agribisnis"],
    "Manufaktur kemasan": ["kemasan", "packaging"],
    "Perdagangan besar farmasi": ["farmasi", "obat", "alat kesehatan"],
    "Logistik dan pergudangan": ["logistik", "pergudangan", "ekspedisi"],
    "Tekstil dan garmen": ["tekstil", "garmen", "konveksi"],
}

# Kata kunci agunan komersial. Diperiksa berurutan; "bangunan" sengaja tidak
# berdiri sendiri supaya frasa "bahan bangunan" tidak tertangkap sebagai agunan.
KUNCI_AGUNAN = [
    ("Deposito / cash collateral", ["deposito", "cash collateral"]),
    ("Tanah dan bangunan pabrik (SHM/SHGB)",
     ["tanah", "pabrik", "shm", "shgb", "hak tanggungan", "gedung", "gudang milik"]),
    ("Mesin dan peralatan", ["mesin", "peralatan", "alat berat"]),
    ("Persediaan (fidusia)", ["persediaan", "inventory", "stok barang"]),
    ("Piutang dagang (fidusia)", ["piutang", "receivable", "tagihan"]),
    ("Penjaminan korporasi grup", ["penjaminan korporasi", "corporate guarantee", "penjaminan grup"]),
]

NAMA_GRUP = [
    "Grup Sumber Logam", "Grup Andalan Niaga", "Grup Cakra Sentosa",
    "Grup Bumi Perkasa", "Grup Mitra Pratama", "Grup Karya Nusantara",
    "Grup Wana Agro", "Grup Samudra Cipta", "Grup Prima Kemasan",
    "Grup Trilogi Mandiri", "Grup Bahtera Jaya", "Grup Adhi Persada",
]

# Pola anomali struktur pada segmen komersial (proposal 7.2 D).
POLA_ANOMALI = {
    "shared_attribute": "Beberapa badan hukum berbagi alamat domisili, pengurus, atau rekening pencairan",
    "circular_payment": "Siklus transaksi melingkar antar pihak berelasi yang menaikkan pendapatan secara artifisial",
    "cross_guarantee_chain": "Penjaminan silang berantai antar afiliasi sehingga nilai agunan berpotensi dihitung ganda",
    "transfer_spike": "Lonjakan transfer antar entitas satu grup sesaat sebelum tanggal laporan keuangan",
    "layered_ownership": "Struktur kepemilikan berlapis melalui entitas tanpa aktivitas usaha",
    "undisclosed_affiliation": "Afiliasi tidak dinyatakan — dua debitur secara topologi berada dalam satu kendali",
}


def _rng(salt: str = "") -> np.random.Generator:
    benih = SEED + int(hashlib.md5(salt.encode()).hexdigest()[:8], 16) % 100_000
    return np.random.default_rng(benih)


# --------------------------------------------------------------------------
# Portofolio pengajuan komersial
# --------------------------------------------------------------------------
def daftar_pengajuan(n: int = 180) -> pd.DataFrame:
    rng = _rng("pengajuan")
    sektor = rng.choice(SEKTOR, n)
    wilayah = rng.choice(WILAYAH, n)
    baris = []
    for idx in range(n):
        penjualan = float(np.clip(rng.lognormal(mean=25.1, sigma=0.55), 30e9, 300e9))
        plafon = float(np.clip(penjualan * rng.uniform(0.10, 0.32), 10e9, 150e9))
        tenor = int(rng.choice([12, 24, 36, 48, 60, 84], p=[0.14, 0.24, 0.26, 0.18, 0.12, 0.06]))
        fasilitas = str(rng.choice(JENIS_FASILITAS, p=[0.30, 0.16, 0.26, 0.16, 0.12]))
        agunan = str(rng.choice(JENIS_AGUNAN, p=[0.06, 0.14, 0.14, 0.18, 0.10, 0.32, 0.06]))
        nilai_agunan = plafon * float(rng.uniform(0.7, 2.0)) if "Tanpa agunan" not in agunan else 0.0
        network_risk = float(np.clip(rng.beta(1.8, 6.0) * 100, 0, 100))
        grup = str(rng.choice(NAMA_GRUP))
        pengajuan = dict(
            penjualan_tahunan=penjualan,
            plafon=plafon,
            tenor_bulan=tenor,
            ebitda_margin=float(np.clip(rng.normal(0.115, 0.032), 0.04, 0.26)),
            der=float(np.clip(rng.gamma(6.0, 0.30), 0.35, 4.2)),
            utang_berbunga_eksisting=plafon * float(rng.uniform(0.10, 0.90)),
            konversi_ebitda_kas=float(np.clip(rng.normal(0.74, 0.13), 0.30, 0.98)),
            utilisasi_plafon=float(np.clip(rng.beta(6.0, 3.0), 0.15, 0.99)),
            saldo_giro_rata=float(np.clip(rng.uniform(10e9, 50e9), 10e9, 50e9)),
            umur_usaha_thn=float(np.clip(rng.gamma(5.0, 3.0), 2, 45)),
            jenis_agunan=agunan,
            nilai_agunan=nilai_agunan,
            buyer_concentration_hhi=float(np.clip(rng.beta(2.4, 3.6), 0.06, 0.97)),
            supplier_concentration_hhi=float(np.clip(rng.beta(2.5, 4.0), 0.05, 0.98)),
            neighbor_default_rate_1hop=float(np.clip(rng.beta(1.6, 18.0), 0.0, 0.6)),
            group_exposure_share=float(np.clip(rng.beta(2.6, 3.4), 0.03, 0.97)),
            network_risk_score=network_risk,
            tenure_nasabah_thn=float(np.clip(rng.gamma(2.6, 2.0), 0.0, 25)),
            jumlah_entitas_grup=int(rng.integers(1, 8)),
            jenis_fasilitas=fasilitas,
        )
        hasil = mock_engine.recommend_limit_pricing(pengajuan)
        gerbang = mock_engine.check_credit_policy(hasil, pengajuan)
        baris.append(
            {
                "application_id": f"APP-2026-{idx + 1001}",
                "cif": f"CIF{rng.integers(10_000_000, 99_999_999)}",
                "nama_debitur": _nama_badan_hukum(rng, sektor[idx], idx),
                "grup_usaha": grup,
                "sektor": sektor[idx],
                "wilayah": wilayah[idx],
                "jenis_fasilitas": fasilitas,
                "tanggal": pd.Timestamp("2026-08-24") - pd.Timedelta(days=int(rng.integers(0, 180))),
                "penjualan_tahunan": penjualan,
                "ebitda_margin": pengajuan["ebitda_margin"],
                "der": pengajuan["der"],
                "plafon_diminta": plafon,
                "tenor_bulan": tenor,
                "jenis_agunan": agunan,
                "nilai_agunan": nilai_agunan,
                "umur_usaha_thn": pengajuan["umur_usaha_thn"],
                "saldo_giro_rata": pengajuan["saldo_giro_rata"],
                "utilisasi_plafon": pengajuan["utilisasi_plafon"],
                "pd": hasil.pd,
                "grade": hasil.grade,
                "lgd": hasil.lgd,
                "icr": hasil.icr,
                "dscr": hasil.dscr,
                "expected_loss": hasil.expected_loss,
                "pricing": hasil.pricing,
                "limit_usulan": hasil.limit_usulan,
                "komite_pemutus": hasil.komite_pemutus,
                "group_exposure_share": pengajuan["group_exposure_share"],
                "network_risk_score": network_risk,
                "community_id": int(rng.integers(1, 13)),
                "kolektibilitas": str(rng.choice(KOLEKTIBILITAS, p=[0.82, 0.10, 0.04, 0.02, 0.02])),
                "posisi_covenant": str(rng.choice(POSISI_COVENANT, p=[0.78, 0.16, 0.06])),
                "keputusan": mock_engine.keputusan_dari_hasil(hasil, gerbang),
            }
        )
    return pd.DataFrame(baris).sort_values("tanggal", ascending=False).reset_index(drop=True)


_KATA_NAMA = [
    "Sumber Logam Perkasa", "Andalan Niaga Utama", "Cakra Sentosa Mandiri",
    "Bumi Perkasa Industri", "Mitra Pratama Sejahtera", "Karya Nusantara Abadi",
    "Wana Agro Lestari", "Samudra Cipta Persada", "Prima Kemasan Nusantara",
    "Trilogi Mandiri Jaya", "Bahtera Jaya Logistik", "Adhi Persada Konstruksi",
]


def _nama_badan_hukum(rng, sektor: str, idx: int) -> str:
    bentuk = str(rng.choice(["PT", "PT", "PT", "CV"]))
    return f"{bentuk} {_KATA_NAMA[idx % len(_KATA_NAMA)]} {idx + 1:03d}"


# --------------------------------------------------------------------------
# Lapisan graf (proposal 7.1)
# --------------------------------------------------------------------------
TIPE_SIMPUL = [
    "Badan hukum", "Grup usaha", "Pemilik manfaat", "Pengurus",
    "Counterparty", "Atribut berbagi", "Agunan",
]
TIPE_RELASI = [
    "memiliki", "mengendalikan", "menjabat_di", "memasok",
    "menjual_ke", "menjamin_silang", "berbagi_atribut", "satu_sektor",
]

_PREFIKS_SIMPUL = {
    "Badan hukum": "BH", "Grup usaha": "GRP", "Pemilik manfaat": "PM",
    "Pengurus": "PNG", "Counterparty": "CP", "Atribut berbagi": "ATR",
    "Agunan": "AGN",
}


def subgraf_ego(entity_id: str, hops: int = 2, batas_simpul: int = 60):
    """Tiruan `get_entity_network(entity_id, hops, min_weight)`.

    Mengembalikan (nodes, edges). Ukuran subgraf dibatasi sesuai catatan
    proposal 9.5 supaya graf besar tidak dikirim ke antarmuka.
    """
    rng = _rng(f"graf-{entity_id}-{hops}")
    komunitas_pusat = int(rng.integers(1, 13))
    grup = str(rng.choice(NAMA_GRUP))

    nodes = [{
        "id": entity_id, "label": entity_id, "tipe": "Badan hukum", "hop": 0,
        "community_id": komunitas_pusat, "pd": float(rng.beta(2, 60)),
        "grup": grup, "peran": "Debitur pengaju",
    }]
    edges = []

    # Hop 1 selalu memuat tulang punggung struktur komersial: grup usaha,
    # pemilik manfaat, pengurus, dan agunan yang diikat.
    inti = [
        ("Grup usaha", "mengendalikan", "Induk grup usaha"),
        ("Pemilik manfaat", "memiliki", "Pemegang saham pengendali"),
        ("Pengurus", "menjabat_di", "Direktur utama"),
        ("Agunan", "menjamin_silang", "Agunan yang diikat"),
    ]
    hop1 = []
    for tipe, relasi, peran in inti:
        nid = f"{_PREFIKS_SIMPUL[tipe]}-{rng.integers(1000, 9999)}"
        nodes.append({
            "id": nid, "label": nid, "tipe": tipe, "hop": 1,
            "community_id": komunitas_pusat, "pd": float("nan"),
            "grup": grup, "peran": peran,
        })
        edges.append({
            "source": entity_id if tipe == "Agunan" else nid,
            "target": nid if tipe == "Agunan" else entity_id,
            "relasi": relasi,
            "bobot": float(np.clip(rng.lognormal(24.0, 0.7), 1e9, 4e11)),
        })
        hop1.append(nid)

    # Sisanya counterparty dagang dan badan hukum afiliasi.
    peran_cp = ["Pemasok utama", "Pembeli utama", "Distributor", "Subkontraktor"]
    for _ in range(int(rng.integers(4, 8))):
        if len(nodes) >= batas_simpul:
            break
        tipe = str(rng.choice(["Counterparty", "Badan hukum", "Atribut berbagi"], p=[0.6, 0.28, 0.12]))
        nid = f"{_PREFIKS_SIMPUL[tipe]}-{rng.integers(1000, 9999)}"
        relasi = str(rng.choice(["memasok", "menjual_ke", "berbagi_atribut", "satu_sektor"],
                                p=[0.38, 0.34, 0.14, 0.14]))
        nodes.append({
            "id": nid, "label": nid, "tipe": tipe, "hop": 1,
            "community_id": komunitas_pusat if rng.random() < 0.7 else int(rng.integers(1, 13)),
            "pd": float(rng.beta(2, 45)) if tipe == "Badan hukum" else float("nan"),
            "grup": grup if tipe == "Badan hukum" and rng.random() < 0.6 else "-",
            "peran": str(rng.choice(peran_cp)) if tipe == "Counterparty" else "Entitas terkait",
        })
        edges.append({
            "source": entity_id if relasi == "menjual_ke" else nid,
            "target": nid if relasi == "menjual_ke" else entity_id,
            "relasi": relasi,
            "bobot": float(np.clip(rng.lognormal(23.2, 0.9), 5e8, 3e11)),
        })
        hop1.append(nid)

    sebelumnya = hop1
    for hop in range(2, hops + 1):
        berikutnya = []
        for induk in sebelumnya:
            if len(nodes) >= batas_simpul:
                break
            for _ in range(int(rng.integers(1, 4))):
                if len(nodes) >= batas_simpul:
                    break
                tipe = str(rng.choice(
                    ["Badan hukum", "Counterparty", "Pengurus", "Atribut berbagi", "Agunan"],
                    p=[0.34, 0.30, 0.14, 0.14, 0.08],
                ))
                nid = f"{_PREFIKS_SIMPUL[tipe]}-{rng.integers(1000, 9999)}"
                nodes.append({
                    "id": nid, "label": nid, "tipe": tipe, "hop": hop,
                    "community_id": komunitas_pusat if rng.random() < 0.55 else int(rng.integers(1, 13)),
                    "pd": float(rng.beta(2, 38)) if tipe == "Badan hukum" else float("nan"),
                    "grup": grup if rng.random() < 0.45 else "-",
                    "peran": "Entitas hop dua",
                })
                edges.append({
                    "source": induk,
                    "target": nid,
                    "relasi": str(rng.choice(TIPE_RELASI, p=[0.14, 0.10, 0.12, 0.18, 0.16, 0.10, 0.12, 0.08])),
                    "bobot": float(np.clip(rng.lognormal(22.4, 1.0), 2e8, 2e11)),
                })
                berikutnya.append(nid)
        sebelumnya = berikutnya
        if len(nodes) >= batas_simpul:
            break

    df_nodes = pd.DataFrame(nodes).drop_duplicates(subset="id").reset_index(drop=True)
    df_edges = pd.DataFrame(edges)
    sah = set(df_nodes["id"])
    df_edges = df_edges[df_edges["target"].isin(sah) & df_edges["source"].isin(sah)].reset_index(drop=True)
    return df_nodes, df_edges


def penelusuran_kepemilikan(entity_id: str) -> pd.DataFrame:
    """Rantai kepemilikan berlapis sampai pemilik manfaat akhir (proposal 7.1).

    Tiruan hasil relasi `mengendalikan` yang diturunkan dari penelusuran
    kepemilikan berlapis.
    """
    rng = _rng(f"owner-{entity_id}")
    lapis = int(rng.integers(2, 5))
    baris = []
    anak = entity_id
    porsi_kumulatif = 1.0
    for tingkat in range(1, lapis + 1):
        porsi = float(np.clip(rng.uniform(0.51, 0.99), 0.51, 0.99))
        porsi_kumulatif *= porsi
        terakhir = tingkat == lapis
        induk = (
            f"PM-{rng.integers(1000, 9999)}" if terakhir
            else f"BH-{rng.integers(1000, 9999)}"
        )
        baris.append({
            "tingkat": tingkat,
            "pemilik": induk,
            "jenis": "Pemilik manfaat akhir (perorangan)" if terakhir else "Badan hukum antara",
            "dimiliki": anak,
            "porsi_langsung": porsi,
            "porsi_efektif": porsi_kumulatif,
            "aktivitas_usaha": "-" if terakhir else str(
                rng.choice(["Ada", "Tidak ada — entitas penampung"], p=[0.6, 0.4])
            ),
            "yurisdiksi": str(rng.choice(["Indonesia", "Indonesia", "Indonesia", "Singapura"])),
        })
        anak = induk
    return pd.DataFrame(baris)


def score_network_risk(application_id: str) -> dict:
    """Tiruan `score_network_risk(application_id)` — skor 0..100 + pola pemicu."""
    rng = _rng(f"netrisk-{application_id}")
    skor = float(np.clip(rng.beta(2.0, 4.0) * 100, 0, 100))
    kunci = list(POLA_ANOMALI)
    jumlah = 0 if skor < 25 else (1 if skor < 50 else int(rng.integers(2, 4)))
    terpicu = list(rng.choice(kunci, size=jumlah, replace=False)) if jumlah else []
    return {
        "application_id": application_id,
        "skor": skor,
        "pola": [
            {
                "kode": k,
                "deskripsi": POLA_ANOMALI[k],
                "bukti": f"{int(rng.integers(2, 7))} badan hukum terkait, "
                         f"nilai transaksi Rp {rng.integers(4, 90)} M dalam "
                         f"{int(rng.integers(3, 30))} hari",
            }
            for k in terpicu
        ],
    }


def daftar_grup(n: int = 12) -> pd.DataFrame:
    """Eksposur gabungan per grup debitur terhadap BMPK (proposal 7.1, 9.3)."""
    rng = _rng("grup")
    baris = []
    for nama in NAMA_GRUP[:n]:
        porsi = float(np.clip(rng.beta(2.6, 3.0), 0.05, 0.98))
        eksposur = mock_engine.BATAS_BMPK_GRUP * porsi
        baris.append({
            "grup_usaha": nama,
            "jumlah_entitas": int(rng.integers(2, 11)),
            "entitas_debitur": int(rng.integers(1, 6)),
            "sektor_inti": str(rng.choice(SEKTOR)),
            "eksposur_grup": eksposur,
            "porsi_bmpk": porsi,
            "ruang_bmpk": mock_engine.BATAS_BMPK_GRUP - eksposur,
            "pd_tertimbang": float(np.clip(rng.beta(2.2, 45.0), 0.002, 0.24)),
            "npl_grup": float(np.clip(rng.beta(1.8, 45.0) * 3, 0.0, 0.20)),
            "covenant_terlanggar": int(rng.integers(0, 4)),
            "pemilik_manfaat": f"PM-{rng.integers(1000, 9999)}",
        })
    return pd.DataFrame(baris).sort_values("eksposur_grup", ascending=False).reset_index(drop=True)


def daftar_komunitas() -> pd.DataFrame:
    rng = _rng("komunitas")
    baris = []
    for cid in range(1, 13):
        baris.append({
            "community_id": cid,
            "nama": f"Klaster {cid} - {rng.choice(SEKTOR)} {rng.choice(WILAYAH)}",
            "jumlah_anggota": int(rng.integers(8, 90)),
            "eksposur_bank": float(rng.uniform(60e9, 900e9)),
            "npl_komunitas": float(np.clip(rng.beta(2.0, 40.0) * 3, 0.001, 0.18)),
            "pd_rata_tetangga": float(np.clip(rng.beta(2.2, 35.0), 0.002, 0.26)),
            "modularitas": float(rng.uniform(0.42, 0.78)),
            "simpul_kritis": f"CP-{rng.integers(1000, 9999)}",
        })
    return pd.DataFrame(baris)


def counterparty_penting() -> pd.DataFrame:
    """Daftar systemically important counterparty (proposal 7.2 A)."""
    rng = _rng("counterparty")
    peran = ["Pembeli utama sektor", "Pemasok bahan baku impor",
             "Pemberi kerja proyek", "Distributor nasional", "Operator logistik"]
    baris = []
    for i in range(15):
        baris.append({
            "entity_id": f"CP-{rng.integers(1000, 9999)}",
            "nama": f"{rng.choice(['PT', 'PT', 'CV'])} {_KATA_NAMA[i % len(_KATA_NAMA)]}",
            "peran": str(rng.choice(peran)),
            "sektor": str(rng.choice(SEKTOR)),
            "wilayah": str(rng.choice(WILAYAH)),
            "debitur_terhubung": int(rng.integers(4, 40)),
            "pagerank": float(rng.uniform(0.004, 0.06)),
            "betweenness": float(rng.uniform(0.01, 0.35)),
            "volume_tahunan": float(rng.uniform(50e9, 900e9)),
            "eksposur_terdampak": float(rng.uniform(40e9, 700e9)),
        })
    return pd.DataFrame(baris).sort_values("pagerank", ascending=False).reset_index(drop=True)


def uji_tekanan(entity_id: str, tingkat_guncangan: float) -> dict:
    """Tiruan skenario 'bila satu pembeli utama sektor menghentikan pesanan'."""
    rng = _rng(f"stress-{entity_id}")
    eksposur = float(rng.uniform(40e9, 700e9))
    debitur = int(rng.integers(4, 40))
    kenaikan_pd = 0.038 * tingkat_guncangan
    return {
        "eksposur_terdampak": eksposur,
        "debitur_terdampak": debitur,
        "grup_terdampak": int(rng.integers(1, 7)),
        "kenaikan_pd_rata": kenaikan_pd,
        "tambahan_pencadangan": eksposur * kenaikan_pd * 0.55,
    }


# --------------------------------------------------------------------------
# Kesehatan model
# --------------------------------------------------------------------------
def metrik_model() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "PD komersial - gradient boosting", "auc": 0.827, "gini": 0.654, "ks": 0.489, "brier": 0.058, "status": "Produksi demo"},
        {"model": "PD komersial - scorecard WOE", "auc": 0.781, "gini": 0.562, "ks": 0.417, "brier": 0.069, "status": "Pembanding auditable"},
        {"model": "LGD - regresi tingkat pemulihan", "auc": float("nan"), "gini": float("nan"), "ks": float("nan"), "brier": 0.084, "status": "Produksi demo"},
        {"model": "Early warning - migrasi kolektibilitas", "auc": 0.798, "gini": 0.596, "ks": 0.448, "brier": 0.071, "status": "Produksi demo"},
        {"model": "Segmentasi portofolio - clustering", "auc": float("nan"), "gini": float("nan"), "ks": float("nan"), "brier": float("nan"), "status": "Pendamping"},
        {"model": "Network anomaly - Isolation Forest", "auc": 0.744, "gini": 0.488, "ks": 0.371, "brier": float("nan"), "status": "Pendamping"},
    ])


def uji_ablasi_graf() -> pd.DataFrame:
    """Selisih metrik tanpa dan dengan blok fitur graf (proposal 7.3)."""
    return pd.DataFrame([
        {"varian": "Tanpa blok fitur graf", "auc": 0.7930, "gini": 0.5860, "ks": 0.4520},
        {"varian": "Dengan blok fitur graf", "auc": 0.8270, "gini": 0.6540, "ks": 0.4890},
    ])


def evaluasi_agen() -> pd.DataFrame:
    """Metrik lapisan agen dan RAG kepatuhan.

    Penilaian mutu jawaban memakai arena berpasangan: dua varian agen menjawab
    kasus uji yang sama, lalu Qwen 14B sebagai model penilai memilih yang lebih
    baik beserta alasannya. Model penilai sengaja berbeda dari model yang
    dipakai agen supaya penilaian tidak menilai gaya tulisannya sendiri.

    Metrik yang bisa diperiksa tanpa penilai — kecocokan angka memo terhadap
    keluaran tool, dan ketepatan atribusi pasal — dihitung dengan pembandingan
    langsung, bukan dengan model.
    """
    return pd.DataFrame([
        {"metrik": "Tingkat menang arena vs agen dasar (juri Qwen 14B)", "nilai": 0.72, "ambang": 0.55},
        {"metrik": "Kesepakatan juri Qwen 14B dengan penilai manusia", "nilai": 0.86, "ambang": 0.80},
        {"metrik": "Ketepatan ekstraksi entitas dari narasi dan dokumen", "nilai": 0.94, "ambang": 0.90},
        {"metrik": "Akurasi pemilihan tool (termasuk tool graf)", "nilai": 0.91, "ambang": 0.85},
        {"metrik": "Recall@5 penelusuran klausul kebijakan", "nilai": 0.88, "ambang": 0.85},
        {"metrik": "Ketepatan atribusi kutipan pasal", "nilai": 0.96, "ambang": 0.95},
        {"metrik": "Tingkat penolakan aman saat dasar kebijakan tidak ada", "nilai": 0.97, "ambang": 0.95},
        {"metrik": "Konsistensi angka memo terhadap keluaran tool", "nilai": 1.00, "ambang": 1.00},
    ])


def population_stability() -> pd.DataFrame:
    rng = _rng("psi")
    fitur = [
        "rasio_plafon_penjualan", "interest_coverage_ratio", "debt_to_ebitda",
        "konversi_ebitda_kas", "utilisasi_plafon", "buyer_concentration_hhi",
        "supplier_concentration_hhi", "neighbor_default_rate_1hop",
        "group_exposure_share", "community_default_rate", "pagerank",
        "shared_attribute_degree",
    ]
    nilai = np.clip(rng.beta(1.6, 12.0, len(fitur)) * 0.9, 0.005, 0.42)
    return pd.DataFrame({"fitur": fitur, "psi": nilai}).sort_values("psi", ascending=False).reset_index(drop=True)


def gerbang_kualitas_data() -> pd.DataFrame:
    return pd.DataFrame([
        {"pemeriksaan": "Skema star schema kredit komersial sesuai kontrak", "hasil": "Lulus", "baris_karantina": 0},
        {"pemeriksaan": "Tidak ada duplikat application_id", "hasil": "Lulus", "baris_karantina": 0},
        {"pemeriksaan": "Nominal tidak negatif dan satuan seragam", "hasil": "Lulus dengan perbaikan", "baris_karantina": 306},
        {"pemeriksaan": "Format tanggal seragam", "hasil": "Lulus dengan perbaikan", "baris_karantina": 412},
        {"pemeriksaan": "Resolusi entitas badan hukum - ejaan dan singkatan", "hasil": "Lulus dengan perbaikan", "baris_karantina": 188},
        {"pemeriksaan": "Rentang segmen komersial (penjualan, plafon, saldo)", "hasil": "Lulus", "baris_karantina": 0},
        {"pemeriksaan": "Anti-kebocoran: edge valid_from <= snapshot", "hasil": "Lulus", "baris_karantina": 0},
        {"pemeriksaan": "Outlier penjualan tahunan ekstrem", "hasil": "Perlu telaah", "baris_karantina": 27},
    ])


def distribusi_skor() -> pd.DataFrame:
    rng = _rng("distribusi")
    return pd.DataFrame({
        "pd": np.concatenate([rng.beta(2.0, 60.0, 900), rng.beta(3.4, 22.0, 220)]),
        "periode": ["Data pelatihan"] * 900 + ["Bulan berjalan"] * 220,
    })


# --------------------------------------------------------------------------
# Lapisan agen (tiruan)
# --------------------------------------------------------------------------
CONTOH_PROMPT = [
    "PT Sumber Logam Perkasa, manufaktur komponen otomotif di Karawang, penjualan Rp 240 miliar, "
    "EBITDA margin 11 persen, DER 1,8x, saldo rata-rata Rp 18 miliar, minta perpanjangan modal kerja "
    "Rp 80 miliar plus investasi mesin Rp 40 miliar tenor 5 tahun, jaminan pabrik dan mesin, "
    "satu grup dengan dua entitas yang sudah menjadi debitur kami",

    "PT Andalan Niaga Utama, distributor bahan bangunan di Cikarang, penjualan Rp 190 miliar, "
    "EBITDA margin 9 persen, DER 2,4x, minta tambahan modal kerja Rp 90 miliar tenor 36 bulan. "
    "Pemegang saham mayoritasnya juga terdaftar di dua debitur kami, dan direktur utamanya sama "
    "dengan salah satu pengajuan bulan lalu. Agunan berupa persediaan dan piutang dagang.",

    "PT Wana Agro Lestari, pengolahan hasil perkebunan di Medan, penjualan Rp 120 miliar, "
    "EBITDA margin 14 persen, DER 1,2x, saldo rata-rata Rp 12 miliar, pengajuan investasi "
    "Rp 60 miliar tenor 60 bulan dengan agunan tanah dan bangunan pabrik. Seluruh pesanan berasal "
    "dari satu pembeli utama di luar negeri.",
]


def ekstraksi_entitas(teks: str) -> dict:
    """Tiruan ekstraksi entitas + validasi skema Pydantic pada lapisan agen."""
    t = teks.lower()

    def semua_nominal():
        for m in re.finditer(r"(?:rp\s*)?([\d.,]+)\s*(miliar|milyar|m|juta|jt)\b", t):
            angka = float(m.group(1).replace(".", "").replace(",", "."))
            pengali = 1_000_000 if m.group(2) in ("juta", "jt") else 1_000_000_000
            yield m, angka * pengali

    def cari_nominal(pola_kata, bawaan: float, jumlahkan: bool = False) -> float:
        total, ketemu = 0.0, False
        for m, nilai in semua_nominal():
            konteks = t[max(0, m.start() - 55): m.start()]
            if any(k in konteks for k in pola_kata):
                if not jumlahkan:
                    return nilai
                total += nilai
                ketemu = True
        return total if ketemu else bawaan

    plafon = cari_nominal(
        ["modal kerja", "minta", "pengajuan", "plafon", "fasilitas", "investasi", "tambahan"],
        80e9, jumlahkan=True,
    )
    penjualan = cari_nominal(["penjualan", "omzet", "pendapatan", "revenue"], 200e9)
    saldo_giro = cari_nominal(["saldo", "giro"], 15e9)

    # Tenor boleh ditulis dalam bulan atau tahun.
    m_tenor_bulan = re.search(r"tenor\s*(\d+)\s*bulan", t) or re.search(r"(\d+)\s*bulan", t)
    m_tenor_tahun = re.search(r"tenor\s*(\d+)\s*tahun", t)
    if m_tenor_tahun:
        tenor = int(m_tenor_tahun.group(1)) * 12
    elif m_tenor_bulan:
        tenor = int(m_tenor_bulan.group(1))
    else:
        tenor = 36

    m_margin = re.search(r"ebitda\s*margin\s*([\d,\.]+)\s*(?:persen|%)", t)
    ebitda_margin = float(m_margin.group(1).replace(",", ".")) / 100 if m_margin else 0.11

    m_der = re.search(r"der\s*([\d,\.]+)\s*x", t)
    der = float(m_der.group(1).replace(",", ".")) if m_der else 1.8

    kata_angka = {
        "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
        "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10,
    }
    m_umur = re.search(r"berdiri\s*(?:sejak\s*)?(\d+)\s*tahun", t)
    umur = float(m_umur.group(1)) if m_umur else 14.0

    sektor = next(
        (s for s, alias in ALIAS_SEKTOR.items()
         if s.lower() in t or any(a in t for a in alias)),
        "Manufaktur komponen otomotif",
    )
    wilayah = next((w for w in WILAYAH if w.lower() in t), "Karawang")

    # Pencocokan agunan komersial: nama lengkap lebih dulu, lalu kata kunci.
    # Bila disebut lebih dari satu, ambil yang tingkat pemulihannya paling tinggi
    # karena itulah yang menentukan LGD pada pengikatan berjenjang.
    kandidat = [a for a in JENIS_AGUNAN if a.lower() in t]
    kandidat += [a for a, kunci in KUNCI_AGUNAN if any(k in t for k in kunci)]
    agunan = (
        max(kandidat, key=lambda a: mock_engine.RECOVERY_AGUNAN[a])
        if kandidat else "Tanpa agunan (clean basis)"
    )

    jumlah_entitas = 1
    m_entitas = re.search(r"(\d+)\s*entitas", t)
    if m_entitas:
        jumlah_entitas = int(m_entitas.group(1)) + 1
    else:
        for kata, nilai in kata_angka.items():
            if f"{kata} entitas" in t or f"{kata} debitur" in t:
                jumlah_entitas = nilai + 1
                break

    rangkap = any(k in t for k in ("direktur utamanya sama", "rangkap jabatan", "pengurus sama",
                                   "direksi sama", "juga terdaftar"))
    konsentrasi_pembeli = any(
        k in t for k in ("satu pembeli", "satu buyer", "satu pemberi kerja",
                         "satu pelanggan", "pembeli utama", "seluruh pesanan")
    )
    konsentrasi_pemasok = any(
        k in t for k in ("satu pemasok", "satu supplier", "pemasok tunggal", "pemasok impor")
    )

    # Jenis fasilitas menentukan apakah kewajiban berjalan berupa bunga saja
    # (revolving) atau angsuran anuitas, jadi ikut diekstraksi dari narasi.
    if "modal kerja" in t:
        fasilitas = "Modal kerja - rekening koran"
    elif "investasi" in t or "term loan" in t:
        fasilitas = "Investasi - term loan"
    elif "lc " in t or "impor" in t or "trade finance" in t:
        fasilitas = "Trade finance - LC impor"
    elif "bank garansi" in t or "proyek" in t:
        fasilitas = "Bank garansi proyek"
    else:
        fasilitas = "Modal kerja - rekening koran"

    return {
        "nama_debitur": _tebak_nama(teks),
        "jenis_fasilitas": fasilitas,
        "sektor": sektor,
        "wilayah": wilayah,
        "umur_usaha_thn": umur,
        "penjualan_tahunan": float(penjualan),
        "ebitda_margin": float(ebitda_margin),
        "der": float(der),
        "plafon": float(plafon),
        "tenor_bulan": int(tenor),
        "jenis_agunan": agunan,
        "nilai_agunan": float(plafon) * 1.3 if "Tanpa agunan" not in agunan else 0.0,
        "saldo_giro_rata": float(saldo_giro),
        "jumlah_entitas_grup": int(jumlah_entitas),
        "indikasi_rangkap_jabatan": rangkap,
        "indikasi_konsentrasi_pembeli": konsentrasi_pembeli,
        "indikasi_konsentrasi_pemasok": konsentrasi_pemasok,
    }


def _tebak_nama(teks: str) -> str:
    m = re.search(r"\b(PT|CV|UD)\s+([A-Z][\w]*(?:\s+[A-Z][\w]*){0,3})", teks)
    return f"{m.group(1)} {m.group(2)}" if m else "Badan hukum tanpa nama"


def rencana_agen(entitas: dict):
    """Urutan tool yang dipilih agen.

    Pada sistem sebenarnya urutan ini datang dari tool calling loop LLM.
    Di sini urutannya ditiru dan berbeda menurut isi masukan supaya demo
    memperlihatkan bahwa jalur pemanggilan tidak selalu sama.
    """
    langkah = [
        {"tool": "get_customer_history", "arg": "cif",
         "keterangan": "Riwayat fasilitas, penarikan, dan pelunasan 12 bulan"},
        {"tool": "query_warehouse", "arg": "sql",
         "keterangan": f"Pembanding rasio keuangan peer sektor {entitas['sektor'].lower()} "
                       f"kelas penjualan Rp {entitas['penjualan_tahunan'] / 1e9:.0f} M"},
        {"tool": "get_entity_network", "arg": "cif, hops=2",
         "keterangan": "Struktur grup usaha dan rantai pasok dua hop"},
    ]
    if entitas.get("indikasi_rangkap_jabatan") or entitas.get("jumlah_entitas_grup", 1) >= 3:
        langkah.append({"tool": "find_community", "arg": "cif",
                        "keterangan": "Klaster ekosistem dan profil risikonya"})
        langkah.append({"tool": "predict_links", "arg": "cif, top_k=5",
                        "keterangan": "Kandidat afiliasi tersembunyi untuk pemeriksaan BMPK"})
    langkah += [
        {"tool": "score_network_risk", "arg": "application_id",
         "keterangan": "Pola anomali struktur dan afiliasi tak dinyatakan"},
        {"tool": "score_pd", "arg": "application",
         "keterangan": "Probability of default terkalibrasi 12 bulan"},
        {"tool": "estimate_lgd", "arg": "collateral",
         "keterangan": f"Tingkat pemulihan agunan {entitas['jenis_agunan'].lower()}"},
        {"tool": "recommend_limit_pricing", "arg": "pd, lgd, ead",
         "keterangan": "Expected loss -> usulan limit grup, tenor, dan pricing"},
    ]
    aspek = ["kewenangan", "bmpk grup", "agunan", "covenant"]
    if entitas.get("indikasi_rangkap_jabatan") or entitas.get("jumlah_entitas_grup", 1) >= 3:
        aspek.append("afiliasi")
    for a in aspek:
        langkah.append({
            "tool": "check_credit_policy", "arg": f'"{a}", konteks_pengajuan',
            "keterangan": f"Gerbang kepatuhan aspek {a} — RAG kebijakan kredit komersial",
        })
    langkah.append({"tool": "explain_prediction", "arg": "model, row",
                    "keterangan": "SHAP -> reason code dan faktor pendorong utama"})
    return langkah


def kutipan_kebijakan(entitas: dict):
    """Tiruan RAG atas korpus kebijakan kredit komersial (proposal 5.3)."""
    dasar = [
        {
            "pasal": "KKK-02.1 Definisi Segmen Komersial",
            "isi": "Segmen komersial mencakup debitur dengan penjualan tahunan Rp 30 miliar sampai "
                   "Rp 300 miliar, plafon Rp 10 miliar sampai Rp 150 miliar, dan saldo rata-rata "
                   "Rp 10 miliar sampai Rp 50 miliar.",
            "skor": 0.93,
            "versi": "berlaku 1 Januari 2026",
        },
        {
            "pasal": "KKK-05.3 Matriks Kewenangan Komite Komersial",
            "isi": "Fasilitas di atas Rp 75 miliar atau berating di bawah BBB diputus oleh Komite "
                   "Kredit Komersial Pusat; di bawahnya cukup pada Komite Kredit Komersial.",
            "skor": 0.90,
            "versi": "berlaku 1 Januari 2026",
        },
        {
            "pasal": "KKK-08.2 BMPK dan Grup Debitur",
            "isi": "Seluruh entitas yang dikendalikan pemilik manfaat yang sama digabungkan sebagai "
                   "satu grup debitur; sisa ruang batas maksimum wajib dicatat pada usulan.",
            "skor": 0.89,
            "versi": "berlaku 1 Januari 2026",
        },
        {
            "pasal": "KKK-09.4 Kebijakan Agunan dan Pengikatan",
            "isi": "Rasio pertanggungan agunan minimum ditetapkan per kelas rating; penjaminan silang "
                   "antar afiliasi tidak boleh diperhitungkan ganda.",
            "skor": 0.87,
            "versi": "berlaku 1 Januari 2026",
        },
    ]
    if entitas.get("indikasi_rangkap_jabatan") or entitas.get("jumlah_entitas_grup", 1) >= 3:
        dasar.append({
            "pasal": "KKK-13.6 Pihak Terafiliasi dan APU-PPT",
            "isi": "Rangkap jabatan pengurus atau atribut identitas yang dipakai bersama antar badan "
                   "hukum menjadi pemicu penelaahan lanjutan dan penelusuran pemilik manfaat akhir.",
            "skor": 0.95,
            "versi": "berlaku 1 Januari 2026",
        })
    if entitas.get("indikasi_konsentrasi_pembeli"):
        dasar.append({
            "pasal": "KKK-10.7 Konsentrasi Pembeli",
            "isi": "Ketergantungan pendapatan pada satu pembeli di atas 60% wajib dicantumkan sebagai "
                   "faktor risiko pada credit memo dan diikat covenant pemberitahuan.",
            "skor": 0.88,
            "versi": "berlaku 1 Januari 2026",
        })
    if entitas.get("indikasi_konsentrasi_pemasok"):
        dasar.append({
            "pasal": "KKK-10.8 Konsentrasi Pemasok Impor",
            "isi": "Ketergantungan bahan baku pada satu pemasok impor wajib disertai analisis "
                   "sensitivitas kurs dan rencana pemasok pengganti.",
            "skor": 0.84,
            "versi": "berlaku 1 Januari 2026",
        })
    return dasar


def dokumen_kurang(entitas: dict):
    kurang = [
        "Laporan keuangan audited dua tahun terakhir beserta catatannya",
        "Proyeksi arus kas selama tenor fasilitas",
        "Rekening koran bank utama 6 bulan terakhir",
    ]
    if "Tanah dan bangunan" in entitas["jenis_agunan"]:
        kurang.append("Laporan penilaian agunan (KJPP) terbaru dan bukti pengikatan hak tanggungan")
    if "Mesin" in entitas["jenis_agunan"]:
        kurang.append("Daftar mesin, invoice pembelian, dan bukti pengikatan fidusia")
    if "Persediaan" in entitas["jenis_agunan"] or "Piutang" in entitas["jenis_agunan"]:
        kurang.append("Aging piutang dan daftar persediaan per akhir bulan terakhir")
    if entitas.get("indikasi_rangkap_jabatan") or entitas.get("jumlah_entitas_grup", 1) >= 3:
        kurang.append("Struktur kepemilikan grup sampai pemilik manfaat akhir beserta akta pendukung")
        kurang.append("Daftar fasilitas aktif seluruh entitas satu grup pada bank lain")
    if entitas.get("indikasi_konsentrasi_pembeli"):
        kurang.append("Salinan kontrak atau purchase order dari pembeli utama")
    kurang.append("Risalah kunjungan relationship manager yang ditandatangani")
    return kurang

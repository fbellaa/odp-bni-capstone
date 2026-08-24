"""Data dummy untuk live demo.

Seluruh isi modul ini sintetis dan dibangkitkan dengan seed tetap supaya
tampilan demo konsisten antar sesi. Tidak ada data nasabah sebenarnya.

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
    "Warung kelontong", "Toko bahan bangunan", "Konveksi", "Kuliner",
    "Bengkel motor", "Agribisnis", "Toko elektronik", "Jasa laundry",
]
WILAYAH = [
    "Bekasi", "Karawang", "Bandung", "Semarang",
    "Surabaya", "Tangerang", "Bogor", "Depok",
]
JENIS_AGUNAN = list(mock_engine.RECOVERY_AGUNAN.keys())
KOLEKTIBILITAS = [
    "1 - Lancar", "2 - Dalam perhatian khusus", "3 - Kurang lancar",
    "4 - Diragukan", "5 - Macet",
]

POLA_ANOMALI = {
    "shared_attribute": "Berbagi atribut identitas (telepon / alamat / rekening) dengan pengajuan lain",
    "repeat_guarantor": "Penjamin yang sama menjamin banyak pengajuan tidak berhubungan",
    "circular_payment": "Siklus pembayaran melingkar yang menaikkan omzet secara artifisial",
    "degree_spike": "Lonjakan derajat simpul tidak wajar menjelang tanggal pengajuan",
    "shared_disbursement": "Rekening penerima pencairan sama pada beberapa debitur berbeda",
}


def _rng(salt: str = "") -> np.random.Generator:
    benih = SEED + int(hashlib.md5(salt.encode()).hexdigest()[:8], 16) % 100_000
    return np.random.default_rng(benih)


# --------------------------------------------------------------------------
# Portofolio pengajuan
# --------------------------------------------------------------------------
def daftar_pengajuan(n: int = 180) -> pd.DataFrame:
    rng = _rng("pengajuan")
    sektor = rng.choice(SEKTOR, n)
    wilayah = rng.choice(WILAYAH, n)
    baris = []
    for idx in range(n):
        omzet = float(np.clip(rng.lognormal(mean=17.4, sigma=0.55), 12_000_000, 900_000_000))
        plafon = float(np.clip(rng.lognormal(mean=18.4, sigma=0.6), 20_000_000, 500_000_000))
        tenor = int(rng.choice([12, 18, 24, 36, 48], p=[0.15, 0.2, 0.35, 0.2, 0.1]))
        agunan = str(rng.choice(JENIS_AGUNAN, p=[0.10, 0.22, 0.20, 0.05, 0.20, 0.13, 0.10]))
        nilai_agunan = plafon * float(rng.uniform(0.4, 1.8)) if agunan != "Tanpa agunan" else 0.0
        network_risk = float(np.clip(rng.beta(1.8, 6.0) * 100, 0, 100))
        pengajuan = dict(
            omzet_bulanan=omzet,
            plafon=plafon,
            tenor_bulan=tenor,
            lama_usaha_thn=float(np.clip(rng.gamma(3.0, 1.7), 0.5, 25)),
            jenis_agunan=agunan,
            nilai_agunan=nilai_agunan,
            margin_usaha=float(np.clip(rng.normal(0.14, 0.035), 0.05, 0.28)),
            stabilitas_arus_kas=float(np.clip(rng.beta(2.5, 6.0), 0.02, 0.95)),
            supplier_concentration_hhi=float(np.clip(rng.beta(2.5, 4.0), 0.05, 0.98)),
            neighbor_default_rate_1hop=float(np.clip(rng.beta(1.6, 18.0), 0.0, 0.6)),
            network_risk_score=network_risk,
            tenure_nasabah_thn=float(np.clip(rng.gamma(2.2, 1.4), 0.0, 15)),
        )
        hasil = mock_engine.recommend_limit_pricing(pengajuan)
        baris.append(
            {
                "application_id": f"APP-2026-{idx + 1001}",
                "cif": f"CIF{rng.integers(10_000_000, 99_999_999)}",
                "nama_usaha": f"{sektor[idx]} {wilayah[idx]} {idx + 1:03d}",
                "sektor": sektor[idx],
                "wilayah": wilayah[idx],
                "tanggal": pd.Timestamp("2026-08-24") - pd.Timedelta(days=int(rng.integers(0, 180))),
                "omzet_bulanan": omzet,
                "plafon_diminta": plafon,
                "tenor_bulan": tenor,
                "jenis_agunan": agunan,
                "nilai_agunan": nilai_agunan,
                "lama_usaha_thn": pengajuan["lama_usaha_thn"],
                "pd": hasil.pd,
                "grade": hasil.grade,
                "lgd": hasil.lgd,
                "expected_loss": hasil.expected_loss,
                "pricing": hasil.pricing,
                "limit_usulan": hasil.limit_usulan,
                "network_risk_score": network_risk,
                "community_id": int(rng.integers(1, 13)),
                "kolektibilitas": str(rng.choice(KOLEKTIBILITAS, p=[0.80, 0.11, 0.05, 0.02, 0.02])),
                "keputusan": mock_engine.keputusan_dari_hasil(hasil),
            }
        )
    return pd.DataFrame(baris).sort_values("tanggal", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Lapisan graf
# --------------------------------------------------------------------------
TIPE_SIMPUL = ["UMKM", "Individu", "Merchant", "Distributor", "Atribut", "Agunan"]
TIPE_RELASI = [
    "membeli_dari", "menjual_ke", "membayar", "berbagi_atribut",
    "menjamin", "memiliki", "satu_wilayah",
]


def subgraf_ego(entity_id: str, hops: int = 2, batas_simpul: int = 60):
    """Tiruan `get_entity_network(entity_id, hops, min_weight)`.

    Mengembalikan (nodes, edges). Ukuran subgraf dibatasi sesuai catatan
    proposal 9.5 supaya graf besar tidak dikirim ke antarmuka.
    """
    rng = _rng(f"graf-{entity_id}-{hops}")
    komunitas_pusat = int(rng.integers(1, 13))
    nodes = [{
        "id": entity_id, "label": entity_id, "tipe": "UMKM", "hop": 0,
        "community_id": komunitas_pusat, "pd": float(rng.beta(2, 30)),
    }]
    edges = []

    sebelumnya = [entity_id]
    for hop in range(1, hops + 1):
        berikutnya = []
        for induk in sebelumnya:
            cabang = int(rng.integers(4, 9) if hop == 1 else rng.integers(2, 5))
            for _ in range(cabang):
                if len(nodes) >= batas_simpul:
                    break
                tipe = str(rng.choice(TIPE_SIMPUL, p=[0.34, 0.12, 0.18, 0.16, 0.14, 0.06]))
                nid = f"{tipe[:3].upper()}-{rng.integers(1000, 9999)}"
                nodes.append({
                    "id": nid,
                    "label": nid,
                    "tipe": tipe,
                    "hop": hop,
                    "community_id": komunitas_pusat if rng.random() < 0.6 else int(rng.integers(1, 13)),
                    "pd": float(rng.beta(2, 25)) if tipe == "UMKM" else float("nan"),
                })
                edges.append({
                    "source": induk,
                    "target": nid,
                    "relasi": str(rng.choice(TIPE_RELASI, p=[0.22, 0.20, 0.14, 0.14, 0.12, 0.08, 0.10])),
                    "bobot": float(np.clip(rng.lognormal(16.5, 0.8), 1e6, 5e9)),
                })
                berikutnya.append(nid)
        sebelumnya = berikutnya
        if len(nodes) >= batas_simpul:
            break

    df_nodes = pd.DataFrame(nodes).drop_duplicates(subset="id").reset_index(drop=True)
    df_edges = pd.DataFrame(edges)
    df_edges = df_edges[df_edges["target"].isin(set(df_nodes["id"]))].reset_index(drop=True)
    return df_nodes, df_edges


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
                "bukti": f"{int(rng.integers(2, 9))} entitas terkait dalam {int(rng.integers(3, 30))} hari",
            }
            for k in terpicu
        ],
    }


def daftar_komunitas() -> pd.DataFrame:
    rng = _rng("komunitas")
    baris = []
    for cid in range(1, 13):
        baris.append({
            "community_id": cid,
            "nama": f"Komunitas {cid} - {rng.choice(SEKTOR)} {rng.choice(WILAYAH)}",
            "jumlah_anggota": int(rng.integers(18, 260)),
            "eksposur_bank": float(rng.uniform(4e9, 90e9)),
            "npl_komunitas": float(np.clip(rng.beta(2.0, 40.0) * 3, 0.001, 0.22)),
            "pd_rata_tetangga": float(np.clip(rng.beta(2.2, 30.0), 0.002, 0.30)),
            "modularitas": float(rng.uniform(0.42, 0.78)),
            "simpul_kritis": f"DIS-{rng.integers(1000, 9999)}",
        })
    return pd.DataFrame(baris)


def counterparty_penting() -> pd.DataFrame:
    """Daftar systemically important counterparty (proposal 7.2 A)."""
    rng = _rng("counterparty")
    baris = []
    for i in range(15):
        baris.append({
            "entity_id": f"DIS-{rng.integers(1000, 9999)}",
            "nama": f"{rng.choice(['CV', 'PT', 'UD'])} Distributor {i + 1:02d}",
            "tipe": str(rng.choice(["Distributor", "Merchant", "Perusahaan pemberi kerja"])),
            "wilayah": str(rng.choice(WILAYAH)),
            "debitur_terhubung": int(rng.integers(6, 90)),
            "pagerank": float(rng.uniform(0.004, 0.06)),
            "betweenness": float(rng.uniform(0.01, 0.35)),
            "eksposur_terdampak": float(rng.uniform(2e9, 65e9)),
        })
    return pd.DataFrame(baris).sort_values("pagerank", ascending=False).reset_index(drop=True)


def uji_tekanan(entity_id: str, tingkat_guncangan: float) -> dict:
    """Tiruan skenario 'bila satu distributor besar gagal'."""
    rng = _rng(f"stress-{entity_id}")
    eksposur = float(rng.uniform(2e9, 65e9))
    debitur = int(rng.integers(6, 90))
    kenaikan_pd = 0.045 * tingkat_guncangan
    return {
        "eksposur_terdampak": eksposur,
        "debitur_terdampak": debitur,
        "kenaikan_pd_rata": kenaikan_pd,
        "tambahan_pencadangan": eksposur * kenaikan_pd * 0.55,
    }


# --------------------------------------------------------------------------
# Kesehatan model
# --------------------------------------------------------------------------
def metrik_model() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "PD - gradient boosting", "auc": 0.812, "gini": 0.624, "ks": 0.471, "brier": 0.061, "status": "Produksi demo"},
        {"model": "PD - scorecard WOE", "auc": 0.774, "gini": 0.548, "ks": 0.408, "brier": 0.070, "status": "Pembanding"},
        {"model": "LGD - regresi pemulihan", "auc": float("nan"), "gini": float("nan"), "ks": float("nan"), "brier": 0.088, "status": "Produksi demo"},
        {"model": "Early warning system", "auc": 0.786, "gini": 0.572, "ks": 0.433, "brier": 0.074, "status": "Produksi demo"},
        {"model": "Network anomaly - Isolation Forest", "auc": 0.731, "gini": 0.462, "ks": 0.362, "brier": float("nan"), "status": "Pendamping"},
    ])


def uji_ablasi_graf() -> pd.DataFrame:
    """Selisih metrik tanpa dan dengan blok fitur graf (proposal 7.3)."""
    return pd.DataFrame([
        {"varian": "Tanpa blok fitur graf", "auc": 0.7840, "gini": 0.5680, "ks": 0.4410},
        {"varian": "Dengan blok fitur graf", "auc": 0.8120, "gini": 0.6240, "ks": 0.4710},
    ])


def population_stability() -> pd.DataFrame:
    rng = _rng("psi")
    fitur = [
        "rasio_plafon_omzet", "stabilitas_arus_kas", "supplier_concentration_hhi",
        "neighbor_default_rate_1hop", "community_default_rate", "pagerank",
        "lama_usaha", "shared_attribute_degree",
    ]
    nilai = np.clip(rng.beta(1.6, 12.0, len(fitur)) * 0.9, 0.005, 0.42)
    return pd.DataFrame({"fitur": fitur, "psi": nilai}).sort_values("psi", ascending=False).reset_index(drop=True)


def gerbang_kualitas_data() -> pd.DataFrame:
    return pd.DataFrame([
        {"pemeriksaan": "Skema tabel gold sesuai kontrak", "hasil": "Lulus", "baris_karantina": 0},
        {"pemeriksaan": "Tidak ada duplikat application_id", "hasil": "Lulus", "baris_karantina": 0},
        {"pemeriksaan": "Nominal tidak negatif", "hasil": "Lulus", "baris_karantina": 0},
        {"pemeriksaan": "Format tanggal seragam", "hasil": "Lulus dengan perbaikan", "baris_karantina": 412},
        {"pemeriksaan": "Resolusi entitas - telepon ganda", "hasil": "Lulus dengan perbaikan", "baris_karantina": 138},
        {"pemeriksaan": "Anti-kebocoran: edge valid_from <= snapshot", "hasil": "Lulus", "baris_karantina": 0},
        {"pemeriksaan": "Outlier omzet ekstrem", "hasil": "Perlu telaah", "baris_karantina": 27},
    ])


def distribusi_skor() -> pd.DataFrame:
    rng = _rng("distribusi")
    return pd.DataFrame({
        "pd": np.concatenate([rng.beta(2.0, 40.0, 900), rng.beta(4.0, 12.0, 220)]),
        "periode": ["Data pelatihan"] * 900 + ["Bulan berjalan"] * 220,
    })


# --------------------------------------------------------------------------
# Lapisan agen (tiruan)
# --------------------------------------------------------------------------
CONTOH_PROMPT = [
    "Warung kelontong di Bekasi, usaha jalan empat tahun, omzet sekitar Rp 45 juta per bulan, "
    "minta modal kerja Rp 150 juta tenor 24 bulan, jaminan BPKB mobil 2019, hasil survei toko ramai dan stok rapi",

    "Toko bahan bangunan di Karawang, minta modal kerja Rp 300 juta tenor 36 bulan. Omzet sekitar "
    "Rp 180 juta per bulan. Pemiliknya juga punya usaha lain, dan penjaminnya sama dengan pengajuan "
    "bulan lalu dari daerah yang sama.",

    "Konveksi di Bandung, berdiri 7 tahun, omzet Rp 120 juta per bulan, pengajuan Rp 250 juta tenor "
    "24 bulan, agunan SHM ruko, seluruh order berasal dari satu buyer besar.",
]


def ekstraksi_entitas(teks: str) -> dict:
    """Tiruan ekstraksi entitas + validasi skema Pydantic pada lapisan agen."""
    t = teks.lower()

    def cari_nominal(pola_kata, bawaan: float) -> float:
        for m in re.finditer(r"(?:rp\s*)?([\d.,]+)\s*(juta|jt|miliar|m)\b", t):
            angka = float(m.group(1).replace(".", "").replace(",", "."))
            pengali = 1_000_000 if m.group(2) in ("juta", "jt") else 1_000_000_000
            konteks = t[max(0, m.start() - 45): m.start()]
            if any(k in konteks for k in pola_kata):
                return angka * pengali
        return bawaan

    plafon = cari_nominal(["modal kerja", "minta", "pengajuan", "plafon", "fasilitas"], 150_000_000)
    omzet = cari_nominal(["omzet", "penjualan", "pendapatan"], 45_000_000)

    m_tenor = re.search(r"tenor\s*(\d+)\s*bulan", t) or re.search(r"(\d+)\s*bulan", t)
    tenor = int(m_tenor.group(1)) if m_tenor else 24

    kata_angka = {
        "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
        "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10,
    }
    m_lama = re.search(r"(\d+)\s*tahun", t)
    if m_lama:
        lama = float(m_lama.group(1))
    else:
        lama = float(next((v for k, v in kata_angka.items() if f"{k} tahun" in t), 4))

    sektor = next((s for s in SEKTOR if s.split()[0].lower() in t), "Warung kelontong")
    wilayah = next((w for w in WILAYAH if w.lower() in t), "Bekasi")
    # Cocokkan nama lengkap agunan lebih dulu supaya "BPKB mobil" tidak
    # tertangkap sebagai "BPKB motor".
    agunan = next(
        (a for a in JENIS_AGUNAN if a.lower() in t),
        next((a for a in JENIS_AGUNAN if a.split()[0].lower() in t), "Tanpa agunan"),
    )
    if "shm" in t or "ruko" in t or "sertifikat" in t:
        agunan = "SHM / SHGB"
    elif "deposito" in t:
        agunan = "Deposito"

    return {
        "sektor": sektor,
        "wilayah": wilayah,
        "lama_usaha_thn": lama,
        "omzet_bulanan": omzet,
        "plafon": float(plafon),
        "tenor_bulan": tenor,
        "jenis_agunan": agunan,
        "nilai_agunan": float(plafon) * 1.2 if agunan != "Tanpa agunan" else 0.0,
        "indikasi_penjamin_berulang": "penjamin" in t and ("sama" in t or "berulang" in t),
        "indikasi_konsentrasi_pembeli": any(
            k in t for k in ("satu buyer", "satu pembeli", "satu perusahaan", "satu distributor")
        ),
    }


def rencana_agen(entitas: dict):
    """Urutan tool yang dipilih agen.

    Pada sistem sebenarnya urutan ini datang dari tool calling loop LLM.
    Di sini urutannya ditiru dan sedikit berbeda menurut isi masukan supaya
    demo memperlihatkan bahwa jalur pemanggilan tidak selalu sama.
    """
    langkah = [
        {"tool": "get_customer_history", "arg": "cif", "keterangan": "Riwayat simpanan dan transaksi 12 bulan"},
        {"tool": "query_warehouse", "arg": "sql", "keterangan": f"Pembanding sektor {entitas['sektor']} wilayah {entitas['wilayah']}"},
        {"tool": "get_entity_network", "arg": "cif, hops=2", "keterangan": "Subgraf relasi entitas dua hop"},
    ]
    if entitas.get("indikasi_penjamin_berulang"):
        langkah.append({"tool": "find_community", "arg": "cif", "keterangan": "Komunitas usaha dan profil risikonya"})
    langkah += [
        {"tool": "score_network_risk", "arg": "application_id", "keterangan": "Pola anomali jaringan"},
        {"tool": "score_pd", "arg": "application", "keterangan": "Probability of default terkalibrasi"},
        {"tool": "estimate_lgd", "arg": "collateral", "keterangan": f"Recovery untuk agunan {entitas['jenis_agunan']}"},
        {"tool": "recommend_limit_pricing", "arg": "pd, lgd, ead", "keterangan": "Expected loss -> usulan limit dan pricing"},
    ]
    kueri = (
        "penjaminan berulang oleh satu penjamin"
        if entitas.get("indikasi_penjamin_berulang")
        else "modal kerja mikro plafon di bawah 500 juta"
    )
    langkah += [
        {"tool": "check_credit_policy", "arg": f'"{kueri}"', "keterangan": "RAG atas dokumen kebijakan kredit"},
        {"tool": "explain_prediction", "arg": "model, row", "keterangan": "SHAP -> reason code"},
    ]
    return langkah


def kutipan_kebijakan(entitas: dict):
    dasar = [
        {
            "pasal": "KK-04.2 Modal Kerja Mikro",
            "isi": "Plafon modal kerja mikro paling tinggi Rp 500 juta dengan tenor maksimum 48 bulan.",
            "skor": 0.91,
        },
        {
            "pasal": "KK-07.1 Kapasitas Pembayaran",
            "isi": "Debt service coverage ratio minimum 1,35x dihitung atas arus kas usaha yang terverifikasi.",
            "skor": 0.88,
        },
    ]
    if entitas.get("indikasi_penjamin_berulang"):
        dasar.append({
            "pasal": "KK-11.5 Penjaminan Pihak Ketiga",
            "isi": "Satu penjamin perorangan hanya dapat menjamin paling banyak dua fasilitas aktif; "
                   "melebihi itu wajib verifikasi ulang dan persetujuan satu tingkat di atas.",
            "skor": 0.94,
        })
    if entitas.get("indikasi_konsentrasi_pembeli"):
        dasar.append({
            "pasal": "KK-09.3 Konsentrasi Pembeli",
            "isi": "Ketergantungan pendapatan pada satu pembeli di atas 60% wajib dicantumkan sebagai "
                   "faktor risiko pada credit memo.",
            "skor": 0.86,
        })
    return dasar


def dokumen_kurang(entitas: dict):
    kurang = ["Rekening koran 6 bulan terakhir", "Foto tempat usaha dan stok"]
    if entitas["jenis_agunan"] in ("SHM / SHGB", "Kios / los pasar"):
        kurang.append("Salinan sertifikat dan bukti penilaian agunan terbaru")
    if entitas["jenis_agunan"] in ("BPKB motor", "BPKB mobil"):
        kurang.append("Salinan BPKB dan bukti cek fisik kendaraan")
    if entitas.get("indikasi_penjamin_berulang"):
        kurang.append("Identitas penjamin dan daftar fasilitas aktif yang dijamin")
    kurang.append("Laporan hasil kunjungan relationship manager yang ditandatangani")
    return kurang

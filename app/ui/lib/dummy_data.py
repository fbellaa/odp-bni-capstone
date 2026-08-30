"""Data dummy untuk live demo — segmen kredit komersial.

Seluruh isi modul ini sintetis dan dibangkitkan dengan seed tetap supaya
tampilan demo konsisten antar sesi. Tidak ada data nasabah sebenarnya.

Rentang penjualan, plafon, dan saldo rata-rata mengikuti batas segmen komersial
pada proposal bagian 3.5 (penjualan Rp 30-300 M, plafon Rp 10-150 M, saldo
rata-rata Rp 10-50 M).

Yang tersisa di sini tinggal tiga: portofolio pengajuan dan eksposur grup untuk
pratinjau Dashboard BI selama Metabase belum jalan, serta rencana tool yang
ditampilkan halaman copilot ketika agen model bahasa tidak dipakai. Sisanya —
subgraf ego, penelusuran kepemilikan, skor risiko jaringan, metrik kesehatan
model, evaluasi agen, gerbang kualitas data, dan kutipan kebijakan — sudah
punya penggantinya yang membaca data atau korpus sungguhan, jadi versi
karangannya dihapus daripada menunggu terpakai lagi tanpa sengaja.

Cara mengganti dengan data asli nanti: tiap fungsi di bawah adalah satu titik
sambung ke FastAPI. Ganti isinya dengan `requests.get(f"{API_URL}/...")` dan
pertahankan bentuk keluarannya (DataFrame / dict) agar halaman tidak perlu
disentuh.
"""
from __future__ import annotations

import hashlib

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
         "keterangan": "Skor default 12 bulan dan pita risikonya"},
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



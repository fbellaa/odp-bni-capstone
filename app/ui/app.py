"""Titik masuk aplikasi demo Streamlit.

Jalankan dari folder app/ui:

    streamlit run app.py

Susunan halaman mengikuti proposal bagian 9.3. Seluruh data masih dummy;
lihat lib/dummy_data.py untuk titik sambung ke FastAPI nanti.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from lib import dummy_data
from lib.format import rupiah
from lib.tampilan import setup_halaman, sidebar_status

setup_halaman("Beranda", "🏦")
sidebar_status()

st.title("Agentic AI Copilot untuk Keputusan Kredit UMKM")
st.caption(
    "Sistem pendukung keputusan kredit usaha mikro dan kecil — "
    "data engineering, pemodelan risiko, analisis jaringan antar entitas, "
    "dan lapisan generative AI sebagai orkestrator model."
)

st.warning(
    "Aplikasi ini berjalan dengan **data dummy**. Model PD/LGD, lapisan graf, dan agen "
    "belum tersambung — seluruh angka dihitung oleh `lib/mock_engine.py` secara deterministik "
    "supaya interaksi demo tetap responsif.",
    icon="⚠️",
)

df = dummy_data.daftar_pengajuan()

st.subheader("Ringkasan portofolio demo")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pengajuan pada pipeline", f"{len(df):,}".replace(",", "."))
k2.metric("Total plafon diminta", rupiah(df["plafon_diminta"].sum(), singkat=True))
k3.metric("Rata-rata PD", f"{df['pd'].mean() * 100:.2f}%".replace(".", ","))
k4.metric("Expected loss portofolio", rupiah(df["expected_loss"].sum(), singkat=True))

st.divider()

st.subheader("Peta halaman")
kiri, kanan = st.columns(2)
halaman = [
    ("1 · Copilot pengajuan",
     "Kolom teks bebas, jejak langkah agen yang muncul bertahap, hasil skor dan reason code, "
     "tombol unduh credit memo."),
    ("2 · Simulasi what-if",
     "Slider plafon, tenor, dan jenis agunan; skor, pricing, dan ekspektasi kerugian "
     "diperbarui seketika."),
    ("3 · Jaringan entitas",
     "Subgraf ego dua hop, penyorotan komunitas, daftar entitas berpengaruh, panel pola "
     "anomali yang terdeteksi."),
    ("4 · Portofolio dan komunitas",
     "Konsentrasi eksposur per komunitas, uji tekanan simpul kritis, tabel counterparty penting."),
    ("5 · Kesehatan model",
     "Metrik berjalan, indeks stabilitas populasi, hasil uji ablasi fitur graf, kelulusan uji "
     "kualitas data."),
    ("6 · Dashboard BI",
     "Metabase disematkan sebagai iframe untuk lapisan eksekutif."),
]
for i, (judul, isi) in enumerate(halaman):
    kolom = kiri if i % 2 == 0 else kanan
    with kolom:
        st.markdown(f'<div class="kotak"><b>{judul}</b><br><span style="opacity:.75">{isi}</span></div>',
                    unsafe_allow_html=True)

st.info("Pilih halaman melalui menu di sisi kiri untuk memulai demo.", icon="👈")

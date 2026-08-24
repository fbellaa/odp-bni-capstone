"""Titik masuk aplikasi demo Streamlit — segmen kredit komersial.

Jalankan dari folder app/ui:

    streamlit run app.py

Susunan halaman mengikuti proposal bagian 9.3. Seluruh data masih dummy;
lihat lib/dummy_data.py untuk titik sambung ke FastAPI nanti.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.express as px
import streamlit as st

from lib import dummy_data, mock_engine
from lib.format import miliar, persen, rupiah
from lib.tampilan import (
    JUDUL_APLIKASI,
    badge,
    plot_bmpk,
    setup_halaman,
    sidebar_status,
)

setup_halaman("Beranda", "🏦")
sidebar_status()

st.title(JUDUL_APLIKASI)
st.caption(
    "Sistem pendukung keputusan kredit segmen komersial — debitur menengah dengan penjualan "
    "tahunan Rp 30 sampai 300 miliar. Memadukan data engineering, pemodelan risiko, analisis "
    "jaringan grup usaha dan rantai pasok, serta lapisan generative AI sebagai orkestrator model."
)

st.markdown(
    " ".join([
        badge("RISK ASSESSMENT", "#2f6f9f"),
        badge("GRAPH ANALYTICS", "#2e8b6f"),
        badge("DECISION SUPPORT", "#7b5ea7"),
        badge("GROUP EXPOSURE", "#c9721c"),
        badge("PERSONALIZATION", "#8e5572"),
    ]),
    unsafe_allow_html=True,
)

st.warning(
    "Aplikasi ini berjalan dengan **data dummy**. Model PD/LGD, lapisan graf, dan agen "
    "belum tersambung — seluruh angka dihitung oleh `lib/mock_engine.py` secara deterministik "
    "supaya interaksi demo tetap responsif.",
    icon="⚠️",
)

df = dummy_data.daftar_pengajuan()
grup = dummy_data.daftar_grup()

st.subheader("Ringkasan portofolio komersial demo")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Pengajuan pada pipeline", f"{len(df):,}".replace(",", "."))
k2.metric("Total plafon diminta", rupiah(df["plafon_diminta"].sum(), singkat=True))
k3.metric("Rata-rata plafon", miliar(df["plafon_diminta"].mean(), 0))
k4.metric("Rata-rata PD", persen(df["pd"].mean()))
k5.metric("Expected loss portofolio", rupiah(df["expected_loss"].sum(), singkat=True))

kiri, kanan = st.columns([3, 2])

with kiri:
    st.markdown("**Sebaran rating internal pada pipeline**")
    urutan = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    hitung = (
        df["grade"].value_counts().reindex(urutan).fillna(0).reset_index()
    )
    hitung.columns = ["grade", "jumlah"]
    fig = px.bar(
        hitung, x="grade", y="jumlah", color="grade",
        color_discrete_map=mock_engine.WARNA_GRADE,
        labels={"grade": "Rating internal", "jumlah": "Jumlah pengajuan"},
    )
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with kanan:
    st.markdown("**Grup usaha dengan ruang BMPK paling tipis**")
    teratas = grup.iloc[0]
    st.plotly_chart(
        plot_bmpk(teratas["eksposur_grup"], 0.0),
        use_container_width=True,
    )
    st.caption(
        f"{teratas['grup_usaha']} — {int(teratas['jumlah_entitas'])} entitas, "
        f"{persen(teratas['porsi_bmpk'], 0)} dari batas terpakai."
    )

st.divider()

st.subheader("Peta halaman")
kiri, kanan = st.columns(2)
halaman = [
    ("1 · Copilot pengajuan",
     "Kolom teks bebas, jejak langkah agen yang muncul bertahap, gerbang kepatuhan, hasil skor "
     "dan reason code, tombol unduh credit memo."),
    ("2 · Simulasi what-if",
     "Slider plafon, tenor, struktur agunan, dan asumsi EBITDA; skor, pricing, covenant, dan "
     "ekspektasi kerugian diperbarui seketika."),
    ("3 · Struktur grup dan jaringan",
     "Subgraf ego dua hop, penelusuran kepemilikan sampai pemilik manfaat, penyorotan klaster, "
     "panel pola anomali yang terdeteksi."),
    ("4 · Portofolio dan eksposur grup",
     "Konsentrasi eksposur per grup dan sektor, posisi terhadap BMPK, uji tekanan simpul kritis, "
     "tabel counterparty penting."),
    ("5 · Kesehatan model",
     "Metrik berjalan, indeks stabilitas populasi, hasil uji ablasi fitur graf, evaluasi agen, "
     "kelulusan uji kualitas data."),
    ("6 · Dashboard BI",
     "Metabase disematkan sebagai iframe untuk lapisan eksekutif."),
]
for i, (judul, isi) in enumerate(halaman):
    kolom = kiri if i % 2 == 0 else kanan
    with kolom:
        st.markdown(f'<div class="kotak"><b>{judul}</b><br><span style="opacity:.75">{isi}</span></div>',
                    unsafe_allow_html=True)

st.info("Pilih halaman melalui menu di sisi kiri untuk memulai demo.", icon="👈")

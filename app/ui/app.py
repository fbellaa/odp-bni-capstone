"""Titik masuk aplikasi Streamlit — segmen kredit komersial.

Jalankan dari folder app/ui:

    streamlit run app.py

Ringkasan portofolio pada halaman ini dibaca dari lapisan emas
(`data/gold/*.parquet`), bukan dari data karangan. Halaman-halaman berikutnya
menyambungnya dengan artefak model pada `ml/models`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.express as px
import streamlit as st

from lib import model_nyata as mn
from lib.format import cacah, miliar, persen, rupiah
from lib.tampilan import (
    ABU,
    DERET_KATEGORI,
    JINGGA,
    JINGGA_GELAP,
    TOSCA,
    TOSCA_GELAP,
    TOSCA_TUA,
    gaya_plot,
    hero,
    judul_bagian,
    kartu,
    setup_halaman,
    sidebar_status,
)

setup_halaman("Beranda")
sidebar_status()

abt = mn.gold("abt_pd")
grup = mn.gold("dim_grup_usaha")
pengajuan = mn.gold("fact_pengajuan")
status = mn.status_lapisan_model()

hero(
    "CC",
    "Agentic AI Copilot untuk Keputusan Kredit Komersial",
    "Sistem pendukung keputusan kredit segmen komersial — debitur menengah dengan penjualan "
    "tahunan Rp 30 sampai 300 miliar. Memadukan data engineering, pemodelan risiko, analisis "
    "jaringan grup usaha, serta lapisan generative AI sebagai orkestrator model.",
    [
        ("model produksi", "PD · EWS · LGD"),
        ("baris ABT PD", cacah(status["baris_abt"]) if status["gold"] else "—"),
        ("lapisan", "Bronze → Silver → Gold"),
        ("keputusan", "Sistem menyarankan, komite memutus"),
    ],
)

if abt is None:
    st.error(
        "Lapisan emas belum dibangun. Jalankan pipeline lebih dulu, atau salin "
        "`data/gold/*.parquet` ke tempatnya.",
    )
    st.stop()

# ------------------------------------------------------------------ ringkas
judul_bagian("Ringkasan portofolio pada lapisan emas",
             "Angka dihitung langsung dari tabel emas setiap kali halaman dibuka.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Pengajuan pada ABT PD", cacah(len(abt)))
k2.metric("Total plafon diminta", rupiah(abt["app_plafon_diminta_rp"].sum(), singkat=True))
k3.metric("Rata-rata plafon", miliar(abt["app_plafon_diminta_rp"].mean(), 0))
k4.metric("Tingkat default 12 bulan", persen(abt["y_default_12bln"].mean()),
          help="Kejadian pada data berlabel — dasar mengapa evaluasi model berpusat pada recall.")
k5.metric("Grup usaha terdaftar", cacah(len(grup)) if grup is not None else "—")

kiri, kanan = st.columns([3, 2], gap="large")

with kiri:
    st.markdown("**Sebaran rating internal pada portofolio**")
    urutan = [r for r in mn.URUTAN_RATING if r in set(abt["app_rating_internal"])]
    hitung = (
        abt.groupby("app_rating_internal")
        .agg(jumlah=("application_id", "size"), tingkat_default=("y_default_12bln", "mean"))
        .reindex(urutan).reset_index()
    )
    fig = px.bar(
        hitung, x="app_rating_internal", y="jumlah", color="tingkat_default",
        color_continuous_scale=[[0, TOSCA], [0.5, JINGGA], [1, JINGGA_GELAP]],
        labels={"app_rating_internal": "Rating internal", "jumlah": "Jumlah pengajuan",
                "tingkat_default": "Tingkat default"},
    )
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%", title=None))
    st.plotly_chart(gaya_plot(fig, 330), use_container_width=True)
    st.caption("Warna batang menyatakan tingkat default historis kelas rating tersebut.")

with kanan:
    st.markdown("**Konsentrasi eksposur grup usaha**")
    if grup is not None and "penjualan_grup_rp" in grup:
        teratas = grup.nlargest(8, "penjualan_grup_rp").sort_values("penjualan_grup_rp")
        fig2 = px.bar(
            teratas, x="penjualan_grup_rp", y="nama_grup", orientation="h",
            labels={"penjualan_grup_rp": "Penjualan grup (Rp)", "nama_grup": ""},
        )
        fig2.update_traces(marker_color=TOSCA_TUA)
        st.plotly_chart(gaya_plot(fig2, 330), use_container_width=True)
        st.caption(
            cacah(len(grup))
            + " grup usaha terbentuk dari resolusi kepemilikan dan pengendalian pada lapisan graf."
        )
    else:
        st.info("Tabel `dim_grup_usaha` belum tersedia.")

if pengajuan is not None and "keputusan" in pengajuan:
    st.markdown("**Keputusan historis pada berkas pengajuan**")
    ringkas = pengajuan["keputusan"].value_counts().reset_index()
    ringkas.columns = ["keputusan", "jumlah"]
    fig3 = px.bar(ringkas, x="jumlah", y="keputusan", orientation="h",
                  color="keputusan", color_discrete_sequence=DERET_KATEGORI,
                  labels={"jumlah": "Jumlah berkas", "keputusan": ""})
    fig3.update_layout(showlegend=False)
    st.plotly_chart(gaya_plot(fig3, 240), use_container_width=True)

# -------------------------------------------------------------- peta halaman
judul_bagian("Peta halaman", "Empat halaman kerja, satu halaman eksekutif.")

halaman = [
    ("01 · Copilot pengajuan",
     "Chat relationship manager plus unggahan PDF laporan keuangan, data kepemilikan, dan "
     "rekening koran. Copilot membaca berkas, memanggil tool, lalu mengeluarkan PD, LGD, "
     "posisi klaster, gerbang kepatuhan, dan draft credit memo.", TOSCA_TUA),
    ("02 · Simulasi what-if",
     "Sandbox perhitungan: plafon, tenor, struktur agunan, dan asumsi EBITDA digeser, lalu "
     "skor, pricing, covenant, dan expected loss diperbarui seketika.", TOSCA),
    ("03 · Struktur grup dan jaringan",
     "Subgraf ego dua hop, penelusuran kepemilikan sampai pemilik manfaat akhir, penyorotan "
     "klaster, dan panel pola anomali struktur.", TOSCA_GELAP),
    ("04 · Kesehatan model",
     "Metrik PD, EWS, dan LGD dihitung ulang dari artefak model di atas data emas, dengan "
     "recall sebagai ukuran utama; ditambah PSI dan hasil judge arena Qwen 14B.", JINGGA),
    ("05 · Dashboard BI",
     "Metabase disematkan untuk lapisan eksekutif — masih dalam pembangunan.", ABU),
]
kiri, kanan = st.columns(2)
for i, (judul, isi, warna) in enumerate(halaman):
    kolom = kiri if i % 2 == 0 else kanan
    with kolom:
        st.markdown(
            kartu(judul, isi, warna=warna), unsafe_allow_html=True,
        )

st.info("Pilih halaman melalui menu di sisi kiri untuk memulai.")

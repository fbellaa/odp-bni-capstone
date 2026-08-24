"""Halaman 5 — Kesehatan model.

Metrik berjalan, indeks stabilitas populasi, hasil uji ablasi fitur graf, dan
kelulusan uji kualitas data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from lib import dummy_data
from lib.tampilan import setup_halaman, sidebar_status

setup_halaman("Kesehatan model", "🩺")
sidebar_status()

st.title("5 · Kesehatan model")
st.caption("Metrik berjalan, stabilitas populasi, kontribusi lapisan graf, dan gerbang kualitas data.")

metrik = dummy_data.metrik_model()
ablasi = dummy_data.uji_ablasi_graf()
psi = dummy_data.population_stability()
gerbang = dummy_data.gerbang_kualitas_data()

utama = metrik.iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("AUC model PD", f"{utama['auc']:.3f}")
k2.metric("Gini", f"{utama['gini']:.3f}")
k3.metric("Kolmogorov-Smirnov", f"{utama['ks']:.3f}")
k4.metric("Brier score", f"{utama['brier']:.3f}", help="Semakin kecil semakin baik kalibrasi.")

st.divider()

tab_metrik, tab_ablasi, tab_psi, tab_kualitas = st.tabs(
    ["Metrik per model", "Uji ablasi fitur graf", "Stabilitas populasi", "Gerbang kualitas data"]
)

with tab_metrik:
    st.dataframe(
        metrik.rename(columns={"model": "Model", "auc": "AUC", "gini": "Gini",
                               "ks": "KS", "brier": "Brier", "status": "Status"}),
        use_container_width=True, hide_index=True,
    )
    dist = dummy_data.distribusi_skor()
    fig = px.histogram(dist, x="pd", color="periode", barmode="overlay", nbins=45,
                       histnorm="probability density",
                       labels={"pd": "Probability of default", "periode": "Periode"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Perbandingan sebaran skor data pelatihan terhadap bulan berjalan.")

with tab_ablasi:
    st.caption(
        "Kontribusi lapisan graf diukur dengan melatih model PD tanpa blok fitur graf, "
        "lalu dengan blok tersebut. Selisihnya dilaporkan apa adanya."
    )
    panjang = ablasi.melt(id_vars="varian", var_name="metrik", value_name="nilai")
    fig = px.bar(panjang, x="metrik", y="nilai", color="varian", barmode="group",
                 labels={"metrik": "Metrik", "nilai": "Nilai", "varian": "Varian model"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[0.3, 0.9])
    st.plotly_chart(fig, use_container_width=True)

    selisih = pd.DataFrame({
        "Metrik": ["AUC", "Gini", "KS"],
        "Tanpa fitur graf": ablasi.iloc[0][["auc", "gini", "ks"]].values,
        "Dengan fitur graf": ablasi.iloc[1][["auc", "gini", "ks"]].values,
    })
    selisih["Selisih"] = (selisih["Dengan fitur graf"] - selisih["Tanpa fitur graf"]).round(4)
    st.dataframe(selisih, use_container_width=True, hide_index=True)

with tab_psi:
    st.caption("PSI < 0,10 stabil · 0,10–0,25 perlu perhatian · > 0,25 pergeseran nyata.")
    warna = ["#2e8b6f" if v < 0.10 else ("#c9721c" if v < 0.25 else "#c0392b") for v in psi["psi"]]
    fig = px.bar(psi, x="psi", y="fitur", orientation="h",
                 labels={"psi": "Population stability index", "fitur": ""})
    fig.update_traces(marker_color=warna)
    fig.add_vline(x=0.10, line_dash="dot", line_color="#c9721c")
    fig.add_vline(x=0.25, line_dash="dot", line_color="#c0392b")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    perhatian = psi[psi["psi"] >= 0.10]
    if len(perhatian):
        st.warning(
            "Fitur yang perlu perhatian: " + ", ".join(f"`{f}`" for f in perhatian["fitur"]),
            icon="⚠️",
        )
    else:
        st.success("Seluruh fitur berada pada rentang stabil.", icon="✅")

with tab_kualitas:
    ikon = {"Lulus": "✅", "Lulus dengan perbaikan": "🛠️", "Perlu telaah": "⚠️"}
    tampil = gerbang.copy()
    tampil["hasil"] = tampil["hasil"].map(lambda h: f"{ikon.get(h, '•')} {h}")
    st.dataframe(
        tampil.rename(columns={"pemeriksaan": "Pemeriksaan", "hasil": "Hasil",
                               "baris_karantina": "Baris masuk karantina"}),
        use_container_width=True, hide_index=True,
    )
    total = int(gerbang["baris_karantina"].sum())
    st.metric("Total baris pada tabel karantina", f"{total:,}".replace(",", "."))
    st.caption(
        "Tingkat kekotoran data diinjeksi sendiri dan didokumentasikan pada sebuah spesifikasi, "
        "sehingga kualitas sebelum dan sesudah pipeline dapat diukur secara objektif."
    )

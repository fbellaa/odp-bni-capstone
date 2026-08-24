"""Halaman 4 — Portofolio dan komunitas.

Konsentrasi eksposur per komunitas, uji tekanan simpul kritis, dan tabel
counterparty penting. Halaman ini ditujukan untuk risk officer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from lib import dummy_data
from lib.format import persen, rupiah
from lib.tampilan import setup_halaman, sidebar_status

setup_halaman("Portofolio dan komunitas", "📊")
sidebar_status()

st.title("4 · Portofolio dan komunitas")
st.caption("Sebaran skor, konsentrasi eksposur per komunitas usaha, dan uji tekanan simpul kritis.")

df = dummy_data.daftar_pengajuan()
komunitas = dummy_data.daftar_komunitas()

with st.sidebar:
    st.divider()
    f_sektor = st.multiselect("Sektor", dummy_data.SEKTOR, default=[])
    f_wilayah = st.multiselect("Wilayah", dummy_data.WILAYAH, default=[])
    f_grade = st.multiselect("Grade", sorted(df["grade"].unique()), default=[])

tersaring = df.copy()
if f_sektor:
    tersaring = tersaring[tersaring["sektor"].isin(f_sektor)]
if f_wilayah:
    tersaring = tersaring[tersaring["wilayah"].isin(f_wilayah)]
if f_grade:
    tersaring = tersaring[tersaring["grade"].isin(f_grade)]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Pengajuan", f"{len(tersaring):,}".replace(",", "."))
m2.metric("Total plafon", rupiah(tersaring["plafon_diminta"].sum(), singkat=True))
m3.metric("Rata-rata PD", persen(tersaring["pd"].mean()))
m4.metric("Expected loss", rupiah(tersaring["expected_loss"].sum(), singkat=True))
m5.metric("Tingkat persetujuan",
          persen((tersaring["keputusan"] != "TOLAK").mean() if len(tersaring) else 0.0))

st.divider()

tab_sebaran, tab_komunitas, tab_tekanan, tab_ambang = st.tabs(
    ["Sebaran risiko", "Konsentrasi komunitas", "Uji tekanan", "Pertukaran ambang skor"]
)

with tab_sebaran:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(tersaring, x="pd", nbins=40, color="grade",
                           labels={"pd": "Probability of default", "count": "Jumlah"})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ringkas = (
            tersaring.groupby("sektor")
            .agg(jumlah=("application_id", "count"), pd_rata=("pd", "mean"),
                 plafon=("plafon_diminta", "sum"))
            .reset_index()
        )
        fig = px.scatter(ringkas, x="pd_rata", y="plafon", size="jumlah", color="sektor",
                         labels={"pd_rata": "PD rata-rata", "plafon": "Total plafon (Rp)"})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    heat = tersaring.pivot_table(index="sektor", columns="wilayah", values="pd", aggfunc="mean")
    fig = px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto",
                    labels=dict(color="PD rata-rata"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab_komunitas:
    st.caption(
        "Komunitas hasil Louvain pada graf berbobot menjadi unit pemantauan konsentrasi risiko."
    )
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.scatter(
            komunitas, x="npl_komunitas", y="eksposur_bank", size="jumlah_anggota",
            color="pd_rata_tetangga", hover_name="nama", color_continuous_scale="RdYlGn_r",
            labels={"npl_komunitas": "NPL komunitas", "eksposur_bank": "Eksposur bank (Rp)",
                    "pd_rata_tetangga": "PD rata tetangga"},
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        atas = komunitas.nlargest(6, "eksposur_bank")
        fig = px.bar(atas, x="eksposur_bank", y="nama", orientation="h",
                     labels={"eksposur_bank": "Eksposur bank (Rp)", "nama": ""})
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    tabel = komunitas.copy()
    tabel["eksposur_bank"] = tabel["eksposur_bank"].map(lambda v: rupiah(v, singkat=True))
    tabel["npl_komunitas"] = tabel["npl_komunitas"].map(persen)
    tabel["pd_rata_tetangga"] = tabel["pd_rata_tetangga"].map(persen)
    tabel["modularitas"] = tabel["modularitas"].round(3)
    st.dataframe(
        tabel.rename(columns={
            "community_id": "ID", "nama": "Komunitas", "jumlah_anggota": "Anggota",
            "eksposur_bank": "Eksposur bank", "npl_komunitas": "NPL",
            "pd_rata_tetangga": "PD rata tetangga", "modularitas": "Modularitas",
            "simpul_kritis": "Simpul kritis",
        }),
        use_container_width=True, hide_index=True,
    )

with tab_tekanan:
    st.caption("Skenario: bila satu simpul kritis terganggu, berapa tambahan pencadangan yang timbul?")
    cp = dummy_data.counterparty_penting()
    c1, c2 = st.columns([2, 1])
    simpul = c1.selectbox(
        "Simpul kritis", cp["entity_id"].tolist(),
        format_func=lambda e: f"{e} — {cp.loc[cp['entity_id'] == e, 'nama'].iat[0]}",
    )
    guncangan = c2.slider("Tingkat guncangan", 0.0, 3.0, 1.0, step=0.25,
                          help="1,0 = simpul berhenti beroperasi; di atas itu efek penularan diperbesar.")

    hasil = dummy_data.uji_tekanan(simpul, guncangan)
    k1, k2, k3 = st.columns(3)
    k1.metric("Debitur terdampak", hasil["debitur_terdampak"])
    k2.metric("Eksposur terdampak", rupiah(hasil["eksposur_terdampak"], singkat=True))
    k3.metric("Tambahan pencadangan", rupiah(hasil["tambahan_pencadangan"], singkat=True),
              delta=f"PD rata +{hasil['kenaikan_pd_rata'] * 100:.2f} pp", delta_color="inverse")

    kurva = pd.DataFrame({
        "guncangan": [g / 4 for g in range(0, 13)],
    })
    kurva["tambahan_pencadangan"] = [
        dummy_data.uji_tekanan(simpul, g)["tambahan_pencadangan"] / 1_000_000 for g in kurva["guncangan"]
    ]
    fig = px.line(kurva, x="guncangan", y="tambahan_pencadangan", markers=True,
                  labels={"guncangan": "Tingkat guncangan",
                          "tambahan_pencadangan": "Tambahan pencadangan (Rp juta)"})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab_ambang:
    st.caption(
        "Pertukaran antara tingkat persetujuan dan ekspektasi kerugian pada setiap ambang skor — "
        "bahan utama kasus bisnis."
    )
    baris = []
    for ambang in [0.02, 0.035, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16]:
        diterima = df[df["pd"] <= ambang]
        baris.append({
            "Ambang PD": persen(ambang),
            "Tingkat persetujuan": persen(len(diterima) / len(df)),
            "Volume penyaluran": rupiah(diterima["limit_usulan"].sum(), singkat=True),
            "Expected loss": rupiah(diterima["expected_loss"].sum(), singkat=True),
            "EL terhadap penyaluran": persen(
                diterima["expected_loss"].sum() / max(diterima["limit_usulan"].sum(), 1)
            ),
        })
    st.dataframe(pd.DataFrame(baris), use_container_width=True, hide_index=True)

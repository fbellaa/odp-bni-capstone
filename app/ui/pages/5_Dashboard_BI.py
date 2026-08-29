"""Halaman 6 — Dashboard BI.

Metabase disematkan sebagai iframe untuk lapisan eksekutif (pola 2 pada
proposal 9.1). Selama Metabase belum dijalankan, halaman ini menampilkan
pratinjau pengganti dengan data dummy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from lib import dummy_data
from lib.format import miliar, persen, rupiah
from lib.tampilan import TOSCA, JINGGA, JINGGA_TUA, JINGGA_GELAP, gaya_plot, hero, setup_halaman, sidebar_status

setup_halaman("Dashboard BI")
sidebar_status()

hero(
    "05",
    "Dashboard BI",
    "Lapisan eksekutif dan operasional portofolio komersial disajikan melalui Metabase di "
    "belakang reverse proxy. Metabase masih dalam pembangunan, jadi halaman ini menahan "
    "tempatnya dengan pratinjau pengganti.",
    [("penyaji", "Metabase"), ("status", "dalam pembangunan"), ("penyematan", "iframe /bi/")],
)

# Di susunan docker-compose, nginx meneruskan /bi/ ke metabase:3000.
METABASE_URL = os.getenv("METABASE_EMBED_URL", "")

if METABASE_URL:
    st.success(f"Menyematkan dashboard dari `{METABASE_URL}`.")
    components.iframe(METABASE_URL, height=900, scrolling=True)
    st.stop()

st.info(
    "`METABASE_EMBED_URL` belum diatur, jadi halaman ini menampilkan **pratinjau pengganti** "
    "dengan data dummy. Setelah Metabase jalan, isi variabel lingkungan tersebut "
    "(misalnya `http://localhost/bi/public/dashboard/<uuid>`) dan iframe akan tampil di sini.",
)

df = dummy_data.daftar_pengajuan()
grup = dummy_data.daftar_grup()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Pengajuan 6 bulan", f"{len(df):,}".replace(",", "."))
k2.metric("Disetujui", persen((df["keputusan"] == "SETUJU").mean()))
k3.metric("Volume usulan limit", rupiah(df["limit_usulan"].sum(), singkat=True))
k4.metric("Expected loss", rupiah(df["expected_loss"].sum(), singkat=True))
k5.metric("Grup di atas 80% BMPK", int((grup["porsi_bmpk"] >= 0.80).sum()))

bulanan = (
    df.assign(bulan=df["tanggal"].dt.to_period("M").dt.to_timestamp())
    .groupby(["bulan", "keputusan"], as_index=False)
    .agg(jumlah=("application_id", "count"), limit=("limit_usulan", "sum"))
)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(bulanan, x="bulan", y="jumlah", color="keputusan",
                 color_discrete_map={"SETUJU": TOSCA, "SETUJU DENGAN SYARAT": JINGGA,
                                     "PERLU PENYESUAIAN": JINGGA_TUA, "TOLAK": JINGGA_GELAP},
                 labels={"bulan": "Bulan", "jumlah": "Jumlah pengajuan", "keputusan": "Keputusan"})
    fig.update_layout(title="Pengajuan komersial per bulan menurut keputusan")
    st.plotly_chart(gaya_plot(fig, 380), use_container_width=True)
with c2:
    per_fasilitas = df.groupby("jenis_fasilitas", as_index=False).agg(
        limit=("limit_usulan", "sum"), pd_rata=("pd", "mean")
    )
    per_fasilitas["limit_miliar"] = per_fasilitas["limit"] / 1e9
    fig = px.bar(per_fasilitas.sort_values("limit_miliar"), x="limit_miliar", y="jenis_fasilitas",
                 orientation="h", color="pd_rata", color_continuous_scale="RdYlGn_r",
                 labels={"limit_miliar": "Usulan limit (Rp miliar)", "jenis_fasilitas": "",
                         "pd_rata": "PD rata"})
    fig.update_layout(title="Usulan penyaluran per jenis fasilitas")
    st.plotly_chart(gaya_plot(fig, 380), use_container_width=True)

c3, c4 = st.columns([3, 2])
with c3:
    # Pengajuan yang ditolak berlimit nol; disaring supaya bobot treemap tidak nol.
    disalurkan = df[df["limit_usulan"] > 0]
    fig = px.treemap(
        disalurkan, path=["sektor", "grade"], values="limit_usulan", color="pd",
        color_continuous_scale="RdYlGn_r",
        labels={"limit_usulan": "Usulan limit", "pd": "PD"},
    )
    fig.update_layout(title="Komposisi usulan penyaluran per sektor dan rating internal")
    st.plotly_chart(gaya_plot(fig, 470), use_container_width=True)
    st.caption("Hanya pengajuan dengan usulan limit di atas nol yang masuk ke komposisi ini.")
with c4:
    atas = grup.nlargest(8, "eksposur_grup").copy()
    atas["eksposur_miliar"] = atas["eksposur_grup"] / 1e9
    fig = px.bar(atas.sort_values("eksposur_miliar"), x="eksposur_miliar", y="grup_usaha",
                 orientation="h", color="porsi_bmpk", color_continuous_scale="RdYlGn_r",
                 labels={"eksposur_miliar": "Eksposur gabungan (Rp miliar)", "grup_usaha": "",
                         "porsi_bmpk": "Porsi BMPK"})
    fig.update_layout(title="Delapan grup usaha dengan eksposur terbesar")
    st.plotly_chart(gaya_plot(fig, 470), use_container_width=True)
    st.caption(
        f"Eksposur gabungan terbesar {miliar(grup['eksposur_grup'].max(), 0)} dari batas "
        f"{miliar(grup['eksposur_grup'].max() / max(grup['porsi_bmpk'].max(), 1e-9), 0)}."
    )

with st.expander("Catatan penempatan Metabase pada susunan layanan"):
    st.code(
        """http://localhost/                 nginx (reverse proxy)
  /            -> streamlit:8501   copilot, what-if, graf, agent trace
  /bi/         -> metabase:3000    dashboard eksekutif dan operasional
  /api/        -> fastapi:8000     skoring, subgraf, agen, SSE
  /mlflow/     -> mlflow:5000      registry model
  /airflow/    -> airflow:8080     orkestrasi pipeline""",
        language="text",
    )

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
from lib.format import rupiah
from lib.tampilan import setup_halaman, sidebar_status

setup_halaman("Dashboard BI", "📈")
sidebar_status()

st.title("6 · Dashboard BI")
st.caption("Lapisan eksekutif dan operasional disajikan melalui Metabase di belakang reverse proxy.")

# Di susunan docker-compose, nginx meneruskan /bi/ ke metabase:3000.
METABASE_URL = os.getenv("METABASE_EMBED_URL", "")

if METABASE_URL:
    st.success(f"Menyematkan dashboard dari `{METABASE_URL}`.", icon="🔗")
    components.iframe(METABASE_URL, height=900, scrolling=True)
    st.stop()

st.info(
    "`METABASE_EMBED_URL` belum diatur, jadi halaman ini menampilkan **pratinjau pengganti** "
    "dengan data dummy. Setelah Metabase jalan, isi variabel lingkungan tersebut "
    "(misalnya `http://localhost/bi/public/dashboard/<uuid>`) dan iframe akan tampil di sini.",
    icon="ℹ️",
)

df = dummy_data.daftar_pengajuan()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Pengajuan 6 bulan", f"{len(df):,}".replace(",", "."))
k2.metric("Disetujui", f"{(df['keputusan'] == 'SETUJU').mean() * 100:.1f}%".replace(".", ","))
k3.metric("Volume usulan limit", rupiah(df["limit_usulan"].sum(), singkat=True))
k4.metric("Expected loss", rupiah(df["expected_loss"].sum(), singkat=True))

bulanan = (
    df.assign(bulan=df["tanggal"].dt.to_period("M").dt.to_timestamp())
    .groupby(["bulan", "keputusan"], as_index=False)
    .agg(jumlah=("application_id", "count"), limit=("limit_usulan", "sum"))
)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(bulanan, x="bulan", y="jumlah", color="keputusan",
                 labels={"bulan": "Bulan", "jumlah": "Jumlah pengajuan", "keputusan": "Keputusan"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                      title="Pengajuan per bulan menurut keputusan")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    per_wilayah = df.groupby("wilayah", as_index=False).agg(
        limit=("limit_usulan", "sum"), pd_rata=("pd", "mean")
    )
    fig = px.bar(per_wilayah.sort_values("limit"), x="limit", y="wilayah", orientation="h",
                 color="pd_rata", color_continuous_scale="RdYlGn_r",
                 labels={"limit": "Usulan limit (Rp)", "wilayah": "", "pd_rata": "PD rata"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                      title="Usulan penyaluran per wilayah")
    st.plotly_chart(fig, use_container_width=True)

# Pengajuan yang ditolak berlimit nol; disaring supaya bobot treemap tidak nol.
disalurkan = df[df["limit_usulan"] > 0]
fig = px.treemap(
    disalurkan, path=["sektor", "grade"], values="limit_usulan", color="pd",
    color_continuous_scale="RdYlGn_r",
    labels={"limit_usulan": "Usulan limit", "pd": "PD"},
)
fig.update_layout(height=470, margin=dict(l=10, r=10, t=40, b=10),
                  title="Komposisi usulan penyaluran per sektor dan grade")
st.plotly_chart(fig, use_container_width=True)
st.caption("Hanya pengajuan dengan usulan limit di atas nol yang masuk ke komposisi ini.")

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

"""Komponen tampilan yang dipakai bersama oleh seluruh halaman."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import mock_engine
from lib.format import persen, rupiah

PALET_TIPE = {
    "UMKM": "#2f6f9f",
    "Individu": "#7b5ea7",
    "Merchant": "#2e8b6f",
    "Distributor": "#c9721c",
    "Atribut": "#b03a48",
    "Agunan": "#7a7a7a",
}

PALET_KOMUNITAS = [
    "#2f6f9f", "#c9721c", "#2e8b6f", "#b03a48", "#7b5ea7", "#8a7f2e",
    "#3f8fa8", "#a4553a", "#5f7d3a", "#8e5572", "#3b6b8f", "#9c6b1f",
]


def setup_halaman(judul: str, ikon: str = "•") -> None:
    st.set_page_config(page_title=f"{judul} · Credit Copilot", page_icon=ikon, layout="wide")
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
          div[data-testid="stMetricValue"] {font-size: 1.5rem;}
          .badge {display:inline-block; padding:.18rem .6rem; border-radius:999px;
                  font-size:.75rem; font-weight:600; letter-spacing:.02em;}
          .kotak {border:1px solid rgba(128,128,128,.28); border-radius:.5rem;
                  padding:.85rem 1rem; margin-bottom:.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_status() -> None:
    with st.sidebar:
        st.caption("AGENTIC AI COPILOT · KEPUTUSAN KREDIT UMKM")
        st.info(
            "**Mode demo — data dummy.**\n\n"
            "Seluruh angka pada aplikasi ini sintetis dan tidak berasal dari data nasabah. "
            "Lapisan FastAPI, model, dan graf belum tersambung.",
            icon="🧪",
        )
        st.caption("Status: under development · Agustus 2026")


def badge(teks: str, warna: str) -> str:
    return (
        f'<span class="badge" style="background:{warna}22;color:{warna};'
        f'border:1px solid {warna}55">{teks}</span>'
    )


def badge_keputusan(keputusan: str) -> str:
    warna = {"SETUJU": "#1b7f4b", "SETUJU DENGAN SYARAT": "#b58900", "TOLAK": "#c0392b"}
    return badge(keputusan, warna.get(keputusan, "#666666"))


def kartu_hasil(hasil: mock_engine.HasilSkor, plafon_diminta: float) -> None:
    """Baris metrik inti keputusan."""
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Probability of default", persen(hasil.pd), help="PD terkalibrasi 12 bulan ke depan")
    k2.metric("Grade risiko", hasil.grade)
    k3.metric("Expected loss", rupiah(hasil.expected_loss, singkat=True),
              help="PD × LGD × EAD")
    k4.metric(
        "Usulan limit",
        rupiah(hasil.limit_usulan, singkat=True),
        delta=None if hasil.limit_usulan >= plafon_diminta
        else f"-{rupiah(plafon_diminta - hasil.limit_usulan, singkat=True)} dari permintaan",
        delta_color="inverse",
    )
    k5.metric("Usulan pricing", persen(hasil.pricing), help="Biaya dana + operasional + margin + expected loss")


def plot_kontribusi(kontribusi, jumlah: int = 7) -> go.Figure:
    """Bar horizontal ala SHAP untuk reason code."""
    data = list(kontribusi)[:jumlah][::-1]
    fig = go.Figure(
        go.Bar(
            x=[k.dampak for k in data],
            y=[k.fitur for k in data],
            orientation="h",
            marker_color=["#c0392b" if k.dampak > 0 else "#2e8b6f" for k in data],
            customdata=[k.nilai for k in data],
            hovertemplate="<b>%{y}</b><br>Nilai: %{customdata}<br>Dampak: %{x:+.3f} log-odds<extra></extra>",
        )
    )
    fig.update_layout(
        height=60 + 34 * len(data),
        margin=dict(l=8, r=8, t=10, b=28),
        xaxis_title="Dampak terhadap risiko (log-odds)",
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def _tata_letak_graf(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Spring layout sederhana (Fruchterman-Reingold ringkas) tanpa dependensi tambahan."""
    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(nodes["id"])
        if not edges.empty:
            g.add_edges_from(zip(edges["source"], edges["target"]))
        return nx.spring_layout(g, seed=7, k=1.1 / math.sqrt(max(len(g), 1)))
    except Exception:
        # Cadangan: tata letak radial per hop bila networkx tidak terpasang.
        posisi = {}
        for hop, kelompok in nodes.groupby("hop"):
            n = len(kelompok)
            for i, nid in enumerate(kelompok["id"]):
                sudut = 2 * math.pi * i / max(n, 1)
                r = float(hop)
                posisi[nid] = (r * math.cos(sudut), r * math.sin(sudut))
        return posisi


def plot_graf(nodes: pd.DataFrame, edges: pd.DataFrame, warnai: str = "tipe",
              sorot: str | None = None) -> go.Figure:
    """Graf simpul-tepi dengan Plotly (lihat proposal 9.4)."""
    posisi = _tata_letak_graf(nodes, edges)

    garis_x, garis_y = [], []
    for _, e in edges.iterrows():
        if e["source"] not in posisi or e["target"] not in posisi:
            continue
        x0, y0 = posisi[e["source"]]
        x1, y1 = posisi[e["target"]]
        garis_x += [x0, x1, None]
        garis_y += [y0, y1, None]

    jejak_edge = go.Scatter(
        x=garis_x, y=garis_y, mode="lines", hoverinfo="skip",
        line=dict(width=0.9, color="rgba(130,130,130,.45)"),
    )

    if warnai == "komunitas":
        warna = [PALET_KOMUNITAS[int(c) % len(PALET_KOMUNITAS)] for c in nodes["community_id"]]
    else:
        warna = [PALET_TIPE.get(t, "#888888") for t in nodes["tipe"]]

    ukuran = [26 if nid == sorot else (16 if h == 1 else 11)
              for nid, h in zip(nodes["id"], nodes["hop"])]
    garis_tepi = ["#111111" if nid == sorot else "rgba(255,255,255,.6)" for nid in nodes["id"]]

    teks = [
        f"<b>{r.id}</b><br>Tipe: {r.tipe}<br>Hop: {r.hop}"
        f"<br>Komunitas: {r.community_id}"
        + (f"<br>PD: {r.pd * 100:.2f}%" if isinstance(r.pd, float) and not math.isnan(r.pd) else "")
        for r in nodes.itertuples()
    ]

    jejak_node = go.Scatter(
        x=[posisi[n][0] for n in nodes["id"]],
        y=[posisi[n][1] for n in nodes["id"]],
        mode="markers",
        marker=dict(size=ukuran, color=warna, line=dict(width=1.4, color=garis_tepi)),
        text=teks,
        hovertemplate="%{text}<extra></extra>",
    )

    fig = go.Figure([jejak_edge, jejak_node])
    fig.update_layout(
        height=520, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        hovermode="closest",
    )
    return fig

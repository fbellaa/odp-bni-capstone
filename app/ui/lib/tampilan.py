"""Komponen tampilan yang dipakai bersama oleh seluruh halaman.

Palet dan komponen di sini mengikuti kosakata segmen komersial: badan hukum,
grup usaha, pemilik manfaat, counterparty dagang, dan agunan.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import mock_engine
from lib.format import kali, miliar, persen

JUDUL_APLIKASI = "Agentic AI Copilot untuk Keputusan Kredit Komersial"
SUBJUDUL_APLIKASI = (
    "Segmen komersial — debitur menengah dengan penjualan tahunan Rp 30 sampai 300 miliar."
)

PALET_TIPE = {
    "Badan hukum": "#2f6f9f",
    "Grup usaha": "#1f4e79",
    "Pemilik manfaat": "#7b5ea7",
    "Pengurus": "#8e5572",
    "Counterparty": "#2e8b6f",
    "Atribut berbagi": "#b03a48",
    "Agunan": "#8a7f2e",
}

SIMBOL_TIPE = {
    "Badan hukum": "circle",
    "Grup usaha": "hexagon",
    "Pemilik manfaat": "diamond",
    "Pengurus": "diamond-open",
    "Counterparty": "square",
    "Atribut berbagi": "x",
    "Agunan": "triangle-up",
}

PALET_KOMUNITAS = [
    "#2f6f9f", "#c9721c", "#2e8b6f", "#b03a48", "#7b5ea7", "#8a7f2e",
    "#3f8fa8", "#a4553a", "#5f7d3a", "#8e5572", "#3b6b8f", "#9c6b1f",
]

WARNA_STATUS = {
    mock_engine.LOLOS: "#1b7f4b",
    mock_engine.TELAAH: "#b58900",
    mock_engine.PENYESUAIAN: "#c0392b",
}

WARNA_KEPUTUSAN = {
    "SETUJU": "#1b7f4b",
    "SETUJU DENGAN SYARAT": "#b58900",
    "PERLU PENYESUAIAN": "#c9721c",
    "TOLAK": "#c0392b",
}


def setup_halaman(judul: str, ikon: str = "•") -> None:
    st.set_page_config(page_title=f"{judul} · Commercial Credit Copilot", page_icon=ikon, layout="wide")
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
          div[data-testid="stMetricValue"] {font-size: 1.45rem;}
          .badge {display:inline-block; padding:.18rem .6rem; border-radius:999px;
                  font-size:.75rem; font-weight:600; letter-spacing:.02em;}
          .kotak {border:1px solid rgba(128,128,128,.28); border-radius:.5rem;
                  padding:.85rem 1rem; margin-bottom:.6rem;}
          .gerbang {border-left:4px solid var(--w); border:1px solid rgba(128,128,128,.24);
                    border-left-width:4px; border-radius:.4rem; padding:.7rem .9rem;
                    margin-bottom:.55rem;}
          .gerbang .aspek {font-weight:700; letter-spacing:.04em; text-transform:uppercase;
                    font-size:.72rem; opacity:.85;}
          .gerbang .pasal {font-size:.76rem; opacity:.65; font-family:ui-monospace,monospace;}
          .tipis {opacity:.72; font-size:.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_status() -> None:
    with st.sidebar:
        st.caption("AGENTIC AI COPILOT · KEPUTUSAN KREDIT KOMERSIAL")
        st.info(
            "**Mode demo — data dummy.**\n\n"
            "Seluruh angka pada aplikasi ini sintetis dan tidak berasal dari data nasabah. "
            "Lapisan FastAPI, model, dan graf belum tersambung.",
            icon="🧪",
        )
        st.caption(
            f"Batas segmen: penjualan Rp {mock_engine.SEGMEN['penjualan_min'] / 1e9:.0f}–"
            f"{mock_engine.SEGMEN['penjualan_maks'] / 1e9:.0f} M · plafon Rp "
            f"{mock_engine.SEGMEN['plafon_min'] / 1e9:.0f}–"
            f"{mock_engine.SEGMEN['plafon_maks'] / 1e9:.0f} M"
        )
        st.caption("Status: under development · Agustus 2026")


def badge(teks: str, warna: str) -> str:
    return (
        f'<span class="badge" style="background:{warna}22;color:{warna};'
        f'border:1px solid {warna}55">{teks}</span>'
    )


def badge_keputusan(keputusan: str) -> str:
    return badge(keputusan, WARNA_KEPUTUSAN.get(keputusan, "#666666"))


def badge_grade(grade: str) -> str:
    return badge(f"Rating {grade}", mock_engine.WARNA_GRADE.get(grade, "#666666"))


def kartu_hasil(hasil: mock_engine.HasilSkor, plafon_diminta: float) -> None:
    """Baris metrik inti keputusan pada segmen komersial."""
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Probability of default", persen(hasil.pd), help="PD terkalibrasi 12 bulan ke depan")
    k2.metric("Rating internal", hasil.grade,
              help="Varian scorecard WOE menjadi rating yang mudah diaudit komite")
    k3.metric("Expected loss", miliar(hasil.expected_loss, 2), help="PD × LGD × EAD")
    k4.metric(
        "Usulan limit grup",
        miliar(hasil.limit_usulan, 0),
        delta=None if hasil.limit_usulan >= plafon_diminta
        else f"-{miliar(plafon_diminta - hasil.limit_usulan, 0)} dari permintaan",
        delta_color="inverse",
    )
    k5.metric("Usulan pricing", persen(hasil.pricing),
              help="Biaya dana + operasional + margin target + expected loss")


def kartu_rasio(hasil: mock_engine.HasilSkor) -> None:
    """Rasio keuangan komersial yang menjadi dasar covenant."""
    cov = hasil.covenant
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Interest coverage", kali(hasil.icr),
              delta=f"min {kali(cov['icr_min'])}",
              delta_color="normal" if hasil.icr >= cov["icr_min"] else "inverse")
    r2.metric("Debt to EBITDA", kali(hasil.debt_to_ebitda))
    r3.metric("Debt to equity", kali(hasil.der),
              delta=f"maks {kali(cov['der_maks'])}",
              delta_color="normal" if hasil.der <= cov["der_maks"] else "inverse")
    r4.metric("Debt service coverage", kali(hasil.dscr),
              delta=f"min {kali(cov['dscr_min'])}",
              delta_color="normal" if hasil.dscr >= cov["dscr_min"] else "inverse")
    r5.metric("Pertanggungan agunan", persen(hasil.coverage_agunan, 0),
              delta=f"min {persen(mock_engine.COVERAGE_MIN[hasil.grade], 0)}",
              delta_color="normal"
              if hasil.coverage_agunan >= mock_engine.COVERAGE_MIN[hasil.grade] else "inverse")


def panel_gerbang(gerbang: list[dict], ringkas: bool = False) -> None:
    """Gerbang kepatuhan pada alur keputusan (proposal 5.3).

    Setiap butir selalu disertai pasal sebagai dasar; tanpa kutipan, tidak ada
    jawaban.
    """
    for a in gerbang:
        warna = WARNA_STATUS.get(a["status"], "#666666")
        isi = (
            f'<div class="gerbang" style="--w:{warna}">'
            f'<span class="aspek" style="color:{warna}">{a["aspek"]} · {a["status"]}</span><br>'
            f'<b>{a["temuan"]}</b><br>'
        )
        if not ringkas:
            isi += f'<span class="tipis">{a["kutipan"]}</span><br>'
        isi += (
            f'<span class="pasal">{a["pasal"]}</span><br>'
            f'<span class="tipis">Tindakan: {a["tindakan"]}</span></div>'
        )
        st.markdown(isi, unsafe_allow_html=True)


def plot_bmpk(eksposur_grup: float, tambahan: float = 0.0,
              batas: float | None = None) -> go.Figure:
    """Posisi eksposur grup terhadap batas maksimum pemberian kredit."""
    batas = batas if batas is not None else mock_engine.BATAS_BMPK_GRUP
    sisa = max(batas - eksposur_grup - tambahan, 0.0)
    fig = go.Figure()
    potongan = [
        ("Eksposur grup berjalan", eksposur_grup, "#2f6f9f"),
        ("Usulan fasilitas ini", tambahan, "#c9721c"),
        ("Sisa ruang BMPK", sisa, "rgba(130,130,130,.28)"),
    ]
    for nama, nilai, warna in potongan:
        fig.add_trace(go.Bar(
            x=[nilai / 1e9], y=["BMPK grup"], name=nama, orientation="h",
            marker_color=warna,
            hovertemplate=f"<b>{nama}</b><br>Rp %{{x:.1f}} M<extra></extra>",
        ))
    fig.add_vline(x=batas / 1e9, line_dash="dash", line_color="#c0392b",
                  annotation_text=f"Batas Rp {batas / 1e9:.0f} M", annotation_position="top right")
    fig.update_layout(
        barmode="stack", height=190, margin=dict(l=8, r=8, t=34, b=8),
        xaxis_title="Rp miliar", yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    return fig


def plot_kontribusi(kontribusi, jumlah: int = 8) -> go.Figure:
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


def plot_kepemilikan(rantai: pd.DataFrame, entity_id: str) -> go.Figure:
    """Sankey penelusuran kepemilikan berlapis sampai pemilik manfaat akhir."""
    label = [entity_id]
    for _, r in rantai.iterrows():
        if r["pemilik"] not in label:
            label.append(r["pemilik"])
    indeks = {n: i for i, n in enumerate(label)}

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18, thickness=16,
            line=dict(color="rgba(120,120,120,.5)", width=0.8),
            label=label,
            color=["#2f6f9f"] + [
                "#7b5ea7" if "Pemilik manfaat" in j else "#3b6b8f"
                for j in rantai["jenis"]
            ],
        ),
        link=dict(
            source=[indeks[r["pemilik"]] for _, r in rantai.iterrows()],
            target=[indeks[r["dimiliki"]] for _, r in rantai.iterrows()],
            value=[float(r["porsi_langsung"]) * 100 for _, r in rantai.iterrows()],
            color="rgba(47,111,159,.32)",
            customdata=[
                [f"{r['porsi_langsung'] * 100:.1f}%", f"{r['porsi_efektif'] * 100:.1f}%", r["jenis"]]
                for _, r in rantai.iterrows()
            ],
            hovertemplate="Kepemilikan langsung %{customdata[0]}<br>"
                          "Kepemilikan efektif %{customdata[1]}<br>"
                          "%{customdata[2]}<extra></extra>",
        ),
    ))
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=18, b=8), font_size=11)
    return fig


def _tata_letak_graf(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Spring layout sederhana (Fruchterman-Reingold ringkas) tanpa dependensi tambahan."""
    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(nodes["id"])
        if not edges.empty:
            g.add_edges_from(zip(edges["source"], edges["target"]))
        return nx.spring_layout(g, seed=7, k=1.25 / math.sqrt(max(len(g), 1)))
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


# Relasi yang layak diberi penekanan visual karena menjadi dasar penggabungan
# grup debitur dan pemeriksaan BMPK.
RELASI_STRUKTURAL = {"memiliki", "mengendalikan", "menjabat_di", "menjamin_silang", "berbagi_atribut"}


def plot_graf(nodes: pd.DataFrame, edges: pd.DataFrame, warnai: str = "tipe",
              sorot: str | None = None) -> go.Figure:
    """Graf simpul-tepi dengan Plotly (lihat proposal 9.4).

    Relasi struktural (kepemilikan, pengendalian, rangkap jabatan, penjaminan
    silang, atribut berbagi) digambar lebih tebal daripada relasi dagang supaya
    tulang punggung grup usaha langsung terbaca.
    """
    posisi = _tata_letak_graf(nodes, edges)
    jejak = []

    for struktural in (False, True):
        gx, gy = [], []
        for _, e in edges.iterrows():
            if (e["relasi"] in RELASI_STRUKTURAL) != struktural:
                continue
            if e["source"] not in posisi or e["target"] not in posisi:
                continue
            x0, y0 = posisi[e["source"]]
            x1, y1 = posisi[e["target"]]
            gx += [x0, x1, None]
            gy += [y0, y1, None]
        if not gx:
            continue
        jejak.append(go.Scatter(
            x=gx, y=gy, mode="lines", hoverinfo="skip",
            line=dict(
                width=2.0 if struktural else 0.9,
                color="rgba(47,111,159,.55)" if struktural else "rgba(130,130,130,.35)",
            ),
        ))

    if warnai == "komunitas":
        warna = [PALET_KOMUNITAS[int(c) % len(PALET_KOMUNITAS)] for c in nodes["community_id"]]
    else:
        warna = [PALET_TIPE.get(t, "#888888") for t in nodes["tipe"]]

    ukuran = [30 if nid == sorot else (18 if h <= 1 else 12)
              for nid, h in zip(nodes["id"], nodes["hop"])]
    garis_tepi = ["#111111" if nid == sorot else "rgba(255,255,255,.65)" for nid in nodes["id"]]
    simbol = [SIMBOL_TIPE.get(t, "circle") for t in nodes["tipe"]]

    teks = []
    for r in nodes.itertuples():
        potong = [f"<b>{r.id}</b>", f"Tipe: {r.tipe}"]
        peran = getattr(r, "peran", None)
        if isinstance(peran, str) and peran not in ("", "-"):
            potong.append(f"Peran: {peran}")
        grup = getattr(r, "grup", None)
        if isinstance(grup, str) and grup not in ("", "-"):
            potong.append(f"Grup: {grup}")
        potong.append(f"Hop: {r.hop} · Klaster: {r.community_id}")
        if isinstance(r.pd, float) and not math.isnan(r.pd):
            potong.append(f"PD: {r.pd * 100:.2f}%")
        teks.append("<br>".join(potong))

    jejak.append(go.Scatter(
        x=[posisi[n][0] for n in nodes["id"]],
        y=[posisi[n][1] for n in nodes["id"]],
        mode="markers",
        marker=dict(size=ukuran, color=warna, symbol=simbol,
                    line=dict(width=1.4, color=garis_tepi)),
        text=teks,
        hovertemplate="%{text}<extra></extra>",
    ))

    fig = go.Figure(jejak)
    fig.update_layout(
        height=540, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        hovermode="closest",
    )
    return fig

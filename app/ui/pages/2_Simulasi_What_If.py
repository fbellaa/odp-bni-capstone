"""Simulasi what-if.

Penguji menggeser plafon, tenor, dan jenis agunan; skor, pricing, dan ekspektasi
kerugian diperbarui seketika. Bagian ini memperlihatkan bahwa model benar-benar
responsif terhadap masukan.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import dummy_data, mock_engine
from lib.format import persen, rupiah
from lib.tampilan import badge_keputusan, kartu_hasil, plot_kontribusi, setup_halaman, sidebar_status

setup_halaman("Simulasi what-if", "🎚️")
sidebar_status()

st.title("2 · Simulasi what-if")
st.caption("Ubah parameter pengajuan dan lihat dampaknya pada skor, limit, pricing, dan expected loss.")

dasar = st.session_state.get("copilot_fitur")
if dasar is None:
    st.info(
        "Belum ada pengajuan dari halaman Copilot. Simulasi dimulai dari kasus contoh.",
        icon="ℹ️",
    )
    dasar = dummy_data.ekstraksi_entitas(dummy_data.CONTOH_PROMPT[0])
    dasar.update(
        margin_usaha=0.14, stabilitas_arus_kas=0.22, supplier_concentration_hhi=0.34,
        neighbor_default_rate_1hop=0.04, tenure_nasabah_thn=2.5, network_risk_score=18.0,
    )

# ---------------------------------------------------------------- kendali
st.subheader("Parameter simulasi")
c1, c2, c3 = st.columns(3)
with c1:
    plafon = st.slider(
        "Plafon (Rp juta)", 20, 500, int(dasar["plafon"] / 1_000_000), step=5,
        help="Batas lingkup proyek: modal kerja mikro dan kecil sampai Rp 500 juta.",
    ) * 1_000_000
    omzet = st.slider(
        "Omzet bulanan (Rp juta)", 10, 900, int(dasar["omzet_bulanan"] / 1_000_000), step=5
    ) * 1_000_000
with c2:
    tenor = st.select_slider("Tenor (bulan)", [12, 18, 24, 36, 48], value=int(dasar["tenor_bulan"]))
    lama_usaha = st.slider("Lama usaha (tahun)", 0.5, 25.0, float(dasar["lama_usaha_thn"]), step=0.5)
with c3:
    jenis_agunan = st.selectbox(
        "Jenis agunan", dummy_data.JENIS_AGUNAN,
        index=dummy_data.JENIS_AGUNAN.index(dasar["jenis_agunan"]),
    )
    coverage = st.slider(
        "Pertanggungan agunan terhadap plafon", 0.0, 2.0,
        float(min(dasar.get("nilai_agunan", 0) / max(dasar["plafon"], 1), 2.0)), step=0.05,
        disabled=(jenis_agunan == "Tanpa agunan"),
    )

with st.expander("Parameter lanjutan (fitur keuangan dan graf)"):
    d1, d2, d3 = st.columns(3)
    margin = d1.slider("Margin usaha", 0.05, 0.28, float(dasar["margin_usaha"]), step=0.01)
    stabilitas = d1.slider("Stabilitas arus kas (0 stabil, 1 bergejolak)", 0.02, 0.95,
                           float(dasar["stabilitas_arus_kas"]), step=0.01)
    hhi = d2.slider("Konsentrasi pemasok (HHI)", 0.05, 0.98,
                    float(dasar["supplier_concentration_hhi"]), step=0.01)
    tetangga = d2.slider("Gagal bayar tetangga 1-hop", 0.0, 0.60,
                         float(dasar["neighbor_default_rate_1hop"]), step=0.01)
    network = d3.slider("Skor risiko jaringan", 0, 100, int(dasar["network_risk_score"]))
    tenure = d3.slider("Lama menjadi nasabah (tahun)", 0.0, 15.0,
                       float(dasar["tenure_nasabah_thn"]), step=0.5)

skenario = dict(
    sektor=dasar["sektor"], wilayah=dasar["wilayah"],
    plafon=float(plafon), omzet_bulanan=float(omzet), tenor_bulan=int(tenor),
    lama_usaha_thn=float(lama_usaha), jenis_agunan=jenis_agunan,
    nilai_agunan=float(plafon * coverage), margin_usaha=float(margin),
    stabilitas_arus_kas=float(stabilitas), supplier_concentration_hhi=float(hhi),
    neighbor_default_rate_1hop=float(tetangga), network_risk_score=float(network),
    tenure_nasabah_thn=float(tenure),
)

hasil = mock_engine.recommend_limit_pricing(skenario)
awal = mock_engine.recommend_limit_pricing(dasar)
keputusan = mock_engine.keputusan_dari_hasil(hasil)

st.divider()
st.markdown(f"### Hasil skenario &nbsp; {badge_keputusan(keputusan)}", unsafe_allow_html=True)
kartu_hasil(hasil, plafon)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Δ PD terhadap kasus awal", persen(hasil.pd), delta=f"{(hasil.pd - awal.pd) * 100:+.2f} pp",
          delta_color="inverse")
b2.metric("Δ Expected loss", rupiah(hasil.expected_loss, singkat=True),
          delta=rupiah(hasil.expected_loss - awal.expected_loss, singkat=True), delta_color="inverse")
b3.metric("Δ Pricing", persen(hasil.pricing), delta=f"{(hasil.pricing - awal.pricing) * 10_000:+.0f} bps",
          delta_color="inverse")
b4.metric("Debt service coverage", f"{hasil.dscr:.2f}x",
          delta=f"{hasil.dscr - awal.dscr:+.2f}x")

if hasil.catatan:
    for c in hasil.catatan:
        st.warning(c, icon="⚠️")

# ---------------------------------------------------------------- kurva
st.divider()
st.subheader("Kurva sensitivitas")
pilih = st.radio("Variabel yang disapu", ["Plafon", "Tenor", "Pertanggungan agunan"],
                 horizontal=True, label_visibility="collapsed")

if pilih == "Plafon":
    sumbu = np.arange(20, 505, 10) * 1_000_000
    varian = [{**skenario, "plafon": float(v), "nilai_agunan": float(v * coverage)} for v in sumbu]
    label, tampil = "Plafon (Rp juta)", sumbu / 1_000_000
elif pilih == "Tenor":
    sumbu = np.array([6, 12, 18, 24, 30, 36, 42, 48])
    varian = [{**skenario, "tenor_bulan": int(v)} for v in sumbu]
    label, tampil = "Tenor (bulan)", sumbu
else:
    sumbu = np.arange(0.0, 2.01, 0.1)
    varian = [{**skenario, "nilai_agunan": float(plafon * v)} for v in sumbu]
    label, tampil = "Pertanggungan agunan (x plafon)", sumbu

hasil_kurva = [mock_engine.recommend_limit_pricing(v) for v in varian]

fig = go.Figure()
fig.add_trace(go.Scatter(x=tampil, y=[h.pd * 100 for h in hasil_kurva],
                         name="PD (%)", mode="lines+markers", line=dict(color="#c0392b")))
fig.add_trace(go.Scatter(x=tampil, y=[h.pricing * 100 for h in hasil_kurva],
                         name="Pricing (%)", mode="lines+markers", line=dict(color="#2f6f9f")))
fig.add_trace(go.Scatter(x=tampil, y=[h.expected_loss / 1_000_000 for h in hasil_kurva],
                         name="Expected loss (Rp juta)", mode="lines+markers",
                         line=dict(color="#c9721c"), yaxis="y2"))
fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title=label, yaxis_title="Persen",
    yaxis2=dict(title="Rp juta", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Reason code skenario berjalan")
st.plotly_chart(plot_kontribusi(hasil.kontribusi), use_container_width=True)

with st.expander("Bandingkan tiga skenario agunan pada plafon yang sama"):
    baris = []
    for agunan in ["Tanpa agunan", "BPKB mobil", "SHM / SHGB"]:
        h = mock_engine.recommend_limit_pricing({**skenario, "jenis_agunan": agunan})
        baris.append({
            "Jenis agunan": agunan,
            "PD": persen(h.pd),
            "LGD": persen(h.lgd),
            "Expected loss": rupiah(h.expected_loss, singkat=True),
            "Limit usulan": rupiah(h.limit_usulan, singkat=True),
            "Pricing": persen(h.pricing),
            "Keputusan": mock_engine.keputusan_dari_hasil(h),
        })
    st.dataframe(pd.DataFrame(baris), use_container_width=True, hide_index=True)

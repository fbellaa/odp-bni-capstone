"""Jaringan entitas.

Subgraf ego dua hop, penyorotan komunitas, daftar entitas berpengaruh, dan panel
pola anomali yang terdeteksi. Setiap visual graf selalu didampingi tabel
peringkat: graf memberi kesan, tabel memberi angka yang dapat dikutip.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib import dummy_data
from lib.format import rupiah
from lib.tampilan import PALET_TIPE, badge, plot_graf, setup_halaman, sidebar_status

setup_halaman("Jaringan entitas", "🕸️")
sidebar_status()

st.title("3 · Jaringan entitas")
st.caption("Relasi antar debitur, pemasok, pembeli, penjamin, dan atribut identitas bersama.")

df = dummy_data.daftar_pengajuan()

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
entity_id = c1.selectbox(
    "Entitas pusat",
    options=df["application_id"].tolist(),
    format_func=lambda a: f"{a} — {df.loc[df['application_id'] == a, 'nama_usaha'].iat[0]}",
)
hops = c2.select_slider("Kedalaman (hop)", [1, 2, 3], value=2)
batas = c3.number_input("Batas simpul", 20, 200, 60, step=10,
                        help="Subgraf dipangkas berdasarkan bobot edge sebelum dikirim ke antarmuka.")
warnai = c4.selectbox("Warna simpul", ["tipe", "komunitas"])

nodes, edges = dummy_data.subgraf_ego(entity_id, hops=int(hops), batas_simpul=int(batas))
network_risk = dummy_data.score_network_risk(entity_id)
baris = df.loc[df["application_id"] == entity_id].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Simpul pada subgraf", len(nodes))
m2.metric("Relasi pada subgraf", len(edges))
m3.metric("Skor risiko jaringan", f"{network_risk['skor']:.0f} / 100")
m4.metric("Komunitas", f"#{int(nodes.loc[0, 'community_id'])}")

kiri, kanan = st.columns([3, 2])

with kiri:
    st.plotly_chart(plot_graf(nodes, edges, warnai=warnai, sorot=entity_id), use_container_width=True)
    if warnai == "tipe":
        st.markdown(
            " ".join(badge(t, w) for t, w in PALET_TIPE.items()),
            unsafe_allow_html=True,
        )
    else:
        st.caption("Warna menandai komunitas hasil Louvain pada snapshot graf bulan sebelumnya.")

with kanan:
    st.markdown("**Pola anomali yang terdeteksi**")
    if network_risk["pola"]:
        for p in network_risk["pola"]:
            st.markdown(
                f'<div class="kotak"><b>{p["deskripsi"]}</b><br>'
                f'<span style="opacity:.7">Bukti: {p["bukti"]}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("Tidak ada pola anomali yang terpicu pada subgraf ini.", icon="✅")

    st.markdown("**Profil entitas pusat**")
    st.dataframe(
        pd.DataFrame({
            "Item": ["Nama usaha", "Sektor", "Wilayah", "Plafon diminta", "PD", "Grade", "Kolektibilitas"],
            "Nilai": [
                baris["nama_usaha"], baris["sektor"], baris["wilayah"],
                rupiah(baris["plafon_diminta"], singkat=True),
                f"{baris['pd'] * 100:.2f}%", baris["grade"], baris["kolektibilitas"],
            ],
        }),
        use_container_width=True, hide_index=True,
    )

st.divider()

tab_relasi, tab_derajat, tab_penting, tab_prediksi = st.tabs(
    ["Daftar relasi", "Peringkat derajat", "Counterparty berpengaruh", "Prediksi relasi"]
)

with tab_relasi:
    tabel = edges.copy()
    tabel["bobot"] = tabel["bobot"].map(lambda v: rupiah(v, singkat=True))
    tabel = tabel.rename(columns={"source": "Dari", "target": "Ke", "relasi": "Tipe relasi", "bobot": "Bobot"})
    st.dataframe(tabel.sort_values("Tipe relasi"), use_container_width=True, hide_index=True)

with tab_derajat:
    derajat = (
        pd.concat([edges["source"], edges["target"]])
        .value_counts().rename_axis("entity_id").reset_index(name="derajat")
        .merge(nodes[["id", "tipe", "hop", "community_id"]], left_on="entity_id", right_on="id", how="left")
        .drop(columns="id")
    )
    st.dataframe(derajat, use_container_width=True, hide_index=True)
    st.caption("Derajat dihitung pada subgraf yang ditampilkan, bukan pada graf penuh.")

with tab_penting:
    cp = dummy_data.counterparty_penting().copy()
    cp["eksposur_terdampak"] = cp["eksposur_terdampak"].map(lambda v: rupiah(v, singkat=True))
    cp["pagerank"] = cp["pagerank"].round(4)
    cp["betweenness"] = cp["betweenness"].round(3)
    st.dataframe(
        cp.rename(columns={
            "entity_id": "Entitas", "nama": "Nama", "tipe": "Tipe", "wilayah": "Wilayah",
            "debitur_terhubung": "Debitur terhubung", "pagerank": "PageRank",
            "betweenness": "Betweenness", "eksposur_terdampak": "Eksposur terdampak",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption("Dihitung batch pada DAG bulanan, dibaca sebagai tabel — bukan dihitung saat halaman dimuat.")

with tab_prediksi:
    st.caption(
        "Kandidat relasi dari link prediction. Prediksi relasi tidak dipakai sebagai dasar penolakan; "
        "keluarannya hanya memunculkan permintaan verifikasi kepada analis."
    )
    kandidat = nodes.iloc[1:6][["id", "tipe"]].copy()
    kandidat["probabilitas"] = [0.71, 0.63, 0.58, 0.51, 0.44][: len(kandidat)]
    kandidat["dasar"] = ["Adamic-Adar tinggi", "Common neighbors 4", "Preferential attachment",
                         "Jaccard 0,38", "node2vec + classifier"][: len(kandidat)]
    st.dataframe(
        kandidat.rename(columns={"id": "Kandidat", "tipe": "Tipe", "probabilitas": "Probabilitas", "dasar": "Dasar"}),
        use_container_width=True, hide_index=True,
    )

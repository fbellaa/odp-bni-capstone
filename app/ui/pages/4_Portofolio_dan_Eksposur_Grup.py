"""Halaman 4 — Portofolio dan eksposur grup.

Konsentrasi eksposur per grup dan sektor, posisi terhadap BMPK, uji tekanan
simpul kritis, dan tabel counterparty penting. Halaman ini ditujukan untuk risk
officer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from lib import dummy_data, mock_engine
from lib.format import miliar, persen, rupiah
from lib.tampilan import setup_halaman, sidebar_status

setup_halaman("Portofolio dan eksposur grup", "📊")
sidebar_status()

st.title("4 · Portofolio dan eksposur grup")
st.caption(
    "Sebaran rating, konsentrasi eksposur per grup usaha dan sektor, kepatuhan BMPK, "
    "dan uji tekanan simpul kritis rantai pasok."
)

df = dummy_data.daftar_pengajuan()
grup = dummy_data.daftar_grup()
komunitas = dummy_data.daftar_komunitas()

with st.sidebar:
    st.divider()
    f_sektor = st.multiselect("Sektor", dummy_data.SEKTOR, default=[])
    f_wilayah = st.multiselect("Wilayah", dummy_data.WILAYAH, default=[])
    f_grade = st.multiselect("Rating internal", sorted(df["grade"].unique()), default=[])
    f_fasilitas = st.multiselect("Jenis fasilitas", dummy_data.JENIS_FASILITAS, default=[])

tersaring = df.copy()
if f_sektor:
    tersaring = tersaring[tersaring["sektor"].isin(f_sektor)]
if f_wilayah:
    tersaring = tersaring[tersaring["wilayah"].isin(f_wilayah)]
if f_grade:
    tersaring = tersaring[tersaring["grade"].isin(f_grade)]
if f_fasilitas:
    tersaring = tersaring[tersaring["jenis_fasilitas"].isin(f_fasilitas)]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Pengajuan", f"{len(tersaring):,}".replace(",", "."))
m2.metric("Total plafon diminta", rupiah(tersaring["plafon_diminta"].sum(), singkat=True))
m3.metric("Rata-rata PD", persen(tersaring["pd"].mean() if len(tersaring) else 0.0))
m4.metric("Expected loss", rupiah(tersaring["expected_loss"].sum(), singkat=True))
m5.metric("Tingkat persetujuan",
          persen((tersaring["keputusan"] != "TOLAK").mean() if len(tersaring) else 0.0))

st.divider()

tab_sebaran, tab_grup, tab_klaster, tab_tekanan, tab_ambang = st.tabs(
    ["Sebaran risiko", "Konsentrasi grup dan BMPK", "Klaster ekosistem",
     "Uji tekanan", "Pertukaran ambang rating"]
)

with tab_sebaran:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(tersaring, x="pd", nbins=40, color="grade",
                           color_discrete_map=mock_engine.WARNA_GRADE,
                           labels={"pd": "Probability of default", "count": "Jumlah"})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), bargap=0.05,
                          title="Sebaran PD menurut rating internal")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ringkas = (
            tersaring.groupby("sektor")
            .agg(jumlah=("application_id", "count"), pd_rata=("pd", "mean"),
                 plafon=("plafon_diminta", "sum"))
            .reset_index()
        )
        ringkas["plafon_miliar"] = ringkas["plafon"] / 1e9
        fig = px.scatter(ringkas, x="pd_rata", y="plafon_miliar", size="jumlah", color="sektor",
                         labels={"pd_rata": "PD rata-rata", "plafon_miliar": "Total plafon (Rp miliar)"})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          title="Eksposur dan risiko per sektor", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        heat = tersaring.pivot_table(index="sektor", columns="wilayah", values="pd", aggfunc="mean")
        fig = px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto",
                        labels=dict(color="PD rata-rata"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                          title="PD rata-rata per sektor dan wilayah")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        covenant = (
            tersaring.groupby(["grade", "posisi_covenant"], as_index=False)
            .agg(jumlah=("application_id", "count"))
        )
        fig = px.bar(covenant, x="grade", y="jumlah", color="posisi_covenant", barmode="stack",
                     category_orders={"grade": ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]},
                     color_discrete_map={"Patuh": "#2e8b6f", "Perlu perhatian": "#c9721c",
                                         "Terlanggar": "#c0392b"},
                     labels={"grade": "Rating internal", "jumlah": "Jumlah debitur",
                             "posisi_covenant": "Posisi covenant"})
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                          title="Posisi covenant per kelas rating")
        st.plotly_chart(fig, use_container_width=True)

with tab_grup:
    st.caption(
        "Eksposur digabungkan per grup debitur sesuai penelusuran pemilik manfaat, lalu "
        "dibandingkan terhadap batas maksimum pemberian kredit."
    )
    g1, g2, g3 = st.columns(3)
    g1.metric("Grup usaha dipantau", len(grup))
    g2.metric("Grup di atas 80% batas", int((grup["porsi_bmpk"] >= 0.80).sum()),
              delta="perlu pembatasan fasilitas baru", delta_color="off")
    g3.metric("Total covenant terlanggar", int(grup["covenant_terlanggar"].sum()))

    tampil = grup.copy()
    tampil["warna"] = tampil["porsi_bmpk"].map(
        lambda p: "Aman" if p < 0.6 else ("Perlu perhatian" if p < 0.85 else "Kritis")
    )
    fig = px.bar(
        tampil.sort_values("porsi_bmpk"), x="porsi_bmpk", y="grup_usaha", orientation="h",
        color="warna",
        color_discrete_map={"Aman": "#2e8b6f", "Perlu perhatian": "#c9721c", "Kritis": "#c0392b"},
        labels={"porsi_bmpk": "Porsi batas BMPK terpakai", "grup_usaha": "",
                "warna": "Status"},
        hover_data={"jumlah_entitas": True, "eksposur_grup": ":,.0f"},
    )
    fig.add_vline(x=0.85, line_dash="dot", line_color="#c0392b",
                  annotation_text="Ambang pembatasan 85%")
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10), xaxis_tickformat=".0%",
                      title="Posisi setiap grup usaha terhadap BMPK")
    st.plotly_chart(fig, use_container_width=True)

    tabel = grup.copy()
    tabel["eksposur_grup"] = tabel["eksposur_grup"].map(lambda v: miliar(v, 0))
    tabel["ruang_bmpk"] = tabel["ruang_bmpk"].map(lambda v: miliar(v, 0))
    tabel["porsi_bmpk"] = tabel["porsi_bmpk"].map(lambda v: persen(v, 0))
    tabel["pd_tertimbang"] = tabel["pd_tertimbang"].map(persen)
    tabel["npl_grup"] = tabel["npl_grup"].map(persen)
    st.dataframe(
        tabel.rename(columns={
            "grup_usaha": "Grup usaha", "jumlah_entitas": "Entitas",
            "entitas_debitur": "Sudah jadi debitur", "sektor_inti": "Sektor inti",
            "eksposur_grup": "Eksposur gabungan", "porsi_bmpk": "Porsi BMPK",
            "ruang_bmpk": "Sisa ruang", "pd_tertimbang": "PD tertimbang",
            "npl_grup": "NPL grup", "covenant_terlanggar": "Covenant terlanggar",
            "pemilik_manfaat": "Pemilik manfaat",
        }),
        use_container_width=True, hide_index=True,
    )

with tab_klaster:
    st.caption(
        "Klaster hasil Louvain pada graf berbobot menjadi unit pemantauan konsentrasi risiko "
        "yang berdampingan dengan definisi grup usaha secara legal — perbedaan keduanya justru "
        "merupakan temuan."
    )
    c1, c2 = st.columns([3, 2])
    with c1:
        tampil = komunitas.copy()
        tampil["eksposur_miliar"] = tampil["eksposur_bank"] / 1e9
        fig = px.scatter(
            tampil, x="npl_komunitas", y="eksposur_miliar", size="jumlah_anggota",
            color="pd_rata_tetangga", hover_name="nama", color_continuous_scale="RdYlGn_r",
            labels={"npl_komunitas": "NPL klaster", "eksposur_miliar": "Eksposur bank (Rp miliar)",
                    "pd_rata_tetangga": "PD rata tetangga"},
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=40, b=10),
                          title="Profil risiko tiap klaster")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        atas = komunitas.nlargest(6, "eksposur_bank").copy()
        atas["eksposur_miliar"] = atas["eksposur_bank"] / 1e9
        fig = px.bar(atas.sort_values("eksposur_miliar"), x="eksposur_miliar", y="nama",
                     orientation="h",
                     labels={"eksposur_miliar": "Eksposur bank (Rp miliar)", "nama": ""})
        fig.update_traces(marker_color="#2f6f9f")
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=40, b=10),
                          title="Klaster dengan eksposur terbesar")
        st.plotly_chart(fig, use_container_width=True)

    tabel = komunitas.copy()
    tabel["eksposur_bank"] = tabel["eksposur_bank"].map(lambda v: miliar(v, 0))
    tabel["npl_komunitas"] = tabel["npl_komunitas"].map(persen)
    tabel["pd_rata_tetangga"] = tabel["pd_rata_tetangga"].map(persen)
    tabel["modularitas"] = tabel["modularitas"].round(3)
    st.dataframe(
        tabel.rename(columns={
            "community_id": "ID", "nama": "Klaster", "jumlah_anggota": "Anggota",
            "eksposur_bank": "Eksposur bank", "npl_komunitas": "NPL",
            "pd_rata_tetangga": "PD rata tetangga", "modularitas": "Modularitas",
            "simpul_kritis": "Simpul kritis",
        }),
        use_container_width=True, hide_index=True,
    )

with tab_tekanan:
    st.caption(
        "Skenario: bila satu pembeli utama sektor atau satu entitas inti grup terganggu, "
        "berapa tambahan pencadangan kerugian yang timbul pada portofolio komersial?"
    )
    cp = dummy_data.counterparty_penting()
    c1, c2 = st.columns([2, 1])
    simpul = c1.selectbox(
        "Simpul kritis", cp["entity_id"].tolist(),
        format_func=lambda e: f"{e} — {cp.loc[cp['entity_id'] == e, 'nama'].iat[0]} "
                              f"({cp.loc[cp['entity_id'] == e, 'peran'].iat[0]})",
    )
    guncangan = c2.slider("Tingkat guncangan", 0.0, 3.0, 1.0, step=0.25,
                          help="1,0 = counterparty berhenti beroperasi; di atas itu efek "
                               "penularan dalam grup diperbesar.")

    hasil = dummy_data.uji_tekanan(simpul, guncangan)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Debitur terdampak", hasil["debitur_terdampak"])
    k2.metric("Grup usaha terdampak", hasil["grup_terdampak"])
    k3.metric("Eksposur terdampak", miliar(hasil["eksposur_terdampak"], 0))
    k4.metric("Tambahan pencadangan", miliar(hasil["tambahan_pencadangan"], 1),
              delta=f"PD rata +{hasil['kenaikan_pd_rata'] * 100:.2f} pp", delta_color="inverse")

    kurva = pd.DataFrame({"guncangan": [g / 4 for g in range(0, 13)]})
    kurva["tambahan_pencadangan"] = [
        dummy_data.uji_tekanan(simpul, g)["tambahan_pencadangan"] / 1e9 for g in kurva["guncangan"]
    ]
    fig = px.line(kurva, x="guncangan", y="tambahan_pencadangan", markers=True,
                  labels={"guncangan": "Tingkat guncangan",
                          "tambahan_pencadangan": "Tambahan pencadangan (Rp miliar)"})
    fig.update_traces(line_color="#c0392b")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                      title="Kurva tambahan pencadangan terhadap tingkat guncangan")
    st.plotly_chart(fig, use_container_width=True)

with tab_ambang:
    st.caption(
        "Pertukaran antara tingkat persetujuan dan ekspektasi kerugian pada setiap ambang "
        "rating — bahan utama kasus bisnis."
    )
    baris = []
    for ambang, rating in mock_engine.BATAS_GRADE[:-1]:
        diterima = df[df["pd"] <= ambang]
        baris.append({
            "Ambang rating": f"{rating} ke atas",
            "Ambang PD": persen(ambang),
            "Tingkat persetujuan": persen(len(diterima) / len(df)),
            "Volume penyaluran": miliar(diterima["limit_usulan"].sum(), 0),
            "Expected loss": miliar(diterima["expected_loss"].sum(), 1),
            "EL terhadap penyaluran": persen(
                diterima["expected_loss"].sum() / max(diterima["limit_usulan"].sum(), 1)
            ),
        })
    st.dataframe(pd.DataFrame(baris), use_container_width=True, hide_index=True)
    st.caption(
        "Pada segmen komersial satu keputusan bernilai puluhan sampai ratusan miliar, sehingga "
        "pergeseran satu kelas rating langsung terlihat pada pencadangan."
    )

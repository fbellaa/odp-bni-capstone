"""Halaman 3 — Struktur grup dan jaringan.

Subgraf ego dua hop, penelusuran kepemilikan sampai pemilik manfaat akhir,
penyorotan klaster, dan panel pola anomali yang terdeteksi. Setiap visual graf
selalu didampingi tabel peringkat: graf memberi kesan, tabel memberi angka yang
dapat dikutip.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib import dummy_data, mock_engine
from lib.format import kali, miliar, persen, rupiah
from lib.tampilan import (
    PALET_TIPE,
    badge,
    hero,
    judul_bagian,
    badge_grade,
    plot_bmpk,
    plot_graf,
    plot_kepemilikan,
    setup_halaman,
    sidebar_status,
)

setup_halaman("Struktur grup dan jaringan", "🕸️")
sidebar_status()

hero(
    "03",
    "Struktur grup dan jaringan",
    "Relasi kepemilikan, pengendalian, rangkap jabatan pengurus, pemasok dan pembeli utama, "
    "serta penjaminan silang antar afiliasi. Setiap visual graf selalu didampingi tabel "
    "peringkat: graf memberi kesan, tabel memberi angka yang dapat dikutip komite.",
    [("kedalaman", "2 hop"), ("penelusuran", "sampai pemilik manfaat"),
     ("keluaran", "pola anomali struktur")],
)

df = dummy_data.daftar_pengajuan()

c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1])
entity_id = c1.selectbox(
    "Badan hukum pusat",
    options=df["application_id"].tolist(),
    format_func=lambda a: f"{a} — {df.loc[df['application_id'] == a, 'nama_debitur'].iat[0]}",
)
hops = c2.select_slider("Kedalaman (hop)", [1, 2, 3], value=2)
batas = c3.number_input("Batas simpul", 20, 200, 60, step=10,
                        help="Subgraf dipangkas berdasarkan bobot edge sebelum dikirim ke antarmuka.")
warnai = c4.selectbox("Warna simpul", ["tipe", "komunitas"])

nodes, edges = dummy_data.subgraf_ego(entity_id, hops=int(hops), batas_simpul=int(batas))
network_risk = dummy_data.score_network_risk(entity_id)
rantai = dummy_data.penelusuran_kepemilikan(entity_id)
baris = df.loc[df["application_id"] == entity_id].iloc[0]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Simpul pada subgraf", len(nodes))
m2.metric("Relasi pada subgraf", len(edges))
m3.metric("Skor risiko jaringan", f"{network_risk['skor']:.0f} / 100")
m4.metric("Lapis kepemilikan", len(rantai),
          help="Jumlah lapisan sampai pemilik manfaat akhir.")
m5.metric("Klaster", f"#{int(nodes.loc[0, 'community_id'])}")

kiri, kanan = st.columns([3, 2])

with kiri:
    st.plotly_chart(plot_graf(nodes, edges, warnai=warnai, sorot=entity_id), use_container_width=True)
    if warnai == "tipe":
        st.markdown(" ".join(badge(t, w) for t, w in PALET_TIPE.items()), unsafe_allow_html=True)
    else:
        st.caption("Warna menandai klaster hasil Louvain pada snapshot graf bulan sebelumnya.")
    st.caption(
        "Garis tebal menandai relasi struktural — kepemilikan, pengendalian, rangkap jabatan, "
        "penjaminan silang, dan atribut yang dipakai bersama. Garis tipis menandai relasi dagang."
    )

with kanan:
    st.markdown("**Pola anomali struktur yang terdeteksi**")
    if network_risk["pola"]:
        for p in network_risk["pola"]:
            st.markdown(
                f'<div class="kotak"><b>{p["deskripsi"]}</b><br>'
                f'<span style="opacity:.7">Bukti: {p["bukti"]}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("Tidak ada pola anomali yang terpicu pada subgraf ini.", icon="✅")

    st.markdown(f"**Profil badan hukum pusat** &nbsp; {badge_grade(baris['grade'])}",
                unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Item": ["Nama debitur", "Grup usaha", "Sektor", "Wilayah", "Jenis fasilitas",
                     "Penjualan tahunan", "Plafon diminta", "DER", "PD", "Kolektibilitas",
                     "Posisi covenant"],
            "Nilai": [
                baris["nama_debitur"], baris["grup_usaha"], baris["sektor"], baris["wilayah"],
                baris["jenis_fasilitas"],
                rupiah(baris["penjualan_tahunan"], singkat=True),
                rupiah(baris["plafon_diminta"], singkat=True),
                kali(baris["der"]), persen(baris["pd"]),
                baris["kolektibilitas"], baris["posisi_covenant"],
            ],
        }),
        use_container_width=True, hide_index=True,
    )

st.divider()

judul_bagian("Penelusuran kepemilikan sampai pemilik manfaat akhir")
st.caption(
    "Relasi `mengendalikan` diturunkan dari penelusuran kepemilikan berlapis. Entitas antara "
    "tanpa aktivitas usaha ditandai karena menjadi pemicu penelaahan APU-PPT."
)
ko, kt = st.columns([3, 2])
with ko:
    st.plotly_chart(plot_kepemilikan(rantai, entity_id), use_container_width=True)
with kt:
    tampil = rantai.copy()
    tampil["porsi_langsung"] = tampil["porsi_langsung"].map(lambda v: persen(v, 1))
    tampil["porsi_efektif"] = tampil["porsi_efektif"].map(lambda v: persen(v, 1))
    st.dataframe(
        tampil.rename(columns={
            "tingkat": "Lapis", "pemilik": "Pemilik", "jenis": "Jenis",
            "dimiliki": "Memiliki", "porsi_langsung": "Porsi langsung",
            "porsi_efektif": "Porsi efektif", "aktivitas_usaha": "Aktivitas usaha",
            "yurisdiksi": "Yurisdiksi",
        }),
        use_container_width=True, hide_index=True,
    )
    penampung = rantai[rantai["aktivitas_usaha"].str.startswith("Tidak ada")]
    if len(penampung):
        st.warning(
            f"{len(penampung)} entitas antara tidak memiliki aktivitas usaha — "
            "penelaahan lanjutan atas struktur kepemilikan berlapis wajib dilakukan.",
            icon="🔎",
        )

st.divider()

judul_bagian("Eksposur grup usaha terhadap BMPK")
grup = dummy_data.daftar_grup()
baris_grup = grup.loc[grup["grup_usaha"] == baris["grup_usaha"]]
baris_grup = baris_grup.iloc[0] if len(baris_grup) else grup.iloc[0]

g1, g2 = st.columns([2, 3])
with g1:
    st.metric("Grup usaha", baris_grup["grup_usaha"])
    st.metric("Entitas dalam grup", int(baris_grup["jumlah_entitas"]),
              delta=f"{int(baris_grup['entitas_debitur'])} sudah menjadi debitur", delta_color="off")
    st.metric("Eksposur gabungan", miliar(baris_grup["eksposur_grup"], 0),
              delta=f"{persen(baris_grup['porsi_bmpk'], 0)} dari batas", delta_color="off")
with g2:
    st.plotly_chart(plot_bmpk(baris_grup["eksposur_grup"], baris["limit_usulan"]),
                    use_container_width=True)
    st.caption(
        f"Batas maksimum pemberian kredit satu grup pada demo ini "
        f"{miliar(mock_engine.BATAS_BMPK_GRUP, 0)}. Usulan fasilitas pengajuan terpilih "
        f"{miliar(baris['limit_usulan'], 0)}."
    )

st.divider()

tab_relasi, tab_derajat, tab_penting, tab_prediksi = st.tabs(
    ["Daftar relasi", "Peringkat derajat", "Counterparty berpengaruh", "Prediksi relasi"]
)

with tab_relasi:
    tabel = edges.copy()
    tabel["bobot"] = tabel["bobot"].map(lambda v: rupiah(v, singkat=True))
    tabel = tabel.rename(columns={"source": "Dari", "target": "Ke",
                                  "relasi": "Tipe relasi", "bobot": "Bobot"})
    st.dataframe(tabel.sort_values("Tipe relasi"), use_container_width=True, hide_index=True)
    st.caption(
        "Bobot relasi dagang adalah nilai transfer 12 bulan; bobot relasi kepemilikan adalah "
        "porsi kepemilikan yang dinormalkan."
    )

with tab_derajat:
    derajat = (
        pd.concat([edges["source"], edges["target"]])
        .value_counts().rename_axis("entity_id").reset_index(name="derajat")
        .merge(nodes[["id", "tipe", "peran", "hop", "community_id"]],
               left_on="entity_id", right_on="id", how="left")
        .drop(columns="id")
    )
    st.dataframe(
        derajat.rename(columns={
            "entity_id": "Entitas", "derajat": "Derajat", "tipe": "Tipe",
            "peran": "Peran", "hop": "Hop", "community_id": "Klaster",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption("Derajat dihitung pada subgraf yang ditampilkan, bukan pada graf penuh.")

with tab_penting:
    cp = dummy_data.counterparty_penting().copy()
    cp["eksposur_terdampak"] = cp["eksposur_terdampak"].map(lambda v: rupiah(v, singkat=True))
    cp["volume_tahunan"] = cp["volume_tahunan"].map(lambda v: rupiah(v, singkat=True))
    cp["pagerank"] = cp["pagerank"].round(4)
    cp["betweenness"] = cp["betweenness"].round(3)
    st.dataframe(
        cp.rename(columns={
            "entity_id": "Entitas", "nama": "Nama", "peran": "Peran", "sektor": "Sektor",
            "wilayah": "Wilayah", "debitur_terhubung": "Debitur terhubung",
            "pagerank": "PageRank", "betweenness": "Betweenness",
            "volume_tahunan": "Volume tahunan", "eksposur_terdampak": "Eksposur terdampak",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption("Dihitung batch pada DAG bulanan, dibaca sebagai tabel — bukan dihitung saat halaman dimuat.")

with tab_prediksi:
    st.caption(
        "Kandidat relasi dari link prediction. Prediksi relasi tidak dipakai sebagai dasar "
        "penolakan; keluarannya hanya memunculkan permintaan verifikasi kepada analis — "
        "termasuk pemeriksaan apakah kandidat tersebut seharusnya masuk ke perhitungan BMPK grup."
    )
    kandidat = nodes.iloc[1:6][["id", "tipe", "peran"]].copy()
    kandidat["probabilitas"] = [0.74, 0.66, 0.59, 0.52, 0.45][: len(kandidat)]
    kandidat["dasar"] = ["Adamic-Adar tinggi", "Common neighbors 4", "Preferential attachment",
                         "Jaccard 0,38", "node2vec + classifier"][: len(kandidat)]
    kandidat["tindak_lanjut"] = [
        "Verifikasi apakah satu kendali dengan debitur", "Peluang supply chain financing",
        "Verifikasi rangkap jabatan pengurus", "Peluang cross-selling trade finance",
        "Periksa penjaminan silang",
    ][: len(kandidat)]
    st.dataframe(
        kandidat.rename(columns={
            "id": "Kandidat", "tipe": "Tipe", "peran": "Peran",
            "probabilitas": "Probabilitas", "dasar": "Dasar", "tindak_lanjut": "Tindak lanjut",
        }),
        use_container_width=True, hide_index=True,
    )

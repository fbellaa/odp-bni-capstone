"""Simulasi what-if — segmen komersial.

Penguji menggeser plafon, tenor, struktur agunan, dan asumsi EBITDA; skor,
pricing, covenant, dan ekspektasi kerugian diperbarui seketika. Bagian ini
memperlihatkan bahwa model benar-benar responsif terhadap masukan.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import dummy_data, mock_engine, model_nyata as mn
from lib.format import kali, miliar, persen
from lib.tampilan import (
    AKSEN,
    AMBER,
    MERAH,
    PRIMER,
    badge_grade,
    badge_keputusan,
    kartu_hasil,
    gaya_plot,
    hero,
    judul_bagian,
    kartu_rasio,
    panel_gerbang,
    plot_bmpk,
    plot_kontribusi,
    setup_halaman,
    sidebar_status,
)

setup_halaman("Simulasi what-if", "🎚️")
sidebar_status()

hero(
    "02",
    "Simulasi what-if",
    "Sandbox perhitungan untuk relationship manager: naikkan atau turunkan plafon, tenor, "
    "struktur agunan, dan asumsi kinerja, lalu lihat dampaknya pada rating, limit grup, "
    "pricing, covenant, dan expected loss.",
    [("skor", "PD model XGBoost"), ("pembanding", "kasus dari halaman 01"),
     ("keluaran", "limit · pricing · covenant")],
)

dasar = st.session_state.get("copilot_fitur")
if dasar is None:
    st.info(
        "Belum ada pengajuan dari halaman Copilot. Simulasi dimulai dari kasus contoh.",
        icon="ℹ️",
    )
    dasar = dummy_data.ekstraksi_entitas(dummy_data.CONTOH_PROMPT[0])
    dasar.update(
        utang_berbunga_eksisting=dasar["plafon"] * 0.25, konversi_ebitda_kas=0.76,
        utilisasi_plafon=0.72, buyer_concentration_hhi=0.32,
        supplier_concentration_hhi=0.30, neighbor_default_rate_1hop=0.035,
        group_exposure_share=0.42, network_risk_score=18.0, tenure_nasabah_thn=8.0,
    )

MIN_PLAFON = int(mock_engine.SEGMEN["plafon_min"] / 1e9)
MAKS_PLAFON = int(mock_engine.SEGMEN["plafon_maks"] / 1e9)
MIN_JUAL = int(mock_engine.SEGMEN["penjualan_min"] / 1e9)
MAKS_JUAL = int(mock_engine.SEGMEN["penjualan_maks"] / 1e9)


def _jepit(nilai: float, bawah: int, atas: int) -> int:
    return int(min(max(round(nilai / 1e9), bawah), atas))


# ---------------------------------------------------------------- kendali
judul_bagian("Parameter simulasi",
             "Setiap geseran langsung dihitung ulang — tidak ada tombol jalankan.")
c1, c2, c3 = st.columns(3)
with c1:
    plafon = st.slider(
        "Plafon fasilitas (Rp miliar)", MIN_PLAFON, MAKS_PLAFON,
        _jepit(dasar["plafon"], MIN_PLAFON, MAKS_PLAFON), step=5,
        help=f"Batas segmen komersial: Rp {MIN_PLAFON} M sampai Rp {MAKS_PLAFON} M.",
    ) * 1e9
    penjualan = st.slider(
        "Penjualan tahunan (Rp miliar)", MIN_JUAL, MAKS_JUAL,
        _jepit(dasar["penjualan_tahunan"], MIN_JUAL, MAKS_JUAL), step=5,
    ) * 1e9
with c2:
    fasilitas = st.selectbox(
        "Jenis fasilitas", dummy_data.JENIS_FASILITAS,
        index=dummy_data.JENIS_FASILITAS.index(
            dasar.get("jenis_fasilitas", dummy_data.JENIS_FASILITAS[0])
        ),
        help="Fasilitas revolving diuji atas beban bunga berjalan; fasilitas beramortisasi "
             "diuji atas angsuran anuitas sepanjang tenor.",
    )
    tenor = st.select_slider("Tenor (bulan)", [12, 24, 36, 48, 60, 84],
                             value=int(min([12, 24, 36, 48, 60, 84],
                                           key=lambda v: abs(v - int(dasar["tenor_bulan"])))))
    ebitda_margin = st.slider("EBITDA margin", 0.04, 0.26, float(dasar["ebitda_margin"]), step=0.005,
                              format="%.3f")
    der = st.slider("Debt to equity ratio", 0.3, 4.0, float(dasar["der"]), step=0.05)
with c3:
    jenis_agunan = st.selectbox(
        "Struktur agunan", dummy_data.JENIS_AGUNAN,
        index=dummy_data.JENIS_AGUNAN.index(dasar["jenis_agunan"]),
    )
    coverage = st.slider(
        "Pertanggungan agunan terhadap plafon", 0.0, 2.5,
        float(min(dasar.get("nilai_agunan", 0) / max(dasar["plafon"], 1), 2.5)), step=0.05,
        disabled=("Tanpa agunan" in jenis_agunan),
    )
    group_share = st.slider(
        "Eksposur grup berjalan terhadap BMPK", 0.0, 0.98,
        float(dasar["group_exposure_share"]), step=0.01,
        help="Porsi batas maksimum pemberian kredit satu grup yang sudah terpakai.",
    )

with st.expander("Parameter lanjutan (kinerja, perilaku fasilitas, dan fitur graf)"):
    d1, d2, d3 = st.columns(3)
    konversi = d1.slider("Konversi EBITDA ke kas", 0.30, 0.98,
                         float(dasar["konversi_ebitda_kas"]), step=0.01)
    utang_eksisting = d1.slider("Utang berbunga eksisting (Rp miliar)", 0, 300,
                                int(dasar["utang_berbunga_eksisting"] / 1e9), step=5) * 1e9
    utilisasi = d1.slider("Tingkat pemakaian plafon", 0.15, 0.99,
                          float(dasar["utilisasi_plafon"]), step=0.01)
    buyer_hhi = d2.slider("Konsentrasi pembeli (HHI)", 0.05, 0.97,
                          float(dasar["buyer_concentration_hhi"]), step=0.01)
    supplier_hhi = d2.slider("Konsentrasi pemasok (HHI)", 0.05, 0.98,
                             float(dasar["supplier_concentration_hhi"]), step=0.01)
    tetangga = d2.slider("Gagal bayar entitas 1-hop", 0.0, 0.60,
                         float(dasar["neighbor_default_rate_1hop"]), step=0.01)
    network = d3.slider("Skor risiko jaringan", 0, 100, int(dasar["network_risk_score"]))
    tenure = d3.slider("Lama menjadi nasabah (tahun)", 0.0, 25.0,
                       float(dasar["tenure_nasabah_thn"]), step=0.5)
    saldo_giro = d3.slider("Saldo giro rata-rata (Rp miliar)", 0, 50,
                           int(min(dasar.get("saldo_giro_rata", 15e9) / 1e9, 50)), step=1) * 1e9
    umur = d3.slider("Umur badan usaha (tahun)", 2.0, 45.0,
                     float(dasar["umur_usaha_thn"]), step=1.0)

skenario = dict(
    nama_debitur=dasar.get("nama_debitur", "-"),
    sektor=dasar["sektor"], wilayah=dasar["wilayah"],
    jenis_fasilitas=fasilitas,
    plafon=float(plafon), penjualan_tahunan=float(penjualan), tenor_bulan=int(tenor),
    ebitda_margin=float(ebitda_margin), der=float(der),
    utang_berbunga_eksisting=float(utang_eksisting),
    konversi_ebitda_kas=float(konversi), utilisasi_plafon=float(utilisasi),
    saldo_giro_rata=float(saldo_giro), umur_usaha_thn=float(umur),
    jenis_agunan=jenis_agunan, nilai_agunan=float(plafon * coverage),
    buyer_concentration_hhi=float(buyer_hhi),
    supplier_concentration_hhi=float(supplier_hhi),
    neighbor_default_rate_1hop=float(tetangga),
    group_exposure_share=float(group_share),
    network_risk_score=float(network), tenure_nasabah_thn=float(tenure),
    jumlah_entitas_grup=int(dasar.get("jumlah_entitas_grup", 1)),
    indikasi_rangkap_jabatan=bool(dasar.get("indikasi_rangkap_jabatan", False)),
)

pakai_model = st.toggle(
    "Pakai model PD dan LGD sungguhan", value=mn.status_lapisan_model()["pd"],
    disabled=not mn.status_lapisan_model()["pd"],
    help="Menyalakan artefak ml/models. Bila dimatikan, simulasi memakai mesin demo "
         "deterministik supaya perilaku tiap rasio terlihat terpisah.",
)


def dengan_model(skenario_uji: dict) -> dict:
    """Tempelkan PD dan LGD hasil model pada satu skenario.

    Simulasi menyapu puluhan titik kurva, jadi SHAP dilewati di sini; reason
    code cukup dihitung sekali untuk skenario yang sedang ditampilkan.
    """
    if not pakai_model:
        return skenario_uji
    hasil_model = mn.skor_pd(skenario_uji, dengan_kontribusi=False)
    lgd_model = mn.skor_lgd(skenario_uji)
    salinan = dict(skenario_uji)
    if hasil_model is not None:
        salinan["pd_model"] = hasil_model.pd_kalibrasi
    if lgd_model is not None:
        salinan["lgd_model"] = lgd_model
    return salinan


skenario = dengan_model(skenario)
hasil = mock_engine.recommend_limit_pricing(skenario)
gerbang = mock_engine.check_credit_policy(hasil, skenario)
awal = mock_engine.recommend_limit_pricing(dengan_model(dict(dasar)))
keputusan = mock_engine.keputusan_dari_hasil(hasil, gerbang)

st.divider()
st.markdown(
    f"### Hasil skenario &nbsp; {badge_keputusan(keputusan)} &nbsp; {badge_grade(hasil.grade)}",
    unsafe_allow_html=True,
)
kartu_hasil(hasil, plafon)
st.markdown("**Rasio keuangan terhadap ambang covenant kelas rating**")
kartu_rasio(hasil)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Δ PD terhadap kasus awal", persen(hasil.pd), delta=f"{(hasil.pd - awal.pd) * 100:+.2f} pp",
          delta_color="inverse")
b2.metric("Δ Expected loss", miliar(hasil.expected_loss, 2),
          delta=miliar(hasil.expected_loss - awal.expected_loss, 2), delta_color="inverse")
b3.metric("Δ Pricing", persen(hasil.pricing), delta=f"{(hasil.pricing - awal.pricing) * 10_000:+.0f} bps",
          delta_color="inverse")
b4.metric("Δ Limit usulan", miliar(hasil.limit_usulan, 0),
          delta=miliar(hasil.limit_usulan - awal.limit_usulan, 0))

if hasil.catatan:
    for c in hasil.catatan:
        st.warning(c, icon="⚠️")

st.divider()
kol_gerbang, kol_bmpk = st.columns([3, 2])
with kol_gerbang:
    judul_bagian("Gerbang kepatuhan skenario")
    panel_gerbang(gerbang, ringkas=True)
with kol_bmpk:
    judul_bagian("Posisi BMPK grup")
    st.plotly_chart(plot_bmpk(hasil.eksposur_grup, hasil.limit_usulan), use_container_width=True)
    st.caption(
        f"Eksposur grup berjalan {miliar(hasil.eksposur_grup, 0)} · usulan fasilitas ini "
        f"{miliar(hasil.limit_usulan, 0)} · sisa ruang {miliar(hasil.ruang_bmpk, 0)}."
    )
    st.markdown("**Covenant wajib kelas rating " + hasil.grade + "**")
    st.dataframe(
        pd.DataFrame([
            {"Covenant": "Debt to equity maksimum", "Ambang": kali(hasil.covenant["der_maks"]),
             "Posisi": kali(hasil.der)},
            {"Covenant": "Interest coverage minimum", "Ambang": kali(hasil.covenant["icr_min"]),
             "Posisi": kali(hasil.icr)},
            {"Covenant": "Debt service coverage minimum", "Ambang": kali(hasil.covenant["dscr_min"]),
             "Posisi": kali(hasil.dscr)},
        ]),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Frekuensi pengujian covenant: {hasil.covenant['uji'].lower()}.")

# ---------------------------------------------------------------- kurva
st.divider()
judul_bagian("Kurva sensitivitas",
             "Satu variabel disapu, sisanya ditahan pada nilai skenario di atas.")
pilih = st.radio(
    "Variabel yang disapu",
    ["Plafon", "Tenor", "Pertanggungan agunan", "EBITDA margin", "Konsentrasi pembeli"],
    horizontal=True, label_visibility="collapsed",
)

if pilih == "Plafon":
    sumbu = np.arange(MIN_PLAFON, MAKS_PLAFON + 1, 5) * 1e9
    varian = [{**skenario, "plafon": float(v), "nilai_agunan": float(v * coverage)} for v in sumbu]
    label, tampil = "Plafon (Rp miliar)", sumbu / 1e9
elif pilih == "Tenor":
    sumbu = np.array([12, 24, 36, 48, 60, 72, 84])
    varian = [{**skenario, "tenor_bulan": int(v)} for v in sumbu]
    label, tampil = "Tenor (bulan)", sumbu
elif pilih == "Pertanggungan agunan":
    sumbu = np.arange(0.0, 2.51, 0.1)
    varian = [{**skenario, "nilai_agunan": float(plafon * v)} for v in sumbu]
    label, tampil = "Pertanggungan agunan (x plafon)", sumbu
elif pilih == "EBITDA margin":
    sumbu = np.arange(0.04, 0.261, 0.01)
    varian = [{**skenario, "ebitda_margin": float(v)} for v in sumbu]
    label, tampil = "EBITDA margin", sumbu
else:
    sumbu = np.arange(0.05, 0.98, 0.05)
    varian = [{**skenario, "buyer_concentration_hhi": float(v)} for v in sumbu]
    label, tampil = "Konsentrasi pembeli (HHI)", sumbu

hasil_kurva = [mock_engine.recommend_limit_pricing(dengan_model(v)) for v in varian]

fig = go.Figure()
fig.add_trace(go.Scatter(x=tampil, y=[h.pd * 100 for h in hasil_kurva],
                         name="PD (%)", mode="lines+markers", line=dict(color=MERAH, width=3)))
fig.add_trace(go.Scatter(x=tampil, y=[h.pricing * 100 for h in hasil_kurva],
                         name="Pricing (%)", mode="lines+markers", line=dict(color=PRIMER, width=3)))
fig.add_trace(go.Scatter(x=tampil, y=[h.expected_loss / 1e9 for h in hasil_kurva],
                         name="Expected loss (Rp miliar)", mode="lines+markers",
                         line=dict(color=AMBER, width=3), yaxis="y2"))
fig.add_trace(go.Scatter(x=tampil, y=[h.limit_usulan / 1e9 for h in hasil_kurva],
                         name="Limit usulan (Rp miliar)", mode="lines",
                         line=dict(color=AKSEN, dash="dot", width=3), yaxis="y2"))
fig.update_layout(
    xaxis_title=label, yaxis_title="Persen",
    yaxis2=dict(title="Rp miliar", overlaying="y", side="right", showgrid=False),
)
st.plotly_chart(gaya_plot(fig, 430), use_container_width=True)

st.divider()
judul_bagian("Reason code skenario berjalan")
kontribusi = hasil.kontribusi
if pakai_model:
    hasil_model = mn.skor_pd(skenario)
    if hasil_model is not None and hasil_model.kontribusi:
        kontribusi = hasil_model.kontribusi
        st.caption("Nilai SHAP dari model PD untuk kombinasi parameter yang sedang dipilih.")
st.plotly_chart(plot_kontribusi(kontribusi), use_container_width=True)

with st.expander("Bandingkan struktur agunan pada plafon yang sama"):
    baris = []
    for agunan in ["Tanpa agunan (clean basis)", "Piutang dagang (fidusia)",
                   "Mesin dan peralatan", "Tanah dan bangunan pabrik (SHM/SHGB)"]:
        varian_agunan = dengan_model({**skenario, "jenis_agunan": agunan})
        h = mock_engine.recommend_limit_pricing(varian_agunan)
        g = mock_engine.check_credit_policy(h, varian_agunan)
        baris.append({
            "Struktur agunan": agunan,
            "PD": persen(h.pd),
            "LGD": persen(h.lgd),
            "Expected loss": miliar(h.expected_loss, 2),
            "Limit usulan": miliar(h.limit_usulan, 0),
            "Pricing": persen(h.pricing),
            "Kepatuhan": mock_engine.status_kepatuhan(g),
            "Keputusan": mock_engine.keputusan_dari_hasil(h, g),
        })
    st.dataframe(pd.DataFrame(baris), use_container_width=True, hide_index=True)

"""Halaman 4 — Kesehatan model.

Metrik dihitung langsung dari artefak `ml/models` di atas `data/gold`, bukan
dari angka yang ditulis tangan:

    PD   XGBoost + kalibrator logistik   -> default 12 bulan
    EWS  Regresi logistik                -> default 6 bulan pada panel bulanan
    LGD  XGBoost regresi                 -> tingkat kerugian saat gagal bayar

Pembacaan halaman ini berpusat pada recall. Pada portofolio dengan kejadian
default sekitar tiga persen, akurasi hampir selalu terlihat bagus dan hampir
tidak berarti apa-apa; yang menentukan adalah berapa banyak debitur bermasalah
yang tertangkap, dan berapa banyak berkas yang harus ditelaah untuk itu.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import dummy_data, model_nyata as mn
from lib.format import cacah
from lib.tampilan import (
    AKSEN,
    AMBER,
    KORAL,
    MERAH,
    PRIMER,
    UNGU,
    gaya_plot,
    hero,
    judul_bagian,
    kartu,
    plot_recall_ambang,
    setup_halaman,
    sidebar_status,
)

setup_halaman("Kesehatan model", "🩺")
sidebar_status()

ev_pd = mn.evaluasi_pd()
ev_ews = mn.evaluasi_ews()
ev_lgd = mn.evaluasi_lgd()

hero(
    "04",
    "Kesehatan model",
    "Metrik berjalan tiga model produksi, stabilitas populasi, dan evaluasi lapisan agen. "
    "Seluruh angka pada halaman ini dihitung ulang dari artefak model dan data emas setiap "
    "kali halaman dibuka.",
    [
        ("model PD", "XGBoost"),
        ("model EWS", "Regresi logistik"),
        ("model LGD", "XGBoost"),
        ("fokus evaluasi", "Recall"),
    ],
)

if ev_pd is None and ev_ews is None and ev_lgd is None:
    st.error(
        "Artefak model atau data emas tidak ditemukan. Pastikan `ml/models/*.joblib` dan "
        "`data/gold/*.parquet` ada, lalu pasang `scikit-learn`, `xgboost`, dan `pyarrow`.",
        icon="⛔",
    )
    st.stop()

# ------------------------------------------------------------- ringkas atas
oot = (ev_pd or {}).get("uji_oot")
k1, k2, k3, k4 = st.columns(4)
if oot:
    k1.metric("Recall PD (ambang q80)", f"{oot['recall']:.1%}",
              help="Porsi debitur yang benar-benar gagal bayar dan berhasil ditandai model.")
    k2.metric("AUC PD out-of-time", f"{oot['auc']:.3f}")
    k3.metric("KS PD", f"{oot['ks']:.3f}")
    k4.metric("Kejadian default pada uji", f"{oot['tingkat_kejadian']:.2%}",
              help="Kelas positif yang sangat jarang inilah alasan recall didahulukan.")

judul_bagian(
    "Tabel model produksi",
    "Tiga model dengan tugas berbeda, karena itu dinilai dengan ukuran yang berbeda pula.",
)

baris = []
if ev_pd and ev_pd.get("uji_oot"):
    p = ev_pd["uji_oot"]
    baris.append({
        "Model": "PD 12 bulan", "Algoritme": "XGBoost + kalibrator logistik",
        "Berkas": "pd_champion.joblib", "Sampel uji": p["n"],
        "Recall": f"{p['recall']:.1%}", "Presisi": f"{p['presisi']:.1%}",
        "AUC": f"{p['auc']:.3f}", "PR-AUC": f"{p['pr_auc']:.3f}", "KS": f"{p['ks']:.3f}",
        "Status": "Produksi",
    })
if ev_ews and ev_ews.get("uji_oot"):
    e = ev_ews["uji_oot"]
    baris.append({
        "Model": "EWS 6 bulan", "Algoritme": "Regresi logistik",
        "Berkas": "ews_logistic_champion.joblib", "Sampel uji": e["n"],
        "Recall": f"{e['recall']:.1%}", "Presisi": f"{e['presisi']:.1%}",
        "AUC": f"{e['auc']:.3f}", "PR-AUC": f"{e['pr_auc']:.3f}", "KS": f"{e['ks']:.3f}",
        "Status": "Produksi",
    })
if ev_lgd:
    baris.append({
        "Model": "LGD", "Algoritme": "XGBoost regresi",
        "Berkas": "final_lgd_xgboost.pkl", "Sampel uji": ev_lgd["n"],
        "Recall": "—", "Presisi": "—",
        "AUC": "—", "PR-AUC": "—", "KS": "—",
        "Status": "Produksi",
    })
st.dataframe(pd.DataFrame(baris), use_container_width=True, hide_index=True)
st.caption(
    "Recall dan presisi PD dihitung pada ambang q80 artefak; EWS pada ambang 2 persen. "
    "LGD adalah model regresi, jadi ukurannya galat, bukan recall — angkanya ada pada tab LGD."
)

tab_pd, tab_ews, tab_lgd, tab_psi, tab_agen, tab_kualitas = st.tabs(
    ["Model PD", "Model EWS", "Model LGD", "Stabilitas populasi",
     "Evaluasi agen (judge arena)", "Gerbang kualitas data"]
)

# ----------------------------------------------------------------- tab PD
with tab_pd:
    if not ev_pd or not ev_pd.get("uji_oot"):
        st.info("Evaluasi PD tidak tersedia.", icon="ℹ️")
    else:
        p = ev_pd["uji_oot"]
        st.markdown(
            kartu(
                "Mengapa recall didahulukan",
                f"Dari {cacah(p['n'])}"
                + f" pengajuan pada periode uji, hanya {int(p['tingkat_kejadian'] * p['n'])} "
                "yang benar-benar gagal bayar. Satu debitur bermasalah yang lolos berbiaya "
                "jauh lebih besar daripada satu berkas sehat yang ikut ditelaah, sehingga "
                "ambang dipilih dari sisi recall lebih dulu, baru beban telaah dihitung.",
                warna=AKSEN, ikon="🎯",
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_recall_ambang(ev_pd["kurva_ambang"],
                               "Recall, presisi, dan beban telaah pada tiap ambang"),
            use_container_width=True,
        )
        st.dataframe(
            ev_pd["kurva_ambang"].assign(
                nilai_ambang=lambda d: d["nilai_ambang"].map(lambda v: f"{v:.2%}"),
                recall=lambda d: d["recall"].map(lambda v: f"{v:.1%}"),
                presisi=lambda d: d["presisi"].map(lambda v: f"{v:.1%}"),
                porsi_alarm=lambda d: d["porsi_alarm"].map(lambda v: f"{v:.1%}"),
            ).rename(columns={
                "ambang": "Ambang", "nilai_ambang": "PD ambang", "recall": "Recall",
                "presisi": "Presisi", "porsi_alarm": "Berkas ditelaah"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("**Sebaran skor: yang gagal bayar vs yang tidak**")
        sebaran = pd.DataFrame({
            "skor": np.concatenate([p["skor_kalibrasi"][p["y"] == 0],
                                    p["skor_kalibrasi"][p["y"] == 1]]),
            "kelompok": ["Tidak default"] * int((p["y"] == 0).sum())
                        + ["Default"] * int((p["y"] == 1).sum()),
        })
        fig = px.histogram(
            sebaran, x="skor", color="kelompok", barmode="overlay", nbins=45,
            histnorm="probability density",
            color_discrete_map={"Tidak default": PRIMER, "Default": MERAH},
            labels={"skor": "PD terkalibrasi", "kelompok": ""},
        )
        ambang_q80 = float(mn.muat_pd()["risk_cutoffs"]["q80"])
        fig.add_vline(x=ambang_q80, line_dash="dot", line_color=AMBER,
                      annotation_text="ambang q80", annotation_position="top right")
        st.plotly_chart(gaya_plot(fig, 380), use_container_width=True)

        latih = ev_pd.get("latih")
        if latih:
            st.caption(
                f"AUC data latih {latih['auc']:.3f} vs out-of-time {p['auc']:.3f} — "
                "selisihnya dilaporkan apa adanya sebagai ukuran overfitting."
            )

# ---------------------------------------------------------------- tab EWS
with tab_ews:
    if not ev_ews:
        st.info("Evaluasi EWS tidak tersedia.", icon="ℹ️")
    else:
        e = ev_ews["uji_oot"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recall (ambang 2%)", f"{e['recall']:.1%}")
        m2.metric("AUC", f"{e['auc']:.3f}")
        m3.metric("PR-AUC", f"{e['pr_auc']:.3f}")
        m4.metric("Snapshot bulanan diuji", cacah(e["n"]))
        st.caption(
            "Regresi logistik dipilih untuk EWS karena keluarannya harus bisa dibaca "
            "petugas pemantauan sebagai daftar penyebab, bukan sekadar angka."
        )
        st.plotly_chart(
            plot_recall_ambang(ev_ews["kurva_ambang"],
                               "Ambang alarm EWS: recall vs beban pemantauan"),
            use_container_width=True,
        )
        st.markdown(
            kartu(
                "Membaca tabel ini",
                "Menurunkan ambang dari 5 persen ke 2 persen menaikkan recall tetapi "
                "melipatgandakan jumlah fasilitas yang harus dipantau tiap bulan. Angka "
                "yang dipakai produksi adalah keputusan kapasitas tim, bukan keputusan model.",
                warna=UNGU, ikon="📌",
            ),
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- tab LGD
with tab_lgd:
    if not ev_lgd:
        st.info("Evaluasi LGD tidak tersedia.", icon="ℹ️")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{ev_lgd['mae']:.3f}")
        m2.metric("RMSE", f"{ev_lgd['rmse']:.3f}")
        m3.metric("R²", f"{ev_lgd['r2']:.3f}")
        m4.metric("Fasilitas default diuji", cacah(ev_lgd["n"]))
        st.caption(
            f"Rata-rata LGD realisasi {ev_lgd['rata_realisasi']:.1%} versus prediksi "
            f"{ev_lgd['rata_prediksi']:.1%}. Untuk kebutuhan expected loss, ketepatan "
            "rata-rata tingkat portofolio lebih penting daripada ketepatan per fasilitas."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ev_lgd["realisasi"], y=ev_lgd["prediksi"], mode="markers",
            marker=dict(size=7, color=PRIMER, opacity=.55, line=dict(width=0)),
            name="Fasilitas", hovertemplate="Realisasi %{x:.2f}<br>Prediksi %{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Prediksi sempurna",
            line=dict(color=KORAL, dash="dash", width=2),
        ))
        fig.update_layout(xaxis_title="LGD realisasi", yaxis_title="LGD prediksi",
                          title="Prediksi terhadap realisasi")
        st.plotly_chart(gaya_plot(fig, 420), use_container_width=True)

        galat = pd.DataFrame({"galat": ev_lgd["prediksi"] - ev_lgd["realisasi"]})
        fig2 = px.histogram(galat, x="galat", nbins=30,
                            color_discrete_sequence=[AKSEN],
                            labels={"galat": "Prediksi − realisasi"})
        fig2.add_vline(x=0, line_dash="dot", line_color=MERAH)
        st.plotly_chart(gaya_plot(fig2, 300), use_container_width=True)

# ---------------------------------------------------------------- tab PSI
with tab_psi:
    psi = mn.psi_fitur()
    if psi is None:
        st.info("PSI tidak bisa dihitung — split latih atau uji tidak tersedia.", icon="ℹ️")
    else:
        st.caption("PSI < 0,10 stabil · 0,10–0,25 perlu perhatian · > 0,25 pergeseran nyata. "
                   "Dihitung antara data latih dan periode uji out-of-time.")
        atas = psi.head(18).iloc[::-1]
        warna = [AKSEN if v < 0.10 else (AMBER if v < 0.25 else MERAH) for v in atas["psi"]]
        fig = go.Figure(go.Bar(
            x=atas["psi"], y=atas["fitur"], orientation="h", marker_color=warna,
            hovertemplate="<b>%{y}</b><br>PSI %{x:.3f}<extra></extra>",
        ))
        fig.add_vline(x=0.10, line_dash="dot", line_color=AMBER)
        fig.add_vline(x=0.25, line_dash="dot", line_color=MERAH)
        fig.update_layout(xaxis_title="Population stability index", yaxis_title=None,
                          title="Delapan belas fitur dengan pergeseran terbesar")
        st.plotly_chart(gaya_plot(fig, 520), use_container_width=True)

        perhatian = psi[psi["psi"] >= 0.10]
        if len(perhatian):
            st.warning("Fitur yang perlu perhatian: "
                       + ", ".join(f"`{f}`" for f in perhatian["fitur"]), icon="⚠️")
        else:
            st.success("Seluruh fitur berada pada rentang stabil.", icon="✅")

# --------------------------------------------------------------- tab agen
with tab_agen:
    st.markdown(
        kartu(
            "Qwen 14B sebagai judge arena",
            "Keluaran agen dinilai berpasangan oleh model penilai <b>Qwen 14B</b> yang terpisah "
            "dari model yang dipakai agen. Tiap kasus uji dijalankan dua varian agen, "
            "keduanya diadu, dan penilai memilih yang lebih baik beserta alasannya. "
            "Nilai di bawah adalah tingkat kemenangan dan skor rubrik dari arena tersebut.",
            warna=UNGU, ikon="⚖️",
        ),
        unsafe_allow_html=True,
    )
    agen = dummy_data.evaluasi_agen()
    tampil = agen.copy()
    tampil["lulus"] = tampil["nilai"] >= tampil["ambang"]
    fig = px.bar(
        tampil, x="nilai", y="metrik", orientation="h",
        color="lulus", color_discrete_map={True: AKSEN, False: MERAH},
        labels={"nilai": "Nilai", "metrik": "", "lulus": "Memenuhi ambang"},
    )
    fig.update_layout(xaxis_range=[0, 1.05], showlegend=False,
                      title="Metrik lapisan agen terhadap ambang penerimaan")
    st.plotly_chart(gaya_plot(fig, 430), use_container_width=True)

    tabel = agen.copy()
    for kolom in ("nilai", "ambang"):
        tabel[kolom] = tabel[kolom].map(lambda v: f"{v:.2f}".replace(".", ","))
    st.dataframe(
        tabel.rename(columns={"metrik": "Metrik", "nilai": "Nilai", "ambang": "Ambang terima"}),
        use_container_width=True, hide_index=True,
    )
    st.info(
        "Penilai hanya memutus mutu penalaran dan kelengkapan sitasi. Angka pada memo tidak "
        "ikut dinilai model — angka dibandingkan langsung dengan keluaran tool, dan selisih "
        "berapa pun dihitung sebagai gagal.",
        icon="🛡️",
    )

# ------------------------------------------------------------ tab kualitas
with tab_kualitas:
    gerbang = dummy_data.gerbang_kualitas_data()
    ikon = {"Lulus": "✅", "Lulus dengan perbaikan": "🛠️", "Perlu telaah": "⚠️"}
    tampil = gerbang.copy()
    tampil["hasil"] = tampil["hasil"].map(lambda h: f"{ikon.get(h, '•')} {h}")
    st.dataframe(
        tampil.rename(columns={"pemeriksaan": "Pemeriksaan", "hasil": "Hasil",
                               "baris_karantina": "Baris masuk karantina"}),
        use_container_width=True, hide_index=True,
    )
    total = int(gerbang["baris_karantina"].sum())
    kol1, kol2 = st.columns(2)
    kol1.metric("Total baris pada tabel karantina", cacah(total))
    abt = mn.gold("abt_pd")
    if abt is not None:
        kol2.metric("Baris ABT PD siap model", cacah(len(abt)))
    st.caption(
        "Tingkat kekotoran data dan pola afiliasi tersembunyi diinjeksi sendiri serta "
        "didokumentasikan pada sebuah spesifikasi, sehingga kualitas sebelum dan sesudah "
        "pipeline maupun recall deteksi dapat diukur secara objektif."
    )

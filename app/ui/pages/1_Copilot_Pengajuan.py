"""Halaman 1 — Copilot pengajuan.

Relationship manager mengetik ringkasan pengajuan dalam bahasa bebas; jejak
langkah agen muncul bertahap, lalu keluar skor, reason code, dan draft memo.

Hasil pemanggilan disimpan di `st.session_state` supaya jejak agen tidak
terhapus saat halaman dijalankan ulang oleh interaksi berikutnya.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib import dummy_data, memo as memo_lib, mock_engine
from lib.format import persen, rupiah
from lib.tampilan import badge_keputusan, kartu_hasil, plot_kontribusi, setup_halaman, sidebar_status

setup_halaman("Copilot pengajuan", "🤖")
sidebar_status()

st.title("1 · Copilot pengajuan")
st.caption("Masukan bahasa bebas → agen merencanakan pemanggilan tool → rekomendasi keputusan.")

with st.sidebar:
    st.divider()
    mode_luring = st.toggle(
        "Mode demo luring", value=True,
        help="Mengalihkan pemanggilan LLM ke respons yang sudah direkam. "
             "Selalu aktif selama layanan agen belum tersedia.",
    )
    kecepatan = st.select_slider("Kecepatan jejak agen", ["lambat", "sedang", "cepat"], value="sedang")
JEDA = {"lambat": 0.75, "sedang": 0.35, "cepat": 0.05}[kecepatan]

# ---------------------------------------------------------------- masukan
pilihan = st.selectbox(
    "Contoh kasus yang sudah disiapkan",
    options=list(range(len(dummy_data.CONTOH_PROMPT))),
    format_func=lambda i: f"Kasus {i + 1} — {dummy_data.CONTOH_PROMPT[i][:70]}…",
)
teks = st.text_area(
    "Ringkasan pengajuan dan hasil kunjungan",
    value=dummy_data.CONTOH_PROMPT[pilihan],
    height=130,
    placeholder="Contoh: Warung kelontong di Bekasi, usaha jalan empat tahun, omzet Rp 45 juta per bulan…",
)

kol_tombol, kol_info = st.columns([1, 3])
jalankan = kol_tombol.button("Jalankan copilot", type="primary", use_container_width=True)
kol_info.caption(
    "Penguji dipersilakan mengetik kasus baru. Ekstraksi entitas dan urutan tool akan menyesuaikan isi masukan."
)

if jalankan:
    entitas = dummy_data.ekstraksi_entitas(teks)
    application_id = f"APP-DEMO-{abs(hash(teks)) % 9000 + 1000}"
    rencana = dummy_data.rencana_agen(entitas)

    st.session_state["copilot_input"] = teks
    st.session_state["copilot_entitas"] = entitas
    st.session_state["copilot_app_id"] = application_id

    with st.status("Agen sedang merencanakan pemanggilan tool...", expanded=True) as status:
        st.write("**Langkah 0 · Ekstraksi entitas dan validasi skema**")
        st.json(entitas, expanded=False)
        time.sleep(JEDA)

        jejak = []
        for i, langkah in enumerate(rencana, start=1):
            st.write(f"**Langkah {i} · `{langkah['tool']}({langkah['arg']})`** — {langkah['keterangan']}")
            time.sleep(JEDA)
            jejak.append(langkah)

        status.update(
            label=f"Selesai — {len(rencana)} tool dipanggil"
            + (" (mode luring)" if mode_luring else ""),
            state="complete",
            expanded=False,
        )

    fitur = dict(entitas)
    fitur.update(
        margin_usaha=0.14,
        stabilitas_arus_kas=0.30 if entitas["indikasi_konsentrasi_pembeli"] else 0.20,
        supplier_concentration_hhi=0.62 if entitas["indikasi_konsentrasi_pembeli"] else 0.34,
        neighbor_default_rate_1hop=0.09 if entitas["indikasi_penjamin_berulang"] else 0.04,
        tenure_nasabah_thn=max(entitas["lama_usaha_thn"] - 1.5, 0.0),
    )
    network_risk = dummy_data.score_network_risk(application_id)
    fitur["network_risk_score"] = network_risk["skor"]

    st.session_state["copilot_fitur"] = fitur
    st.session_state["copilot_network"] = network_risk
    st.session_state["copilot_hasil"] = mock_engine.recommend_limit_pricing(fitur)
    st.session_state["copilot_jejak"] = jejak

# ---------------------------------------------------------------- keluaran
if "copilot_hasil" not in st.session_state:
    st.info("Tekan **Jalankan copilot** untuk memulai.", icon="▶️")
    st.stop()

entitas = st.session_state["copilot_entitas"]
fitur = st.session_state["copilot_fitur"]
hasil: mock_engine.HasilSkor = st.session_state["copilot_hasil"]
network_risk = st.session_state["copilot_network"]
application_id = st.session_state["copilot_app_id"]
keputusan = mock_engine.keputusan_dari_hasil(hasil)

st.divider()
st.markdown(f"### Rekomendasi &nbsp; {badge_keputusan(keputusan)}", unsafe_allow_html=True)
st.caption(f"Nomor pengajuan demo: `{application_id}` · Sistem merekomendasikan, pejabat pemutus memutuskan.")

kartu_hasil(hasil, entitas["plafon"])

k1, k2, k3 = st.columns(3)
k1.metric("Loss given default", persen(hasil.lgd))
k2.metric("Debt service coverage", f"{hasil.dscr:.2f}x", help="Ambang kebijakan minimum 1,35x")
k3.metric("Angsuran per bulan", rupiah(hasil.angsuran, singkat=True))

tab_alasan, tab_jaringan, tab_kebijakan, tab_memo = st.tabs(
    ["Reason code", "Risiko jaringan", "Kebijakan yang dirujuk", "Draft credit memo"]
)

with tab_alasan:
    st.markdown("**Faktor pendorong utama keputusan** — merah menaikkan risiko, hijau menurunkan.")
    st.plotly_chart(plot_kontribusi(hasil.kontribusi), use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            [{"Fitur": k.fitur, "Nilai": k.nilai, "Dampak (log-odds)": round(k.dampak, 4)}
             for k in hasil.kontribusi]
        ),
        use_container_width=True, hide_index=True,
    )

with tab_jaringan:
    skor = network_risk["skor"]
    st.metric("Skor risiko jaringan", f"{skor:.0f} / 100")
    st.progress(min(skor / 100, 1.0))
    st.caption(
        "Skor ini dilaporkan terpisah dan tidak dilebur ke dalam PD, agar alasannya tetap dapat dibaca."
    )
    if network_risk["pola"]:
        for p in network_risk["pola"]:
            st.markdown(
                f'<div class="kotak"><b>{p["deskripsi"]}</b><br>'
                f'<span style="opacity:.7">Bukti: {p["bukti"]} · kode <code>{p["kode"]}</code></span></div>',
                unsafe_allow_html=True,
            )
        st.page_link("pages/3_Jaringan_Entitas.py", label="Lihat subgraf sebagai bukti", icon="🕸️")
    else:
        st.success("Tidak ada pola anomali jaringan yang terpicu.", icon="✅")

with tab_kebijakan:
    st.caption("Hasil RAG atas dokumen kebijakan kredit internal.")
    for p in dummy_data.kutipan_kebijakan(entitas):
        st.markdown(
            f'<div class="kotak"><b>{p["pasal"]}</b> '
            f'<span style="opacity:.6">· kemiripan {p["skor"]:.2f}</span><br>{p["isi"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("**Dokumen yang masih kurang**")
    for d in dummy_data.dokumen_kurang(entitas):
        st.checkbox(d, value=False, key=f"dok-{d}")

with tab_memo:
    teks_memo = memo_lib.susun_memo(
        application_id, entitas, hasil, network_risk,
        dummy_data.kutipan_kebijakan(entitas), dummy_data.dokumen_kurang(entitas),
    )
    st.download_button(
        "Unduh draft credit memo (.md)",
        data=teks_memo.encode("utf-8"),
        file_name=f"credit_memo_{application_id}.md",
        mime="text/markdown",
        type="primary",
    )
    st.markdown(teks_memo)

"""Halaman 1 — Copilot pengajuan komersial.

Relationship manager mengetik ringkasan pengajuan dalam bahasa bebas; jejak
langkah agen muncul bertahap, lalu keluar skor, gerbang kepatuhan, reason code,
dan draft credit memo.

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
from lib.format import miliar
from lib.tampilan import (
    badge_grade,
    badge_keputusan,
    kartu_hasil,
    kartu_rasio,
    panel_gerbang,
    plot_bmpk,
    plot_kontribusi,
    setup_halaman,
    sidebar_status,
)

setup_halaman("Copilot pengajuan", "🤖")
sidebar_status()

st.title("1 · Copilot pengajuan")
st.caption(
    "Masukan bahasa bebas → agen merencanakan pemanggilan tool → gerbang kepatuhan → "
    "rekomendasi keputusan."
)

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
    format_func=lambda i: f"Kasus {i + 1} — {dummy_data.CONTOH_PROMPT[i][:80]}…",
)
teks = st.text_area(
    "Ringkasan pengajuan dan hasil kunjungan relationship manager",
    value=dummy_data.CONTOH_PROMPT[pilihan],
    height=150,
    placeholder="Contoh: PT Sumber Logam Perkasa, manufaktur komponen otomotif di Karawang, "
                "penjualan Rp 240 miliar, EBITDA margin 11 persen, DER 1,8x…",
)

kol_tombol, kol_info = st.columns([1, 3])
jalankan = kol_tombol.button("Jalankan copilot", type="primary", use_container_width=True)
kol_info.caption(
    "Penguji dipersilakan mengetik kasus baru. Ekstraksi entitas dan urutan tool akan menyesuaikan "
    "isi masukan — termasuk memanggil tool graf tambahan bila terdeteksi indikasi afiliasi."
)

if jalankan:
    entitas = dummy_data.ekstraksi_entitas(teks)
    application_id = f"APP-DEMO-{abs(hash(teks)) % 9000 + 1000}"
    rencana = dummy_data.rencana_agen(entitas)

    st.session_state["copilot_input"] = teks
    st.session_state["copilot_entitas"] = entitas
    st.session_state["copilot_app_id"] = application_id

    with st.status("Agen sedang merencanakan pemanggilan tool...", expanded=True) as status:
        st.write("**Langkah 0 · Ekstraksi entitas dan validasi skema (Pydantic)**")
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

    # Fitur yang tidak muncul pada narasi diisi dari riwayat nasabah dan lapisan
    # graf; pada sistem sebenarnya nilainya datang dari warehouse dan Redis.
    fitur = dict(entitas)
    fitur.update(
        utang_berbunga_eksisting=entitas["plafon"] * 0.25,
        konversi_ebitda_kas=0.62 if entitas["indikasi_konsentrasi_pembeli"] else 0.76,
        utilisasi_plafon=0.72,
        buyer_concentration_hhi=0.71 if entitas["indikasi_konsentrasi_pembeli"] else 0.32,
        supplier_concentration_hhi=0.66 if entitas["indikasi_konsentrasi_pemasok"] else 0.30,
        neighbor_default_rate_1hop=0.09 if entitas["indikasi_rangkap_jabatan"] else 0.035,
        group_exposure_share=min(0.28 + 0.11 * entitas["jumlah_entitas_grup"], 0.95),
        tenure_nasabah_thn=max(entitas["umur_usaha_thn"] - 6.0, 0.0),
    )
    network_risk = dummy_data.score_network_risk(application_id)
    fitur["network_risk_score"] = network_risk["skor"]

    hasil = mock_engine.recommend_limit_pricing(fitur)
    st.session_state["copilot_fitur"] = fitur
    st.session_state["copilot_network"] = network_risk
    st.session_state["copilot_hasil"] = hasil
    st.session_state["copilot_gerbang"] = mock_engine.check_credit_policy(hasil, fitur)
    st.session_state["copilot_jejak"] = jejak

# ---------------------------------------------------------------- keluaran
if "copilot_hasil" not in st.session_state:
    st.info("Tekan **Jalankan copilot** untuk memulai.", icon="▶️")
    st.stop()

entitas = st.session_state["copilot_entitas"]
fitur = st.session_state["copilot_fitur"]
hasil: mock_engine.HasilSkor = st.session_state["copilot_hasil"]
network_risk = st.session_state["copilot_network"]
gerbang = st.session_state["copilot_gerbang"]
application_id = st.session_state["copilot_app_id"]
keputusan = mock_engine.keputusan_dari_hasil(hasil, gerbang)
status_patuh = mock_engine.status_kepatuhan(gerbang)

st.divider()
st.markdown(
    f"### Rekomendasi &nbsp; {badge_keputusan(keputusan)} &nbsp; {badge_grade(hasil.grade)}",
    unsafe_allow_html=True,
)
st.caption(
    f"{entitas.get('nama_debitur', '-')} · nomor pengajuan demo `{application_id}` · "
    f"kewenangan {hasil.komite_pemutus} — sistem merekomendasikan, komite memutuskan."
)

if status_patuh == mock_engine.PENYESUAIAN:
    st.error(
        "Rekomendasi tidak lolos gerbang kepatuhan, sehingga tidak ditampilkan sebagai usulan "
        "setuju. Lihat tab **Gerbang kepatuhan** untuk pasal dan penyesuaian angka yang "
        "membuatnya patuh.",
        icon="⛔",
    )
elif status_patuh == mock_engine.TELAAH:
    st.warning(
        "Gerbang kepatuhan memicu penelaahan lanjutan atas struktur afiliasi sebelum akad.",
        icon="🔎",
    )

kartu_hasil(hasil, entitas["plafon"])
st.markdown("**Rasio keuangan terhadap ambang covenant kelas rating**")
kartu_rasio(hasil)

tab_gerbang, tab_alasan, tab_jaringan, tab_grup, tab_kebijakan, tab_memo = st.tabs(
    ["Gerbang kepatuhan", "Reason code", "Risiko jaringan", "Eksposur grup",
     "Kebijakan yang dirujuk", "Draft credit memo"]
)

with tab_gerbang:
    st.caption(
        "Setiap rekomendasi melewati gerbang ini sebelum ditampilkan — kepatuhan diperiksa "
        "sebelum keluar, bukan sesudah komite menemukan masalahnya."
    )
    panel_gerbang(gerbang)

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
        "Skor ini dilaporkan terpisah dan tidak dilebur ke dalam PD, agar alasannya tetap "
        "dapat dibaca komite kredit."
    )
    if network_risk["pola"]:
        for p in network_risk["pola"]:
            st.markdown(
                f'<div class="kotak"><b>{p["deskripsi"]}</b><br>'
                f'<span style="opacity:.7">Bukti: {p["bukti"]} · kode <code>{p["kode"]}</code></span></div>',
                unsafe_allow_html=True,
            )
        st.page_link("pages/3_Struktur_Grup_dan_Jaringan.py",
                     label="Lihat subgraf struktur grup sebagai bukti", icon="🕸️")
    else:
        st.success("Tidak ada pola anomali struktur yang terpicu.", icon="✅")

with tab_grup:
    g1, g2, g3 = st.columns(3)
    g1.metric("Entitas satu grup", entitas.get("jumlah_entitas_grup", 1))
    g2.metric("Eksposur grup berjalan", miliar(hasil.eksposur_grup, 0))
    g3.metric("Sisa ruang BMPK", miliar(hasil.ruang_bmpk, 0),
              delta=f"batas {miliar(mock_engine.BATAS_BMPK_GRUP, 0)}", delta_color="off")
    st.plotly_chart(
        plot_bmpk(hasil.eksposur_grup, hasil.limit_usulan), use_container_width=True
    )
    st.caption(
        "Seluruh entitas yang dikendalikan pemilik manfaat yang sama digabungkan sebagai satu "
        "grup debitur sebelum sisa ruang batas dihitung."
    )

with tab_kebijakan:
    st.caption(
        "Hasil RAG atas korpus kebijakan kredit komersial. Versi kebijakan diikat pada tanggal "
        "pengajuan, bukan yang berlaku hari ini."
    )
    for p in dummy_data.kutipan_kebijakan(entitas):
        st.markdown(
            f'<div class="kotak"><b>{p["pasal"]}</b> '
            f'<span style="opacity:.6">· kemiripan {p["skor"]:.2f} · {p.get("versi", "")}</span><br>'
            f'{p["isi"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("**Dokumen yang masih kurang**")
    for d in dummy_data.dokumen_kurang(entitas):
        st.checkbox(d, value=False, key=f"dok-{d}")

with tab_memo:
    teks_memo = memo_lib.susun_memo(
        application_id, entitas, hasil, network_risk,
        dummy_data.kutipan_kebijakan(entitas), dummy_data.dokumen_kurang(entitas),
        gerbang=gerbang,
    )
    st.download_button(
        "Unduh draft credit memo (.md)",
        data=teks_memo.encode("utf-8"),
        file_name=f"credit_memo_{application_id}.md",
        mime="text/markdown",
        type="primary",
    )
    st.markdown(teks_memo)

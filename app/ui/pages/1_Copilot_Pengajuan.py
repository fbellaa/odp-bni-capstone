"""Halaman 1 — Copilot pengajuan komersial.

Satu halaman untuk seluruh rantai pengajuan, gabungan dari dua halaman lama
(copilot demo dan copilot lokal):

    unggahan empat berkas PDF
        -> pembacaan dokumen (model bahasa lokal, atau sapuan pola bila luring)
        -> entitas gabungan
        -> agen memanggil tool perhitungan
        -> model PD (XGBoost terkalibrasi) dan LGD (XGBoost)
        -> pemetaan klaster portofolio
        -> gerbang kepatuhan, reason code, draft credit memo

Angka PD, LGD, dan posisi klaster berasal dari artefak `ml/models` di atas data
`data/gold`, bukan dari rumus tiruan. Bagian yang belum punya model — skor
risiko jaringan dan kutipan kebijakan tanpa index — masih memakai lapisan demo
dan diberi tanda pada tampilannya.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib import copilot_lokal as ck
from lib import dummy_data, memo as memo_lib, mock_engine, model_nyata as mn
from lib import pipeline_copilot as pc
from lib.format import cacah, miliar, persen
from lib.tampilan import (
    ABU,
    JINGGA,
    JINGGA_GELAP,
    TOSCA_TUA,
    badge,
    baris_status,
    badge_grade,
    badge_keputusan,
    hero,
    judul_bagian,
    kartu,
    kartu_hasil,
    kartu_rasio,
    meter_pd,
    panel_gerbang,
    plot_bmpk,
    plot_jarak_klaster,
    plot_klaster,
    plot_kontribusi,
    setup_halaman,
    sidebar_status,
)

setup_halaman("Copilot pengajuan")
sidebar_status()

status = pc.status_lengkap()

hero(
    "01",
    "Copilot pengajuan",
    "Unggah empat berkas pengajuan: laporan keuangan tiga periode, data kepemilikan, "
    "rekening koran, dan nota analisa kredit. Copilot membaca berkas, memanggil tool "
    "perhitungan, lalu menutupnya dengan skor model dan draft credit memo.",
    [
        ("model PD", "XGBoost" if status["pd"] else "belum ada"),
        ("model LGD", "XGBoost" if status["lgd"] else "belum ada"),
        ("baris ABT emas", cacah(status["baris_abt"])),
        ("pembaca dokumen", "LLM lokal" if status["llm_siap"] else "sapuan pola"),
    ],
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.divider()
    st.caption("KESIAPAN LAPISAN")
    for label, siap, pesan_kurang in [
        ("Model PD", status["pd"], "ml/models/pd_champion.joblib"),
        ("Model LGD", status["lgd"], "ml/models/final_lgd_xgboost.pkl"),
        ("Data emas", status["gold"], "data/gold/*.parquet"),
        ("Model bahasa lokal", status["llm_siap"],
         "jalankan `ollama serve`" if not status["ollama"]
         else "tarik model: `ollama pull " + " ".join(status.get("model_kurang") or []) + "`"),
        ("Index kebijakan", status["index"], "python -m copilot.rag.indeks"),
    ]:
        st.markdown(baris_status(label, siap, pesan_kurang), unsafe_allow_html=True)
    for nama_model, pesan in (status.get("galat_muat") or {}).items():
        st.error(f"Model {nama_model.upper()} gagal dimuat — `{pesan}`")
    if not status["copilot"]:
        st.warning("Paket `copilot` tidak bisa diimpor — unggahan PDF dinonaktifkan.")
        st.code(status["galat_impor"] or "-", language="text")
    st.divider()
    kecepatan = st.select_slider("Kecepatan jejak agen", ["lambat", "sedang", "cepat"],
                                 value="sedang")
JEDA = {"lambat": 0.6, "sedang": 0.28, "cepat": 0.03}[kecepatan]

# ------------------------------------------------------------------ masukan
judul_bagian(
    "Berkas pengajuan",
    "Seluruh masukan model datang dari berkas. Tidak ada angka yang diketik ulang, "
    "sehingga tiap angka di memo bisa ditunjuk balik ke halaman PDF-nya.",
)

kol_unggah, kol_daftar = st.columns([1.15, 1], gap="large")

with kol_unggah:
    st.markdown("**Unggah berkas (PDF)**")
    unggahan = st.file_uploader(
        "Berkas PDF", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=not status["copilot"],
        help="Laporan keuangan tiga periode, data kepemilikan, rekening koran, dan "
             "nota analisa kredit. Berkas hanya diproses di mesin ini.",
    )
    jenis_manual: dict[str, str] = {}
    if unggahan:
        opsi = ["(tebak otomatis)"] + list(pc.JENIS_DOKUMEN)
        for berkas_unggah in unggahan:
            pilih = st.selectbox(
                berkas_unggah.name, opsi,
                format_func=lambda o: pc.JENIS_DOKUMEN.get(o, o),
                key=f"jenis-{berkas_unggah.name}",
            )
            if pilih != "(tebak otomatis)":
                jenis_manual[berkas_unggah.name] = pilih
    st.caption(
        "Berkas yang tidak diunggah tidak membuat halaman gagal — field yang "
        "seharusnya ada di sana ditandai `bawaan` dan memakai asumsi sistem."
    )

with kol_daftar:
    # Kelengkapan ditampilkan sebelum tombol ditekan, bukan sesudahnya: analis
    # perlu tahu berkas mana yang kurang selagi masih bisa menambahkannya.
    st.markdown("**Kelengkapan berkas**")
    terunggah = pc.jenis_unggahan(unggahan, jenis_manual) if unggahan else set()
    st.markdown(
        "".join(
            kartu(nama, "sudah diunggah" if kunci in terunggah else "belum diunggah",
                  warna=TOSCA_TUA if kunci in terunggah else ABU)
            for kunci, nama in pc.JENIS_DOKUMEN.items()
        ),
        unsafe_allow_html=True,
    )

# Jalur LLM hanya dipilih kalau modelnya benar-benar ada. Ollama yang menjawab
# tanpa model terpasang menghasilkan dokumen kosong tanpa galat di layar.
pakai_llm = status["llm_siap"] and status["copilot"]
kol_tombol, kol_opsi, kol_info = st.columns([1.1, 1.2, 2.4])
jalankan = kol_tombol.button("Jalankan copilot", type="primary", use_container_width=True)
pakai_agen = kol_opsi.toggle(
    "Agen tool calling", value=pakai_llm, disabled=not pakai_llm,
    help="Menyalakan agen perhitungan berbasis model bahasa lokal. Tanpa Ollama, "
         "urutan tool tetap ditampilkan tetapi dijalankan secara deterministik.",
)
if status["ollama"] and not status["llm_siap"]:
    st.warning(
        "Ollama hidup tetapi model belum ditarik: "
        f"`{', '.join(status.get('model_kurang') or [])}`. Dokumen dibaca dengan "
        "sapuan pola. Jalankan `ollama pull "
        f"{' '.join(status.get('model_kurang') or [])}` untuk memakai jalur model bahasa."
    )
kol_info.caption(
    ("Model bahasa lokal aktif — dokumen dibaca dan agen memanggil tool sungguhan."
     if pakai_llm else
     "Dokumen dibaca dengan sapuan pola dan tool dijalankan langsung tanpa model "
     "bahasa; skor PD, LGD, dan klaster tetap dari model asli.")
)

# ----------------------------------------------------------------- eksekusi
if jalankan:
    # Nomor pengajuan datang dari nota analisa. Tanpa nota, nomor diturunkan dari
    # nama berkas supaya satu berkas yang sama selalu menghasilkan nomor yang sama
    # antar pemutaran demo.
    kunci_berkas = "|".join(sorted(u.name for u in unggahan)) if unggahan else "kosong"
    application_id = f"APP-{abs(hash(kunci_berkas)) % 9000 + 1000}"
    with st.status("Copilot mulai bekerja…", expanded=True) as kotak:
        # 1. dokumen
        dokumen = None
        if unggahan and status["copilot"]:
            st.write(f"**Langkah 1 · Membaca {len(unggahan)} berkas PDF**")
            path_list = [pc.simpan_unggahan(u) for u in unggahan]
            try:
                dokumen = (pc.baca_dengan_llm(path_list, jenis_manual) if pakai_llm
                           else pc.baca_dengan_pola(path_list, jenis_manual))
                st.write(
                    f"Jalur `{dokumen.jalur}` · {len(dokumen.per_berkas)} berkas terbaca · "
                    f"{len(dokumen.fakta)} pos keuangan ditemukan"
                )
            except Exception as exc:
                st.warning(f"Pembacaan dokumen gagal: {exc}")
        else:
            st.write("**Langkah 1 · Tidak ada berkas diunggah — seluruh field memakai asumsi sistem**")
        time.sleep(JEDA)

        # 2. entitas
        st.write("**Langkah 2 · Ekstraksi entitas dan validasi skema**")
        entitas, asal = pc.entitas_dari_dokumen(dokumen)
        if dokumen is not None and dokumen.pengajuan.get("nomor_pengajuan"):
            application_id = str(dokumen.pengajuan["nomor_pengajuan"])
        st.json({k: v for k, v in entitas.items() if not isinstance(v, dict)}, expanded=False)
        time.sleep(JEDA)

        # 3. rencana dan pemanggilan tool
        rencana = dummy_data.rencana_agen(entitas)
        st.write(f"**Langkah 3 · Agen memanggil {len(rencana)} tool**")
        jejak_agen = None
        if pakai_agen:
            try:
                pengajuan_agen = {
                    "plafon": entitas["plafon"],
                    "tenor_bulan": int(entitas["tenor_bulan"]),
                    "bunga_tahunan": 0.105,
                    "jenis_fasilitas": entitas["jenis_fasilitas"],
                    "jenis_agunan": entitas["jenis_agunan"],
                    "nilai_agunan": entitas["nilai_agunan"],
                    "eksposur_grup_berjalan": mock_engine.BATAS_BMPK_GRUP * 0.42,
                    "kewajiban_tahunan_eksisting": entitas["plafon"] * 0.08,
                    "pd_12bulan": 0.04,
                }
                # Agen membaca fakta sebagai teks datar. Tanpa dokumen, berkas
                # kosong tetap dikirim supaya kontraknya tidak berubah dan agen
                # tahu bahwa dokumen memang tidak ada.
                berkas_agen = (dokumen.berkas if dokumen and dokumen.berkas
                               else ck.BerkasPengajuan(nama_debitur=entitas.get("nama_debitur")))
                konteks = ck.memo_copilot.konteks_pengajuan(berkas_agen, pengajuan_agen, None)
                penampung, terkumpul = st.empty(), []

                def catat(j) -> None:
                    terkumpul.append(f"{'✔' if j.berhasil else '✘'} `{j.nama}` — {j.ringkas()}")
                    penampung.markdown("\n\n".join(terkumpul))

                jejak_agen = ck.AgenPerhitungan().jalankan(konteks, saat_alat=catat)
            except Exception as exc:
                st.warning(f"Agen tool calling tidak bisa dijalankan: {exc}")
        if jejak_agen is None:
            for i, langkah in enumerate(rencana, start=1):
                st.write(f"`{langkah['tool']}({langkah['arg']})` — {langkah['keterangan']}")
                time.sleep(JEDA / 3)

        # 4. model
        st.write("**Langkah 4 · Model PD, LGD, dan pemetaan klaster**")
        fitur = pc.lengkapi_fitur_graf(entitas, application_id)
        hasil_pd = mn.skor_pd(entitas)
        lgd_model = mn.skor_lgd(entitas)
        posisi = mn.posisi_klaster(entitas, hasil_pd) if hasil_pd else None
        if hasil_pd:
            fitur["pd_model"] = hasil_pd.pd_kalibrasi
            st.write(
                f"PD terkalibrasi **{persen(hasil_pd.pd_kalibrasi)}** · {hasil_pd.band}"
                + (f" · LGD **{persen(lgd_model)}**" if lgd_model is not None else "")
                + (f" · klaster terdekat **{posisi.nama}**" if posisi else "")
            )
        else:
            st.warning("Artefak model PD tidak ditemukan — skor memakai mesin demo.")
        if lgd_model is not None:
            fitur["lgd_model"] = lgd_model

        hasil = mock_engine.recommend_limit_pricing(fitur)
        gerbang = mock_engine.check_credit_policy(hasil, fitur)
        jaringan = dummy_data.score_network_risk(application_id)
        time.sleep(JEDA)

        kotak.update(
            label=f"Selesai — {application_id} · "
                  f"{'agen LLM' if jejak_agen else 'jalur deterministik'}",
            state="complete", expanded=False,
        )

    st.session_state.update(
        copilot_berkas=[u.name for u in unggahan] if unggahan else [],
        copilot_entitas=entitas, copilot_asal=asal,
        copilot_app_id=application_id, copilot_fitur=fitur, copilot_hasil=hasil,
        copilot_gerbang=gerbang, copilot_network=jaringan, copilot_dokumen=dokumen,
        copilot_pd=hasil_pd, copilot_lgd=lgd_model, copilot_posisi=posisi,
        copilot_jejak_agen=jejak_agen, copilot_rencana=rencana,
    )

# ------------------------------------------------------------------ hasil
if "copilot_hasil" not in st.session_state:
    st.info("Unggah berkas pengajuan, lalu tekan **Jalankan copilot**.")
    st.stop()

entitas = st.session_state["copilot_entitas"]
asal = st.session_state["copilot_asal"]
fitur = st.session_state["copilot_fitur"]
hasil: mock_engine.HasilSkor = st.session_state["copilot_hasil"]
gerbang = st.session_state["copilot_gerbang"]
jaringan = st.session_state["copilot_network"]
application_id = st.session_state["copilot_app_id"]
dokumen: pc.HasilDokumen | None = st.session_state.get("copilot_dokumen")
hasil_pd: mn.HasilPD | None = st.session_state.get("copilot_pd")
lgd_model = st.session_state.get("copilot_lgd")
posisi = st.session_state.get("copilot_posisi")
jejak_agen = st.session_state.get("copilot_jejak_agen")

keputusan = mock_engine.keputusan_dari_hasil(hasil, gerbang)
status_patuh = mock_engine.status_kepatuhan(gerbang)

st.divider()
st.markdown(
    f"### Rekomendasi &nbsp; {badge_keputusan(keputusan)} &nbsp; {badge_grade(hasil.grade)}"
    + (f" &nbsp; {badge(hasil_pd.band, hasil_pd.warna)}" if hasil_pd else ""),
    unsafe_allow_html=True,
)
st.caption(
    f"{entitas.get('nama_debitur', '-')} · nomor pengajuan `{application_id}` · "
    f"kewenangan {hasil.komite_pemutus} — sistem merekomendasikan, komite memutuskan."
)

if status_patuh == mock_engine.PENYESUAIAN:
    st.error(
        "Rekomendasi tidak lolos gerbang kepatuhan sehingga tidak ditampilkan sebagai usulan "
        "setuju. Lihat tab **Gerbang kepatuhan** untuk pasal dan penyesuaian angkanya.",
    )
elif status_patuh == mock_engine.TELAAH:
    st.warning("Gerbang kepatuhan memicu penelaahan struktur afiliasi sebelum akad.")

kartu_hasil(hasil, entitas["plafon"])
st.markdown("**Rasio keuangan terhadap ambang covenant kelas rating**")
kartu_rasio(hasil)

# --------------------------------------------------- skor model dan klaster
judul_bagian(
    "Skor model dan posisi klaster",
    "PD dari XGBoost terkalibrasi; klaster dibangun tanpa label default, "
    "label hanya dipakai untuk menamai tiap klaster sesudahnya.",
)
kol_meter, kol_peta = st.columns([1, 2], gap="large")

with kol_meter:
    if hasil_pd:
        st.plotly_chart(meter_pd(hasil_pd.pd_kalibrasi, hasil_pd.cutoffs, hasil_pd.warna),
                        use_container_width=True)
        st.caption(
            f"PD sebelum kalibrasi {persen(hasil_pd.pd_mentah)} · ambang portofolio "
            f"q50 {persen(hasil_pd.cutoffs['q50'])}, q80 {persen(hasil_pd.cutoffs['q80'])}, "
            f"q95 {persen(hasil_pd.cutoffs['q95'])}."
        )
        if lgd_model is not None:
            st.metric("LGD model XGBoost", persen(lgd_model),
                      help="Kerugian bila gagal bayar, dari model LGD terlatih.")
            st.caption(
                "Model LGD dilatih atas tenor, porsi penjaminan, jenis fasilitas, sektor, "
                "skala pegawai, dan kelengkapan dokumen. Jenis agunan belum menjadi fiturnya, "
                "jadi perbedaan struktur agunan baru terasa pada rantai limit dan covenant."
            )
    else:
        st.info("Model PD belum tersedia; angka pada kartu di atas berasal dari mesin demo.")

with kol_peta:
    ruang = mn.ruang_klaster()
    if ruang is not None and posisi is not None:
        st.plotly_chart(plot_klaster(ruang, posisi), use_container_width=True)
        condong = posisi.condong_default
        warna = JINGGA_GELAP if condong > 0.6 else (JINGGA if condong > 0.45 else TOSCA_TUA)
        st.markdown(
            kartu(
                f"Klaster terdekat: {posisi.nama}",
                f"Tingkat default historis klaster ini <b>{persen(posisi.tingkat_default_klaster, 1)}</b>. "
                f"Kecondongan ke kantong default <b>{persen(condong, 0)}</b> — dihitung dari jarak "
                f"relatif terhadap klaster paling berisiko dan klaster paling sehat.",
                warna=warna,
            ),
            unsafe_allow_html=True,
        )
    else:
        st.info("Ruang klaster belum bisa dibangun — data emas atau scikit-learn tidak tersedia.")

# ------------------------------------------------------------------- tab
tab_klaster, tab_gerbang, tab_alasan, tab_dokumen, tab_jaringan, tab_grup, \
    tab_kebijakan, tab_tool, tab_memo = st.tabs(
        ["Peta klaster", "Gerbang kepatuhan", "Reason code", "Dokumen terbaca",
         "Risiko jaringan", "Eksposur grup", "Tanya kebijakan", "Jejak tool",
         "Draft credit memo"]
    )

with tab_klaster:
    if ruang is None:
        st.info("Ruang klaster belum tersedia.")
    else:
        st.caption(
            f"K-Means atas {cacah(len(ruang.titik))}"
            + " pengajuan berlabel pada data emas, dengan 12 fitur keuangan dan graf. "
            "Sumbu grafik adalah dua komponen utama, jadi jarak pada gambar hanyalah "
            "bayangan dari jarak sebenarnya — tabel di bawah memakai jarak penuh."
        )
        if posisi is not None:
            st.plotly_chart(plot_jarak_klaster(posisi.jarak), use_container_width=True)
        tampil = ruang.ringkas.copy()
        tampil["tingkat_default"] = tampil["tingkat_default"].map(lambda v: persen(v, 2))
        kolom_tampil = ["nama", "jumlah", "tingkat_default"] + mn.FITUR_KLASTER[:6]
        st.dataframe(
            tampil[kolom_tampil].rename(columns={
                "nama": "Klaster", "jumlah": "Anggota", "tingkat_default": "Tingkat default",
                **{k: mn.NAMA_FITUR_KLASTER[k] for k in mn.FITUR_KLASTER[:6]},
            }).round(3),
            use_container_width=True, hide_index=True,
        )

with tab_gerbang:
    st.caption("Kepatuhan diperiksa sebelum rekomendasi keluar, bukan sesudah komite "
               "menemukan masalahnya.")
    panel_gerbang(gerbang)

with tab_alasan:
    if hasil_pd and hasil_pd.kontribusi:
        st.markdown("**Faktor pendorong PD menurut nilai SHAP model** — merah menaikkan "
                    "risiko, hijau menurunkan.")
        st.plotly_chart(plot_kontribusi(hasil_pd.kontribusi), use_container_width=True)
        st.dataframe(
            pd.DataFrame([
                {"Fitur": k.fitur, "Nilai": k.nilai, "Dampak (log-odds)": round(k.dampak, 4)}
                for k in hasil_pd.kontribusi
            ]),
            use_container_width=True, hide_index=True,
        )
        if hasil_pd.fitur_rujukan:
            with st.expander(
                f"{len(hasil_pd.fitur_rujukan)} fitur diisi median portofolio, bukan dari berkas"
            ):
                st.caption(
                    "Fitur graf dan riwayat tidak ada pada dokumen pengajuan. Pada sistem "
                    "sebenarnya nilainya datang dari warehouse dan lapisan graf."
                )
                st.write(", ".join(f"`{mn.label_fitur(f)}`" for f in hasil_pd.fitur_rujukan))
    else:
        st.plotly_chart(plot_kontribusi(hasil.kontribusi), use_container_width=True)
        st.caption("Reason code dari mesin demo karena model PD tidak tersedia.")

with tab_dokumen:
    if dokumen is None:
        st.info("Tidak ada dokumen diunggah pada jalannya copilot terakhir.")
    else:
        jalur = "model bahasa lokal" if dokumen.jalur == "llm" else "sapuan pola tanpa LLM"
        st.caption(f"Dibaca lewat {jalur}.")
        # Jumlah kolom mengikuti jumlah jenis dokumen, bukan angka tetap: dengan
        # tiga kolom, jenis keempat hilang diam-diam dari daftar kelengkapan.
        kelengkapan = dokumen.kelengkapan()
        kol = st.columns(len(kelengkapan))
        for k, (jenis, ada) in zip(kol, kelengkapan.items()):
            k.metric(pc.JENIS_DOKUMEN[jenis], "ada" if ada else "kurang")
        if dokumen.per_berkas:
            st.dataframe(
                pd.DataFrame(dokumen.per_berkas).rename(columns={
                    "berkas": "Berkas", "jenis": "Jenis", "halaman": "Halaman terbaca",
                    "total_halaman": "Total halaman", "pos_terbaca": "Catatan",
                }),
                use_container_width=True, hide_index=True,
            )
        if dokumen.fakta:
            st.markdown("**Pos keuangan hasil pembacaan**")
            st.dataframe(
                pd.DataFrame([
                    {"Pos": k.replace("_", " ").capitalize(),
                     "Nilai": miliar(v) if isinstance(v, (int, float)) else v}
                    for k, v in dokumen.fakta.items()
                ]),
                use_container_width=True, hide_index=True,
            )
        if dokumen.pemegang_saham:
            st.markdown("**Pemegang saham**")
            st.dataframe(
                pd.DataFrame([
                    {"Nama": p["nama"], "Porsi": f"{p['porsi']:.1%}",
                     "Jenis": p.get("jenis", "-")}
                    for p in dokumen.pemegang_saham
                ]),
                use_container_width=True, hide_index=True,
            )
        if dokumen.pengurus:
            st.markdown("**Pengurus**")
            st.write(" · ".join(dokumen.pengurus))
        for catatan in dokumen.catatan:
            st.warning(catatan)

        st.markdown("**Asal-usul angka yang dipakai model**")
        # Yang berasal dari berkas ditampilkan lebih dulu, lalu yang memakai
        # asumsi sistem. Bagian kedua itu yang paling perlu dilihat komite:
        # ia menyebut persis field mana yang tidak ada di berkas mana pun.
        dari_berkas = {k: s for k, s in asal.items()
                       if s in ("dokumen", "turunan dokumen", "pengajuan")}
        bawaan = [k for k, s in asal.items() if s == "bawaan"]
        st.markdown(
            " ".join(
                badge(f"{kunci}: {sumber}", pc.SUMBER_WARNA.get(sumber, "#8b97a6"))
                for kunci, sumber in dari_berkas.items()
            ) or "<span class='tipis'>Tidak ada angka yang berhasil diambil dari berkas.</span>",
            unsafe_allow_html=True,
        )
        if bawaan:
            st.caption(
                f"{len(bawaan)} field memakai asumsi sistem karena tidak tertulis di "
                f"berkas mana pun: {', '.join(sorted(bawaan))}."
            )

with tab_jaringan:
    skor = jaringan["skor"]
    st.metric("Skor risiko jaringan", f"{skor:.0f} / 100")
    st.progress(min(skor / 100, 1.0))
    st.caption(
        "Skor ini dilaporkan terpisah dan tidak dilebur ke dalam PD, supaya alasannya tetap "
        "dapat dibaca komite. Lapisan graf demo — belum memakai model anomali terlatih."
    )
    if jaringan["pola"]:
        for p in jaringan["pola"]:
            st.markdown(
                kartu(p["deskripsi"], f"Bukti: {p['bukti']} · kode <code>{p['kode']}</code>",
                      warna=JINGGA),
                unsafe_allow_html=True,
            )
        st.page_link("pages/3_Struktur_Grup_dan_Jaringan.py",
                     label="Lihat subgraf struktur grup sebagai bukti")
    else:
        st.success("Tidak ada pola anomali struktur yang terpicu.")

with tab_grup:
    g1, g2, g3 = st.columns(3)
    g1.metric("Entitas satu grup", entitas.get("jumlah_entitas_grup", 1))
    g2.metric("Eksposur grup berjalan", miliar(hasil.eksposur_grup, 0))
    g3.metric("Sisa ruang BMPK", miliar(hasil.ruang_bmpk, 0),
              delta=f"batas {miliar(mock_engine.BATAS_BMPK_GRUP, 0)}", delta_color="off")
    st.plotly_chart(plot_bmpk(hasil.eksposur_grup, hasil.limit_usulan),
                    use_container_width=True)
    st.caption(
        "Seluruh entitas yang dikendalikan pemilik manfaat yang sama digabung sebagai satu "
        "grup debitur sebelum sisa ruang batas dihitung."
    )

with tab_kebijakan:
    st.caption(
        "Jawaban disusun hanya dari korpus `docs/policies` dengan sitasi pasal. Bila aturannya "
        "tidak ada di korpus, copilot mengatakannya — bukan mengarang."
    )
    if status["index"] and status["copilot"]:
        for peran, isi in st.session_state.get("ck_chat", []):
            with st.chat_message(peran):
                st.markdown(isi)
        pertanyaan = st.chat_input("Contoh: kapan kredit wajib digolongkan kurang lancar?")
        if pertanyaan:
            riwayat = st.session_state.setdefault("ck_chat", [])
            riwayat.append(("user", pertanyaan))
            with st.chat_message("user"):
                st.markdown(pertanyaan)
            with st.chat_message("assistant"):
                with st.spinner("Mencari pasal yang relevan…"):
                    jawaban = ck.jawab_kebijakan(pertanyaan)
                sitasi = "\n".join(
                    f"- `{s['rujukan']}` (kemiripan {s['skor']})" for s in jawaban["sitasi"]
                )
                isi = jawaban["jawaban"] + (f"\n\n**Sitasi**\n{sitasi}" if sitasi else "")
                st.markdown(isi)
            riwayat.append(("assistant", isi))
    else:
        st.info("Index kebijakan belum dibangun. Jalankan sekali:\n\n"
                "```\npython -m copilot.rag.indeks\n```")
        st.markdown("**Kutipan kebijakan pada jalur demo**")
        for p in dummy_data.kutipan_kebijakan(entitas):
            st.markdown(
                kartu(p["pasal"], f"{p['isi']}<br><span class='tipis'>kemiripan "
                                  f"{p['skor']:.2f} · {p.get('versi', '')}</span>",
                      warna=TOSCA_TUA),
                unsafe_allow_html=True,
            )

with tab_tool:
    if jejak_agen is not None:
        st.caption("Setiap angka di bawah dihitung tool, bukan ditulis model bahasa.")
        if jejak_agen.ada_kegagalan:
            st.warning(
                f"{len(jejak_agen.rekaman.gagal())} pemanggilan tool gagal; angkanya tidak "
                "masuk memo dan harus dihitung manual.")
        st.dataframe(
            pd.DataFrame([
                {"Tool": j.nama, "Status": "berhasil" if j.berhasil else "gagal",
                 "Argumen": ", ".join(f"{k}={v}" for k, v in j.argumen.items()),
                 "Keterangan": (j.hasil or {}).get("rumus", j.galat or "-"),
                 "ms": j.durasi_ms}
                for j in jejak_agen.rekaman.jejak
            ]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("Urutan tool yang dipilih agen pada jalannya copilot terakhir.")
        st.dataframe(
            pd.DataFrame(st.session_state.get("copilot_rencana", [])).rename(columns={
                "tool": "Tool", "arg": "Argumen", "keterangan": "Keterangan"}),
            use_container_width=True, hide_index=True,
        )
    if status["copilot"]:
        with st.expander("Katalog tool yang boleh dipanggil agen"):
            st.dataframe(pd.DataFrame(ck.ringkas_katalog()),
                         use_container_width=True, hide_index=True)

with tab_memo:
    teks_memo = memo_lib.susun_memo(
        application_id, entitas, hasil, jaringan,
        dummy_data.kutipan_kebijakan(entitas), dummy_data.dokumen_kurang(entitas),
        gerbang=gerbang,
    )
    st.download_button(
        "Unduh draft credit memo (.md)",
        data=teks_memo.encode("utf-8"),
        file_name=f"credit_memo_{application_id}.md",
        mime="text/markdown", type="primary",
    )
    st.markdown(teks_memo)

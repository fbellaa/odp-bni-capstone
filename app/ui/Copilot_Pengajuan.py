"""Skrip entry aplikasi — Copilot pengajuan komersial.

Jalankan dari akar proyek:

    streamlit run app/ui/Copilot_Pengajuan.py

Halaman ini sekaligus beranda. Beranda ringkasan portofolio yang dulu ada
dihapus karena ia mengulang angka yang toh muncul lagi pada halaman kerja, dan
menaruh satu klik di depan pekerjaan yang sebenarnya.

Satu halaman untuk seluruh rantai pengajuan, gabungan dari dua halaman lama
(copilot demo dan copilot lokal):

    unggahan empat berkas PDF
        -> pembacaan dokumen (model bahasa lokal, atau sapuan pola bila luring)
        -> entitas gabungan
        -> agen memanggil tool perhitungan
        -> model PD dan LGD (XGBoost, artefak ml/artifacts)
        -> pemetaan klaster portofolio
        -> gerbang kepatuhan, reason code, draft credit memo

Angka PD, LGD, dan posisi klaster berasal dari artefak `ml/artifacts` di atas data
`data/gold`, bukan dari rumus tiruan. Rujukan kebijakan pada memo dikutip dari
korpus `docs/policies` yang terindeks; bila korpus tidak bisa ditelusuri,
bagian itu kosong beserta sebabnya, bukan diisi pasal karangan. Yang belum
punya model — skor risiko jaringan, serta rantai limit dan pricing sesudah PD
dan LGD — masih memakai lapisan demo dan diberi tanda pada tampilannya.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from lib import copilot_lokal as ck
from lib import dummy_data, kebijakan as kb, memo as memo_lib, mock_engine
from lib import model_nyata as mn
from lib import parameter_kebijakan as pk
from lib import graf_nyata as gn
from lib import pipeline_copilot as pc
from lib import risiko_jaringan as rj
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
# Parameter kebijakan diturunkan dari lapisan emas sekali per proses, sebelum
# mesin skoring dipakai sama sekali. Laporannya menyebut mana yang sudah datang
# dari tabel dan mana yang masih asumsi bawaan.
pk.terapkan()

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
        ("Model PD", status["pd"], "ml/artifacts/pd/pd_champion_new.joblib"),
        ("Model LGD", status["lgd"], "ml/artifacts/lgd/final_lgd_xgboost_new.pkl"),
        ("Klaster portofolio", status["klaster"],
         "ml/artifacts/pd_cluster/pd_cluster_champion.joblib"),
        ("Peringatan dini afiliasi", status["ews"],
         "ml/artifacts/ews/ews_xgboost_champion.joblib"),
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
# Jeda antar langkah pada jejak agen. Dulu bisa digeser dari sidebar; sekarang
# tetap, karena satu-satunya nilai yang pernah dipakai saat demo adalah ini.
JEDA = 0.28

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
        "Laporan keuangan dan nota analisa pengajuan wajib. Dua berkas sisanya "
        "boleh menyusul — field yang seharusnya ada di sana ditandai `bawaan` dan "
        "memakai asumsi sistem."
    )

with kol_daftar:
    # Kelengkapan ditampilkan sebelum tombol ditekan, bukan sesudahnya: analis
    # perlu tahu berkas mana yang kurang selagi masih bisa menambahkannya.
    st.markdown("**Kelengkapan berkas**")
    terunggah = pc.jenis_unggahan(unggahan, jenis_manual) if unggahan else set()
    st.markdown(
        "".join(
            kartu(
                nama + (" · wajib" if kunci in pc.DOKUMEN_WAJIB else ""),
                "sudah diunggah" if kunci in terunggah
                else ("belum diunggah — wajib" if kunci in pc.DOKUMEN_WAJIB
                      else "belum diunggah"),
                warna=TOSCA_TUA if kunci in terunggah
                else (JINGGA_GELAP if kunci in pc.DOKUMEN_WAJIB else ABU),
            )
            for kunci, nama in pc.JENIS_DOKUMEN.items()
        ),
        unsafe_allow_html=True,
    )

# Dua berkas wajib karena keduanya menentukan angka, bukan sekadar melengkapi:
# laporan keuangan mengisi seluruh rasio yang dipakai model, nota analisa
# mengisi struktur fasilitas yang diminta. Tanpa keduanya copilot masih bisa
# jalan - dulu memang begitu - tetapi yang keluar adalah skor atas asumsi
# segmen yang terlihat persis seperti skor atas berkas nasabah.
wajib_kurang = [pc.JENIS_DOKUMEN[k] for k in pc.DOKUMEN_WAJIB if k not in terunggah]

# Jalur LLM hanya dipilih kalau modelnya benar-benar ada. Ollama yang menjawab
# tanpa model terpasang menghasilkan dokumen kosong tanpa galat di layar.
pakai_llm = status["llm_siap"] and status["copilot"]
kol_tombol, kol_opsi, kol_info = st.columns([1.1, 1.2, 2.4])
jalankan = kol_tombol.button(
    "Jalankan copilot", type="primary", use_container_width=True,
    disabled=bool(wajib_kurang) or not status["copilot"],
    help=("Belum bisa dijalankan — " + ", ".join(wajib_kurang) + " belum diunggah."
          if wajib_kurang else None),
)
if wajib_kurang:
    st.warning(
        "Copilot menunggu berkas wajib: **" + "**, **".join(wajib_kurang) + "**. "
        "Berkas lain boleh menyusul."
    )
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
                dokumen = pc.baca_dokumen_pengajuan(
                    path_list, jenis_manual, boleh_llm=pakai_llm
                )
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

        # 3. risiko jaringan — resolusi afiliasi atas data graf nyata.
        # Pemohon baru belum punya simpul di graf, jadi dokumennya dicocokkan
        # dulu ke debitur eksisting; indikatornya mengukur lingkungan hasil
        # cocokan itu. Dijalankan sebelum model karena fitur graf yang sama
        # ikut mengisi payload skoring.
        st.write("**Langkah 3 · Resolusi afiliasi dan indikator risiko jaringan**")
        snapshot = gn.snapshot_tersedia()
        tanggal_telaah = snapshot[0] if snapshot else None
        resolusi = None
        if tanggal_telaah is not None and dokumen is not None and dokumen.berkas is not None:
            arg = dokumen.berkas.argumen_resolusi()
            resolusi = gn.resolusi_calon(
                tanggal_telaah,
                alamat_operasional=arg["alamat_operasional"],
                nama_pengurus=tuple(arg["nama_pengurus"]),
                rekening_lawan=tuple(arg["rekening_lawan"]),
            )
        hasil_jaringan = rj.skor_jaringan(
            application_id, resolusi, tanggal_telaah or pd.Timestamp.today()
        )
        jaringan = hasil_jaringan.sebagai_dict()
        # Peringatan dini dibaca atas afiliasi, bukan atas pemohon: fitur EWS
        # seluruhnya perilaku fasilitas, dan pemohon baru belum punya satu pun.
        pantauan = mn.ews_afiliasi(
            tuple(hasil_jaringan.cif_tercocok),
            tanggal_telaah or pd.Timestamp.today(),
        )
        if pantauan is not None and not pantauan.tabel.empty:
            st.write(
                f"Peringatan dini afiliasi: **{pantauan.jumlah_alarm}** dari "
                f"{len(pantauan.tabel)} fasilitas di atas ambang alarm"
            )
        fitur = pc.lengkapi_fitur_graf(
            entitas, application_id, hasil_jaringan, tanggal_telaah)
        if fitur.get("eksposur_grup_rp") is not None:
            st.write(
                f"Grup debitur `{fitur['bmpk_grup_id']}` · eksposur berjalan "
                f"**{miliar(fitur['eksposur_grup_rp'], 0)}** dari batas "
                f"{miliar(fitur['batas_bmpk_rp'], 0)} "
                f"(`fact_eksposur_grup`, snapshot {fitur['bmpk_snapshot']:%b %Y})"
            )
            for c in fitur.get("catatan_bmpk") or []:
                st.caption(c)
        else:
            st.caption(
                "Eksposur BMPK grup tidak terukur — " + str(fitur.get("asal_bmpk"))
                + ". Batas kredit memakai porsi asumsi segmen."
            )
        st.write(
            f"{hasil_jaringan.jumlah_afiliasi} afiliasi tercocok · indikator "
            + (f"**{hasil_jaringan.skor:.0f}/100**" if hasil_jaringan.skor is not None
               else "**tidak dapat dihitung**")
        )
        time.sleep(JEDA)

        # 4. rencana dan pemanggilan tool
        rencana = dummy_data.rencana_agen(entitas)
        # Judulnya dipesan dulu dan diisi belakangan. Berapa tool yang dipanggil
        # baru diketahui setelah agen selesai: ia memilih sendiri, mengulang tool
        # yang argumennya ditolak, dan berhenti kapan ia menganggap cukup.
        # Mencetak panjang `rencana` di sini - rencana tiruan yang pada jalur
        # agen nyata tidak pernah dieksekusi - berarti mengumumkan angka yang
        # kebetulan tidak sama dengan yang lewat di bawahnya.
        kepala_agen = st.empty()
        kepala_agen.write("**Langkah 4 · Agen memilih tool…**")
        jejak_agen, terkumpul = None, []
        if pakai_agen:
            try:
                pengajuan_agen = {
                    "plafon": entitas["plafon"],
                    "tenor_bulan": int(entitas["tenor_bulan"]),
                    "bunga_tahunan": 0.105,
                    "jenis_fasilitas": entitas["jenis_fasilitas"],
                    "jenis_agunan": entitas["jenis_agunan"],
                    "nilai_agunan": entitas["nilai_agunan"],
                    # Eksposur grup nyata bila cocokan afiliasi menunjuk satu
                    # grup debitur; tanpa itu kuncinya bernilai None dan tidak
                    # ikut dikirim ke agen - lebih baik agen tahu angkanya tidak
                    # ada daripada menerima porsi karangan.
                    "eksposur_grup_berjalan": fitur.get("eksposur_grup_rp"),
                    "kewajiban_tahunan_eksisting": entitas["plafon"] * 0.08,
                    "pd_12bulan": 0.04,
                }
                # Agen membaca fakta sebagai teks datar. Tanpa dokumen, berkas
                # kosong tetap dikirim supaya kontraknya tidak berubah dan agen
                # tahu bahwa dokumen memang tidak ada.
                # `fitur`, bukan `entitas`: turunan seperti utang berbunga
                # eksisting hanya ada di sana, dan tanpanya agen menghitung
                # rasio atas nol.
                berkas_agen = pc.berkas_untuk_agen(dokumen, fitur)
                konteks = ck.memo_copilot.konteks_pengajuan(berkas_agen, pengajuan_agen, None)
                penampung = st.empty()

                def catat(j) -> None:
                    terkumpul.append(f"{'✔' if j.berhasil else '✘'} `{j.nama}` — {j.ringkas()}")
                    penampung.markdown("\n\n".join(terkumpul))

                jejak_agen = ck.AgenPerhitungan().jalankan(konteks, saat_alat=catat)
            except Exception as exc:
                st.warning(f"Agen tool calling tidak bisa dijalankan: {exc}")
        if jejak_agen is None:
            kepala_agen.write(
                f"**Langkah 4 · Rencana tiruan: {len(rencana)} tool** — agen tidak "
                "dijalankan, daftar di bawah belum dieksekusi."
            )
            for i, langkah in enumerate(rencana, start=1):
                st.write(f"`{langkah['tool']}({langkah['arg']})` — {langkah['keterangan']}")
                time.sleep(JEDA / 3)
        elif not terkumpul:
            kepala_agen.write(
                "**Langkah 4 · Agen tidak memanggil tool apa pun** — "
                f"berhenti setelah {jejak_agen.putaran} putaran "
                f"(`{jejak_agen.berhenti_karena}`). Lihat tab Jejak tool."
            )
        else:
            gagal = sum(1 for b in terkumpul if b.startswith("✘"))
            kepala_agen.write(
                f"**Langkah 4 · Agen memanggil {len(terkumpul)} tool**"
                + (f" · {gagal} ditolak dan diulang" if gagal else "")
            )

        # 5. model
        st.write("**Langkah 5 · Model PD, LGD, dan pemetaan klaster**")
        hasil_pd = mn.skor_pd(entitas)
        lgd_model = mn.skor_lgd(entitas)
        posisi = mn.posisi_klaster(entitas, hasil_pd) if hasil_pd else None
        if hasil_pd:
            fitur["pd_model"] = hasil_pd.skor
            st.write(
                f"Skor default 12 bulan **{persen(hasil_pd.skor)}** · {hasil_pd.band}"
                + (f" · LGD **{persen(lgd_model)}**" if lgd_model is not None else "")
                + (f" · klaster terdekat **{posisi.nama}**" if posisi else "")
            )
        else:
            st.warning("Artefak model PD tidak ditemukan — skor memakai mesin demo.")
        if lgd_model is not None:
            fitur["lgd_model"] = lgd_model

        hasil = mock_engine.recommend_limit_pricing(fitur)
        gerbang = kb.lampirkan_rujukan(
            mock_engine.check_credit_policy(hasil, fitur))
        time.sleep(JEDA)

        # 6. rujukan kebijakan. Ditelusuri di sini, bukan saat tab memo dibuka,
        # karena tiap topik satu panggilan embedding: dijalankan pada tiap
        # rerun Streamlit, tab akan terasa menggantung tanpa sebab yang jelas.
        st.write("**Langkah 6 · Penelusuran korpus kebijakan**")
        rujukan, catatan_kebijakan = kb.rujukan_pengajuan(entitas)
        st.write(
            f"{len(rujukan)} pasal terkutip dari korpus"
            if rujukan else "**Tidak ada pasal terkutip** — memo menyebutkan sebabnya"
        )
        for c in catatan_kebijakan:
            st.caption(c)
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
        copilot_jaringan=hasil_jaringan, copilot_resolusi=resolusi,
        copilot_pd=hasil_pd, copilot_lgd=lgd_model, copilot_posisi=posisi,
        copilot_jejak_agen=jejak_agen, copilot_rencana=rencana,
        copilot_rujukan=rujukan, copilot_catatan_kebijakan=catatan_kebijakan,
        copilot_pantauan=pantauan,
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
hasil_jaringan: rj.HasilJaringan | None = st.session_state.get("copilot_jaringan")
pantauan: mn.PantauanEWS | None = st.session_state.get("copilot_pantauan")
rujukan = st.session_state.get("copilot_rujukan", [])
catatan_kebijakan = st.session_state.get("copilot_catatan_kebijakan", [])

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
st.markdown("**Rasio keuangan terhadap ambang covenant pita risiko**")
kartu_rasio(hasil)

# --------------------------------------------------- skor model dan klaster
judul_bagian(
    "Skor model dan posisi klaster",
    "Skor default dari XGBoost tanpa kalibrasi — yang dibaca pitanya; klaster "
    "dibangun tanpa label default, "
    "label hanya dipakai untuk menamai tiap klaster sesudahnya.",
)
kol_meter, kol_peta = st.columns([1, 2], gap="large")

with kol_meter:
    if hasil_pd:
        st.plotly_chart(meter_pd(hasil_pd.skor, hasil_pd.cutoffs, hasil_pd.warna),
                        use_container_width=True)
        st.caption(
            f"Pita **{hasil_pd.band}** · ambang portofolio q50 "
            f"{persen(hasil_pd.cutoffs['q50'])}, q80 {persen(hasil_pd.cutoffs['q80'])}, "
            f"q95 {persen(hasil_pd.cutoffs['q95'])}."
        )
        if not hasil_pd.terkalibrasi:
            st.caption(
                "Model versi ini tidak terkalibrasi: angkanya skor peringkat, bukan "
                "probabilitas gagal bayar yang boleh dibaca apa adanya. Yang bermakna "
                "adalah pita risikonya — posisi pengajuan ini terhadap portofolio."
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
         "Risiko jaringan", "Eksposur grup", "Rujukan kebijakan", "Jejak tool",
         "Draft credit memo"]
    )

with tab_klaster:
    if ruang is None:
        st.info("Ruang klaster belum tersedia.")
    else:
        st.caption(
            f"Artefak `pd_cluster` atas {cacah(len(ruang.titik))}"
            + f" pengajuan berlabel pada data emas: {len((mn.muat_klaster() or {}).get('features', []))} fitur "
            "diringkas PCA, lalu dikelompokkan K-Means. Sumbu grafik adalah dua komponen "
            "utama, jadi jarak pada gambar hanyalah bayangan dari jarak sebenarnya — "
            "tabel di bawah memakai jarak penuh pada ruang PCA."
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

    # Fitur perilaku dan graf tidak tertulis di berkas mana pun. Yang menentukan
    # apakah angkanya boleh dibela adalah asalnya — diukur atas pemohon, diukur
    # atas afiliasi tercocok, atau dipinjam dari median portofolio.
    asal_fitur = fitur.get("asal_fitur") or {}
    if asal_fitur:
        st.divider()
        st.markdown("**Asal fitur yang tidak tertulis di berkas**")
        st.caption(
            "Median portofolio bukan pengukuran atas pemohon ini. Ia dipakai supaya "
            "rantai perhitungan bisa jalan, dan disebut apa adanya supaya tidak "
            "terbaca sebagai angka nasabah."
        )
        st.dataframe(
            pd.DataFrame([
                {"Fitur": mn.label_fitur(k),
                 "Nilai": (miliar(fitur[k], 1) if k.endswith("_rp")
                           or k == "utang_berbunga_eksisting" else f"{fitur[k]:.4g}"),
                 "Asal": v}
                for k, v in asal_fitur.items() if k in fitur
            ]),
            use_container_width=True, hide_index=True,
        )

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
    if hasil_jaringan is None or not hasil_jaringan.tersedia:
        # "Tidak bisa diperiksa" tidak boleh terbaca sebagai "aman". Panelnya
        # menolak menampilkan angka apa pun, dan menyebut sebabnya.
        st.warning("Indikator risiko jaringan tidak dapat dihitung.")
        for c in (hasil_jaringan.catatan if hasil_jaringan else
                  ["Copilot belum dijalankan pada sesi ini."]):
            st.markdown(f"- {c}")
        st.caption(
            "Pemohon baru belum punya simpul di graf. Pencocokan butuh dokumen "
            "domisili usaha, akta/kepemilikan, atau rekening koran."
        )
    else:
        skor = hasil_jaringan.skor
        k1, k2, k3 = st.columns(3)
        k1.metric("Indikator risiko jaringan", f"{skor:.0f} / 100")
        k2.metric("Afiliasi tercocok", cacah(hasil_jaringan.jumlah_afiliasi))
        k3.metric("Sudah gagal bayar", cacah(hasil_jaringan.afiliasi_gagal_bayar))
        st.progress(min(skor / 100, 1.0))
        st.caption(
            "Komponennya dibaca dari data gold; **pembobotannya keputusan kebijakan, "
            "bukan model terlatih** — karena itu disebut indikator. Dilaporkan terpisah "
            "dan tidak dilebur ke dalam PD supaya alasannya tetap dapat dibaca komite."
        )
        if hasil_jaringan.perlu_telaah:
            st.warning("Ambang KKK-13.6 terpicu — afiliasi wajib ditelaah analis.")

        st.markdown("**Komponen penyusun indikator**")
        st.dataframe(
            pd.DataFrame([
                {"Komponen": k.label, "Nilai": k.mentah,
                 "Ternormalisasi": round(k.nilai, 3), "Bobot": k.bobot,
                 "Sumbangan ke skor": round(k.sumbangan, 1), "Sumber": k.sumber}
                for k in hasil_jaringan.komponen
            ]),
            use_container_width=True, hide_index=True,
        )

        if hasil_jaringan.pola:
            st.markdown("**Pola terdeteksi**")
            for p in hasil_jaringan.pola:
                st.markdown(
                    kartu(p["deskripsi"], f"Bukti: {p['bukti']} · kode <code>{p['kode']}</code>",
                          warna=JINGGA),
                    unsafe_allow_html=True,
                )
            st.page_link("pages/2_Struktur_Grup_dan_Jaringan.py",
                         label="Lihat subgraf struktur grup sebagai bukti")
        else:
            st.success("Tidak ada pola anomali struktur yang terpicu pada afiliasi tercocok.")

        st.divider()
        st.markdown("**Peringatan dini pada afiliasi tercocok**")
        st.caption(
            "Model EWS menilai fasilitas yang sudah berjalan, jadi ia tidak bisa dan "
            "tidak boleh dipakai menilai pemohon — pemohon baru tidak punya tunggakan, "
            "pemakaian plafon, atau covenant untuk dilanggar. Yang punya perilaku adalah "
            "afiliasinya, dan memburuknya mereka adalah pertanyaan kredit yang sah."
        )
        if pantauan is None:
            st.info("Artefak EWS atau panel `abt_ews` tidak tersedia.")
        elif pantauan.tabel.empty:
            st.info(
                f"{pantauan.cif_tanpa_fasilitas} debitur tercocok tidak punya fasilitas "
                f"pada panel bulanan sampai {pantauan.tanggal:%b %Y} — tidak ada yang "
                "bisa dipantau, dan itu bukan berarti bersih."
            )
        else:
            cacah_pita = pantauan.cacah_pita()
            e1, e2, e3 = st.columns(3)
            e1.metric("Fasilitas terpantau", cacah(len(pantauan.tabel)))
            e2.metric("Di atas ambang alarm", cacah(pantauan.jumlah_alarm),
                      delta=f"ambang {pantauan.ambang:.4f}", delta_color="off")
            e3.metric("Peringatan dini (HIGH)", cacah(cacah_pita.get("HIGH", 0)))
            if pantauan.jumlah_alarm:
                st.warning(
                    f"{pantauan.jumlah_alarm} fasilitas afiliasi berstatus alarm. Penularan "
                    "dalam satu grup wajib ditelaah sebelum fasilitas baru diputus."
                )
            tampil = pantauan.tabel.copy()
            tampil["skor"] = tampil["skor"].map(lambda v: f"{v:.4f}")
            tampil["pita"] = tampil["pita"].map(lambda v: f"{v} · {mn.PITA_EWS.get(v, v)}")
            tampil["snapshot"] = pd.to_datetime(tampil["snapshot"]).dt.strftime("%b %Y")
            tampil["pemakaian_plafon"] = tampil["pemakaian_plafon"].map(
                lambda v: "-" if pd.isna(v) else f"{float(v) * 100:.0f}%")
            st.dataframe(
                tampil[["cif_sk", "facility_id", "snapshot", "skor", "pita", "alarm",
                        "dpd", "kolektibilitas", "pemakaian_plafon"]].rename(columns={
                    "cif_sk": "CIF", "facility_id": "Fasilitas", "snapshot": "Snapshot",
                    "skor": "Skor EWS", "pita": "Pita", "alarm": "Alarm",
                    "dpd": "DPD (hari)", "kolektibilitas": "Kolektibilitas",
                    "pemakaian_plafon": "Pemakaian plafon"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Skor dibaca pada snapshot terakhir yang tidak melewati tanggal telaah, "
                "supaya penelaahan tidak memakai perilaku yang belum terjadi. Snapshot "
                "lama berarti fasilitasnya memang sudah lama tidak berjalan."
                + (f" {pantauan.cif_tanpa_fasilitas} debitur tercocok tidak punya fasilitas "
                   "pada panel." if pantauan.cif_tanpa_fasilitas else "")
            )

        for kode, sebab in hasil_jaringan.tak_diperiksa.items():
            st.caption(f"Belum diperiksa — `{kode}`: {sebab}")
        for c in hasil_jaringan.catatan:
            st.caption(c)

with tab_grup:
    g1, g2, g3 = st.columns(3)
    g1.metric("Entitas satu grup", entitas.get("jumlah_entitas_grup", 1))
    g2.metric("Eksposur grup berjalan", miliar(hasil.eksposur_grup, 0))
    g3.metric("Sisa ruang BMPK", miliar(hasil.ruang_bmpk, 0),
              delta=f"batas {miliar(hasil.batas_bmpk, 0)}", delta_color="off")
    st.plotly_chart(plot_bmpk(hasil.eksposur_grup, hasil.limit_usulan,
                              batas=hasil.batas_bmpk),
                    use_container_width=True)
    st.caption(
        "Seluruh entitas yang dikendalikan pemilik manfaat yang sama digabung sebagai satu "
        "grup debitur sebelum sisa ruang batas dihitung. Sumber angka eksposur: "
        f"**{hasil.sumber_bmpk}**"
        + (f" · grup `{fitur['bmpk_grup_id']}`" if fitur.get("bmpk_grup_id") else "")
        + "."
    )

with tab_kebijakan:
    st.caption(
        "Seluruh isi bagian ini dikutip dari korpus `docs/policies` dengan sitasi pasal. "
        "Bila aturannya tidak ada di korpus, copilot mengatakannya — bukan mengarang."
    )

    st.markdown("**Rujukan yang terkutip untuk pengajuan ini**")
    st.caption(
        "Daftar yang sama masuk ke bagian 6 draft credit memo. Tiap baris menyebut topik "
        "yang membuatnya terambil, supaya komite tahu kenapa pasal itu ada di sana."
    )
    for p in rujukan:
        st.markdown(
            kartu(
                p["pasal"],
                f"{p['isi']}<br><span class='tipis'>{kb.jejak_sumber(p)}"
                f"<br>ditelusuri untuk: {', '.join(p.get('topik') or ['-'])}</span>",
                warna=TOSCA_TUA,
            ),
            unsafe_allow_html=True,
        )
    if not rujukan:
        st.warning(
            "Tidak ada pasal yang bisa dikutip untuk pengajuan ini. Bagian rujukan pada memo "
            "dibiarkan kosong dan wajib diisi analis — sistem tidak menyusun pasal sendiri."
        )
    for c in catatan_kebijakan:
        st.caption(c)

    if not status["index"]:
        st.info("Index kebijakan belum dibangun. Jalankan sekali:\n\n"
                "```\npython -m copilot.rag.indeks\n```")

with tab_tool:
    if jejak_agen is not None and not jejak_agen.rekaman.jejak:
        # Agen jalan tetapi tidak memanggil satu tool pun. Tanpa penjelasan,
        # yang terlihat cuma tabel kosong - tidak terbedakan dari agen yang
        # tidak dinyalakan, padahal sebabnya bisa jauh berbeda: model tidak
        # bisa dihubungi, atau model menjawab dengan prosa alih-alih memanggil
        # tool. Yang kedua lazim pada model kecil, dan itu keterangan yang
        # menentukan apakah profilnya perlu dinaikkan.
        SEBAB = {
            "galat_model": "model bahasa tidak bisa dihubungi saat agen berjalan",
            "batas_putaran": f"batas {jejak_agen.putaran} putaran tercapai "
                             "sebelum satu tool pun dipanggil",
            "selesai": "model menjawab langsung tanpa memanggil tool — lazim pada "
                       "model kecil; naikkan peran agen lewat COPILOT_MODEL_AGEN",
        }
        st.warning(
            "Agen tidak memanggil tool apa pun — "
            + SEBAB.get(jejak_agen.berhenti_karena, jejak_agen.berhenti_karena)
            + f". Berhenti setelah {jejak_agen.putaran} putaran."
        )
        if (jejak_agen.ringkasan or "").strip():
            st.caption("Yang dijawab model alih-alih memanggil tool:")
            st.code(jejak_agen.ringkasan.strip(), language="text")
        st.caption(
            "Angka pada memo karena itu datang dari mesin deterministik, bukan "
            "dari tool yang dipanggil agen."
        )
    elif jejak_agen is not None:
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
    # Reason code memo mengikuti apa yang benar-benar menghitungnya: SHAP model
    # PD bila artefaknya ada, dan tidak ada apa-apa bila tidak. Pendorong dari
    # mesin demo tidak ikut ke dokumen yang diunduh.
    teks_memo = memo_lib.susun_memo(
        application_id, entitas, hasil, jaringan,
        rujukan, kb.dokumen_kurang(entitas, dokumen),
        gerbang=gerbang, catatan_kebijakan=catatan_kebijakan,
        pantauan_ews=pantauan,
        kontribusi=hasil_pd.kontribusi if hasil_pd else None,
        sumber_kontribusi=("nilai SHAP model PD XGBoost atas baris fitur pengajuan ini"
                           if hasil_pd else None),
    )
    st.download_button(
        "Unduh draft credit memo (.md)",
        data=teks_memo.encode("utf-8"),
        file_name=f"credit_memo_{application_id}.md",
        mime="text/markdown", type="primary",
    )
    st.markdown(teks_memo)

"""Halaman 3 — Hubungan pengaju dengan nasabah eksisting.

Halaman ini menjawab satu pertanyaan, bukan menyediakan penjelajahan graf:
**calon pengaju ini terhubung ke debitur mana di buku kita, lewat apa, dan apa
konsekuensinya.** Jawabannya punya akibat konkret - penggabungan BMPK,
penelaahan pihak terafiliasi (KKK-13.6), dan penularan gagal bayar.

DUA MODE, karena sumber relasinya memang berbeda:

    Calon pengaju     belum punya satu pun edge di GOLD_GRAPH_EDGES. Relasinya
                      dicari lewat `pipelines.graph.resolusi.telusuri_afiliasi()`
                      dari tiga berkas CDD: akta, domisili usaha, rekening koran.

    Nasabah eksisting sudah menjadi simpul di graf. Relasinya dibaca langsung
                      sebagai subgraf ego dari GOLD_GRAPH_NODES/EDGES.

Yang WAJIB dijaga di layar: "tidak ada kecocokan" dan "dokumen tidak disertakan"
tidak boleh sama-sama tampil sebagai "tidak ada afiliasi". Yang pertama temuan,
yang kedua batas penelaahan - dan menyamakannya mengubah *tidak tahu* menjadi
*aman*. Karena itu status tiap jalur selalu ditampilkan terpisah.

Setiap visual graf tetap didampingi tabel: graf memberi kesan, tabel memberi
angka yang dapat dikutip komite.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib import graf_nyata as gn
from lib.format import miliar, persen, rupiah
from lib.tampilan import (
    PALET_TIPE,
    badge,
    hero,
    judul_bagian,
    plot_bmpk,
    plot_graf,
    plot_kepemilikan,
    setup_halaman,
    sidebar_status,
)

setup_halaman("Hubungan pengaju dan nasabah eksisting")
sidebar_status()

hero(
    "03",
    "Hubungan pengaju dengan nasabah eksisting",
    "Calon pengaju dicocokkan ke debitur yang sudah ada di buku lewat akta, domisili usaha, "
    "dan rekening koran — tiga berkas yang memang wajib dikumpulkan saat CDD. Keluarannya "
    "daftar yang layak diperiksa analis, lengkap dengan alasan tiap barisnya, bukan skor risiko.",
    [("jalur pencocokan", "3"), ("dasar", "identitas & transaksi"),
     ("keluaran", "kandidat afiliasi + bukti")],
)

# ---------------------------------------------------------------- penjaga data
if not gn.tersedia():
    st.error(
        "Tabel graf belum ada di `data/gold`. Jalankan pipeline lebih dulu:\n\n"
        "```\npython -m pipelines.flows.main_flow\n```",
    )
    hilang = gn.tabel_hilang()
    if hilang:
        st.caption("Tidak ditemukan: " + ", ".join(f"`{t}`" for t in hilang))
    st.stop()

info = gn.ringkas_ketersediaan()
snapshots = gn.snapshot_tersedia()
snapshot_terbaru = snapshots[0] if snapshots else pd.Timestamp.today().normalize()

with st.expander("Data yang sedang dibaca", expanded=False):
    d1, d2, d3 = st.columns(3)
    d1.metric("Simpul", f"{info['jumlah_simpul']:,}".replace(",", "."))
    d2.metric("Relasi", f"{info['jumlah_edge']:,}".replace(",", "."))
    d3.metric("Snapshot bulanan", info["jumlah_snapshot"])
    st.caption(
        "Simpul menurut tipe: "
        + " · ".join(f"{k} {v:,}".replace(",", ".") for k, v in info["per_tipe_simpul"].items())
    )
    st.caption(
        "Relasi menurut tipe: "
        + " · ".join(f"{k} {v:,}".replace(",", ".") for k, v in info["per_tipe_relasi"].items())
    )
    if info["tabel_hilang"]:
        st.warning("Tabel pelengkap tidak ada: " + ", ".join(info["tabel_hilang"]))

tab_calon, tab_eksisting = st.tabs(
    ["Calon pengaju — pencocokan ke buku", "Nasabah eksisting — subgraf ego"]
)

# ==========================================================================
# MODE 1 — calon pengaju
# ==========================================================================
with tab_calon:
    dokumen = st.session_state.get("copilot_dokumen")
    entitas = st.session_state.get("copilot_entitas") or {}

    # Bahan pencocokan diambil dari berkas yang sudah dibaca halaman 1.
    # `BerkasPengajuan.argumen_resolusi()` sengaja dibentuk sama dengan tanda
    # tangan `telusuri_afiliasi()`, jadi tidak ada penyesuaian di sini.
    alamat_awal, pengurus_awal, rekening_awal = "", [], []
    asal = "isian manual"
    if dokumen is not None:
        berkas = getattr(dokumen, "berkas", None)
        if berkas is not None and hasattr(berkas, "argumen_resolusi"):
            arg = berkas.argumen_resolusi()
            alamat_awal = str(arg.get("alamat_operasional") or "")
            pengurus_awal = list(arg.get("nama_pengurus") or [])
            rekening_awal = list(arg.get("rekening_lawan") or [])
            asal = "berkas pengajuan (jalur pembacaan LLM)"
        elif getattr(dokumen, "pengurus", None):
            # Jalur pola menyimpan "Nama — Jabatan"; yang dipakai hanya namanya.
            pengurus_awal = [
                str(p).split("—")[0].split(" - ")[0].strip() for p in dokumen.pengurus
            ]
            asal = "berkas pengajuan (jalur pembacaan pola)"

    nama_calon = str(entitas.get("nama_debitur") or "Calon pengaju")

    if dokumen is None:
        st.info(
            "Belum ada berkas pengajuan yang dibaca di halaman 1. Isi bahan pencocokan "
            "di bawah untuk mencoba, atau jalankan Copilot Pengajuan lebih dulu.",
        )
    else:
        st.caption(f"Bahan pencocokan diambil dari **{asal}** untuk **{nama_calon}**.")

    with st.expander("Bahan pencocokan", expanded=dokumen is None):
        f1, f2 = st.columns([3, 2])
        alamat = f1.text_input(
            "Alamat operasional (dokumen domisili usaha)",
            value=alamat_awal,
            placeholder="Jalan …, Kota …",
        )
        tanggal = f2.date_input(
            "Tanggal penilaian",
            value=snapshot_terbaru.date(),
            help="Pencocokan hanya melihat relasi yang sudah berlaku dan gagal bayar "
                 "yang sudah terjadi pada tanggal ini.",
        )
        g1, g2 = st.columns(2)
        teks_pengurus = g1.text_area(
            "Nama pengurus / pemegang saham (akta) — satu per baris",
            value="\n".join(pengurus_awal), height=120,
        )
        teks_rekening = g2.text_area(
            "Rekening lawan transaksi (rekening koran) — satu per baris",
            value="\n".join(rekening_awal), height=120,
        )
        nama_calon = st.text_input("Nama calon pengaju", value=nama_calon)

    pengurus = tuple(b.strip() for b in teks_pengurus.splitlines() if b.strip())
    rekening = tuple(b.strip() for b in teks_rekening.splitlines() if b.strip())

    hasil = gn.resolusi_calon(
        pd.Timestamp(tanggal),
        alamat_operasional=alamat.strip() or None,
        nama_pengurus=pengurus,
        rekening_lawan=rekening,
    )

    if hasil.galat:
        st.error(f"Pencocokan gagal dijalankan: {hasil.galat}")
        st.stop()

    # ---- status tiap jalur. Inilah yang membedakan "tidak tahu" dari "aman".
    judul_bagian("Jalur pencocokan yang terpakai")
    lencana = []
    for j in hasil.jalur:
        warna = "#1b7f4b" if j["dipakai"] else (
            "#b58900" if "tidak disertakan" in j["keterangan"] else "#666666"
        )
        lencana.append(badge(f"{j['nama']} — {j['keterangan']}", warna))
    st.markdown(" ".join(lencana), unsafe_allow_html=True)

    # Jalur yang benar-benar dijalankan - entah ketemu atau tidak. Jalur yang
    # dokumennya tidak dilampirkan TIDAK termasuk: ia belum diperiksa.
    diperiksa = [j for j in hasil.jalur if "tidak disertakan" not in j["keterangan"]]
    tidak_disertakan = [j for j in hasil.jalur if "tidak disertakan" in j["keterangan"]]
    if tidak_disertakan:
        st.warning(
            "Batas penelaahan: "
            + ", ".join(j["nama"].lower() for j in tidak_disertakan)
            + " tidak disertakan, sehingga jalur itu belum diperiksa. "
            "Hasil di bawah bukan pernyataan bahwa tidak ada afiliasi.",
        )

    # ---- vonis kebijakan
    if hasil.perlu_telaah:
        st.error(
            f"**Penelaahan pihak terafiliasi diperlukan (KKK-13.6)** — {hasil.jumlah_kandidat} "
            "debitur eksisting terkait. Ambang ini aturan kebijakan, bukan keluaran model.",
        )
    elif hasil.jumlah_kandidat:
        st.info(
            f"{hasil.jumlah_kandidat} debitur terkait ditemukan, di bawah ambang penelaahan "
            "wajib. Tetap perlu dilihat untuk penggabungan BMPK.",
        )
    elif diperiksa:
        st.success(
            "Tidak ada kecocokan pada jalur yang diperiksa: "
            + ", ".join(j["nama"].lower() for j in diperiksa) + ".",
        )
    else:
        # Nol jalur diperiksa bukan hasil bersih. Menampilkannya hijau adalah
        # persis kekeliruan yang halaman ini ada untuk mencegahnya.
        st.warning(
            "Belum ada bahan pencocokan yang bisa diperiksa. Lampirkan akta, dokumen "
            "domisili, atau rekening koran — status di atas belum menyatakan apa pun "
            "tentang ada-tidaknya afiliasi.",
        )

    if hasil.jumlah_kandidat:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Debitur terkait", hasil.jumlah_kandidat)
        m2.metric("Sudah gagal bayar", hasil.ada_gagal_bayar,
                  help="Gagal bayar yang tanggalnya sudah lewat pada tanggal penilaian.")
        m3.metric("Eksposur terkait", miliar(float(hasil.tabel["eksposur_rp"].sum()), 0))
        m4.metric("Bukti berupa hub", int(hasil.tabel["hub"].sum()),
                  help=f"Objek bukti yang menempel pada ≥ {gn.AMBANG_HUB} debitur — "
                       "nominee atau alamat bersama massal, bukan afiliasi spesifik.")

        # Tabel dulu dan selebar halaman: dua belas kolom bukti tidak terbaca di
        # kolom sempit, dan angka inilah yang dikutip komite. Graf menyusul
        # sebagai ilustrasinya.
        st.markdown("**Debitur eksisting yang terkait**")
        tampil = hasil.tabel.copy()
        tampil["eksposur_rp"] = tampil["eksposur_rp"].map(
            lambda v: rupiah(v, singkat=True) if v else "—"
        )
        tampil["skor"] = tampil["skor"].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
        )
        tampil["ukuran_hub"] = tampil["ukuran_hub"].map(
            lambda v: f"{int(v)}" if pd.notna(v) else "—"
        )
        tampil["sudah_gagal_bayar"] = tampil["sudah_gagal_bayar"].map(
            {True: "ya", False: "—"}
        )
        tampil["hub"] = tampil["hub"].map({True: "hub", False: "—"})
        st.dataframe(
            tampil.rename(columns={
                "cif_sk": "CIF", "nama_debitur": "Debitur", "grup": "Grup",
                "dasar_utama": "Dasar terkuat", "semua_dasar": "Semua dasar",
                "bukti": "Bukti", "skor": "Skor cocok",
                "sudah_gagal_bayar": "Gagal bayar", "eksposur_rp": "Eksposur",
                "jumlah_fasilitas": "Fasilitas", "ukuran_hub": "Debitur pada objek ini",
                "hub": "Catatan",
            }),
            use_container_width=True, hide_index=True,
        )
        if hasil.dipangkas:
            st.caption(
                f"Menampilkan {len(hasil.tabel)} dari {hasil.jumlah_kandidat} kandidat. "
                "Urutan: bukti spesifik lebih dulu, hub di bawah."
            )
        st.caption(
            "Keluaran ini bukan skor risiko dan tidak dipakai sebagai dasar penolakan — "
            "ia memunculkan permintaan verifikasi kepada analis, termasuk pemeriksaan "
            "apakah debitur terkait seharusnya masuk perhitungan BMPK grup."
        )

        if bool(hasil.tabel["hub"].any()):
            st.caption(
                f"Perhatian: sebagian bukti menempel pada objek yang terkait ≥ {gn.AMBANG_HUB} debitur. "
                "`resolusi.py` sudah membuang alamat agen registrasi lewat `is_alamat_agen`, "
                "tetapi belum punya penyaring setara untuk pihak berderajat tinggi — baris "
                "seperti ini ditandai dan diturunkan peringkatnya, tidak dibuang."
            )

        judul_bagian("Peta hubungan")
        simpul_g, edge_g = gn.graf_resolusi(hasil.tabel, nama_calon)
        st.plotly_chart(
            plot_graf(simpul_g, edge_g, warnai="tipe", sorot=nama_calon),
            use_container_width=True,
        )
        st.markdown(
            " ".join(badge(t, w) for t, w in PALET_TIPE.items()
                     if t in set(simpul_g["tipe"])),
            unsafe_allow_html=True,
        )
        st.caption(
            "Objek bukti digambar sebagai simpul tersendiri, bukan label pada garis: "
            "debitur yang berbagi pengurus atau alamat yang sama langsung terlihat "
            "mengumpul pada satu titik. Simpul bertepi tebal adalah calon pengaju."
        )

# ==========================================================================
# MODE 2 — nasabah eksisting
# ==========================================================================
with tab_eksisting:
    daftar = gn.daftar_debitur()
    if daftar is None or daftar.empty:
        st.warning("Tidak ada debitur yang bisa dijadikan pusat subgraf.")
        st.stop()

    c1, c2, c3, c4 = st.columns([2.6, 1, 1, 1])
    pilihan = c1.selectbox(
        "Badan hukum pusat",
        options=daftar["node_id"].tolist(),
        format_func=lambda nid: (
            f"{daftar.loc[daftar['node_id'] == nid, 'nama'].iat[0]} "
            f"({int(daftar.loc[daftar['node_id'] == nid, 'derajat'].iat[0])} relasi)"
        ),
    )
    hops = c2.select_slider("Kedalaman (hop)", [1, 2, 3], value=2)
    batas = c3.number_input("Batas simpul", 20, 200, 60, step=10,
                            help="Subgraf dipangkas sebelum dikirim ke antarmuka.")
    warnai = c4.selectbox("Warna simpul", ["tipe", "komunitas"])

    baris_pilih = daftar.loc[daftar["node_id"] == pilihan].iloc[0]
    cif = int(baris_pilih["ref_id"])

    s1, s2 = st.columns([1, 3])
    mode_waktu = s1.radio(
        "Kondisi graf", ["Terkini", "Pada snapshot"], horizontal=False,
        help="Simpul dan relasi bertanggal. 'Terkini' memakai seluruh riwayat; "
             "'Pada snapshot' hanya relasi yang berlaku di akhir bulan terpilih.",
    )
    if mode_waktu == "Pada snapshot" and snapshots:
        pada = s2.select_slider(
            "Snapshot", options=list(reversed(snapshots)), value=snapshots[0],
            format_func=lambda t: pd.Timestamp(t).strftime("%b %Y"),
        )
        pada = pd.Timestamp(pada)
    else:
        pada = None
        s2.caption("Memakai seluruh riwayat relasi. Metrik klaster hanya tersedia "
                   "pada mode snapshot.")

    # Filter lapisan relasi. Memilih satu tipe berarti melihat jaringan pada
    # lapisan itu saja - "siapa yang terhubung ke debitur ini HANYA lewat
    # rangkap jabatan", misalnya. Filter bekerja pada penelusuran, bukan pada
    # gambar: simpul yang cuma bisa dicapai lewat relasi yang disaring keluar
    # ikut hilang, bukan tertinggal melayang tanpa garis.
    semua_rel = sorted(info["per_tipe_relasi"])
    rel_pilih = st.multiselect(
        "Lapisan relasi",
        options=semua_rel,
        default=semua_rel,
        format_func=lambda r: f"{r} ({info['per_tipe_relasi'][r]:,})".replace(",", "."),
        help="Kosongkan atau pilih semua untuk melihat seluruh lapisan sekaligus.",
    )
    rel_dipakai = (
        None if not rel_pilih or len(rel_pilih) == len(semua_rel)
        else tuple(rel_pilih)
    )

    hasil_sub = gn.subgraf_ego(int(pilihan), hops=int(hops),
                               batas_simpul=int(batas), pada=pada,
                               rel_dipakai=rel_dipakai)
    if hasil_sub is None or hasil_sub.nodes.empty:
        keterangan = "pada kondisi yang dipilih"
        if rel_dipakai:
            keterangan = "pada lapisan " + ", ".join(f"`{r}`" for r in rel_dipakai)
        st.warning(
            f"Debitur ini tidak punya relasi {keterangan}. "
            "Lapisan lain mungkin berisi — tambahkan tipe relasi di atas.",
        )
        st.stop()

    nodes, edges = hasil_sub.nodes, hasil_sub.edges

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Simpul pada subgraf", len(nodes))
    m2.metric("Relasi pada subgraf", len(edges))
    m3.metric(
        "Tetangga langsung", hasil_sub.total_tetangga,
        help="Dihitung pada lapisan relasi yang sedang dipilih, bukan pada graf penuh.",
    )
    m4.metric("Tipe relasi", edges["relasi"].nunique() if len(edges) else 0)

    if rel_dipakai:
        st.caption(
            "Lapisan aktif: " + ", ".join(f"`{r}`" for r in rel_dipakai)
            + ". Simpul yang hanya terhubung lewat lapisan lain tidak ikut ditelusuri."
        )

    if hasil_sub.dipangkas:
        st.info(
            f"Subgraf dipangkas ke {len(nodes)} simpul — debitur ini punya "
            f"{hasil_sub.total_tetangga} tetangga langsung. Yang tidak tampil bukan "
            "berarti tidak ada; naikkan batas simpul untuk melihat lebih banyak.",
        )

    kiri, kanan = st.columns([3, 2])
    with kiri:
        st.plotly_chart(
            plot_graf(nodes, edges, warnai=warnai, sorot=nodes["id"].iat[0]),
            use_container_width=True,
        )
        if warnai == "tipe":
            st.markdown(
                " ".join(badge(t, w) for t, w in PALET_TIPE.items()
                         if t in set(nodes["tipe"])),
                unsafe_allow_html=True,
            )
        else:
            st.caption("Warna menandai klaster Louvain pada snapshot terpilih.")

    with kanan:
        st.markdown("**Relasi menurut tipe**")
        if len(edges):
            ringkas = (
                edges["relasi"].value_counts()
                .rename_axis("Tipe relasi").reset_index(name="Jumlah")
            )
            st.dataframe(ringkas, use_container_width=True, hide_index=True)
        st.markdown("**Simpul menurut tipe**")
        st.dataframe(
            nodes["tipe"].value_counts().rename_axis("Tipe").reset_index(name="Jumlah"),
            use_container_width=True, hide_index=True,
        )

    # ---- kepemilikan
    st.divider()
    judul_bagian(
        "Pemilik tercatat",
        "Relasi kepemilikan pada data ini berhenti di satu lapis — pihak pemegang saham "
        "tidak pernah sekaligus menjadi debitur, sehingga tidak ada rantai berlapis untuk "
        "ditelusuri. Yang ditampilkan pemilik langsung, bukan penelusuran sampai "
        "pemilik manfaat akhir. Identitas relasinya berasal dari ICIJ; "
        "**angka porsinya sintesis** — ICIJ tidak memuat persentase saham, jadi porsi "
        "diundi lalu dinormalkan per tanggal sampai berjumlah 100%. Yang bermakna di "
        "sini urutan dan ukuran relatif antarpemilik, bukan besaran permodalannya.",
    )
    rantai = gn.kepemilikan_langsung(cif, pada)
    if rantai is None or rantai.empty:
        st.info("Tidak ada pemilik tercatat untuk debitur ini pada kondisi terpilih.")
    else:
        jumlah_pemilik = int(rantai["jumlah_pemilik"].iat[0])
        porsi_total = float(rantai["porsi_total"].iat[0])
        besar = rantai[rantai["porsi_langsung"] >= gn.AMBANG_PEMILIK_MANFAAT]

        k1, k2, k3 = st.columns(3)
        k1.metric("Pemilik tercatat", jumlah_pemilik)
        k2.metric(f"Porsi ≥ {gn.AMBANG_PEMILIK_MANFAAT:.0%}", len(besar),
                  help="Ambang yang lazim dipakai penelaahan APU-PPT untuk pemilik manfaat.")
        k3.metric("Total porsi tercatat", persen(porsi_total, 1))

        # Porsi dinormalkan per debitur per segmen waktu di pipeline, jadi ini
        # harus selalu 100%. Kalau tidak, yang rusak pipeline-nya - bukan sifat
        # datanya - dan halaman perlu menyebutnya, bukan diam-diam menampilkannya.
        if abs(porsi_total - 1.0) > 0.01:
            st.warning(
                f"Total porsi kepemilikan {persen(porsi_total, 1)}, seharusnya 100%. "
                "Porsi dinormalkan per debitur per rentang tanggal di "
                "`pipelines.graph.struktur._kapitalisasi_pit`; selisih di sini berarti "
                "normalisasi itu tidak berlaku untuk kondisi terpilih.",
            )

        ko, kt = st.columns([3, 2])
        with ko:
            st.plotly_chart(plot_kepemilikan(rantai, baris_pilih["nama"]),
                            use_container_width=True)
            if jumlah_pemilik > len(rantai):
                st.caption(
                    f"Menampilkan {len(rantai)} porsi terbesar dari {jumlah_pemilik} "
                    f"pemilik tercatat — mencakup "
                    f"{persen(float(rantai['porsi_langsung'].sum()), 1)} kepemilikan."
                )
        with kt:
            tampil = rantai.drop(columns=["porsi_total", "jumlah_pemilik", "tingkat",
                                          "aktivitas_usaha", "yurisdiksi"], errors="ignore")
            tampil["porsi_langsung"] = tampil["porsi_langsung"].map(lambda v: persen(v, 1))
            tampil["porsi_efektif"] = tampil["porsi_efektif"].map(lambda v: persen(v, 1))
            tampil["pengendali_efektif"] = tampil["pengendali_efektif"].map(
                {True: "ya", False: "—"}
            )
            st.dataframe(
                tampil.rename(columns={
                    "pemilik": "Pemilik", "jenis": "Jenis", "dimiliki": "Memiliki",
                    "porsi_langsung": "Porsi", "porsi_efektif": "Porsi efektif",
                    "pengendali_efektif": "Pengendali", "valid_from": "Berlaku dari",
                    "valid_to": "Sampai",
                }),
                use_container_width=True, hide_index=True,
            )

    # ---- BMPK
    st.divider()
    judul_bagian("Eksposur grup usaha terhadap BMPK")
    ringkas_grup = gn.eksposur_grup(baris_pilih.get("grup_id"), pada or snapshot_terbaru)
    if ringkas_grup is None:
        st.info(
            "Grup usaha debitur ini tidak punya baris eksposur pada tanggal tersebut. "
            "Angka BMPK tidak ditampilkan daripada menampilkan nol yang menyesatkan.",
        )
    else:
        g1, g2 = st.columns([2, 3])
        with g1:
            st.metric("Grup usaha", ringkas_grup["nama_grup"])
            if ringkas_grup["jumlah_entitas"] is not None:
                st.metric("Entitas dalam grup", ringkas_grup["jumlah_entitas"])
            st.metric("Eksposur gabungan", miliar(ringkas_grup["total_eksposur_rp"], 1),
                      delta=f"{persen(ringkas_grup['group_exposure_share'], 1)} dari batas",
                      delta_color="off")
            st.metric("Sisa ruang", miliar(ringkas_grup["sisa_ruang_rp"], 0))
        with g2:
            st.plotly_chart(
                plot_bmpk(ringkas_grup["total_eksposur_rp"],
                          batas=ringkas_grup["batas_bmpk_rp"]),
                use_container_width=True,
            )
            st.caption(
                f"Batas maksimum pemberian kredit grup "
                f"{miliar(ringkas_grup['batas_bmpk_rp'], 0)}, posisi per "
                f"{pd.Timestamp(ringkas_grup['snapshot_date']).strftime('%d %b %Y')}."
            )

    # ---- daftar relasi
    st.divider()
    judul_bagian("Daftar relasi pada subgraf")
    tabel = edges.copy()
    # Bobot beda satuan per tipe relasi: rupiah pada `memasok`, porsi pada
    # `memiliki`, penanda 1,0 pada sisanya. Memformat semuanya sebagai rupiah
    # membuat porsi 0,36 tampil sebagai "Rp 0".
    def _bobot(baris: pd.Series) -> str:
        rel, nilai = baris["relasi"], float(baris["bobot"])
        if rel == "memasok":
            return rupiah(nilai, singkat=True)
        if rel == "memiliki":
            return persen(nilai, 1)
        return "—"

    tabel["Bobot"] = tabel.apply(_bobot, axis=1)
    tabel["Berlaku dari"] = pd.to_datetime(tabel["valid_from"]).dt.strftime("%b %Y")
    tabel["Sampai"] = pd.to_datetime(tabel["valid_to"]).dt.strftime("%b %Y").fillna("—")
    st.dataframe(
        tabel[["source", "target", "relasi", "Bobot", "Berlaku dari", "Sampai"]]
        .rename(columns={"source": "Dari", "target": "Ke", "relasi": "Tipe relasi"})
        .sort_values("Tipe relasi"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Bobot `memasok` adalah nilai transfer, `memiliki` adalah porsi kepemilikan, "
        "sisanya penanda tanpa satuan."
    )

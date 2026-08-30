"""Rujukan kebijakan dan daftar dokumen kurang untuk satu pengajuan.

Modul ini menggantikan dua fungsi terakhir `lib/dummy_data.py` yang masih ikut
masuk ke draft credit memo: `kutipan_kebijakan()` dan `dokumen_kurang()`.

Bedanya bukan sekadar sumber data. Yang lama mengembalikan pasal bernomor
"KKK-08.2" lengkap dengan skor kemiripan 0,89 — seluruhnya ditulis tangan, tanpa
satu pun berkas di belakangnya. Memo yang memuatnya terbaca seperti hasil
penelusuran korpus, dan itulah yang berbahaya: dokumen yang paling terlihat
resmi justru bagian yang paling tidak bisa dipertanggungjawabkan di depan
komite.

Di sini pasal hanya boleh datang dari `copilot.rag.pencarian.kutipan()`, yaitu
korpus pada `docs/policies` yang sudah diindeks. Ketika index belum dibangun,
Ollama mati, atau sebuah topik memang tidak ada padanannya di korpus, hasilnya
adalah daftar kosong beserta sebabnya — bukan pasal pengganti. Memo lalu
mencetak sebab itu apa adanya, sehingga "belum ditelusuri" tidak pernah menyamar
menjadi "sudah sesuai kebijakan".
"""
from __future__ import annotations

from lib import copilot_lokal as ck

# --------------------------------------------------------------------------
# Topik yang ditelusuri untuk tiap pengajuan
# --------------------------------------------------------------------------
# Tiap topik adalah satu kueri ke korpus, bukan satu pasal yang sudah diketahui
# jawabannya. Kalimatnya sengaja ditulis seperti pertanyaan analis supaya
# peringkat hibrida (kosinus + kecocokan istilah) punya cukup kata kunci.
TOPIK_WAJIB = [
    ("Kualitas kredit dan penggolongan kolektibilitas",
     "penetapan kualitas kredit debitur berdasarkan prospek usaha, kinerja "
     "keuangan, dan kemampuan membayar"),
    ("Penilaian agunan sebagai pengurang penyisihan",
     "syarat agunan yang dapat diperhitungkan sebagai pengurang dalam "
     "pembentukan penyisihan penghapusan aset, penilaian dan pengikatannya"),
    ("Eksposur satu debitur dan pihak terkait",
     "penetapan kualitas aset untuk debitur dengan beberapa fasilitas dan "
     "penyediaan dana kepada pihak terkait dalam satu grup"),
]

# Topik tambahan yang hanya ditelusuri bila kasusnya memang memicunya. Kunci
# pertama adalah nama topik, kedua kueri, ketiga uji atas entitas pengajuan.
TOPIK_BERSYARAT = [
    ("Pihak terafiliasi dan penelusuran pengendali",
     "hubungan keterkaitan kepemilikan dan kepengurusan antar debitur, "
     "pengendalian oleh pihak yang sama",
     lambda e: bool(e.get("indikasi_rangkap_jabatan"))
     or int(e.get("jumlah_entitas_grup", 1)) >= 3),
    ("Konsentrasi pendapatan pada satu pembeli",
     "prospek usaha debitur yang bergantung pada sedikit pelanggan atau "
     "konsentrasi pasar",
     lambda e: bool(e.get("indikasi_konsentrasi_pembeli"))),
    ("Ketergantungan pada pemasok dan risiko kurs",
     "kemampuan membayar debitur yang terpengaruh perubahan nilai tukar dan "
     "ketergantungan pemasok",
     lambda e: bool(e.get("indikasi_konsentrasi_pemasok"))),
    ("Restrukturisasi dan penyelamatan kredit",
     "kualitas kredit yang direstrukturisasi dan syarat perbaikan "
     "penggolongannya",
     lambda e: bool(e.get("riwayat_restrukturisasi"))),
]


def _sebab_tak_tersedia() -> str | None:
    """Alasan korpus tidak bisa ditelusuri, atau None kalau ia siap."""
    if not ck.TERSEDIA:
        return (
            "lapisan copilot tidak terpasang pada lingkungan ini "
            f"({ck.GALAT_IMPOR})"
        )
    if not ck.index_kebijakan_tersedia():
        return (
            "index korpus kebijakan belum dibangun — jalankan "
            "`python -m copilot.rag.indeks`"
        )
    return None


def rujukan_pengajuan(
    entitas: dict, *, top_k: int = 2
) -> tuple[list[dict], list[str]]:
    """Pasal yang relevan bagi satu pengajuan, beserta batas penelusurannya.

    Mengembalikan `(rujukan, catatan)`. `rujukan` memakai bentuk yang sama
    dengan `copilot.rag.pencarian.kutipan()` — kunci `pasal`, `isi`, `skor`,
    `versi`, `halaman` — ditambah `topik`, yaitu pertanyaan yang membuat pasal
    itu terambil. Tanpa `topik`, komite membaca daftar pasal tanpa tahu kenapa
    masing-masing ada di sana.

    `catatan` berisi sebab kenapa penelusuran tidak lengkap: korpus tidak siap,
    atau sebuah topik tidak punya padanan. Keduanya harus sampai ke memo;
    daftar pendek tanpa keterangan terbaca sebagai "hanya ini yang berlaku".
    """
    sebab = _sebab_tak_tersedia()
    if sebab:
        return [], [f"Korpus kebijakan tidak ditelusuri: {sebab}."]

    topik = [(nama, kueri) for nama, kueri in TOPIK_WAJIB]
    topik += [
        (nama, kueri) for nama, kueri, uji in TOPIK_BERSYARAT if uji(entitas)
    ]

    # Satu pasal bisa menjawab beberapa topik sekaligus. Dikumpulkan menurut
    # rujukan supaya memo tidak mencetak isi yang sama dua kali, sementara
    # seluruh topik yang membuatnya terambil tetap tercatat.
    terkumpul: dict[str, dict] = {}
    catatan: list[str] = []
    for nama, kueri in topik:
        try:
            hasil = ck.cari_kebijakan(kueri, top_k=top_k)
        except Exception as exc:  # Ollama mati di tengah sesi
            catatan.append(f"Topik “{nama}” gagal ditelusuri: {exc}.")
            continue
        if not hasil:
            catatan.append(
                f"Topik “{nama}” tidak punya padanan pada korpus — rujukannya "
                "harus dicari analis di luar sistem."
            )
            continue
        for p in hasil:
            simpul = terkumpul.setdefault(p["pasal"], {**p, "topik": []})
            simpul["topik"].append(nama)
            simpul["skor"] = max(simpul["skor"], p["skor"])

    rujukan = sorted(terkumpul.values(), key=lambda p: p["skor"], reverse=True)
    return rujukan, catatan


# --------------------------------------------------------------------------
# Rujukan untuk gerbang kepatuhan
# --------------------------------------------------------------------------
# Aturan yang diuji `mock_engine.check_credit_policy()` adalah kebijakan kredit
# internal: ambang segmen, matriks kewenangan, batas BMPK, pertanggungan agunan,
# covenant per rating. Yang dicari di sini bukan "pasal yang mirip", melainkan
# pasal yang memang mendasari aturan itu — dan untuk sebagian aspek, peraturan
# yang mendasarinya tidak ada di korpus repo (yang ada baru POJK 40/2019).
#
# Pemetaannya ditulis tangan, bukan diserahkan ke peringkat kemiripan. Sekali
# dicoba dengan kueri bebas, aspek "Batas segmen" mendapat Pasal 31 dengan skor
# 0,82 sementara Pasal 47 yang benar-benar mengatur agunan hanya 0,69: skor
# kemiripan tidak bisa membedakan "mendasari" dari "kebetulan sebidang". Yang
# dikerjakan korpus adalah mengambil teks pasal yang sudah ditunjuk — bukan
# menebak pasal mana yang berlaku.
#
# Nilai None berarti aturannya memang internal atau diatur peraturan di luar
# korpus; `peraturan_luar` menyebut peraturan mana, tanpa mengarang kutipannya.
RUJUKAN_GERBANG = {
    "Batas segmen": {"pasal": (), "peraturan_luar": None},
    "Kewenangan": {"pasal": (), "peraturan_luar": None},
    "BMPK grup": {
        "pasal": (),
        "peraturan_luar": "POJK 32/POJK.03/2018 tentang BMPK dan penyediaan dana besar",
    },
    "Agunan": {
        # Pasal 45 syarat agunan pengurang PPKA, 46 pengikatan dan dokumen,
        # 47 siapa yang boleh menilai. Ketiganya dasar langsung ambang
        # pertanggungan yang diuji gerbang.
        "pasal": ("Pasal 45", "Pasal 46", "Pasal 47"),
        "peraturan_luar": None,
    },
    "Covenant": {
        # Kinerja dan kemampuan membayar sebagai faktor kualitas kredit; ambang
        # DER/ICR/DSCR-nya sendiri tetap kebijakan internal.
        "pasal": ("Pasal 10", "Pasal 11"),
        "peraturan_luar": None,
    },
    "Afiliasi": {
        # Kualitas yang sama untuk debitur yang sama pada beberapa fasilitas,
        # dan syarat pengecualiannya - dasar penggabungan pihak terafiliasi.
        "pasal": ("Pasal 6", "Pasal 7"),
        "peraturan_luar": "POJK 12/POJK.01/2017 tentang APU-PPT untuk penelusuran "
                          "pemilik manfaat",
    },
}

# Pasal yang ditunjuk tetap perlu diambil dari korpus, dan pengambilan itu satu
# panggilan embedding. Halaman what-if menghitung ulang gerbang pada tiap
# geseran slider, jadi hasilnya disinggahi — kueri tiap aspek tetap, tidak
# bergantung pada pengajuan.
_singgahan_gerbang: dict[str, list[dict]] = {}


def _potongan_pasal(nama: str) -> dict | None:
    """Ambil satu pasal dari korpus menurut nomornya, bukan menurut kemiripan."""
    try:
        hasil = ck.cari_kebijakan(nama, top_k=3)
    except Exception:
        return None
    # Index memberi jalur khusus untuk kueri yang menyebut nomor pasal, tetapi
    # tetangganya ikut terbawa. Hanya potongan yang rujukannya benar-benar
    # pasal itu yang dipakai.
    for p in hasil:
        if f"· {nama} ·" in str(p.get("pasal", "")):
            return p
    return None


def rujukan_aspek(aspek: str) -> list[dict]:
    """Pasal korpus yang mendasari satu aspek gerbang; kosong bila tidak ada."""
    if aspek in _singgahan_gerbang:
        return _singgahan_gerbang[aspek]
    peta = RUJUKAN_GERBANG.get(aspek)
    if not peta or not peta["pasal"]:
        return []
    if _sebab_tak_tersedia():
        # Korpus tidak siap tidak disinggahi: index bisa saja dibangun di tengah
        # sesi, dan jawaban "tidak ada" tidak boleh ikut awet.
        return []
    ambil = [p for p in (_potongan_pasal(n) for n in peta["pasal"]) if p]
    _singgahan_gerbang[aspek] = ambil
    return ambil


def lampirkan_rujukan(gerbang: list[dict]) -> list[dict]:
    """Isi `pasal` dan `kutipan` tiap butir gerbang dari korpus terindeks.

    Butir tanpa dasar di korpus tetap lewat dengan `pasal` kosong dan `sumber`
    "internal". Aturannya tetap berlaku — ia kebijakan internal bank — tetapi
    layar tidak boleh menyiratkan ada pasal yang mendasarinya.
    """
    for a in gerbang:
        aspek = a.get("aspek", "")
        peta = RUJUKAN_GERBANG.get(aspek) or {}
        a["peraturan_luar"] = peta.get("peraturan_luar")
        rujukan = rujukan_aspek(aspek)
        if not rujukan:
            a["pasal"] = None
            a["kutipan"] = None
            a["sumber"] = "internal"
            continue
        # Ketiganya dari berkas yang sama; nama berkasnya ditulis sekali supaya
        # sel tabel memo tetap terbaca.
        berkas = str(rujukan[0].get("versi") or "").strip()
        nomor = [str(n) for n in peta["pasal"]
                 if any(f"· {n} ·" in str(p["pasal"]) for p in rujukan)]
        a["pasal"] = f"{berkas} · {', '.join(nomor)}" if berkas else ", ".join(nomor)
        a["kutipan"] = rujukan[0]["isi"]
        a["sumber"] = "korpus"
    return gerbang


def label_dasar(a: dict) -> str:
    """Satu baris untuk kolom dasar pada tabel gerbang."""
    if a.get("sumber") == "korpus" and a.get("pasal"):
        return str(a["pasal"])
    if a.get("peraturan_luar"):
        return f"kebijakan internal · diatur {a['peraturan_luar']} (di luar korpus)"
    return "kebijakan internal — tidak ada padanan pada korpus"


def jejak_sumber(p: dict) -> str:
    """Asal satu kutipan: berkas, halaman, kemiripan — tanpa mengulang rujukan.

    `pasal` yang datang dari index sudah memuat nama berkas dan halaman pada
    sebagian korpus. Mengulangnya membuat baris memo panjang tanpa menambah apa
    pun yang bisa dibuka analis.
    """
    pasal = str(p.get("pasal", ""))
    bagian = []
    versi = str(p.get("versi") or "")
    if versi and versi not in pasal:
        bagian.append(versi)
    halaman = p.get("halaman") or []
    if halaman and "hal." not in pasal:
        bagian.append("hlm " + ", ".join(str(h) for h in halaman))
    bagian.append(f"kemiripan {p.get('skor', 0):.3f}")
    return " · ".join(bagian)


# --------------------------------------------------------------------------
# Dokumen yang masih kurang
# --------------------------------------------------------------------------
# Daftar ini aturan, bukan hasil pencarian: isinya turun dari struktur agunan
# dan struktur grup pengajuan yang bersangkutan. Karena itu ia boleh disusun
# sistem — yang tidak boleh adalah menandai berkas sebagai kurang padahal ia
# baru saja diunggah, dan itulah gunanya `dokumen`.
DOKUMEN_DASAR = [
    ("laporan_keuangan",
     "Laporan keuangan audited dua tahun terakhir beserta catatannya"),
    (None, "Proyeksi arus kas selama tenor fasilitas"),
    ("rekening_koran", "Rekening koran bank utama 6 bulan terakhir"),
]


def dokumen_kurang(entitas: dict, dokumen=None) -> list[str]:
    """Ceklis dokumen yang belum ada, disesuaikan dengan berkas yang diunggah.

    `dokumen` adalah `pipeline_copilot.HasilDokumen` bila copilot dijalankan
    atas unggahan. Jenis berkas yang sudah masuk dicoret dari ceklis supaya
    memo tidak meminta ulang rekening koran yang barusan dibaca.
    """
    ada = dokumen.kelengkapan() if dokumen is not None else {}
    kurang = [teks for jenis, teks in DOKUMEN_DASAR if not ada.get(jenis)]

    agunan = str(entitas.get("jenis_agunan", ""))
    if "Tanah dan bangunan" in agunan:
        kurang.append(
            "Laporan penilaian agunan (KJPP) terbaru dan bukti pengikatan hak tanggungan")
    if "Mesin" in agunan:
        kurang.append("Daftar mesin, invoice pembelian, dan bukti pengikatan fidusia")
    if "Persediaan" in agunan or "Piutang" in agunan:
        kurang.append("Aging piutang dan daftar persediaan per akhir bulan terakhir")

    if entitas.get("indikasi_rangkap_jabatan") or int(entitas.get("jumlah_entitas_grup", 1)) >= 3:
        if not ada.get("akta"):
            kurang.append(
                "Struktur kepemilikan grup sampai pemilik manfaat akhir beserta akta pendukung")
        kurang.append("Daftar fasilitas aktif seluruh entitas satu grup pada bank lain")
    if entitas.get("indikasi_konsentrasi_pembeli"):
        kurang.append("Salinan kontrak atau purchase order dari pembeli utama")

    kurang.append("Risalah kunjungan relationship manager yang ditandatangani")
    return kurang

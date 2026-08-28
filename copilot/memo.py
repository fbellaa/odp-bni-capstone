"""Penyusunan draft credit memo dari dokumen, kebijakan, dan hasil tool.

Aturan yang dipegang - sama dengan `app/ui/lib/memo.py`: memo hanya merangkai
angka yang keluar dari tool. Tidak ada angka baru yang lahir di lapisan ini,
dan tidak ada angka yang datang dari model bahasa.

Pembagiannya:

    konteks_pengajuan()   fakta dokumen -> teks yang dibaca agen
    susun_draft()         kerangka memo + narasi (boleh dari LLM)
    bagian_perhitungan()  tabel hasil tool, lengkap dengan rumusnya

Narasi yang ditulis model ditempatkan pada bagian yang jelas batasnya dan tidak
pernah menggantikan tabel angka - supaya pembaca memo tahu persis bagian mana
yang dihasilkan model dan bagian mana yang dihitung.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from copilot.agen.perhitungan import HasilAgen
from copilot.dokumen.jembatan import ringkas_untuk_memo
from copilot.dokumen.skema import BerkasPengajuan
from copilot.llm.klien import GalatOllama, KlienOllama, klien

PERINTAH_NARASI = """\
Kamu analis kredit komersial yang menulis bagian naratif sebuah credit memo.

Aturan:
- Tulis 2 sampai 4 paragraf dalam bahasa Indonesia formal perbankan.
- JANGAN menyebut angka apa pun yang tidak ada pada data yang diberikan.
- JANGAN menghitung rasio atau menyimpulkan angka baru.
- Bahas: profil usaha dan kepengurusan, kualitas informasi dokumen, serta hal
  yang perlu diperhatikan analis. Jangan menyatakan keputusan setuju atau tolak.
- Jangan memakai daftar berpoin; tulis sebagai paragraf.
"""


BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def tanggal_indonesia(d: date) -> str:
    """Format tanggal tanpa bergantung locale sistem.

    `%B` mengikuti locale proses, yang di server demo hampir selalu bahasa
    Inggris - satu-satunya kata Inggris di memo berbahasa Indonesia.
    """
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def rupiah(nilai: float | None) -> str:
    if nilai is None:
        return "-"
    return f"Rp {nilai:,.0f}".replace(",", ".")


def kali(nilai: float | None) -> str:
    return "-" if nilai is None else f"{nilai:,.2f}x".replace(".", ",")


def persen(nilai: float | None) -> str:
    return "-" if nilai is None else f"{nilai * 100:,.2f}%".replace(".", ",")


# ------------------------------------------------------------------ konteks
def konteks_pengajuan(
    berkas: BerkasPengajuan,
    pengajuan: dict[str, Any] | None = None,
    afiliasi: dict[str, Any] | None = None,
) -> str:
    """Fakta pengajuan dalam bentuk teks datar untuk dibaca agen.

    Sengaja datar dan berlabel eksplisit. Model kecil jauh lebih jarang salah
    memasangkan angka ke argumen tool bila tiap angka datang dengan namanya
    sendiri di baris terpisah, dibanding bila disajikan sebagai JSON bersarang.
    """
    pengajuan = pengajuan or {}
    baris: list[str] = []

    baris.append(f"Nama debitur: {berkas.nama_debitur or 'tidak tercantum'}")

    akta = berkas.akta_utama
    if akta:
        baris.append(f"Alamat operasional: {akta.alamat_operasional or 'tidak tercantum'}")
        if akta.pengurus:
            pengurus = "; ".join(
                f"{p.nama} ({p.jabatan or 'jabatan tidak tercantum'})" for p in akta.pengurus
            )
            baris.append(f"Pengurus: {pengurus}")
        if akta.pemegang_saham:
            saham = "; ".join(
                f"{s.nama} {s.persentase}%" if s.persentase is not None else s.nama
                for s in akta.pemegang_saham
            )
            baris.append(f"Pemegang saham: {saham}")

    lapkeu = berkas.lapkeu_terbaru
    if lapkeu:
        baris.append(f"Periode laporan keuangan: {lapkeu.periode or 'tidak tercantum'}")
        for label, nilai in (
            ("Penjualan", lapkeu.penjualan),
            ("EBITDA", lapkeu.ebitda),
            ("Laba bersih", lapkeu.laba_bersih),
            ("Beban bunga", lapkeu.beban_bunga),
            ("Total aset", lapkeu.total_aset),
            ("Total liabilitas", lapkeu.total_liabilitas),
            ("Utang berbunga", lapkeu.utang_berbunga),
            ("Ekuitas", lapkeu.ekuitas),
            ("Kas dan setara kas", lapkeu.kas_dan_setara),
            ("Arus kas operasi", lapkeu.arus_kas_operasi),
        ):
            if nilai is not None:
                baris.append(f"{label}: {nilai:.0f}")
    else:
        baris.append("Laporan keuangan: tidak tersedia pada berkas ini.")

    for i, rk in enumerate(berkas.semua_rekening_koran, start=1):
        awal = f"{rk.periode_awal}" if rk.periode_awal else "?"
        akhir = f"{rk.periode_akhir}" if rk.periode_akhir else "?"
        baris.append(
            f"Rekening koran {i}: {rk.nomor_rekening or 'nomor tidak tercantum'} "
            f"({rk.bank or 'bank tidak tercantum'}), periode {awal} s.d. {akhir}, "
            f"{len(rk.mutasi)} baris mutasi, total kredit {rk.total_kredit:.0f}, "
            f"total debit {rk.total_debit:.0f}"
        )
        if rk.saldo_rata_rata is not None:
            baris.append(f"Saldo rata-rata rekening {i}: {rk.saldo_rata_rata:.0f}")

    for kunci, label in (
        ("plafon", "Plafon diajukan"),
        ("tenor_bulan", "Tenor (bulan)"),
        ("jenis_fasilitas", "Jenis fasilitas"),
        ("jenis_agunan", "Jenis agunan"),
        ("nilai_agunan", "Nilai agunan"),
        ("bunga_tahunan", "Indikasi suku bunga tahunan (pecahan)"),
        ("kewajiban_tahunan_eksisting", "Kewajiban tahunan fasilitas eksisting"),
        ("eksposur_grup_berjalan", "Eksposur grup berjalan"),
        ("pd_12bulan", "PD 12 bulan (pecahan)"),
    ):
        nilai = pengajuan.get(kunci)
        if nilai is not None:
            baris.append(f"{label}: {nilai}")

    if afiliasi:
        baris.append(f"Hasil penelusuran afiliasi: {ringkas_untuk_memo(afiliasi)}")

    return "\n".join(baris)


# ------------------------------------------------------------------- narasi
def tulis_narasi(
    berkas: BerkasPengajuan,
    konteks: str,
    *,
    kl: KlienOllama | None = None,
) -> str:
    """Paragraf naratif dari model bahasa. Gagal dengan anggun, bukan gagal total."""
    kl = kl or klien()
    try:
        balasan = kl.chat(
            [
                {"role": "system", "content": PERINTAH_NARASI},
                {"role": "user", "content": f"Data pengajuan:\n{konteks}"},
            ],
            peran="chat",
        )
        return (balasan.get("content") or "").strip()
    except GalatOllama as exc:
        return (
            f"_Bagian naratif tidak dapat disusun pada sesi ini ({exc}). "
            "Isi manual sebelum memo diteruskan ke komite._"
        )


# ------------------------------------------------------------- perhitungan
# Angka mana dari hasil tool mana yang naik ke tabel ringkas memo.
SOROTAN: list[tuple[str, str, str, str]] = [
    ("hitung_rasio_keuangan", "der", "DER", "kali"),
    ("hitung_rasio_keuangan", "interest_coverage", "Interest coverage", "kali"),
    ("hitung_rasio_keuangan", "ebitda_margin", "Marjin EBITDA", "persen"),
    ("hitung_angsuran", "kewajiban_tahunan", "Kewajiban tahunan fasilitas", "rupiah"),
    ("hitung_dscr", "dscr", "DSCR", "kali"),
    ("estimasi_lgd", "lgd", "LGD", "persen"),
    ("estimasi_lgd", "coverage", "Tingkat pertanggungan agunan", "kali"),
    ("grade_dari_pd", "grade", "Rating internal", "teks"),
    ("hitung_expected_loss", "expected_loss", "Expected loss", "rupiah"),
    ("usulkan_pricing", "pricing", "Pricing usulan", "persen"),
    ("periksa_bmpk", "sisa_ruang", "Sisa ruang BMPK grup", "rupiah"),
    ("kewenangan_komite", "komite_pemutus", "Kewenangan pemutus", "teks"),
]

_FORMAT = {"rupiah": rupiah, "kali": kali, "persen": persen, "teks": lambda v: str(v)}


def bagian_perhitungan(hasil: HasilAgen) -> str:
    """Tabel angka + jejak tool. Inti pertanggungjawaban memo ini."""
    angka = hasil.angka

    sorotan = [
        f"| {label} | {_FORMAT[bentuk](angka[tool][kunci])} | `{tool}` |"
        for tool, kunci, label, bentuk in SOROTAN
        if tool in angka and angka[tool].get(kunci) is not None
    ]
    tabel_sorotan = (
        "| Besaran | Nilai | Dihitung oleh |\n| --- | --- | --- |\n" + "\n".join(sorotan)
        if sorotan
        else "_Belum ada angka yang berhasil dihitung._"
    )

    jejak = "\n".join(
        f"{i}. {'✔' if j.berhasil else '✘'} `{j.nama}` — {j.ringkas().split('-> ', 1)[-1]}"
        for i, j in enumerate(hasil.rekaman.jejak, start=1)
    ) or "_Tidak ada tool yang dipanggil._"

    gerbang = _tabel_gerbang(angka)

    catatan_gagal = ""
    if hasil.ada_kegagalan:
        daftar = "\n".join(f"- `{j.nama}`: {j.galat}" for j in hasil.rekaman.gagal())
        catatan_gagal = (
            "\n**Perhitungan yang gagal dan wajib dikerjakan manual**\n\n" + daftar + "\n"
        )

    return f"""### Angka hasil perhitungan

{tabel_sorotan}

{gerbang}
{catatan_gagal}
<details>
<summary>Jejak pemanggilan tool ({len(hasil.rekaman.jejak)} panggilan, {hasil.putaran} putaran, berhenti: {hasil.berhenti_karena})</summary>

{jejak}

</details>

### Ringkasan agen

{hasil.ringkasan or '_Agen tidak menuliskan ringkasan._'}
"""


def _tabel_gerbang(angka: dict[str, dict[str, Any]]) -> str:
    """Gerbang kepatuhan dari hasil tool yang relevan."""
    baris = []

    if "periksa_batas_segmen" in angka:
        s = angka["periksa_batas_segmen"]
        baris.append(
            ("Batas segmen", s["lolos"], "Seluruh dimensi di dalam batas"
             if s["lolos"] else f"Di luar batas: {', '.join(s['di_luar_batas'])}")
        )
    if "periksa_bmpk" in angka:
        b = angka["periksa_bmpk"]
        baris.append(
            ("BMPK grup debitur", b["lolos"],
             f"Sisa ruang {rupiah(b['sisa_ruang'])} setelah pencairan")
        )
    if "periksa_covenant" in angka:
        c = angka["periksa_covenant"]
        baris.append(
            ("Covenant keuangan", c["lolos"],
             f"Uji {c['frekuensi_uji']}; " + ("seluruh butir terpenuhi" if c["lolos"]
              else f"dilanggar: {', '.join(c['dilanggar'])}"))
        )
    if "periksa_agunan" in angka:
        a = angka["periksa_agunan"]
        baris.append(
            ("Agunan", a["lolos"],
             f"Coverage {kali(a['coverage'])} terhadap minimum {kali(a['coverage_minimum'])}")
        )
    if "hitung_dscr" in angka:
        d = angka["hitung_dscr"]
        baris.append(
            ("Kapasitas arus kas", d["lolos"],
             f"DSCR {kali(d['dscr'])} terhadap ambang {kali(d['ambang_kebijakan'])}")
        )

    if not baris:
        return "_Gerbang kepatuhan belum dijalankan._"

    isi = "\n".join(
        f"| {aspek} | {'LOLOS' if lolos else 'PERLU PENYESUAIAN'} | {temuan} |"
        for aspek, lolos, temuan in baris
    )
    return "**Gerbang kepatuhan**\n\n| Aspek | Status | Temuan |\n| --- | --- | --- |\n" + isi


# --------------------------------------------------------------------- memo
def susun_draft(
    nomor_pengajuan: str,
    berkas: BerkasPengajuan,
    *,
    hasil_agen: HasilAgen | None = None,
    afiliasi: dict[str, Any] | None = None,
    kebijakan: list[dict[str, Any]] | None = None,
    narasi: str | None = None,
) -> str:
    """Rangkai draft credit memo lengkap dalam Markdown."""
    kebijakan = kebijakan or []
    akta = berkas.akta_utama
    lapkeu = berkas.lapkeu_terbaru
    kelengkapan = berkas.kelengkapan()

    baris_dokumen = "\n".join(
        f"| {d.sumber.berkas} | {d.jenis} | {len(d.sumber.halaman)} dari {d.sumber.jumlah_halaman} | "
        f"{'; '.join(d.catatan) or '-'} |"
        for d in berkas.dokumen
    ) or "| - | - | - | Tidak ada dokumen yang dibaca |"

    baris_kurang = "\n".join(
        f"- [ ] {nama.replace('_', ' ').capitalize()}"
        for nama, ada in kelengkapan.items()
        if not ada
    ) or "- Seluruh jenis dokumen wajib sudah tersedia."

    baris_pengurus = "\n".join(
        f"- {p.nama} — {p.jabatan or 'jabatan tidak tercantum'}"
        for p in (akta.pengurus if akta else [])
    ) or "- Susunan pengurus tidak terbaca dari dokumen."

    baris_saham = "\n".join(
        f"- {s.nama}" + (f" — {s.persentase}%" if s.persentase is not None else "")
        for s in (akta.pemegang_saham if akta else [])
    ) or "- Susunan pemegang saham tidak terbaca dari dokumen."

    baris_lapkeu = "\n".join(
        f"| {label} | {rupiah(nilai)} |"
        for label, nilai in (
            ("Penjualan", lapkeu.penjualan if lapkeu else None),
            ("EBITDA", lapkeu.ebitda if lapkeu else None),
            ("Laba bersih", lapkeu.laba_bersih if lapkeu else None),
            ("Beban bunga", lapkeu.beban_bunga if lapkeu else None),
            ("Utang berbunga", lapkeu.utang_berbunga if lapkeu else None),
            ("Ekuitas", lapkeu.ekuitas if lapkeu else None),
            ("Total aset", lapkeu.total_aset if lapkeu else None),
        )
        if nilai is not None
    ) or "| - | Laporan keuangan tidak terbaca |"

    baris_kebijakan = "\n".join(
        f"- **{k['pasal']}** (kemiripan {k['skor']}) — {k['isi']}" for k in kebijakan
    ) or "- Index kebijakan belum dibangun; sitasi wajib dilengkapi manual."

    bagian_hitung = (
        bagian_perhitungan(hasil_agen)
        if hasil_agen
        else "_Perhitungan belum dijalankan._"
    )

    return f"""# DRAFT CREDIT MEMO — SEGMEN KOMERSIAL

**Nomor pengajuan:** {nomor_pengajuan}
**Tanggal disusun:** {tanggal_indonesia(date.today())}
**Debitur:** {berkas.nama_debitur or '(nama tidak terbaca dari dokumen)'}
**NPWP:** {(akta.npwp if akta else None) or '-'}
**Alamat operasional:** {(akta.alamat_operasional if akta else None) or '-'}
**Disusun oleh:** Copilot kredit komersial — LLM lokal + agen tool calling

> **Draft, bukan keputusan.** Angka pada memo ini berasal dari tool perhitungan
> deterministik; narasi disusun model bahasa lokal. Seluruh isinya wajib
> ditelaah analis kredit sebelum diteruskan ke komite. Model PD/LGD/EWS belum
> dilatih, sehingga PD yang dipakai berasal dari masukan analis, bukan model.

---

## 1. Dokumen yang dibaca

| Berkas | Jenis | Halaman terbaca | Catatan |
| --- | --- | --- | --- |
{baris_dokumen}

**Dokumen yang masih kurang**

{baris_kurang}

## 2. Profil dan kepengurusan

{baris_pengurus}

**Pemegang saham**

{baris_saham}

## 3. Posisi keuangan (angka dokumen, belum diolah)

| Pos | Nilai |
| --- | --- |
{baris_lapkeu}

## 4. Perhitungan dan gerbang kepatuhan

{bagian_hitung}

## 5. Penelusuran afiliasi dan grup debitur

{ringkas_untuk_memo(afiliasi) if afiliasi else '_Penelusuran afiliasi belum dijalankan._'}

## 6. Kebijakan yang dirujuk

{baris_kebijakan}

## 7. Catatan analis

{narasi or '_Bagian naratif belum disusun._'}

---

_Dokumen ini dihasilkan otomatis. Tanda tangan analis dan pemutus wajib
dibubuhkan pada versi final._
"""

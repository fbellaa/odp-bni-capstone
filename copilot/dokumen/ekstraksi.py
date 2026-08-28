"""PDF -> struktur, dengan model bahasa lokal.

Satu dokumen dibaca per kelompok halaman, lalu hasilnya digabung. Penggabungan
dilakukan di Python, bukan dengan meminta model meringkas ulang: menambah satu
lapisan model di atas keluaran model hanya menambah kesempatan angka berubah
tanpa jejak.

Prinsip yang dipegang di seluruh modul ini: **model hanya boleh menyalin, tidak
menghitung.** Rasio, saldo rata-rata, dan seluruh turunan angka dikerjakan
`copilot.alat.keuangan` secara deterministik. Yang diminta dari model hanyalah
memindahkan angka dari halaman PDF ke field yang benar.
"""

from __future__ import annotations

import logging
from pathlib import Path

from copilot.dokumen import pdf as baca_pdf
from copilot.dokumen.skema import (
    Akta,
    BerkasPengajuan,
    DokumenTerstruktur,
    JenisDokumen,
    LaporanKeuangan,
    RekeningKoran,
)
from copilot.llm.klien import GalatOllama, KlienOllama, klien

LOG = logging.getLogger(__name__)

ATURAN_UMUM = """\
Kamu adalah pembaca dokumen kredit di sebuah bank komersial Indonesia.

Aturan yang tidak boleh dilanggar:
1. Salin angka apa adanya dari dokumen. JANGAN menghitung, menjumlah, merata-rata,
   atau menyimpulkan angka yang tidak tertulis.
2. Bila sebuah field tidak ada di dokumen, isi null. Jangan menebak.
3. Angka dikirim sebagai bilangan tanpa pemisah ribuan dan tanpa simbol mata uang.
   "Rp 1.250.000.000" menjadi 1250000000. "(1.500)" berarti negatif: -1500.
4. Bila dokumen memakai satuan ("dalam jutaan Rupiah", "dalam ribuan"), kalikan
   HANYA untuk mengembalikan angka ke satuan Rupiah penuh, lalu sebutkan hal itu
   pada catatan.
5. Tanggal ditulis YYYY-MM-DD.
6. Jawab HANYA dengan objek JSON sesuai skema. Tanpa penjelasan, tanpa markdown.
"""

PERINTAH: dict[JenisDokumen, str] = {
    "rekening_koran": """\
Dokumen ini rekening koran / mutasi rekening.

Ambil identitas rekening dan SELURUH baris mutasi yang terlihat pada potongan ini.

Untuk tiap baris mutasi:
- `debit` diisi bila dana keluar, `kredit` bila dana masuk. Yang tidak terpakai diisi 0.
- `rekening_lawan` diisi HANYA bila nomor rekening pihak lawan benar-benar tertulis
  pada keterangan transaksi. Jangan mengarang nomor.
- `nama_lawan` diisi nama pihak lawan bila tertulis.

`saldo_rata_rata` diisi hanya bila dokumen mencantumkannya secara eksplisit.
Jangan menghitungnya sendiri.
""",
    "laporan_keuangan": """\
Dokumen ini laporan keuangan.

Ambil pos-pos untuk SATU periode terakhir yang tersedia. Bila laporan menyajikan
dua kolom tahun berdampingan, ambil kolom tahun terbaru saja dan tulis tahunnya
pada `periode`.

Catatan pos:
- `utang_berbunga` = pinjaman bank + obligasi + sewa pembiayaan, jangka pendek
  maupun panjang. JANGAN memakai total liabilitas.
- `ebitda` diisi hanya bila tertulis. Bila tidak ada, biarkan null - sistem akan
  menghitungnya sendiri dari laba usaha dan penyusutan.
- `ekuitas` adalah total ekuitas yang dapat diatribusikan ke pemilik.
""",
    "akta": """\
Dokumen ini akta pendirian / anggaran dasar / dokumen kepemilikan perusahaan.

Ambil identitas perusahaan, alamat, susunan pengurus, dan pemegang saham.

- `alamat_operasional` adalah alamat tempat usaha dijalankan. Bila hanya ada satu
  alamat, isikan alamat itu ke `alamat_operasional` dan `alamat_domisili` sekaligus.
- `pengurus` mencakup direksi DAN komisaris, masing-masing dengan jabatannya.
- `persentase` pemegang saham diisi dalam persen (25.5 berarti 25,5 persen), bukan
  pecahan.
""",
}

SKEMA_PER_JENIS: dict[JenisDokumen, type] = {
    "rekening_koran": RekeningKoran,
    "laporan_keuangan": LaporanKeuangan,
    "akta": Akta,
}


def baca_dokumen(
    path: str | Path,
    *,
    jenis: JenisDokumen | None = None,
    kl: KlienOllama | None = None,
) -> DokumenTerstruktur:
    """Baca satu PDF menjadi satu `DokumenTerstruktur`."""
    kl = kl or klien()
    path = Path(path)
    halaman = baca_pdf.baca_halaman(path)

    catatan: list[str] = []
    if jenis is None:
        jenis, skor = baca_pdf.tebak_jenis(halaman)
        catatan.append(f"Jenis ditebak otomatis: {jenis} (skor penanda {skor}).")

    sumber = baca_pdf.sumber_dari(path, halaman)

    if jenis == "tidak_dikenali":
        catatan.append(
            "Jenis dokumen tidak dikenali dari kata kunci, sehingga isinya tidak "
            "diekstraksi. Tetapkan jenisnya secara manual bila dokumen ini memang "
            "salah satu dari rekening koran, laporan keuangan, atau akta."
        )
        return DokumenTerstruktur(jenis="tidak_dikenali", sumber=sumber, catatan=catatan)

    kelompok = baca_pdf.kelompokkan(halaman)
    skema = SKEMA_PER_JENIS[jenis]
    bagian: list = []

    for i, kel in enumerate(kelompok, start=1):
        LOG.info("%s: kelompok %s/%s (halaman %s)", path.name, i, len(kelompok),
                 [h.nomor for h in kel])
        pesan = [
            {"role": "system", "content": ATURAN_UMUM + "\n" + PERINTAH[jenis]},
            {
                "role": "user",
                "content": (
                    f"Berkas: {path.name}\n"
                    f"Potongan {i} dari {len(kelompok)}.\n\n"
                    f"{baca_pdf.gabung_teks(kel)}"
                ),
            },
        ]
        try:
            bagian.append(kl.terstruktur(pesan, skema, peran="ekstraksi"))
        except GalatOllama as exc:
            # Satu potongan gagal tidak boleh membatalkan seluruh dokumen -
            # rekening koran 30 halaman kerap punya satu halaman lampiran yang
            # membingungkan model.
            LOG.warning("potongan %s pada %s gagal: %s", i, path.name, exc)
            catatan.append(f"Potongan {i} (halaman {[h.nomor for h in kel]}) gagal dibaca.")

    if not bagian:
        catatan.append("Tidak ada potongan yang berhasil diekstraksi.")
        return DokumenTerstruktur(jenis=jenis, sumber=sumber, catatan=catatan)

    hasil = DokumenTerstruktur(jenis=jenis, sumber=sumber, catatan=catatan)
    if jenis == "rekening_koran":
        hasil.rekening_koran = _gabung_rekening(bagian)
        catatan.append(f"{len(hasil.rekening_koran.mutasi)} baris mutasi terbaca.")
    elif jenis == "laporan_keuangan":
        hasil.laporan_keuangan = _gabung_lapkeu(bagian)
    else:
        hasil.akta = _gabung_akta(bagian)
    return hasil


def baca_berkas_pengajuan(
    daftar_path: list[str | Path],
    *,
    jenis_per_berkas: dict[str, JenisDokumen] | None = None,
    kl: KlienOllama | None = None,
) -> BerkasPengajuan:
    """Baca banyak PDF menjadi satu berkas pengajuan."""
    kl = kl or klien()
    jenis_per_berkas = jenis_per_berkas or {}
    dokumen = []

    for path in daftar_path:
        nama = Path(path).name
        try:
            dokumen.append(
                baca_dokumen(path, jenis=jenis_per_berkas.get(nama), kl=kl)
            )
        except baca_pdf.GalatPDF as exc:
            LOG.error("%s dilewati: %s", nama, exc)
            dokumen.append(
                DokumenTerstruktur(
                    jenis="tidak_dikenali",
                    sumber=baca_pdf.Sumber(berkas=nama),
                    catatan=[str(exc)],
                )
            )

    berkas = BerkasPengajuan(dokumen=dokumen)
    akta = berkas.akta_utama
    if akta and akta.nama_perusahaan:
        berkas.nama_debitur = akta.nama_perusahaan
    else:
        pemilik = next(
            (rk.nama_pemilik for rk in berkas.semua_rekening_koran if rk.nama_pemilik), None
        )
        berkas.nama_debitur = pemilik
    return berkas


# ------------------------------------------------------------- penggabungan
def _pertama(nilai_nilai):
    """Nilai pertama yang tidak kosong. Identitas biasanya hanya di halaman awal."""
    return next((v for v in nilai_nilai if v not in (None, "", [])), None)


def _gabung_rekening(bagian: list[RekeningKoran]) -> RekeningKoran:
    mutasi = [m for b in bagian for m in b.mutasi]
    tanggal = [b.periode_awal for b in bagian if b.periode_awal]
    akhir = [b.periode_akhir for b in bagian if b.periode_akhir]
    return RekeningKoran(
        nomor_rekening=_pertama(b.nomor_rekening for b in bagian),
        nama_pemilik=_pertama(b.nama_pemilik for b in bagian),
        bank=_pertama(b.bank for b in bagian),
        periode_awal=min(tanggal) if tanggal else None,
        periode_akhir=max(akhir) if akhir else None,
        saldo_rata_rata=_pertama(b.saldo_rata_rata for b in bagian),
        mutasi=mutasi,
    )


def _gabung_lapkeu(bagian: list[LaporanKeuangan]) -> LaporanKeuangan:
    """Isi tiap pos dari potongan pertama yang menyebutnya.

    Lapkeu tersebar: neraca di satu halaman, laba rugi di halaman lain. Pos
    yang sudah terisi tidak ditimpa potongan berikutnya, karena potongan
    belakangan biasanya catatan atas laporan - angkanya rincian, bukan total.
    """
    hasil = LaporanKeuangan()
    for b in bagian:
        for nama_field in LaporanKeuangan.model_fields:
            if getattr(hasil, nama_field, None) in (None, "IDR"):
                nilai = getattr(b, nama_field, None)
                if nilai is not None:
                    setattr(hasil, nama_field, nilai)
    return hasil


def _gabung_akta(bagian: list[Akta]) -> Akta:
    pengurus, saham = [], []
    terlihat_p, terlihat_s = set(), set()

    for b in bagian:
        for p in b.pengurus:
            kunci = p.nama.strip().lower()
            if kunci and kunci not in terlihat_p:
                terlihat_p.add(kunci)
                pengurus.append(p)
        for s in b.pemegang_saham:
            kunci = s.nama.strip().lower()
            if kunci and kunci not in terlihat_s:
                terlihat_s.add(kunci)
                saham.append(s)

    return Akta(
        nama_perusahaan=_pertama(b.nama_perusahaan for b in bagian),
        npwp=_pertama(b.npwp for b in bagian),
        nomor_akta=_pertama(b.nomor_akta for b in bagian),
        tanggal_akta=_pertama(b.tanggal_akta for b in bagian),
        alamat_operasional=_pertama(b.alamat_operasional for b in bagian),
        alamat_domisili=_pertama(b.alamat_domisili for b in bagian),
        pengurus=pengurus,
        pemegang_saham=saham,
    )

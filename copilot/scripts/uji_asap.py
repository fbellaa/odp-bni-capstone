"""Uji asap rantai copilot dari ujung ke ujung.

Dijalankan sekali sesudah `siapkan_ollama.sh`, sebelum demo. Tujuannya
menemukan masalah yang hanya muncul saat model benar-benar dipanggil - tag
model salah, index dibangun dengan embedding lain, model agen yang ternyata
tidak mendukung function calling - selagi masih ada waktu memperbaikinya.

    python -m copilot.scripts.uji_asap
"""

from __future__ import annotations

import logging
import sys

from copilot.agen.perhitungan import AgenPerhitungan
from copilot.alat.registrasi import jalankan
from copilot.konfigurasi import PENGATURAN
from copilot.llm.klien import GalatOllama, klien, ringkas_anggaran
from copilot.rag import indeks as rag_indeks
from copilot.rag import pencarian as rag_cari

LOG = logging.getLogger("uji_asap")

KONTEKS_UJI = """\
Nama debitur: PT Sumber Logam Perkasa
Periode laporan keuangan: 2025
Penjualan: 240000000000
EBITDA: 26400000000
Beban bunga: 4200000000
Utang berbunga: 45000000000
Ekuitas: 25000000000
Plafon diajukan: 80000000000
Tenor (bulan): 36
Jenis fasilitas: Investasi - term loan
Jenis agunan: Tanah dan bangunan pabrik (SHM/SHGB)
Nilai agunan: 100000000000
Indikasi suku bunga tahunan (pecahan): 0.105
Kewajiban tahunan fasilitas eksisting: 6000000000
Eksposur grup berjalan: 320000000000
PD 12 bulan (pecahan): 0.042
"""


def _lapor(nama: str, lolos: bool, keterangan: str = "") -> bool:
    print(f"[{'OK  ' if lolos else 'GAGAL'}] {nama}" + (f" - {keterangan}" if keterangan else ""))
    return lolos


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    anggaran = ringkas_anggaran()
    print(f"Profil {anggaran['profil']}, perkiraan {anggaran['total_gb']} GB")
    print(f"Model: {anggaran['model']}\n")

    hasil: list[bool] = []
    kl = klien()

    # 1. Server dan model.
    hidup = kl.hidup()
    hasil.append(_lapor("Ollama merespons", hidup, PENGATURAN.host_ollama))
    if not hidup:
        print("\nHentikan di sini: jalankan `bash copilot/scripts/siapkan_ollama.sh`.")
        return 1

    for peran in ("ekstraksi", "chat", "agen", "embedding"):
        try:
            nama = PENGATURAN.model_untuk(peran)
            kl.pastikan_model(nama)
            hasil.append(_lapor(f"Model peran {peran}", True, nama))
        except (GalatOllama, RuntimeError) as exc:
            hasil.append(_lapor(f"Model peran {peran}", False, str(exc).splitlines()[0]))

    # 2. Tool deterministik - tidak butuh model sama sekali.
    j = jalankan(
        "hitung_rasio_keuangan",
        {"penjualan": 240e9, "ebitda": 26.4e9, "utang_berbunga": 45e9, "ekuitas": 25e9},
    )
    hasil.append(_lapor("Tool perhitungan", j.berhasil and j.hasil["der"] == 1.8, j.ringkas()))

    # 3. Embedding dan index.
    try:
        vektor = kl.embed(["uji embedding"])
        hasil.append(_lapor("Embedding", bool(vektor and vektor[0]), f"dimensi {len(vektor[0])}"))
    except GalatOllama as exc:
        hasil.append(_lapor("Embedding", False, str(exc).splitlines()[0]))

    if rag_indeks.index_tersedia():
        try:
            kutipan = rag_cari.kutipan("penggolongan kualitas kredit", top_k=3)
            hasil.append(
                _lapor("Pencarian kebijakan", bool(kutipan),
                       kutipan[0]["pasal"] if kutipan else "tidak ada hasil")
            )
        except Exception as exc:
            hasil.append(_lapor("Pencarian kebijakan", False, str(exc).splitlines()[0]))
    else:
        _lapor("Index kebijakan", False, "belum dibangun; jalankan python -m copilot.rag.indeks")

    # 4. Agen tool calling. Inilah yang paling sering gagal diam-diam: model
    #    yang tidak mendukung function calling akan menjawab dengan kalimat
    #    berisi angka, bukan memanggil tool.
    print("\nMenjalankan agen (bisa memakan waktu di CPU)...")
    try:
        keluaran = AgenPerhitungan(kl).jalankan(KONTEKS_UJI)
    except Exception as exc:
        hasil.append(_lapor("Agen tool calling", False, str(exc).splitlines()[0]))
    else:
        dipanggil = len(keluaran.rekaman.jejak)
        berhasil = len([x for x in keluaran.rekaman.jejak if x.berhasil])
        hasil.append(
            _lapor(
                "Agen tool calling",
                berhasil > 0,
                f"{dipanggil} panggilan, {berhasil} berhasil, "
                f"{keluaran.putaran} putaran, berhenti: {keluaran.berhenti_karena}",
            )
        )
        for x in keluaran.rekaman.jejak:
            print(f"    {'v' if x.berhasil else 'x'} {x.ringkas()[:150]}")
        if dipanggil == 0:
            print(
                "\n    Model tidak memanggil satu tool pun. Model agen kemungkinan\n"
                "    tidak mendukung function calling; pakai qwen2.5 (3b/7b) untuk\n"
                "    peran `agen` lewat COPILOT_MODEL_AGEN."
            )

    print(f"\n{sum(hasil)}/{len(hasil)} pemeriksaan lolos.")
    return 0 if all(hasil) else 1


if __name__ == "__main__":
    sys.exit(main())

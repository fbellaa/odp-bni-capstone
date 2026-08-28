"""Agen perhitungan: putaran function calling atas tool di `copilot.alat`.

Pembagian peran di sistem ini:

    SahabatAI (model bahasa pada peran `ekstraksi`/`chat`)
        membaca PDF, menyusun narasi, menjawab pertanyaan analis

    Qwen (peran `agen`)
        memilih tool, mengisi argumennya, dan merangkai hasilnya

    copilot.alat
        seluruh angka

Qwen dipisah untuk peran ini karena tugasnya berbeda jenis: bukan menulis
kalimat yang enak dibaca, melainkan menghasilkan argumen fungsi yang bentuknya
persis benar. Model yang fasih berbahasa Indonesia belum tentu patuh pada
skema, dan sebaliknya.

Tidak ada tool model ML di sini - PD, LGD, dan EWS belum dilatih. Yang dipanggil
agen semata aritmetika kebijakan kredit. Ketika model asli siap, ia masuk
sebagai tool tambahan di `copilot.alat` tanpa mengubah putaran ini.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from copilot.alat.registrasi import JejakAlat, Rekaman, daftar_tool, jalankan
from copilot.konfigurasi import PENGATURAN
from copilot.llm.klien import GalatOllama, KlienOllama, klien

LOG = logging.getLogger(__name__)

PERINTAH_SISTEM = """\
Kamu analis kredit komersial yang bekerja HANYA lewat tool.

Aturan:
1. JANGAN pernah menghitung sendiri. Setiap angka turunan - rasio, angsuran,
   DSCR, LGD, expected loss, pricing, sisa BMPK - wajib berasal dari tool.
   Menuliskan hasil hitunganmu sendiri dianggap kesalahan fatal.
2. Pakai HANYA angka yang tertera pada data pengajuan. Bila sebuah angka tidak
   ada, jangan menebak: katakan angka itu tidak tersedia dan sebutkan tool mana
   yang jadi tidak bisa dijalankan.
3. Urutan yang wajar: rasio keuangan -> angsuran -> DSCR -> LGD -> rating ->
   expected loss -> pricing -> gerbang kepatuhan (segmen, BMPK, covenant,
   agunan, kewenangan).
4. Satu tool boleh dipanggil ulang bila argumennya salah. Bila sebuah tool
   mengembalikan galat, baca pesannya dan perbaiki argumennya.
5. Setelah seluruh tool yang relevan selesai, tulis ringkasan dalam bahasa
   Indonesia: sebutkan tiap angka beserta nama tool yang menghasilkannya.
   Jangan menambahkan angka baru di ringkasan.
"""


@dataclass
class HasilAgen:
    ringkasan: str
    rekaman: Rekaman = field(default_factory=Rekaman)
    putaran: int = 0
    berhenti_karena: str = "selesai"
    percakapan: list[dict[str, Any]] = field(default_factory=list)

    @property
    def angka(self) -> dict[str, dict[str, Any]]:
        """Hasil tool terakhir per nama - inilah sumber angka draft memo."""
        return self.rekaman.hasil_terakhir

    @property
    def ada_kegagalan(self) -> bool:
        return bool(self.rekaman.gagal())


class AgenPerhitungan:
    def __init__(self, kl: KlienOllama | None = None) -> None:
        self.kl = kl or klien()

    def jalankan(
        self,
        konteks: str,
        *,
        instruksi: str | None = None,
        maks_putaran: int | None = None,
        saat_alat: Callable[[JejakAlat], None] | None = None,
    ) -> HasilAgen:
        """Jalankan putaran tool calling sampai model berhenti memanggil tool.

        `saat_alat` dipanggil tiap satu tool selesai, dipakai antarmuka untuk
        menampilkan jejak secara bertahap alih-alih menunggu seluruh putaran.
        """
        maks_putaran = maks_putaran or PENGATURAN.maks_putaran_agen
        rekaman = Rekaman()
        pesan: list[dict[str, Any]] = [
            {"role": "system", "content": PERINTAH_SISTEM},
            {
                "role": "user",
                "content": (
                    f"{instruksi or 'Hitung seluruh angka yang dibutuhkan draft credit memo.'}\n\n"
                    f"Data pengajuan:\n{konteks}"
                ),
            },
        ]

        alat = daftar_tool()
        for putaran in range(1, maks_putaran + 1):
            try:
                balasan = self.kl.chat(pesan, peran="agen", alat=alat)
            except GalatOllama as exc:
                return HasilAgen(
                    ringkasan=f"Agen berhenti karena model tidak bisa dihubungi: {exc}",
                    rekaman=rekaman,
                    putaran=putaran,
                    berhenti_karena="galat_model",
                    percakapan=pesan,
                )

            pesan.append(balasan)
            panggilan = balasan.get("tool_calls") or []

            if not panggilan:
                return HasilAgen(
                    ringkasan=(balasan.get("content") or "").strip(),
                    rekaman=rekaman,
                    putaran=putaran,
                    berhenti_karena="selesai",
                    percakapan=pesan,
                )

            for p in panggilan:
                fungsi = p.get("function", {})
                nama = fungsi.get("name", "")
                argumen = _argumen(fungsi.get("arguments"))
                jejak = rekaman.tambah(jalankan(nama, argumen))
                LOG.info("putaran %s: %s", putaran, jejak.ringkas())
                if saat_alat:
                    saat_alat(jejak)
                pesan.append(
                    {"role": "tool", "name": nama, "content": jejak.untuk_model()}
                )

        # Batas putaran tercapai. Jejaknya tetap berguna - angka yang sudah
        # dihitung tidak hilang hanya karena model tidak tahu kapan berhenti.
        return HasilAgen(
            ringkasan=_ringkasan_darurat(rekaman),
            rekaman=rekaman,
            putaran=maks_putaran,
            berhenti_karena="batas_putaran",
            percakapan=pesan,
        )


def _argumen(mentah: Any) -> dict[str, Any]:
    """Ollama mengirim arguments sebagai dict, sebagian model sebagai string JSON."""
    if isinstance(mentah, dict):
        return mentah
    if isinstance(mentah, str):
        try:
            terurai = json.loads(mentah)
        except json.JSONDecodeError:
            return {}
        return terurai if isinstance(terurai, dict) else {}
    return {}


def _ringkasan_darurat(rekaman: Rekaman) -> str:
    """Ringkasan dari jejak saat model tidak sempat menuliskannya sendiri."""
    berhasil = [j for j in rekaman.jejak if j.berhasil]
    if not berhasil:
        return (
            "Agen mencapai batas putaran tanpa satu pun tool berhasil dijalankan. "
            "Perhitungan wajib dikerjakan manual."
        )
    baris = "\n".join(f"- {j.ringkas()}" for j in berhasil)
    return (
        "Agen mencapai batas putaran sebelum menutup analisis. Perhitungan yang "
        f"sudah selesai:\n{baris}"
    )


def jalankan_agen(konteks: str, **kwargs) -> HasilAgen:
    """Pintasan satu baris untuk antarmuka dan notebook."""
    return AgenPerhitungan().jalankan(konteks, **kwargs)

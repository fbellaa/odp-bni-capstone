"""Skema keluaran pembacaan dokumen.

Ini kontrak antara model bahasa dan sisa sistem. Model boleh salah; yang tidak
boleh adalah kesalahan itu lolos diam-diam ke perhitungan. Karena itu setiap
field yang dipakai hitungan bertipe angka - bukan string - dan setiap dokumen
membawa `sumber` supaya tiap angka di memo bisa ditunjuk balik ke halaman
PDF-nya.

Bentuk `BerkasPengajuan.argumen_resolusi()` sengaja dibuat persis sama dengan
tanda tangan `pipelines.graph.resolusi.telusuri_afiliasi()`: alamat operasional,
daftar nama pengurus, dan daftar rekening lawan. Kontrak itu sudah ada sebelum
lapisan ini ditulis; tugas parser hanya mengisinya.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

JenisDokumen = Literal["rekening_koran", "laporan_keuangan", "akta", "tidak_dikenali"]


class Sumber(BaseModel):
    """Asal-usul satu dokumen terstruktur."""

    berkas: str
    halaman: list[int] = Field(default_factory=list)
    jumlah_halaman: int = 0


# ------------------------------------------------------------ rekening koran
class BarisMutasi(BaseModel):
    tanggal: date | None = None
    keterangan: str = ""
    debit: float = 0.0
    kredit: float = 0.0
    saldo: float | None = None
    # Dua field inilah yang menyalakan jalur "counterparty bersama" pada
    # resolusi afiliasi. Sisanya hanya konteks bagi analis.
    rekening_lawan: str | None = None
    nama_lawan: str | None = None


class RekeningKoran(BaseModel):
    nomor_rekening: str | None = None
    nama_pemilik: str | None = None
    bank: str | None = None
    periode_awal: date | None = None
    periode_akhir: date | None = None
    saldo_rata_rata: float | None = None
    mutasi: list[BarisMutasi] = Field(default_factory=list)

    @property
    def daftar_rekening_lawan(self) -> list[str]:
        terlihat = {
            m.rekening_lawan.strip()
            for m in self.mutasi
            if m.rekening_lawan and m.rekening_lawan.strip()
        }
        return sorted(terlihat)

    @property
    def total_kredit(self) -> float:
        return sum(m.kredit for m in self.mutasi)

    @property
    def total_debit(self) -> float:
        return sum(m.debit for m in self.mutasi)


# --------------------------------------------------------- laporan keuangan
class LaporanKeuangan(BaseModel):
    """Pos yang benar-benar dipakai perhitungan, bukan seluruh isi lapkeu.

    Menambah pos berarti menambah kesempatan model salah baca. Yang masuk ke
    sini hanya yang punya konsumen di `copilot.alat.keuangan`.
    """

    periode: str | None = None
    mata_uang: str = "IDR"
    penjualan: float | None = None
    laba_kotor: float | None = None
    ebitda: float | None = None
    laba_bersih: float | None = None
    beban_bunga: float | None = None
    total_aset: float | None = None
    total_liabilitas: float | None = None
    utang_berbunga: float | None = None
    ekuitas: float | None = None
    kas_dan_setara: float | None = None
    arus_kas_operasi: float | None = None
    piutang_usaha: float | None = None
    persediaan: float | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _kosong_jadi_none(cls, v):
        # Model kerap mengisi "-", "n/a", atau "" untuk pos yang tidak ada.
        if isinstance(v, str) and v.strip().lower() in {"", "-", "n/a", "na", "null", "tidak ada"}:
            return None
        return v


# ---------------------------------------------------------------------- akta
class Pengurus(BaseModel):
    nama: str
    jabatan: str | None = None


class PemegangSaham(BaseModel):
    nama: str
    persentase: float | None = None
    jenis: Literal["perorangan", "badan_hukum", "tidak_diketahui"] = "tidak_diketahui"


class Akta(BaseModel):
    nama_perusahaan: str | None = None
    npwp: str | None = None
    nomor_akta: str | None = None
    tanggal_akta: date | None = None
    alamat_operasional: str | None = None
    alamat_domisili: str | None = None
    pengurus: list[Pengurus] = Field(default_factory=list)
    pemegang_saham: list[PemegangSaham] = Field(default_factory=list)


# ----------------------------------------------------------------- gabungan
class DokumenTerstruktur(BaseModel):
    jenis: JenisDokumen
    sumber: Sumber
    rekening_koran: RekeningKoran | None = None
    laporan_keuangan: LaporanKeuangan | None = None
    akta: Akta | None = None
    catatan: list[str] = Field(default_factory=list)


class BerkasPengajuan(BaseModel):
    """Seluruh dokumen satu pengajuan, sesudah dibaca dan digabung."""

    nama_debitur: str | None = None
    dokumen: list[DokumenTerstruktur] = Field(default_factory=list)

    # -------------------------------------------------------------- pintasan
    @property
    def semua_rekening_koran(self) -> list[RekeningKoran]:
        return [d.rekening_koran for d in self.dokumen if d.rekening_koran]

    @property
    def lapkeu_terbaru(self) -> LaporanKeuangan | None:
        daftar = [d.laporan_keuangan for d in self.dokumen if d.laporan_keuangan]
        if not daftar:
            return None
        # Periode ditulis bebas ("2025", "FY2025", "31 Des 2025"); urutan
        # leksikal atas string yang diawali tahun sudah cukup untuk demo, dan
        # kegagalannya kelihatan (periode ikut tercetak di memo).
        return sorted(daftar, key=lambda lap: lap.periode or "")[-1]

    @property
    def akta_utama(self) -> Akta | None:
        daftar = [d.akta for d in self.dokumen if d.akta]
        return daftar[-1] if daftar else None

    # --------------------------------------------- kontrak resolusi afiliasi
    def argumen_resolusi(self) -> dict[str, object]:
        """Bahan untuk `telusuri_afiliasi()`.

        Nilai kosong dikembalikan sebagai None / list kosong dengan sengaja:
        `telusuri_afiliasi` membedakan "dokumen tidak disertakan" dari "tidak
        ada kecocokan", dan pembedaan itu ikut muncul di memo sebagai batas
        penelaahan.
        """
        akta = self.akta_utama
        rekening_lawan: list[str] = []
        for rk in self.semua_rekening_koran:
            rekening_lawan.extend(rk.daftar_rekening_lawan)

        return {
            "alamat_operasional": (akta.alamat_operasional if akta else None) or None,
            "nama_pengurus": [p.nama for p in akta.pengurus] if akta else [],
            "rekening_lawan": sorted(set(rekening_lawan)),
        }

    def kelengkapan(self) -> dict[str, bool]:
        """Dipakai antarmuka untuk menandai dokumen yang masih kurang."""
        return {
            "rekening_koran": bool(self.semua_rekening_koran),
            "laporan_keuangan": self.lapkeu_terbaru is not None,
            "akta_dan_kepemilikan": self.akta_utama is not None,
        }

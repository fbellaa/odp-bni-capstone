"""Rantai copilot pengajuan: dokumen PDF -> fakta -> model -> keputusan.

Modul ini yang menyambung tiga lapisan yang sebelumnya terpisah di dua halaman:

    unggahan PDF ──> pembacaan dokumen ──┐
                                          ├──> entitas gabungan ──> model PD/LGD
    chat relationship manager ───────────┘                          + klaster
                                                                   + agen tool

Pembacaan dokumen punya dua jalur. Jalur penuh memakai model bahasa lokal lewat
paket `copilot` (hasilnya terstruktur dan rapi). Jalur cadangan hanya membaca
teks PDF dan menyapu angka dengan pola — dipakai saat Ollama tidak hidup, supaya
demo tetap berjalan dan halaman tetap jujur menyebut jalur mana yang dipakai.

Yang datang dari dokumen selalu ditandai sumbernya, sehingga analis bisa
membedakan angka hasil pembacaan berkas, angka dari narasi, dan angka isian
median portofolio.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from lib import copilot_lokal as ck
from lib import dummy_data, model_nyata as mn

# Jenis dokumen yang diminta pada pengajuan komersial.
JENIS_DOKUMEN = {
    "laporan_keuangan": "Laporan keuangan / home statement",
    "akta": "Data kepemilikan (pemegang saham, pengurus)",
    "rekening_koran": "Rekening koran perusahaan",
}

SUMBER_WARNA = {"dokumen": "#1f8a5f", "narasi": "#1d5fa8", "rujukan": "#8b97a6"}


@dataclass
class HasilDokumen:
    """Keluaran pembacaan berkas, apa pun jalur yang dipakai."""

    jalur: str                      # "llm" atau "pola"
    berkas: object | None = None    # BerkasPengajuan bila jalur llm
    per_berkas: list[dict] = field(default_factory=list)
    fakta: dict = field(default_factory=dict)
    sumber_fakta: dict = field(default_factory=dict)
    pemegang_saham: list[dict] = field(default_factory=list)
    pengurus: list[str] = field(default_factory=list)
    catatan: list[str] = field(default_factory=list)

    def kelengkapan(self) -> dict[str, bool]:
        ada = {j: False for j in JENIS_DOKUMEN}
        for d in self.per_berkas:
            if d.get("jenis") in ada:
                ada[d["jenis"]] = True
        return ada


# --------------------------------------------------------------------------
# Kesiapan lingkungan
# --------------------------------------------------------------------------
def status_lengkap() -> dict:
    """Gabungan kesiapan lapisan model dan lapisan copilot lokal."""
    status = {"copilot": ck.TERSEDIA, "ollama": False, "index": False,
              "galat_impor": ck.GALAT_IMPOR, "host": "-", "profil": "-"}
    if ck.TERSEDIA:
        try:
            lingkungan = ck.status_lingkungan()
            status.update(
                ollama=bool(lingkungan.get("ollama")),
                index=bool(lingkungan.get("index")),
                host=lingkungan.get("host", "-"),
                profil=lingkungan.get("profil", "-"),
                model_kurang=lingkungan.get("model_kurang", []),
            )
        except Exception as exc:
            status["galat_impor"] = f"{type(exc).__name__}: {exc}"
    status.update(mn.status_lapisan_model())
    return status


# --------------------------------------------------------------------------
# Jalur cadangan: baca teks PDF dan sapu angka dengan pola
# --------------------------------------------------------------------------
_SKALA = [
    (re.compile(r"dalam\s+(jutaan|juta)\s+rupiah", re.I), 1e6),
    (re.compile(r"dalam\s+(ribuan|ribu)\s+rupiah", re.I), 1e3),
    (re.compile(r"dalam\s+(miliar|milyar)\s+rupiah", re.I), 1e9),
]

# Pos yang punya konsumen di perhitungan; menambah pos berarti menambah
# kesempatan salah baca, jadi daftarnya sengaja pendek.
_POS = {
    "penjualan": ["penjualan bersih", "pendapatan usaha", "penjualan neto",
                  "total pendapatan", "penjualan"],
    "ebitda": ["ebitda"],
    "laba_bersih": ["laba bersih", "laba tahun berjalan", "laba periode berjalan"],
    "beban_bunga": ["beban bunga", "biaya bunga"],
    "total_aset": ["jumlah aset", "total aset"],
    "total_liabilitas": ["jumlah liabilitas", "total liabilitas"],
    "ekuitas": ["jumlah ekuitas", "total ekuitas"],
    "utang_berbunga": ["utang bank", "pinjaman bank", "utang berbunga"],
    "arus_kas_operasi": ["arus kas dari aktivitas operasi", "kas bersih dari operasi"],
    "saldo_rata_rata": ["saldo rata-rata", "rata-rata saldo", "saldo akhir"],
}

_ANGKA = re.compile(r"\(?\s*(?:rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*\)?", re.I)


def _ke_float(teks: str, negatif: bool) -> float | None:
    """Angka Indonesia: titik ribuan, koma desimal. Kurung berarti negatif."""
    bersih = teks.strip()
    if bersih.count(",") == 1 and len(bersih.split(",")[-1]) <= 2:
        bersih = bersih.replace(".", "").replace(",", ".")
    else:
        bersih = bersih.replace(".", "").replace(",", "")
    try:
        nilai = float(bersih)
    except ValueError:
        return None
    return -nilai if negatif else nilai


def _angka_pada_baris(baris: str) -> float | None:
    cocok = _ANGKA.search(baris)
    if not cocok:
        return None
    return _ke_float(cocok.group(1), negatif=cocok.group(0).strip().startswith("("))


def fakta_dari_teks(teks: str) -> dict:
    """Sapu pos keuangan dari teks PDF apa adanya.

    Nilai yang terlalu kecil untuk segmen komersial dinaikkan skalanya menurut
    keterangan "dalam jutaan/ribuan rupiah" pada kepala laporan; kalau tidak ada
    keterangan itu, angka dipakai apa adanya dan halaman menandainya.
    """
    faktor = 1.0
    for pola, skala in _SKALA:
        if pola.search(teks):
            faktor = skala
            break

    hasil: dict[str, float] = {}
    for baris in teks.splitlines():
        rendah = baris.lower().strip()
        if not rendah:
            continue
        for pos, kunci in _POS.items():
            if pos in hasil:
                continue
            if any(rendah.startswith(k) or f" {k}" in rendah for k in kunci):
                nilai = _angka_pada_baris(baris[len(baris) - len(rendah):])
                if nilai is not None and nilai != 0:
                    hasil[pos] = nilai * faktor
    return hasil


_SAHAM = re.compile(
    r"(?:pt|cv|tuan|nyonya|ny\.|tn\.)?\s*([A-Z][\w.\- ]{2,60}?)\s*[-:•]?\s*"
    r"(\d{1,3}(?:[.,]\d+)?)\s*%",
    re.M,
)


def pemegang_saham_dari_teks(teks: str) -> list[dict]:
    """Baris "Nama ... 60%" pada akta atau daftar pemegang saham."""
    hasil, terlihat = [], set()
    for nama, porsi in _SAHAM.findall(teks):
        bersih = " ".join(nama.split())
        if len(bersih) < 3 or bersih.lower() in terlihat:
            continue
        try:
            nilai = float(porsi.replace(",", "."))
        except ValueError:
            continue
        if not 0 < nilai <= 100:
            continue
        terlihat.add(bersih.lower())
        hasil.append({"nama": bersih, "porsi": nilai / 100})
    return sorted(hasil, key=lambda p: p["porsi"], reverse=True)[:12]


def baca_dengan_pola(path_list: list[Path], jenis_manual: dict[str, str]) -> HasilDokumen:
    """Jalur cadangan tanpa model bahasa."""
    from copilot.dokumen import pdf as pdf_util

    hasil = HasilDokumen(jalur="pola")
    fakta: dict[str, float] = {}
    for path in path_list:
        try:
            halaman = pdf_util.baca_halaman(path)
        except Exception as exc:
            hasil.catatan.append(f"{Path(path).name}: {exc}")
            continue
        teks = pdf_util.gabung_teks(halaman)
        jenis = jenis_manual.get(Path(path).name) or pdf_util.tebak_jenis(halaman)[0]
        temuan = fakta_dari_teks(teks)
        for kunci, nilai in temuan.items():
            fakta.setdefault(kunci, nilai)
        if jenis == "akta":
            hasil.pemegang_saham.extend(pemegang_saham_dari_teks(teks))
        hasil.per_berkas.append({
            "berkas": Path(path).name,
            "jenis": jenis,
            "halaman": len([h for h in halaman if h.teks.strip()]),
            "total_halaman": len(halaman),
            "pos_terbaca": ", ".join(temuan) or "-",
        })
    hasil.fakta = fakta
    hasil.sumber_fakta = {k: "dokumen" for k in fakta}
    if fakta:
        hasil.catatan.append(
            "Angka disapu dengan pola dari teks PDF, tanpa model bahasa. "
            "Periksa ulang sebelum dipakai untuk keputusan."
        )
    return hasil


def baca_dengan_llm(path_list: list[Path], jenis_manual: dict[str, str]) -> HasilDokumen:
    """Jalur penuh: `copilot.dokumen.ekstraksi` dengan model bahasa lokal."""
    berkas = ck.baca_dokumen(list(path_list), jenis_manual)
    hasil = HasilDokumen(jalur="llm", berkas=berkas)
    for d in berkas.dokumen:
        hasil.per_berkas.append({
            "berkas": d.sumber.berkas,
            "jenis": d.jenis,
            "halaman": len(d.sumber.halaman),
            "total_halaman": d.sumber.jumlah_halaman,
            "pos_terbaca": "; ".join(d.catatan) or "terbaca",
        })

    lapkeu = berkas.lapkeu_terbaru
    fakta: dict[str, float] = {}
    if lapkeu is not None:
        for pos in ("penjualan", "ebitda", "laba_bersih", "beban_bunga", "total_aset",
                    "total_liabilitas", "ekuitas", "utang_berbunga", "arus_kas_operasi"):
            nilai = getattr(lapkeu, pos, None)
            if nilai:
                fakta[pos] = float(nilai)
    rekening = berkas.semua_rekening_koran
    if rekening:
        saldo = [r.saldo_rata_rata for r in rekening if r.saldo_rata_rata]
        if saldo:
            fakta["saldo_rata_rata"] = float(sum(saldo) / len(saldo))
        else:
            # Tanpa baris saldo rata-rata, mutasi kredit dipakai sebagai
            # perkiraan kasar perputaran rekening dan ditandai demikian.
            masuk = sum(r.total_kredit for r in rekening)
            if masuk:
                fakta["mutasi_kredit"] = float(masuk)

    akta = berkas.akta_utama
    if akta is not None:
        hasil.pemegang_saham = [
            {"nama": p.nama, "porsi": (p.persentase or 0) / 100 if (p.persentase or 0) > 1
             else (p.persentase or 0), "jenis": p.jenis}
            for p in akta.pemegang_saham
        ]
        hasil.pengurus = [f"{p.nama} — {p.jabatan or 'pengurus'}" for p in akta.pengurus]
    hasil.fakta = fakta
    hasil.sumber_fakta = {k: "dokumen" for k in fakta}
    if berkas.nama_debitur:
        hasil.fakta["nama_debitur"] = berkas.nama_debitur
        hasil.sumber_fakta["nama_debitur"] = "dokumen"
    return hasil


def simpan_unggahan(unggahan) -> Path:
    """Tulis unggahan Streamlit ke cakram; pembaca PDF butuh path."""
    if ck.TERSEDIA:
        return ck.simpan_unggahan(unggahan)
    tujuan = Path(st.session_state.get("_dir_unggahan", ".")) / unggahan.name
    tujuan.write_bytes(unggahan.getbuffer())
    return tujuan


# --------------------------------------------------------------------------
# Penggabungan narasi dan dokumen
# --------------------------------------------------------------------------
def gabung_entitas(teks_chat: str, dokumen: HasilDokumen | None) -> tuple[dict, dict]:
    """Entitas final untuk model, plus asal-usul tiap angka.

    Aturannya satu kalimat: angka dari dokumen menang atas angka dari narasi,
    karena narasi adalah ingatan relationship manager sedangkan dokumen adalah
    berkas yang bisa dibuka ulang saat komite bertanya.
    """
    entitas = dummy_data.ekstraksi_entitas(teks_chat)
    asal = {k: "narasi" for k in entitas}

    if dokumen is None or not dokumen.fakta:
        return entitas, asal

    f = dokumen.fakta
    if f.get("nama_debitur"):
        entitas["nama_debitur"] = f["nama_debitur"]
        asal["nama_debitur"] = "dokumen"
    if f.get("penjualan"):
        entitas["penjualan_tahunan"] = float(f["penjualan"])
        asal["penjualan_tahunan"] = "dokumen"
    if f.get("ebitda"):
        entitas["ebitda_rp"] = float(f["ebitda"])
        asal["ebitda_rp"] = "dokumen"
        if entitas.get("penjualan_tahunan"):
            entitas["ebitda_margin"] = float(f["ebitda"]) / entitas["penjualan_tahunan"]
            asal["ebitda_margin"] = "dokumen"
    for pos, kunci in [
        ("ekuitas", "ekuitas_rp"), ("total_aset", "total_aset_rp"),
        ("total_liabilitas", "total_liabilitas_rp"), ("beban_bunga", "beban_bunga_rp"),
        ("laba_bersih", "laba_bersih_rp"),
    ]:
        if f.get(pos):
            entitas[kunci] = float(f[pos])
            asal[kunci] = "dokumen"
    if f.get("ekuitas") and f.get("total_liabilitas"):
        entitas["der"] = float(f["total_liabilitas"]) / max(float(f["ekuitas"]), 1.0)
        asal["der"] = "dokumen"
    if f.get("saldo_rata_rata"):
        entitas["saldo_giro_rata"] = float(f["saldo_rata_rata"])
        asal["saldo_giro_rata"] = "dokumen"

    if dokumen.pemegang_saham:
        entitas["jumlah_pemegang_saham"] = len(dokumen.pemegang_saham)
        entitas["porsi_pengendali"] = float(dokumen.pemegang_saham[0]["porsi"])
        asal["porsi_pengendali"] = "dokumen"
    return entitas, asal


def lengkapi_fitur_graf(entitas: dict, application_id: str) -> dict:
    """Fitur yang datangnya dari lapisan graf dan riwayat, bukan dari berkas."""
    fitur = dict(entitas)
    fitur.update(
        utang_berbunga_eksisting=entitas.get(
            "total_liabilitas_rp", entitas["plafon"] * 0.25) * 0.5,
        konversi_ebitda_kas=0.62 if entitas.get("indikasi_konsentrasi_pembeli") else 0.76,
        utilisasi_plafon=0.72,
        buyer_concentration_hhi=0.71 if entitas.get("indikasi_konsentrasi_pembeli") else 0.32,
        supplier_concentration_hhi=0.66 if entitas.get("indikasi_konsentrasi_pemasok") else 0.30,
        neighbor_default_rate_1hop=0.09 if entitas.get("indikasi_rangkap_jabatan") else 0.035,
        group_exposure_share=min(0.28 + 0.11 * entitas.get("jumlah_entitas_grup", 1), 0.95),
        tenure_nasabah_thn=max(entitas.get("umur_usaha_thn", 10.0) - 6.0, 0.0),
    )
    jaringan = dummy_data.score_network_risk(application_id)
    fitur["network_risk_score"] = jaringan["skor"]
    return fitur

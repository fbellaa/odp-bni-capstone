"""Langkah 7: injeksi afiliasi tersembunyi.

SPESIFIKASI INI SENGAJA DIPISAH DARI KODE DETEKSI (proposal §4). Modul ini hanya
menanam struktur dan mencatat ground truth-nya; tidak ada satu baris pun di sini
yang membaca hasil deteksi.

---

Arah kausal - inilah yang membedakan injeksi yang sah dari kebocoran:

    SALAH  ambil debitur yang gagal bayar -> hubungkan mereka
           struktur mengkodekan label, model membaca label balik, AUC palsu.

    BENAR  bentuk klaster -> atur URUTAN WAKTU gagal bayarnya
           pada saat penilaian yang terlihat hanya gagal bayar MASA LALU.

Karena itu tiap klaster memakai dua angkatan:

    2 sumber      buku lama, benar-benar gagal bayar, jatuh 2022-2024
    2 terinfeksi  buku baru, benar-benar gagal bayar, mengajukan 2025
    4 sehat       buku baru, tidak gagal bayar (dilusi)

Saat anggota buku baru mengajukan di 2025, sumbernya sudah kolaps dan terlihat
lewat neighbor_default_rate_1hop. Label tiap debitur tetap NYATA dari panel US -
yang disintesis adalah struktur ketergantungan dan urutan waktunya.

KEBOCORAN RESIDUAL YANG HARUS DILAPORKAN: keanggotaan klaster berkorelasi dengan
label (2 dari 6 anggota buku baru gagal bayar, versus base rate ~7%). Ukur dan
laporkan dua AUC terpisah - dari keanggotaan saja, dan dari gagal bayar afiliasi
masa lalu saja. Yang pertama artefak, yang kedua sinyal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipelines.config import settings
from pipelines.utils import read_table, write_table

LOG = logging.getLogger("pipelines.afiliasi")

MEKANISME = ("nominee_bersama", "alamat_operasional_bersama", "siklus_pembayaran")

# Edge afiliasi harus sudah berlaku sebelum anggota buku baru mengajukan.
BULAN_EDGE_MENDAHULUI = 6


def _pilih_anggota(
    kandidat: pd.DataFrame, jumlah: int, terpakai: set[int], rng: np.random.Generator
) -> list[int]:
    """Ambil `jumlah` cif yang belum terpakai, dari grup usaha yang berbeda-beda."""
    tersedia = kandidat[~kandidat["cif_sk"].isin(terpakai)]
    if len(tersedia) < jumlah:
        return []
    urut = tersedia.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))

    dipilih: list[int] = []
    grup_dipakai: set[int] = set()
    for cif, grup in zip(urut["cif_sk"], urut["grup_id"]):
        # Afiliasi tersembunyi bermakna kalau ia MELINTASI grup yang kasat mata.
        if grup in grup_dipakai:
            continue
        dipilih.append(int(cif))
        grup_dipakai.add(int(grup))
        if len(dipilih) == jumlah:
            return dipilih
    return []


def bentuk_klaster(rng: np.random.Generator) -> pd.DataFrame:
    """Bentuk klaster afiliasi tersembunyi lintas grup dan lintas angkatan."""
    peta = read_table(
        "silver", "sl_peta_cif", columns=["cif_sk", "grup_id", "angkatan", "label_default_debitur"]
    )

    sumber_pool = peta[(peta["angkatan"] == "buku_lama") & (peta["label_default_debitur"] == 1)]
    infeksi_pool = peta[(peta["angkatan"] == "buku_baru") & (peta["label_default_debitur"] == 1)]
    sehat_pool = peta[(peta["angkatan"] == "buku_baru") & (peta["label_default_debitur"] == 0)]

    n_sumber = settings.afiliasi_default_per_klaster
    n_infeksi = settings.afiliasi_default_per_klaster
    n_sehat = settings.afiliasi_sehat_per_klaster

    # Batasi penyerapan: klaster tidak boleh menghabiskan kolam gagal bayar,
    # kalau tidak keanggotaan klaster berubah menjadi label.
    porsi = settings.afiliasi_porsi_default_terpakai
    kuota_sumber = int(len(sumber_pool) * porsi)
    kuota_infeksi = int(len(infeksi_pool) * porsi)
    maks_klaster = min(kuota_sumber // n_sumber, kuota_infeksi // n_infeksi)
    LOG.info(
        "kolam afiliasi: sumber=%s terinfeksi=%s sehat=%s -> maksimum %s klaster",
        len(sumber_pool),
        len(infeksi_pool),
        len(sehat_pool),
        maks_klaster,
    )

    baris = []
    terpakai: set[int] = set()
    jeda_min, jeda_maks = settings.afiliasi_jeda_bulan

    for klaster_id in range(1, maks_klaster + 1):
        sumber = _pilih_anggota(sumber_pool, n_sumber, terpakai, rng)
        infeksi = _pilih_anggota(infeksi_pool, n_infeksi, terpakai | set(sumber), rng)
        sehat = _pilih_anggota(sehat_pool, n_sehat, terpakai | set(sumber) | set(infeksi), rng)
        if not (sumber and infeksi and sehat):
            break

        mekanisme = MEKANISME[int(rng.integers(0, len(MEKANISME)))]
        terpakai.update(sumber + infeksi + sehat)

        for urutan, cif in enumerate(sumber):
            baris.append(
                {
                    "afiliasi_id": klaster_id,
                    "cif_sk": cif,
                    "peran": "sumber",
                    "mekanisme": mekanisme,
                    "urutan": urutan,
                    "jeda_bulan": 0,
                }
            )
        for urutan, cif in enumerate(infeksi):
            baris.append(
                {
                    "afiliasi_id": klaster_id,
                    "cif_sk": cif,
                    "peran": "terinfeksi",
                    "mekanisme": mekanisme,
                    "urutan": urutan,
                    "jeda_bulan": int(rng.integers(jeda_min, jeda_maks + 1)),
                }
            )
        for urutan, cif in enumerate(sehat):
            baris.append(
                {
                    "afiliasi_id": klaster_id,
                    "cif_sk": cif,
                    "peran": "sehat",
                    "mekanisme": mekanisme,
                    "urutan": urutan,
                    "jeda_bulan": 0,
                }
            )

    klaster = pd.DataFrame(baris)
    if klaster.empty:
        LOG.warning("tidak ada klaster afiliasi yang terbentuk")
        return klaster

    LOG.info(
        "afiliasi tersembunyi: %s klaster, %s debitur tersentuh, mekanisme %s",
        klaster["afiliasi_id"].nunique(),
        klaster["cif_sk"].nunique(),
        klaster.drop_duplicates("afiliasi_id")["mekanisme"].value_counts().to_dict(),
    )
    return klaster


def umur_default_paksa(klaster: pd.DataFrame) -> pd.Series:
    """Paksa sumber gagal bayar lebih awal supaya sempat terlihat sebelum 2025.

    Dikembalikan sebagai umur (hari sejak pencairan) per cif_sk. Hanya sumber
    yang dipaksa; anggota terinfeksi memakai umur alaminya, dan urutannya sudah
    terjamin karena mereka baru mencairkan fasilitas dua tahun kemudian.
    """
    if klaster.empty:
        return pd.Series(dtype="float64")
    sumber = klaster[klaster["peran"] == "sumber"]
    # 120-420 hari: cukup lama untuk realistis, cukup cepat untuk jatuh sebelum
    # angkatan buku baru mengajukan.
    umur = 120 + (sumber["cif_sk"] * 7919 % 300)
    return pd.Series(umur.to_numpy(dtype="float64"), index=sumber["cif_sk"].to_numpy())


def build_afiliasi() -> pd.DataFrame:
    """Bentuk klaster dan simpan ground truth-nya ke layer silver."""
    rng = np.random.default_rng(settings.seed + 11)
    klaster = bentuk_klaster(rng)
    write_table(klaster, "silver", "sl_afiliasi_tersembunyi")
    return klaster


# --------------------------------------------------------------------- edge
def edge_afiliasi(
    klaster: pd.DataFrame, rng: np.random.Generator
) -> dict[str, pd.DataFrame]:
    """Terjemahkan tiap klaster menjadi relasi yang menyamar sebagai relasi biasa.

    Edge hasil injeksi TIDAK diberi penanda apa pun di GOLD_GRAPH_EDGES - kalau
    ditandai, ia tidak lagi tersembunyi. Ground truth-nya hidup terpisah di
    FACT_AFILIASI_TERSEMBUNYI dan terdaftar sebagai kolom terlarang.

    Mekanisme alamat_operasional_bersama tidak dikembalikan di sini. Klaster itu
    ditangani graph/alamat.py, yang memberinya baris DIM_ALAMAT sungguhan lalu
    menurunkan edge berbagi_atribut-nya lewat jalur yang sama dengan alamat ICIJ.
    Menyambung pasangannya langsung dari sini akan menghasilkan klaster yang
    berbagi alamat TANPA punya simpul alamat - penanda yang membocorkan injeksi.
    """
    if klaster.empty:
        return {"kepengurusan": pd.DataFrame(), "pasokan": pd.DataFrame()}

    pengajuan_awal = pd.Timestamp(settings.buku_baru_awal_pengajuan)
    valid_from = pengajuan_awal - pd.DateOffset(months=BULAN_EDGE_MENDAHULUI)

    kepengurusan, pasokan = [], []
    for afiliasi_id, sub in klaster.groupby("afiliasi_id"):
        anggota = sub["cif_sk"].tolist()
        mekanisme = sub["mekanisme"].iloc[0]

        if mekanisme == "nominee_bersama":
            # Satu pihak yang sama menjabat di seluruh anggota klaster.
            for cif in anggota:
                kepengurusan.append(
                    {"afiliasi_id": afiliasi_id, "cif_sk": cif, "valid_from": valid_from}
                )
        elif mekanisme == "alamat_operasional_bersama":
            continue  # ditangani graph/alamat.py lewat DIM_ALAMAT
        else:  # siklus_pembayaran
            for i, cif in enumerate(anggota):
                pasokan.append(
                    {
                        "afiliasi_id": afiliasi_id,
                        "cif_dari": cif,
                        "cif_ke": anggota[(i + 1) % len(anggota)],
                        "bobot": float(rng.uniform(2e9, 2e10)),
                        "valid_from": valid_from,
                    }
                )

    return {
        "kepengurusan": pd.DataFrame(kepengurusan),
        "pasokan": pd.DataFrame(pasokan),
    }

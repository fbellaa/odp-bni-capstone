"""DIM_ALAMAT dan FACT_ALAMAT_DEBITUR - alamat sebagai entitas, bukan kunci join.

Sebelum modul ini ada, alamat hanya dipakai sekali lalu dibuang: dua debitur yang
berbagi node alamat ICIJ langsung disambung menjadi edge berbagi_atribut, dan
teks alamatnya tidak pernah sampai ke layer gold. Akibatnya afiliasi lewat alamat
hanya bisa ditemukan untuk debitur yang SUDAH ada di graf. Calon nasabah yang
baru mengisi formulir tidak punya apa pun untuk dicocokkan - RM mengetik alamat,
dan tidak ada tabel untuk mencarinya.

Di sini alamat menjadi dimensi tersendiri: teksnya disimpan, dan di sebelahnya
bentuk ternormalisasi yang menjadi kunci pencocokan. 'Jl. Merdeka No. 12, Blok A'
dan 'JALAN MERDEKA NOMOR 12 BLOK A' menghasilkan string yang sama, sehingga
alamat calon nasabah baru bisa di-join ke portofolio yang sudah ada (lihat
pipelines/graph/resolusi.py).

Teks yang disimpan adalah alamat SINTETIS, bukan alamat asli ICIJ - lihat
_samarkan_alamat untuk alasannya dan untuk cara strukturnya dipertahankan.

Satu konsekuensi yang disengaja: klaster afiliasi tersembunyi bermekanisme
alamat_operasional_bersama sekarang ikut mendapat baris DIM_ALAMAT sungguhan.
Kalau tidak, klaster injeksi menjadi satu-satunya kelompok debitur yang berbagi
alamat TANPA punya simpul alamat - itu penanda yang membuatnya tidak lagi
tersembunyi.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from pipelines.config import settings
from pipelines.utils import read_table

LOG = logging.getLogger("pipelines.graph.alamat")

# Alamat dengan entitas sebanyak ini biasanya kantor agen registrasi, bukan
# tanda keterkaitan usaha. Ambangnya sama dengan MAKS_ENTITAS_PER_ALAMAT di
# transform/joins.py supaya dua tempat tidak berbeda pendapat.
MAKS_DEBITUR_PER_ALAMAT = 20

# Varian penulisan yang harus runtuh menjadi satu token. Dicampur Indonesia dan
# Inggris karena alamat ICIJ berbahasa Inggris sedangkan alamat sintetis dan
# formulir RM berbahasa Indonesia.
KANONIK_TOKEN = {
    "JALAN": "JL",
    "JLN": "JL",
    "JL": "JL",
    "GANG": "GANG",
    "GG": "GANG",
    "PERUMAHAN": "PERUM",
    "PERUM": "PERUM",
    "DUSUN": "DUSUN",
    "DSN": "DUSUN",
    "DESA": "DESA",
    "DS": "DESA",
    "NOMOR": "NO",
    "NOMER": "NO",
    "NRO": "NO",
    "NO": "NO",
    "GD": "GEDUNG",
    "GDG": "GEDUNG",
    "GEDUNG": "GEDUNG",
    "KOMP": "KOMPLEK",
    "KOMPLEKS": "KOMPLEK",
    "KOMPLEK": "KOMPLEK",
    "KEC": "KEC",
    "KECAMATAN": "KEC",
    "KEL": "KEL",
    "KELURAHAN": "KEL",
    "KAB": "KAB",
    "KABUPATEN": "KAB",
    "PROP": "PROV",
    "PROV": "PROV",
    "PROVINSI": "PROV",
    "BLK": "BLOK",
    "BLOK": "BLOK",
    "LT": "LANTAI",
    "LANTAI": "LANTAI",
    "STREET": "ST",
    "ST": "ST",
    "ROAD": "RD",
    "RD": "RD",
    "AVENUE": "AVE",
    "AV": "AVE",
    "AVE": "AVE",
    "SUITE": "SUITE",
    "STE": "SUITE",
    "FLOOR": "FLOOR",
    "FL": "FLOOR",
    "BUILDING": "BUILDING",
    "BLDG": "BUILDING",
}

_BUKAN_ALNUM = re.compile(r"[^0-9A-Z]+")


def normalisasi_alamat(seri: pd.Series) -> pd.Series:
    """Bentuk kanonik satu alamat, dipakai sebagai kunci pencocokan.

    Bukan geocoding - ini normalisasi tekstual. Cukup untuk menangkap variasi
    penulisan yang lazim ('Jl.' vs 'Jalan'), tidak cukup untuk menangkap alamat
    yang benar-benar ditulis ulang. Untuk kasus itu resolusi.py menyediakan
    pencocokan berbasis kemiripan token.
    """
    teks = seri.astype("string").fillna("").str.upper()
    teks = teks.str.replace(_BUKAN_ALNUM, " ", regex=True).str.strip()

    def _kanonik(baris: str) -> str:
        if not baris:
            return ""
        return " ".join(KANONIK_TOKEN.get(t, t) for t in baris.split())

    return teks.map(_kanonik).astype("string")


KOLOM_ALAMAT_MENTAH = [
    "cif_sk",
    "alamat_teks",
    "negara",
    "src_icij_node_id",
    "valid_from",
    "valid_to",
    "sumber",
]


def _kosong() -> pd.DataFrame:
    return pd.DataFrame(columns=KOLOM_ALAMAT_MENTAH)


def _samarkan_alamat(alamat_asli: pd.Series) -> pd.Series:
    """Ganti alamat ICIJ dengan alamat Indonesia sintetis, strukturnya dijaga.

    Dua alasan, dan keduanya wajib:

    1. PRIVASI. Alamat di berkas ICIJ adalah data nyata dari dokumen bocoran.
       Kebijakan proyek ini sudah menolak membawa nama asli ICIJ ke gold (lihat
       buat_pihak_dan_relasi) - alamat tidak boleh diperlakukan lebih longgar.
    2. KONSISTENSI. Debitur di sini badan hukum Indonesia sintetis. Alamat asli
       ICIJ membuat mereka berkantor di Sliema dan Tortola, yang langsung
       terlihat janggal begitu ditampilkan di layar RM.

    Yang DIPERTAHANKAN adalah strukturnya: penyamaran dikunci pada bentuk
    ternormalisasi alamat aslinya, jadi dua entitas yang berbagi satu alamat di
    ICIJ tetap berbagi satu alamat sesudah disamarkan. Justru keterkaitan itulah
    yang ingin ditemukan, dan ia lolos utuh.
    """
    from faker import Faker

    fk = Faker("id_ID")
    Faker.seed(settings.seed + 29)

    kunci_asli = normalisasi_alamat(alamat_asli)
    unik = sorted(k for k in kunci_asli.dropna().unique() if k)
    peta = {k: fk.address().replace("\n", ", ") for k in unik}
    return kunci_asli.map(peta).astype("string")


def _alamat_icij() -> pd.DataFrame:
    """Alamat terdaftar ICIJ untuk debitur yang terpilih, sudah jadi per-cif."""
    rel = read_table("silver", "sl_icij_alamat_terpilih")
    node = read_table("silver", "sl_icij_address", columns=["node_id", "address"])
    peta = read_table("silver", "sl_peta_cif", columns=["cif_sk", "node_id"])

    entitas_ke_cif = peta.set_index("node_id")["cif_sk"]
    df = rel.copy()
    df["cif_sk"] = df["node_id_start"].map(entitas_ke_cif)
    df = df.dropna(subset=["cif_sk"])
    if df.empty:
        return _kosong()

    teks = node.set_index("node_id")
    df["alamat_asli"] = df["node_id_end"].map(teks["address"])
    df = df[df["alamat_asli"].notna() & (df["alamat_asli"].astype(str).str.strip() != "")]
    if df.empty:
        return _kosong()

    # Teks aslinya berhenti di sini dan tidak pernah ikut ke gold.
    df["alamat_teks"] = _samarkan_alamat(df["alamat_asli"])

    return pd.DataFrame(
        {
            "cif_sk": df["cif_sk"].astype("int64"),
            "alamat_teks": df["alamat_teks"].astype("string"),
            "negara": pd.Series("Indonesia", index=df.index, dtype="string"),
            "src_icij_node_id": df["node_id_end"].astype("Int64"),
            "valid_from": pd.to_datetime(df["valid_from"]),
            "valid_to": pd.to_datetime(df["valid_to"]),
            "sumber": "icij",
        }
    )


def _alamat_afiliasi(klaster: pd.DataFrame) -> pd.DataFrame:
    """Alamat operasional bersama untuk klaster afiliasi tersembunyi.

    Dibangkitkan Faker supaya berbentuk alamat Indonesia yang wajar, dan
    diperlakukan persis seperti alamat ICIJ mulai dari sini - tidak ada kolom
    penanda apa pun yang membedakannya di DIM_ALAMAT.
    """
    if klaster is None or klaster.empty:
        return _kosong()

    pakai_alamat = klaster[klaster["mekanisme"] == "alamat_operasional_bersama"]
    if pakai_alamat.empty:
        return _kosong()

    from faker import Faker

    fk = Faker("id_ID")
    Faker.seed(settings.seed + 23)

    # Edge afiliasi harus sudah berlaku sebelum anggota buku baru mengajukan;
    # angka jedanya milik generators/afiliasi.py, jadi diimpor dari sana.
    from pipelines.generators.afiliasi import BULAN_EDGE_MENDAHULUI

    valid_from = pd.Timestamp(settings.buku_baru_awal_pengajuan) - pd.DateOffset(
        months=BULAN_EDGE_MENDAHULUI
    )

    baris = []
    for afiliasi_id, sub in pakai_alamat.groupby("afiliasi_id"):
        teks = fk.address().replace("\n", ", ")
        for cif in sub["cif_sk"]:
            baris.append(
                {
                    "cif_sk": int(cif),
                    "alamat_teks": teks,
                    "negara": "Indonesia",
                    "src_icij_node_id": pd.NA,
                    "valid_from": valid_from,
                    "valid_to": pd.NaT,
                    "sumber": "icij",
                }
            )
    LOG.info("alamat bersama untuk %s klaster afiliasi", pakai_alamat["afiliasi_id"].nunique())
    df = pd.DataFrame(baris)
    # Samakan dtype dengan jalur ICIJ, kalau tidak concat di bangun_dim_alamat
    # menebak sendiri dan kolom yang seluruhnya kosong berubah jenis.
    df["src_icij_node_id"] = df["src_icij_node_id"].astype("Int64")
    df["negara"] = df["negara"].astype("string")
    df["alamat_teks"] = df["alamat_teks"].astype("string")
    df["valid_to"] = df["valid_to"].astype("datetime64[ns]")
    return df


def bangun_dim_alamat(klaster: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """DIM_ALAMAT (satu baris per alamat unik) + FACT_ALAMAT_DEBITUR (jembatan)."""
    bagian = [_alamat_icij(), _alamat_afiliasi(klaster)]
    gabung = pd.concat([b for b in bagian if len(b)], ignore_index=True)
    if gabung.empty:
        return {
            "dim_alamat": pd.DataFrame(
                columns=[
                    "alamat_id",
                    "alamat_teks",
                    "alamat_normal",
                    "negara",
                    "jumlah_debitur",
                    "is_alamat_agen",
                    "src_icij_node_id",
                ]
            ),
            "fact_alamat_debitur": pd.DataFrame(
                columns=["cif_sk", "alamat_id", "jenis_alamat", "valid_from", "valid_to", "sumber"]
            ),
        }

    gabung["alamat_normal"] = normalisasi_alamat(gabung["alamat_teks"])
    gabung = gabung[gabung["alamat_normal"].str.len() > 0]

    # Identitas alamat adalah bentuk ternormalisasinya, bukan node ICIJ-nya.
    # Dua node ICIJ berbeda yang menuliskan alamat yang sama adalah satu tempat,
    # dan justru itulah keterkaitan yang ingin ditemukan.
    dim = (
        gabung.groupby("alamat_normal", as_index=False)
        .agg(
            alamat_teks=("alamat_teks", "first"),
            negara=("negara", "first"),
            src_icij_node_id=("src_icij_node_id", "first"),
            jumlah_debitur=("cif_sk", "nunique"),
        )
        .sort_values("alamat_normal")
        .reset_index(drop=True)
    )
    dim.insert(0, "alamat_id", range(1, len(dim) + 1))
    dim["is_alamat_agen"] = dim["jumlah_debitur"] > MAKS_DEBITUR_PER_ALAMAT

    peta_id = dim.set_index("alamat_normal")["alamat_id"]
    jembatan = pd.DataFrame(
        {
            "cif_sk": gabung["cif_sk"].astype("int64"),
            "alamat_id": gabung["alamat_normal"].map(peta_id).astype("int64"),
            "jenis_alamat": "operasional",
            "valid_from": pd.to_datetime(gabung["valid_from"]).fillna(
                pd.Timestamp(settings.tanggal_default_edge)
            ),
            "valid_to": pd.to_datetime(gabung["valid_to"]),
            "sumber": gabung["sumber"].astype("string"),
        }
    ).drop_duplicates(subset=["cif_sk", "alamat_id"], keep="first")

    LOG.info(
        "alamat: %s alamat unik, %s pasangan cif-alamat, %s alamat agen (>%s debitur)",
        len(dim),
        len(jembatan),
        int(dim["is_alamat_agen"].sum()),
        MAKS_DEBITUR_PER_ALAMAT,
    )
    return {
        "dim_alamat": dim[
            [
                "alamat_id",
                "alamat_teks",
                "alamat_normal",
                "negara",
                "jumlah_debitur",
                "is_alamat_agen",
                "src_icij_node_id",
            ]
        ],
        "fact_alamat_debitur": jembatan.reset_index(drop=True),
    }


def pasangan_seralamat(
    dim_alamat: pd.DataFrame, jembatan: pd.DataFrame
) -> pd.DataFrame:
    """Pasangan debitur yang berbagi alamat, sumber edge berbagi_atribut.

    Alamat agen registrasi dibuang: ratusan badan hukum yang memakai satu kantor
    notaris bukan satu grup usaha, dan kalau ikut dipasangkan ia menghasilkan
    klik raksasa yang membuat seluruh metrik graf tidak berarti.
    """
    if jembatan.empty:
        return pd.DataFrame(columns=["cif_a", "cif_b", "alamat_id", "valid_from"])

    layak = set(dim_alamat[~dim_alamat["is_alamat_agen"]]["alamat_id"])
    sub = jembatan[jembatan["alamat_id"].isin(layak)]

    pasangan = []
    for alamat_id, grup in sub.groupby("alamat_id"):
        anggota = sorted(set(grup["cif_sk"].astype(int)))
        if len(anggota) < 2:
            continue
        tanggal = grup["valid_from"].min()
        for i in range(len(anggota)):
            for j in range(i + 1, len(anggota)):
                pasangan.append(
                    {
                        "cif_a": anggota[i],
                        "cif_b": anggota[j],
                        "alamat_id": int(alamat_id),
                        "valid_from": tanggal,
                    }
                )
    return pd.DataFrame(pasangan, columns=["cif_a", "cif_b", "alamat_id", "valid_from"])

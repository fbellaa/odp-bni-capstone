"""Lapisan graf sungguhan: tabel graf `data/gold` untuk halaman Struktur Grup.

Berbeda dari `lib/dummy_data.py` yang mengarang simpul dan relasi dari RNG
bersalt, modul ini membaca graf yang benar-benar dibangun pipeline:

    gold_graph_nodes        38.731 simpul  (pihak, counterparty, badan hukum, alamat)
    gold_graph_edges        93.506 edge    (menjabat_di, memasok, memiliki, ...)
    graph_snapshot_bulanan  metrik PIT per simpul per akhir bulan
    fact_kepemilikan        porsi kepemilikan bertanggal + flag pengendali efektif

Bentuk keluarannya sengaja dibuat sama dengan `dummy_data` supaya
`tampilan.plot_graf` dan `tampilan.plot_kepemilikan` bisa dipakai tanpa diubah:
`subgraf_ego` mengembalikan nodes/edges dengan kolom yang sama, dan halaman
cukup menukar sumbernya.

Seperti `model_nyata`, semua fungsi berat dibungkus cache Streamlit dan
mengembalikan `None` bila tabelnya tidak ada - halaman menyebut apa yang kurang,
bukan melempar traceback.

BATAS DATA YANG HARUS DISEBUT DI ANTARMUKA
------------------------------------------
1. KEPEMILIKAN HANYA SATU LAPIS. Tidak ada satu pun dari 23.219 `pihak` yang
   sekaligus menjadi debitur (irisan `dim_pihak.src_icij_node_id` dengan
   `map_entitas_graf.src_icij_node_id` = 0). Relasi `memiliki` karena itu selalu
   pihak -> badan hukum dan berhenti di situ. Rantai kepemilikan berlapis 2-5
   tingkat tidak punya padanan pada data nyata, dan `kepemilikan_langsung()` di
   sini TIDAK mengarang lapisan tambahan.

2. PORSI KEPEMILIKAN ADALAH SINTESIS, BUKAN DATA ICIJ. ICIJ tidak memuat
   persentase saham sama sekali - yang nyata di sana hanya FAKTA relasi "A
   pemilik B" beserta rentang tanggalnya. Angka porsinya diundi pipeline, lalu
   dinormalkan per debitur PER SEGMEN WAKTU (`struktur._kapitalisasi_pit`),
   sehingga `porsi_total` berjumlah tepat 1,0 pada setiap tanggal snapshot.
   Konsekuensinya satu relasi ICIJ bisa muncul sebagai beberapa baris bertanggal
   (18.845 relasi -> 54.153 segmen); baris yang berasal dari relasi yang sama
   berbagi `rel_id`. Yang boleh dibaca dari angka ini hanya URUTAN dan ukuran
   relatif antarpemilik, bukan besaran permodalan sungguhan - dan itu wajib
   disebut di antarmuka. Kolom `porsi_total` tetap disediakan sebagai jaring
   pengaman kalau normalisasinya suatu saat jebol.

3. BOBOT EDGE BEDA SATUAN per rel_type - rupiah pada `memasok`, porsi pada
   `memiliki`, penanda 1,0 pada sisanya. Pemangkasan subgraf memakai peringkat
   persentil DI DALAM tiap rel_type (`prioritas`), bukan bobot mentah, supaya
   satu transfer besar tidak menyingkirkan seluruh relasi kepemilikan.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

AKAR = Path(__file__).resolve().parents[3]
DIR_GOLD = AKAR / "data" / "gold"

# `pipelines` hidup di AKAR, bukan di app/ui. `streamlit run app/ui/Copilot_Pengajuan.py`
# hanya menaruh direktori skrip di sys.path - bukan direktori kerja - sehingga
# `import pipelines` di `resolusi_calon()` gagal dengan ModuleNotFoundError.
# Dua cara peluncuran lain kebetulan menutupinya: `python -m streamlit ...` dari
# akar menambahkan cwd, dan Dockerfile menyetel PYTHONPATH=/opt/banking-copilot.
# Menambahkannya di sini membuat halaman jalan pada ketiganya.
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))

# Galat pemuatan disimpan, bukan ditelan: halaman perlu bisa menyebut alasannya.
GALAT_MUAT: dict[str, str] = {}

TABEL_WAJIB = ("gold_graph_nodes", "gold_graph_edges")
TABEL_OPSIONAL = (
    "graph_snapshot_bulanan", "fact_kepemilikan", "dim_pihak", "dim_debitur",
    "dim_counterparty", "dim_alamat", "dim_grup_usaha", "fact_pengajuan",
    "fact_default",
)

# node_type gold -> label tipe yang dikenal PALET_TIPE / SIMBOL_TIPE di tampilan.py.
# `alamat` sengaja dipetakan ke "Atribut berbagi": alamat operasional bersama
# memang atribut yang dipakai bersama, dan paletnya sudah menyediakan warnanya.
TIPE_TAMPIL = {
    "badan_hukum": "Badan hukum",
    "counterparty": "Counterparty",
    "alamat": "Atribut berbagi",
    "pihak": "Pengurus",          # ditimpa jadi "Pemilik manfaat" bila punya edge `memiliki`
}

AWALAN_ID = {
    "badan_hukum": "BH", "counterparty": "CP", "alamat": "AL", "pihak": "PH",
}

# Ambang pemilik manfaat yang lazim dipakai penelaahan APU-PPT.
AMBANG_PEMILIK_MANFAAT = 0.25


def _berkas(nama: str) -> Path:
    return DIR_GOLD / f"{nama}.parquet"


def tersedia() -> bool:
    """True bila dua tabel graf inti ada. Dipanggil halaman sebelum apa pun."""
    return all(_berkas(t).exists() for t in TABEL_WAJIB)


def tabel_hilang() -> list[str]:
    """Daftar tabel yang tidak ditemukan, untuk pesan di antarmuka."""
    return [t for t in TABEL_WAJIB + TABEL_OPSIONAL if not _berkas(t).exists()]


@st.cache_data(show_spinner=False)
def _baca(nama: str, kolom: tuple[str, ...] | None = None) -> pd.DataFrame | None:
    berkas = _berkas(nama)
    if not berkas.exists():
        GALAT_MUAT[nama] = f"{berkas} tidak ada"
        return None
    try:
        return pd.read_parquet(berkas, columns=list(kolom) if kolom else None)
    except Exception as exc:                      # noqa: BLE001 - dilaporkan ke UI
        GALAT_MUAT[nama] = f"{type(exc).__name__}: {exc}"
        return None


# --------------------------------------------------------------- simpul & nama
@st.cache_data(show_spinner=False)
def _nama_simpul() -> pd.DataFrame | None:
    """Satu tabel (node_type, ref_id) -> nama & peran, dari tiap dimensi sumber."""
    bagian: list[pd.DataFrame] = []

    deb = _baca("dim_debitur", ("cif_sk", "nama_badan_hukum", "sektor_deskripsi", "is_current"))
    if deb is not None:
        if "is_current" in deb.columns:
            deb = deb[deb["is_current"].fillna(True)]
        bagian.append(pd.DataFrame({
            "node_type": "badan_hukum", "ref_id": deb["cif_sk"].to_numpy(),
            "nama": deb["nama_badan_hukum"].to_numpy(),
            "peran": deb["sektor_deskripsi"].to_numpy(),
        }))

    pih = _baca("dim_pihak", ("pihak_id", "nama", "tipe"))
    if pih is not None:
        bagian.append(pd.DataFrame({
            "node_type": "pihak", "ref_id": pih["pihak_id"].to_numpy(),
            "nama": pih["nama"].to_numpy(),
            "peran": pih["tipe"].map({"individu": "Perorangan", "badan": "Badan"}).to_numpy(),
        }))

    cp = _baca("dim_counterparty", ("cp_id", "nama", "peran"))
    if cp is not None:
        bagian.append(pd.DataFrame({
            "node_type": "counterparty", "ref_id": cp["cp_id"].to_numpy(),
            "nama": cp["nama"].to_numpy(), "peran": cp["peran"].to_numpy(),
        }))

    al = _baca("dim_alamat", ("alamat_id", "alamat_teks", "jumlah_debitur"))
    if al is not None:
        bagian.append(pd.DataFrame({
            "node_type": "alamat", "ref_id": al["alamat_id"].to_numpy(),
            "nama": al["alamat_teks"].to_numpy(),
            "peran": al["jumlah_debitur"].map(lambda n: f"Dipakai {int(n)} debitur").to_numpy(),
        }))

    if not bagian:
        return None
    gab = pd.concat(bagian, ignore_index=True)
    gab["ref_id"] = gab["ref_id"].astype("int64")
    gab["nama"] = gab["nama"].fillna("(tanpa nama)").astype(str)
    gab["peran"] = gab["peran"].fillna("-").astype(str)
    return gab


@st.cache_data(show_spinner=False)
def simpul() -> pd.DataFrame | None:
    """GOLD_GRAPH_NODES + nama, peran, dan id tampilan yang stabil."""
    n = _baca("gold_graph_nodes")
    if n is None:
        return None
    n = n.copy()
    n["ref_id"] = n["ref_id"].astype("int64")
    nama = _nama_simpul()
    if nama is not None:
        n = n.merge(nama, on=["node_type", "ref_id"], how="left")
    else:
        n["nama"], n["peran"] = "(tanpa nama)", "-"
    n["nama"] = n["nama"].fillna("(tanpa nama)")
    n["peran"] = n["peran"].fillna("-")
    # id tampilan: dibaca manusia di hover, tetap unik karena membawa ref_id.
    n["id_tampil"] = (
        n["node_type"].map(AWALAN_ID).fillna("NA") + "-"
        + n["ref_id"].astype(str) + " " + n["nama"].str.slice(0, 34)
    )
    return n


@st.cache_data(show_spinner=False)
def edge() -> pd.DataFrame | None:
    """GOLD_GRAPH_EDGES + `prioritas`: peringkat persentil bobot DI DALAM rel_type.

    Bobot mentah tidak sebanding antar rel_type (rupiah vs porsi vs penanda),
    jadi pemangkasan subgraf memakai kolom ini. Penanda konstan 1,0 memperoleh
    prioritas 0,5 - netral, tidak otomatis kalah dari transfer terkecil.
    """
    e = _baca("gold_graph_edges")
    if e is None:
        return None
    e = e.copy()
    e["prioritas"] = (
        e.groupby("rel_type")["bobot"].rank(pct=True, method="average").fillna(0.5)
    )
    return e


# ------------------------------------------------------------------- snapshot
@st.cache_data(show_spinner=False)
def snapshot_tersedia() -> list[pd.Timestamp]:
    """Akhir bulan yang punya baris di GRAPH_SNAPSHOT_BULANAN, terbaru dulu."""
    s = _baca("graph_snapshot_bulanan", ("snapshot_date",))
    if s is None:
        return []
    return sorted(pd.to_datetime(pd.Series(s["snapshot_date"].unique())), reverse=True)


KOLOM_METRIK = ("node_id", "snapshot_date", "degree", "weighted_degree", "pagerank",
                "betweenness", "community_id", "community_default_rate")


@st.cache_data(show_spinner=False)
def metrik_snapshot(tanggal: pd.Timestamp | None) -> pd.DataFrame | None:
    """Metrik PIT per simpul pada satu snapshot - dibaca, tidak dihitung ulang.

    Hanya delapan kolom yang dibaca dari 24 yang tersedia: enam belas kolom
    `emb_*` melayani uji ablasi model, bukan halaman ini, dan memuatnya membuat
    tiap pemuatan halaman menyeret 1,39 juta baris x 24 kolom tanpa guna.
    """
    if tanggal is None:
        return None
    s = _baca("graph_snapshot_bulanan", KOLOM_METRIK)
    if s is None:
        return None
    sel = s[pd.to_datetime(s["snapshot_date"]) == pd.Timestamp(tanggal)]
    return sel.drop(columns=["snapshot_date"]).reset_index(drop=True)


def _edge_aktif(e: pd.DataFrame, pada: pd.Timestamp | None) -> pd.DataFrame:
    """Edge yang berlaku pada satu tanggal. `pada=None` berarti seluruh riwayat."""
    if pada is None:
        return e
    pada = pd.Timestamp(pada)
    mulai = pd.to_datetime(e["valid_from"]) <= pada
    belum_usai = e["valid_to"].isna() | (pd.to_datetime(e["valid_to"]) > pada)
    return e[mulai & belum_usai]


@st.cache_data(show_spinner=False)
def _edge_pada(
    pada: pd.Timestamp | None, rel_dipakai: tuple[str, ...] | None = None
) -> pd.DataFrame | None:
    """Edge yang berlaku pada satu tanggal, sudah disaring per rel_type."""
    e = edge()
    if e is None:
        return None
    e = _edge_aktif(e, pada)
    if rel_dipakai:
        e = e[e["rel_type"].isin(rel_dipakai)]
    return e.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _adjacency(pada: pd.Timestamp | None) -> dict[int, list[tuple[int, str]]]:
    """Peta tetangga tak berarah, terurut prioritas menurun.

    Di-cache terpisah dari `subgraf_ego` dan sengaja TIDAK bergantung pada simpul
    pusat: membangunnya butuh concat + groupby atas 187 ribu baris, dan kalau
    ikut menempel pada argumen `node_id`, tiap kali analis mengganti debitur di
    pemilih halaman ongkos itu dibayar penuh lagi (terukur 4-8 detik). Dipisah
    begini, hanya pergantian tanggal snapshot yang membayarnya.

    Juga TIDAK bergantung pada filter tipe relasi. Tiap tetangga sudah membawa
    `rel_type`-nya, jadi penyaringan bisa dikerjakan saat penelusuran - kalau
    filternya ikut jadi kunci cache, lima tipe relasi berarti 31 kombinasi yang
    masing-masing membangun ulang peta ini dari nol.
    """
    e = _edge_pada(pada, None)
    if e is None:
        return {}
    sisi = pd.concat([
        e[["src_node_id", "dst_node_id", "rel_type", "prioritas"]].rename(
            columns={"src_node_id": "dari", "dst_node_id": "ke"}),
        e[["dst_node_id", "src_node_id", "rel_type", "prioritas"]].rename(
            columns={"dst_node_id": "dari", "src_node_id": "ke"}),
    ], ignore_index=True).sort_values("prioritas", ascending=False)
    return {
        int(k): list(zip(v["ke"].astype("int64"), v["rel_type"]))
        for k, v in sisi.groupby("dari")
    }


# ------------------------------------------------------------------ pencarian
@st.cache_data(show_spinner=False)
def daftar_debitur(batas: int = 400) -> pd.DataFrame | None:
    """Debitur yang layak jadi pusat subgraf, diurut dari yang paling terhubung.

    Debitur tanpa relasi apa pun menghasilkan subgraf kosong dan hanya membuat
    pemilih halaman menyesatkan, jadi urutannya berdasarkan derajat.
    """
    n, e = simpul(), edge()
    if n is None or e is None:
        return None
    bh = n[n["node_type"] == "badan_hukum"]
    derajat = (
        pd.concat([e["src_node_id"], e["dst_node_id"]])
        .value_counts().rename_axis("node_id").reset_index(name="derajat")
    )
    keluar = bh.merge(derajat, on="node_id", how="left")
    keluar["derajat"] = keluar["derajat"].fillna(0).astype(int)

    peng = _baca("fact_pengajuan", ("application_id", "cif_sk", "tanggal_pengajuan"))
    if peng is not None:
        peng = peng.sort_values("tanggal_pengajuan").groupby("cif_sk", as_index=False).last()
        keluar = keluar.merge(peng, left_on="ref_id", right_on="cif_sk", how="left")

    kolom = [k for k in ["node_id", "ref_id", "nama", "grup_id", "derajat",
                         "application_id", "tanggal_pengajuan"] if k in keluar.columns]
    return (
        keluar.sort_values("derajat", ascending=False)
        .head(batas)[kolom].reset_index(drop=True)
    )


# -------------------------------------------------------------------- subgraf
@dataclass
class HasilSubgraf:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    dipangkas: bool          # True bila batas simpul menghentikan penelusuran
    total_tetangga: int      # tetangga langsung sebelum pemangkasan


@st.cache_data(show_spinner=False)
def subgraf_ego(
    node_id: int,
    hops: int = 2,
    batas_simpul: int = 60,
    pada: pd.Timestamp | None = None,
    rel_dipakai: tuple[str, ...] | None = None,
) -> HasilSubgraf | None:
    """Subgraf ego di sekitar satu simpul, siap dikirim ke `tampilan.plot_graf`.

    Penelusuran lebar per hop; kalau tetangga satu hop melebihi sisa anggaran
    simpul, yang diambil adalah yang `prioritas`-nya tertinggi. Nilai kembalinya
    membawa `dipangkas` supaya halaman bisa mengatakan graf ini tidak utuh -
    subgraf terpangkas yang disajikan seolah lengkap adalah cara termudah
    membuat analis salah menyimpulkan "tidak ada relasi lain".
    """
    n = simpul()
    e = _edge_pada(pada, rel_dipakai)
    if n is None or e is None:
        return None
    tetangga = _adjacency(pada)
    # Filter diterapkan pada PENELUSURAN, bukan pada gambar. Menyembunyikan
    # garis tapi menyisakan simpulnya akan menampilkan simpul melayang yang
    # sampai ke sini lewat relasi yang justru sedang disaring keluar. Yang
    # dicari analis saat memilih `menjabat_di` adalah jaringan rangkap jabatan
    # itu sendiri - siapa yang terhubung HANYA lewat lapisan tersebut.
    saring = set(rel_dipakai) if rel_dipakai else None

    def tetangga_sah(simpul_id: int) -> list[tuple[int, str]]:
        daftar = tetangga.get(simpul_id, [])
        if saring is None:
            return daftar
        return [(t, r) for t, r in daftar if r in saring]

    pusat = int(node_id)
    terpilih: dict[int, int] = {pusat: 0}
    dipangkas = False
    total_tetangga = len(tetangga_sah(pusat))

    # Anggaran dibagi per hop, bukan dilahap habis oleh hop pertama. Tanpa ini
    # satu debitur berderajat 2.337 memenuhi seluruh batas simpul dengan
    # tetangga langsung, dan subgraf "dua hop" yang tampil sebenarnya bintang
    # satu lapis - persis kesan yang paling menyesatkan bagi analis.
    sisa = batas_simpul - 1
    lapis = [pusat]
    for hop in range(1, hops + 1):
        if sisa <= 0 or not lapis:
            break
        # Hop terakhir boleh memakai sisa anggaran seluruhnya.
        jatah = sisa if hop == hops else max(1, int(sisa * 0.6))

        # Kandidat dikumpulkan per rel_type lalu diambil bergiliran, supaya
        # relasi yang jumlahnya sedikit (beralamat_di, berbagi_atribut) tidak
        # habis tersingkir oleh `menjabat_di` dan `memasok` yang berlimpah.
        per_rel: dict[str, list[int]] = {}
        for induk in lapis:
            for tujuan, rel in tetangga_sah(induk):
                if tujuan in terpilih:
                    continue
                per_rel.setdefault(rel, []).append(tujuan)

        jumlah_kandidat = sum(len(v) for v in per_rel.values())
        baru: list[int] = []
        antrean_rel = [iter(v) for v in per_rel.values()]
        while antrean_rel and len(baru) < jatah:
            for it in list(antrean_rel):
                if len(baru) >= jatah:
                    break
                for tujuan in it:
                    if tujuan not in terpilih:
                        terpilih[tujuan] = hop
                        baru.append(tujuan)
                        break
                else:
                    antrean_rel.remove(it)

        if jumlah_kandidat > len(baru):
            dipangkas = True
        sisa -= len(baru)
        lapis = baru

    ids = list(terpilih)
    sub_n = n[n["node_id"].isin(ids)].copy()
    sub_e = e[e["src_node_id"].isin(ids) & e["dst_node_id"].isin(ids)].copy()

    # `pihak` yang memegang saham ditampilkan sebagai pemilik manfaat, sisanya
    # sebagai pengurus - keduanya punya warna dan simbol sendiri di palet.
    pemilik = set(sub_e.loc[sub_e["rel_type"] == "memiliki", "src_node_id"])
    sub_n["tipe"] = sub_n["node_type"].map(TIPE_TAMPIL).fillna("Badan hukum")
    sub_n.loc[sub_n["node_id"].isin(pemilik) & (sub_n["node_type"] == "pihak"),
              "tipe"] = "Pemilik manfaat"

    sub_n["hop"] = sub_n["node_id"].map(terpilih).astype(int)

    metrik = metrik_snapshot(pada)
    if metrik is not None and not metrik.empty and "community_id" in metrik.columns:
        sub_n = sub_n.merge(metrik[["node_id", "community_id"]], on="node_id", how="left")
    if "community_id" not in sub_n.columns:
        sub_n["community_id"] = 0
    sub_n["community_id"] = sub_n["community_id"].fillna(0).astype(int)

    # PD per simpul graf tidak ada di lapisan ini; kolomnya tetap disediakan
    # karena `plot_graf` membacanya, dan NaN berarti "tidak ditampilkan".
    sub_n["pd"] = float("nan")

    grup = _baca("dim_grup_usaha", ("grup_id", "nama_grup"))
    if grup is not None:
        sub_n = sub_n.merge(grup, on="grup_id", how="left")
        sub_n["grup"] = sub_n["nama_grup"].fillna("-")
    else:
        sub_n["grup"] = "-"

    peta = dict(zip(sub_n["node_id"], sub_n["id_tampil"]))
    nodes = sub_n.rename(columns={"id_tampil": "id"})[
        ["id", "node_id", "node_type", "tipe", "hop", "community_id", "pd",
         "grup", "peran", "nama"]
    ].copy()
    nodes["label"] = nodes["nama"]
    nodes = nodes.sort_values("hop").reset_index(drop=True)

    edges = pd.DataFrame({
        "source": sub_e["src_node_id"].map(peta),
        "target": sub_e["dst_node_id"].map(peta),
        "relasi": sub_e["rel_type"].to_numpy(),
        "bobot": sub_e["bobot"].to_numpy(),
        "berarah": sub_e["berarah"].to_numpy(),
        "valid_from": sub_e["valid_from"].to_numpy(),
        "valid_to": sub_e["valid_to"].to_numpy(),
    }).dropna(subset=["source", "target"]).reset_index(drop=True)

    return HasilSubgraf(nodes, edges, dipangkas, total_tetangga)


# ---------------------------------------------------------------- kepemilikan
KOLOM_KEPEMILIKAN = [
    "tingkat", "pemilik", "jenis", "dimiliki", "porsi_langsung", "porsi_efektif",
    "pengendali_efektif", "aktivitas_usaha", "yurisdiksi", "porsi_total",
    "jumlah_pemilik", "valid_from", "valid_to",
]


@st.cache_data(show_spinner=False)
def kepemilikan_langsung(
    cif_sk: int, pada: pd.Timestamp | None = None, batas_baris: int = 15
) -> pd.DataFrame | None:
    """Pemilik tercatat satu debitur pada satu tanggal, porsi terbesar dulu.

    Namanya `langsung`, bukan `penelusuran`, karena data ini memang berhenti di
    satu lapis (lihat batas data 1 di docstring modul). `porsi_efektif` karena
    itu identik dengan `porsi_langsung`; kolomnya tetap disediakan supaya
    `tampilan.plot_kepemilikan` bisa dipakai apa adanya, dan supaya rumusnya
    tinggal diganti kalau suatu saat ada lapis kedua.

    `batas_baris` memotong tampilan, TIDAK memotong hitungan: `porsi_total` dan
    `jumlah_pemilik` selalu dihitung atas seluruh pemilik aktif. Pemotongan ini
    bukan kosmetik - CIF 3833 punya 184 pemilik tercatat pada satu tanggal, dan
    Sankey berisi 184 link tidak menyampaikan apa pun. Halaman wajib menampilkan
    `jumlah_pemilik` di samping grafiknya supaya yang terpotong tetap terlihat,
    dan menyebut berapa persen yang sedang ditampilkan.

    `porsi_total` kini berjumlah 1,0 pada setiap tanggal (lihat batas data 2 di
    docstring modul). Halaman tetap memeriksanya: kalau suatu saat tidak, itu
    tanda pipeline-nya yang rusak, bukan sifat datanya.
    """
    kep = _baca("fact_kepemilikan")
    if kep is None:
        return None
    sel = kep[kep["cif_sk"] == int(cif_sk)].copy()
    if pada is not None:
        pada = pd.Timestamp(pada)
        sel = sel[
            (pd.to_datetime(sel["valid_from"]) <= pada)
            & (sel["valid_to"].isna() | (pd.to_datetime(sel["valid_to"]) > pada))
        ]
    if sel.empty:
        return pd.DataFrame(columns=KOLOM_KEPEMILIKAN)

    pih = _baca("dim_pihak", ("pihak_id", "nama", "tipe"))
    if pih is not None:
        sel = sel.merge(pih, on="pihak_id", how="left")
    else:
        sel["nama"], sel["tipe"] = "(tanpa nama)", "individu"

    deb = _baca("dim_debitur", ("cif_sk", "nama_badan_hukum", "is_current"))
    nama_deb = f"CIF {int(cif_sk)}"
    if deb is not None:
        cocok = deb[deb["cif_sk"] == int(cif_sk)]
        if "is_current" in cocok.columns:
            kini = cocok[cocok["is_current"].fillna(True)]
            cocok = kini if len(kini) else cocok
        if len(cocok):
            nama_deb = str(cocok["nama_badan_hukum"].iat[0])

    sel = sel.sort_values("porsi_kepemilikan", ascending=False).reset_index(drop=True)
    # Dihitung SEBELUM pemotongan: yang dilaporkan halaman harus seluruh pemilik.
    porsi_total = float(sel["porsi_kepemilikan"].sum())
    jumlah_pemilik = int(len(sel))
    if batas_baris and jumlah_pemilik > batas_baris:
        sel = sel.head(batas_baris).reset_index(drop=True)

    tipe = sel["tipe"].fillna("individu")
    porsi = sel["porsi_kepemilikan"].astype(float)

    jenis = pd.Series("Pemegang saham perorangan", index=sel.index)
    jenis[tipe == "badan"] = "Badan hukum pemegang saham"
    jenis[(tipe == "individu") & (porsi >= AMBANG_PEMILIK_MANFAAT)] = (
        "Pemilik manfaat (perorangan)"
    )

    return pd.DataFrame({
        "tingkat": 1,
        "pemilik": sel["nama"].fillna("(tanpa nama)").astype(str),
        "jenis": jenis,
        "dimiliki": nama_deb,
        "porsi_langsung": porsi,
        "porsi_efektif": porsi,
        "pengendali_efektif": sel["pengendali_efektif"],
        "aktivitas_usaha": "-",
        "yurisdiksi": "-",
        "porsi_total": porsi_total,
        "jumlah_pemilik": jumlah_pemilik,
        "valid_from": sel["valid_from"],
        "valid_to": sel["valid_to"],
    })


@st.cache_data(show_spinner=False)
def ringkas_ketersediaan() -> dict:
    """Angka untuk panel 'data apa yang sedang dibaca' di halaman."""
    n, e = simpul(), edge()
    snap = snapshot_tersedia()
    return {
        "tersedia": n is not None and e is not None,
        "tabel_hilang": tabel_hilang(),
        "jumlah_simpul": 0 if n is None else int(len(n)),
        "jumlah_edge": 0 if e is None else int(len(e)),
        "per_tipe_simpul": {} if n is None else n["node_type"].value_counts().to_dict(),
        "per_tipe_relasi": {} if e is None else e["rel_type"].value_counts().to_dict(),
        "snapshot_terbaru": None if not snap else snap[0],
        "jumlah_snapshot": len(snap),
        "galat": dict(GALAT_MUAT),
    }


# ============================================================ resolusi calon
# Pengaju baru belum punya satu pun edge di GOLD_GRAPH_EDGES, jadi `subgraf_ego`
# di atas tidak bisa menjawab "calon ini terafiliasi dengan debitur mana".
# Yang menjawabnya `pipelines.graph.resolusi.telusuri_afiliasi()`, lewat tiga
# berkas yang memang sudah wajib dikumpulkan saat CDD. Bagian di bawah ini
# pembungkus tipisnya: menambah nama, eksposur berjalan, dan penanda hub - tanpa
# menyentuh logika pencocokannya.

# Urutan kekuatan bukti. Identitas (pengurus, pemilik) mengalahkan lokasi, dan
# lokasi mengalahkan perilaku transaksi - sesuai catatan resolusi.py bahwa jalur
# identitas sudah sahih pada hari pengajuan sedangkan jalur transaksi baru
# berguna setelah ada riwayat mutasi.
BOBOT_DASAR = {
    "pengurus_bersama": 5,
    "pemilik_bersama": 4,
    "alamat_persis": 3,
    "alamat_mirip": 2,
    "counterparty_bersama": 1,
}

LABEL_DASAR = {
    "pengurus_bersama": "Pengurus bersama",
    "pemilik_bersama": "Pemilik bersama",
    "alamat_persis": "Alamat sama persis",
    "alamat_mirip": "Alamat mirip",
    "counterparty_bersama": "Lawan transaksi sama",
}

LABEL_JALUR = {
    "alamat": "Dokumen domisili usaha",
    "pengurus": "Akta / data kepemilikan",
    "rekening_koran": "Rekening koran",
}

# Di atas ambang ini, satu pihak atau satu alamat terhubung ke begitu banyak
# debitur sehingga keterkaitannya berhenti bermakna sebagai afiliasi spesifik.
# resolusi.py sudah membuang alamat agen registrasi lewat `is_alamat_agen`,
# tetapi TIDAK punya penyaring setara untuk pihak berderajat tinggi - dan
# sintesis grup memang memakai hub pengurus ICIJ (lihat
# `map_entitas_graf.metode_pemetaan` = "klaster_hub_pengurus_icij"), sehingga
# satu pihak bisa muncul pada 641 debitur sekaligus. Barisnya tidak dibuang,
# hanya ditandai dan diturunkan peringkatnya: menyembunyikannya berarti analis
# tidak pernah tahu kenapa daftarnya panjang.
AMBANG_HUB = 25


@dataclass
class HasilResolusiUI:
    """Kandidat afiliasi calon nasabah, siap ditabelkan halaman."""

    tanggal: pd.Timestamp
    tabel: pd.DataFrame
    jalur: list[dict]              # nama, dipakai, keterangan - per jalur dokumen
    jumlah_kandidat: int
    perlu_telaah: bool
    dipangkas: bool
    galat: str | None = None

    @property
    def ada_gagal_bayar(self) -> int:
        if self.tabel.empty or "sudah_gagal_bayar" not in self.tabel:
            return 0
        return int(self.tabel["sudah_gagal_bayar"].sum())


@st.cache_data(show_spinner=False)
def _eksposur_berjalan(tanggal: pd.Timestamp) -> pd.DataFrame | None:
    """Outstanding per debitur atas fasilitas yang masih hidup pada `tanggal`."""
    fas = _baca("fact_fasilitas", ("cif_sk", "outstanding_rp", "plafon_rp",
                                   "tanggal_pencairan", "tanggal_jatuh_tempo"))
    if fas is None:
        return None
    tanggal = pd.Timestamp(tanggal)
    cair = pd.to_datetime(fas["tanggal_pencairan"]) <= tanggal
    belum_lunas = (
        fas["tanggal_jatuh_tempo"].isna()
        | (pd.to_datetime(fas["tanggal_jatuh_tempo"]) > tanggal)
    )
    hidup = fas[cair & belum_lunas]
    return (
        hidup.groupby("cif_sk", as_index=False)
        .agg(eksposur_rp=("outstanding_rp", "sum"), jumlah_fasilitas=("outstanding_rp", "size"))
    )


@st.cache_data(show_spinner=False)
def _derajat_pihak(tanggal: pd.Timestamp) -> pd.Series:
    """Berapa debitur yang terhubung ke tiap pihak pada `tanggal` - pendeteksi hub."""
    bagian = []
    for tabel in ("fact_kepengurusan", "fact_kepemilikan"):
        df = _baca(tabel, ("pihak_id", "cif_sk", "valid_from", "valid_to"))
        if df is None:
            continue
        t = pd.Timestamp(tanggal)
        aktif = df[
            (pd.to_datetime(df["valid_from"]) <= t)
            & (df["valid_to"].isna() | (pd.to_datetime(df["valid_to"]) > t))
        ]
        bagian.append(aktif[["pihak_id", "cif_sk"]])
    if not bagian:
        return pd.Series(dtype="int64")
    gab = pd.concat(bagian, ignore_index=True).drop_duplicates()
    return gab.groupby("pihak_id")["cif_sk"].nunique()


def _bukti(baris: pd.Series) -> str:
    """Kalimat bukti yang bisa dikutip analis, sesuai jalur yang memunculkannya."""
    dasar = baris.get("dasar")
    if dasar in ("pengurus_bersama", "pemilik_bersama"):
        nama = baris.get("nama_pihak") or "(tanpa nama)"
        masukan = baris.get("nama_input")
        cocok = f" - cocok dengan {masukan}" if masukan and masukan != nama else ""
        return f"{nama}{cocok}"
    if dasar in ("alamat_persis", "alamat_mirip"):
        return str(baris.get("alamat_teks") or "-")
    if dasar == "counterparty_bersama":
        n = baris.get("jumlah_transfer")
        rek = baris.get("rekening_lawan") or "-"
        return f"{rek} ({int(n)} transfer)" if pd.notna(n) else str(rek)
    return "-"


@st.cache_data(show_spinner=False)
def resolusi_calon(
    tanggal: pd.Timestamp | str,
    alamat_operasional: str | None = None,
    nama_pengurus: tuple[str, ...] = (),
    rekening_lawan: tuple[str, ...] = (),
    batas_baris: int = 40,
) -> HasilResolusiUI:
    """Debitur eksisting yang terkait calon pengaju, beserta buktinya.

    Pembungkus `pipelines.graph.resolusi.telusuri_afiliasi()` - seluruh
    pencocokan dan aturan titik-waktu tetap milik modul itu. Yang ditambahkan di
    sini hanya yang dibutuhkan layar: nama debitur, eksposur berjalan, status
    gagal bayar, penanda hub, dan peringkat.

    Keluarannya BUKAN skor risiko. Ia daftar yang layak diperiksa analis - sama
    seperti yang ditegaskan docstring `telusuri_afiliasi`. `perlu_telaah`
    memicu KKK-13.6, dan itu ambang kebijakan, bukan ambang model.

    Status tiap jalur dikembalikan utuh di `jalur`, termasuk sebab kosongnya.
    Halaman WAJIB menampilkannya: "tidak ada kecocokan" dan "dokumen tidak
    disertakan" tidak boleh sama-sama tampil sebagai "tidak ada afiliasi".
    """
    tanggal = pd.Timestamp(tanggal)
    kosong = pd.DataFrame(columns=[
        "cif_sk", "nama_debitur", "grup", "dasar_utama", "semua_dasar", "bukti",
        "skor", "sudah_gagal_bayar", "eksposur_rp", "jumlah_fasilitas",
        "ukuran_hub", "hub",
    ])

    try:
        from pipelines.graph.resolusi import telusuri_afiliasi
    except Exception as exc:                      # noqa: BLE001 - dilaporkan ke UI
        return HasilResolusiUI(
            tanggal, kosong, [], 0, False, False,
            galat=f"modul resolusi tidak dapat dimuat - {type(exc).__name__}: {exc}",
        )

    try:
        hasil = telusuri_afiliasi(
            tanggal,
            alamat_operasional=alamat_operasional or None,
            nama_pengurus=list(nama_pengurus),
            rekening_lawan=list(rekening_lawan),
        )
    except Exception as exc:                      # noqa: BLE001 - dilaporkan ke UI
        return HasilResolusiUI(
            tanggal, kosong, [], 0, False, False,
            galat=f"{type(exc).__name__}: {exc}",
        )

    # Status per jalur, apa adanya dari HasilResolusi - termasuk sebab kosongnya.
    jalur = [
        {"nama": LABEL_JALUR.get(j, j), "kunci": j, "dipakai": True,
         "keterangan": "ada kecocokan"}
        for j in hasil.jalur_terpakai
    ]
    for baris in hasil.jalur_kosong:
        kunci, _, sebab = baris.partition(" (")
        jalur.append({
            "nama": LABEL_JALUR.get(kunci.strip(), kunci.strip()),
            "kunci": kunci.strip(),
            "dipakai": False,
            "keterangan": sebab.rstrip(")") or "kosong",
        })

    if hasil.kandidat.empty:
        return HasilResolusiUI(tanggal, kosong, jalur, 0, hasil.perlu_telaah, False)

    k = hasil.kandidat.copy()
    for kolom in ("nama_pihak", "nama_input", "alamat_teks", "alamat_id",
                  "rekening_lawan", "jumlah_transfer", "pihak_id", "skor"):
        if kolom not in k.columns:
            k[kolom] = pd.NA
    k["bobot"] = k["dasar"].map(BOBOT_DASAR).fillna(0).astype(int)
    k["bukti"] = k.apply(_bukti, axis=1)

    # Ukuran hub: berapa debitur yang menempel pada objek bukti yang sama.
    derajat = _derajat_pihak(tanggal)
    ukuran = pd.Series(pd.NA, index=k.index, dtype="Float64")
    pihak_ada = k["pihak_id"].notna()
    if pihak_ada.any() and len(derajat):
        ukuran[pihak_ada] = (
            k.loc[pihak_ada, "pihak_id"].astype("int64").map(derajat).astype("Float64")
        )
    alamat = _baca("dim_alamat", ("alamat_id", "jumlah_debitur"))
    if alamat is not None:
        peta_al = alamat.set_index("alamat_id")["jumlah_debitur"]
        al_ada = k["alamat_id"].notna()
        if al_ada.any():
            ukuran[al_ada] = k.loc[al_ada, "alamat_id"].map(peta_al).astype("Float64")
    k["ukuran_hub"] = ukuran

    # Satu baris per debitur: bukti terkuat jadi wakil, sisanya diringkas.
    k = k.sort_values(["bobot", "skor"], ascending=False)
    utama = k.drop_duplicates(subset=["cif_sk"], keep="first").set_index("cif_sk")
    semua = k.groupby("cif_sk")["dasar"].apply(
        lambda s: ", ".join(LABEL_DASAR.get(d, d) for d in dict.fromkeys(s))
    )
    gagal = k.groupby("cif_sk")["afiliasi_sudah_gagal_bayar"].any()

    tabel = pd.DataFrame({
        "cif_sk": utama.index.astype("int64"),
        "dasar_utama": utama["dasar"].map(lambda d: LABEL_DASAR.get(d, d)).to_numpy(),
        "bukti": utama["bukti"].to_numpy(),
        "skor": pd.to_numeric(utama["skor"], errors="coerce").to_numpy(),
        "bobot": utama["bobot"].to_numpy(),
        "ukuran_hub": pd.to_numeric(utama["ukuran_hub"], errors="coerce").to_numpy(),
        "semua_dasar": semua.reindex(utama.index).to_numpy(),
        "sudah_gagal_bayar": gagal.reindex(utama.index).fillna(False).to_numpy(),
    })
    tabel["hub"] = tabel["ukuran_hub"].fillna(0) >= AMBANG_HUB

    deb = _baca("dim_debitur", ("cif_sk", "nama_badan_hukum", "grup_id", "is_current"))
    if deb is not None:
        if "is_current" in deb.columns:
            deb = deb[deb["is_current"].fillna(True)]
            deb = deb.drop(columns=["is_current"])
        tabel = tabel.merge(deb, on="cif_sk", how="left")
        tabel = tabel.rename(columns={"nama_badan_hukum": "nama_debitur"})
    else:
        tabel["nama_debitur"], tabel["grup_id"] = pd.NA, pd.NA

    grup = _baca("dim_grup_usaha", ("grup_id", "nama_grup"))
    if grup is not None:
        tabel = tabel.merge(grup, on="grup_id", how="left")
        tabel["grup"] = tabel["nama_grup"].fillna("-")
    else:
        tabel["grup"] = "-"

    eks = _eksposur_berjalan(tanggal)
    if eks is not None:
        tabel = tabel.merge(eks, on="cif_sk", how="left")
    for kolom in ("eksposur_rp", "jumlah_fasilitas"):
        if kolom not in tabel.columns:
            tabel[kolom] = 0.0
        tabel[kolom] = tabel[kolom].fillna(0)

    # Peringkat: bukti hub turun ke bawah, gagal bayar naik ke atas, lalu
    # kekuatan bukti dan eksposur. Hub tidak dibuang - hanya tidak boleh
    # menghalangi nama-nama yang benar-benar spesifik.
    tabel = tabel.sort_values(
        ["hub", "sudah_gagal_bayar", "bobot", "eksposur_rp", "skor"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    jumlah = int(len(tabel))
    dipangkas = bool(batas_baris and jumlah > batas_baris)
    if dipangkas:
        tabel = tabel.head(batas_baris)

    kolom_akhir = ["cif_sk", "nama_debitur", "grup", "dasar_utama", "semua_dasar",
                   "bukti", "skor", "sudah_gagal_bayar", "eksposur_rp",
                   "jumlah_fasilitas", "ukuran_hub", "hub"]
    tabel = tabel[[c for c in kolom_akhir if c in tabel.columns]]

    return HasilResolusiUI(
        tanggal=tanggal,
        tabel=tabel.reset_index(drop=True),
        jalur=jalur,
        jumlah_kandidat=jumlah,
        perlu_telaah=hasil.perlu_telaah,
        dipangkas=dipangkas,
    )


# Tipe simpul objek bukti, dipetakan ke palet yang sudah ada di tampilan.py.
TIPE_BUKTI = {
    "Pengurus bersama": "Pengurus",
    "Pemilik bersama": "Pemilik manfaat",
    "Alamat sama persis": "Atribut berbagi",
    "Alamat mirip": "Atribut berbagi",
    "Lawan transaksi sama": "Counterparty",
}


def graf_resolusi(
    tabel: pd.DataFrame, nama_calon: str = "Calon pengaju", batas_debitur: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Graf tripartit calon -> objek bukti -> debitur eksisting.

    Bukan subgraf ego: calon pengaju belum punya edge apa pun di graf, jadi yang
    digambar adalah hasil pencocokan. Objek bukti (nama pengurus, alamat,
    rekening lawan) sengaja dijadikan simpul tersendiri, bukan label pada garis -
    dengan begitu beberapa debitur yang berbagi objek yang sama langsung terlihat
    mengumpul pada satu titik, dan itulah bentuk yang dicari analis.

    Keluarannya memenuhi kontrak `tampilan.plot_graf`.
    """
    kolom_n = ["id", "label", "tipe", "hop", "community_id", "pd", "grup", "peran"]
    kolom_e = ["source", "target", "relasi", "bobot"]
    if tabel is None or tabel.empty:
        return pd.DataFrame(columns=kolom_n), pd.DataFrame(columns=kolom_e)

    sel = tabel.head(batas_debitur)
    nodes = [{
        "id": nama_calon, "label": nama_calon, "tipe": "Badan hukum", "hop": 0,
        "community_id": 0, "pd": float("nan"), "grup": "-", "peran": "Calon pengaju",
    }]
    edges: list[dict] = []
    bukti_dilihat: dict[str, str] = {}

    for _, r in sel.iterrows():
        dasar = str(r.get("dasar_utama") or "-")
        teks = str(r.get("bukti") or "-")
        kunci = f"{dasar}|{teks}"
        if kunci not in bukti_dilihat:
            id_bukti = f"{dasar}: {teks[:40]}"
            bukti_dilihat[kunci] = id_bukti
            nodes.append({
                "id": id_bukti, "label": teks[:40], "tipe": TIPE_BUKTI.get(dasar, "Atribut berbagi"),
                "hop": 1, "community_id": 1, "pd": float("nan"), "grup": "-",
                "peran": dasar + (" (hub)" if bool(r.get("hub")) else ""),
            })
            edges.append({
                "source": nama_calon, "target": id_bukti,
                "relasi": "berbagi_atribut", "bobot": 1.0,
            })
        id_bukti = bukti_dilihat[kunci]

        nama = str(r.get("nama_debitur") or f"CIF {r.get('cif_sk')}")
        gagal = bool(r.get("sudah_gagal_bayar"))
        nodes.append({
            "id": nama, "label": nama, "tipe": "Badan hukum", "hop": 2,
            "community_id": 2 if gagal else 3, "pd": float("nan"),
            "grup": str(r.get("grup") or "-"),
            "peran": "Debitur eksisting - sudah gagal bayar" if gagal else "Debitur eksisting",
        })
        edges.append({
            "source": id_bukti, "target": nama,
            "relasi": "berbagi_atribut", "bobot": float(r.get("eksposur_rp") or 0.0),
        })

    df_n = pd.DataFrame(nodes).drop_duplicates(subset="id").reset_index(drop=True)
    df_e = pd.DataFrame(edges)
    sah = set(df_n["id"])
    df_e = df_e[df_e["source"].isin(sah) & df_e["target"].isin(sah)].reset_index(drop=True)
    return df_n, df_e


@st.cache_data(show_spinner=False)
def eksposur_grup(grup_id: int | None, pada: pd.Timestamp | None = None) -> dict | None:
    """Eksposur gabungan satu grup terhadap BMPK, pada snapshot terdekat <= `pada`.

    FACT_EKSPOSUR_GRUP bertanggal bulanan dan tidak selalu punya baris tepat di
    tanggal yang diminta, jadi yang diambil baris terakhir yang tidak melewati
    `pada` - bukan baris terbaru begitu saja, supaya tampilan "kondisi saat
    pengajuan" tidak diam-diam memakai angka masa depan.
    """
    if grup_id is None or pd.isna(grup_id):
        return None
    f = _baca("fact_eksposur_grup")
    if f is None:
        return None
    sel = f[f["grup_id"] == int(grup_id)].copy()
    if sel.empty:
        return None
    sel["snapshot_date"] = pd.to_datetime(sel["snapshot_date"])
    if pada is not None:
        sel = sel[sel["snapshot_date"] <= pd.Timestamp(pada)]
        if sel.empty:
            return None
    baris = sel.sort_values("snapshot_date").iloc[-1]

    nama = None
    jumlah_entitas = None
    g = _baca("dim_grup_usaha", ("grup_id", "nama_grup", "jumlah_entitas"))
    if g is not None:
        cocok = g[g["grup_id"] == int(grup_id)]
        if len(cocok):
            nama = str(cocok["nama_grup"].iat[0])
            jumlah_entitas = int(cocok["jumlah_entitas"].iat[0])

    return {
        "grup_id": int(grup_id),
        "nama_grup": nama or f"Grup {int(grup_id)}",
        "jumlah_entitas": jumlah_entitas,
        "snapshot_date": baris["snapshot_date"],
        "total_eksposur_rp": float(baris["total_eksposur_rp"]),
        "batas_bmpk_rp": float(baris["batas_bmpk_rp"]),
        "group_exposure_share": float(baris["group_exposure_share"]),
        "sisa_ruang_rp": float(baris["sisa_ruang_rp"]),
    }

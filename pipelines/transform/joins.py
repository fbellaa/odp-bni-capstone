"""Langkah 1-5 rencana data: menjahit tujuh dataset publik pada satu CIF sintetis.

PERINGATAN YANG WAJIB IKUT KE README DAN DOKUMENTASI MODEL:
join di bawah ini TIDAK ADA DI DUNIA NYATA. Baris rasio Taiwan tidak punya
hubungan apa pun dengan akun di dataset AML maupun dengan simpul ICIJ. Yang
dilakukan modul ini adalah MENEMPELKAN mereka pada satu cif sintetis. Sinyal dan
topologinya nyata; keterkaitan antar sumbernya tidak.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipelines.config import settings
from pipelines.utils import kuantil_bucket, read_table, write_table

LOG = logging.getLogger("pipelines.joins")

# Rentang jumlah entitas per "hub" ICIJ yang dianggap satu grup usaha.
MIN_ENTITAS_PER_HUB = 2
MAKS_ENTITAS_PER_HUB = 30
# Alamat dengan ratusan entitas adalah alamat agen registrasi, bukan grup.
MAKS_ENTITAS_PER_ALAMAT = 20


class UnionFind:
    def __init__(self) -> None:
        self.induk: dict[int, int] = {}

    def cari(self, x: int) -> int:
        self.induk.setdefault(x, x)
        while self.induk[x] != x:
            self.induk[x] = self.induk[self.induk[x]]
            x = self.induk[x]
        return x

    def gabung(self, a: int, b: int) -> None:
        ra, rb = self.cari(a), self.cari(b)
        if ra != rb:
            self.induk[rb] = ra


# ---------------------------------------------------- langkah 1: pilih panel
def pilih_panel_debitur(rng: np.random.Generator) -> pd.DataFrame:
    """Ambil n_debitur perusahaan dengan panel_years tahun berturut-turut."""
    panel = read_table("silver", "sl_us_panel")
    panel = panel[
        (panel["total_assets"] > 0)
        & (panel["total_revenue"] > 0)
        & panel["der"].notna()
        & panel["icr"].notna()
    ].copy()

    n = settings.panel_years
    panel = panel.sort_values(["company_name", "year"])
    grup = panel.groupby("company_name")
    # Hanya perusahaan yang punya n tahun terakhir berturut-turut.
    layak = grup["year"].agg(["max", "count"])
    layak = layak[layak["count"] >= n]
    panel = panel[panel["company_name"].isin(layak.index)]
    panel = panel.merge(
        layak["max"].rename("tahun_maks"), left_on="company_name", right_index=True
    )
    panel = panel[panel["year"] > panel["tahun_maks"] - n]
    hitung = panel.groupby("company_name")["year"].nunique()
    berturut = hitung[hitung == n].index
    panel = panel[panel["company_name"].isin(berturut)].copy()

    perusahaan = np.sort(panel["company_name"].unique())
    jumlah = min(settings.n_debitur, len(perusahaan))
    terpilih = rng.choice(perusahaan, size=jumlah, replace=False)
    panel = panel[panel["company_name"].isin(set(terpilih))].copy()

    # Bagi ke dua angkatan. Buku lama mengajukan lebih dulu dan menghasilkan
    # riwayat gagal bayar; buku baru mengajukan setelah riwayat itu ada, jadi
    # fitur penularan graf punya sesuatu untuk dilihat pada snapshot-nya.
    urut = np.sort(terpilih)
    acak = rng.permutation(len(urut))
    batas = int(round(settings.porsi_buku_lama * len(urut)))
    angkatan = pd.Series(
        np.where(acak < batas, "buku_lama", "buku_baru"), index=urut, name="angkatan"
    )
    panel["angkatan"] = panel["company_name"].map(angkatan)

    # Tahun buku dipetakan supaya panel berakhir di tahun buku angkatannya.
    panel["urutan_tahun"] = panel.groupby("company_name")["year"].rank(method="dense").astype(int)
    akhir = panel["angkatan"].map(
        {k: v["tahun_buku_terakhir"] for k, v in settings.angkatan.items()}
    )
    panel["tahun_buku"] = akhir - (n - panel["urutan_tahun"])
    panel["is_tahun_terakhir"] = panel["urutan_tahun"] == n

    LOG.info(
        "panel terpilih: %s perusahaan x %s tahun = %s firm-year | angkatan: %s",
        panel["company_name"].nunique(),
        n,
        len(panel),
        panel.drop_duplicates("company_name")["angkatan"].value_counts().to_dict(),
    )
    return panel


def terbitkan_cif(panel: pd.DataFrame) -> pd.DataFrame:
    """Terbitkan CIF-000001... untuk tiap perusahaan panel."""
    perusahaan = pd.DataFrame({"company_name": np.sort(panel["company_name"].unique())})
    perusahaan["cif_sk"] = np.arange(1, len(perusahaan) + 1, dtype="int64")
    perusahaan["cif"] = "CIF-" + perusahaan["cif_sk"].astype(str).str.zfill(6)

    tahun_akhir = panel[panel["is_tahun_terakhir"]]
    perusahaan = perusahaan.merge(
        tahun_akhir[
            [
                "company_name",
                "angkatan",
                "der",
                "roa",
                "icr",
                "debt_to_ebitda",
                "total_revenue",
                "total_assets",
                "status_label",
                "label_default",
            ]
        ],
        on="company_name",
        how="left",
    )
    # Label debitur = status pada tahun buku terakhir panel (NYATA).
    perusahaan = perusahaan.rename(columns={"label_default": "label_default_debitur"})
    return perusahaan


# ------------------------------------------- langkah 2: pencocokan rasio Taiwan
def cocokkan_taiwan(debitur: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Pencocokan longgar berbasis kuintil, bukan nilai - hindari korelasi palsu."""
    taiwan = read_table("silver", "sl_taiwan_ratio")

    debitur = debitur.copy()
    debitur["bucket_der"] = kuantil_bucket(debitur["der"], 5)
    debitur["bucket_roa"] = kuantil_bucket(debitur["roa"], 5)
    debitur["kunci_match"] = (
        debitur["label_default_debitur"].astype(int).astype(str)
        + "-"
        + debitur["bucket_der"].astype(str)
        + "-"
        + debitur["bucket_roa"].astype(str)
    )

    kandidat_per_kunci = taiwan.groupby("kunci_match")["taiwan_row_id"].apply(np.array).to_dict()
    kandidat_per_label = (
        taiwan.groupby("label_default_taiwan")["taiwan_row_id"].apply(np.array).to_dict()
    )
    semua = taiwan["taiwan_row_id"].to_numpy()

    dipilih, metode = [], []
    for kunci, label in zip(debitur["kunci_match"], debitur["label_default_debitur"]):
        opsi = kandidat_per_kunci.get(kunci)
        cara = "label+kuintil_der+kuintil_roa"
        if opsi is None or len(opsi) == 0:
            opsi = kandidat_per_label.get(int(label))
            cara = "label_saja"
        if opsi is None or len(opsi) == 0:
            opsi, cara = semua, "acak"
        dipilih.append(int(rng.choice(opsi)))
        metode.append(cara)

    debitur["taiwan_row_id"] = dipilih
    debitur["metode_match_taiwan"] = metode
    LOG.info("pencocokan Taiwan: %s", pd.Series(metode).value_counts().to_dict())
    return debitur


# ------------------------------------------- langkah 3: pemetaan simpul ICIJ
def petakan_icij(debitur: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Petakan tiap cif ke satu entitas ICIJ, lewat klaster hub pengurus/alamat.

    Sampling acak murni akan menghasilkan 3.000 entitas yang hampir tidak saling
    terhubung. Karena itu entitas diambil per klaster hub (satu pengurus yang
    memegang 2-30 badan hukum) sehingga struktur kepemilikan berlapis, rangkap
    jabatan, dan alamat yang dibagi ikut terbawa apa adanya.
    """
    rel = read_table("silver", "sl_icij_relationship")
    entity = read_table("silver", "sl_icij_entity")
    entity_ids = set(entity["node_id"].dropna().astype("int64").tolist())

    rel = rel.dropna(subset=["node_id_start", "node_id_end"]).copy()
    rel["node_id_start"] = rel["node_id_start"].astype("int64")
    rel["node_id_end"] = rel["node_id_end"].astype("int64")

    pengurus = rel[
        rel["kategori"].isin(["kepemilikan", "kepengurusan"])
        & rel["node_id_end"].isin(entity_ids)
    ]
    alamat = rel[
        (rel["kategori"] == "berbagi_atribut") & rel["node_id_start"].isin(entity_ids)
    ]

    ukuran_hub = pengurus.groupby("node_id_start")["node_id_end"].nunique()
    hub = ukuran_hub[
        ukuran_hub.between(MIN_ENTITAS_PER_HUB, MAKS_ENTITAS_PER_HUB)
    ].index.to_numpy()
    rng.shuffle(hub)

    entitas_per_hub = (
        pengurus[pengurus["node_id_start"].isin(set(hub.tolist()))]
        .groupby("node_id_start")["node_id_end"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )

    butuh = len(debitur)
    terpilih: list[int] = []
    uf = UnionFind()
    dilihat: set[int] = set()
    for h in hub:
        anggota = entitas_per_hub.get(h, [])
        baru = [e for e in anggota if e not in dilihat]
        if not baru:
            continue
        for e in anggota:
            uf.gabung(anggota[0], e)
        for e in baru:
            dilihat.add(e)
            terpilih.append(e)
        if len(terpilih) >= butuh:
            break
    terpilih = terpilih[:butuh]
    if len(terpilih) < butuh:
        raise RuntimeError(
            f"hanya {len(terpilih)} entitas ICIJ terkumpul untuk {butuh} debitur; "
            "turunkan N_DEBITUR atau longgarkan MAKS_ENTITAS_PER_HUB"
        )

    # Alamat yang dibagi ikut menggabungkan grup, selama bukan alamat agen.
    set_terpilih = set(terpilih)
    alamat_sel = alamat[alamat["node_id_start"].isin(set_terpilih)]
    per_alamat = alamat_sel.groupby("node_id_end")["node_id_start"].apply(lambda s: sorted(set(s)))
    for anggota in per_alamat:
        if MIN_ENTITAS_PER_HUB <= len(anggota) <= MAKS_ENTITAS_PER_ALAMAT:
            for e in anggota[1:]:
                uf.gabung(anggota[0], e)

    peta = pd.DataFrame({"node_id": terpilih})
    peta["akar_grup"] = [uf.cari(e) for e in terpilih]
    peta["grup_id"] = peta.groupby("akar_grup").ngroup() + 1
    peta["metode_pemetaan"] = "klaster_hub_pengurus_icij"

    debitur = debitur.sort_values("cif_sk").reset_index(drop=True)
    debitur["node_id"] = peta["node_id"].to_numpy()
    debitur["grup_id"] = peta["grup_id"].to_numpy()

    LOG.info(
        "ICIJ: %s entitas dipetakan ke %s grup usaha",
        len(peta),
        peta["grup_id"].nunique(),
    )
    return {
        "debitur": debitur,
        "map_entitas": peta[["node_id", "grup_id", "metode_pemetaan"]],
        "rel_pengurus": pengurus[pengurus["node_id_end"].isin(set_terpilih)].copy(),
        "rel_alamat": alamat_sel.copy(),
    }


# ---------------------------------------- langkah 4: pemetaan rekening giro AML
def petakan_rekening(debitur: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Tiap cif dapat 1-3 rekening giro dari subgraf AML.

    Rekening berderajat tinggi diberikan ke debitur berpenjualan besar supaya
    korelasi ukuran usaha vs aktivitas transfer tidak terbalik.
    """
    rekening = read_table("bronze", "br_aml_rekening").sort_values(
        "derajat", ascending=False, kind="stable"
    )
    tersedia = rekening["rekening"].tolist()

    debitur = debitur.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    jumlah_rek = rng.choice([1, 2, 3], size=len(debitur), p=[0.5, 0.35, 0.15])
    total_butuh = int(jumlah_rek.sum())
    if total_butuh > len(tersedia):
        raise RuntimeError(
            f"butuh {total_butuh} rekening AML tapi hanya {len(tersedia)} tersedia"
        )

    baris = []
    kursor = 0
    for cif_sk, n in zip(debitur["cif_sk"], jumlah_rek):
        for i in range(int(n)):
            baris.append(
                {
                    "rekening_id": f"REK-{cif_sk:06d}-{i + 1}",
                    "cif_sk": int(cif_sk),
                    "src_aml_account": tersedia[kursor],
                    "rekening_utama": i == 0,
                }
            )
            kursor += 1
    return pd.DataFrame(baris)


# ------------------------------------------------ langkah 5: penarikan baris SBA
def tarik_sba(debitur: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Ambil satu baris SBA per cif: CHGOFF untuk yang default, PIF untuk sisanya."""
    sba = read_table(
        "silver",
        "sl_sba",
        columns=[
            "sba_loan_nr",
            "Term",
            "revolving",
            "jenis_fasilitas",
            "dokumen_ringkas",
            "perusahaan_baru",
            "NoEmp",
            "skala_pegawai",
            "kbli_kategori",
            "kbli_deskripsi",
            "NAICS",
            "DisbursementGross",
            "GrAppv",
            "porsi_penjaminan",
            "ChgOffPrinGr",
            "is_default",
            "lgd_realisasi",
            "hari_ke_default",
            "ApprovalDate",
            "ChgOffDate",
        ],
    )
    sba = sba[sba["Term"].between(6, 360)]

    kolam = {
        1: sba[sba["is_default"] == 1].reset_index(drop=True),
        0: sba[sba["is_default"] == 0].reset_index(drop=True),
    }
    bagian = []
    for label, sub in debitur.groupby("label_default_debitur"):
        pool = kolam[int(label)]
        idx = rng.integers(0, len(pool), size=len(sub))
        diambil = pool.iloc[idx].reset_index(drop=True)
        diambil.insert(0, "cif_sk", sub["cif_sk"].to_numpy())
        bagian.append(diambil)

    hasil = pd.concat(bagian, ignore_index=True).sort_values("cif_sk").reset_index(drop=True)
    return hasil.rename(columns={"sba_loan_nr": "src_sba_loannr"})


# --------------------------------------------------------------------- orkestra
def build_peta_cif() -> dict[str, int]:
    """Jalankan langkah 1-5 dan tulis tabel pemetaan ke layer silver."""
    rng = np.random.default_rng(settings.seed)

    panel = pilih_panel_debitur(rng)
    debitur = terbitkan_cif(panel)
    debitur = cocokkan_taiwan(debitur, rng)

    hasil_icij = petakan_icij(debitur, rng)
    debitur = hasil_icij["debitur"]

    rekening = petakan_rekening(debitur, rng)
    sba = tarik_sba(debitur, rng)

    panel = panel.merge(debitur[["company_name", "cif_sk", "cif"]], on="company_name", how="inner")

    write_table(panel, "silver", "sl_panel_terpilih")
    write_table(debitur, "silver", "sl_peta_cif")
    write_table(hasil_icij["map_entitas"], "silver", "sl_map_entitas_graf")
    write_table(hasil_icij["rel_pengurus"], "silver", "sl_icij_rel_terpilih")
    write_table(hasil_icij["rel_alamat"], "silver", "sl_icij_alamat_terpilih")
    write_table(rekening, "silver", "sl_peta_rekening")
    write_table(sba, "silver", "sl_peta_sba")

    return {
        "debitur": len(debitur),
        "firm_year": len(panel),
        "grup": int(debitur["grup_id"].nunique()),
        "rekening": len(rekening),
    }

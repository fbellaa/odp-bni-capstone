"""Lapisan graf (ERD B): simpul, edge, dan tabel relasi bertanggal.

Kolom valid_from / valid_to di GOLD_GRAPH_NODES dan GOLD_GRAPH_EDGES bukan
hiasan. Seluruh fitur graf titik-waktu dihitung dari kolom ini, dan uji
pipelines.quality.checks akan GAGAL kalau ada edge dengan valid_from lebih baru
dari snapshot yang memakainya.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipelines.config import settings
from pipelines.graph import alamat as mod_alamat
from pipelines.utils import read_table, write_table

LOG = logging.getLogger("pipelines.graph")

TIPE_NODE_DEBITUR = "badan_hukum"
TIPE_NODE_PIHAK = "pihak"
TIPE_NODE_COUNTERPARTY = "counterparty"
TIPE_NODE_ALAMAT = "alamat"


def _tanggal_default() -> pd.Timestamp:
    return pd.Timestamp(settings.tanggal_default_edge)


# ------------------------------------------------------ DIM_PIHAK & relasi ICIJ
def buat_pihak_dan_relasi(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """DIM_PIHAK, FACT_KEPEMILIKAN, FACT_KEPENGURUSAN dari relasi ICIJ terpilih."""
    peta = read_table("silver", "sl_peta_cif", columns=["cif_sk", "node_id", "grup_id"])
    rel = read_table("silver", "sl_icij_rel_terpilih")
    entity = read_table("silver", "sl_icij_entity", columns=["node_id", "name", "jurisdiction"])
    inter = read_table("silver", "sl_icij_intermediary", columns=["node_id", "name", "countries"])

    entitas_ke_cif = peta.set_index("node_id")["cif_sk"]
    rel = rel[rel["node_id_end"].isin(entitas_ke_cif.index)].copy()
    rel["cif_sk"] = rel["node_id_end"].map(entitas_ke_cif).astype("int64")

    # DIM_PIHAK: nama asli ICIJ tidak dipakai (data pribadi nyata); yang dibawa
    # hanya id sumber sebagai provenance, namanya disintesis oleh Faker.
    from faker import Faker

    fk = Faker("id_ID")
    Faker.seed(settings.seed)

    id_pihak = np.sort(rel["node_id_start"].unique())
    tipe_sumber = pd.Series("individu", index=id_pihak)
    tipe_sumber[np.isin(id_pihak, entity["node_id"].to_numpy())] = "badan"
    tipe_sumber[np.isin(id_pihak, inter["node_id"].to_numpy())] = "badan"

    pihak = pd.DataFrame(
        {
            "pihak_id": np.arange(1, len(id_pihak) + 1),
            "src_icij_node_id": id_pihak,
            "tipe": tipe_sumber.to_numpy(),
        }
    )
    pihak["nama"] = [
        fk.name() if t == "individu" else f"PT {fk.last_name()} Holding"
        for t in pihak["tipe"]
    ]
    peta_pihak = pihak.set_index("src_icij_node_id")["pihak_id"]
    rel["pihak_id"] = rel["node_id_start"].map(peta_pihak).astype("int64")

    kepemilikan = rel[rel["kategori"] == "kepemilikan"].copy()
    kepemilikan["porsi_kepemilikan"] = np.round(
        rng.dirichlet(np.ones(3), size=len(kepemilikan))[:, 0] * 0.9 + 0.05, 4
    )
    kepemilikan["pengendali_efektif"] = kepemilikan["porsi_kepemilikan"] >= 0.25
    fact_kepemilikan = kepemilikan[
        ["pihak_id", "cif_sk", "porsi_kepemilikan", "pengendali_efektif", "valid_from", "valid_to"]
    ].copy()
    fact_kepemilikan["src_icij_rel"] = kepemilikan["link"].astype("string")

    pengurus = rel[rel["kategori"] == "kepengurusan"].copy()
    jabatan = np.where(
        pengurus["link"].astype(str).str.contains("secretary", case=False), "komisaris", "direktur"
    )
    fact_kepengurusan = pd.DataFrame(
        {
            "pihak_id": pengurus["pihak_id"],
            "cif_sk": pengurus["cif_sk"],
            "jenis_jabatan": jabatan,
            "valid_from": pengurus["valid_from"],
            "valid_to": pengurus["valid_to"],
            "src_icij_rel": pengurus["link"].astype("string"),
        }
    )

    LOG.info(
        "pihak %s, kepemilikan %s, kepengurusan %s",
        len(pihak),
        len(fact_kepemilikan),
        len(fact_kepengurusan),
    )
    return {
        "dim_pihak": pihak[["pihak_id", "nama", "tipe", "src_icij_node_id"]],
        "fact_kepemilikan": fact_kepemilikan.reset_index(drop=True),
        "fact_kepengurusan": fact_kepengurusan.reset_index(drop=True),
    }


def hitung_kedalaman_grup() -> pd.DataFrame:
    """Kedalaman rantai kepemilikan per grup, dari struktur berlapis ICIJ."""
    import networkx as nx

    peta = read_table("silver", "sl_peta_cif", columns=["cif_sk", "node_id", "grup_id"])
    rel = read_table("silver", "sl_icij_rel_terpilih", columns=["node_id_start", "node_id_end"])

    g = nx.DiGraph()
    g.add_edges_from(zip(rel["node_id_start"], rel["node_id_end"]))

    hasil = []
    for grup_id, sub in peta.groupby("grup_id"):
        simpul = set(sub["node_id"])
        tetangga = set()
        for n in simpul:
            if g.has_node(n):
                tetangga |= set(g.predecessors(n))
        sg = g.subgraph(simpul | tetangga)
        kedalaman = 1
        if sg.number_of_nodes() > 1:
            dag = nx.DiGraph(sg)
            while not nx.is_directed_acyclic_graph(dag):
                dag.remove_edge(*next(iter(nx.find_cycle(dag, orientation="original")))[:2])
            kedalaman = max(1, nx.dag_longest_path_length(dag))
        hasil.append({"grup_id": grup_id, "kedalaman_kepemilikan": int(kedalaman)})
    return pd.DataFrame(hasil)


# ------------------------------------- BRIDGE_REKENING, DIM_COUNTERPARTY, transfer
def buat_lapisan_giro(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Rekening giro debitur, counterparty, dan FACT_TRANSFER_GIRO dari AML."""
    from faker import Faker

    fk = Faker("id_ID")
    Faker.seed(settings.seed + 1)

    peta_rek = read_table("silver", "sl_peta_rekening")
    transfer = read_table("silver", "sl_aml_transfer")

    rek_debitur = peta_rek.set_index("src_aml_account")
    akun_debitur = set(rek_debitur.index)

    akun_lain = sorted(
        set(transfer["rekening_pengirim"]).union(transfer["rekening_penerima"]) - akun_debitur
    )
    counterparty = pd.DataFrame(
        {
            "cp_id": np.arange(1, len(akun_lain) + 1),
            "src_aml_account": akun_lain,
        }
    )
    counterparty["nama"] = [f"PT {fk.last_name()} {fk.last_name()}" for _ in range(len(akun_lain))]
    counterparty["peran"] = rng.choice(["pemasok", "pembeli"], size=len(akun_lain), p=[0.5, 0.5])
    counterparty["sektor"] = rng.choice(
        ["C", "G", "F", "H", "M", "N"], size=len(akun_lain), p=[0.3, 0.3, 0.15, 0.1, 0.1, 0.05]
    )

    bridge_cp = pd.DataFrame(
        {
            "rekening_id": "CP-" + counterparty["cp_id"].astype(str).str.zfill(6),
            "cif_sk": pd.NA,
            "cp_id": counterparty["cp_id"],
            "src_aml_account": counterparty["src_aml_account"],
            "rekening_utama": True,
        }
    )
    bridge_debitur = peta_rek.assign(cp_id=pd.NA)[
        ["rekening_id", "cif_sk", "cp_id", "src_aml_account", "rekening_utama"]
    ]
    bridge = pd.concat([bridge_debitur, bridge_cp], ignore_index=True)

    peta_akun = bridge.set_index("src_aml_account")
    transfer = transfer.copy()
    transfer["rekening_id_pengirim"] = transfer["rekening_pengirim"].map(peta_akun["rekening_id"])
    transfer["rekening_id_penerima"] = transfer["rekening_penerima"].map(peta_akun["rekening_id"])

    fact_transfer = pd.DataFrame(
        {
            "transfer_id": np.arange(1, len(transfer) + 1),
            "rekening_id_pengirim": transfer["rekening_id_pengirim"],
            "rekening_id_penerima": transfer["rekening_id_penerima"],
            "waktu": transfer["waktu"],
            "nominal_rp": transfer["nominal_rp"].round(0),
            "format_pembayaran": transfer["format_pembayaran"],
            "src_aml_row_id": transfer["aml_row_id"],
            # Kolom evaluasi saja. TIDAK BOLEH masuk tabel fitur model PD.
            "src_is_laundering": transfer["src_is_laundering"].astype("int8"),
        }
    )

    LOG.info("counterparty %s, transfer %s", len(counterparty), len(fact_transfer))
    return {
        "dim_counterparty": counterparty[["cp_id", "nama", "peran", "sektor", "src_aml_account"]],
        "bridge_rekening": bridge,
        "fact_transfer_giro": fact_transfer,
    }


# --------------------------------------------- GOLD_GRAPH_NODES / GOLD_GRAPH_EDGES
def buat_nodes_dan_edges(
    lapisan: dict[str, pd.DataFrame], afiliasi: dict[str, pd.DataFrame] | None = None
) -> dict[str, pd.DataFrame]:
    """Satukan debitur, pihak, counterparty menjadi satu graf bertanggal."""
    peta = read_table("silver", "sl_peta_cif", columns=["cif_sk", "grup_id"])
    pihak = lapisan["dim_pihak"]
    counterparty = lapisan["dim_counterparty"]
    bridge = lapisan["bridge_rekening"]
    transfer = lapisan["fact_transfer_giro"]
    kepemilikan = lapisan["fact_kepemilikan"]
    kepengurusan = lapisan["fact_kepengurusan"]
    dim_alamat = lapisan["dim_alamat"]
    jembatan_alamat = lapisan["fact_alamat_debitur"]

    nodes = []
    nodes.append(
        pd.DataFrame(
            {
                "node_type": TIPE_NODE_DEBITUR,
                "ref_id": peta["cif_sk"],
                "grup_id": peta["grup_id"],
                "valid_from": pd.Timestamp(settings.tanggal_default_edge),
            }
        )
    )
    nodes.append(
        pd.DataFrame(
            {
                "node_type": TIPE_NODE_PIHAK,
                "ref_id": pihak["pihak_id"],
                "grup_id": pd.NA,
                "valid_from": pd.Timestamp(settings.tanggal_default_edge),
            }
        )
    )
    nodes.append(
        pd.DataFrame(
            {
                "node_type": TIPE_NODE_COUNTERPARTY,
                "ref_id": counterparty["cp_id"],
                "grup_id": pd.NA,
                "valid_from": pd.Timestamp(settings.tanggal_default_edge),
            }
        )
    )
    # Alamat adalah simpul, bukan sekadar kunci join. Tanpa ini teks alamatnya
    # berhenti di layer silver dan alamat calon nasabah baru tidak punya apa pun
    # untuk dicocokkan (lihat graph/alamat.py dan graph/resolusi.py).
    if len(dim_alamat):
        nodes.append(
            pd.DataFrame(
                {
                    "node_type": TIPE_NODE_ALAMAT,
                    "ref_id": dim_alamat["alamat_id"],
                    "grup_id": pd.NA,
                    "valid_from": pd.Timestamp(settings.tanggal_default_edge),
                }
            )
        )

    # Pihak nominee untuk mekanisme afiliasi tersembunyi. Bentuknya sama persis
    # dengan pihak ICIJ lain supaya tidak bisa dibedakan dari struktur.
    peta_nominee: dict[int, int] = {}
    if afiliasi and len(afiliasi.get("kepengurusan", [])):
        id_klaster = sorted(afiliasi["kepengurusan"]["afiliasi_id"].unique())
        mulai = int(pihak["pihak_id"].max()) + 1
        peta_nominee = {k: mulai + i for i, k in enumerate(id_klaster)}
        nodes.append(
            pd.DataFrame(
                {
                    "node_type": TIPE_NODE_PIHAK,
                    "ref_id": list(peta_nominee.values()),
                    "grup_id": pd.NA,
                    "valid_from": pd.Timestamp(settings.tanggal_default_edge),
                }
            )
        )

    gold_nodes = pd.concat(nodes, ignore_index=True)
    gold_nodes.insert(0, "node_id", np.arange(1, len(gold_nodes) + 1))
    gold_nodes["valid_to"] = pd.NaT

    kunci = gold_nodes.set_index(["node_type", "ref_id"])["node_id"]

    def id_node(tipe: str, ref: pd.Series) -> pd.Series:
        return pd.Series(
            kunci.reindex(pd.MultiIndex.from_arrays([[tipe] * len(ref), ref])).to_numpy(),
            index=ref.index,
        )

    edges = []

    # memiliki: pihak -> debitur
    if len(kepemilikan):
        edges.append(
            pd.DataFrame(
                {
                    "src_node_id": id_node(TIPE_NODE_PIHAK, kepemilikan["pihak_id"]),
                    "dst_node_id": id_node(TIPE_NODE_DEBITUR, kepemilikan["cif_sk"]),
                    "rel_type": "memiliki",
                    "bobot": kepemilikan["porsi_kepemilikan"],
                    "berarah": True,
                    "valid_from": kepemilikan["valid_from"],
                    "valid_to": kepemilikan["valid_to"],
                    "sumber": "icij",
                }
            )
        )

    # menjabat_di: pihak -> debitur
    if len(kepengurusan):
        edges.append(
            pd.DataFrame(
                {
                    "src_node_id": id_node(TIPE_NODE_PIHAK, kepengurusan["pihak_id"]),
                    "dst_node_id": id_node(TIPE_NODE_DEBITUR, kepengurusan["cif_sk"]),
                    "rel_type": "menjabat_di",
                    "bobot": 1.0,
                    "berarah": True,
                    "valid_from": kepengurusan["valid_from"],
                    "valid_to": kepengurusan["valid_to"],
                    "sumber": "icij",
                }
            )
        )

    # beralamat_di: debitur -> alamat. Edge inilah yang membuat alamat bisa
    # ditelusuri dua arah - dari debitur ke alamatnya, dan dari satu alamat ke
    # seluruh debitur yang memakainya.
    if len(jembatan_alamat):
        edges.append(
            pd.DataFrame(
                {
                    "src_node_id": id_node(TIPE_NODE_DEBITUR, jembatan_alamat["cif_sk"]),
                    "dst_node_id": id_node(TIPE_NODE_ALAMAT, jembatan_alamat["alamat_id"]),
                    "rel_type": "beralamat_di",
                    "bobot": 1.0,
                    "berarah": True,
                    "valid_from": jembatan_alamat["valid_from"],
                    "valid_to": jembatan_alamat["valid_to"],
                    "sumber": "icij",
                }
            )
        )

    # berbagi_atribut: debitur <-> debitur yang berbagi alamat operasional.
    # Pasangannya diturunkan dari DIM_ALAMAT, jadi alamat ICIJ dan alamat klaster
    # afiliasi tersembunyi melewati jalur yang sama persis - tidak ada bentuk
    # edge yang membedakan keduanya.
    pa = mod_alamat.pasangan_seralamat(dim_alamat, jembatan_alamat)
    if len(pa):
        edges.append(
            pd.DataFrame(
                {
                    "src_node_id": id_node(TIPE_NODE_DEBITUR, pa["cif_a"]),
                    "dst_node_id": id_node(TIPE_NODE_DEBITUR, pa["cif_b"]),
                    "rel_type": "berbagi_atribut",
                    "bobot": 1.0,
                    "berarah": False,
                    "valid_from": pa["valid_from"],
                    "valid_to": pd.NaT,
                    "sumber": "icij",
                }
            )
        )

    # memasok / menjual_ke: agregasi transfer giro per pasangan rekening
    peta_bridge = bridge.set_index("rekening_id")
    tr = transfer.dropna(subset=["rekening_id_pengirim", "rekening_id_penerima"]).copy()
    tr["cif_pengirim"] = tr["rekening_id_pengirim"].map(peta_bridge["cif_sk"])
    tr["cp_pengirim"] = tr["rekening_id_pengirim"].map(peta_bridge["cp_id"])
    tr["cif_penerima"] = tr["rekening_id_penerima"].map(peta_bridge["cif_sk"])
    tr["cp_penerima"] = tr["rekening_id_penerima"].map(peta_bridge["cp_id"])

    def node_dari(cif: pd.Series, cp: pd.Series) -> pd.Series:
        """Satu rekening milik debitur ATAU counterparty, tidak pernah keduanya."""
        cif_num = pd.to_numeric(cif, errors="coerce").fillna(-1)
        cp_num = pd.to_numeric(cp, errors="coerce").fillna(-1)
        sisi_debitur = pd.to_numeric(id_node(TIPE_NODE_DEBITUR, cif_num), errors="coerce")
        sisi_cp = pd.to_numeric(id_node(TIPE_NODE_COUNTERPARTY, cp_num), errors="coerce")
        return pd.Series(
            np.where(sisi_debitur.notna(), sisi_debitur, sisi_cp), index=cif.index
        )

    tr["src_node_id"] = node_dari(tr["cif_pengirim"], tr["cp_pengirim"])
    tr["dst_node_id"] = node_dari(tr["cif_penerima"], tr["cp_penerima"])
    tr = tr.dropna(subset=["src_node_id", "dst_node_id"])

    agregat = (
        tr.groupby(["src_node_id", "dst_node_id"])
        .agg(bobot=("nominal_rp", "sum"), valid_from=("waktu", "min"), jumlah_transfer=("transfer_id", "size"))
        .reset_index()
    )
    edges.append(
        pd.DataFrame(
            {
                "src_node_id": agregat["src_node_id"],
                "dst_node_id": agregat["dst_node_id"],
                "rel_type": "memasok",
                "bobot": agregat["bobot"],
                "berarah": True,
                "valid_from": agregat["valid_from"],
                "valid_to": pd.NaT,
                "sumber": "aml",
                "jumlah_transfer": agregat["jumlah_transfer"],
            }
        )
    )

    # ---- afiliasi tersembunyi (langkah 7)
    # Edge di bawah sengaja TIDAK diberi penanda: kalau ditandai, ia tidak lagi
    # tersembunyi. Ground truth-nya ada di FACT_AFILIASI_TERSEMBUNYI.
    if afiliasi:
        kep = afiliasi.get("kepengurusan")
        if kep is not None and len(kep):
            # Nominee bersama muncul sebagai pihak yang menjabat di banyak debitur.
            nominee_node = id_node(
                TIPE_NODE_PIHAK, kep["afiliasi_id"].map(peta_nominee)
            )
            edges.append(
                pd.DataFrame(
                    {
                        "src_node_id": nominee_node,
                        "dst_node_id": id_node(TIPE_NODE_DEBITUR, kep["cif_sk"]),
                        "rel_type": "menjabat_di",
                        "bobot": 1.0,
                        "berarah": True,
                        "valid_from": kep["valid_from"],
                        "valid_to": pd.NaT,
                        "sumber": "icij",
                    }
                )
            )
        # Mekanisme alamat_operasional_bersama tidak ditangani di sini. Klaster
        # itu mendapat baris DIM_ALAMAT sungguhan lewat graph/alamat.py, lalu
        # ikut jalur berbagi_atribut di atas bersama alamat ICIJ biasa.
        pas = afiliasi.get("pasokan")
        if pas is not None and len(pas):
            edges.append(
                pd.DataFrame(
                    {
                        "src_node_id": id_node(TIPE_NODE_DEBITUR, pas["cif_dari"]),
                        "dst_node_id": id_node(TIPE_NODE_DEBITUR, pas["cif_ke"]),
                        "rel_type": "memasok",
                        "bobot": pas["bobot"],
                        "berarah": True,
                        "valid_from": pas["valid_from"],
                        "valid_to": pd.NaT,
                        "sumber": "aml",
                    }
                )
            )

    gold_edges = pd.concat(edges, ignore_index=True)
    gold_edges = gold_edges.dropna(subset=["src_node_id", "dst_node_id"])
    gold_edges["src_node_id"] = gold_edges["src_node_id"].astype("int64")
    gold_edges["dst_node_id"] = gold_edges["dst_node_id"].astype("int64")
    gold_edges["valid_from"] = pd.to_datetime(gold_edges["valid_from"]).fillna(_tanggal_default())
    gold_edges["jumlah_transfer"] = gold_edges.get(
        "jumlah_transfer", pd.Series(np.nan, index=gold_edges.index)
    )
    gold_edges.insert(0, "edge_id", np.arange(1, len(gold_edges) + 1))

    LOG.info(
        "graf: %s simpul, %s edge (%s)",
        len(gold_nodes),
        len(gold_edges),
        gold_edges["rel_type"].value_counts().to_dict(),
    )
    return {"gold_graph_nodes": gold_nodes, "gold_graph_edges": gold_edges}


def build_struktur_graf() -> dict[str, int]:
    """Bangun seluruh tabel struktur graf dan tulis ke layer gold."""
    rng = np.random.default_rng(settings.seed + 7)

    relasi = buat_pihak_dan_relasi(rng)
    giro = buat_lapisan_giro(rng)
    lapisan = {**relasi, **giro}

    edge_afiliasi = None
    klaster = None
    if settings.injeksi_afiliasi:
        from pipelines.generators import afiliasi as mod_afiliasi

        klaster = mod_afiliasi.build_afiliasi()
        edge_afiliasi = mod_afiliasi.edge_afiliasi(klaster, rng)
        lapisan["fact_afiliasi_tersembunyi"] = klaster

    # Alamat dibangun setelah klaster afiliasi supaya alamat operasional bersama
    # milik klaster ikut masuk DIM_ALAMAT lewat pintu yang sama.
    lapisan.update(mod_alamat.bangun_dim_alamat(klaster))

    graf = buat_nodes_dan_edges(lapisan, edge_afiliasi)

    peta = read_table("silver", "sl_peta_cif", columns=["cif_sk", "node_id", "grup_id"])
    map_entitas = peta.rename(columns={"node_id": "src_icij_node_id"})
    map_entitas["metode_pemetaan"] = "klaster_hub_pengurus_icij (SINTESIS - wajib didokumentasi)"

    write_table(map_entitas, "gold", "map_entitas_graf")
    for nama, df in {**lapisan, **graf}.items():
        write_table(df, "gold", nama)

    write_table(hitung_kedalaman_grup(), "silver", "sl_kedalaman_grup")

    return {nama: len(df) for nama, df in {**lapisan, **graf}.items()}

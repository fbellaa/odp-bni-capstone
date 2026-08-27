"""GRAPH_SNAPSHOT_BULANAN dan FEAT_GRAF_PIT - fitur graf titik-waktu.

Aturan anti-bocor (§7.4 proposal):
1. Fitur graf untuk pengajuan bertanggal T dihitung pada snapshot akhir bulan
   SEBELUM T - tidak pernah pada bulan T itu sendiri.
2. Sebuah edge hanya masuk snapshot bila valid_from <= snapshot_date dan
   (valid_to kosong atau valid_to > snapshot_date).
3. community_default_rate dan neighbor_default_rate hanya menghitung default yang
   TANGGAL DEFAULT-nya sudah lewat pada snapshot itu.
4. src_is_laundering tidak pernah masuk tabel ini. Kolom itu hidup di
   FACT_TRANSFER_GIRO dan hanya dipakai untuk evaluasi.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from pipelines.config import settings
from pipelines.utils import akhir_bulan_sebelum, read_table, write_table

LOG = logging.getLogger("pipelines.graph.pit")

REL_PEMBAYARAN = "memasok"
K_BETWEENNESS = 64
PANJANG_SIKLUS_MAKS = 4
DIMENSI_EMBEDDING = 16


def _edges_pada(edges: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    aktif = edges["valid_from"] <= snapshot
    belum_berakhir = edges["valid_to"].isna() | (edges["valid_to"] > snapshot)
    return edges[aktif & belum_berakhir]


def _hhi(bobot: np.ndarray) -> float:
    total = bobot.sum()
    if total <= 0:
        return np.nan
    porsi = bobot / total
    return float((porsi**2).sum())


def _embedding_svd(g: nx.Graph, urutan: list[int]) -> np.ndarray | None:
    """Pengganti node2vec yang murah dan deterministik: SVD tersaring dari adjacency.

    node2vec asli butuh dependensi tambahan dan waktu latih; untuk varian model
    PD (§ERD B: 'PD varian saja') embedding spektral ini cukup dan reproducible.
    """
    try:
        from scipy.sparse.linalg import svds
    except ImportError:
        return None
    if g.number_of_edges() == 0 or len(urutan) <= DIMENSI_EMBEDDING + 1:
        return None
    a = nx.to_scipy_sparse_array(g, nodelist=urutan, weight="bobot", format="csr").astype(float)
    k = min(DIMENSI_EMBEDDING, min(a.shape) - 1)
    u, s, _ = svds(a, k=k, random_state=settings.seed)

    # ARPACK mengembalikan vektor singular dengan tanda sembarang, urutan menaik,
    # dan bebas berotasi di dalam subruang yang nilai singularnya kembar. Tanpa
    # dikanonikkan, 16 kolom embedding berubah tiap kali pipeline dijalankan
    # ulang - fitur yang dipakai model diam-diam bergeser padahal datanya sama.
    urut = np.argsort(s)[::-1]
    u, s = u[:, urut], s[urut]
    baris_dominan = np.abs(u).argmax(axis=0)
    tanda = np.sign(u[baris_dominan, np.arange(u.shape[1])])
    tanda[tanda == 0] = 1.0
    emb = (u * tanda) * s
    if emb.shape[1] < DIMENSI_EMBEDDING:
        emb = np.pad(emb, ((0, 0), (0, DIMENSI_EMBEDDING - emb.shape[1])))
    return emb


def hitung_snapshot(
    snapshot: pd.Timestamp,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    default_per_node: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Metrik graf satu snapshot + bahan mentah untuk FEAT_GRAF_PIT."""
    aktif = _edges_pada(edges, snapshot)

    g = nx.Graph()
    g.add_nodes_from(nodes["node_id"].tolist())
    for src, dst, bobot in zip(aktif["src_node_id"], aktif["dst_node_id"], aktif["bobot"]):
        if g.has_edge(src, dst):
            g[src][dst]["bobot"] += float(bobot)
        else:
            g.add_edge(src, dst, bobot=float(bobot))

    derajat = dict(g.degree())
    derajat_berbobot = dict(g.degree(weight="bobot"))
    pagerank = nx.pagerank(g, weight="bobot") if g.number_of_edges() else {}
    betweenness = (
        nx.betweenness_centrality(g, k=min(K_BETWEENNESS, g.number_of_nodes()), seed=settings.seed)
        if g.number_of_edges()
        else {}
    )
    komunitas = (
        nx.community.louvain_communities(g, weight="bobot", seed=settings.seed)
        if g.number_of_edges()
        else []
    )
    peta_komunitas = {n: i + 1 for i, kom in enumerate(komunitas) for n in kom}

    # Default yang SUDAH terjadi pada snapshot ini saja.
    sudah_default = set(
        default_per_node[default_per_node["tanggal_default"] <= snapshot]["node_id"]
    )
    node_debitur = set(nodes[nodes["node_type"] == "badan_hukum"]["node_id"])

    tingkat_default_komunitas: dict[int, float] = {}
    for i, kom in enumerate(komunitas, start=1):
        anggota = kom & node_debitur
        if anggota:
            tingkat_default_komunitas[i] = len(anggota & sudah_default) / len(anggota)

    snap = pd.DataFrame({"node_id": nodes["node_id"]})
    snap["snapshot_date"] = snapshot
    snap["degree"] = snap["node_id"].map(derajat).fillna(0).astype("int32")
    snap["weighted_degree"] = snap["node_id"].map(derajat_berbobot).fillna(0.0)
    snap["pagerank"] = snap["node_id"].map(pagerank).fillna(0.0)
    snap["betweenness"] = snap["node_id"].map(betweenness).fillna(0.0)
    snap["community_id"] = snap["node_id"].map(peta_komunitas).astype("Int64")
    snap["community_default_rate"] = snap["community_id"].map(tingkat_default_komunitas).astype(
        "float64"
    )

    # ---- bahan FEAT_GRAF_PIT yang butuh arah edge dan jenis relasi
    bayar = aktif[aktif["rel_type"] == REL_PEMBAYARAN]
    keluar = bayar.groupby("src_node_id")["bobot"].apply(lambda s: _hhi(s.to_numpy()))
    masuk = bayar.groupby("dst_node_id")["bobot"].apply(lambda s: _hhi(s.to_numpy()))

    atribut = aktif[aktif["rel_type"] == "berbagi_atribut"]
    derajat_atribut = pd.concat(
        [atribut["src_node_id"], atribut["dst_node_id"]]
    ).value_counts()

    dg = nx.DiGraph()
    dg.add_edges_from(zip(bayar["src_node_id"], bayar["dst_node_id"]))
    simpul_siklus: set[int] = set()
    if dg.number_of_edges():
        for siklus in nx.simple_cycles(dg, length_bound=PANJANG_SIKLUS_MAKS):
            simpul_siklus.update(siklus)

    tetangga_default = {}
    for n in node_debitur:
        if not g.has_node(n):
            continue
        tetangga = set(g.neighbors(n)) & node_debitur
        if tetangga:
            tetangga_default[n] = len(tetangga & sudah_default) / len(tetangga)

    urutan = nodes["node_id"].tolist()
    emb = _embedding_svd(g, urutan)
    if emb is not None:
        for d in range(DIMENSI_EMBEDDING):
            snap[f"emb_{d:02d}"] = emb[:, d]

    tambahan = {
        "supplier_hhi": keluar.to_dict(),
        "buyer_hhi": masuk.to_dict(),
        "shared_attribute_degree": derajat_atribut.to_dict(),
        "circular": simpul_siklus,
        "neighbor_default_rate": tetangga_default,
    }
    return snap, tambahan


def build_fitur_pit() -> dict[str, int]:
    """Bangun GRAPH_SNAPSHOT_BULANAN dan FEAT_GRAF_PIT."""
    edges = read_table("gold", "gold_graph_edges")
    nodes = read_table("gold", "gold_graph_nodes")
    pengajuan = read_table("gold", "fact_pengajuan", columns=["application_id", "cif_sk", "tanggal_pengajuan"])
    default = read_table("gold", "fact_default", columns=["cif_sk", "tanggal_default"])
    eksposur = read_table("gold", "fact_eksposur_grup")
    debitur = read_table("gold", "dim_debitur", columns=["cif_sk", "grup_id", "is_current"])

    edges["valid_from"] = pd.to_datetime(edges["valid_from"])
    edges["valid_to"] = pd.to_datetime(edges["valid_to"])

    node_debitur = nodes[nodes["node_type"] == "badan_hukum"]
    cif_ke_node = node_debitur.set_index("ref_id")["node_id"]

    default_per_node = default.copy()
    default_per_node["node_id"] = default_per_node["cif_sk"].map(cif_ke_node)
    default_per_node = default_per_node.dropna(subset=["node_id"])
    default_per_node = (
        default_per_node.groupby("node_id")["tanggal_default"].min().reset_index()
    )

    # Kunci anti-bocor: akhir bulan SEBELUM tanggal pengajuan.
    pengajuan = pengajuan.copy()
    pengajuan["snapshot_date"] = akhir_bulan_sebelum(pengajuan["tanggal_pengajuan"])
    snapshot_dibutuhkan = sorted(pengajuan["snapshot_date"].unique())
    LOG.info("menghitung %s snapshot graf", len(snapshot_dibutuhkan))

    grup_kini = debitur[debitur["is_current"]][["cif_sk", "grup_id"]]
    eksposur = eksposur.copy()
    eksposur["snapshot_date"] = pd.to_datetime(eksposur["snapshot_date"])

    semua_snap, baris_fitur = [], []
    for snapshot in snapshot_dibutuhkan:
        snapshot = pd.Timestamp(snapshot)
        snap, tambahan = hitung_snapshot(snapshot, edges, nodes, default_per_node)
        semua_snap.append(snap)

        sub = pengajuan[pengajuan["snapshot_date"] == snapshot]
        indeks = snap.set_index("node_id")
        for _, app in sub.iterrows():
            node_id = cif_ke_node.get(app["cif_sk"])
            if node_id is None or node_id not in indeks.index:
                continue
            metrik = indeks.loc[node_id]
            fitur = {
                "application_id": int(app["application_id"]),
                "cif_sk": int(app["cif_sk"]),
                "snapshot_date": snapshot,
                "supplier_concentration_hhi": tambahan["supplier_hhi"].get(node_id, np.nan),
                "buyer_concentration_hhi": tambahan["buyer_hhi"].get(node_id, np.nan),
                "neighbor_default_rate_1hop": tambahan["neighbor_default_rate"].get(node_id, np.nan),
                "community_default_rate": metrik["community_default_rate"],
                "community_id": metrik["community_id"],
                "degree": int(metrik["degree"]),
                "weighted_degree": float(metrik["weighted_degree"]),
                "pagerank": float(metrik["pagerank"]),
                "betweenness": float(metrik["betweenness"]),
                "shared_attribute_degree": int(tambahan["shared_attribute_degree"].get(node_id, 0)),
                "circular_payment_flag": bool(node_id in tambahan["circular"]),
            }
            for d in range(DIMENSI_EMBEDDING):
                kolom = f"emb_{d:02d}"
                if kolom in metrik.index:
                    fitur[f"node_emb_{d:02d}"] = float(metrik[kolom])
            baris_fitur.append(fitur)

    graph_snapshot = pd.concat(semua_snap, ignore_index=True)
    feat = pd.DataFrame(baris_fitur)

    # group_exposure_share diambil dari snapshot yang sama, bukan dari bulan T.
    feat = feat.merge(grup_kini, on="cif_sk", how="left").merge(
        eksposur[["grup_id", "snapshot_date", "group_exposure_share"]],
        on=["grup_id", "snapshot_date"],
        how="left",
    )
    feat["group_exposure_share"] = feat["group_exposure_share"].fillna(0.0)
    feat = feat.drop(columns=["grup_id"])

    write_table(graph_snapshot, "gold", "graph_snapshot_bulanan")
    write_table(feat, "gold", "feat_graf_pit")

    LOG.info(
        "FEAT_GRAF_PIT: %s baris, circular_payment_flag %s, rata-rata degree %.1f",
        len(feat),
        int(feat["circular_payment_flag"].sum()),
        feat["degree"].mean(),
    )
    return {"graph_snapshot_bulanan": len(graph_snapshot), "feat_graf_pit": len(feat)}

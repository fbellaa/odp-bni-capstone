"""Layer GOLD - ERD A: inti kredit komersial (star schema).

Modul ini hanya merangkai: sinyal nyata datang dari layer silver, konteks
Indonesia datang dari pipelines.generators.sintesis.
"""

from __future__ import annotations

import logging

import numpy as np

from pipelines.config import settings
from pipelines.generators import sintesis
from pipelines.utils import write_table

LOG = logging.getLogger("pipelines.gold")


def build_gold_core() -> dict[str, int]:
    rng = np.random.default_rng(settings.seed + 3)

    dim_debitur = sintesis.buat_dim_debitur(rng)
    dim_produk = sintesis.buat_dim_produk()
    fact_lk = sintesis.buat_fact_laporan_keuangan(dim_debitur)
    fact_pengajuan = sintesis.buat_fact_pengajuan(dim_debitur, dim_produk, rng)
    fact_fasilitas = sintesis.buat_fact_fasilitas(fact_pengajuan, dim_debitur, rng)
    fact_agunan = sintesis.buat_fact_agunan(fact_fasilitas, dim_debitur, rng)
    fact_covenant = sintesis.buat_fact_covenant(fact_fasilitas, fact_lk, dim_debitur, rng)
    fact_kolektibilitas, fact_default = sintesis.buat_kolektibilitas_dan_default(
        fact_fasilitas, dim_debitur, rng
    )
    dim_grup, fact_eksposur = sintesis.buat_grup_dan_eksposur(
        dim_debitur, fact_fasilitas, fact_kolektibilitas, rng
    )

    tabel = {
        "dim_debitur": dim_debitur,
        "dim_grup_usaha": dim_grup,
        "dim_produk_fasilitas": dim_produk,
        "fact_laporan_keuangan": fact_lk,
        "fact_pengajuan": fact_pengajuan,
        "fact_fasilitas": fact_fasilitas,
        "fact_agunan": fact_agunan,
        "fact_covenant": fact_covenant,
        "fact_kolektibilitas": fact_kolektibilitas,
        "fact_default": fact_default,
        "fact_eksposur_grup": fact_eksposur,
    }
    for nama, df in tabel.items():
        write_table(df, "gold", nama)

    return {nama: len(df) for nama, df in tabel.items()}

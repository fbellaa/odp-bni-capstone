"""Flow utama: bronze -> silver -> gold -> gerbang kualitas.

Jalankan langsung:
    python -m pipelines.flows.main_flow
atau lewat deployment Prefect:
    prefect deploy --all && prefect deployment run 'banking-copilot-data-pipeline/harian'
"""

from __future__ import annotations

import logging

from prefect import flow, get_run_logger

from pipelines.config import ensure_dirs, settings
from pipelines.flows.bronze_flow import bronze_flow
from pipelines.flows.gold_flow import gold_flow
from pipelines.flows.quality_flow import quality_flow
from pipelines.flows.silver_flow import silver_flow


@flow(
    name="banking-copilot-data-pipeline",
    description=(
        "Pipeline data end-to-end untuk Agentic AI Copilot Kredit Komersial: "
        "tujuh dataset publik dijahit pada CIF sintetis, ERD gold layer, "
        "fitur graf titik-waktu, dan gerbang kualitas anti-bocor."
    ),
)
def main_flow(
    jalankan_bronze: bool = True,
    jalankan_silver: bool = True,
    jalankan_gold: bool = True,
    paksa_ulang_bronze: bool = False,
    muat_postgres: bool | None = None,
    strict: bool = True,
) -> dict[str, object]:
    log = get_run_logger()
    ensure_dirs()
    log.info(
        "mulai pipeline | n_debitur=%s panel=%s tahun seed=%s sample_mode=%s",
        settings.n_debitur,
        settings.panel_years,
        settings.seed,
        settings.sample_mode,
    )

    hasil: dict[str, object] = {}
    if jalankan_bronze:
        hasil["bronze"] = bronze_flow(paksa_ulang=paksa_ulang_bronze)
    if jalankan_silver:
        hasil["silver"] = silver_flow()
    if jalankan_gold:
        hasil["gold"] = gold_flow(muat_postgres=muat_postgres)
    hasil["kualitas"] = quality_flow(strict=strict)

    log.info("pipeline selesai")
    return hasil


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main_flow()

"""Flow Prefect: layer silver (pembersihan + penurunan rasio) dan penjahitan CIF."""

from __future__ import annotations

from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner

from pipelines.transform import joins, silver

RETRY = {"retries": 2, "retry_delay_seconds": 15}


@task(name="silver-us-panel", **RETRY)
def silver_us_panel() -> int:
    return silver.build_silver_us_panel()


@task(name="silver-taiwan", **RETRY)
def silver_taiwan() -> int:
    return silver.build_silver_taiwan()


@task(name="silver-rating", **RETRY)
def silver_rating() -> int:
    return silver.build_silver_rating()


@task(name="silver-sba", timeout_seconds=1800, **RETRY)
def silver_sba() -> int:
    return silver.build_silver_sba()


@task(name="silver-aml", timeout_seconds=1800, **RETRY)
def silver_aml() -> int:
    return silver.build_silver_aml()


@task(name="silver-icij", timeout_seconds=1800, **RETRY)
def silver_icij() -> dict[str, int]:
    return silver.build_silver_icij()


@task(name="jahit-cif", timeout_seconds=3600, **RETRY)
def jahit_cif() -> dict[str, int]:
    """Langkah 1-5 rencana data. Menunggu seluruh tabel silver siap."""
    return joins.build_peta_cif()


@flow(
    name="silver-transform",
    description="Bersihkan tiap sumber, turunkan rasio, lalu jahit semuanya pada satu CIF sintetis.",
    task_runner=ThreadPoolTaskRunner(max_workers=3),
)
def silver_flow() -> dict[str, object]:
    log = get_run_logger()
    futures = [
        silver_us_panel.submit(),
        silver_taiwan.submit(),
        silver_rating.submit(),
        silver_sba.submit(),
        silver_aml.submit(),
        silver_icij.submit(),
    ]
    jumlah = [f.result() for f in futures]

    peta = jahit_cif()
    log.info("silver selesai: %s | pemetaan cif: %s", jumlah, peta)
    return {"silver": jumlah, "peta_cif": peta}

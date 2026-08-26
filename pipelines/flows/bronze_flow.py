"""Flow Prefect: layer bronze (CSV mentah -> parquet)."""

from __future__ import annotations

from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner

from pipelines.config import RAW_FILES, ensure_dirs
from pipelines.ingestion import bronze
from pipelines.utils import table_exists

RETRY = {"retries": 2, "retry_delay_seconds": 15}


def _lewati(nama_tabel: str, paksa_ulang: bool) -> bool:
    return not paksa_ulang and table_exists("bronze", nama_tabel)


@task(name="ingest-taiwan", **RETRY)
def ingest_taiwan(paksa_ulang: bool) -> int:
    if _lewati("br_taiwan_ratio", paksa_ulang):
        return -1
    return bronze.ingest_taiwan()


@task(name="ingest-us-panel", **RETRY)
def ingest_us_panel(paksa_ulang: bool) -> int:
    if _lewati("br_us_panel", paksa_ulang):
        return -1
    return bronze.ingest_us_panel()


@task(name="ingest-rating", **RETRY)
def ingest_rating(paksa_ulang: bool) -> int:
    if _lewati("br_rating", paksa_ulang):
        return -1
    return bronze.ingest_rating()


@task(name="ingest-sba", timeout_seconds=3600, **RETRY)
def ingest_sba(paksa_ulang: bool) -> int:
    if _lewati("br_sba", paksa_ulang):
        return -1
    return bronze.ingest_sba()


@task(name="ingest-aml", timeout_seconds=7200, **RETRY)
def ingest_aml(paksa_ulang: bool) -> int:
    if _lewati("br_aml_transfer", paksa_ulang):
        return -1
    return bronze.ingest_aml()


@task(name="ingest-icij", timeout_seconds=3600, **RETRY)
def ingest_icij(paksa_ulang: bool) -> dict[str, int]:
    if _lewati("br_icij_relationship", paksa_ulang):
        return {}
    return bronze.ingest_icij()


@task(name="periksa-file-mentah")
def periksa_file_mentah() -> list[str]:
    """Gagal cepat kalau ada file mentah yang belum diunduh ke data/raw."""
    hilang = [nama for nama, path in RAW_FILES.items() if not path.exists()]
    if hilang:
        raise FileNotFoundError(
            "file mentah belum ada di data/raw: "
            + ", ".join(f"{n} ({RAW_FILES[n].name})" for n in hilang)
        )
    return sorted(RAW_FILES)


@flow(
    name="bronze-ingestion",
    description="Baca tujuh dataset publik dari data/raw menjadi parquet bronze.",
    task_runner=ThreadPoolTaskRunner(max_workers=3),
)
def bronze_flow(paksa_ulang: bool = False) -> dict[str, object]:
    log = get_run_logger()
    ensure_dirs()
    periksa_file_mentah()

    futures = {
        "taiwan": ingest_taiwan.submit(paksa_ulang),
        "us_panel": ingest_us_panel.submit(paksa_ulang),
        "rating": ingest_rating.submit(paksa_ulang),
        "sba": ingest_sba.submit(paksa_ulang),
        "aml": ingest_aml.submit(paksa_ulang),
        "icij": ingest_icij.submit(paksa_ulang),
    }
    hasil = {nama: f.result() for nama, f in futures.items()}
    log.info("bronze selesai: %s", hasil)
    return hasil

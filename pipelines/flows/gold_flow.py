"""Flow Prefect: layer gold (ERD A + ERD B) dan pemuatan ke Postgres.

Urutan di sini tidak boleh diacak:
struktur graf -> inti kredit -> fitur graf titik-waktu.
FEAT_GRAF_PIT butuh FACT_PENGAJUAN (tanggal T) dan FACT_DEFAULT, sedangkan
DIM_GRUP_USAHA butuh kedalaman kepemilikan yang lahir dari struktur graf.
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task

from pipelines.config import settings
from pipelines.exports import abt
from pipelines.graph import fitur_pit, struktur
from pipelines.transform import gold_core

RETRY = {"retries": 1, "retry_delay_seconds": 15}


@task(name="graf-struktur", timeout_seconds=3600, **RETRY)
def graf_struktur() -> dict[str, int]:
    return struktur.build_struktur_graf()


@task(name="gold-inti-kredit", timeout_seconds=3600, **RETRY)
def gold_inti_kredit() -> dict[str, int]:
    return gold_core.build_gold_core()


@task(name="fitur-graf-pit", timeout_seconds=7200, **RETRY)
def fitur_graf_pit() -> dict[str, int]:
    return fitur_pit.build_fitur_pit()


@task(name="bangun-abt", timeout_seconds=3600, **RETRY)
def bangun_abt() -> dict[str, int]:
    """Paket serah terima untuk data scientist (ABT PD / EWS / LGD + kamus)."""
    return abt.build_abt()


@task(name="muat-ke-postgres", timeout_seconds=3600, retries=1)
def muat_ke_postgres() -> dict[str, int]:
    from pipelines.loaders.postgres import muat_gold

    return muat_gold()


@flow(
    name="gold-build",
    description="Bangun star schema kredit komersial dan lapisan graf titik-waktu.",
)
def gold_flow(muat_postgres: bool | None = None) -> dict[str, object]:
    log = get_run_logger()

    graf = graf_struktur()
    inti = gold_inti_kredit()
    pit = fitur_graf_pit()

    paket_abt = bangun_abt()

    hasil: dict[str, object] = {
        "graf": graf,
        "inti_kredit": inti,
        "pit": pit,
        "abt": paket_abt,
    }

    if settings.tulis_ke_postgres if muat_postgres is None else muat_postgres:
        hasil["postgres"] = muat_ke_postgres()

    log.info("gold selesai: %s tabel graf, %s tabel inti", len(graf), len(inti))
    return hasil

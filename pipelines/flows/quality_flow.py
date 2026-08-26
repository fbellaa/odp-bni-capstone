"""Flow Prefect: gerbang kualitas data.

Flow ini sengaja dibuat bisa dijalankan sendiri, supaya uji anti-bocor bisa
dieksekusi berulang kali tanpa membangun ulang seluruh layer.
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact

from pipelines.quality import checks


@task(name="jalankan-uji-kualitas", retries=0)
def jalankan_uji(strict: bool) -> list[dict]:
    df = checks.jalankan_semua(strict=strict)
    return df.assign(dijalankan_pada=df["dijalankan_pada"].astype(str)).to_dict("records")


@flow(
    name="data-quality-gate",
    description="Uji kunci, integritas referensial, rentang nilai, dan kebocoran waktu lapisan graf.",
)
def quality_flow(strict: bool = True) -> dict[str, int]:
    log = get_run_logger()
    hasil = jalankan_uji(strict)

    gagal = [r for r in hasil if not r["lolos"]]
    try:
        create_table_artifact(
            key="laporan-kualitas-data",
            table=hasil,
            description="Hasil gerbang kualitas data layer gold.",
        )
    except Exception as exc:  # artifact hanya tersedia bila terhubung ke server
        log.warning("artifact Prefect tidak dibuat: %s", exc)

    log.info("uji kualitas: %s lolos, %s gagal", len(hasil) - len(gagal), len(gagal))
    return {"total": len(hasil), "gagal": len(gagal)}

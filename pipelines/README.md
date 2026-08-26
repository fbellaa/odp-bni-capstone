# Data Pipeline - Banking Copilot

Pipeline data untuk *Agentic AI Copilot Keputusan Kredit Komersial*. Orkestrasi
memakai **Prefect 3** (bukan Airflow) supaya bisa dijalankan sebagai proses
Python biasa tanpa scheduler terpisah - pilihan sadar mengingat tenggat tujuh
hari kerja pada proposal §11.

> **Peringatan yang wajib dibaca lebih dulu.**
> Join antar tujuh dataset publik di bawah ini **tidak ada di dunia nyata**.
> Baris rasio Taiwan tidak punya hubungan apa pun dengan akun di dataset AML,
> maupun dengan simpul ICIJ. Pipeline ini **menempelkan** mereka pada satu `cif`
> sintetis. Sinyal (label gagal bayar, rasio, LGD) dan topologi relasinya nyata;
> keterkaitan antar sumbernya sintesis. Lihat
> [docs/data-lineage.md](../docs/data-lineage.md).

---

## 1. Arsitektur

```
data/raw/ (7 dataset publik)
    |
    v  pipelines/ingestion/bronze.py         BRONZE  - salinan mentah -> parquet
data/bronze/
    |
    v  pipelines/transform/silver.py         SILVER  - bersih + rasio diturunkan
    v  pipelines/transform/joins.py                  - langkah 1-5: jahit ke CIF
data/silver/
    |
    v  pipelines/graph/struktur.py           GOLD B  - simpul & edge bertanggal
    v  pipelines/transform/gold_core.py      GOLD A  - star schema kredit
    v  pipelines/graph/fitur_pit.py          GOLD B  - snapshot + FEAT_GRAF_PIT
    v  pipelines/exports/abt.py             ABT     - paket data scientist
data/gold/  -> (opsional) Postgres skema `gold`
    |
    v  pipelines/quality/checks.py           GERBANG - termasuk uji anti-bocor
```

| Modul | Isi |
|---|---|
| `config.py` | Semua path, skala rupiah, timeline, kamus kolom. Satu sumber kebenaran. |
| `utils.py` | Helper parquet, pembersih kolom uang, rasio aman, `akhir_bulan_sebelum`. |
| `ingestion/bronze.py` | Enam fungsi ingest; SBA dan AML dibaca per potongan. |
| `transform/silver.py` | Penurunan DER / ICR / debt-to-EBITDA / siklus modal kerja. |
| `transform/joins.py` | Langkah 1-5 rencana data (penjahitan CIF). |
| `generators/sintesis.py` | Langkah 6: konteks Indonesia, agunan, covenant, kolektibilitas. |
| `graph/struktur.py` | `GOLD_GRAPH_NODES`, `GOLD_GRAPH_EDGES`, relasi ICIJ, transfer giro. |
| `graph/fitur_pit.py` | `GRAPH_SNAPSHOT_BULANAN`, `FEAT_GRAF_PIT`. |
| `exports/abt.py` | Analytic Base Table PD / EWS / LGD + kamus data. |
| `quality/checks.py` | 61 uji, termasuk tiga lapis uji kebocoran waktu. |
| `loaders/postgres.py` | Materialisasi parquet gold ke skema `gold`. |
| `flows/` | Flow Prefect per layer + `main_flow`. |

---

## 2. Menjalankan

### Persiapan

```bash
pip install -r pipelines/requirements.txt
```

Pastikan tujuh berkas mentah ada di `data/raw/` (`bronze_flow` gagal cepat kalau
ada yang hilang):

`data.csv`, `american_bankruptcy.csv`, `corporate_rating.csv`, `SBAnational.csv`,
`LI-Small_Trans.csv`, `nodes-entities.csv`, `nodes-officers.csv`,
`nodes-addresses.csv`, `nodes-intermediaries.csv`, `relationships.csv`

### Sekali jalan (tanpa server Prefect)

```bash
python -m pipelines.flows.main_flow
```

Kira-kira 20-30 menit pada laptop; sebagian besar untuk membaca
`LI-Small_Trans.csv` (650 MB, dua lintasan). Layer bronze dilewati otomatis bila
parquet-nya sudah ada.

### Mode cepat untuk uji coba

```bash
SAMPLE_MODE=1 N_DEBITUR=300 python -m pipelines.flows.main_flow
```

### Dengan server + UI Prefect

Butuh tiga terminal. Pakai `python -m prefect`, bukan `prefect` saja, supaya CLI
dan pipeline berjalan di interpreter yang sama.

**Terminal 1 - server** (biarkan hidup, UI di http://127.0.0.1:4200):

```bash
python -m prefect server start
```

**Terminal 2 - daftarkan deployment** (sekali saja, tersimpan di database Prefect):

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
export PREFECT_PROJECT_DIR="$(pwd)"
python -m prefect work-pool create default-agent-pool --type process
python -m prefect deploy --all
```

`PREFECT_PROJECT_DIR` wajib diisi - langkah `pull` di `prefect.yaml` memakainya
untuk menentukan direktori kerja flow. Tanpa itu, flow run gagal saat
`set_working_directory`.

**Terminal 3 - worker** (biarkan hidup, ini yang mengeksekusi flow):

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
export PREFECT_PROJECT_DIR="C:/path/ke/banking-copilot"
python -m prefect worker start --pool default-agent-pool
```

Lalu picu run dari terminal mana pun:

```bash
python -m prefect deployment run 'data-quality-gate/gerbang-kualitas'
python -m prefect deployment run 'banking-copilot-data-pipeline/harian'
```

> **Windows.** Beberapa perintah CLI Prefect mencetak karakter Unicode dan
> menabrak codepage cp1252 (`UnicodeEncodeError: 'charmap' codec`). Perintahnya
> tetap jalan, hanya cetakannya yang gagal. Setel `PYTHONIOENCODING=utf-8`
> supaya bersih.

Ekuivalen PowerShell untuk baris `export`:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:PREFECT_PROJECT_DIR = (Get-Location).Path
$env:PYTHONIOENCODING = "utf-8"
```

Jadwal yang sudah didefinisikan di `prefect.yaml`:

| Deployment | Jadwal | Guna |
|---|---|---|
| `harian` | 02:00 WIB tiap hari | pipeline penuh |
| `gerbang-kualitas` | tiap jam, 08:00-17:00 hari kerja | uji anti-bocor berulang |
| `refresh-gold` | manual | bangun ulang gold tanpa baca CSV besar |
| `bronze-refresh` | manual | baca ulang seluruh CSV mentah |

### Memuat ke Postgres

```bash
docker compose up -d postgres
LOAD_TO_POSTGRES=1 python -m pipelines.flows.main_flow
# atau hanya memuat:
python -m pipelines.loaders.postgres
```

---

## 3. Parameter (environment variable)

| Variabel | Default | Arti |
|---|---|---|
| `N_DEBITUR` | `3000` | Jumlah CIF sintetis (3.000 x 3 tahun = 9.000 firm-year). |
| `PANEL_YEARS` | `3` | Panjang panel laporan keuangan. |
| `PIPELINE_SEED` | `42` | Seed tunggal untuk seluruh sintesis - hasilnya reproducible. |
| `SAMPLE_MODE` | `0` | `1` membatasi baris yang dibaca dari SBA dan AML. |
| `AML_MAX_ROWS` | - | Batas baris AML secara eksplisit. |
| `LOAD_TO_POSTGRES` | `0` | `1` memuat gold ke Postgres di akhir flow. |

---

## 4. Uji

```bash
python -m pytest
```

`test_gold_layer.py` dan `test_abt.py` otomatis di-skip kalau `data/gold` masih
kosong. Uji yang paling penting:

- `test_fitur_graf_memakai_snapshot_bulan_sebelumnya`
- `test_tidak_ada_edge_dengan_valid_from_setelah_snapshot`
- `test_feat_graf_pit_tidak_memuat_label`
- `test_abt_pd_tidak_memuat_perilaku_pasca_pencairan`
- `test_bad_rate_cukup_untuk_dilatih`

Uji pertama pernah **gagal betulan** saat pipeline ini dibangun: helper
`akhir_bulan_sebelum` sempat mengembalikan akhir bulan yang sama dengan bulan
pengajuan. Itu persis kebocoran yang diperingatkan proposal §7.4, dan itulah
alasan uji tersebut dibuat sebelum fiturnya ada.

---

## 5. Serah terima ke data scientist

Flow gold menghasilkan tiga Analytic Base Table siap model:

| Berkas | Grain | Baris | Target |
|---|---|---|---|
| `data/gold/abt_pd.parquet` | `application_id` | 2.192 | `y_default_12bln` |
| `data/gold/abt_ews.parquet` | `facility_id` x `snapshot_date` | 38.092 | `y_default_6bln` |
| `data/gold/abt_lgd.parquet` | `facility_id` yang default | 75 | `y_lgd_realisasi` |

Setiap kolom fitur berprefiks `fin_`, `app_`, `perilaku_`, atau `graf_`, sehingga
uji ablasi §7.3 cukup `X.drop(columns=X.filter(like="graf_").columns)`.

Panduan lengkapnya - termasuk arti `y = NaN` (tersensor kanan) dan kolom yang
sebaiknya di-drop - ada di
[docs/serah-terima-data-scientist.md](../docs/serah-terima-data-scientist.md).

---

## 6. Yang belum dikerjakan

- `pipelines/dbt/` masih kosong. Transformasi gold ditulis dengan pandas, bukan
  dbt; uji `valid_from > snapshot_date` sudah ada padanannya di
  `quality/checks.py`. Kalau dbt tetap dipakai nanti, uji itu yang pertama
  diporting.
- `node2vec_emb[0..15]` pada ERD B diisi embedding SVD dari adjacency
  (`node_emb_00..15`), bukan node2vec sungguhan - deterministik dan tanpa
  dependensi tambahan. Statusnya varian, bukan fitur utama.
- Injeksi kekotoran data & afiliasi tersembunyi (langkah 7) belum masuk; sesuai
  proposal §4 spesifikasinya harus dipisah dari kode deteksi, jadi menunggu
  dokumen spesifikasi tersendiri.
- Narasi RM dan korpus kebijakan kredit (LLM-generated) di luar cakupan pipeline
  ini.

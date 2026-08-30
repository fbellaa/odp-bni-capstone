# Banking Copilot

Agentic AI Copilot untuk Keputusan Kredit Komersial - proyek capstone.

```
app/ui/        Streamlit (7 halaman: copilot pengajuan, what-if, graf, portofolio,
               kesehatan model, BI, copilot lokal)
copilot/       LLM lokal (Ollama): baca PDF, RAG kebijakan, agen tool calling
pipelines/     Data pipeline Prefect 3: bronze -> silver -> gold + gerbang kualitas
ml/            Model PD, LGD, EWS (belum diisi)
docs/          Silsilah data dan catatan keputusan
data/          raw / bronze / silver / gold / quality / index (tidak di-commit)
```

---

## Peringatan data (wajib dibaca)

Proyek ini menggabungkan tujuh dataset publik: Taiwan Economic Journal
(kebangkrutan), American companies bankruptcy (panel), Corporate Credit Rating,
US SBA (899 ribu pinjaman usaha), IBM AML transactions, dan ICIJ Offshore Leaks.

**Join antar dataset itu tidak ada di dunia nyata.** Baris rasio Taiwan tidak
punya hubungan apa pun dengan akun di dataset AML maupun simpul ICIJ. Pipeline
*menempelkan* mereka pada satu `cif` sintetis.

- **Nyata**: label gagal bayar, rasio keuangan dan trennya, distribusi rating,
  tingkat pemulihan (LGD), topologi transfer, struktur kepemilikan berlapis.
- **Sintesis**: seluruh konteks Indonesia (nama PT, CIF, NPWP, sektor KBLI,
  skala rupiah), pengajuan, agunan, covenant, kolektibilitas - dan **keterkaitan
  antar sumber itu sendiri**.

Rinciannya, termasuk setiap asumsi yang menempel pada angka "nyata", ada di
[docs/data-lineage.md](docs/data-lineage.md).

Sebelum memakai datanya, baca panduan yang sesuai peran Anda:

| Peran | Dokumen |
|---|---|
| Melatih model PD / LGD / EWS | [serah-terima-data-scientist.md](docs/serah-terima-data-scientist.md) |
| Membuat dashboard & laporan portofolio | [panduan-business-analyst.md](docs/panduan-business-analyst.md) |
| Menelusuri asal-usul tiap angka | [data-lineage.md](docs/data-lineage.md) |

---

## Menjalankan

### Data pipeline

```bash
pip install -r pipelines/requirements.txt
python -m pipelines.flows.main_flow
```

Hasilnya: 30 tabel gold di `data/gold/`, plus tiga Analytic Base Table siap
model (`abt_pd`, `abt_ews`, `abt_lgd`) dan kamus datanya.

Lihat [pipelines/README.md](pipelines/README.md) untuk mode Prefect server,
jadwal deployment, parameter, dan pemuatan ke Postgres.

### UI

```bash
pip install -r app/ui/requirements.txt
streamlit run app/ui/Copilot_Pengajuan.py
```

Keempat halaman berjalan tanpa Ollama: PD, LGD, dan klaster dibaca dari
`ml/artifacts` di atas `data/gold`. Yang butuh Ollama adalah pembacaan PDF dengan
model bahasa, agen tool calling, dan penelusuran korpus kebijakan pada halaman
copilot; tanpa Ollama, dokumen dibaca dengan sapuan pola, tool dijalankan tanpa
model bahasa, dan bagian rujukan kebijakan pada memo dibiarkan kosong beserta
sebabnya.

### Copilot lokal (halaman entry)

```bash
bash copilot/scripts/siapkan_ollama.sh    # pasang Ollama + tarik model
pip install -r copilot/requirements.txt
python -m copilot.rag.indeks              # bangun index kebijakan, sekali saja
python -m copilot.scripts.uji_asap        # periksa rantai sebelum demo
```

Rantainya: PDF pengajuan -> model bahasa lokal -> struktur -> `telusuri_afiliasi()`
-> RAG atas `docs/policies/` -> agen tool calling -> draft credit memo.

Model bahasa tidak pernah menghitung. Seluruh angka pada memo berasal dari 12
tool deterministik di `copilot/alat/`, dan tiap angka mencantumkan nama tool
yang menghasilkannya. Profil bawaan `hemat` (~2,3 GB) dipilih supaya muat di
Kaggle / Colab free tier.

Rincian, termasuk cara memakai SahabatAI dan apa yang belum ditangani, ada di
[copilot/README.md](copilot/README.md).

### Semuanya lewat Docker

```bash
cp .env.example .env
docker compose up -d warehouse prefect-server
docker compose --profile batch run --rm pipeline   # bangun data warehouse
```

`pipeline` sengaja ditaruh di balik profil `batch`. Ia pekerjaan sekali jalan
yang **menimpa `data/gold`**, jadi ia tidak boleh ikut terbawa `docker compose
up` polos - apalagi menjelang demo, ketika tabel gold yang ada justru yang mau
dipakai.

Prefect UI: http://localhost:4200

Antarmuka beserta lapisan copilot punya image sendiri:

```bash
docker compose up -d ui
```

Dibangun dari `Dockerfile` di akar, membawa Streamlit, `copilot/`, dan
`pipelines/graph` sekaligus - ketiganya satu image karena halaman 1 mengimpor
keduanya langsung. Data pipeline punya image terpisah di `pipelines/Dockerfile`
karena dependensinya (Prefect, psycopg2) tidak beririsan. `./data` di-mount, jadi
tabel gold dan index kebijakan dipakai bersama dengan yang di host.

**Model bahasa tidak ikut di dalam image.** Bawaannya, container menghubungi
Ollama yang terpasang di host lewat `host.docker.internal:11434`. Di Windows itu
cara yang paling sedikit risikonya: installer resmi Ollama memakai GPU tanpa
konfigurasi apa pun.

Kalau Ollama juga ingin dikontainerkan - butuh Docker Desktop dengan backend
WSL2 dan driver NVIDIA yang cocok:

```bash
docker compose --profile ollama up -d ollama ui
```

lalu setel `OLLAMA_HOST=http://ollama:11434` di `.env`, dan tarik modelnya ke
dalam container:

```bash
docker compose exec ollama ollama pull qwen2.5:3b-instruct
```

Index kebijakan dibangun dari dalam container supaya artefaknya mendarat di
`./data/index` yang sama:

```bash
docker compose run --rm ui python -m copilot.rag.indeks
```

---

## Uji

```bash
python -m pytest
```

Termasuk uji anti-bocor lapisan graf (§7.4 proposal): fitur graf untuk pengajuan
bertanggal `T` wajib dihitung pada snapshot akhir bulan **sebelum** `T`, dan
tidak boleh ada edge dengan `valid_from > snapshot_date` yang ikut terhitung.

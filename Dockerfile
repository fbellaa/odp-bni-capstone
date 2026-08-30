# Antarmuka Streamlit + lapisan copilot. Ini image aplikasi yang dijalankan;
# data pipeline (Prefect) punya image sendiri di pipelines/Dockerfile.
#
# Pembagiannya bukan soal rapi-rapian, melainkan soal dependensi yang memang
# tidak beririsan. Pipeline butuh Prefect, SQLAlchemy, psycopg2, dan karenanya
# build-essential plus libpq-dev; antarmuka butuh Streamlit, Plotly, dan pypdf.
# Menyatukannya berarti setiap image memikul dependensi yang tidak dipakainya,
# dan container antarmuka membawa driver Postgres yang tidak pernah disentuh.
#
# Streamlit, `copilot/`, dan `pipelines/graph` justru DIsatukan di sini karena
# halaman 1 mengimpor `copilot.*` langsung, dan `copilot.dokumen.jembatan`
# mengimpor `telusuri_afiliasi()` dari pipelines. Memecahnya jadi tiga image
# hanya akan memindahkan impor itu ke jaringan.
#
# Yang TIDAK ada di sini: model bahasa. Inferensi dikerjakan Ollama di luar
# container - lihat layanan `ollama` pada docker-compose.yaml, atau pakai
# Ollama yang terpasang di host lewat host.docker.internal.

FROM python:3.11-slim

WORKDIR /opt/banking-copilot

# curl dipakai HEALTHCHECK. Tidak ada build-essential: seluruh dependensi
# tersedia sebagai wheel, dan menariknya menambah ~300 MB tanpa guna.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Requirement disalin lebih dulu supaya lapisan pip ikut cache selama daftar
# dependensinya tidak berubah - build ulang setelah mengedit kode jadi hitungan
# detik, bukan menit.
COPY copilot/requirements.txt copilot/requirements.txt
COPY app/ui/requirements.txt app/ui/requirements.txt
RUN pip install --no-cache-dir \
    -r copilot/requirements.txt \
    -r app/ui/requirements.txt

COPY copilot/ copilot/
COPY app/ app/
COPY docs/ docs/
# Artefak model PD/EWS/LGD/klaster. Disalin, bukan di-mount: 3,2 MB, sudah
# ter-track di git, dan `lib/model_nyata.py` membacanya lewat AKAR/ml/artifacts.
# Tanpa baris ini seluruh lapisan model nyata jatuh ke mock tanpa suara.
COPY ml/artifacts/ ml/artifacts/
# Lapisan graf; `telusuri_afiliasi()` diimpor jembatan dokumen.
COPY pipelines/ pipelines/

# Akar proyek harus ada di sys.path supaya `import copilot` dan `import
# pipelines` bekerja dari mana pun skrip dijalankan.
ENV PYTHONPATH=/opt/banking-copilot \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/ui/Copilot_Pengajuan.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

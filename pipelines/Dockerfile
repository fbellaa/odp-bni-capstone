# Image untuk data pipeline (Prefect). UI Streamlit memakai Dockerfile di root.
FROM python:3.11-slim

WORKDIR /opt/banking-copilot

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pipelines/requirements.txt pipelines/requirements.txt
RUN pip install --no-cache-dir -r pipelines/requirements.txt

COPY pipelines/ pipelines/
COPY prefect.yaml pytest.ini ./

ENV PYTHONPATH=/opt/banking-copilot \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "pipelines.flows.main_flow"]

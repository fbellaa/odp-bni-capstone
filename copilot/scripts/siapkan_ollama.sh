#!/usr/bin/env bash
# Pasang Ollama, hidupkan servernya, dan tarik model sesuai profil.
#
# Dipakai di Kaggle / Colab free tier, dan sama saja di mesin Linux biasa.
# Idempoten: aman dijalankan ulang pada sesi yang sama.
#
#   bash copilot/scripts/siapkan_ollama.sh            # profil hemat (~2,5 GB)
#   COPILOT_PROFIL=seimbang \
#   COPILOT_MODEL_SAHABAT=hf.co/<repo-gguf>:Q4_K_M \
#     bash copilot/scripts/siapkan_ollama.sh          # profil seimbang (~8 GB)

set -euo pipefail

PROFIL="${COPILOT_PROFIL:-hemat}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

echo "== Profil: ${PROFIL}"

# --------------------------------------------------------------- pasang
if ! command -v ollama >/dev/null 2>&1; then
  echo "== Memasang Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "== Ollama sudah terpasang: $(ollama --version 2>/dev/null || echo '?')"
fi

# ------------------------------------------------------------- hidupkan
# Notebook tidak punya systemd, jadi server dijalankan sebagai proses latar.
if curl -sf "${HOST}/api/tags" >/dev/null 2>&1; then
  echo "== Server sudah hidup di ${HOST}"
else
  echo "== Menghidupkan server"
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    if curl -sf "${HOST}/api/tags" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if ! curl -sf "${HOST}/api/tags" >/dev/null 2>&1; then
    echo "!! Server tidak merespons dalam 30 detik. Isi /tmp/ollama.log:" >&2
    tail -20 /tmp/ollama.log >&2
    exit 1
  fi
  echo "== Server siap"
fi

# ----------------------------------------------------------- tarik model
MODEL=("nomic-embed-text")

case "${PROFIL}" in
  hemat)
    MODEL+=("qwen2.5:3b-instruct")
    ;;
  seimbang)
    MODEL+=("qwen2.5:7b-instruct")
    if [ -n "${COPILOT_MODEL_SAHABAT:-}" ]; then
      MODEL+=("${COPILOT_MODEL_SAHABAT}")
    else
      echo "!! COPILOT_MODEL_SAHABAT belum disetel." >&2
      echo "   Profil seimbang memakai SahabatAI untuk peran bahasa Indonesia," >&2
      echo "   dan tag GGUF-nya harus Anda pilih sendiri. Contoh bentuknya:" >&2
      echo "     export COPILOT_MODEL_SAHABAT=hf.co/<pengguna>/<repo-gguf>:Q4_K_M" >&2
      echo "   Atau pakai profil hemat: export COPILOT_PROFIL=hemat" >&2
      exit 1
    fi
    ;;
  *)
    echo "!! Profil '${PROFIL}' tidak dikenal. Pilihan: hemat, seimbang" >&2
    exit 1
    ;;
esac

# Nilai COPILOT_MODEL_* menimpa pilihan profil; ikutkan supaya tidak ada model
# yang diminta konfigurasi tapi tidak pernah ditarik.
for VAR in COPILOT_MODEL_EKSTRAKSI COPILOT_MODEL_CHAT COPILOT_MODEL_AGEN COPILOT_MODEL_EMBEDDING; do
  NILAI="${!VAR:-}"
  if [ -n "${NILAI}" ]; then MODEL+=("${NILAI}"); fi
done

for M in $(printf '%s\n' "${MODEL[@]}" | awk '!seen[$0]++'); do
  echo "== Menarik ${M}"
  ollama pull "${M}"
done

echo
echo "== Model terpasang"
ollama list

cat <<'PESAN'

Langkah berikutnya:

  python -m copilot.rag.indeks        # bangun index kebijakan (sekali saja)
  streamlit run app/ui/app.py         # buka halaman "7 · Copilot lokal"

PESAN

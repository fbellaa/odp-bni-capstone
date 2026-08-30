from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

import requests

from copilot.konfigurasi import PENGATURAN


class KlienOllama:
    def __init__(self, host: str | None = None):
        raw_host = (
            host
            or os.getenv("RAG_OLLAMA_HOST")
            or os.getenv("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )
        if not str(raw_host).startswith(("http://", "https://")):
            raw_host = "http://" + str(raw_host)
        self.host = str(raw_host).rstrip("/")
        self.session = requests.Session()

    def pastikan_model(self, model: str):
        response = self.session.get(
            f"{self.host}/api/tags",
            timeout=10,
        )
        response.raise_for_status()

        installed = [
            item.get("name", "")
            for item in response.json().get("models", [])
        ]

        if any(
            name == model
            or name.startswith(model + ":")
            for name in installed
        ):
            return

        if shutil.which("ollama") is None:
            raise RuntimeError(
                f"Model {model!r} belum tersedia dan binary Ollama tidak ditemukan."
            )

        subprocess.run(
            ["ollama", "pull", model],
            check=True,
        )

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)

        model = PENGATURAN.model_untuk("embedding")
        self.pastikan_model(model)

        response = self.session.post(
            f"{self.host}/api/embed",
            json={
                "model": model,
                "input": texts,
            },
            timeout=300,
        )

        if response.ok:
            embeddings = response.json().get("embeddings")
            if embeddings is not None:
                return embeddings

        # Backward compatibility with older Ollama.
        outputs = []
        for text in texts:
            response = self.session.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                },
                timeout=300,
            )
            response.raise_for_status()
            outputs.append(
                response.json()["embedding"]
            )

        return outputs

    def chat(
        self,
        messages,
        *,
        peran: str = "chat",
    ) -> dict[str, Any]:
        model = PENGATURAN.model_untuk(peran)
        self.pastikan_model(model)

        response = self.session.post(
            f"{self.host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]


_CLIENT = None


def klien() -> KlienOllama:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = KlienOllama()
    return _CLIENT

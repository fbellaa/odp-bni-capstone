from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(
    os.getenv("ODP_REPO_ROOT", "/content/odp-bni-capstone")
).resolve()

POLICY_DIR = Path(
    os.getenv(
        "RAG_POLICY_DIR",
        str(REPO_ROOT / "docs" / "policies"),
    )
).resolve()

INDEX_DIR = Path(
    os.getenv(
        "RAG_INDEX_DIR",
        str(REPO_ROOT / "data" / "index"),
    )
).resolve()

POLICY_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Pengaturan:
    model_embedding: str = os.getenv(
        "RAG_EMBED_MODEL",
        "nomic-embed-text",
    )
    model_chat: str = os.getenv(
        "RAG_CHAT_MODEL",
        os.getenv(
            "AI_QWEN_AGENT_MODEL",
            os.getenv(
                "QWEN_AGENT_MODEL",
                "qwen2.5:7b-instruct",
            ),
        ),
    )
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    ukuran_potongan: int = int(
        os.getenv("RAG_CHUNK_SIZE", "2400")
    )
    tumpang_tindih: int = int(
        os.getenv("RAG_CHUNK_OVERLAP", "300")
    )

    def model_untuk(self, peran: str) -> str:
        peran = str(peran).strip().lower()
        if peran == "embedding":
            return self.model_embedding
        return self.model_chat


PENGATURAN = Pengaturan()

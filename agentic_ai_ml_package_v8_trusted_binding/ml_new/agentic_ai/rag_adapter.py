from __future__ import annotations

from typing import Any


class PolicyRAGError(RuntimeError):
    """Runtime/configuration error in the policy RAG layer."""


def rag_preflight() -> dict[str, Any]:
    """Check whether the supplied copilot RAG implementation is importable and indexed."""
    try:
        from copilot.rag.indeks import index_tersedia
        from copilot.konfigurasi import PENGATURAN
    except Exception as exc:
        return {
            "rag_importable": False,
            "index_available": False,
            "embedding_model": None,
            "error": str(exc),
        }

    try:
        embedding_model = PENGATURAN.model_untuk("embedding")
    except Exception:
        embedding_model = None

    try:
        available = bool(index_tersedia())
    except Exception as exc:
        return {
            "rag_importable": True,
            "index_available": False,
            "embedding_model": embedding_model,
            "error": str(exc),
        }

    return {
        "rag_importable": True,
        "index_available": available,
        "embedding_model": embedding_model,
        "error": None,
    }


def query_credit_policy(
    query: str,
    *,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Run real policy retrieval, preserving citations even if synthesis fails."""
    query = str(query or "").strip()

    if not query:
        raise PolicyRAGError(
            "query_credit_policy membutuhkan query yang tidak kosong."
        )

    try:
        from copilot.rag.indeks import index_tersedia
    except Exception as exc:
        raise PolicyRAGError(
            f"copilot.rag.indeks tidak dapat di-import: {exc}"
        ) from exc

    try:
        if not index_tersedia():
            return {
                "status": "index_not_ready",
                "query": query,
                "answer": (
                    "Index kebijakan belum tersedia. Bangun index RAG dari "
                    "docs/policies sebelum meminta policy retrieval."
                ),
                "citations": [],
                "citation_count": 0,
                "top_k": top_k,
            }
    except Exception as exc:
        raise PolicyRAGError(
            f"Gagal mengecek index kebijakan: {exc}"
        ) from exc

    try:
        from copilot.rag.pencarian import jawab, kutipan
    except Exception as exc:
        raise PolicyRAGError(
            "copilot.rag.pencarian tidak dapat di-import. "
            f"Detail: {exc}"
        ) from exc

    # 1. Real retrieval
    try:
        excerpts = kutipan(
            query,
            top_k=top_k,
        )
    except Exception as exc:
        raise PolicyRAGError(
            f"Policy retrieval gagal: {exc}"
        ) from exc

    if not excerpts:
        return {
            "status": "no_match",
            "query": query,
            "answer": (
                "Tidak ditemukan kutipan kebijakan yang relevan "
                "pada index yang tersedia."
            ),
            "citations": [],
            "citation_count": 0,
            "top_k": top_k,
        }

    retrieval_citations = [
        {
            "rujukan": x.get("pasal"),
            "skor": x.get("skor"),
            "halaman": x.get("halaman") or [],
            "versi": x.get("versi"),
        }
        for x in excerpts
    ]

    # 2. Grounded answer synthesis. Retrieval remains valid if this fails.
    synthesis_warning = None

    try:
        out = jawab(
            query,
            top_k=top_k,
        )
        answer = str(
            out.get("jawaban") or ""
        ).strip()
        citations = (
            out.get("sitasi")
            or retrieval_citations
        )
    except Exception as exc:
        answer = (
            "Retrieval kebijakan berhasil, tetapi penyusunan jawaban "
            "naratif RAG gagal. Gunakan citation hasil retrieval."
        )
        citations = retrieval_citations
        synthesis_warning = str(exc)

    return {
        "status": "retrieved",
        "query": query,
        "answer": answer,
        "citations": citations,
        "citation_count": len(citations),
        "retrieved_excerpts": excerpts,
        "top_k": top_k,
        "synthesis_warning": synthesis_warning,
    }


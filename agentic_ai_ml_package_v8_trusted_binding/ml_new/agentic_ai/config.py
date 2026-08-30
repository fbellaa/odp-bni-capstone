from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    explicit = os.getenv("ODP_REPO_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
ARTIFACT_ROOT = Path(
    os.getenv("ML_ARTIFACT_ROOT", str(REPO_ROOT / "ml" / "artifacts"))
).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    # V7 role-specialized models. Deterministic mapping is primary; Qwen extractor is optional fallback.
    qwen_extractor_model: str = os.getenv("QWEN_EXTRACTOR_MODEL", "qwen2.5:7b-instruct")
    qwen_agent_model: str = os.getenv("QWEN_AGENT_MODEL", "qwen2.5:7b-instruct")
    vlm_model: str = os.getenv("VLM_MODEL", "qwen3-vl:4b-instruct")

    # Qwen is the official final narrator in V9.3.
    # SAHABAT_MODEL is accepted only as a legacy compatibility fallback.
    qwen_narrator_model: str = os.getenv(
        "QWEN_NARRATOR_MODEL",
        os.getenv(
            "SAHABAT_MODEL",
            os.getenv(
                "QWEN_AGENT_MODEL",
                "qwen2.5:7b-instruct",
            ),
        ),
    )

    timeout_seconds: int = int(os.getenv("AI_TIMEOUT", "300"))
    ollama_keep_alive: str = os.getenv("AI_OLLAMA_KEEP_ALIVE", "5m")
    max_tool_rounds: int = int(os.getenv("AI_MAX_TOOL_ROUNDS", "4"))

    extractor_temperature: float = float(os.getenv("AI_EXTRACTOR_TEMPERATURE", "0"))
    agent_temperature: float = float(os.getenv("AI_AGENT_TEMPERATURE", "0"))
    narrator_temperature: float = float(os.getenv("AI_NARRATOR_TEMPERATURE", "0.2"))

    # Fast mode keeps the extraction request compact instead of sending whole long documents.
    extraction_mode: str = os.getenv("AI_EXTRACTION_MODE", "fast").strip().lower()
    extraction_max_chars: int = int(os.getenv("AI_EXTRACTION_MAX_CHARS", "9000"))
    use_qwen_semantic_fallback: bool = os.getenv("AI_QWEN_SEMANTIC_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "y"}

    # Policy RAG tool calling.
    rag_enabled: bool = os.getenv("AI_RAG_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y"}
    rag_top_k: int = int(os.getenv("AI_RAG_TOP_K", "5"))
    rag_tool_rounds: int = int(os.getenv("AI_RAG_TOOL_ROUNDS", "2"))

    def require_qwen_narrator(self) -> str:
        if not self.qwen_narrator_model:
            raise RuntimeError(
                "QWEN_NARRATOR_MODEL belum diset. "
                "Set ke tag Qwen yang tersedia di Ollama."
            )
        return self.qwen_narrator_model

    @property
    def sahabat_model(self) -> str:
        """Legacy compatibility alias. V9.3 narrator is Qwen."""
        return self.qwen_narrator_model

    def require_sahabat(self) -> str:
        """Legacy compatibility alias. V9.3 narrator is Qwen."""
        return self.require_qwen_narrator()


SETTINGS = Settings()

MODEL_LAYOUT = {
    "pd": {
        "task": "classification",
        "folder": ARTIFACT_ROOT / "pd",
        "champion": "pd_champion_new.joblib",
        "schema": "pd_feature_schema.json",
        "metadata": "pd_metadata.json",
        "manifest": "pd_manifest.json",
        "metrics": "pd_metrics.json",
        "policy": "pd_decision_policy.json",
        "reference": "pd_reference_stats.json",
        "importance": "pd_feature_importance.csv",
        "requirements": "pd_requirements.txt",
    },
    "ews": {
        "task": "classification",
        "folder": ARTIFACT_ROOT / "ews",
        "champion": "ews_xgboost_champion.joblib",
        "schema": "ews_feature_schema.json",
        "metadata": "ews_metadata.json",
        "manifest": "ews_manifest.json",
        "metrics": "ews_metrics.json",
        "policy": "ews_decision_policy.json",
        "reference": "ews_reference_stats.json",
        "importance": "ews_feature_importance.csv",
        "requirements": "ews_requirements.txt",
    },
    "lgd": {
        "task": "regression",
        "folder": ARTIFACT_ROOT / "lgd",
        "champion": "final_lgd_xgboost_new.pkl",
        "schema": "lgd_feature_schema.json",
        "metadata": "lgd_metadata.json",
        "manifest": "lgd_manifest.json",
        "metrics": "lgd_metrics.json",
        "policy": "lgd_decision_policy.json",
        "reference": "lgd_reference_stats.json",
        "importance": "lgd_feature_importance.csv",
        "requirements": "lgd_requirements.txt",
    },
    "pd_cluster": {
        "task": "clustering",
        "folder": ARTIFACT_ROOT / "pd_cluster",
        "champion": "pd_cluster_champion.joblib",
        "schema": "pd_cluster_feature_schema.json",
        "metadata": "pd_cluster_metadata.json",
        "manifest": "pd_cluster_manifest.json",
        "metrics": "pd_cluster_metrics.json",
        "reference": "pd_cluster_reference_stats.json",
        "profiles": "pd_cluster_profiles.json",
        "profile_csv": "pd_cluster_profile.csv",
        "summary_csv": "pd_cluster_summary.csv",
        "importance": None,
        "requirements": "pd_cluster_requirements.txt",
    },
}

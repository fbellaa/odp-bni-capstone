from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


MODEL_KEYS = ("pd", "ews", "lgd", "pd_cluster")
REQUIRED_ML_TOOLS = ("predict_pd", "predict_ews", "predict_lgd", "predict_pd_cluster")
POLICY_RAG_TOOL = "query_credit_policy"
REQUIRED_AGENT_TOOLS = REQUIRED_ML_TOOLS + (POLICY_RAG_TOOL,)


class Evidence(BaseModel):
    """Provenance for one value extracted from RM input/document."""

    model_config = ConfigDict(extra="forbid")

    source_document: str | None = None
    page: int | None = None
    quote: str | None = None
    extraction_method: str | None = None


class ExtractedValue(BaseModel):
    """A value copied from source material; not a value invented by an LLM."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    unit: str | None = None
    explicit_in_source: bool = True
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: Evidence | None = None


class BorrowerExtraction(BaseModel):
    """Structured borrower facts produced by the structured extractor from documents/manual notes.

    `direct_model_features` may only contain values that are explicitly written in
    the source. Derived ratios/calculations belong to deterministic Python feature
    engineering, not the LLM.
    """

    model_config = ConfigDict(extra="forbid")

    borrower_name: str | None = None
    raw_facts: dict[str, ExtractedValue] = Field(default_factory=dict)
    direct_model_features: dict[str, dict[str, ExtractedValue]] = Field(
        default_factory=lambda: {k: {} for k in MODEL_KEYS}
    )
    missing_or_ambiguous: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelExplanation(BaseModel):
    available: bool = False
    scope: str = "none"
    basis: str | None = None
    top_factors: list[dict[str, Any]] = Field(default_factory=list)
    warning: str | None = None

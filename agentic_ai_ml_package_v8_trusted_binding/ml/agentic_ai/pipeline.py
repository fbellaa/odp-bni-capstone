from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .agent import AgentResult, QwenMLAgent
from .artifacts import ArtifactStore
from .config import SETTINGS, Settings
from .document_extraction import (
    DocumentExtractionResult,
    extract_documents,
    extract_documents_multimodal,
)
from .document_mapper import DeterministicDocumentMapper, merge_extractions
from .document_reducer import ReductionResult, reduce_documents_for_extraction
from .feature_engineering import FeatureEngineer
from .narrator import SahabatNarrator
from .parser import QwenStructuredExtractor
from .schemas import BorrowerExtraction, MODEL_KEYS


@dataclass
class PipelineResult:
    documents: DocumentExtractionResult | None
    extraction: BorrowerExtraction
    feature_context: dict[str, dict[str, Any]]
    agent: AgentResult
    answer: str
    extraction_input_text: str = ""
    reduction: ReductionResult | None = None
    extraction_strategy: str = "deterministic_document_mapper"

    @property
    def model_results(self) -> dict[str, dict[str, Any]]:
        return self.agent.record.last_success_by_name


class CreditRiskAIPipeline:
    """V7 hybrid pipeline.

    Documents -> deterministic mapper -> optional Qwen semantic fallback ->
    deterministic feature engineering -> Qwen mandatory ML tool calling ->
    verified results -> SahabatAI narrator.

    Qwen is deliberately NOT responsible for calculating/mapping the primary model
    features from normal financial statements. That responsibility lives in auditable
    Python rules. Qwen remains the agent/tool orchestrator.
    """

    def __init__(
        self,
        extractor: QwenStructuredExtractor | None = None,
        document_mapper: DeterministicDocumentMapper | None = None,
        feature_engineer: FeatureEngineer | None = None,
        agent: QwenMLAgent | None = None,
        narrator: SahabatNarrator | None = None,
        store: ArtifactStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.s = settings or SETTINGS
        self.store = store or ArtifactStore()
        self.extractor = extractor or QwenStructuredExtractor(settings=self.s)
        self.document_mapper = document_mapper or DeterministicDocumentMapper()
        self.feature_engineer = feature_engineer or FeatureEngineer(self.store)
        self.agent = agent or QwenMLAgent(settings=self.s)
        self.narrator = narrator or SahabatNarrator(settings=self.s)

    def feature_catalog(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for key in MODEL_KEYS:
            out[key] = [
                {"name": f.name, "dtype": f.dtype, "description": f.description}
                for f in self.store.feature_defs(key)
            ]
        return out

    @staticmethod
    def _qwen_context(feature_context: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            k: {
                "features": v.get("features", {}),
                "missing_feature_names": v.get("missing_feature_names", []),
            }
            for k, v in feature_context.items()
        }

    def _reduced_text(self, docs: DocumentExtractionResult) -> tuple[str, ReductionResult]:
        reduction = reduce_documents_for_extraction(
            docs,
            feature_catalog=None,  # raw accounting/application keywords only
            max_chars=self.s.extraction_max_chars,
        )
        return reduction.text, reduction

    def extract_from_documents(
        self,
        doc_result: DocumentExtractionResult,
        *,
        manual_input: dict[str, Any] | None = None,
        use_qwen_semantic_fallback: bool | None = None,
    ) -> tuple[BorrowerExtraction, str, ReductionResult | None, str]:
        primary = self.document_mapper.extract(doc_result)
        use_fallback = (
            self.s.use_qwen_semantic_fallback
            if use_qwen_semantic_fallback is None
            else bool(use_qwen_semantic_fallback)
        )
        if not use_fallback:
            return primary, "", None, "deterministic_document_mapper"

        # Optional fallback sees only a compact document excerpt and can fill missing
        # canonical facts. Deterministic values always win on merge.
        llm_text, reduction = self._reduced_text(doc_result)
        fallback = self.extractor.extract(
            llm_text,
            feature_catalog=None,
            manual_input=manual_input,
        )
        merged = merge_extractions(primary, fallback)
        return merged, llm_text, reduction, "deterministic_mapper_plus_qwen_fallback"

    def run_text(
        self,
        document_text: str,
        *,
        manual_input: dict[str, Any] | None = None,
        execute_tools: bool = True,
        document_warnings: list[str] | None = None,
    ) -> PipelineResult:
        # Text-only mode uses Qwen because no page-aware DocumentExtractionResult exists.
        extraction = self.extractor.extract(
            document_text,
            feature_catalog=None,
            manual_input=manual_input,
        )
        feature_context = self.feature_engineer.build(extraction, manual_input=manual_input)
        agent = self.agent.run(self._qwen_context(feature_context), execute_tools=execute_tools)
        answer = self.narrator.narrate(
            extraction=extraction,
            feature_context=feature_context,
            agent=agent,
            document_warnings=document_warnings,
        )
        return PipelineResult(
            None,
            extraction,
            feature_context,
            agent,
            answer,
            extraction_input_text=document_text,
            extraction_strategy="qwen_text_only",
        )

    def run_extracted_documents(
        self,
        doc_result: DocumentExtractionResult,
        *,
        manual_input: dict[str, Any] | None = None,
        execute_tools: bool = True,
        use_qwen_semantic_fallback: bool | None = None,
        extraction_mode: str | None = None,  # kept for backward compatibility
    ) -> PipelineResult:
        extraction, llm_text, reduction, strategy = self.extract_from_documents(
            doc_result,
            manual_input=manual_input,
            use_qwen_semantic_fallback=use_qwen_semantic_fallback,
        )
        feature_context = self.feature_engineer.build(extraction, manual_input=manual_input)
        agent = self.agent.run(self._qwen_context(feature_context), execute_tools=execute_tools)
        answer = self.narrator.narrate(
            extraction=extraction,
            feature_context=feature_context,
            agent=agent,
            document_warnings=doc_result.warnings,
        )
        return PipelineResult(
            documents=doc_result,
            extraction=extraction,
            feature_context=feature_context,
            agent=agent,
            answer=answer,
            extraction_input_text=llm_text,
            reduction=reduction,
            extraction_strategy=strategy,
        )

    def run_documents(
        self,
        documents: Iterable[Any],
        *,
        manual_input: dict[str, Any] | None = None,
        execute_tools: bool = True,
        multimodal: bool = True,
        use_qwen_semantic_fallback: bool | None = None,
        extraction_mode: str | None = None,
    ) -> PipelineResult:
        if multimodal:
            doc_result = extract_documents_multimodal(
                documents,
                ollama_url=self.s.ollama_host,
                vlm_model=self.s.vlm_model,
            )
        else:
            doc_result = extract_documents(documents)
        return self.run_extracted_documents(
            doc_result,
            manual_input=manual_input,
            execute_tools=execute_tools,
            use_qwen_semantic_fallback=use_qwen_semantic_fallback,
        )

    def run(
        self,
        document_text: str,
        manual_input: dict[str, Any] | None = None,
        *,
        execute_tools: bool = True,
    ) -> PipelineResult:
        return self.run_text(document_text, manual_input=manual_input, execute_tools=execute_tools)

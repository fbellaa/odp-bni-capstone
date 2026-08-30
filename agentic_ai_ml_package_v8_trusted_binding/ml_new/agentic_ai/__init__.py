"""V9 Agentic AI credit-risk package with policy RAG tool calling."""

__version__ = "9.3.0"

from .document_mapper import DeterministicDocumentMapper
from .pipeline import CreditRiskAIPipeline, PipelineResult
from .parser import QwenStructuredExtractor
from .narrator import QwenNarrator

__all__ = [
    "CreditRiskAIPipeline",
    "PipelineResult",
    "DeterministicDocumentMapper",
    "QwenStructuredExtractor",
    "QwenNarrator",
    "__version__",
]

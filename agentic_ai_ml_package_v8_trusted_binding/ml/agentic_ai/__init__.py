"""V8 trusted-binding Agentic AI credit-risk package."""

__version__ = "8.0.0"

from .document_mapper import DeterministicDocumentMapper
from .pipeline import CreditRiskAIPipeline, PipelineResult
from .parser import QwenStructuredExtractor

__all__ = [
    "CreditRiskAIPipeline",
    "PipelineResult",
    "DeterministicDocumentMapper",
    "QwenStructuredExtractor",
    "__version__",
]

"""Core services package initialization."""

from pdf_pipeline.core.exceptions import (
    PipelineError,
    GeminiAPIError,
    GeminiContextLimitError,
    GeminiMaxTokensExceededError,
    GeminiSchemaValidationError,
    JSONRepairFailedError,
    BatchSplittingError,
    StorageError,
    UnrecoverablePageError,
)
from pdf_pipeline.core.audit_logger import TokenAuditLogger, InMemoryAuditSink, AuditSink
from pdf_pipeline.core.gemini_client import GeminiClientWrapper

__all__ = [
    "PipelineError",
    "GeminiAPIError",
    "GeminiContextLimitError",
    "GeminiMaxTokensExceededError",
    "GeminiSchemaValidationError",
    "JSONRepairFailedError",
    "BatchSplittingError",
    "StorageError",
    "UnrecoverablePageError",
    "TokenAuditLogger",
    "InMemoryAuditSink",
    "AuditSink",
    "GeminiClientWrapper",
]

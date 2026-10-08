"""BigQuery LLM Call Audit Model.

Authoritative schema for tracking every Gemini invocation (success, retried, failed)
with detailed token breakdown and decoupled cost attribution.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CallStatus(str, Enum):
    """Execution status of an LLM call."""

    SUCCESS = "SUCCESS"
    RETRYING = "RETRYING"
    FAILED = "FAILED"
    FALLBACK_TRIGGERED = "FALLBACK_TRIGGERED"


class OperationType(str, Enum):
    """Categorization of Gemini invocation purpose."""

    TOKEN_COUNT = "TOKEN_COUNT"
    STAGE1_CLASSIFICATION = "STAGE1_CLASSIFICATION"
    STAGE2_EXTRACTION = "STAGE2_EXTRACTION"
    JSON_REPAIR = "JSON_REPAIR"
    EMBEDDING_GENERATION = "EMBEDDING_GENERATION"


class LLMCallAuditRecord(BaseModel):
    """Audit record directly serializable into BigQuery project_id.dataset_id.llm_call_audit."""

    audit_id: str = Field(description="Unique UUID for this specific call attempt")
    correlation_id: str = Field(description="Distributed trace or workflow correlation identifier")
    document_id: str = Field(description="Parent document identifier")
    processing_run_id: str = Field(description="Pipeline execution run ID")
    batch_id: str = Field(description="Batch identifier")
    operation_type: str = Field(description="e.g. STAGE1_CLASSIFICATION, STAGE2_EXTRACTION")
    model_name: str = Field(description="Configured model name requested")
    model_version: Optional[str] = Field(default=None, description="Actual returned model version")
    page_start: Optional[int] = Field(default=None, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)

    # Token Usage Metrics (authoritative post-execution metrics)
    prompt_token_count: int = Field(default=0, ge=0)
    output_token_count: int = Field(default=0, ge=0)
    thinking_token_count: int = Field(default=0, ge=0, description="Tokens consumed during internal thinking")
    cached_token_count: int = Field(default=0, ge=0, description="Tokens read from context cache")
    total_token_count: int = Field(default=0, ge=0)

    # Financial Cost Estimation (Derived from decoupled pricing registry)
    estimated_input_cost: Decimal = Field(default=Decimal("0.0"))
    estimated_output_cost: Decimal = Field(default=Decimal("0.0"))
    estimated_thinking_cost: Decimal = Field(default=Decimal("0.0"))
    estimated_total_cost: Decimal = Field(default=Decimal("0.0"))
    currency: str = Field(default="USD")

    # Timing & Latency
    request_timestamp: datetime = Field(description="UTC timestamp when request was initiated")
    response_timestamp: datetime = Field(description="UTC timestamp when response was received")
    latency_ms: int = Field(ge=0, description="End-to-end duration of the network invocation")

    # Reliability & Execution Diagnostics
    retry_count: int = Field(default=0, ge=0, description="Attempt sequence number (0 = initial call)")
    status: CallStatus = Field(description="SUCCESS, RETRYING, FAILED, or FALLBACK_TRIGGERED")
    error_code: Optional[str] = Field(default=None, description="Standardized error code or HTTP status")
    error_message: Optional[str] = Field(default=None, description="Error diagnostics without sensitive payload")

    # Audit Trail
    created_by: str = Field(default="gemini-pdf-pipeline-worker")
    created_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_bigquery_row(self) -> dict:
        """Converts model to BigQuery client compatible row dictionary."""
        return {
            "audit_id": self.audit_id,
            "correlation_id": self.correlation_id,
            "document_id": self.document_id,
            "processing_run_id": self.processing_run_id,
            "batch_id": self.batch_id,
            "operation_type": self.operation_type,
            "model_name": self.model_name,
            "model_version": self.model_version or "",
            "page_start": self.page_start,
            "page_end": self.page_end,
            "prompt_token_count": self.prompt_token_count,
            "output_token_count": self.output_token_count,
            "thinking_token_count": self.thinking_token_count,
            "cached_token_count": self.cached_token_count,
            "total_token_count": self.total_token_count,
            "estimated_input_cost": float(self.estimated_input_cost),
            "estimated_output_cost": float(self.estimated_output_cost),
            "estimated_thinking_cost": float(self.estimated_thinking_cost),
            "estimated_total_cost": float(self.estimated_total_cost),
            "currency": self.currency,
            "request_timestamp": self.request_timestamp.isoformat(),
            "response_timestamp": self.response_timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "error_code": self.error_code or "",
            "error_message": self.error_message or "",
            "created_by": self.created_by,
            "created_timestamp": self.created_timestamp.isoformat(),
            "updated_timestamp": self.updated_timestamp.isoformat(),
        }

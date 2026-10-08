"""BigQuery Token Audit Logger.

Logs every Gemini invocation (successful, retried, failed) with post-execution
token usage metadata and decoupled financial cost calculations.
Never stores hidden Chain-of-Thought reasoning; logs thinking token count only.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from pdf_pipeline.config.pricing import pricing_registry, PricingRegistry
from pdf_pipeline.config.settings import settings
from pdf_pipeline.models.audit import CallStatus, LLMCallAuditRecord

logger = logging.getLogger("pdf_pipeline.audit")


@runtime_checkable
class AuditSink(Protocol):
    """Protocol for persisting audit records (e.g. BigQuery or In-Memory)."""

    def insert_audit_record(self, record: LLMCallAuditRecord) -> None:
        """Persist a single audit record."""
        ...


class InMemoryAuditSink:
    """Thread-safe in-memory sink for testing and local runs."""

    def __init__(self):
        self.records: list[LLMCallAuditRecord] = []

    def insert_audit_record(self, record: LLMCallAuditRecord) -> None:
        self.records.append(record)
        logger.info(
            f"[AUDIT] op={record.operation_type} status={record.status} "
            f"tokens={record.total_token_count} (prompt={record.prompt_token_count}, "
            f"out={record.output_token_count}, think={record.thinking_token_count}) "
            f"cost=${record.estimated_total_cost:.6f} latency={record.latency_ms}ms"
        )


class TokenAuditLogger:
    """Service responsible for standardizing and recording all LLM call telemetry."""

    def __init__(
        self,
        sink: Optional[AuditSink] = None,
        pricing: Optional[PricingRegistry] = None,
    ):
        self.sink = sink or InMemoryAuditSink()
        self.pricing = pricing or pricing_registry

    def set_sink(self, sink: AuditSink) -> None:
        """Hot-swap the audit sink (e.g., once BigQuery client initializes)."""
        self.sink = sink

    def log_call(
        self,
        operation_type: str,
        document_id: str,
        processing_run_id: str,
        batch_id: str,
        model_name: str,
        request_timestamp: datetime,
        response_timestamp: datetime,
        status: CallStatus,
        model_version: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        prompt_token_count: int = 0,
        output_token_count: int = 0,
        thinking_token_count: int = 0,
        cached_token_count: int = 0,
        total_token_count: int = 0,
        retry_count: int = 0,
        correlation_id: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> LLMCallAuditRecord:
        """Create, cost-evaluate, and persist an audit record."""
        # Calculate latency
        latency_ms = int(
            (response_timestamp - request_timestamp).total_seconds() * 1000
        )
        if latency_ms < 0:
            latency_ms = 0

        # Calculate costs using decoupled pricing registry
        cost_estimate = self.pricing.calculate_cost(
            model_name=model_name,
            prompt_tokens=prompt_token_count,
            output_tokens=output_token_count,
            thinking_tokens=thinking_token_count,
            cached_tokens=cached_token_count,
        )

        record = LLMCallAuditRecord(
            audit_id=f"audit_{uuid.uuid4().hex[:16]}",
            correlation_id=correlation_id or f"corr_{uuid.uuid4().hex[:12]}",
            document_id=document_id,
            processing_run_id=processing_run_id,
            batch_id=batch_id,
            operation_type=operation_type,
            model_name=model_name,
            model_version=model_version or "",
            page_start=page_start,
            page_end=page_end,
            prompt_token_count=prompt_token_count,
            output_token_count=output_token_count,
            thinking_token_count=thinking_token_count,
            cached_token_count=cached_token_count,
            total_token_count=total_token_count or (prompt_token_count + output_token_count + thinking_token_count),
            estimated_input_cost=cost_estimate.estimated_input_cost,
            estimated_output_cost=cost_estimate.estimated_output_cost,
            estimated_thinking_cost=cost_estimate.estimated_thinking_cost,
            estimated_total_cost=cost_estimate.estimated_total_cost,
            currency=cost_estimate.currency,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
            latency_ms=latency_ms,
            retry_count=retry_count,
            status=status,
            error_code=error_code,
            error_message=error_message,
            created_by=created_by or settings.created_by,
            created_timestamp=datetime.now(timezone.utc),
            updated_timestamp=datetime.now(timezone.utc),
        )

        try:
            self.sink.insert_audit_record(record)
        except Exception as e:
            logger.error(f"Failed to record audit log to sink: {e}", exc_info=True)

        return record

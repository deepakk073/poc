"""Unit tests for Decoupled Pricing Registry and BigQuery Token-Audit Logger."""

from datetime import datetime, timezone
from decimal import Decimal
from pdf_pipeline.config.pricing import ModelPricing, PricingRegistry
from pdf_pipeline.core.audit_logger import InMemoryAuditSink, TokenAuditLogger
from pdf_pipeline.models.audit import CallStatus, OperationType


def test_pricing_calculation_gemini_flash():
    """Validates cost formula for prompt, candidate, thinking, and cached tokens."""
    registry = PricingRegistry()
    # gemini-3.8-flash pricing:
    # Input: $0.15 / 1M ($0.00000015/tok)
    # Output: $0.60 / 1M ($0.0000006/tok)
    # Thinking: $0.60 / 1M ($0.0000006/tok)
    # Cached: $0.0375 / 1M ($0.0000000375/tok)

    cost = registry.calculate_cost(
        model_name="gemini-3.8-flash",
        prompt_tokens=100_000,
        output_tokens=10_000,
        thinking_tokens=5_000,
        cached_tokens=20_000,
    )

    # Active input = 100k - 20k = 80k. 80,000 * 0.15 / 1M = 0.012000
    assert cost.estimated_input_cost == Decimal("0.012000")
    # Cached = 20,000 * 0.0375 / 1M = 0.000750
    assert cost.estimated_cached_cost == Decimal("0.000750")
    # Output = 10,000 * 0.60 / 1M = 0.006000
    assert cost.estimated_output_cost == Decimal("0.006000")
    # Thinking = 5,000 * 0.60 / 1M = 0.003000
    assert cost.estimated_thinking_cost == Decimal("0.003000")
    # Total = 0.012 + 0.00075 + 0.006 + 0.003 = 0.021750
    assert cost.estimated_total_cost == Decimal("0.021750")
    assert cost.currency == "USD"


def test_audit_logger_records_tokens_and_cost():
    """Validates that audit record contains authoritative token metrics and no raw reasoning text."""
    sink = InMemoryAuditSink()
    logger = TokenAuditLogger(sink=sink)

    req_time = datetime(2026, 10, 8, 10, 0, 0, tzinfo=timezone.utc)
    resp_time = datetime(2026, 10, 8, 10, 0, 2, tzinfo=timezone.utc)  # 2000ms latency

    record = logger.log_call(
        operation_type=OperationType.STAGE1_CLASSIFICATION.value,
        document_id="doc_audit_test",
        processing_run_id="run_audit_001",
        batch_id="batch_001",
        model_name="gemini-3.8-flash",
        request_timestamp=req_time,
        response_timestamp=resp_time,
        status=CallStatus.SUCCESS,
        page_start=1,
        page_end=10,
        prompt_token_count=5000,
        output_token_count=1200,
        thinking_token_count=450,
        cached_token_count=1000,
        total_token_count=6650,
        retry_count=0,
    )

    assert len(sink.records) == 1
    assert record.latency_ms == 2000
    assert record.prompt_token_count == 5000
    assert record.thinking_token_count == 450
    assert record.cached_token_count == 1000
    assert record.total_token_count == 6650
    assert record.status == CallStatus.SUCCESS

    # Verify BigQuery row conversion
    bq_row = record.to_bigquery_row()
    assert bq_row["audit_id"].startswith("audit_")
    assert bq_row["document_id"] == "doc_audit_test"
    assert bq_row["operation_type"] == "STAGE1_CLASSIFICATION"
    assert "chain_of_thought" not in bq_row  # Invariant: Never store raw thinking reasoning
    assert bq_row["thinking_token_count"] == 450

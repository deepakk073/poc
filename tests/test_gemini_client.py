"""Unit tests for Unified Gemini Client Wrapper and JSON Repair Fallback."""

import json
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel, Field

from pdf_pipeline.core.audit_logger import InMemoryAuditSink, TokenAuditLogger
from pdf_pipeline.core.exceptions import GeminiAPIError, GeminiMaxTokensExceededError
from pdf_pipeline.core.gemini_client import GeminiClientWrapper
from pdf_pipeline.models.audit import CallStatus


class SampleSchema(BaseModel):
    title: str
    amount: float


def test_gemini_client_successful_structured_call():
    sink = InMemoryAuditSink()
    audit_logger = TokenAuditLogger(sink=sink)
    mock_client = MagicMock()

    # Create mock response
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"title": "Test Invoice", "amount": 99.95})
    mock_resp.model_version = "gemini-3.8-flash-001"
    mock_resp.candidates = [MagicMock(finish_reason="STOP")]

    usage = MagicMock()
    usage.prompt_token_count = 120
    usage.candidates_token_count = 35
    usage.thoughts_token_count = 10
    usage.cached_content_token_count = 0
    usage.total_token_count = 165
    mock_resp.usage_metadata = usage

    mock_client.models.generate_content.return_value = mock_resp

    wrapper = GeminiClientWrapper(audit_logger=audit_logger, raw_client=mock_client)

    result = wrapper.call_gemini(
        model_name="gemini-3.8-flash",
        prompt="Extract title and amount",
        contents="Sample text",
        operation_type="STAGE2_EXTRACTION",
        document_id="doc_1",
        processing_run_id="run_1",
        batch_id="b_1",
        response_schema=SampleSchema,
    )

    assert result["title"] == "Test Invoice"
    assert result["amount"] == 99.95
    assert len(sink.records) == 1
    assert sink.records[0].status == CallStatus.SUCCESS
    assert sink.records[0].thinking_token_count == 10


def test_gemini_client_json_repair_fallback():
    """Validates that a malformed response triggers JSON repair prompt and successfully recovers."""
    sink = InMemoryAuditSink()
    audit_logger = TokenAuditLogger(sink=sink)
    mock_client = MagicMock()

    # First call returns malformed JSON with markdown and trailing comma
    bad_resp = MagicMock()
    bad_resp.text = '```json\n{"title": "Repaired Doc", "amount": 150.0,}\n```'  # invalid JSON due to trailing comma
    bad_resp.candidates = [MagicMock(finish_reason="STOP")]
    bad_resp.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=30, thoughts_token_count=0, cached_content_token_count=0, total_token_count=130)

    # Second call (repair call) returns clean JSON
    repair_resp = MagicMock()
    repair_resp.text = '{"title": "Repaired Doc", "amount": 150.0}'
    repair_resp.candidates = [MagicMock(finish_reason="STOP")]
    repair_resp.usage_metadata = MagicMock(prompt_token_count=150, candidates_token_count=25, thoughts_token_count=0, cached_content_token_count=0, total_token_count=175)

    mock_client.models.generate_content.side_effect = [bad_resp, repair_resp]

    wrapper = GeminiClientWrapper(audit_logger=audit_logger, raw_client=mock_client)

    result = wrapper.call_gemini(
        model_name="gemini-3.8-flash",
        prompt="Extract",
        contents="Text",
        operation_type="STAGE2_EXTRACTION",
        document_id="doc_2",
        processing_run_id="run_2",
        batch_id="b_2",
        response_schema=SampleSchema,
    )

    assert result["title"] == "Repaired Doc"
    assert result["amount"] == 150.0
    # Both initial call and repair call must be audited
    assert len(sink.records) == 2
    assert sink.records[0].operation_type == "JSON_REPAIR"
    assert sink.records[1].status == CallStatus.SUCCESS


def test_gemini_client_max_tokens_truncation():
    """Validates that MAX_TOKENS finish_reason raises GeminiMaxTokensExceededError."""
    sink = InMemoryAuditSink()
    audit_logger = TokenAuditLogger(sink=sink)
    mock_client = MagicMock()

    truncated_resp = MagicMock()
    truncated_resp.text = '{"title": "Incomplete...'
    truncated_resp.candidates = [MagicMock(finish_reason="MAX_TOKENS")]
    truncated_resp.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=8192, thoughts_token_count=0, cached_content_token_count=0, total_token_count=8292)

    mock_client.models.generate_content.return_value = truncated_resp

    wrapper = GeminiClientWrapper(audit_logger=audit_logger, raw_client=mock_client)

    with pytest.raises(GeminiMaxTokensExceededError):
        wrapper.call_gemini(
            model_name="gemini-3.8-flash",
            prompt="Extract",
            contents="Text",
            operation_type="STAGE2_EXTRACTION",
            document_id="doc_3",
            processing_run_id="run_3",
            batch_id="b_3",
            response_schema=SampleSchema,
        )

    # Invariant: Failed/truncated call must be logged in audit table
    assert len(sink.records) == 1
    assert sink.records[0].status == CallStatus.FALLBACK_TRIGGERED
    assert sink.records[0].error_code == "FINISH_REASON_MAX_TOKENS"

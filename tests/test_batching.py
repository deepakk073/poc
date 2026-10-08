"""Unit tests for PDF Batching and Overlap Management."""

import pytest
from unittest.mock import MagicMock
from pdf_pipeline.config.settings import PipelineSettings
from pdf_pipeline.models.document import PageBatch
from pdf_pipeline.services.batching_service import PDFBatchingService


def test_batching_28_pages_with_1_page_overlap():
    """Validates exact specification: 28 pages, batch_size=10, overlap=1.

    Expected batches:
    Batch 1: pages 1–10
    Batch 2: pages 10–19
    Batch 3: pages 19–28
    """
    settings = PipelineSettings(batch_size=10, overlap_size=1)
    service = PDFBatchingService(pipeline_settings=settings)

    batches = service.create_batches(
        document_id="doc_test_123",
        processing_run_id="run_test_001",
        total_pages=28,
        batch_size=10,
        overlap_size=1,
    )

    assert len(batches) == 3

    # Batch 1: 1-10
    assert batches[0].batch_id == "batch_001"
    assert batches[0].page_start == 1
    assert batches[0].page_end == 10
    assert len(batches[0].pages) == 10
    assert [p.page_number for p in batches[0].pages] == list(range(1, 11))
    # In Batch 1, all 1-10 are primary
    assert batches[0].primary_page_numbers == list(range(1, 11))

    # Batch 2: 10-19
    assert batches[1].batch_id == "batch_002"
    assert batches[1].page_start == 10
    assert batches[1].page_end == 19
    assert len(batches[1].pages) == 10
    assert [p.page_number for p in batches[1].pages] == list(range(10, 20))
    # In Batch 2, page 10 is overlap (already primary in batch 1), 11-19 are primary
    p10 = next(p for p in batches[1].pages if p.page_number == 10)
    assert p10.is_primary is False
    assert batches[1].primary_page_numbers == list(range(11, 20))

    # Batch 3: 19-28
    assert batches[2].batch_id == "batch_003"
    assert batches[2].page_start == 19
    assert batches[2].page_end == 28
    assert len(batches[2].pages) == 10
    assert [p.page_number for p in batches[2].pages] == list(range(19, 29))
    # In Batch 3, page 19 is overlap, 20-28 are primary
    p19 = next(p for p in batches[2].pages if p.page_number == 19)
    assert p19.is_primary is False
    assert batches[2].primary_page_numbers == list(range(20, 29))


def test_primary_page_uniqueness_across_all_batches():
    """Guarantees every page from 1 to N has exactly one primary batch assignment."""
    service = PDFBatchingService()
    total_pages = 45
    batches = service.create_batches(
        document_id="doc_unique_test",
        processing_run_id="run_001",
        total_pages=total_pages,
        batch_size=10,
        overlap_size=1,
    )

    primary_pages_collected = []
    for b in batches:
        primary_pages_collected.extend(b.primary_page_numbers)

    # Invariant: exactly 45 primary pages, no duplicates, perfectly sorted 1..45
    assert len(primary_pages_collected) == total_pages
    assert len(set(primary_pages_collected)) == total_pages
    assert primary_pages_collected == list(range(1, total_pages + 1))


def test_batch_splitting_on_proactive_token_limit():
    """Validates that proactive token threshold triggers recursive batch splitting."""
    settings = PipelineSettings(
        batch_size=10,
        overlap_size=1,
        token_safety_threshold=1000,
        min_batch_size=1,
    )
    mock_audit = MagicMock()
    service = PDFBatchingService(pipeline_settings=settings, audit_logger=mock_audit)

    batch = service.create_batches("doc_split", "run_1", 10, batch_size=10, overlap_size=1)[0]
    assert len(batch.pages) == 10

    # Mock client wrapper returning 2500 tokens (exceeding 1000 threshold)
    mock_client = MagicMock()
    # First check returns 2500, then child checks return 400 (under threshold)
    mock_client.estimate_tokens.side_effect = [2500, 400, 400]

    optimized = service.estimate_and_optimize_batch(
        batch=batch,
        gemini_client=mock_client,
        batch_content_bytes=b"dummy",
        model_name="gemini-3.8-flash",
    )

    # 10 pages split into two 5-page child batches
    assert len(optimized) == 2
    assert len(optimized[0].pages) == 5
    assert len(optimized[1].pages) == 5
    assert optimized[0].is_split_child is True
    assert optimized[1].is_split_child is True
    assert mock_audit.log_call.called

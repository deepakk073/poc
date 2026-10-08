"""End-to-End Integration Tests for PDF Processing Orchestrator."""

import json
import os
import tempfile
from unittest.mock import MagicMock
import pytest

from pdf_pipeline.config.settings import PipelineSettings
from pdf_pipeline.core.gemini_client import GeminiClientWrapper
from pdf_pipeline.models.classification import BatchClassificationResponse, ClassifiedPage
from pdf_pipeline.models.citation import Citation
from pdf_pipeline.models.document import DocumentStatus
from pdf_pipeline.models.extraction import BatchExtractionResponse, ExtractedPage
from pdf_pipeline.repositories.bigquery_repo import BigQueryRepository
from pdf_pipeline.repositories.gcs_repo import GCSStorageRepository
from pdf_pipeline.repositories.vector_store_repo import InMemoryVectorStoreRepository
from pdf_pipeline.services.orchestrator import PDFProcessingOrchestrator
from tests.conftest import create_mock_pdf_bytes


def test_end_to_end_orchestrator_pipeline():
    """Runs full pipeline on a 28-page PDF, verifying batching, extraction, overlap reconciliation,

    lineage preservation, and the 11 production validation checks.
    """
    settings = PipelineSettings(
        project_id="test-proj",
        batch_size=10,
        overlap_size=1,
        token_safety_threshold=100000,
    )

    db = BigQueryRepository(settings)
    storage = GCSStorageRepository(settings, local_fallback_dir=tempfile.mkdtemp())
    vector_store = InMemoryVectorStoreRepository()

    # Create mock Gemini client with realistic structured responses
    mock_raw_client = MagicMock()

    def mock_generate_content(model, contents, config):
        resp = MagicMock()
        resp.candidates = [MagicMock(finish_reason="STOP")]
        usage = MagicMock()
        usage.prompt_token_count = 1500
        usage.candidates_token_count = 600
        usage.thoughts_token_count = 120
        usage.cached_content_token_count = 0
        usage.total_token_count = 2220
        resp.usage_metadata = usage

        resp_schema = config.get("response_schema")
        # Stage 1 Classification mock
        if resp_schema == BatchClassificationResponse:
            # Detect which batch pages are being requested
            prompt_str = contents[0] if contents else ""
            pages_list = []
            if "1 through 10" in prompt_str or "1, 2" in prompt_str:
                page_nums = list(range(1, 11))
            elif "10 through 19" in prompt_str:
                page_nums = list(range(10, 20))
            else:
                page_nums = list(range(19, 29))

            for p in page_nums:
                pages_list.append(
                    {
                        "page_number": p,
                        "classification": "invoice" if p % 2 == 0 else "work_order",
                        "classification_description": "Standard document",
                        "classification_source": "known",
                        "confidence_score": 0.95,
                        "content_types": ["typed_text", "table"],
                        "page_metadata": {"has_header": True, "has_footer": True, "language": "en"},
                    }
                )
            resp.text = json.dumps({"pages": pages_list, "document_metadata": {"title": "Sample Doc"}})
            return resp

        # Stage 2 Extraction mock
        elif resp_schema == BatchExtractionResponse:
            prompt_str = contents[0] if contents else ""
            if "1 through 10" in prompt_str:
                page_nums = list(range(1, 11))
                b_id = "batch_001"
            elif "10 through 19" in prompt_str:
                page_nums = list(range(10, 20))
                b_id = "batch_002"
            else:
                page_nums = list(range(19, 29))
                b_id = "batch_003"

            ext_pages = []
            for p in page_nums:
                ext_pages.append(
                    {
                        "page_number": p,
                        "classification": "invoice" if p % 2 == 0 else "work_order",
                        "extracted_data": {"doc_num": f"DOC-{p}", "amount": float(p * 100)},
                        "text_content": f"This is full text content for page {p}.",
                        "tables": [],
                        "images": [],
                        "citations": [
                            {
                                "document_id": "test_doc",
                                "source_filename": "test.pdf",
                                "page_number": p,
                                "batch_id": b_id,
                                "classification": "invoice" if p % 2 == 0 else "work_order",
                                "source_text": f"DOC-{p}",
                                "field_path": "doc_num",
                                "confidence_score": 0.98,
                            }
                        ],
                        "confidence_score": 0.95,
                    }
                )
            resp.text = json.dumps({"document_id": "test_doc", "processing_run_id": "run_1", "batch_id": b_id, "pages": ext_pages})
            return resp

        # Default fallback
        resp.text = "{}"
        return resp

    mock_raw_client.models.generate_content.side_effect = mock_generate_content
    count_resp = MagicMock()
    count_resp.total_tokens = 800
    mock_raw_client.models.count_tokens.return_value = count_resp

    # Initialize Gemini wrapper
    from pdf_pipeline.core.audit_logger import TokenAuditLogger
    audit_logger = TokenAuditLogger(sink=db)
    gemini_client = GeminiClientWrapper(audit_logger=audit_logger, pipeline_settings=settings, raw_client=mock_raw_client)

    orchestrator = PDFProcessingOrchestrator(
        pipeline_settings=settings,
        db_repo=db,
        storage_repo=storage,
        vector_store=vector_store,
        gemini_client=gemini_client,
    )

    # Create temporary PDF file with 28 pages
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(create_mock_pdf_bytes(28))
        pdf_path = tmp.name

    try:
        result = orchestrator.process_pdf(pdf_path)

        # 1. Validation Report Verification (all 11 production criteria)
        assert result.validation_report.is_valid is True
        assert result.document_metadata.status == DocumentStatus.COMPLETED
        assert result.document_metadata.page_count == 28

        # 2. Page Completeness & Deduplication (Pages 1 through 28)
        assert len(result.classifications) == 28
        assert len(result.extractions) == 28
        assert [p.page_number for p in result.classifications] == list(range(1, 29))
        assert [p.page_number for p in result.extractions] == list(range(1, 29))

        # 3. Citations Present & Grounded
        assert len(result.citations) >= 28
        for cite in result.citations:
            assert cite.document_id == result.document_metadata.document_id
            assert 1 <= cite.page_number <= 28

        # 4. Embeddings Stored with Lineage
        assert len(result.embeddings) >= 28
        for emb in result.embeddings:
            assert emb.document_id == result.document_metadata.document_id
            assert 1 <= emb.page_number <= 28
            assert len(emb.embedding) == 768

        # 5. BigQuery Persistence
        assert len(db.documents_table) >= 1
        assert len(db.classifications_table) == 28
        assert len(db.extractions_table) == 28
        assert len(db.citations_table) >= 28

        # 6. Audit Trail: every call logged with thinking tokens
        assert len(db.audit_table) >= 6  # 3 batches * 2 stages
        for audit_row in db.audit_table:
            assert audit_row["thinking_token_count"] == 120
            assert audit_row["total_token_count"] > 0
            assert "chain_of_thought" not in audit_row

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

"""Pytest fixtures and mock test doubles."""

import io
from unittest.mock import MagicMock
import pytest

from pdf_pipeline.config.settings import PipelineSettings
from pdf_pipeline.core.audit_logger import InMemoryAuditSink, TokenAuditLogger
from pdf_pipeline.core.gemini_client import GeminiClientWrapper
from pdf_pipeline.models.classification import (
    BatchClassificationResponse,
    ClassifiedPage,
    PageMetadata,
    ProposedClassification,
)
from pdf_pipeline.models.citation import Citation
from pdf_pipeline.models.extraction import (
    BatchExtractionResponse,
    ExtractedPage,
    TableData,
)
from pdf_pipeline.repositories.bigquery_repo import BigQueryRepository
from pdf_pipeline.repositories.gcs_repo import GCSStorageRepository
from pdf_pipeline.repositories.vector_store_repo import InMemoryVectorStoreRepository
from pdf_pipeline.services.schema_manager import DynamicSchemaManager


@pytest.fixture
def test_settings():
    """Isolated settings fixture for unit testing."""
    return PipelineSettings(
        project_id="test-vertex-project",
        location="us-central1",
        batch_size=10,
        overlap_size=1,
        token_safety_threshold=5000,
        max_retries=2,
    )


@pytest.fixture
def mock_audit_sink():
    return InMemoryAuditSink()


@pytest.fixture
def test_audit_logger(mock_audit_sink):
    return TokenAuditLogger(sink=mock_audit_sink)


@pytest.fixture
def mock_raw_gemini_client():
    """Mock simulating google-genai Client behavior."""
    client = MagicMock()

    # Default count_tokens response
    count_resp = MagicMock()
    count_resp.total_tokens = 450
    client.models.count_tokens.return_value = count_resp

    return client


@pytest.fixture
def test_gemini_wrapper(test_audit_logger, test_settings, mock_raw_gemini_client):
    return GeminiClientWrapper(
        audit_logger=test_audit_logger,
        pipeline_settings=test_settings,
        raw_client=mock_raw_gemini_client,
    )


@pytest.fixture
def test_repositories(test_settings):
    """In-memory repositories for database, object storage, and vector store."""
    db = BigQueryRepository(pipeline_settings=test_settings)
    storage = GCSStorageRepository(pipeline_settings=test_settings, local_fallback_dir="/tmp/test_pdf_store")
    vector_store = InMemoryVectorStoreRepository()
    return db, storage, vector_store


def create_mock_pdf_bytes(page_count: int = 28) -> bytes:
    """Generates synthetic minimal PDF file bytes with /Type /Page markers."""
    header = b"%PDF-1.4\n"
    body = b""
    for i in range(1, page_count + 1):
        body += f"\n{i} 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n".encode("utf-8")
    body += b"\n2 0 obj\n<< /Type /Pages /Count " + str(page_count).encode("utf-8") + b" >>\nendobj\n"
    trailer = b"trailer\n<< /Root 2 0 R >>\n%%EOF\n"
    return header + body + trailer

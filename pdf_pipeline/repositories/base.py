"""Repository base interfaces and abstraction protocols."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pdf_pipeline.models.audit import LLMCallAuditRecord
from pdf_pipeline.models.citation import Citation, LineageRecord
from pdf_pipeline.models.classification import ClassifiedPage, ProposedClassification
from pdf_pipeline.models.document import DocumentMetadata
from pdf_pipeline.models.extraction import ExtractedPage


class ObjectStorageRepository(ABC):
    """Abstract interface for storing raw files, page images, and JSON payloads."""

    @abstractmethod
    def upload_bytes(self, destination_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Uploads raw bytes and returns the stored URI (e.g. gs://... or file://...)."""
        pass

    @abstractmethod
    def upload_json(self, destination_path: str, data: Any) -> str:
        """Uploads serialized JSON object and returns the stored URI."""
        pass

    @abstractmethod
    def read_bytes(self, source_path: str) -> bytes:
        """Reads file bytes from storage."""
        pass


class DatabaseRepository(ABC):
    """Abstract interface for persisting relational and structured documents in BigQuery."""

    @abstractmethod
    def save_document_metadata(self, metadata: DocumentMetadata) -> None:
        pass

    @abstractmethod
    def save_classifications(self, classifications: List[ClassifiedPage], document_id: str) -> None:
        pass

    @abstractmethod
    def save_extractions(self, extractions: List[ExtractedPage], document_id: str) -> None:
        pass

    @abstractmethod
    def save_citations(self, citations: List[Citation]) -> None:
        pass

    @abstractmethod
    def save_proposed_classifications(self, proposals: List[ProposedClassification], document_id: str) -> None:
        pass

    @abstractmethod
    def save_audit_record(self, record: LLMCallAuditRecord) -> None:
        pass


class VectorRecord:
    """Standardized vector payload with complete lineage back to PDF and page."""

    def __init__(
        self,
        vector_id: str,
        embedding: List[float],
        text_chunk: str,
        document_id: str,
        page_number: int,
        classification: str,
        section: Optional[str] = None,
        citation_ref: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.vector_id = vector_id
        self.embedding = embedding
        self.text_chunk = text_chunk
        self.document_id = document_id
        self.page_number = page_number
        self.classification = classification
        self.section = section or ""
        self.citation_ref = citation_ref or ""
        self.source_uri = source_uri or ""
        self.metadata = metadata or {}


class VectorStoreRepository(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def upsert_vectors(self, records: List[VectorRecord]) -> int:
        """Persists vector records with metadata and lineage; returns count upserted."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        """Performs nearest-neighbor search with optional metadata filtering."""
        pass

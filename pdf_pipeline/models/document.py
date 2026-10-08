"""Document intake, page batching, and lifecycle models."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle processing status of a document."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"


class DocumentMetadata(BaseModel):
    """Metadata captured during initial PDF intake and preprocessing."""

    document_id: str = Field(description="Unique immutable identifier for this document")
    processing_run_id: str = Field(description="Unique identifier for this specific pipeline run")
    source_filename: str = Field(description="Original filename of the ingested file")
    source_uri: str = Field(description="GCS URI (gs://...) or local filesystem path")
    page_count: int = Field(ge=1, description="Total number of physical pages in the PDF")
    file_size_bytes: int = Field(ge=0, description="File size in bytes")
    mime_type: str = Field(default="application/pdf", description="MIME type of the source file")
    sha256_checksum: str = Field(description="Cryptographic SHA256 checksum of the file bytes")
    ingestion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of intake registration",
    )
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)


class BatchPage(BaseModel):
    """Represents a single page within a slice/batch."""

    page_number: int = Field(ge=1, description="1-indexed PDF page number")
    is_primary: bool = Field(
        default=True,
        description="True if this page is owned by this batch; False if it is an overlap boundary page",
    )


class PageBatch(BaseModel):
    """A contiguous or overlapping slice of PDF pages submitted to Gemini."""

    batch_id: str = Field(description="Unique identifier for the batch (e.g. batch_001)")
    document_id: str = Field(description="Parent document ID")
    processing_run_id: str = Field(description="Pipeline execution run ID")
    page_start: int = Field(ge=1, description="First page number in batch (inclusive)")
    page_end: int = Field(ge=1, description="Last page number in batch (inclusive)")
    pages: List[BatchPage] = Field(description="List of pages included in this batch with primary/overlap flags")
    is_split_child: bool = Field(
        default=False,
        description="Whether this batch was created through fallback splitting of an oversized parent batch",
    )
    parent_batch_id: Optional[str] = Field(
        default=None,
        description="Parent batch ID if created through recursive reduction",
    )
    estimated_input_tokens: Optional[int] = Field(
        default=None,
        description="Estimated token count calculated before Gemini submission",
    )

    @property
    def primary_page_numbers(self) -> List[int]:
        """Returns the list of authoritative primary page numbers for this batch."""
        return [p.page_number for p in self.pages if p.is_primary]

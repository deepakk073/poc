"""PDF Intake and Preprocessing Service.

Registers PDF files from local storage or Google Cloud Storage (GCS), calculates cryptographic
checksums, measures page counts, and mints document_id and processing_run_id.
"""

import hashlib
import io
import logging
import os
import uuid
from typing import Any, Optional, Tuple, Union

from pdf_pipeline.core.exceptions import PipelineError
from pdf_pipeline.models.document import DocumentMetadata, DocumentStatus

logger = logging.getLogger("pdf_pipeline.intake")


class PDFIntakeService:
    """Handles intake, metadata extraction, and registration of PDFs."""

    def __init__(self, gcs_client: Optional[Any] = None):
        self.gcs_client = gcs_client

    def ingest_pdf(
        self,
        source_path_or_uri: str,
        custom_document_id: Optional[str] = None,
        custom_run_id: Optional[str] = None,
    ) -> Tuple[DocumentMetadata, bytes]:
        """Ingests a PDF, computes metadata, and returns metadata along with file content bytes."""
        is_gcs = source_path_or_uri.startswith("gs://")
        source_filename = os.path.basename(source_path_or_uri)

        if is_gcs:
            file_bytes = self._read_from_gcs(source_path_or_uri)
        else:
            if not os.path.exists(source_path_or_uri):
                raise PipelineError(f"Local PDF file not found at path: {source_path_or_uri}")
            with open(source_path_or_uri, "rb") as f:
                file_bytes = f.read()

        file_size = len(file_bytes)
        if file_size == 0:
            raise PipelineError(f"PDF source {source_path_or_uri} is empty (0 bytes)")

        # Compute cryptographic SHA256 checksum
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        # Deterministic document_id ensures idempotence, or use custom if provided
        document_id = custom_document_id or f"doc_{sha256_hash[:16]}"
        processing_run_id = custom_run_id or f"run_{uuid.uuid4().hex[:12]}"

        # Extract page count
        page_count = self._get_page_count(file_bytes)

        metadata = DocumentMetadata(
            document_id=document_id,
            processing_run_id=processing_run_id,
            source_filename=source_filename,
            source_uri=source_path_or_uri,
            page_count=page_count,
            file_size_bytes=file_size,
            mime_type="application/pdf",
            sha256_checksum=sha256_hash,
            status=DocumentStatus.PROCESSING,
        )

        logger.info(
            f"Successfully ingested {source_filename}: "
            f"pages={page_count}, size={file_size} bytes, doc_id={document_id}, run_id={processing_run_id}"
        )
        return metadata, file_bytes

    def _get_page_count(self, file_bytes: bytes) -> int:
        """Determines the number of physical pages in the PDF."""
        # Method 1: Try pypdf
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(file_bytes))
            return len(reader.pages)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"pypdf reader failed: {e}. Trying fallback methods.")

        # Method 2: Try PyMuPDF (fitz)
        try:
            import fitz

            doc = fitz.open(stream=file_bytes, filetype="pdf")
            count = len(doc)
            doc.close()
            return count
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"fitz reader failed: {e}. Trying fallback methods.")

        # Method 3: Lightweight byte stream regex scanning for /Type /Page
        import re

        matches = re.findall(rb"/Type\s*/Page\b", file_bytes)
        count = len(matches)
        if count > 0:
            return count

        # Minimum baseline
        logger.warning("Could not reliably inspect PDF structure; defaulting page count to 1.")
        return 1

    def _read_from_gcs(self, gcs_uri: str) -> bytes:
        """Reads file bytes from a Google Cloud Storage URI."""
        from google.cloud import storage

        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        client = self.gcs_client or storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.download_as_bytes()

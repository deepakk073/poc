"""BigQuery Database Repository.

Persists document metadata, page classifications, extracted structured data,
proposed taxonomies, citations, and LLM call token/cost audit records.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.models.audit import LLMCallAuditRecord
from pdf_pipeline.models.citation import Citation
from pdf_pipeline.models.classification import ClassifiedPage, ProposedClassification
from pdf_pipeline.models.document import DocumentMetadata
from pdf_pipeline.models.extraction import ExtractedPage
from pdf_pipeline.repositories.base import DatabaseRepository

logger = logging.getLogger("pdf_pipeline.bigquery")


class BigQueryRepository(DatabaseRepository):
    """Production BigQuery repository with in-memory fallback for local execution."""

    def __init__(
        self,
        pipeline_settings: Optional[PipelineSettings] = None,
        bq_client: Optional[Any] = None,
    ):
        self.settings = pipeline_settings or settings
        self._client = bq_client

        # Local in-memory storage buffers
        self.documents_table: List[Dict[str, Any]] = []
        self.classifications_table: List[Dict[str, Any]] = []
        self.extractions_table: List[Dict[str, Any]] = []
        self.citations_table: List[Dict[str, Any]] = []
        self.proposed_classifications_table: List[Dict[str, Any]] = []
        self.audit_table: List[Dict[str, Any]] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import bigquery

                self._client = bigquery.Client(project=self.settings.project_id)
            except Exception as e:
                logger.warning(f"Could not connect to BigQuery: {e}. Running in local in-memory mode.")
                self._client = None
        return self._client

    def _insert_rows_json(self, table_id: str, rows: List[Dict[str, Any]]) -> None:
        """Helper to insert rows to BigQuery if client is available."""
        if not rows:
            return
        if self.client is not None:
            try:
                full_table_id = f"{self.settings.project_id}.{self.settings.bq_dataset_id}.{table_id}"
                errors = self.client.insert_rows_json(full_table_id, rows)
                if errors:
                    logger.error(f"BigQuery insertion errors into {table_id}: {errors}")
            except Exception as e:
                logger.warning(f"Failed live BigQuery write to {table_id}: {e}. Retaining in local buffer.")

    def save_document_metadata(self, metadata: DocumentMetadata) -> None:
        row = {
            "document_id": metadata.document_id,
            "processing_run_id": metadata.processing_run_id,
            "source_filename": metadata.source_filename,
            "source_uri": metadata.source_uri,
            "page_count": metadata.page_count,
            "file_size_bytes": metadata.file_size_bytes,
            "mime_type": metadata.mime_type,
            "sha256_checksum": metadata.sha256_checksum,
            "ingestion_timestamp": metadata.ingestion_timestamp.isoformat(),
            "status": metadata.status.value,
        }
        self.documents_table.append(row)
        self._insert_rows_json(self.settings.bq_documents_table, [row])
        logger.info(f"Persisted document metadata for {metadata.document_id}")

    def save_classifications(self, classifications: List[ClassifiedPage], document_id: str) -> None:
        rows = []
        for p in classifications:
            row = {
                "document_id": document_id,
                "page_number": p.page_number,
                "classification": p.classification,
                "classification_description": p.classification_description or "",
                "classification_source": p.classification_source,
                "confidence_score": float(p.confidence_score),
                "content_types": p.content_types,
                "page_metadata": p.page_metadata.model_dump(),
                "proposed_extraction_schema": json.dumps(p.proposed_extraction_schema or {}),
            }
            rows.append(row)
            self.classifications_table.append(row)
        self._insert_rows_json(self.settings.bq_classifications_table, rows)
        logger.info(f"Persisted {len(rows)} page classifications for {document_id}")

    def save_extractions(self, extractions: List[ExtractedPage], document_id: str) -> None:
        rows = []
        for p in extractions:
            row = {
                "document_id": document_id,
                "page_number": p.page_number,
                "classification": p.classification,
                "extracted_data": json.dumps(p.extracted_data),
                "text_content": p.text_content or "",
                "handwritten_content": p.handwritten_content or "",
                "is_scanned": p.is_scanned,
                "tables": [t.model_dump() for t in p.tables],
                "images": [img.model_dump() for img in p.images],
                "warnings": p.warnings,
                "confidence_score": float(p.confidence_score),
            }
            rows.append(row)
            self.extractions_table.append(row)
        self._insert_rows_json(self.settings.bq_extractions_table, rows)
        logger.info(f"Persisted {len(rows)} extracted pages for {document_id}")

    def save_citations(self, citations: List[Citation]) -> None:
        rows = []
        for c in citations:
            row = {
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
                "batch_id": c.batch_id,
                "classification": c.classification,
                "field_path": c.field_path or "",
                "source_text": c.source_text or "",
                "bounding_box": c.bounding_box.model_dump() if c.bounding_box else None,
                "extraction_method": c.extraction_method,
                "confidence_score": float(c.confidence_score),
                "is_inferred": c.is_inferred,
            }
            rows.append(row)
            self.citations_table.append(row)
        self._insert_rows_json(self.settings.bq_citations_table, rows)
        logger.info(f"Persisted {len(rows)} citations")

    def save_proposed_classifications(self, proposals: List[ProposedClassification], document_id: str) -> None:
        rows = []
        for prop in proposals:
            row = {
                "classification_name": prop.classification_name,
                "document_id": document_id,
                "classification_description": prop.classification_description,
                "distinguishing_features": prop.distinguishing_features,
                "extraction_schema": json.dumps(prop.extraction_schema),
                "confidence_score": float(prop.confidence_score),
                "classification_status": prop.classification_status,
            }
            rows.append(row)
            self.proposed_classifications_table.append(row)
        self._insert_rows_json(self.settings.bq_proposed_classifications_table, rows)
        logger.info(f"Persisted {len(rows)} proposed classifications for review")

    def save_audit_record(self, record: LLMCallAuditRecord) -> None:
        row = record.to_bigquery_row()
        self.audit_table.append(row)
        self._insert_rows_json(self.settings.bq_audit_table, [row])

    # Method matching AuditSink protocol
    def insert_audit_record(self, record: LLMCallAuditRecord) -> None:
        self.save_audit_record(record)

"""Embedding Generation and Vector Persistence Service.

Generates embeddings for extracted page texts and table summaries using Gemini embedding models,
attaches full source lineage, and writes vectors to the configured vector database backend.
"""

import hashlib
import logging
from typing import Any, List, Optional

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.core.audit_logger import TokenAuditLogger
from pdf_pipeline.core.gemini_client import GeminiClientWrapper
from pdf_pipeline.models.audit import CallStatus, OperationType
from pdf_pipeline.models.citation import LineageRecord
from pdf_pipeline.models.document import DocumentMetadata
from pdf_pipeline.models.extraction import ExtractedPage
from pdf_pipeline.repositories.base import VectorRecord, VectorStoreRepository

logger = logging.getLogger("pdf_pipeline.embedding")


class EmbeddingService:
    """Chunks extracted content, creates Gemini embeddings, and persists lineage-grounded vectors."""

    def __init__(
        self,
        gemini_client: GeminiClientWrapper,
        vector_store: VectorStoreRepository,
        audit_logger: TokenAuditLogger,
        pipeline_settings: Optional[PipelineSettings] = None,
    ):
        self.gemini_client = gemini_client
        self.vector_store = vector_store
        self.audit_logger = audit_logger
        self.settings = pipeline_settings or settings

    def generate_and_store_embeddings(
        self,
        extracted_pages: List[ExtractedPage],
        doc_metadata: DocumentMetadata,
    ) -> List[VectorRecord]:
        """Generates embeddings for all extracted pages and stores them with lineage."""
        records_to_upsert: List[VectorRecord] = []

        for page in extracted_pages:
            # Build representative text chunks for this page
            chunks = self._chunk_page(page)

            for chunk_idx, (chunk_text, section_name, citation_ref) in enumerate(chunks):
                vector_id = f"{doc_metadata.document_id}_p{page.page_number}_c{chunk_idx}"

                # Generate embedding vector
                embedding_vector = self._embed_text(
                    text=chunk_text,
                    document_id=doc_metadata.document_id,
                    run_id=doc_metadata.processing_run_id,
                    page_number=page.page_number,
                )

                record = VectorRecord(
                    vector_id=vector_id,
                    embedding=embedding_vector,
                    text_chunk=chunk_text,
                    document_id=doc_metadata.document_id,
                    page_number=page.page_number,
                    classification=page.classification,
                    section=section_name,
                    citation_ref=citation_ref,
                    source_uri=doc_metadata.source_uri,
                    metadata={
                        "source_filename": doc_metadata.source_filename,
                        "confidence_score": page.confidence_score,
                        "is_scanned": page.is_scanned,
                    },
                )
                records_to_upsert.append(record)

        # Upsert batch to vector store
        if records_to_upsert:
            count = self.vector_store.upsert_vectors(records_to_upsert)
            logger.info(
                f"Generated and persisted {count} lineage-grounded embeddings "
                f"for document {doc_metadata.document_id}"
            )

        return records_to_upsert

    def _chunk_page(self, page: ExtractedPage) -> List[tuple[str, str, str]]:
        """Breaks a page into embeddable text segments (text content, tables)."""
        chunks = []

        # 1. Main body text
        if page.text_content and page.text_content.strip():
            cite_str = f"Page {page.page_number} ({page.classification})"
            chunks.append((page.text_content.strip(), "body", cite_str))

        # 2. Handwritten annotations
        if page.handwritten_content and page.handwritten_content.strip():
            chunks.append((f"Handwritten annotations: {page.handwritten_content.strip()}", "handwritten", f"Page {page.page_number}"))

        # 3. Tables
        for tbl in page.tables:
            tbl_text = tbl.markdown_representation or f"Table {tbl.table_id}: headers {tbl.headers}"
            chunks.append((tbl_text, f"table_{tbl.table_id}", f"Page {page.page_number} Table {tbl.table_id}"))

        # If page had no text (e.g. pure drawing/image page), embed classification & extracted entity summary
        if not chunks and page.extracted_data:
            summary = f"Page {page.page_number} classified as {page.classification}: {page.extracted_data}"
            chunks.append((summary, "structured_summary", f"Page {page.page_number}"))

        return chunks

    def _embed_text(self, text: str, document_id: str, run_id: str, page_number: int) -> List[float]:
        """Calls Gemini embedding API or returns deterministic synthetic vector in offline mode."""
        model_name = self.settings.model_embedding
        if self.gemini_client.client is not None:
            try:
                response = self.gemini_client.client.models.embed_content(
                    model=model_name,
                    contents=text,
                )
                if hasattr(response, "embedding") and hasattr(response.embedding, "values"):
                    return response.embedding.values
                if hasattr(response, "embeddings") and response.embeddings:
                    return response.embeddings[0].values
            except Exception as e:
                logger.warning(f"Live embedding generation failed: {e}. Falling back to deterministic embedding.")

        # Offline deterministic normalized embedding (768 dimensions)
        dim = self.settings.vector_dimension
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        import random

        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec]

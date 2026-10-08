"""Stage 2: Classification-Aware Information Extraction Service.

Extracts typed and handwritten text, scanned indicators, tables, embedded imagery,
and classification-specific entities using the dynamic schema discovered in Stage 1.
Preserves reading order, binds field-level citations, and enforces strict null-semantics for missing values.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.core.gemini_client import GeminiClientWrapper
from pdf_pipeline.models.audit import OperationType
from pdf_pipeline.models.classification import BatchClassificationResponse
from pdf_pipeline.models.document import DocumentMetadata, PageBatch
from pdf_pipeline.models.extraction import BatchExtractionResponse, ExtractedPage
from pdf_pipeline.services.citation_manager import CitationManager
from pdf_pipeline.services.schema_manager import DynamicSchemaManager

logger = logging.getLogger("pdf_pipeline.extraction")


class ExtractionService:
    """Executes Stage 2 classification-grounded information extraction."""

    def __init__(
        self,
        gemini_client: GeminiClientWrapper,
        schema_manager: DynamicSchemaManager,
        citation_manager: CitationManager,
        pipeline_settings: Optional[PipelineSettings] = None,
    ):
        self.gemini_client = gemini_client
        self.schema_manager = schema_manager
        self.citation_manager = citation_manager
        self.settings = pipeline_settings or settings

    def extract_batch(
        self,
        batch: PageBatch,
        doc_metadata: DocumentMetadata,
        classification_result: BatchClassificationResponse,
        batch_pdf_contents: Any,
        retry_attempt: int = 0,
    ) -> BatchExtractionResponse:
        """Invokes Gemini to extract structured entities grounded in Stage 1 classification context."""
        # Compile page-by-page schema instructions
        page_instructions = []
        for p in classification_result.pages:
            schema = self.schema_manager.get_schema_for_classification(p.classification)
            page_instructions.append(
                f"Page {p.page_number} (Classification: '{p.classification}', Modalities: {p.content_types}):\n"
                f"Target Schema:\n{json.dumps(schema, indent=2)}"
            )

        schemas_str = "\n\n".join(page_instructions)

        prompt = (
            "You are a Senior Multimodal Document Information Extraction specialist.\n"
            f"You are extracting structured information from document '{doc_metadata.source_filename}' "
            f"(Batch {batch.batch_id}, pages {batch.page_start} through {batch.page_end}).\n\n"
            "STAGE 1 CLASSIFICATION CONTEXT & SCHEMAS:\n"
            f"{schemas_str}\n\n"
            "CRITICAL EXTRACTION DIRECTIVES:\n"
            "1. Grounded Entity Extraction:\n"
            "   - Extract structured entities for each page according to its classification schema into 'extracted_data'.\n"
            "   - Distinguish explicitly observed values from inferred values. Set 'is_inferred: true' on citations for inferred facts.\n"
            "   - If a requested field is absent from the page, return null. NEVER fabricate, hallucinate, or assume missing values.\n"
            "2. Text Modalities:\n"
            "   - 'text_content': Machine-readable typed text, preserving logical reading order (multi-column aware).\n"
            "   - 'handwritten_content': Isolate handwritten remarks, dates, technician signatures, or marginalia.\n"
            "   - 'is_scanned': Set to true if the page shows visual scan artifacts or physical paper distortion.\n"
            "3. Tabular & Visual Data:\n"
            "   - 'tables': Extract complete table structures (headers, rows) and provide a clean markdown representation.\n"
            "   - 'images': Identify and describe drawings, schematics, photos, maps, diagrams, logos, and signatures.\n"
            "4. Citations & Provenance:\n"
            "   - For every extracted field, table, or visual element, return a citation with the page_number, "
            "     source_text evidence, extraction_method='gemini_multimodal', confidence_score (0.0 to 1.0), "
            "     and bounding_box [ymin, xmin, ymax, xmax] normalized to 0-1000 if available.\n\n"
            "Return the output strictly adhering to the requested JSON structure."
        )

        response_dict = self.gemini_client.call_gemini(
            model_name=self.settings.model_extraction,
            prompt=prompt,
            contents=batch_pdf_contents,
            operation_type=OperationType.STAGE2_EXTRACTION,
            document_id=batch.document_id,
            processing_run_id=batch.processing_run_id,
            batch_id=batch.batch_id,
            page_start=batch.page_start,
            page_end=batch.page_end,
            response_schema=BatchExtractionResponse,
            retry_attempt=retry_attempt,
        )

        # Re-validate with Pydantic model
        extraction_result = BatchExtractionResponse.model_validate(response_dict)
        extraction_result.document_id = batch.document_id
        extraction_result.processing_run_id = batch.processing_run_id
        extraction_result.batch_id = batch.batch_id

        # Post-process pages: ensure proper lineage & citation consistency
        for page in extraction_result.pages:
            # Bind classification if missing
            matching_class = next(
                (p.classification for p in classification_result.pages if p.page_number == page.page_number),
                "unclassified",
            )
            page.classification = page.classification or matching_class

            # Validate citations against page number
            self.citation_manager.validate_page_citations(page)

            # Ensure every citation references document_id, filename, and batch_id
            for cite in page.citations:
                cite.document_id = batch.document_id
                cite.source_filename = doc_metadata.source_filename
                cite.batch_id = batch.batch_id
                if not cite.classification:
                    cite.classification = page.classification

        logger.info(
            f"Successfully extracted batch {batch.batch_id} "
            f"({len(extraction_result.pages)} pages extracted)"
        )
        return extraction_result

"""Stage 1: Classification, Metadata, and Citation Service.

Performs document-level and page-level classification, metadata extraction,
content-type profiling, and dynamic extraction schema proposals using Gemini on Vertex AI.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.core.gemini_client import GeminiClientWrapper
from pdf_pipeline.models.audit import OperationType
from pdf_pipeline.models.citation import Citation
from pdf_pipeline.models.classification import (
    BatchClassificationResponse,
    ClassifiedPage,
    PageMetadata,
    ProposedClassification,
)
from pdf_pipeline.models.document import DocumentMetadata, PageBatch
from pdf_pipeline.services.schema_manager import DynamicSchemaManager

logger = logging.getLogger("pdf_pipeline.classification")


class ClassificationService:
    """Executes Stage 1 classification and metadata enrichment across page batches."""

    def __init__(
        self,
        gemini_client: GeminiClientWrapper,
        schema_manager: DynamicSchemaManager,
        pipeline_settings: Optional[PipelineSettings] = None,
    ):
        self.gemini_client = gemini_client
        self.schema_manager = schema_manager
        self.settings = pipeline_settings or settings

    def classify_batch(
        self,
        batch: PageBatch,
        doc_metadata: DocumentMetadata,
        batch_pdf_contents: Any,
        retry_attempt: int = 0,
    ) -> BatchClassificationResponse:
        """Invokes Gemini to classify a batch of PDF pages according to semi-dynamic taxonomy."""
        known_taxonomies_str = ", ".join(self.schema_manager.known_classifications)
        pages_in_batch = [p.page_number for p in batch.pages]

        prompt = (
            "You are an expert Document Classification and Metadata Intelligence system.\n"
            f"You are analyzing pages {batch.page_start} through {batch.page_end} of document '{doc_metadata.source_filename}'.\n"
            f"The specific page numbers in this batch are: {pages_in_batch}.\n\n"
            "INSTRUCTIONS:\n"
            "1. Document & Page Classification:\n"
            f"   - Match each page against this approved list of known classifications: [{known_taxonomies_str}].\n"
            "   - If a page fits one of the known types, set 'classification' to that name and 'classification_source' to 'known'.\n"
            "   - If NO known type fits, propose a new category in normalized snake_case (e.g. 'flight_itinerary', 'soil_test_report').\n"
            "     Set 'classification_source' to 'proposed'.\n"
            "   - For every proposed category, add a corresponding entry to 'proposed_taxonomies' with 'classification_name', "
            "     'classification_description', 'distinguishing_features', and a proposed 'extraction_schema' JSON schema.\n"
            "2. Content Types Profiling:\n"
            "   - For each page, identify all content modalities present from: "
            "     ['typed_text', 'handwritten', 'scanned', 'table', 'form', 'drawing', 'diagram', 'map', 'photograph', 'mixed'].\n"
            "3. Page & Document Metadata:\n"
            "   - Extract headers, footers, section names, page orientation, and language.\n"
            "4. Source Citation:\n"
            "   - Provide an authoritative citation anchoring the classification decision to specific evidence found on that page.\n"
            "5. Confidence Scoring:\n"
            "   - Provide a realistic confidence_score between 0.0 and 1.0 for each page.\n\n"
            "Return the output strictly adhering to the requested JSON structure."
        )

        response_dict = self.gemini_client.call_gemini(
            model_name=self.settings.model_classification,
            prompt=prompt,
            contents=batch_pdf_contents,
            operation_type=OperationType.STAGE1_CLASSIFICATION,
            document_id=batch.document_id,
            processing_run_id=batch.processing_run_id,
            batch_id=batch.batch_id,
            page_start=batch.page_start,
            page_end=batch.page_end,
            response_schema=BatchClassificationResponse,
            retry_attempt=retry_attempt,
        )

        # Re-validate with Pydantic model
        classification_result = BatchClassificationResponse.model_validate(response_dict)

        # Register any dynamically proposed taxonomies
        for proposed in classification_result.proposed_taxonomies:
            self.schema_manager.register_proposed_classification(proposed)

        # Standardize page citations and classifications
        for page in classification_result.pages:
            norm_name = self.schema_manager.normalize_name(page.classification)
            page.classification = norm_name

            # Ensure citation is properly populated
            if page.citation is None:
                page.citation = Citation(
                    document_id=batch.document_id,
                    source_filename=doc_metadata.source_filename,
                    page_number=page.page_number,
                    batch_id=batch.batch_id,
                    classification=page.classification,
                    source_text=f"Classified as {page.classification} with confidence {page.confidence_score}",
                    extraction_method="gemini_multimodal",
                    confidence_score=page.confidence_score,
                )
            else:
                page.citation.document_id = batch.document_id
                page.citation.source_filename = doc_metadata.source_filename
                page.citation.batch_id = batch.batch_id

        logger.info(
            f"Successfully classified batch {batch.batch_id} "
            f"({len(classification_result.pages)} pages classified)"
        )
        return classification_result

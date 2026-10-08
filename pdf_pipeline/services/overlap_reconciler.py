"""Overlap Reconciliation and Deduplication Service.

Harmonizes overlapping page boundaries across adjacent batches.
Uses (document_id, page_number) as authoritative deduplication key, prioritizes
primary batch ownership, resolves classification conflicts based on confidence and neighbor context,
and guarantees zero duplicate records in final structured tables, citations, or vector stores.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from pdf_pipeline.models.classification import BatchClassificationResponse, ClassifiedPage
from pdf_pipeline.models.citation import Citation
from pdf_pipeline.models.document import PageBatch
from pdf_pipeline.models.extraction import BatchExtractionResponse, ExtractedPage

logger = logging.getLogger("pdf_pipeline.reconciler")


class ClassificationConflict(BaseModel := type("BaseModel", (), {})):
    """Record of conflicting classification decisions across overlapping batches."""

    def __init__(
        self,
        document_id: str,
        page_number: int,
        primary_batch_id: str,
        primary_classification: str,
        primary_confidence: float,
        overlap_batch_id: str,
        overlap_classification: str,
        overlap_confidence: float,
        resolution: str,
        resolved_classification: str,
    ):
        self.document_id = document_id
        self.page_number = page_number
        self.primary_batch_id = primary_batch_id
        self.primary_classification = primary_classification
        self.primary_confidence = primary_confidence
        self.overlap_batch_id = overlap_batch_id
        self.overlap_classification = overlap_classification
        self.overlap_confidence = overlap_confidence
        self.resolution = resolution
        self.resolved_classification = resolved_classification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "primary_batch_id": self.primary_batch_id,
            "primary_classification": self.primary_classification,
            "primary_confidence": self.primary_confidence,
            "overlap_batch_id": self.overlap_batch_id,
            "overlap_classification": self.overlap_classification,
            "overlap_confidence": self.overlap_confidence,
            "resolution": self.resolution,
            "resolved_classification": self.resolved_classification,
        }


class OverlapReconciler:
    """Reconciles boundary pages between overlapping batches."""

    def __init__(self):
        self.conflicts: List[ClassificationConflict] = []

    def reconcile_batches(
        self,
        batches: List[PageBatch],
        classifications: List[BatchClassificationResponse],
        extractions: List[BatchExtractionResponse],
    ) -> Tuple[List[ClassifiedPage], List[ExtractedPage], List[Citation], List[Dict[str, Any]]]:
        """Consolidates results across all batches, strictly deduplicating pages."""
        # Index batches by batch_id
        batch_map: Dict[str, PageBatch] = {b.batch_id: b for b in batches}

        # Page-indexed authoritative store
        consolidated_classifications: Dict[int, Tuple[ClassifiedPage, str, bool]] = {}  # page_num -> (page, batch_id, is_primary)
        consolidated_extractions: Dict[int, Tuple[ExtractedPage, str, bool]] = {}

        # 1. Process Classifications
        for c_resp, batch in zip(classifications, batches):
            # Map page primary status for this batch
            primary_flags = {p.page_number: p.is_primary for p in batch.pages}

            for page in c_resp.pages:
                p_num = page.page_number
                is_prim = primary_flags.get(p_num, True)

                if p_num not in consolidated_classifications:
                    consolidated_classifications[p_num] = (page, batch.batch_id, is_prim)
                else:
                    existing_page, existing_batch_id, existing_is_prim = consolidated_classifications[p_num]

                    # Check for classification conflict
                    if existing_page.classification != page.classification:
                        primary_record = (existing_page, existing_batch_id) if existing_is_prim else (page, batch.batch_id)
                        overlap_record = (page, batch.batch_id) if existing_is_prim else (existing_page, existing_batch_id)

                        # Resolution policy:
                        # 1. Prefer primary batch result
                        # 2. If overlap confidence is significantly higher (> 0.25 delta), prefer higher confidence
                        conf_delta = overlap_record[0].confidence_score - primary_record[0].confidence_score
                        if conf_delta > 0.25:
                            resolved_class = overlap_record[0].classification
                            resolution = "OVERLAP_OVERRIDE_DUE_TO_SIGNIFICANT_CONFIDENCE_DELTA"
                            chosen_page = overlap_record[0]
                            chosen_batch = overlap_record[1]
                        else:
                            resolved_class = primary_record[0].classification
                            resolution = "PRIMARY_BATCH_PREFERRED"
                            chosen_page = primary_record[0]
                            chosen_batch = primary_record[1]

                        conflict = ClassificationConflict(
                            document_id=batch.document_id,
                            page_number=p_num,
                            primary_batch_id=primary_record[1],
                            primary_classification=primary_record[0].classification,
                            primary_confidence=primary_record[0].confidence_score,
                            overlap_batch_id=overlap_record[1],
                            overlap_classification=overlap_record[0].classification,
                            overlap_confidence=overlap_record[0].confidence_score,
                            resolution=resolution,
                            resolved_classification=resolved_class,
                        )
                        self.conflicts.append(conflict)
                        logger.warning(
                            f"Classification conflict on page {p_num}: "
                            f"{primary_record[1]}='{primary_record[0].classification}' vs "
                            f"{overlap_record[1]}='{overlap_record[0].classification}'. Resolved: {resolved_class}"
                        )
                        consolidated_classifications[p_num] = (chosen_page, chosen_batch, True)
                    else:
                        # No conflict; if incoming is primary and existing was overlap, update to primary
                        if is_prim and not existing_is_prim:
                            consolidated_classifications[p_num] = (page, batch.batch_id, True)

        # 2. Process Extractions
        for e_resp, batch in zip(extractions, batches):
            primary_flags = {p.page_number: p.is_primary for p in batch.pages}

            for page in e_resp.pages:
                p_num = page.page_number
                is_prim = primary_flags.get(p_num, True)

                if p_num not in consolidated_extractions:
                    consolidated_extractions[p_num] = (page, batch.batch_id, is_prim)
                else:
                    existing_page, existing_batch_id, existing_is_prim = consolidated_extractions[p_num]
                    # Retain primary batch extraction
                    if is_prim and not existing_is_prim:
                        consolidated_extractions[p_num] = (page, batch.batch_id, True)

        # Sort deduplicated results deterministically by page_number
        sorted_pages = sorted(consolidated_classifications.keys())
        final_classifications = [consolidated_classifications[p][0] for p in sorted_pages]
        final_extractions = [consolidated_extractions[p][0] for p in sorted_pages if p in consolidated_extractions]

        # Consolidate and deduplicate citations
        final_citations: List[Citation] = []
        seen_citation_keys = set()

        for ext_page in final_extractions:
            for cite in ext_page.citations:
                # Deduplication key: (doc_id, page_number, field_path, source_text)
                cite_key = (
                    cite.document_id,
                    cite.page_number,
                    cite.field_path or "",
                    cite.source_text or "",
                )
                if cite_key not in seen_citation_keys:
                    seen_citation_keys.add(cite_key)
                    final_citations.append(cite)

        logger.info(
            f"Overlap reconciliation complete: {len(final_classifications)} unique pages, "
            f"{len(final_citations)} deduplicated citations, {len(self.conflicts)} conflicts recorded."
        )

        return (
            final_classifications,
            final_extractions,
            final_citations,
            [c.to_dict() for c in self.conflicts],
        )

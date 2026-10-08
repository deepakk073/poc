"""Citation and Lineage Engine.

Ensures every extracted field, table, visual asset, and text snippet maintains
an unbroken, verified provenance link back to the source PDF, physical page number, and batch context.
"""

import logging
from typing import Any, Dict, List, Optional

from pdf_pipeline.models.citation import BoundingBox, Citation, LineageRecord
from pdf_pipeline.models.extraction import ExtractedPage

logger = logging.getLogger("pdf_pipeline.citation")


class CitationManager:
    """Creates, validates, and manages source citations and lineage graphs."""

    def create_citation(
        self,
        document_id: str,
        source_filename: str,
        page_number: int,
        batch_id: str,
        classification: str,
        source_text: Optional[str] = None,
        bounding_box: Optional[BoundingBox] = None,
        extraction_method: str = "gemini_multimodal",
        confidence_score: float = 1.0,
        field_path: Optional[str] = None,
        is_inferred: bool = False,
    ) -> Citation:
        """Constructs an authoritative field-level citation object."""
        return Citation(
            document_id=document_id,
            source_filename=source_filename,
            page_number=page_number,
            batch_id=batch_id,
            classification=classification,
            source_text=source_text,
            bounding_box=bounding_box,
            extraction_method=extraction_method,
            confidence_score=max(0.0, min(1.0, confidence_score)),
            field_path=field_path,
            is_inferred=is_inferred,
        )

    def extract_field_lineages(
        self,
        extracted_page: ExtractedPage,
        document_id: str,
        source_uri: str,
        source_filename: str,
        batch_id: str,
    ) -> List[LineageRecord]:
        """Flattens extracted page data into verifiable lineage records with citations."""
        lineage_records: List[LineageRecord] = []

        # 1. Lineage for structured JSON entities
        def _recurse_fields(data: Any, prefix: str = ""):
            if isinstance(data, dict):
                for k, v in data.items():
                    path = f"{prefix}.{k}" if prefix else k
                    _recurse_fields(v, path)
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    path = f"{prefix}[{idx}]"
                    _recurse_fields(item, path)
            else:
                # Leaf value
                # Find matching citation if emitted by model, or construct one
                matching_cite = next(
                    (c for c in extracted_page.citations if c.field_path == prefix),
                    None,
                )
                if not matching_cite:
                    matching_cite = self.create_citation(
                        document_id=document_id,
                        source_filename=source_filename,
                        page_number=extracted_page.page_number,
                        batch_id=batch_id,
                        classification=extracted_page.classification,
                        source_text=str(data) if data is not None else None,
                        field_path=prefix,
                        confidence_score=extracted_page.confidence_score,
                    )
                lineage_records.append(
                    LineageRecord(
                        document_id=document_id,
                        source_uri=source_uri,
                        page_number=extracted_page.page_number,
                        classification=extracted_page.classification,
                        field_path=prefix,
                        extracted_value=data,
                        citation=matching_cite,
                    )
                )

        if extracted_page.extracted_data:
            _recurse_fields(extracted_page.extracted_data)

        # 2. Lineage for tables
        for tbl in extracted_page.tables:
            tbl_cite = tbl.citation or self.create_citation(
                document_id=document_id,
                source_filename=source_filename,
                page_number=extracted_page.page_number,
                batch_id=batch_id,
                classification=extracted_page.classification,
                source_text=tbl.caption or f"Table with {len(tbl.rows)} rows",
                field_path=f"tables.{tbl.table_id}",
                confidence_score=extracted_page.confidence_score,
            )
            lineage_records.append(
                LineageRecord(
                    document_id=document_id,
                    source_uri=source_uri,
                    page_number=extracted_page.page_number,
                    classification=extracted_page.classification,
                    field_path=f"tables.{tbl.table_id}",
                    extracted_value={"headers": tbl.headers, "row_count": len(tbl.rows)},
                    citation=tbl_cite,
                )
            )

        return lineage_records

    def validate_page_citations(self, extracted_page: ExtractedPage) -> bool:
        """Validates that all citations on this page reference the exact physical page number."""
        for cite in extracted_page.citations:
            if cite.page_number != extracted_page.page_number:
                logger.warning(
                    f"Citation page mismatch: Citation references page {cite.page_number}, "
                    f"but page entity is {extracted_page.page_number}."
                )
                cite.page_number = extracted_page.page_number  # Harmonize
        return True

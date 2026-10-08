"""Stage 1: Classification, Metadata, and Schema Proposal data models."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import re
from pdf_pipeline.models.citation import Citation


class ProposedClassification(BaseModel):
    """Dynamic schema and taxonomy proposed by Gemini when no known category matches."""

    classification_name: str = Field(description="Normalized snake_case identifier (e.g., equipment_log_sheet)")
    classification_description: str = Field(description="Detailed explanation of what this page represents")
    distinguishing_features: List[str] = Field(
        default_factory=list,
        description="Visual or structural hallmarks identifying this classification",
    )
    extraction_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="Proposed JSON schema for extracting fields from this document category",
    )
    confidence_score: float = Field(ge=0.0, le=1.0, description="Model confidence in proposing this new category")
    classification_status: Literal["proposed", "approved", "rejected"] = Field(
        default="proposed",
        description="Status of classification; defaults to proposed and is never automatically promoted",
    )

    @field_validator("classification_name")
    @classmethod
    def validate_snake_case(cls, v: str) -> str:
        """Enforces normalized snake_case name formatting."""
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", v).strip("_").lower()
        if not cleaned:
            raise ValueError(f"Invalid classification name: '{v}'. Must resolve to valid snake_case.")
        return cleaned


class PageMetadata(BaseModel):
    """Page-specific structural and physical attributes."""

    has_header: bool = Field(default=False)
    has_footer: bool = Field(default=False)
    page_boundary_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    section_name: Optional[str] = Field(default=None, description="Logical document section (e.g. Terms & Conditions)")
    is_section_start: bool = Field(default=False)
    language: str = Field(default="en")
    page_orientation: str = Field(default="portrait", description="portrait, landscape, or rotated")


class ClassifiedPage(BaseModel):
    """Stage 1 classification output for an individual page."""

    page_number: int = Field(ge=1, description="1-indexed physical PDF page number")
    classification: str = Field(description="Canonical or proposed snake_case classification name")
    classification_description: Optional[str] = Field(
        default=None, description="Explanation of why this classification was assigned"
    )
    classification_source: Literal["known", "proposed"] = Field(
        description="Whether this was selected from known taxonomy or proposed newly"
    )
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score for this page's classification")
    content_types: List[str] = Field(
        default_factory=list,
        description="Detected modalities: typed_text, handwritten, scanned, table, form, drawing, image, map, mixed",
    )
    page_metadata: PageMetadata = Field(default_factory=PageMetadata)
    proposed_extraction_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema proposal for fields that should be extracted from this page type",
    )
    citation: Optional[Citation] = Field(
        default=None,
        description="Authoritative citation anchoring the classification decision to page evidence",
    )


class BatchClassificationResponse(BaseModel):
    """Structured output expected from Gemini for Stage 1 classification."""

    document_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document-level inferred metadata (e.g., document_title, document_type, overall_language)",
    )
    batch_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="High-level summary of the pages in this batch",
    )
    pages: List[ClassifiedPage] = Field(description="Page-level classification results")
    proposed_taxonomies: List[ProposedClassification] = Field(
        default_factory=list,
        description="List of newly discovered page types proposed in this batch",
    )

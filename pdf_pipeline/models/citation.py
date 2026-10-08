"""Citation and lineage data models.

Ensures every extracted field and observation is groundable with authoritative PDF page attribution.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates on the PDF page (0 to 1000 or 0.0 to 1.0)."""

    ymin: float = Field(description="Top coordinate")
    xmin: float = Field(description="Left coordinate")
    ymax: float = Field(description="Bottom coordinate")
    xmax: float = Field(description="Right coordinate")
    coordinate_system: str = Field(
        default="normalized_1000",
        description="Coordinate space ('normalized_1000' [0-1000] or 'normalized_1' [0.0-1.0])",
    )


class Citation(BaseModel):
    """Authoritative source citation for an extracted datum."""

    document_id: str = Field(description="Unique ID of the parent document")
    source_filename: str = Field(description="Filename of the source PDF")
    page_number: int = Field(ge=1, description="1-indexed authoritative PDF page number")
    batch_id: str = Field(description="Batch identifier during which extraction occurred")
    classification: str = Field(description="Identified page classification")
    source_text: Optional[str] = Field(default=None, description="Exact textual evidence extracted from the page")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Coordinates of the bounding box if detected")
    extraction_method: str = Field(
        default="gemini_multimodal",
        description="Extraction method (e.g., 'gemini_multimodal', 'ocr_text', 'table_grid')",
    )
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in this extraction (0.0 to 1.0)")
    field_path: Optional[str] = Field(
        default=None,
        description="Target JSON path or field name in the structured extraction schema",
    )
    is_inferred: bool = Field(
        default=False,
        description="Whether this value was inferred rather than explicitly observed on the page",
    )


class LineageRecord(BaseModel):
    """End-to-end lineage tracing record from document ingestion to vector embedding."""

    document_id: str
    source_uri: str
    page_number: int
    classification: str
    field_path: str
    extracted_value: Any
    citation: Citation
    vector_id: Optional[str] = None

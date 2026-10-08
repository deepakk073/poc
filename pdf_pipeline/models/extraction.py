"""Stage 2: Classification-Aware Information Extraction data models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from pdf_pipeline.models.citation import Citation


class TableCell(BaseModel):
    """Single cell within an extracted table."""

    row_index: int
    column_index: int
    text: str
    is_header: bool = False
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class TableData(BaseModel):
    """Extracted table representation."""

    table_id: str = Field(description="Unique identifier for the table on this page")
    caption: Optional[str] = Field(default=None, description="Table title or caption")
    headers: List[str] = Field(default_factory=list, description="Column header labels")
    rows: List[List[Optional[str]]] = Field(
        default_factory=list,
        description="2D array of rows containing extracted string values or null",
    )
    markdown_representation: Optional[str] = Field(
        default=None, description="Markdown rendering of the table grid"
    )
    citation: Optional[Citation] = Field(default=None, description="Citation anchoring table location")


class VisualAssetMetadata(BaseModel):
    """Metadata for embedded images, diagrams, drawings, maps, or photographs."""

    asset_id: str
    asset_type: str = Field(
        description="diagram | technical_drawing | map | photograph | signature | stamp | logo | barcode"
    )
    description: str = Field(description="Visual description of the asset content")
    coordinates_hint: Optional[str] = Field(default=None, description="Location description or coordinates")
    text_within_asset: Optional[str] = Field(default=None, description="Any text detected inside the drawing/image")


class ExtractedField(BaseModel):
    """Fine-grained field-level extracted value with confidence and lineage."""

    field_name: str
    value: Any = Field(default=None, description="Extracted value or null if unavailable")
    is_inferred: bool = Field(
        default=False,
        description="Distinguishes explicitly observed facts from inferred/calculated values",
    )
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    citation: Optional[Citation] = None


class ExtractedPage(BaseModel):
    """Stage 2 extraction output for a single page."""

    page_number: int = Field(ge=1, description="1-indexed physical PDF page number")
    classification: str = Field(description="Page classification used to guide extraction")
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Classification-specific structured JSON entities adhering to the extraction schema",
    )
    text_content: Optional[str] = Field(
        default=None,
        description="Full typed machine-readable text in reading order",
    )
    handwritten_content: Optional[str] = Field(
        default=None,
        description="Isolated handwritten annotations, signatures, or notes",
    )
    is_scanned: bool = Field(default=False, description="Flag indicating if the page is scanned content")
    tables: List[TableData] = Field(default_factory=list, description="All tables discovered on this page")
    images: List[VisualAssetMetadata] = Field(
        default_factory=list,
        description="Embedded images, schematics, drawings, maps, or photos",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations for every extracted entity and field value",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings regarding illegible handwriting, partial tears, or low contrast",
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall aggregate confidence score for page extraction"
    )


class BatchExtractionResponse(BaseModel):
    """Structured output expected from Gemini for Stage 2 extraction."""

    document_id: str = Field(description="Parent document identifier")
    processing_run_id: str = Field(description="Pipeline run identifier")
    batch_id: str = Field(description="Batch identifier")
    pages: List[ExtractedPage] = Field(description="Page-level structured extraction records")

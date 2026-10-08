-- BigQuery DDL: Page Classifications Table
-- Stores Stage 1 classification, confidence scores, detected modalities, and proposed schemas.

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.page_classifications`
(
    document_id STRING NOT NULL OPTIONS(description="Foreign key to document_metadata.document_id"),
    page_number INT64 NOT NULL OPTIONS(description="1-indexed physical PDF page number"),
    classification STRING NOT NULL OPTIONS(description="Canonical or proposed snake_case classification"),
    classification_description STRING OPTIONS(description="Detailed description or justification"),
    classification_source STRING NOT NULL OPTIONS(description="'known' or 'proposed'"),
    confidence_score FLOAT64 NOT NULL OPTIONS(description="Model confidence between 0.0 and 1.0"),
    content_types ARRAY<STRING> OPTIONS(description="List of detected content modalities (typed_text, table, handwritten, etc.)"),
    page_metadata JSON OPTIONS(description="Structural metadata (header, footer, orientation, section)"),
    proposed_extraction_schema JSON OPTIONS(description="Dynamic schema proposed for this classification")
)
CLUSTER BY document_id, classification;

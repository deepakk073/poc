-- BigQuery DDL: Citations & Provenance Table
-- Anchors every extracted datum to physical page evidence with bounding box coordinates.

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.citations`
(
    document_id STRING NOT NULL OPTIONS(description="Foreign key to document_metadata.document_id"),
    source_filename STRING NOT NULL OPTIONS(description="Filename of the source document"),
    page_number INT64 NOT NULL OPTIONS(description="1-indexed authoritative physical PDF page number"),
    batch_id STRING NOT NULL OPTIONS(description="Processing batch ID"),
    classification STRING NOT NULL OPTIONS(description="Classification at time of extraction"),
    field_path STRING OPTIONS(description="JSONPath or attribute name of the extracted datum"),
    source_text STRING OPTIONS(description="Verbatim text evidence observed on the page"),
    bounding_box JSON OPTIONS(description="Normalized coordinates (ymin, xmin, ymax, xmax)"),
    extraction_method STRING NOT NULL DEFAULT 'gemini_multimodal',
    confidence_score FLOAT64 NOT NULL OPTIONS(description="Citation confidence (0.0 to 1.0)"),
    is_inferred BOOL NOT NULL DEFAULT FALSE OPTIONS(description="Flags whether value was inferred rather than explicitly printed")
)
CLUSTER BY document_id, page_number;

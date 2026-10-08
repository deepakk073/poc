-- BigQuery DDL: Extracted Pages Table
-- Stores Stage 2 classification-grounded extracted JSON, reading-order text, tables, and visual assets.

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.extracted_pages`
(
    document_id STRING NOT NULL OPTIONS(description="Foreign key to document_metadata.document_id"),
    page_number INT64 NOT NULL OPTIONS(description="1-indexed physical PDF page number"),
    classification STRING NOT NULL OPTIONS(description="Classification driving the schema"),
    extracted_data JSON NOT NULL OPTIONS(description="Structured entities adhering to the extraction schema"),
    text_content STRING OPTIONS(description="Machine-readable text in natural reading order"),
    handwritten_content STRING OPTIONS(description="Isolated handwritten marginalia, signatures, notes"),
    is_scanned BOOL NOT NULL DEFAULT FALSE OPTIONS(description="Indicator if the page is a visual paper scan"),
    tables JSON OPTIONS(description="Extracted table structures including 2D grid arrays and markdown"),
    images JSON OPTIONS(description="Discovered visual elements (drawings, schematics, logos, maps)"),
    warnings ARRAY<STRING> OPTIONS(description="Quality notices regarding low contrast, torn paper, etc."),
    confidence_score FLOAT64 NOT NULL OPTIONS(description="Aggregate extraction confidence score")
)
CLUSTER BY document_id, classification;

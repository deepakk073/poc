-- BigQuery DDL: Proposed Classifications Table
-- Stores dynamic taxonomies proposed by Gemini for human evaluation without auto-promotion.

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.proposed_classifications`
(
    classification_name STRING NOT NULL OPTIONS(description="Normalized snake_case identifier"),
    document_id STRING NOT NULL OPTIONS(description="Document ID where this taxonomy was first discovered"),
    classification_description STRING NOT NULL OPTIONS(description="Model rationale for creating this category"),
    distinguishing_features ARRAY<STRING> OPTIONS(description="Visual hallmarks and structural patterns"),
    extraction_schema JSON NOT NULL OPTIONS(description="Proposed JSON schema for extraction"),
    confidence_score FLOAT64 NOT NULL OPTIONS(description="Model confidence score in proposing this category"),
    classification_status STRING NOT NULL DEFAULT 'proposed' OPTIONS(description="'proposed', 'approved', or 'rejected'"),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY classification_status, classification_name;

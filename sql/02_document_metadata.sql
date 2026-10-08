-- BigQuery DDL: Document Metadata Table
-- Tracks intake registration, cryptographic integrity, page counts, and overall status.

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.document_metadata`
(
    document_id STRING NOT NULL OPTIONS(description="Deterministic cryptographic identifier for the document"),
    processing_run_id STRING NOT NULL OPTIONS(description="Unique pipeline execution run ID"),
    source_filename STRING NOT NULL OPTIONS(description="Original uploaded filename"),
    source_uri STRING NOT NULL OPTIONS(description="GCS or local filesystem source URI"),
    page_count INT64 NOT NULL OPTIONS(description="Total number of physical PDF pages"),
    file_size_bytes INT64 NOT NULL OPTIONS(description="File size in bytes"),
    mime_type STRING NOT NULL DEFAULT 'application/pdf',
    sha256_checksum STRING NOT NULL OPTIONS(description="Cryptographic SHA256 checksum"),
    ingestion_timestamp TIMESTAMP NOT NULL OPTIONS(description="Intake timestamp in UTC"),
    status STRING NOT NULL OPTIONS(description="PENDING, PROCESSING, COMPLETED, FAILED, PARTIALLY_COMPLETED")
)
PARTITION BY DATE(ingestion_timestamp)
CLUSTER BY status, document_id
OPTIONS(
    description="Intake registration and processing lifecycle state for ingested documents"
);

-- BigQuery DDL: LLM Call Audit Table
-- Logs token consumption, internal thinking token counts, decoupled cost estimations, and latency.
-- Partitioned by DATE(request_timestamp) and clustered by document_id, operation_type, status.

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.llm_call_audit`
(
    audit_id STRING NOT NULL OPTIONS(description="Unique UUID for this specific Gemini call attempt"),
    correlation_id STRING NOT NULL OPTIONS(description="Distributed trace or workflow correlation identifier"),
    document_id STRING NOT NULL OPTIONS(description="Parent document identifier"),
    processing_run_id STRING NOT NULL OPTIONS(description="Pipeline execution run ID"),
    batch_id STRING NOT NULL OPTIONS(description="Batch identifier (e.g. batch_001)"),
    operation_type STRING NOT NULL OPTIONS(description="Operation category: STAGE1_CLASSIFICATION, STAGE2_EXTRACTION, JSON_REPAIR, TOKEN_COUNT"),
    model_name STRING NOT NULL OPTIONS(description="Gemini model requested (e.g. gemini-3.8-flash)"),
    model_version STRING OPTIONS(description="Specific model version returned by Vertex AI"),
    page_start INT64 OPTIONS(description="First page in batch (1-indexed)"),
    page_end INT64 OPTIONS(description="Last page in batch (1-indexed)"),
    prompt_token_count INT64 NOT NULL OPTIONS(description="Authoritative prompt/input tokens consumed"),
    output_token_count INT64 NOT NULL OPTIONS(description="Authoritative candidates/output tokens consumed"),
    thinking_token_count INT64 NOT NULL OPTIONS(description="Tokens consumed by model during internal thinking"),
    cached_token_count INT64 NOT NULL OPTIONS(description="Tokens read from context cache"),
    total_token_count INT64 NOT NULL OPTIONS(description="Authoritative total tokens consumed"),
    estimated_input_cost NUMERIC NOT NULL OPTIONS(description="Estimated USD cost for prompt tokens"),
    estimated_output_cost NUMERIC NOT NULL OPTIONS(description="Estimated USD cost for output tokens"),
    estimated_thinking_cost NUMERIC NOT NULL OPTIONS(description="Estimated USD cost for thinking tokens"),
    estimated_total_cost NUMERIC NOT NULL OPTIONS(description="Total estimated USD cost"),
    currency STRING NOT NULL DEFAULT 'USD',
    request_timestamp TIMESTAMP NOT NULL OPTIONS(description="Timestamp when request was dispatched"),
    response_timestamp TIMESTAMP NOT NULL OPTIONS(description="Timestamp when response was received"),
    latency_ms INT64 NOT NULL OPTIONS(description="End-to-end invocation latency in milliseconds"),
    retry_count INT64 NOT NULL DEFAULT 0 OPTIONS(description="Attempt sequence number (0=initial, 1+=retries)"),
    status STRING NOT NULL OPTIONS(description="SUCCESS, RETRYING, FAILED, or FALLBACK_TRIGGERED"),
    error_code STRING OPTIONS(description="Standardized error code (e.g. RESOURCE_EXHAUSTED, TIMEOUT)"),
    error_message STRING OPTIONS(description="Sanitized diagnostic error message"),
    created_by STRING NOT NULL OPTIONS(description="Service account or worker identity"),
    created_timestamp TIMESTAMP NOT NULL OPTIONS(description="Audit row creation timestamp"),
    updated_timestamp TIMESTAMP NOT NULL OPTIONS(description="Audit row update timestamp")
)
PARTITION BY DATE(request_timestamp)
CLUSTER BY document_id, operation_type, status
OPTIONS(
    description="Authoritative audit trail for Vertex AI Gemini API token usage and financial cost estimation"
);

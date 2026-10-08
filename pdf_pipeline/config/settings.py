"""Configuration model for Vertex AI Gemini PDF Classification & Extraction Pipeline.

All settings can be overridden via environment variables or a .env file.
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Production settings for the intelligent PDF pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="PDF_PIPELINE_",
    )

    # Google Cloud Platform Identity & Location
    project_id: str = Field(default="test-gcp-project", description="GCP Project ID")
    location: str = Field(default="us-central1", description="Vertex AI & GCP Resource Location")
    use_vertex_ai: bool = Field(default=True, description="Whether to route calls through Vertex AI SDK")
    api_key: Optional[str] = Field(default=None, description="Optional Google AI Studio API key for local fallback")

    # Storage & BigQuery Identifiers
    bq_dataset_id: str = Field(default="pdf_intelligence", description="BigQuery Dataset ID")
    bq_audit_table: str = Field(default="llm_call_audit", description="Audit table for Gemini token logging")
    bq_pricing_table: str = Field(default="llm_pricing_reference", description="Pricing reference table")
    bq_documents_table: str = Field(default="document_metadata", description="Table storing intake document metadata")
    bq_classifications_table: str = Field(default="page_classifications", description="Table storing page classifications")
    bq_extractions_table: str = Field(default="extracted_pages", description="Table storing extracted structured data")
    bq_citations_table: str = Field(default="citations", description="Table storing field-level citations")
    bq_proposed_classifications_table: str = Field(
        default="proposed_classifications", description="Table storing dynamic proposed classifications"
    )

    # Cloud Storage Buckets
    gcs_bucket_name: str = Field(default="pdf-intelligence-artifacts", description="Primary GCS bucket for artifacts")
    gcs_raw_prefix: str = Field(default="raw_pdfs", description="GCS prefix for original PDF storage")
    gcs_pages_prefix: str = Field(default="rendered_pages", description="GCS prefix for rendered page image artifacts")
    gcs_dead_letter_prefix: str = Field(default="dead_letter", description="GCS prefix for unrecoverable payloads")

    # Gemini Models
    model_classification: str = Field(default="gemini-3.8-flash", description="Model for Stage 1 classification")
    model_extraction: str = Field(default="gemini-3.8-flash", description="Model for Stage 2 extraction")
    model_repair: str = Field(default="gemini-3.8-flash", description="Model for JSON repair fallbacks")
    model_embedding: str = Field(default="gemini-embedding-001", description="Model for multimodal/text embeddings")

    # Batching Configuration
    batch_size: int = Field(default=10, ge=1, description="Initial page batch size")
    overlap_size: int = Field(default=1, ge=0, description="Page overlap between consecutive batches")
    min_batch_size: int = Field(default=1, ge=1, description="Minimum batch size allowed during recursive halving")

    # Token Safety & Budgets
    token_safety_threshold: int = Field(
        default=600_000,
        description="Threshold of estimated input tokens beyond which proactive batch splitting triggers",
    )
    max_output_tokens: int = Field(default=8192, description="Max candidate tokens requested from Gemini")
    prompt_reserve_tokens: int = Field(default=16384, description="Reserved tokens for prompt formatting")

    # Resilience & Exponential Backoff
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts on retryable errors")
    retry_delay_seconds: float = Field(default=2.0, ge=0.1, description="Initial backoff delay in seconds")
    retry_exponential_base: float = Field(default=2.0, ge=1.0, description="Exponential multiplier")
    request_timeout_seconds: float = Field(default=120.0, ge=1.0, description="Per-call client timeout")

    # Vector Storage Abstraction
    vector_store_type: str = Field(
        default="in_memory",
        description="Vector database backend: 'vertex_vector_search', 'bigquery_vector_search', or 'in_memory'",
    )
    vector_dimension: int = Field(default=768, description="Embedding vector dimensionality")
    vertex_index_endpoint_id: Optional[str] = Field(default=None, description="Vertex AI Vector Search Index Endpoint")
    vertex_deployed_index_id: Optional[str] = Field(default=None, description="Vertex AI Deployed Index ID")

    # Semi-Dynamic Taxonomy Configuration
    known_classifications: List[str] = Field(
        default_factory=lambda: [
            "invoice",
            "receipt",
            "purchase_order",
            "contract_agreement",
            "bank_statement",
            "work_order",
            "technical_schematic",
            "architectural_drawing",
            "tax_document",
            "medical_record",
            "shipping_manifest",
            "utility_bill",
            "government_id",
            "correspondence_letter",
            "table_of_contents",
            "legal_brief",
        ],
        description="Known canonical classifications that Gemini will prioritize",
    )

    # Auditing & User Identification
    created_by: str = Field(default="gemini-pdf-pipeline-worker", description="System actor logging audit records")


# Global singleton instance
settings = PipelineSettings()

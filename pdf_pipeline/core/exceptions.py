"""Custom exception hierarchy for the intelligent PDF pipeline."""


class PipelineError(Exception):
    """Base exception for all pipeline failures."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class GeminiAPIError(PipelineError):
    """Raised when Vertex AI Gemini API invocation fails."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message, details)
        self.error_code = error_code


class GeminiContextLimitError(GeminiAPIError):
    """Raised when prompt or batch exceeds Gemini model context or token threshold."""
    pass


class GeminiMaxTokensExceededError(GeminiAPIError):
    """Raised when output generation is truncated due to MAX_TOKENS limit."""
    pass


class GeminiSchemaValidationError(GeminiAPIError):
    """Raised when Gemini returns output that violates the strict JSON Pydantic schema."""

    def __init__(self, message: str, raw_response: str, validation_errors: list = None):
        super().__init__(message, error_code="SCHEMA_VALIDATION_FAILED")
        self.raw_response = raw_response
        self.validation_errors = validation_errors or []


class JSONRepairFailedError(GeminiAPIError):
    """Raised when JSON repair prompt fails to recover valid schema from raw output."""
    pass


class BatchSplittingError(PipelineError):
    """Raised when batch cannot be subdivided further (already at min_batch_size)."""
    pass


class StorageError(PipelineError):
    """Raised when BigQuery, GCS, or Vector Store storage operations fail."""
    pass


class UnrecoverablePageError(PipelineError):
    """Raised when a page cannot be processed even after recursive halving down to 1 page."""
    pass

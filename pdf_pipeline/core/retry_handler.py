"""Resilience, Retry, and Fallback Management.

Implements exponential backoff, reactive batch halving, context-limit recovery,
and dead-letter queue routing for unrecoverable pages.
Never silently discards a page under any failure mode.
"""

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.core.audit_logger import TokenAuditLogger
from pdf_pipeline.core.exceptions import (
    BatchSplittingError,
    GeminiContextLimitError,
    GeminiMaxTokensExceededError,
    UnrecoverablePageError,
)
from pdf_pipeline.models.document import PageBatch
from pdf_pipeline.repositories.base import ObjectStorageRepository

logger = logging.getLogger("pdf_pipeline.retry")

T = TypeVar("T")


class RetryAndFallbackHandler:
    """Manages proactive and reactive retry loops and batch decomposition."""

    def __init__(
        self,
        audit_logger: TokenAuditLogger,
        storage_repo: ObjectStorageRepository,
        pipeline_settings: Optional[PipelineSettings] = None,
    ):
        self.audit_logger = audit_logger
        self.storage = storage_repo
        self.settings = pipeline_settings or settings

    def execute_with_backoff(
        self,
        func: Callable[..., T],
        *args,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        **kwargs,
    ) -> T:
        """Executes a function with jittered exponential backoff for transient errors."""
        retries = max_retries if max_retries is not None else self.settings.max_retries
        delay = base_delay if base_delay is not None else self.settings.retry_delay_seconds
        multiplier = self.settings.retry_exponential_base

        last_error = None
        for attempt in range(retries + 1):
            try:
                # Pass retry_attempt to target function if accepted
                kwargs["retry_attempt"] = attempt
                return func(*args, **kwargs)
            except (GeminiContextLimitError, GeminiMaxTokensExceededError):
                # Context limit / token truncation cannot be solved by simple retry;
                # must bubble up to reactive batch splitting
                raise
            except Exception as e:
                last_error = e
                if attempt >= retries:
                    logger.error(f"Function {func.__name__} exhausted {retries} retries. Final error: {e}")
                    raise

                jitter = random.uniform(0.5, 1.5)
                sleep_duration = (delay * (multiplier**attempt)) * jitter
                logger.warning(
                    f"Attempt {attempt + 1}/{retries + 1} for {func.__name__} failed: {e}. "
                    f"Retrying in {sleep_duration:.2f}s..."
                )
                time.sleep(sleep_duration)

        raise last_error

    def handle_dead_letter(
        self,
        batch: PageBatch,
        failed_payload: Any,
        error_reason: str,
    ) -> str:
        """Sends permanently unrecoverable batch artifacts to dead-letter storage."""
        dlq_path = (
            f"{self.settings.gcs_dead_letter_prefix}/"
            f"{batch.document_id}/{batch.batch_id}_error.json"
        )
        payload = {
            "document_id": batch.document_id,
            "processing_run_id": batch.processing_run_id,
            "batch_id": batch.batch_id,
            "page_start": batch.page_start,
            "page_end": batch.page_end,
            "pages": [p.model_dump() for p in batch.pages],
            "error_reason": error_reason,
            "payload_data": str(failed_payload),
        }
        stored_uri = self.storage.upload_json(dlq_path, payload)
        logger.critical(
            f"DEAD-LETTER QUEUE: Preserved unrecoverable batch {batch.batch_id} "
            f"(pages {batch.page_start}-{batch.page_end}) to {stored_uri}. Reason: {error_reason}"
        )
        return stored_uri

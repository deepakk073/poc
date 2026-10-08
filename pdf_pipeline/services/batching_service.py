"""PDF Batching and Overlap Management Service.

Generates deterministic page batches with configurable overlap (e.g., 10-page window with 1-page overlap:
1-10, 10-19, 19-28). Unambiguously marks primary vs overlap pages so deduplication is strictly enforced.
Provides proactive token estimation and recursive batch-halving when capacity exceeds safety thresholds.
"""

from datetime import datetime, timezone
import logging
from typing import Any, List, Optional, Tuple

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.core.audit_logger import TokenAuditLogger
from pdf_pipeline.core.exceptions import BatchSplittingError
from pdf_pipeline.core.gemini_client import GeminiClientWrapper
from pdf_pipeline.models.audit import CallStatus, OperationType
from pdf_pipeline.models.document import BatchPage, PageBatch

logger = logging.getLogger("pdf_pipeline.batching")


class PDFBatchingService:
    """Calculates batch intervals, assigns primary/overlap tags, and manages recursive splitting."""

    def __init__(
        self,
        pipeline_settings: Optional[PipelineSettings] = None,
        audit_logger: Optional[TokenAuditLogger] = None,
    ):
        self.settings = pipeline_settings or settings
        self.audit_logger = audit_logger

    def create_batches(
        self,
        document_id: str,
        processing_run_id: str,
        total_pages: int,
        batch_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
    ) -> List[PageBatch]:
        """Generates the initial sequence of page batches with overlap."""
        bs = batch_size or self.settings.batch_size
        ov = overlap_size if overlap_size is not None else self.settings.overlap_size

        if total_pages <= 0:
            return []

        # Validate parameters
        if bs <= 0:
            bs = 10
        if ov < 0:
            ov = 0
        if ov >= bs:
            ov = bs - 1  # Overlap cannot be greater than or equal to batch size

        stride = bs - ov
        batches: List[PageBatch] = []
        batch_idx = 1
        curr_start = 1

        # Track pages that have already been claimed as primary
        claimed_primary_pages = set()

        while curr_start <= total_pages:
            curr_end = min(curr_start + bs - 1, total_pages)
            batch_id = f"batch_{batch_idx:03d}"

            batch_pages: List[BatchPage] = []
            for p_num in range(curr_start, curr_end + 1):
                # If page was already primary in a previous batch, it acts as overlap context here
                if p_num in claimed_primary_pages:
                    batch_pages.append(BatchPage(page_number=p_num, is_primary=False))
                else:
                    batch_pages.append(BatchPage(page_number=p_num, is_primary=True))
                    claimed_primary_pages.add(p_num)

            batch = PageBatch(
                batch_id=batch_id,
                document_id=document_id,
                processing_run_id=processing_run_id,
                page_start=curr_start,
                page_end=curr_end,
                pages=batch_pages,
                is_split_child=False,
                parent_batch_id=None,
            )
            batches.append(batch)

            if curr_end >= total_pages:
                break

            curr_start += stride
            batch_idx += 1

        logger.info(
            f"Created {len(batches)} batches for document {document_id} "
            f"({total_pages} pages, batch_size={bs}, overlap={ov})"
        )
        return batches

    def estimate_and_optimize_batch(
        self,
        batch: PageBatch,
        gemini_client: GeminiClientWrapper,
        batch_content_bytes: Any,
        model_name: str,
    ) -> List[PageBatch]:
        """Proactive Fallback: Estimates tokens before Gemini call.

        If estimated tokens exceed safety threshold, recursively reduces batch size
        down to min_batch_size and logs split decisions to the audit log.
        """
        estimated_tokens = gemini_client.estimate_tokens(model_name, batch_content_bytes)
        batch.estimated_input_tokens = estimated_tokens

        threshold = self.settings.token_safety_threshold
        page_count = len(batch.pages)

        if estimated_tokens <= threshold or page_count <= self.settings.min_batch_size:
            # Batch fits safely within threshold or cannot be split further
            return [batch]

        logger.warning(
            f"Batch {batch.batch_id} estimated tokens ({estimated_tokens}) exceeds threshold ({threshold}). "
            f"Proactively splitting batch of {page_count} pages."
        )

        # Audit the split decision
        if self.audit_logger:
            self.audit_logger.log_call(
                operation_type=OperationType.TOKEN_COUNT.value,
                document_id=batch.document_id,
                processing_run_id=batch.processing_run_id,
                batch_id=batch.batch_id,
                model_name=model_name,
                request_timestamp=datetime.now(timezone.utc),
                response_timestamp=datetime.now(timezone.utc),
                status=CallStatus.FALLBACK_TRIGGERED,
                page_start=batch.page_start,
                page_end=batch.page_end,
                prompt_token_count=estimated_tokens,
                error_code="PROACTIVE_SPLIT_THRESHOLD_EXCEEDED",
                error_message=f"Estimated tokens {estimated_tokens} exceeded threshold {threshold}",
            )

        # Split batch into two smaller child batches
        children = self.split_batch(batch)
        optimized_batches: List[PageBatch] = []

        for child in children:
            # Recursive check on child
            sub_results = self.estimate_and_optimize_batch(
                batch=child,
                gemini_client=gemini_client,
                batch_content_bytes=batch_content_bytes,
                model_name=model_name,
            )
            optimized_batches.extend(sub_results)

        return optimized_batches

    def split_batch(self, batch: PageBatch) -> List[PageBatch]:
        """Subdivides a batch into two halves while preserving page continuity and primary ownership."""
        total_pages = len(batch.pages)
        if total_pages <= 1:
            raise BatchSplittingError(f"Cannot split batch {batch.batch_id} containing only 1 page.")

        mid = total_pages // 2
        pages_left = batch.pages[:mid]
        pages_right = batch.pages[mid:]

        child_1 = PageBatch(
            batch_id=f"{batch.batch_id}_c1",
            document_id=batch.document_id,
            processing_run_id=batch.processing_run_id,
            page_start=pages_left[0].page_number,
            page_end=pages_left[-1].page_number,
            pages=pages_left,
            is_split_child=True,
            parent_batch_id=batch.batch_id,
        )

        child_2 = PageBatch(
            batch_id=f"{batch.batch_id}_c2",
            document_id=batch.document_id,
            processing_run_id=batch.processing_run_id,
            page_start=pages_right[0].page_number,
            page_end=pages_right[-1].page_number,
            pages=pages_right,
            is_split_child=True,
            parent_batch_id=batch.batch_id,
        )

        logger.info(
            f"Split batch {batch.batch_id} (pages {batch.page_start}-{batch.page_end}) "
            f"into {child_1.batch_id} (pages {child_1.page_start}-{child_1.page_end}) and "
            f"{child_2.batch_id} (pages {child_2.page_start}-{child_2.page_end})"
        )
        return [child_1, child_2]

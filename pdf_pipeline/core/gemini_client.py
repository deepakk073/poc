"""Gemini Client Wrapper for Vertex AI.

Provides the unified `call_gemini` interface, enforces structured output validation,
extracts post-execution usage metadata (prompt, output, thinking, cached tokens),
and guarantees every invocation (success, retried, failed) is audited.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.core.audit_logger import TokenAuditLogger
from pdf_pipeline.core.exceptions import (
    GeminiAPIError,
    GeminiContextLimitError,
    GeminiMaxTokensExceededError,
    GeminiSchemaValidationError,
    JSONRepairFailedError,
)
from pdf_pipeline.models.audit import CallStatus, OperationType

logger = logging.getLogger("pdf_pipeline.gemini")


class GeminiClientWrapper:
    """Unified Vertex AI Gemini invocation service."""

    def __init__(
        self,
        audit_logger: TokenAuditLogger,
        pipeline_settings: Optional[PipelineSettings] = None,
        raw_client: Optional[Any] = None,
    ):
        self.audit_logger = audit_logger
        self.settings = pipeline_settings or settings
        self._client = raw_client

    @property
    def client(self) -> Any:
        """Lazy initialization of the google-genai client."""
        if self._client is None:
            try:
                from google import genai

                if self.settings.use_vertex_ai:
                    self._client = genai.Client(
                        vertexai=True,
                        project=self.settings.project_id,
                        location=self.settings.location,
                    )
                elif self.settings.api_key:
                    self._client = genai.Client(api_key=self.settings.api_key)
                else:
                    self._client = genai.Client()
            except ImportError:
                logger.warning("google-genai SDK not installed. Running in mock/offline mode.")
                self._client = None
            except Exception as e:
                logger.warning(f"Could not initialize google-genai client: {e}. Running with mock capability.")
                self._client = None
        return self._client

    def estimate_tokens(self, model_name: str, contents: Any) -> int:
        """Estimates input tokens prior to invocation for proactive capacity planning."""
        if self.client is not None:
            try:
                response = self.client.models.count_tokens(
                    model=model_name,
                    contents=contents,
                )
                if hasattr(response, "total_tokens"):
                    return response.total_tokens
            except Exception as e:
                logger.warning(f"Live count_tokens failed: {e}. Falling back to heuristic estimation.")

        # Heuristic fallback: ~4 characters per token for text, ~258 tokens per standard page image
        if isinstance(contents, str):
            return max(1, len(contents) // 4)
        if isinstance(contents, (list, tuple)):
            total = 0
            for item in contents:
                if isinstance(item, str):
                    total += len(item) // 4
                elif isinstance(item, bytes):
                    total += 258  # approximate tokens per PDF page image
                else:
                    total += len(str(item)) // 4
            return max(1, total)
        return 1000

    def call_gemini(
        self,
        model_name: str,
        prompt: str,
        contents: Any,
        operation_type: Union[str, OperationType],
        document_id: str,
        processing_run_id: str,
        batch_id: str,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        response_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        retry_attempt: int = 0,
    ) -> Dict[str, Any]:
        """Unified Gemini invocation matching requirement #5.

        Logs every call to BigQuery audit table with authoritative usage metadata.
        Validates against response_schema, triggering repair on validation failures.
        """
        op_type_str = operation_type.value if isinstance(operation_type, OperationType) else str(operation_type)
        corr_id = correlation_id or f"corr_{uuid.uuid4().hex[:12]}"
        req_time = datetime.now(timezone.utc)

        # Merge input prompt and contents into standardized contents payload
        full_contents = []
        if prompt:
            full_contents.append(prompt)
        if contents:
            if isinstance(contents, list):
                full_contents.extend(contents)
            else:
                full_contents.append(contents)

        # Build generation configuration
        config = dict(generation_config or {})
        if response_schema is not None:
            # Handle Pydantic model vs raw dict schema
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                config["response_mime_type"] = "application/json"
                config["response_schema"] = response_schema
            elif isinstance(response_schema, dict):
                config["response_mime_type"] = "application/json"
                config["response_schema"] = response_schema

        if "max_output_tokens" not in config:
            config["max_output_tokens"] = self.settings.max_output_tokens

        try:
            if self.client is None:
                raise GeminiAPIError("google-genai client is not configured or authenticated")

            # Execute model call
            response = self.client.models.generate_content(
                model=model_name,
                contents=full_contents,
                config=config,
            )
            resp_time = datetime.now(timezone.utc)

            # Extract authoritative token metrics from usage_metadata
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
            cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + output_tokens + thinking_tokens)

            # Check for output truncation due to token limits
            finish_reason = None
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
                if finish_reason in ("MAX_TOKENS", "LENGTH"):
                    self.audit_logger.log_call(
                        operation_type=op_type_str,
                        document_id=document_id,
                        processing_run_id=processing_run_id,
                        batch_id=batch_id,
                        model_name=model_name,
                        request_timestamp=req_time,
                        response_timestamp=resp_time,
                        status=CallStatus.FALLBACK_TRIGGERED,
                        page_start=page_start,
                        page_end=page_end,
                        prompt_token_count=prompt_tokens,
                        output_token_count=output_tokens,
                        thinking_token_count=thinking_tokens,
                        cached_token_count=cached_tokens,
                        total_token_count=total_tokens,
                        retry_count=retry_attempt,
                        correlation_id=corr_id,
                        error_code="FINISH_REASON_MAX_TOKENS",
                        error_message="Response was cut off due to MAX_TOKENS candidate limit",
                    )
                    raise GeminiMaxTokensExceededError("Generation truncated due to max output tokens limit")

            # Parse and validate response text
            raw_text = response.text if hasattr(response, "text") else str(response)
            parsed_data = self._parse_and_validate(
                raw_text=raw_text,
                response_schema=response_schema,
                document_id=document_id,
                processing_run_id=processing_run_id,
                batch_id=batch_id,
                model_name=model_name,
                page_start=page_start,
                page_end=page_end,
                correlation_id=corr_id,
                retry_attempt=retry_attempt,
            )

            # Log authoritative successful call
            self.audit_logger.log_call(
                operation_type=op_type_str,
                document_id=document_id,
                processing_run_id=processing_run_id,
                batch_id=batch_id,
                model_name=model_name,
                model_version=getattr(response, "model_version", None),
                request_timestamp=req_time,
                response_timestamp=resp_time,
                status=CallStatus.SUCCESS,
                page_start=page_start,
                page_end=page_end,
                prompt_token_count=prompt_tokens,
                output_token_count=output_tokens,
                thinking_token_count=thinking_tokens,
                cached_token_count=cached_tokens,
                total_token_count=total_tokens,
                retry_count=retry_attempt,
                correlation_id=corr_id,
            )

            return parsed_data

        except (GeminiMaxTokensExceededError, GeminiSchemaValidationError, JSONRepairFailedError):
            # Already logged at fine-grained layer or will be handled by caller
            raise

        except Exception as e:
            resp_time = datetime.now(timezone.utc)
            error_msg = str(e)
            error_code = type(e).__name__

            status = CallStatus.FAILED
            is_context_limit = (
                "context length" in error_msg.lower()
                or "token limit" in error_msg.lower()
                or "resourceexhausted" in error_msg.lower()
            )

            if is_context_limit:
                status = CallStatus.FALLBACK_TRIGGERED
                error_code = "CONTEXT_LIMIT_EXCEEDED"

            self.audit_logger.log_call(
                operation_type=op_type_str,
                document_id=document_id,
                processing_run_id=processing_run_id,
                batch_id=batch_id,
                model_name=model_name,
                request_timestamp=req_time,
                response_timestamp=resp_time,
                status=status,
                page_start=page_start,
                page_end=page_end,
                prompt_token_count=0,
                output_token_count=0,
                thinking_token_count=0,
                cached_token_count=0,
                total_token_count=0,
                retry_count=retry_attempt,
                correlation_id=corr_id,
                error_code=error_code,
                error_message=error_msg,
            )

            if is_context_limit:
                raise GeminiContextLimitError(error_msg, error_code=error_code)
            raise GeminiAPIError(error_msg, error_code=error_code)

    def _parse_and_validate(
        self,
        raw_text: str,
        response_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]],
        document_id: str,
        processing_run_id: str,
        batch_id: str,
        model_name: str,
        page_start: Optional[int],
        page_end: Optional[int],
        correlation_id: str,
        retry_attempt: int,
    ) -> Dict[str, Any]:
        """Parses JSON output, validates against schema, and attempts repair if invalid."""
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            parsed = json.loads(cleaned_text)
            if response_schema is not None and isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                # Validate with Pydantic
                validated_model = response_schema.model_validate(parsed)
                return validated_model.model_dump()
            return parsed
        except (json.JSONDecodeError, ValidationError) as parse_err:
            logger.warning(f"Gemini response failed JSON/schema validation: {parse_err}. Triggering repair prompt.")
            return self._repair_json(
                malformed_text=cleaned_text,
                response_schema=response_schema,
                document_id=document_id,
                processing_run_id=processing_run_id,
                batch_id=batch_id,
                page_start=page_start,
                page_end=page_end,
                correlation_id=correlation_id,
                retry_attempt=retry_attempt,
            )

    def _repair_json(
        self,
        malformed_text: str,
        response_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]],
        document_id: str,
        processing_run_id: str,
        batch_id: str,
        page_start: Optional[int],
        page_end: Optional[int],
        correlation_id: str,
        retry_attempt: int,
    ) -> Dict[str, Any]:
        """Invokes Gemini with a dedicated JSON repair prompt to fix syntax or schema errors."""
        repair_prompt = (
            "You are a strict JSON repair engine. The following text was intended to be valid JSON "
            "matching a strict schema, but it has syntax errors, truncation, or schema mismatches.\n"
            "Analyze and fix it so it returns strictly valid JSON without any markdown formatting or commentary.\n\n"
            f"MALFORMED CONTENT:\n{malformed_text}"
        )

        repair_schema = response_schema
        req_time = datetime.now(timezone.utc)
        repair_model = self.settings.model_repair

        try:
            if self.client is None:
                raise JSONRepairFailedError("No client available for JSON repair")

            repair_config = {
                "response_mime_type": "application/json",
                "max_output_tokens": self.settings.max_output_tokens,
            }
            if repair_schema and isinstance(repair_schema, type) and issubclass(repair_schema, BaseModel):
                repair_config["response_schema"] = repair_schema

            response = self.client.models.generate_content(
                model=repair_model,
                contents=repair_prompt,
                config=repair_config,
            )
            resp_time = datetime.now(timezone.utc)

            usage = getattr(response, "usage_metadata", None)
            p_tokens = getattr(usage, "prompt_token_count", 0) or 0
            o_tokens = getattr(usage, "candidates_token_count", 0) or 0
            th_tokens = getattr(usage, "thoughts_token_count", 0) or 0

            self.audit_logger.log_call(
                operation_type=OperationType.JSON_REPAIR.value,
                document_id=document_id,
                processing_run_id=processing_run_id,
                batch_id=batch_id,
                model_name=repair_model,
                request_timestamp=req_time,
                response_timestamp=resp_time,
                status=CallStatus.SUCCESS,
                page_start=page_start,
                page_end=page_end,
                prompt_token_count=p_tokens,
                output_token_count=o_tokens,
                thinking_token_count=th_tokens,
                retry_count=retry_attempt + 1,
                correlation_id=correlation_id,
            )

            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text.strip())

            if repair_schema and isinstance(repair_schema, type) and issubclass(repair_schema, BaseModel):
                return repair_schema.model_validate(parsed).model_dump()
            return parsed

        except Exception as repair_err:
            resp_time = datetime.now(timezone.utc)
            self.audit_logger.log_call(
                operation_type=OperationType.JSON_REPAIR.value,
                document_id=document_id,
                processing_run_id=processing_run_id,
                batch_id=batch_id,
                model_name=repair_model,
                request_timestamp=req_time,
                response_timestamp=resp_time,
                status=CallStatus.FAILED,
                page_start=page_start,
                page_end=page_end,
                error_code="JSON_REPAIR_FAILED",
                error_message=str(repair_err),
                retry_count=retry_attempt + 1,
                correlation_id=correlation_id,
            )
            raise JSONRepairFailedError(f"JSON repair failed: {repair_err}") from repair_err

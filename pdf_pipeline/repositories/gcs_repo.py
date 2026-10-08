"""Google Cloud Storage (GCS) and Local Object Storage Repository.

Persists raw PDFs, rendered page images, raw Gemini responses, failed payloads,
and normalized JSON artifacts with transparent local fallback.
"""

import json
import logging
import os
from typing import Any, Optional

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.repositories.base import ObjectStorageRepository

logger = logging.getLogger("pdf_pipeline.gcs")


class GCSStorageRepository(ObjectStorageRepository):
    """Production GCS repository with local filesystem emulation support."""

    def __init__(
        self,
        pipeline_settings: Optional[PipelineSettings] = None,
        storage_client: Optional[Any] = None,
        local_fallback_dir: str = "/tmp/pdf_pipeline_storage",
    ):
        self.settings = pipeline_settings or settings
        self.local_dir = local_fallback_dir
        self._client = storage_client
        self._bucket = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import storage

                self._client = storage.Client(project=self.settings.project_id)
            except Exception as e:
                logger.warning(f"Could not connect to Google Cloud Storage: {e}. Using local disk repository.")
                self._client = None
        return self._client

    def upload_bytes(self, destination_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Uploads raw bytes to GCS bucket or local directory."""
        if self.client is not None:
            try:
                bucket = self.client.bucket(self.settings.gcs_bucket_name)
                blob = bucket.blob(destination_path)
                blob.upload_from_string(data, content_type=content_type)
                uri = f"gs://{self.settings.gcs_bucket_name}/{destination_path}"
                logger.debug(f"Uploaded bytes to {uri}")
                return uri
            except Exception as e:
                logger.warning(f"GCS upload failed: {e}. Falling back to local storage.")

        # Local storage fallback
        full_path = os.path.join(self.local_dir, destination_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(data)
        uri = f"file://{os.path.abspath(full_path)}"
        logger.debug(f"Saved bytes locally to {uri}")
        return uri

    def upload_json(self, destination_path: str, data: Any) -> str:
        """Serializes and uploads JSON payload."""
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        return self.upload_bytes(destination_path, json_bytes, content_type="application/json")

    def read_bytes(self, source_path: str) -> bytes:
        """Reads file bytes from GCS URI or local path."""
        if source_path.startswith("gs://"):
            parts = source_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ""
            if self.client is not None:
                bucket = self.client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                return blob.download_as_bytes()

        clean_path = source_path.replace("file://", "")
        with open(clean_path, "rb") as f:
            return f.read()

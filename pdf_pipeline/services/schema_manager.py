"""Dynamic Schema and Taxonomy Manager.

Maintains canonical approved classifications, registers dynamic schemas proposed by Gemini,
enforces normalized snake_case naming conventions, and ensures proposed schemas are never
auto-promoted without human review.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pdf_pipeline.config.settings import settings, PipelineSettings
from pdf_pipeline.models.classification import ProposedClassification

logger = logging.getLogger("pdf_pipeline.schema")


class DynamicSchemaManager:
    """Manages document taxonomy, default extraction schemas, and proposed classifications."""

    # Default canonical extraction schemas for common known document types
    DEFAULT_SCHEMAS: Dict[str, Dict[str, Any]] = {
        "invoice": {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string", "description": "Invoice reference number"},
                "invoice_date": {"type": "string", "description": "Billing date"},
                "due_date": {"type": ["string", "null"], "description": "Payment due date"},
                "vendor_name": {"type": "string", "description": "Supplier or vendor legal entity"},
                "vendor_address": {"type": ["string", "null"], "description": "Supplier physical address"},
                "customer_name": {"type": "string", "description": "Billed recipient customer name"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": ["number", "null"]},
                            "unit_price": {"type": ["number", "null"]},
                            "total_amount": {"type": ["number", "null"]},
                        },
                    },
                },
                "subtotal": {"type": ["number", "null"]},
                "tax_amount": {"type": ["number", "null"]},
                "total_amount": {"type": "number", "description": "Grand total charged"},
                "currency": {"type": "string", "default": "USD"},
            },
            "required": ["invoice_number", "vendor_name", "total_amount"],
        },
        "receipt": {
            "type": "object",
            "properties": {
                "merchant_name": {"type": "string"},
                "transaction_datetime": {"type": "string"},
                "total_amount": {"type": "number"},
                "payment_method": {"type": ["string", "null"]},
                "tax_amount": {"type": ["number", "null"]},
            },
            "required": ["merchant_name", "total_amount"],
        },
        "purchase_order": {
            "type": "object",
            "properties": {
                "po_number": {"type": "string"},
                "order_date": {"type": "string"},
                "buyer_name": {"type": "string"},
                "vendor_name": {"type": "string"},
                "items": {"type": "array"},
                "total_amount": {"type": "number"},
            },
            "required": ["po_number", "vendor_name"],
        },
        "work_order": {
            "type": "object",
            "properties": {
                "work_order_number": {"type": "string"},
                "created_date": {"type": "string"},
                "assigned_technician": {"type": ["string", "null"]},
                "priority": {"type": ["string", "null"]},
                "task_description": {"type": "string"},
                "estimated_labor_hours": {"type": ["number", "null"]},
                "completion_status": {"type": "string"},
            },
            "required": ["work_order_number", "task_description"],
        },
        "contract_agreement": {
            "type": "object",
            "properties": {
                "agreement_title": {"type": "string"},
                "effective_date": {"type": "string"},
                "expiration_date": {"type": ["string", "null"]},
                "parties_involved": {"type": "array", "items": {"type": "string"}},
                "governing_law": {"type": ["string", "null"]},
                "key_obligations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["agreement_title", "parties_involved"],
        },
    }

    def __init__(self, pipeline_settings: Optional[PipelineSettings] = None):
        self.settings = pipeline_settings or settings
        self._known_classifications = set(self.settings.known_classifications)
        self._schemas: Dict[str, Dict[str, Any]] = dict(self.DEFAULT_SCHEMAS)
        self._proposed_registry: Dict[str, ProposedClassification] = {}

    @property
    def known_classifications(self) -> List[str]:
        """Returns sorted list of approved known classifications."""
        return sorted(list(self._known_classifications))

    def is_known(self, classification_name: str) -> bool:
        """Checks if a classification is in the approved canonical list."""
        return classification_name.lower().strip() in self._known_classifications

    def normalize_name(self, name: str) -> str:
        """Enforces normalized snake_case name formatting."""
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()
        return cleaned or "unclassified_document"

    def register_proposed_classification(
        self, proposed: ProposedClassification
    ) -> ProposedClassification:
        """Registers a newly discovered classification proposal without auto-promotion."""
        norm_name = self.normalize_name(proposed.classification_name)
        proposed.classification_name = norm_name
        proposed.classification_status = "proposed"  # Strict non-auto-promotion invariant

        # Validate schema structure
        if not isinstance(proposed.extraction_schema, dict) or "properties" not in proposed.extraction_schema:
            # Wrap in standard object format if not already well-formed
            proposed.extraction_schema = {
                "type": "object",
                "properties": proposed.extraction_schema if isinstance(proposed.extraction_schema, dict) else {},
                "additionalProperties": True,
            }

        self._proposed_registry[norm_name] = proposed
        # Save schema in memory for stage 2 extraction
        if norm_name not in self._schemas:
            self._schemas[norm_name] = proposed.extraction_schema

        logger.info(
            f"Registered proposed classification: '{norm_name}' "
            f"(confidence={proposed.confidence_score}, status={proposed.classification_status})"
        )
        return proposed

    def get_schema_for_classification(self, classification_name: str) -> Dict[str, Any]:
        """Fetches the extraction schema for an approved or proposed classification."""
        norm_name = self.normalize_name(classification_name)
        if norm_name in self._schemas:
            return self._schemas[norm_name]

        # Generic default schema if no custom schema exists
        return {
            "type": "object",
            "properties": {
                "document_title": {"type": ["string", "null"]},
                "primary_date": {"type": ["string", "null"]},
                "entities": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
                "key_value_pairs": {"type": "object", "additionalProperties": True},
            },
        }

    def get_all_proposed(self) -> List[ProposedClassification]:
        """Returns all proposed classifications registered during execution."""
        return list(self._proposed_registry.values())

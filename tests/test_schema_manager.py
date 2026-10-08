"""Unit tests for Dynamic Schema and Taxonomy Manager."""

import pytest
from pdf_pipeline.models.classification import ProposedClassification
from pdf_pipeline.services.schema_manager import DynamicSchemaManager


def test_schema_manager_known_classifications():
    manager = DynamicSchemaManager()
    assert manager.is_known("invoice") is True
    assert manager.is_known("work_order") is True
    assert manager.is_known("completely_random_unknown_type") is False


def test_schema_manager_snake_case_normalization():
    manager = DynamicSchemaManager()
    assert manager.normalize_name("Equipment Inspection Log Sheet") == "equipment_inspection_log_sheet"
    assert manager.normalize_name("  TAX---Form 1040-EZ  ") == "tax_form_1040_ez"
    assert manager.normalize_name("!!!Special#Case???") == "special_case"


def test_proposed_classification_non_auto_promotion():
    """Validates that proposed classifications are stored as 'proposed' and not auto-promoted."""
    manager = DynamicSchemaManager()
    proposal = ProposedClassification(
        classification_name="Soil Boring Log",
        classification_description="Geotechnical core sample drill data",
        distinguishing_features=["Depth columns", "Blow count graphs", "Soil stratification symbols"],
        extraction_schema={"properties": {"borehole_id": {"type": "string"}, "sample_depth_feet": {"type": "number"}}},
        confidence_score=0.92,
    )

    registered = manager.register_proposed_classification(proposal)

    # Invariants
    assert registered.classification_name == "soil_boring_log"
    assert registered.classification_status == "proposed"  # Strict invariant
    assert manager.is_known("soil_boring_log") is False  # Not promoted to canonical known list

    # Schema is available for stage 2 extraction
    schema = manager.get_schema_for_classification("soil_boring_log")
    assert "properties" in schema
    assert "borehole_id" in schema["properties"]

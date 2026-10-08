"""Unit tests for Citation and Lineage Engine."""

from pdf_pipeline.models.citation import BoundingBox, Citation
from pdf_pipeline.models.extraction import ExtractedPage, TableData
from pdf_pipeline.services.citation_manager import CitationManager


def test_create_citation_and_lineage():
    manager = CitationManager()
    bbox = BoundingBox(ymin=120.0, xmin=45.0, ymax=180.0, xmax=350.0)

    cite = manager.create_citation(
        document_id="doc_cite_1",
        source_filename="invoice_2026.pdf",
        page_number=3,
        batch_id="batch_001",
        classification="invoice",
        source_text="Invoice Number: INV-998877",
        bounding_box=bbox,
        confidence_score=0.98,
        field_path="invoice_number",
    )

    assert cite.page_number == 3
    assert cite.document_id == "doc_cite_1"
    assert cite.field_path == "invoice_number"
    assert cite.bounding_box.ymin == 120.0


def test_extract_field_lineages():
    manager = CitationManager()
    page = ExtractedPage(
        page_number=2,
        classification="purchase_order",
        extracted_data={
            "po_number": "PO-5544",
            "vendor": {"name": "Acme Industrial Supplies", "id": "V-12"},
            "total_amount": 12500.50,
        },
        tables=[
            TableData(
                table_id="tbl_1",
                caption="Purchased Line Items",
                headers=["Item", "Qty", "Cost"],
                rows=[["Steel Bolts", "1000", "$500"], ["Gaskets", "50", "$250"]],
            )
        ],
        confidence_score=0.95,
    )

    lineages = manager.extract_field_lineages(
        extracted_page=page,
        document_id="doc_po_1",
        source_uri="gs://bucket/sample.pdf",
        source_filename="sample.pdf",
        batch_id="batch_001",
    )

    # 4 field paths: po_number, vendor.name, vendor.id, total_amount, plus tables.tbl_1
    field_paths = [l.field_path for l in lineages]
    assert "po_number" in field_paths
    assert "vendor.name" in field_paths
    assert "vendor.id" in field_paths
    assert "total_amount" in field_paths
    assert "tables.tbl_1" in field_paths

    # Lineage must point to page 2 and document_id
    assert all(l.page_number == 2 for l in lineages)
    assert all(l.document_id == "doc_po_1" for l in lineages)

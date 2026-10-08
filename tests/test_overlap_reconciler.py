"""Unit tests for Overlap Reconciliation and Deduplication."""

from pdf_pipeline.models.citation import Citation
from pdf_pipeline.models.classification import BatchClassificationResponse, ClassifiedPage
from pdf_pipeline.models.document import BatchPage, PageBatch
from pdf_pipeline.models.extraction import BatchExtractionResponse, ExtractedPage
from pdf_pipeline.services.overlap_reconciler import OverlapReconciler


def test_overlap_deduplication_and_conflict_resolution():
    reconciler = OverlapReconciler()

    # Create two overlapping batches (shared page 10)
    batch_1 = PageBatch(
        batch_id="batch_001",
        document_id="doc_test",
        processing_run_id="run_1",
        page_start=1,
        page_end=10,
        pages=[BatchPage(page_number=i, is_primary=True) for i in range(1, 11)],
    )

    batch_2 = PageBatch(
        batch_id="batch_002",
        document_id="doc_test",
        processing_run_id="run_1",
        page_start=10,
        page_end=19,
        # Page 10 is overlap (is_primary=False), 11-19 are primary
        pages=[BatchPage(page_number=10, is_primary=False)] + [BatchPage(page_number=i, is_primary=True) for i in range(11, 20)],
    )

    # In Batch 1, page 10 is classified as "invoice" with confidence 0.95
    c_resp_1 = BatchClassificationResponse(
        pages=[
            ClassifiedPage(
                page_number=i,
                classification="invoice" if i == 10 else "correspondence_letter",
                classification_source="known",
                confidence_score=0.95 if i == 10 else 0.90,
            )
            for i in range(1, 11)
        ]
    )

    # In Batch 2, page 10 is seen again as overlap, but misclassified as "receipt" with confidence 0.70
    c_resp_2 = BatchClassificationResponse(
        pages=[
            ClassifiedPage(
                page_number=i,
                classification="receipt" if i == 10 else "work_order",
                classification_source="known",
                confidence_score=0.70 if i == 10 else 0.88,
            )
            for i in range(10, 20)
        ]
    )

    e_resp_1 = BatchExtractionResponse(
        document_id="doc_test",
        processing_run_id="run_1",
        batch_id="batch_001",
        pages=[
            ExtractedPage(page_number=i, classification="invoice" if i == 10 else "letter", extracted_data={"val": i})
            for i in range(1, 11)
        ],
    )

    e_resp_2 = BatchExtractionResponse(
        document_id="doc_test",
        processing_run_id="run_1",
        batch_id="batch_002",
        pages=[
            ExtractedPage(page_number=i, classification="receipt" if i == 10 else "work_order", extracted_data={"val": i * 10})
            for i in range(10, 20)
        ],
    )

    final_classes, final_exts, final_cites, conflicts = reconciler.reconcile_batches(
        batches=[batch_1, batch_2],
        classifications=[c_resp_1, c_resp_2],
        extractions=[e_resp_1, e_resp_2],
    )

    # Deduplication Invariant: exactly 19 unique pages (1..19)
    assert len(final_classes) == 19
    assert len(final_exts) == 19
    assert [p.page_number for p in final_classes] == list(range(1, 20))
    assert [p.page_number for p in final_exts] == list(range(1, 20))

    # Page 10 must resolve to "invoice" (from primary batch 1)
    page_10_class = next(p for p in final_classes if p.page_number == 10)
    assert page_10_class.classification == "invoice"

    # Conflict must be recorded
    assert len(conflicts) == 1
    assert conflicts[0]["page_number"] == 10
    assert conflicts[0]["primary_classification"] == "invoice"
    assert conflicts[0]["overlap_classification"] == "receipt"
    assert conflicts[0]["resolution"] == "PRIMARY_BATCH_PREFERRED"

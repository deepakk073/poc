# Stage 1: Test Cases & Implementation Guide

## Test Suite for Document Classification

### Test Case 1: Digital Invoice (Good Quality, No Handwriting)
```
Test ID: T1-DI-GOOD
Filename: digital_invoice_clean.pdf
Characteristics:
  - Native digital PDF (searchable text)
  - Invoice #INV-2026-12345
  - High resolution (300 DPI)
  - No skew (<1°)
  - No handwritten content
  
Expected Output:
  form_type: "digital_invoice"
  confidence: 0.95+
  quality_rating: "good" (0.90+)
  handwriting: false
  manual_review: false
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 2: Digital Invoice (Poor Quality, Heavy Skew)
```
Test ID: T2-DI-POOR
Filename: digital_invoice_skewed_blurry.pdf
Characteristics:
  - Digital format but poor scan
  - Skew: ~15°
  - Blur detected (high variance)
  - OCR confidence: 75%
  
Expected Output:
  form_type: "digital_invoice"
  confidence: 0.85+
  quality_rating: "poor" (<0.70)
  issues: ["high_skew", "blurred"]
  manual_review: true
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 3: Manual Invoice (Scanned, <10% Handwriting)
```
Test ID: T3-MI-HANDWRITING-LOW
Filename: manual_invoice_mostly_typed.pdf
Characteristics:
  - Scan of hardcopy form
  - 95% typed, 5% handwritten notes
  - Good scan quality (300 DPI)
  - No overwrites
  - Clean scan (no shadows, creases)
  
Expected Output:
  form_type: "manual_invoice"
  confidence: 0.90+
  quality_rating: "good"
  handwriting_percentage: 0.05
  has_handwriting: false (threshold >10%)
  manual_review: false
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 4: Manual Invoice (>10% Handwriting, No Overwrites)
```
Test ID: T4-MI-HANDWRITING-MID
Filename: manual_invoice_mixed_handwritten.pdf
Characteristics:
  - Scan of form with handwritten fields
  - ~20% handwritten (fields, notes)
  - Handwritten text clear, legible
  - No strike-throughs or overwrites
  - Quality: 200 DPI scan
  
Expected Output:
  form_type: "manual_invoice"
  confidence: 0.88+
  quality_rating: "acceptable"
  handwriting_percentage: 0.20
  has_handwriting: true
  overwrites_detected: []
  llm_routing: "claude"
  manual_review: false (but flagged for special handling)
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 5: Purchase Order with Overwrite (CRITICAL YOUR USE CASE)
```
Test ID: T5-PO-OVERWRITE
Filename: purchase_order_quantity_overwrite.pdf
Characteristics:
  - Digital PO form
  - Quantity field shows: "10" (typed), "8" (handwritten on top)
  - Overwrite clearly visible
  - Cross-out indicator present
  - Image analysis can distinguish both values
  
Expected Output:
  form_type: "purchase_order"
  confidence: 0.90+
  handwriting_percentage: 0.15
  has_handwriting: true
  
  overwrites_detected: [
    {
      "type": "overwrite",
      "page": 1,
      "field_name": "quantity",
      "original_value": "10",
      "overwritten_value": "8",
      "confidence": 0.70,
      "action": "extract_both_values_for_llm_decision"
    }
  ]
  
  confidence_baseline: 0.88 (reduced due to uncertainty)
  manual_review: true (for overwrite confirmation)
  llm_routing: "claude"
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 6: Consent Form (Signature Detection)
```
Test ID: T6-CF-SIGNATURE
Filename: consent_form_signed.pdf
Characteristics:
  - Long-form text document
  - Contains "Consent", "Agreement", "Terms"
  - Signature field present (handwritten)
  - Date field (typed)
  - High quality scan
  
Expected Output:
  form_type: "consent_form"
  confidence: 0.92+
  quality_rating: "good"
  handwriting: true (signature region)
  form_fields_detected: ["signature", "date"]
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 7: Vendor Delivery Form (Handwritten Notes)
```
Test ID: T7-VDF-HANDWRITTEN
Filename: delivery_form_handwritten_notes.pdf
Characteristics:
  - Short form (half page)
  - Structured fields (mostly typed)
  - Handwritten notes in "Special Instructions" section (~15%)
  - Signature present
  - Variable quality (mixed typed + handwritten)
  
Expected Output:
  form_type: "vendor_delivery_form"
  confidence: 0.90+
  quality_rating: "acceptable"
  handwriting_percentage: 0.15
  has_handwriting: true
  form_fields: ["delivery_date", "signature", "recipient"]
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 8: Feedback Form (Survey)
```
Test ID: T8-FF-SURVEY
Filename: feedback_form_response.pdf
Characteristics:
  - Survey form with ratings
  - Checkmarks / filled circles
  - Open text response area (handwritten)
  - Mixed quality
  
Expected Output:
  form_type: "feedback_form"
  confidence: 0.85+
  handwriting: true (open text section)
  form_fields: ["rating", "comment_field"]
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 9: Unknown Form Type
```
Test ID: T9-UNKNOWN
Filename: generic_document.pdf
Characteristics:
  - Document with no clear invoice/PO/consent keywords
  - Mixed content types
  - Could be any form type
  
Expected Output:
  form_type: "unknown"
  confidence: 0.50 (low)
  alternative_forms: [
    { "form_type": "generic_document", "confidence": 0.40 },
    { "form_type": "invoice", "confidence": 0.10 }
  ]
  manual_review: true
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 10: Multi-Page Document (Variable Quality)
```
Test ID: T10-MULTI-PAGE
Filename: invoice_3pages_mixed_quality.pdf
Characteristics:
  - Page 1: 300 DPI, good quality
  - Page 2: 200 DPI, acceptable quality
  - Page 3: 150 DPI, poor quality (scanned from copy)
  - Mixed handwriting across pages
  
Expected Output:
  form_type: "digital_invoice"
  overall_quality_score: 0.70 (average)
  quality_rating: "acceptable"
  page_quality_breakdown: [
    { "page": 1, "score": 0.92 },
    { "page": 2, "score": 0.78 },
    { "page": 3, "score": 0.48 }
  ]
  issues: [
    { "page": 3, "issue": "low_dpi", "severity": "high" }
  ]
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 11: Strike-Through Overwrite
```
Test ID: T11-STRIKETHROUGH
Filename: order_with_strikethrough.pdf
Characteristics:
  - Order form
  - Original value: "10 units"
  - User strikes through and writes "8 units"
  - Clear visual strike-through line
  
Expected Output:
  form_type: "purchase_order"
  
  overwrites_detected: [
    {
      "type": "cross_out",
      "page": 1,
      "field": "quantity",
      "original_text": "10 units",
      "new_text": "8 units",
      "action": "flag_for_manual_review"
    }
  ]
  
Status: [ ] PASS [ ] FAIL
```

### Test Case 12: Multiple Overwrites in Single Document
```
Test ID: T12-MULTI-OVERWRITES
Filename: purchase_order_heavily_edited.pdf
Characteristics:
  - PO with 3+ overwritten fields
  - Date changed
  - Quantity changed
  - Delivery address corrected
  
Expected Output:
  overwrites_detected: [
    { "field": "date", ... },
    { "field": "quantity", ... },
    { "field": "delivery_address", ... }
  ]
  
  confidence_baseline: 0.75 (significantly reduced)
  manual_review: true (required)
  recommendation: "high_touch_review_needed"
  
Status: [ ] PASS [ ] FAIL
```

---

## Implementation Checklist

### Week 1: Setup & Core Infrastructure

- [ ] **GCP Project Setup**
  - [ ] Create Document AI processors:
    - [ ] Layout Processor (for content analysis)
    - [ ] OCR Processor (for text extraction)
    - [ ] Forms Processor (for form field detection)
  - [ ] Enable Vision API
  - [ ] Create service account with appropriate IAM roles
  - [ ] Test Document AI credentials

- [ ] **Install Dependencies**
  ```bash
  pip install google-cloud-documentai google-cloud-vision google-cloud-storage Pillow numpy opencv-python PyPDF2 pdf2image
  ```

- [ ] **Document AI Processor Configuration**
  - [ ] Document the processor IDs (save in config)
  - [ ] Test each processor with sample documents
  - [ ] Document expected outputs

### Week 2: Development & Testing

- [ ] **Implement Core Classification Logic**
  - [ ] Form type classifier (keyword-based)
  - [ ] Quality assessment module
  - [ ] Handwriting detector
  - [ ] Overwrite detector (CRITICAL)

- [ ] **Testing**
  - [ ] Unit tests for each detection function
  - [ ] Integration test with Document AI + Vision API
  - [ ] End-to-end test with 50-doc sample
  - [ ] Edge case testing (overwrites, mixed quality, etc.)

- [ ] **Performance Optimization**
  - [ ] Batch processing for multiple documents
  - [ ] Caching strategy for repeated images
  - [ ] Parallel page processing

- [ ] **Error Handling**
  - [ ] Fallback logic when Document AI fails
  - [ ] Graceful degradation for Vision API errors
  - [ ] Timeout handling for slow documents

- [ ] **Validation Against Test Cases**
  - [ ] Run all 12 test cases
  - [ ] Document results
  - [ ] Adjust thresholds based on results

---

## Quick Reference: Key Functions

### Main Entry Point
```python
classifier = DocumentClassifier(project_id="your-project")
result = classifier.classify_document(
    pdf_path="gs://bucket/document.pdf",
    document_id="doc_12345"
)
```

### Critical Detection Functions

**Form Type Classification:**
```python
form_type, confidence = classifier._classify_form_type(pdf_bytes)
```

**Quality Assessment:**
```python
quality = classifier._assess_quality(pdf_bytes, pdf_pages)
# Returns: overall_score, rating, issues
```

**Handwriting Detection:**
```python
hw_analysis = classifier._analyze_handwriting_and_overwrites(pdf_pages)
# Returns: has_handwriting, percentage, overwrites[], regions[]
```

**Overwrite Detection (YOUR KEY FUNCTION):**
```python
overwrites = classifier._detect_overwrites(image, page_num, hw_regions)
# Returns: [{type, location, original_value, overwritten_value, action}]
```

**Page Content Analysis:**
```python
content_summary, routing = classifier._analyze_page_content(pdf_pages)
# Returns: text%, table%, image%, dominant_type, segments[]
```

---

## Configuration Tuning

### Quality Thresholds
```python
# Adjust based on your test results
quality_thresholds = {
    "good": 0.85,      # Score >= 85% is "good"
    "acceptable": 0.70 # Score >= 70% is "acceptable"
    # Score < 70% is "poor"
}
```

### Handwriting Detection Threshold
```python
# Change if you want to flag < or > 10% handwriting
handwriting_threshold = 0.10  # 10% of document
```

### Confidence Minimums by Form Type
```python
min_confidence = {
    FormType.DIGITAL_INVOICE: 0.95,
    FormType.MANUAL_INVOICE: 0.85,
    FormType.PURCHASE_ORDER: 0.90,
    FormType.CONSENT_FORM: 0.85,
    FormType.VENDOR_DELIVERY_FORM: 0.90,
    FormType.FEEDBACK_FORM: 0.85,
}
```

---

## Expected Outputs

### Success Output (Good Document)
```json
{
  "document_id": "doc_123",
  "form_type": "digital_invoice",
  "form_type_confidence": 0.95,
  "quality_metrics": {
    "overall_score": 0.92,
    "rating": "good",
    "issues": []
  },
  "handwriting_analysis": {
    "has_handwriting": false,
    "overwrites_detected": []
  },
  "next_stage_input": {
    "extraction_strategy": "multi_method_routing",
    "manual_review_required": false
  }
}
```

### Flag Output (Overwrite Detected)
```json
{
  "document_id": "doc_456",
  "form_type": "purchase_order",
  "form_type_confidence": 0.90,
  "handwriting_analysis": {
    "has_handwriting": true,
    "overwrites_detected": [
      {
        "type": "overwrite",
        "page": 1,
        "field_name": "Order_Quantity",
        "original_value": "10",
        "overwritten_value": "8",
        "confidence": 0.70,
        "action": "extract_both_values_for_llm_decision"
      }
    ]
  },
  "next_stage_input": {
    "manual_review_required": true,
    "llm_routing": "claude"
  }
}
```

---

## Integration with Next Stages

### Input to Stage 2: Extraction
```
Stage 1 Output:
  - form_type: Tells Stage 2 which extraction method to use
  - extraction_routing: Page-by-page routing (table vs text vs image)
  - metadata_schema: What fields to extract
  - confidence_baseline: Quality indicator for result validation

Stage 2 uses this to:
  - Select appropriate Document AI processor
  - Prepare LLM prompts for refinement
  - Set confidence thresholds for acceptance
```

### Input to Stage 3: Metadata + Embeddings
```
Stage 1 provides:
  - Suggested field schema
  - LLM routing decision (Gemini vs Claude vs others)
  - Quality indicators for weighting

Stage 3 uses to:
  - Guide LLM metadata generation
  - Weight embeddings based on quality
  - Flag low-confidence regions for manual review
```

---

## Monitoring & Metrics

### Key Metrics to Track

1. **Classification Accuracy**
   - % of correct form type detections
   - Confidence score distribution
   - False positive / false negative rates

2. **Quality Assessment Correlation**
   - Does quality_score correlate with extraction accuracy?
   - (Should be R² > 0.80)

3. **Handwriting Detection Sensitivity**
   - % of handwritten documents correctly identified
   - % false positives (flagging non-handwritten as handwritten)

4. **Overwrite Detection**
   - % of overwrites correctly identified
   - % missed overwrites (critical!)
   - False positive overwrites

5. **Processing Performance**
   - Avg latency per document
   - (Target: <2 seconds)
   - % documents requiring manual review

### Sample Dashboard Metrics
```
Week 2 Results (50 docs):
  ✓ Form type accuracy: 96% (48/50)
  ✓ Handwriting detection: 95% sensitivity
  ✓ Overwrite detection: 88% sensitivity (7/8 caught)
  ✓ Avg processing time: 1.8 seconds/doc
  ✓ Manual review rate: 8% (4/50 docs)
```

---

## Troubleshooting

### Issue: Low Confidence on Digital Invoices
```
Solution:
  1. Check keyword list — are your invoice keywords comprehensive?
  2. Add more keywords to self.form_keywords
  3. Lower confidence threshold if appropriate
```

### Issue: Handwriting Not Detected
```
Solution:
  1. Check Vision API handwriting detection settings
  2. Verify image quality (need clear images)
  3. Increase handwriting percentage threshold
```

### Issue: Overwrites Missed
```
Solution:
  1. Check strike-through detection logic
  2. Verify overlapping text detection sensitivity
  3. Implement additional overwrite detection methods
  4. Manual inspection of false negatives
```

### Issue: Quality Score Too Low
```
Solution:
  1. Review threshold values
  2. Check if documents naturally have lower quality
  3. Adjust weights in quality_score calculation
  4. Consider per-form-type thresholds
```


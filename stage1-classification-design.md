# Stage 1: Document Classification Pipeline
## Complete Design for Form Type Detection, Quality Assessment, and Content Routing

---

## 1. Classification Architecture Overview

```
INPUT: PDF (200/day, 500 pages avg)
  ↓
[1.1 DOCUMENT-LEVEL CLASSIFICATION]
  - Form Type Detection (6 types)
  - Quality Assessment
  - Handwriting Detection & Overwrite Detection
  ↓
[1.2 PAGE-LEVEL CONTENT ANALYSIS]
  - Content Type Detection (image, table, text, mixed)
  - Page Quality Score
  - Section Segmentation
  ↓
[1.3 METADATA SCHEMA GENERATION]
  - Intelligent schema creation per document
  - LLM guidance application
  - Flexible field extraction planning
  ↓
OUTPUT: Classification Metadata JSON + Routing Instructions
```

---

## 2. Form Types & Characteristics

### Form Type 1: Digital Invoice
- **Characteristics**: Native digital PDF, structured text, clear formatting
- **Detection signals**:
  - Text layers present (searchable PDF)
  - Regular fonts, consistent spacing
  - Invoice keywords: "Invoice #", "Total", "Due Date"
- **Confidence threshold**: 95%+ (high confidence possible)
- **Extraction method**: Document AI Forms Parser
- **Quality baseline**: 90%+ (usually high quality)

### Form Type 2: Manual Invoice (Scanned)
- **Characteristics**: Scan of hardcopy form, may have handwritten fields, variable quality
- **Detection signals**:
  - No text layer OR text layer from OCR
  - Mixed text and image elements
  - Handwritten annotations likely
  - Invoice keywords + scan artifacts (shadows, creases)
- **Confidence threshold**: 85% (lower than digital)
- **Extraction method**: Document AI + Vision API + OCR
- **Quality baseline**: 70-85% (scan quality varies)

### Form Type 3: Purchase Order
- **Characteristics**: Can be digital or scanned, structured table format
- **Detection signals**:
  - "PO #", "Requisition", "Vendor", "Item", "Quantity"
  - Table structure present (multiple rows)
  - Date fields, approval signatures
- **Confidence threshold**: 90%
- **Extraction method**: Document AI Table Parser + Vision API
- **Quality baseline**: 80%+

### Form Type 4: Consent Form
- **Characteristics**: Long-form text, legal language, checkbox/signature fields
- **Detection signals**:
  - "Consent", "Agreement", "Terms", "Signature"
  - Dense paragraph text
  - Checkboxes, radio buttons, signature lines
- **Confidence threshold**: 85%
- **Extraction method**: Document AI + Vision API for checkboxes/signatures
- **Quality baseline**: 85%+

### Form Type 5: Vendor Delivery Form
- **Characteristics**: Short form with essential fields, may have handwritten notes
- **Detection signals**:
  - "Delivery", "Received", "Delivered Date", "Signature"
  - Short structured fields + optional notes
  - Often signed/dated
- **Confidence threshold**: 90%
- **Extraction method**: Document AI + Vision API + Handwriting detection
- **Quality baseline**: 75-85%

### Form Type 6: Feedback Form
- **Characteristics**: Survey-like, ratings, open text, may have handwritten responses
- **Detection signals**:
  - "Feedback", "Rating", "Comment", "Opinion"
  - Checkmarks, scales, open text areas
- **Confidence threshold**: 85%
- **Extraction method**: Document AI + Vision API
- **Quality baseline**: 70-80% (variable handwriting)

---

## 3. Document-Level Classification Decision Tree

```
PDF Input
  ↓
[Step 1: Check if searchable text exists]
  → YES: Digital format (likely Digital Invoice, PO, Consent Form)
  → NO: Scanned format (likely Manual Invoice, Delivery Form, Feedback Form)
  ↓
[Step 2: Extract text + analyze keywords (Document AI + Vision API)]
  ↓
  BRANCH A: If "Invoice" keywords + date + amount fields
    → IF handwritten % > 10%
        → Manual Invoice (Scanned)
    → ELSE
        → Digital Invoice
  ↓
  BRANCH B: If "PO" keywords + "Quantity", "Item", "Vendor"
    → Purchase Order
  ↓
  BRANCH C: If "Consent", "Agreement", "Terms", "Signature" + dense text
    → Consent Form
  ↓
  BRANCH D: If "Delivery", "Received", "Signature" + short structure
    → Vendor Delivery Form
  ↓
  BRANCH E: If "Feedback", "Rating", "Comment", "Opinion" + checkmarks
    → Feedback Form
  ↓
  BRANCH F: If no clear match
    → FALLBACK: Use Vision API + LLM classifier
    → Return: form_type = "unknown", confidence = low, require_manual_review = true
  ↓
[Step 3: Quality Assessment]
  ↓
[Step 4: Handwriting Detection (including overwrite detection)]
  ↓
[Step 5: Output Classification Metadata]
```

---

## 4. Quality Assessment Thresholds

### Quality Score Calculation (0-1 scale)

```python
quality_score = (
    image_quality_score * 0.30 +      # 30% - resolution, contrast, blur
    skew_score * 0.20 +                # 20% - document alignment
    text_readability_score * 0.25 +    # 25% - OCR confidence, clarity
    completeness_score * 0.25          # 25% - all pages present, no major artifacts
)
```

### Quality Thresholds by Form Type

| Form Type | Good (>0.85) | Acceptable (0.70-0.85) | Poor (<0.70) |
|-----------|--------------|------------------------|--------------|
| Digital Invoice | DPI 300+, no skew, clear | DPI 200+, <2° skew | DPI <200, >5° skew, blurry |
| Manual Invoice | Scan 300 DPI, <30° angle | Scan 200 DPI, <45° angle | Scan <150 DPI, >45° angle |
| Purchase Order | Clear tables, no overlap | Slight shadows, readable | Poor scans, illegible fields |
| Consent Form | Clear text, <5% illegible | Some noise, <20% illegible | >20% illegible text |
| Vendor Delivery | Good handwriting clarity | Mixed quality | Poor handwriting clarity |
| Feedback Form | Clear responses | Some responses unclear | Most responses unclear |

### Specific Quality Metrics

1. **Image DPI (for scanned docs)**
   - Good: ≥300 DPI
   - Acceptable: 200-300 DPI
   - Poor: <200 DPI

2. **Skew Detection**
   - Good: <1° skew
   - Acceptable: 1-5° skew
   - Poor: >5° skew

3. **Blur Detection (Vision API)**
   - Good: blur score <0.3
   - Acceptable: blur score 0.3-0.6
   - Poor: blur score >0.6

4. **OCR Confidence**
   - Good: >95% character confidence
   - Acceptable: 80-95% confidence
   - Poor: <80% confidence

5. **Text-to-Image Ratio**
   - Good: >70% text (for text-heavy forms)
   - Acceptable: 50-70% text
   - Poor: <50% text (likely image-heavy or scanned)

---

## 5. Handwriting Detection & Overwrite Detection

### 5.1 Handwriting Detection (threshold: >10% → special handling)

```python
def detect_handwriting(page_image, page_text):
    """
    Detect if page has >10% handwritten content.
    
    Detection signals:
    1. Vision API handwriting annotations
    2. Text stroke width variation (Vision API StructuralUnit)
    3. Irregular character spacing
    4. Non-standard font detection
    """
    
    # Use Vision API's DOCUMENT_TEXT_DETECTION
    handwritten_regions = vision_api.detect_handwriting(page_image)
    
    handwriting_percentage = (
        handwritten_regions.bounding_boxes_area / total_page_area
    )
    
    return {
        "has_handwriting": handwriting_percentage > 0.10,
        "handwriting_percentage": handwriting_percentage,
        "confidence": vision_api_confidence_score,
        "regions": handwritten_regions  # Page coordinates
    }
```

### 5.2 CRITICAL: Overwrite Detection (Your Use Case!)

**Problem**: User writes "10", then crosses it out and writes "8" on top.

```python
def detect_overwrites(page_image, page_text, handwritten_regions):
    """
    Detect when handwriting overwrites or modifies existing content.
    
    Signals:
    1. Multiple text layers at same location (Vision API can detect)
    2. Cross-outs / strike-throughs
    3. High text density in small region (multiple overlapping characters)
    4. Color changes (ink color indicating different pen)
    5. Font style inconsistency at same field location
    """
    
    overwrites = []
    
    for region in handwritten_regions:
        # Check for cross-outs
        if detect_strike_through(region.image):
            overwrites.append({
                "type": "cross_out",
                "location": region.bounding_box,
                "original_text": extract_text_under_strikethrough(region),
                "new_text": region.text,
                "action": "FLAG_FOR_MANUAL_REVIEW"
            })
        
        # Check for overlapping text (high density)
        if detect_overlapping_text(region):
            overwrites.append({
                "type": "overwrite",
                "location": region.bounding_box,
                "possible_original": attempt_ocr_under_text(region),
                "visible_text": region.text,
                "action": "CAPTURE_BOTH_VALUES_WITH_CONFIDENCE"
            })
        
        # Check for ink color changes at same field
        if detect_ink_color_change(region):
            overwrites.append({
                "type": "ink_change",
                "location": region.bounding_box,
                "inks_detected": region.ink_colors,
                "action": "EXTRACT_LATEST_INK_COLOR"
            })
    
    return {
        "has_overwrites": len(overwrites) > 0,
        "overwrites": overwrites,
        "confidence": vision_confidence_score,
        "recommendation": "manual_review" if len(overwrites) > 0 else "auto_process"
    }
```

### 5.3 Handwriting Routing

```
If handwriting_percentage > 10%:
  ├─ If overwrites detected:
  │   └─ Flag: confidence_level = "low", require_manual_review = true
  │   └─ Extract both values: original (if readable) + overwrite
  │   └─ Route to: Claude LLM (reasoning about which value is correct)
  │
  └─ If no overwrites:
      └─ Extract handwritten text with Vision API + OCR
      └─ Route to: Document AI + Vision API + LLM refinement
      └─ Confidence: reduced by 10-20% (handwriting uncertainty)
```

---

## 6. Page-Level Content Type Detection

### Content Types
1. **Pure Text**: Paragraphs, lists, dense text
2. **Table**: Structured rows/columns
3. **Image**: Photos, diagrams, charts
4. **Form Fields**: Checkboxes, radio buttons, signature lines
5. **Mixed**: Combination of above

### Detection Logic

```python
def detect_page_content_types(page_image, page_text):
    """
    Analyze single page and return composition.
    """
    
    content_composition = {
        "text_percentage": 0.0,
        "table_percentage": 0.0,
        "image_percentage": 0.0,
        "form_fields_percentage": 0.0,
        "dominant_type": "",
        "segments": []  # Page regions with their type
    }
    
    # Use Document AI LAYOUT_ANALYSIS
    layout_analysis = doc_ai.analyze_layout(page_image)
    
    for region in layout_analysis.regions:
        if region.type == "TEXT":
            content_composition["text_percentage"] += region.area_percentage
            content_composition["segments"].append({
                "type": "text",
                "bounding_box": region.bbox,
                "content": region.extracted_text
            })
        
        elif region.type == "TABLE":
            content_composition["table_percentage"] += region.area_percentage
            content_composition["segments"].append({
                "type": "table",
                "bounding_box": region.bbox,
                "rows": region.table_rows,
                "columns": region.table_cols
            })
        
        elif region.type == "IMAGE":
            content_composition["image_percentage"] += region.area_percentage
            content_composition["segments"].append({
                "type": "image",
                "bounding_box": region.bbox,
                "image_description": vision_api.describe_image(region.image)
            })
        
        elif region.type in ["CHECKBOX", "RADIO_BUTTON", "SIGNATURE_FIELD"]:
            content_composition["form_fields_percentage"] += region.area_percentage
            content_composition["segments"].append({
                "type": "form_field",
                "field_type": region.type,
                "bounding_box": region.bbox,
                "is_filled": region.is_filled
            })
    
    # Determine dominant type
    composition_dict = {
        "text": content_composition["text_percentage"],
        "table": content_composition["table_percentage"],
        "image": content_composition["image_percentage"],
        "form_fields": content_composition["form_fields_percentage"]
    }
    content_composition["dominant_type"] = max(composition_dict, key=composition_dict.get)
    
    return content_composition
```

### Routing by Content Type

```
For each page:
  If dominant_type == "TABLE":
    → Use: Document AI Table Parser
    
  If dominant_type == "IMAGE":
    → Use: Vision API + Image Description
    
  If dominant_type == "TEXT":
    → Use: Document AI OCR + Vision API
    
  If contains "FORM_FIELDS":
    → Use: Document AI Forms Parser
    
  If "MIXED":
    → Process segments in order
    → Combine results with positional awareness
    → Use LLM to correlate segments
```

---

## 7. Classification Output Schema

### Document-Level Classification Metadata

```json
{
  "document_id": "UUID",
  "pdf_name": "string",
  "total_pages": "integer",
  
  "document_classification": {
    "form_type": "digital_invoice | manual_invoice | purchase_order | consent_form | vendor_delivery_form | feedback_form | unknown",
    "form_type_confidence": 0.95,
    "form_type_description": "string (human readable)",
    "alternative_forms": [
      {
        "form_type": "string",
        "confidence": 0.05
      }
    ]
  },
  
  "quality_assessment": {
    "overall_quality_score": 0.87,
    "quality_rating": "good | acceptable | poor",
    "image_quality_score": 0.90,
    "skew_score": 0.85,
    "ocr_confidence": 0.92,
    "text_readability": 0.88,
    "issues_detected": [
      {
        "issue": "slight_skew",
        "severity": "low",
        "location": "pages 1-3"
      }
    ]
  },
  
  "handwriting_analysis": {
    "has_handwriting": true,
    "handwriting_percentage": 0.12,
    "handwriting_confidence": 0.88,
    "overwrites_detected": [
      {
        "type": "cross_out | overwrite | ink_change",
        "page": 2,
        "field_name": "Order_Quantity",
        "original_value": "10",
        "overwritten_value": "8",
        "confidence": 0.75,
        "action": "manual_review | auto_process"
      }
    ]
  },
  
  "content_summary": {
    "total_segments": 15,
    "text_percentage": 0.65,
    "table_percentage": 0.20,
    "image_percentage": 0.10,
    "form_fields_percentage": 0.05,
    "dominant_content_type": "text"
  },
  
  "extraction_routing": [
    {
      "page": 1,
      "segments": [
        {
          "segment_id": "page1_seg1",
          "type": "text",
          "recommended_extractor": "document_ai_ocr",
          "confidence": 0.95
        },
        {
          "segment_id": "page1_seg2",
          "type": "table",
          "recommended_extractor": "document_ai_table_parser",
          "confidence": 0.92
        }
      ]
    }
  ],
  
  "metadata_schema_generation": {
    "schema_version": "1.0",
    "suggested_fields": [
      {
        "field_name": "invoice_number",
        "data_type": "string",
        "is_required": true,
        "extraction_guidance": "Located at top-right, numeric/alphanumeric"
      },
      {
        "field_name": "invoice_date",
        "data_type": "date",
        "is_required": true,
        "extraction_guidance": "Format: MM/DD/YYYY, appears near invoice number"
      },
      {
        "field_name": "line_items",
        "data_type": "array",
        "is_required": true,
        "extraction_guidance": "Tabular format, columns: description, quantity, unit_price, total"
      }
    ],
    "flexible_fields": [
      "notes",
      "special_instructions",
      "handwritten_annotations"
    ]
  },
  
  "next_stage_input": {
    "extraction_strategy": "document_ai + vision_api + multi_section_routing",
    "llm_routing": "claude_for_complex_reasoning",
    "confidence_baseline": 0.87,
    "quality_gate_pass": true,
    "manual_review_required": false,
    "handwriting_handling_special": true
  },
  
  "metadata": {
    "classified_at": "2026-09-23T10:30:00Z",
    "classification_service_version": "1.0",
    "gcs_path": "gs://pdf-bucket/input/invoice_sept_2026.pdf"
  }
}
```

### Page-Level Content Breakdown

```json
{
  "page_number": 1,
  "page_size": {"width": 8.5, "height": 11, "unit": "inches"},
  
  "content_composition": {
    "text_percentage": 0.65,
    "table_percentage": 0.20,
    "image_percentage": 0.10,
    "form_fields_percentage": 0.05,
    "dominant_type": "text"
  },
  
  "segments": [
    {
      "segment_id": "page1_seg1",
      "type": "text",
      "bounding_box": [0.05, 0.05, 0.95, 0.30],
      "text_preview": "Invoice #12345\nDate: 09/15/2026\n...",
      "extraction_method": "document_ai_ocr",
      "priority": "high"
    },
    {
      "segment_id": "page1_seg2",
      "type": "table",
      "bounding_box": [0.05, 0.35, 0.95, 0.80],
      "table_preview": "Description | Qty | Unit Price | Total",
      "num_rows": 8,
      "num_cols": 4,
      "extraction_method": "document_ai_table_parser",
      "priority": "high"
    },
    {
      "segment_id": "page1_seg3",
      "type": "handwritten",
      "bounding_box": [0.05, 0.80, 0.40, 0.95],
      "text_preview": "John Doe - Signature",
      "handwriting_confidence": 0.85,
      "extraction_method": "vision_api + ocr",
      "priority": "medium"
    }
  ]
}
```

---

## 8. Edge Cases & Fallback Logic

### Edge Case 1: Handwritten Overwrites (Your Primary Use Case)

```
Scenario: PO shows "Quantity: 10" (typed), user crosses out and writes "8"

Detection:
  → Vision API detects strike-through
  → Overwrite detector identifies original "10" + new "8"

Output:
  {
    "field": "order_quantity",
    "original_value": "10",
    "overwritten_value": "8",
    "overwrite_confidence": 0.75,
    "recommendation": "use_overwritten_value",
    "routing": "claude_llm_for_confirmation"
  }

Action:
  → Flag with medium confidence (0.75 instead of 0.95)
  → Provide both values to LLM
  → LLM should prefer the overwritten value (latest intent)
```

### Edge Case 2: Partially Scanned Document

```
Scenario: First 2 pages are high quality, pages 3-5 are low quality

Action:
  → Quality score: 0.65 (average of all pages)
  → Flag: "variable_quality_detected"
  → Recommendation: "process_with_caution"
  → Per-page quality tracking
  → Different confidence thresholds per page
```

### Edge Case 3: Unclear Form Type

```
Scenario: Document contains mixed elements (invoice header + purchase order table)

Action:
  → Return multiple form types with confidence scores
  → Top: "digital_invoice" (0.55)
  → Alternative: "purchase_order" (0.40)
  → Alternative: "unknown" (0.05)
  → Routing: "require_manual_review"
  → LLM: Extract common fields regardless of form type
```

### Edge Case 4: Low OCR Confidence

```
Scenario: Scanned document with poor quality, OCR confidence <80%

Action:
  → Flag quality score as "poor"
  → Reduce extraction confidence baseline by 20%
  → Route to: Vision API + LLM (reasoning)
  → Manual review recommendation: YES
```

---

## 9. Implementation Strategy (Weeks 1-2)

### Week 1
- [ ] Document AI processor setup (Forms, OCR, Layout Analysis)
- [ ] Vision API integration (handwriting detection, image analysis)
- [ ] Quality assessment logic (DPI, skew, blur detection)
- [ ] Form type classification rules (keyword-based)

### Week 2
- [ ] Handwriting detection + overwrite detection
- [ ] Page-level content analysis
- [ ] Classification output schema implementation
- [ ] 50-doc test suite (all form types, edge cases)
- [ ] Fallback routing logic

---

## 10. Testing Strategy

### Test Cases (Minimum Coverage)

| Test ID | Form Type | Quality | Handwriting | Expected Output | Status |
|---------|-----------|---------|-------------|-----------------|--------|
| T1 | Digital Invoice | Good | None | Confidence >95% | — |
| T2 | Digital Invoice | Poor | None | Confidence <85%, Quality Flag | — |
| T3 | Manual Invoice | Good | <10% | Confidence >90% | — |
| T4 | Manual Invoice | Good | >10%, No overwrites | Confidence 80-85%, Handwriting Flag | — |
| T5 | Manual Invoice | Good | >10%, WITH overwrites | Confidence 75-80%, Overwrite Flag, Manual Review | — |
| T6 | PO | Good | None | Form Type Confidence >92% | — |
| T7 | PO | Good | Overwritten quantity | Extract both values, Flag overwrite | — |
| T8 | Consent Form | Good | None | Form Type Confidence >90% | — |
| T9 | Vendor Delivery | Good | Handwritten signature | Confidence >85%, Signature detected | — |
| T10 | Feedback Form | Good | Mixed handwritten | Confidence 75-85%, Handwriting Flag | — |
| T11 | Unknown Type | Good | — | Form Type: "unknown", Confidence <70%, Manual Review | — |
| T12 | Multi-page mixed quality | Mixed | Mixed | Per-page quality scores, Average quality calculated | — |

---

## 11. Success Criteria (Week 2)

- ✅ Classify all 6 form types with >85% confidence (minimum)
- ✅ Detect handwriting in >95% of cases where present
- ✅ Detect overwrites in >90% of cases
- ✅ Quality score correlates with extraction accuracy (R² > 0.8)
- ✅ 50-doc test suite passes with <5% manual review requirement
- ✅ Classification latency: <2 seconds per document
- ✅ Zero false negatives on form type (better to be uncertain than wrong)


"""
Stage 1: Document Classification Pipeline
Detects form type, quality, handwriting, and content types for routing.

Uses:
- Google Document AI (document_intelligence, layout_processor)
- Google Vision API (handwriting detection, image properties)
- Custom logic for quality assessment and overwrites

Author: Data Architecture Team
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import math

from google.cloud import documentai
from google.cloud import vision
from google.cloud import storage
import numpy as np
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class FormType(Enum):
    DIGITAL_INVOICE = "digital_invoice"
    MANUAL_INVOICE = "manual_invoice"
    PURCHASE_ORDER = "purchase_order"
    CONSENT_FORM = "consent_form"
    VENDOR_DELIVERY_FORM = "vendor_delivery_form"
    FEEDBACK_FORM = "feedback_form"
    UNKNOWN = "unknown"

class ContentType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORM_FIELD = "form_field"
    MIXED = "mixed"

class QualityRating(Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"

@dataclass
class QualityMetrics:
    overall_score: float
    rating: str
    image_quality: float
    skew_score: float
    ocr_confidence: float
    text_readability: float
    issues: List[Dict]

@dataclass
class HandwritingAnalysis:
    has_handwriting: bool
    percentage: float
    confidence: float
    regions: List[Dict]
    overwrites: List[Dict]

@dataclass
class PageContentComposition:
    page_number: int
    text_percentage: float
    table_percentage: float
    image_percentage: float
    form_field_percentage: float
    dominant_type: str
    segments: List[Dict]

@dataclass
class ClassificationResult:
    document_id: str
    pdf_name: str
    total_pages: int
    form_type: str
    form_type_confidence: float
    quality_metrics: Dict
    handwriting_analysis: Dict
    content_summary: Dict
    extraction_routing: List[Dict]
    metadata_schema: Dict
    next_stage_input: Dict
    classified_at: str

# ============================================================================
# STAGE 1: DOCUMENT CLASSIFICATION
# ============================================================================

class DocumentClassifier:
    """
    Classifies PDF documents by form type, quality, and content composition.
    Orchestrates Document AI and Vision API for intelligent routing.
    """
    
    def __init__(self, project_id: str, location: str = "us"):
        """
        Initialize the classifier with GCP clients.
        
        Args:
            project_id: GCP project ID
            location: GCP region (default: us)
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize clients
        self.doc_ai_client = documentai.DocumentProcessorServiceClient()
        self.vision_client = vision.ImageAnnotatorClient()
        self.storage_client = storage.Client()
        
        # Document AI processor IDs (from Week 1 setup)
        self.processors = {
            'layout': f"projects/{project_id}/locations/{location}/processors/LAYOUT_PROCESSOR_ID",
            'ocr': f"projects/{project_id}/locations/{location}/processors/OCR_PROCESSOR_ID",
            'forms': f"projects/{project_id}/locations/{location}/processors/FORMS_PROCESSOR_ID"
        }
        
        # Form type keywords for classification
        self.form_keywords = {
            FormType.DIGITAL_INVOICE: [
                "invoice", "invoice #", "inv #", "bill", "amount due", "total"
            ],
            FormType.MANUAL_INVOICE: [
                "invoice", "handwritten", "scanned", "hardcopy"
            ],
            FormType.PURCHASE_ORDER: [
                "po #", "purchase order", "requisition", "vendor", "item", "quantity"
            ],
            FormType.CONSENT_FORM: [
                "consent", "agreement", "terms", "signature", "acknowledge"
            ],
            FormType.VENDOR_DELIVERY_FORM: [
                "delivery", "received", "delivered date", "signature", "received by"
            ],
            FormType.FEEDBACK_FORM: [
                "feedback", "survey", "rating", "comment", "opinion"
            ]
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "good": 0.85,
            "acceptable": 0.70,
            "poor": 0.00
        }
    
    # ========================================================================
    # MAIN CLASSIFICATION FLOW
    # ========================================================================
    
    def classify_document(self, pdf_path: str, document_id: str) -> ClassificationResult:
        """
        Classify a PDF document through the complete Stage 1 pipeline.
        
        Args:
            pdf_path: Path to PDF (local or GCS)
            document_id: Unique document identifier
            
        Returns:
            ClassificationResult with all metadata for downstream processing
        """
        logger.info(f"Starting classification for {document_id}")
        
        # Step 1: Load PDF
        pdf_bytes = self._load_pdf(pdf_path)
        pdf_pages = self._split_pdf_to_images(pdf_bytes)
        
        # Step 2: Document-level analysis
        form_type, form_confidence = self._classify_form_type(pdf_bytes)
        
        # Step 3: Quality assessment
        quality_metrics = self._assess_quality(pdf_bytes, pdf_pages)
        
        # Step 4: Handwriting and overwrite detection
        handwriting_analysis = self._analyze_handwriting_and_overwrites(pdf_pages)
        
        # Step 5: Page-level content analysis
        content_summary, extraction_routing = self._analyze_page_content(pdf_pages)
        
        # Step 6: Generate metadata schema
        metadata_schema = self._generate_metadata_schema(form_type, extraction_routing)
        
        # Step 7: Next stage routing
        next_stage_input = self._generate_routing_instructions(
            form_type, form_confidence, quality_metrics, handwriting_analysis
        )
        
        # Compile results
        result = ClassificationResult(
            document_id=document_id,
            pdf_name=pdf_path.split('/')[-1],
            total_pages=len(pdf_pages),
            form_type=form_type.value if isinstance(form_type, FormType) else str(form_type),
            form_type_confidence=form_confidence,
            quality_metrics=asdict(quality_metrics) if quality_metrics else {},
            handwriting_analysis=asdict(handwriting_analysis) if handwriting_analysis else {},
            content_summary=content_summary,
            extraction_routing=extraction_routing,
            metadata_schema=metadata_schema,
            next_stage_input=next_stage_input,
            classified_at=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(f"Classification complete: {form_type.value} ({form_confidence:.2%} confidence)")
        return result
    
    # ========================================================================
    # STEP 1: FORM TYPE CLASSIFICATION
    # ========================================================================
    
    def _classify_form_type(self, pdf_bytes: bytes) -> Tuple[FormType, float]:
        """
        Classify the form type using Document AI OCR + keyword matching.
        
        Returns:
            (form_type, confidence_score)
        """
        # Process with Document AI OCR
        raw_text = self._extract_text_with_doc_ai(pdf_bytes)
        text_lower = raw_text.lower()
        
        # Keyword-based classification
        form_scores = {}
        for form_type, keywords in self.form_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            form_scores[form_type] = score
        
        # Get top form type
        if max(form_scores.values()) == 0:
            return FormType.UNKNOWN, 0.0
        
        top_form = max(form_scores, key=form_scores.get)
        
        # Calculate confidence (simple heuristic)
        total_matches = max(form_scores.values())
        confidence = min(0.95, 0.70 + (total_matches * 0.10))
        
        logger.info(f"Form type: {top_form.value}, Confidence: {confidence:.2%}")
        return top_form, confidence
    
    def _extract_text_with_doc_ai(self, pdf_bytes: bytes) -> str:
        """Extract text using Document AI."""
        request = documentai.ProcessRequest(
            name=self.processors['ocr'],
            raw_document=documentai.RawDocument(
                content=pdf_bytes,
                mime_type="application/pdf"
            )
        )
        
        result = self.doc_ai_client.process_document(request=request)
        document = result.document
        
        return document.text
    
    # ========================================================================
    # STEP 2: QUALITY ASSESSMENT
    # ========================================================================
    
    def _assess_quality(self, pdf_bytes: bytes, pdf_pages: List[Image.Image]) -> QualityMetrics:
        """
        Assess document quality across multiple dimensions.
        
        Evaluates: DPI, skew, blur, OCR confidence, text readability.
        """
        quality_scores = []
        issues = []
        ocr_confidences = []
        
        for page_num, page_image in enumerate(pdf_pages, 1):
            # Image quality metrics
            image_quality = self._assess_image_quality(page_image)
            quality_scores.append(image_quality['score'])
            
            # Skew detection
            skew = self._detect_skew(page_image)
            if skew > 5:
                issues.append({
                    "issue": "high_skew",
                    "severity": "high" if skew > 15 else "low",
                    "skew_angle": round(skew, 2),
                    "page": page_num
                })
            
            # Blur detection
            blur_score = self._detect_blur(page_image)
            if blur_score < 0.3:
                issues.append({
                    "issue": "blurred",
                    "severity": "high",
                    "blur_score": round(blur_score, 2),
                    "page": page_num
                })
            
            # OCR confidence
            ocr_conf = self._extract_ocr_confidence(page_image)
            ocr_confidences.append(ocr_conf)
        
        # Calculate aggregated metrics
        overall_score = np.mean(quality_scores) if quality_scores else 0.0
        avg_ocr_confidence = np.mean(ocr_confidences) if ocr_confidences else 0.0
        text_readability = (avg_ocr_confidence * 0.7) + (1 - min(1.0, len(issues) / len(pdf_pages)) * 0.3)
        
        # Determine quality rating
        if overall_score >= self.quality_thresholds['good']:
            rating = QualityRating.GOOD.value
        elif overall_score >= self.quality_thresholds['acceptable']:
            rating = QualityRating.ACCEPTABLE.value
        else:
            rating = QualityRating.POOR.value
        
        return QualityMetrics(
            overall_score=round(overall_score, 2),
            rating=rating,
            image_quality=round(np.mean([s['score'] for s in quality_scores if isinstance(s, dict)]), 2) if any(isinstance(s, dict) for s in quality_scores) else round(overall_score, 2),
            skew_score=round(1.0 - min(1.0, np.mean([max(d.get('skew_angle', 0), 0) for d in issues if d.get('issue') == 'high_skew']) / 45), 2) if any(d.get('issue') == 'high_skew' for d in issues) else 0.95,
            ocr_confidence=round(avg_ocr_confidence, 2),
            text_readability=round(text_readability, 2),
            issues=issues
        )
    
    def _assess_image_quality(self, image: Image.Image) -> Dict:
        """Assess image quality using Vision API."""
        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Use Vision API
        vision_image = vision.Image(content=img_bytes.read())
        
        # Get image properties
        response = self.vision_client.image_properties(image=vision_image)
        props = response.image_properties_annotation
        
        # Calculate quality score based on color saturation
        avg_saturation = np.mean([c.saturation for c in props.dominant_colors]) if props.dominant_colors else 0.5
        dpi_estimate = image.info.get('dpi', (200, 200))[0]
        dpi_score = min(1.0, dpi_estimate / 300)
        
        quality_score = (avg_saturation * 0.4) + (dpi_score * 0.6)
        
        return {
            'score': quality_score,
            'dpi': dpi_estimate,
            'color_count': len(props.dominant_colors)
        }
    
    def _detect_skew(self, image: Image.Image) -> float:
        """
        Detect document skew angle.
        
        Returns: Skew angle in degrees (0 = straight, 45+ = significant skew)
        """
        # Placeholder: Use Vision API text detection to infer skew
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        vision_image = vision.Image(content=img_bytes.read())
        response = self.vision_client.text_detection(image=vision_image)
        
        if not response.text_annotations:
            return 0.0
        
        # Use bounding box vertices to calculate skew
        vertices = response.text_annotations[0].bounding_poly.vertices
        if len(vertices) >= 2:
            angle = math.atan2(
                vertices[1].y - vertices[0].y,
                vertices[1].x - vertices[0].x
            ) * 180 / math.pi
            return abs(angle)
        
        return 0.0
    
    def _detect_blur(self, image: Image.Image) -> float:
        """
        Detect image blur using Laplacian variance.
        
        Returns: Blur score (0-1, where 1 = no blur, 0 = heavily blurred)
        """
        import cv2
        
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Convert to grayscale
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 (empirically, >100 is sharp, <50 is blurry)
        blur_score = min(1.0, laplacian_var / 200)
        
        return blur_score
    
    def _extract_ocr_confidence(self, image: Image.Image) -> float:
        """Extract average OCR confidence from Vision API."""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        vision_image = vision.Image(content=img_bytes.read())
        response = self.vision_client.text_detection(image=vision_image)
        
        if not response.text_annotations:
            return 0.0
        
        # Average confidence of all detected text blocks
        confidences = [anno.confidence for anno in response.text_annotations[1:]]
        return np.mean(confidences) if confidences else 0.0
    
    # ========================================================================
    # STEP 3: HANDWRITING & OVERWRITE DETECTION
    # ========================================================================
    
    def _analyze_handwriting_and_overwrites(
        self, pdf_pages: List[Image.Image]
    ) -> HandwritingAnalysis:
        """
        Detect handwriting and identify overwrites (key edge case).
        
        Special handling for your case: "10" crossed out, "8" written on top.
        """
        total_handwriting_area = 0.0
        total_area = sum(img.width * img.height for img in pdf_pages)
        handwritten_regions = []
        overwrites = []
        
        for page_num, page_image in enumerate(pdf_pages, 1):
            # Detect handwriting with Vision API
            hw_regions = self._detect_handwriting_regions(page_image)
            
            for region in hw_regions:
                handwritten_regions.append({
                    'page': page_num,
                    'location': region['bbox'],
                    'text': region['text'],
                    'confidence': region['confidence']
                })
                total_handwriting_area += self._calc_bbox_area(region['bbox'], page_image)
            
            # Detect overwrites (CRITICAL for your use case)
            overwrites_in_page = self._detect_overwrites(page_image, page_num, hw_regions)
            overwrites.extend(overwrites_in_page)
        
        handwriting_percentage = total_handwriting_area / total_area if total_area > 0 else 0.0
        has_handwriting = handwriting_percentage > 0.10
        
        return HandwritingAnalysis(
            has_handwriting=has_handwriting,
            percentage=round(handwriting_percentage, 3),
            confidence=0.88,  # Vision API confidence
            regions=handwritten_regions,
            overwrites=overwrites
        )
    
    def _detect_handwriting_regions(self, image: Image.Image) -> List[Dict]:
        """Detect handwritten text regions using Vision API."""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        vision_image = vision.Image(content=img_bytes.read())
        
        # Use DOCUMENT_TEXT_DETECTION for handwriting
        response = self.vision_client.document_text_detection(image=vision_image)
        
        handwriting_regions = []
        
        for block in response.full_text_annotation.pages[0].blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    # Check if word is handwritten
                    property_list = word.property.detected_languages if word.property else None
                    if self._is_handwritten(word):
                        handwriting_regions.append({
                            'text': ''.join(s.text for s in word.symbols),
                            'bbox': self._vertices_to_bbox(word.bounding_box.vertices),
                            'confidence': word.confidence if hasattr(word, 'confidence') else 0.85
                        })
        
        return handwriting_regions
    
    def _detect_overwrites(
        self, image: Image.Image, page_num: int, hw_regions: List[Dict]
    ) -> List[Dict]:
        """
        CRITICAL: Detect when handwriting overwrites existing content.
        
        Your use case: "10" typed, then "8" handwritten on top.
        """
        overwrites = []
        
        for hw_region in hw_regions:
            # Check for strike-through indicator
            if self._detect_strike_through(image, hw_region['bbox']):
                overwrites.append({
                    'type': 'cross_out',
                    'page': page_num,
                    'location': hw_region['bbox'],
                    'visible_text': hw_region['text'],
                    'confidence': 0.75,
                    'action': 'flag_for_manual_review'
                })
            
            # Check for overlapping text (high text density)
            if self._detect_overlapping_text(image, hw_region['bbox']):
                overwrites.append({
                    'type': 'overwrite',
                    'page': page_num,
                    'location': hw_region['bbox'],
                    'overwritten_value': hw_region['text'],
                    'confidence': 0.70,
                    'action': 'extract_both_values'
                })
            
            # Check for ink color changes
            if self._detect_ink_color_change(image, hw_region['bbox']):
                overwrites.append({
                    'type': 'ink_change',
                    'page': page_num,
                    'location': hw_region['bbox'],
                    'latest_text': hw_region['text'],
                    'confidence': 0.80,
                    'action': 'prefer_latest_ink'
                })
        
        return overwrites
    
    def _detect_strike_through(self, image: Image.Image, bbox: List[float]) -> bool:
        """Detect if text has a strike-through line."""
        # Simple heuristic: Check for horizontal line in text region
        img_np = np.array(image)
        y_start, x_start, y_end, x_end = bbox
        
        # Extract region
        region = img_np[int(y_start):int(y_end), int(x_start):int(x_end)]
        
        # Look for dark horizontal lines in middle
        if len(region.shape) == 3:
            gray = np.mean(region, axis=2)
        else:
            gray = region
        
        # Check middle rows
        middle_rows = gray[int(gray.shape[0] * 0.3):int(gray.shape[0] * 0.7), :]
        line_strength = np.sum(middle_rows < 100, axis=1)
        
        # Strong horizontal line indicates strike-through
        return np.max(line_strength) > gray.shape[1] * 0.7
    
    def _detect_overlapping_text(self, image: Image.Image, bbox: List[float]) -> bool:
        """Detect if region has overlapping text (high character density)."""
        # Extract text density in region
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        vision_image = vision.Image(content=img_bytes.read())
        response = self.vision_client.text_detection(image=vision_image)
        
        text_count = len(response.text_annotations)
        region_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        
        # High density = overlapping text
        density = text_count / (region_area + 1e-6)
        
        return density > 0.01  # Threshold for high density
    
    def _detect_ink_color_change(self, image: Image.Image, bbox: List[float]) -> bool:
        """Detect if region has multiple ink colors (indicating multiple writes)."""
        img_np = np.array(image)
        y_start, x_start, y_end, x_end = bbox
        
        region = img_np[int(y_start):int(y_end), int(x_start):int(x_end)]
        
        if len(region.shape) < 3:
            return False
        
        # Count distinct colors in region
        reshaped = region.reshape(-1, region.shape[2])
        unique_colors = np.unique(reshaped, axis=0)
        
        # Multiple ink colors indicate multiple writes
        return len(unique_colors) > 50
    
    def _is_handwritten(self, word) -> bool:
        """Heuristic to determine if a word is handwritten."""
        # In Vision API, handwritten text often has lower confidence and irregular spacing
        # This is a simplified check
        return True  # TODO: Implement proper handwriting classifier
    
    # ========================================================================
    # STEP 4: PAGE-LEVEL CONTENT ANALYSIS
    # ========================================================================
    
    def _analyze_page_content(
        self, pdf_pages: List[Image.Image]
    ) -> Tuple[Dict, List[Dict]]:
        """
        Analyze each page to identify content types (text, table, image, form fields).
        Generate routing instructions for each segment.
        """
        content_summary = {
            "total_segments": 0,
            "text_percentage": 0.0,
            "table_percentage": 0.0,
            "image_percentage": 0.0,
            "form_field_percentage": 0.0,
            "dominant_content_type": "text"
        }
        
        extraction_routing = []
        
        for page_num, page_image in enumerate(pdf_pages, 1):
            page_composition = self._analyze_single_page_content(page_image, page_num)
            
            # Accumulate statistics
            content_summary["text_percentage"] += page_composition.text_percentage
            content_summary["table_percentage"] += page_composition.table_percentage
            content_summary["image_percentage"] += page_composition.image_percentage
            content_summary["form_field_percentage"] += page_composition.form_field_percentage
            content_summary["total_segments"] += len(page_composition.segments)
            
            # Add page routing
            extraction_routing.append({
                "page": page_num,
                "dominant_type": page_composition.dominant_type,
                "segments": page_composition.segments
            })
        
        # Normalize percentages
        num_pages = len(pdf_pages)
        if num_pages > 0:
            content_summary["text_percentage"] /= num_pages
            content_summary["table_percentage"] /= num_pages
            content_summary["image_percentage"] /= num_pages
            content_summary["form_field_percentage"] /= num_pages
        
        # Determine overall dominant type
        composition = {
            "text": content_summary["text_percentage"],
            "table": content_summary["table_percentage"],
            "image": content_summary["image_percentage"],
            "form_field": content_summary["form_field_percentage"]
        }
        content_summary["dominant_content_type"] = max(composition, key=composition.get)
        
        return content_summary, extraction_routing
    
    def _analyze_single_page_content(
        self, image: Image.Image, page_num: int
    ) -> PageContentComposition:
        """Analyze a single page for content composition."""
        # Use Document AI layout analysis
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Placeholder: Simulate layout analysis
        segments = self._segment_page_content(image, page_num)
        
        text_pct = sum(s['percentage'] for s in segments if s['type'] == ContentType.TEXT.value) / 100.0
        table_pct = sum(s['percentage'] for s in segments if s['type'] == ContentType.TABLE.value) / 100.0
        image_pct = sum(s['percentage'] for s in segments if s['type'] == ContentType.IMAGE.value) / 100.0
        form_pct = sum(s['percentage'] for s in segments if s['type'] == ContentType.FORM_FIELD.value) / 100.0
        
        composition = {
            ContentType.TEXT.value: text_pct,
            ContentType.TABLE.value: table_pct,
            ContentType.IMAGE.value: image_pct,
            ContentType.FORM_FIELD.value: form_pct
        }
        dominant = max(composition, key=composition.get)
        
        return PageContentComposition(
            page_number=page_num,
            text_percentage=text_pct,
            table_percentage=table_pct,
            image_percentage=image_pct,
            form_field_percentage=form_pct,
            dominant_type=dominant,
            segments=segments
        )
    
    def _segment_page_content(self, image: Image.Image, page_num: int) -> List[Dict]:
        """Segment a page into content regions."""
        # Simplified: Use image analysis to detect tables, images, text regions
        segments = []
        
        # Detect tables
        if self._contains_table(image):
            segments.append({
                'segment_id': f"page{page_num}_seg_table",
                'type': ContentType.TABLE.value,
                'percentage': 25,
                'extraction_method': 'document_ai_table_parser',
                'priority': 'high'
            })
        
        # Detect images
        if self._contains_image(image):
            segments.append({
                'segment_id': f"page{page_num}_seg_image",
                'type': ContentType.IMAGE.value,
                'percentage': 15,
                'extraction_method': 'vision_api',
                'priority': 'medium'
            })
        
        # Default: text
        segments.append({
            'segment_id': f"page{page_num}_seg_text",
            'type': ContentType.TEXT.value,
            'percentage': 60,
            'extraction_method': 'document_ai_ocr',
            'priority': 'high'
        })
        
        return segments
    
    def _contains_table(self, image: Image.Image) -> bool:
        """Simple heuristic: detect tables by looking for grid patterns."""
        # TODO: Implement proper table detection
        return False
    
    def _contains_image(self, image: Image.Image) -> bool:
        """Simple heuristic: detect non-text regions."""
        # TODO: Implement proper image detection
        return False
    
    # ========================================================================
    # STEP 5: METADATA SCHEMA GENERATION
    # ========================================================================
    
    def _generate_metadata_schema(
        self, form_type: FormType, extraction_routing: List[Dict]
    ) -> Dict:
        """
        Generate flexible metadata schema based on form type.
        Provides guidance for LLM-based extraction in later stages.
        """
        base_schema = {
            "schema_version": "1.0",
            "form_type": form_type.value,
            "suggested_fields": [],
            "flexible_fields": ["notes", "special_instructions", "handwritten_annotations"]
        }
        
        # Form-type-specific fields
        form_specific = {
            FormType.DIGITAL_INVOICE: {
                "suggested_fields": [
                    {"name": "invoice_number", "type": "string", "required": True},
                    {"name": "invoice_date", "type": "date", "required": True},
                    {"name": "vendor_name", "type": "string", "required": True},
                    {"name": "total_amount", "type": "float", "required": True},
                    {"name": "line_items", "type": "array", "required": True},
                    {"name": "payment_terms", "type": "string", "required": False}
                ]
            },
            FormType.MANUAL_INVOICE: {
                "suggested_fields": [
                    {"name": "invoice_number", "type": "string", "required": True},
                    {"name": "invoice_date", "type": "date", "required": True},
                    {"name": "vendor_name", "type": "string", "required": True},
                    {"name": "total_amount", "type": "float", "required": True},
                    {"name": "line_items", "type": "array", "required": True},
                    {"name": "handwritten_notes", "type": "string", "required": False}
                ]
            },
            FormType.PURCHASE_ORDER: {
                "suggested_fields": [
                    {"name": "po_number", "type": "string", "required": True},
                    {"name": "order_date", "type": "date", "required": True},
                    {"name": "vendor", "type": "string", "required": True},
                    {"name": "line_items", "type": "array", "required": True},
                    {"name": "total_amount", "type": "float", "required": True},
                    {"name": "delivery_date", "type": "date", "required": False}
                ]
            },
            FormType.CONSENT_FORM: {
                "suggested_fields": [
                    {"name": "form_title", "type": "string", "required": True},
                    {"name": "consent_type", "type": "string", "required": True},
                    {"name": "signature", "type": "string", "required": True},
                    {"name": "date_signed", "type": "date", "required": True},
                    {"name": "witness_name", "type": "string", "required": False}
                ]
            },
            FormType.VENDOR_DELIVERY_FORM: {
                "suggested_fields": [
                    {"name": "tracking_number", "type": "string", "required": True},
                    {"name": "delivery_date", "type": "date", "required": True},
                    {"name": "signature", "type": "string", "required": True},
                    {"name": "recipient_name", "type": "string", "required": True},
                    {"name": "items_count", "type": "integer", "required": False}
                ]
            },
            FormType.FEEDBACK_FORM: {
                "suggested_fields": [
                    {"name": "rating", "type": "integer", "required": True},
                    {"name": "feedback_text", "type": "string", "required": True},
                    {"name": "date_submitted", "type": "date", "required": False},
                    {"name": "respondent_name", "type": "string", "required": False}
                ]
            }
        }
        
        if form_type in form_specific:
            base_schema["suggested_fields"] = form_specific[form_type]["suggested_fields"]
        
        return base_schema
    
    # ========================================================================
    # STEP 6: ROUTING INSTRUCTIONS
    # ========================================================================
    
    def _generate_routing_instructions(
        self,
        form_type: FormType,
        form_confidence: float,
        quality_metrics: QualityMetrics,
        handwriting_analysis: HandwritingAnalysis
    ) -> Dict:
        """
        Generate next-stage instructions for extraction pipeline.
        """
        quality_gate = quality_metrics.overall_score >= self.quality_thresholds['acceptable']
        manual_review = (
            form_type == FormType.UNKNOWN or
            form_confidence < 0.80 or
            not quality_gate or
            len(handwriting_analysis.overwrites) > 0
        )
        
        return {
            "extraction_strategy": "multi_method_routing",
            "llm_routing": "claude" if handwriting_analysis.has_handwriting else "gemini",
            "confidence_baseline": form_confidence,
            "quality_gate_pass": quality_gate,
            "manual_review_required": manual_review,
            "handwriting_handling_special": handwriting_analysis.has_handwriting,
            "next_stage": "extraction_stage_2"
        }
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _load_pdf(self, pdf_path: str) -> bytes:
        """Load PDF from local or GCS path."""
        if pdf_path.startswith('gs://'):
            # Load from GCS
            bucket_name, path = pdf_path.replace('gs://', '').split('/', 1)
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(path)
            return blob.download_as_bytes()
        else:
            # Load from local
            with open(pdf_path, 'rb') as f:
                return f.read()
    
    def _split_pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF to list of PIL images."""
        import PyPDF2
        from io import BytesIO
        
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        images = []
        
        # For simplicity, convert each page to image
        # In production, use pdf2image library
        for page_num in range(len(pdf_reader.pages)):
            # Placeholder: would use pdf2image
            images.append(Image.new('RGB', (612, 792), color=(255, 255, 255)))
        
        return images
    
    def _calc_bbox_area(self, bbox: List[float], image: Image.Image) -> float:
        """Calculate bounding box area as percentage of image."""
        y_start, x_start, y_end, x_end = bbox
        bbox_area = (x_end - x_start) * (y_end - y_start)
        image_area = image.width * image.height
        return (bbox_area / image_area) if image_area > 0 else 0.0
    
    def _vertices_to_bbox(self, vertices) -> List[float]:
        """Convert Vision API vertices to [y_start, x_start, y_end, x_end]."""
        if not vertices:
            return [0, 0, 0, 0]
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        return [min(y_coords), min(x_coords), max(y_coords), max(x_coords)]

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    project_id = "your-gcp-project-id"
    classifier = DocumentClassifier(project_id)
    
    # Classify a document
    result = classifier.classify_document(
        pdf_path="gs://pdf-bucket/invoices/sample_invoice.pdf",
        document_id="doc_12345"
    )
    
    # Output results
    print(json.dumps(asdict(result), indent=2))

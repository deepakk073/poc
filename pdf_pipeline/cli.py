"""Command-line interface (CLI) entrypoint for running the PDF pipeline."""

import argparse
import json
import logging
import sys

from pdf_pipeline.config.settings import settings
from pdf_pipeline.services.orchestrator import PDFProcessingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pdf_pipeline.cli")


def main():
    parser = argparse.ArgumentParser(
        description="Vertex AI Gemini Intelligent PDF Classification & Extraction Pipeline"
    )
    parser.add_argument("pdf_source", help="Local PDF file path or Google Cloud Storage URI (gs://...)")
    parser.add_argument("--doc-id", dest="doc_id", default=None, help="Optional custom document ID")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Batch size in pages")
    parser.add_argument("--overlap-size", dest="overlap_size", type=int, default=None, help="Overlap in pages")
    parser.add_argument("--output-json", dest="output_json", default=None, help="Path to write final summary JSON")

    args = parser.parse_args()

    if args.batch_size:
        settings.batch_size = args.batch_size
    if args.overlap_size:
        settings.overlap_size = args.overlap_size

    orchestrator = PDFProcessingOrchestrator(pipeline_settings=settings)
    try:
        result = orchestrator.process_pdf(args.pdf_source, custom_document_id=args.doc_id)
        summary = result.to_dict()
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 60)
        print(json.dumps(summary, indent=2))

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nExecution summary written to {args.output_json}")

        if not result.validation_report.is_valid:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()

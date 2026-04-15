from __future__ import annotations

r"""Generate inspectable OCR smoke-test assets for the workflow-ingest pipeline.

This script writes:

- two PNG page images with clear text blocks
- one combined two-page PDF built from those images

Output defaults to:
    tests/.tmp_workflow_ingest_ocr/generated_smoke_assets

Example:
    .venv\Scripts\python.exe scripts/generate_workflow_ingest_ocr_smoke_assets.py
"""

import argparse
from pathlib import Path

from kg_doc_parser.workflow_ingest.smoke_assets import generate_ocr_smoke_assets


DEFAULT_OUTPUT_DIR = Path("tests") / ".tmp_workflow_ingest_ocr" / "generated_smoke_assets"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG and PDF smoke assets should be written.",
    )
    args = parser.parse_args(argv)
    outputs = generate_ocr_smoke_assets(args.output_dir)
    print(f"output_dir: {args.output_dir}")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print("ready: OCR smoke assets generated for manual workflow-ingest testing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

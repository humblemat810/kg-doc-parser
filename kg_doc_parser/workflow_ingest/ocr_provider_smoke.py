"""Produce a sanitized OCR provider capability report."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from pydantic import ValidationError

from .ocr_pipeline import _run_live_ocr_page
from .ocr_providers import list_ocr_provider_capabilities
from .ocr_quality import OCRGroundTruth, evaluate_ocr_ground_truth
from .providers import WorkflowProviderSettings
from .serialization import JsonObject, JsonValue, json_safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ocr-provider-smoke"))
    parser.add_argument("--live", action="store_true", help="run one real OCR request")
    parser.add_argument("--image", type=Path, help="one image required with --live")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="independent OCRGroundTruth JSON to evaluate against the live response",
    )
    parser.add_argument("--page-number", type=int, default=1)
    args = parser.parse_args()
    image_path: Path | None = None
    if args.ground_truth is not None and not args.live:
        parser.error("--ground-truth requires --live")
    if args.live:
        image_path = args.image
        if image_path is None:
            parser.error("--image is required with --live")
        elif not image_path.is_file():
            parser.error(f"image file does not exist: {image_path}")
    if args.page_number < 1:
        parser.error("--page-number must be positive")
    if args.live and args.ground_truth is not None and not args.ground_truth.is_file():
        parser.error(f"ground-truth file does not exist: {args.ground_truth}")
    ground_truth = None
    if args.live and args.ground_truth is not None:
        try:
            ground_truth = OCRGroundTruth.model_validate_json(
                args.ground_truth.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            parser.error(f"invalid ground-truth file: {exc}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provider_records: list[JsonValue] = [
        {
            "provider": item.provider,
            "engine": item.engine,
            "vision": item.vision,
            "local": item.local,
            "coordinates": item.coordinates,
            "structured_output": item.structured_output,
            "optional_dependency": item.optional_dependency,
            "platforms": list(item.platforms),
        }
        for item in list_ocr_provider_capabilities()
    ]
    report: JsonObject = {
        "created_at": time.time(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "live_requested": args.live,
        "live_status": "not_requested",
        "providers": provider_records,
    }
    if args.live:
        assert image_path is not None
        live_result: JsonObject = {
            "provider": "configured",
            "image": str(image_path),
            "page_number": args.page_number,
            "credentials_recorded": False,
        }
        try:
            response = _run_live_ocr_page(
                image_path,
                args.page_number,
                WorkflowProviderSettings.from_env(),
            )
        except Exception as exc:  # noqa: BLE001 - smoke reports provider failures without a traceback
            live_result.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            report["live_status"] = "failed"
        else:
            live_result.update(
                {
                    "status": "succeeded",
                    "text_clusters": len(response.OCR_text_clusters),
                    "non_text_objects": len(response.non_text_objects),
                    "empty_page": response.is_empty_page,
                }
            )
            report["live_status"] = "succeeded"
            if ground_truth is not None:
                quality_value = json_safe(
                    evaluate_ocr_ground_truth(response, ground_truth).model_dump(
                        mode="json"
                    )
                )
                if not isinstance(quality_value, dict):
                    raise TypeError("OCR ground-truth report must serialize as an object")
                quality_path = args.output_dir / "quality-report.json"
                quality_path.write_text(
                    json.dumps(quality_value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                live_result["quality_report"] = str(quality_path)
        (args.output_dir / "live-result.json").write_text(
            json.dumps(live_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    path = args.output_dir / "capability-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

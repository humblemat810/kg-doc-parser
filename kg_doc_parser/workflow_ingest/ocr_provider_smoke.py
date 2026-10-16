"""Produce a sanitized OCR provider capability report."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from .ocr_pipeline import _run_live_ocr_page
from .ocr_providers import list_ocr_provider_capabilities
from .providers import WorkflowProviderSettings
from .serialization import JsonObject, JsonValue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ocr-provider-smoke"))
    parser.add_argument("--live", action="store_true", help="run one real OCR request")
    parser.add_argument("--image", type=Path, help="one image required with --live")
    parser.add_argument("--page-number", type=int, default=1)
    args = parser.parse_args()
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
        if args.image is None:
            parser.error("--image is required with --live")
        live_result: JsonObject = {
            "provider": "configured",
            "image": str(args.image),
            "page_number": args.page_number,
            "credentials_recorded": False,
        }
        try:
            response = _run_live_ocr_page(
                args.image,
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

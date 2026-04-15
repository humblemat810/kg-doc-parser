from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_doc_parser.workflow_ingest import DemoHarnessConfig, run_demo_harness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the workflow ingest demo harness.")
    parser.add_argument("--output-dir", default="logs/workflow_ingest_demo", help="Artifact output directory.")
    parser.add_argument("--document-id", default="demo-doc")
    parser.add_argument("--title", default="Workflow Ingest Demo")
    parser.add_argument("--text", default="Alpha clause\nBeta clause\nGamma clause")
    parser.add_argument(
        "--parser-mode",
        choices=["fake_layered", "legacy_cached"],
        default="fake_layered",
    )
    parser.add_argument(
        "--server-mode",
        choices=["testclient", "subprocess_http", "external_http"],
        default="testclient",
    )
    parser.add_argument(
        "--external-base-url",
        default=None,
        help="Base URL for a live external Kogwistar server when --server-mode external_http.",
    )
    parser.add_argument(
        "--disable-sys-monitoring",
        action="store_true",
        help="Disable optional sys.monitoring probe hooks.",
    )
    args = parser.parse_args()

    artifacts = run_demo_harness(
        DemoHarnessConfig(
            output_dir=Path(args.output_dir),
            document_id=args.document_id,
            title=args.title,
            text=args.text,
            parser_mode=args.parser_mode,
            server_mode=args.server_mode,
            external_base_url=args.external_base_url,
            enable_sys_monitoring=not args.disable_sys_monitoring,
        )
    )
    payload = {
        "status": artifacts.status,
        "run_id": artifacts.run_id,
        "probe_path": str(artifacts.probe_path),
        "summary_path": str(artifacts.summary_path),
        "cache_dir": str(artifacts.cache_dir),
        "canonical_write_confirmed": artifacts.canonical_write_confirmed,
        "persistence_mode": artifacts.persistence_mode,
        "kg_authority": artifacts.kg_authority,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

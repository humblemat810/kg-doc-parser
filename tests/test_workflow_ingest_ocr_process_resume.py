"""Fresh-process OCR and root-workflow checkpoint resume tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ocr_workflow_support import (
    draw_test_image as _draw_test_image,
)
from ocr_workflow_support import (
    scratch_path as _scratch,
)

from kg_doc_parser.models import SplitPage

pytestmark = [pytest.mark.workflow]

@pytest.mark.ci
def test_ocr_page_resume_reopens_state_in_a_fresh_process() -> None:
    """A fresh process reuses accepted pages and processes only the gap."""
    scratch = _scratch("ocr_page_fresh_process_resume")
    for page_number, text in ((1, "fresh page one"), (2, "fresh page two")):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[text],
        )
    worker = Path(__file__).with_name("_ocr_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            str(repo_root.parent / "kogwistar"),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )
    interrupted = subprocess.run(
        [sys.executable, str(worker), str(scratch), "interrupt"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
    )
    assert interrupted.returncode != 0
    assert "fresh-process interruption" in interrupted.stderr

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((scratch / "resume-result.json").read_text(encoding="utf-8"))
    assert result == {
        "pages": 2,
        "completed": [1, 2],
        "reused": [1],
        "run_id": "ocr|fresh-process-document",
        "calls": [2],
    }


@pytest.mark.ci
def test_ocr_root_resume_reopens_same_lineage_in_a_fresh_process() -> None:
    """A fresh process retries only the failed page and accepts one root result."""
    scratch = _scratch("ocr_root_fresh_process_resume")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"root page {page_number}"],
        )
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )

    interrupted = subprocess.run(
        [sys.executable, str(worker), str(scratch), "interrupt"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert interrupted.returncode == 17, interrupted.stderr
    interruption = json.loads(
        (scratch / "interrupt-result.json").read_text(encoding="utf-8")
    )
    assert interruption["calls"] == [1, 2], interruption

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((scratch / "resume-result.json").read_text(encoding="utf-8"))
    assert result == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [2],
        "has_grouping_tree": True,
    }
    lineage = json.loads((scratch / "resume-lineage.json").read_text(encoding="utf-8"))
    lineage_rows = lineage["lineage"]
    assert len({row["id"] for row in lineage_rows}) == len(lineage_rows)
    execution_root_id = "ocr|fresh-root-document|recovery-1"
    expected_stage_run_ids = {
        f"{execution_root_id}|ocr-pages|recovery-1",
        f"{execution_root_id}|grouping|recovery-1",
        f"{execution_root_id}|semantic|recovery-1",
    }
    actual_stage_run_ids = {
        row["child_run_id"]
        for row in lineage_rows
        if row["parent_run_id"] == execution_root_id
    }
    assert actual_stage_run_ids == expected_stage_run_ids
    assert set(lineage["workflow_step_run_ids"]) == {
        execution_root_id,
        *expected_stage_run_ids,
    }
    assert {
        "kg_doc_parser.ocr_pages.v1",
        "kg_doc_parser.ocr_grouping.v1",
        "kg_doc_parser.ingest.v1",
    }.issubset({row["workflow_id"] for row in lineage_rows})
    assert lineage["graph_node_ids"]
    assert set(lineage["graph_node_ids"]) == set(lineage["persisted_graph_node_ids"])


@pytest.mark.ci
def test_ocr_root_resume_replays_checkpointed_page_when_artifact_is_missing() -> None:
    """A runtime checkpoint cannot make a missing page artifact authoritative."""
    scratch = _scratch("ocr_root_checkpoint_without_artifact")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"checkpoint page {page_number}"],
        )
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            str(repo_root.parent / "kogwistar"),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )

    interrupted = subprocess.run(
        [sys.executable, str(worker), str(scratch), "interrupt"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert interrupted.returncode == 17, interrupted.stderr
    accepted_page = (
        scratch
        / "artifacts"
        / "legacy_split_pages"
        / "fresh-root-document"
        / "page_1.json"
    )
    assert accepted_page.is_file()
    accepted_page.unlink()

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume-missing-artifact"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(
        (scratch / "resume-missing-artifact-result.json").read_text(encoding="utf-8")
    )
    assert result == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [1, 2],
        "has_grouping_tree": True,
    }


@pytest.mark.ci
def test_ocr_root_resume_replays_checkpointed_page_when_artifact_digest_is_tampered() -> (
    None
):
    """A valid-looking edited artifact cannot supersede its accepted digest."""
    scratch = _scratch("ocr_root_checkpoint_with_tampered_artifact")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"checkpoint page {page_number}"],
        )
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            str(repo_root.parent / "kogwistar"),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )

    interrupted = subprocess.run(
        [sys.executable, str(worker), str(scratch), "interrupt"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert interrupted.returncode == 17, interrupted.stderr
    accepted_page = (
        scratch
        / "artifacts"
        / "legacy_split_pages"
        / "fresh-root-document"
        / "page_1.json"
    )
    payload = json.loads(accepted_page.read_text(encoding="utf-8"))
    clusters = payload["OCR_text_clusters"]
    assert isinstance(clusters, list) and clusters
    clusters[0]["text"] = "tampered but structurally valid text"
    SplitPage.model_validate(payload)
    accepted_page.write_text(json.dumps(payload), encoding="utf-8")

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume-missing-artifact"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = json.loads(
        (scratch / "resume-missing-artifact-result.json").read_text(encoding="utf-8")
    )
    assert result == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [1, 2],
        "has_grouping_tree": True,
    }
    accepted_after_resume = json.loads(accepted_page.read_text(encoding="utf-8"))
    assert accepted_after_resume["OCR_text_clusters"][0]["text"] == ("root page 1")

"""Typed per-page OCR processing and durable acceptance boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import SplitPage, SplitPageMeta
from .adapters import OCRPageJSON, _validated_ocr_page_json
from .ocr_artifacts import read_json_object as _read_json_object
from .ocr_generators import _provider_endpoint_configuration_identity
from .ocr_page_runner import _retry_feedback_from_attempts, _safe_ocr_retry_feedback
from .ocr_pipeline_contracts import (
    OCRRunner,
)
from .ocr_quality_policy import (
    OCRQualityPolicy,
)
from .ocr_quality_policy import (
    OCRQualityPolicyRejection as _OCRQualityPolicyRejection,
)
from .ocr_quality_policy import (
    evaluate_ocr_quality as _evaluate_ocr_quality,
)
from .ocr_state_store import OCRWorkflowStateStore
from .probe import WorkflowProbe, emit_probe_event
from .providers import WorkflowProviderSettings

_LOGGER = logging.getLogger(__name__)
def _ocr_configuration_fingerprint(
    settings: WorkflowProviderSettings,
    *,
    candidate_models: Sequence[str] = (),
    quality_policy: OCRQualityPolicy | None = None,
) -> str:
    """Return the non-secret identity of an OCR execution configuration."""
    config = {
        "provider_configuration": _provider_endpoint_configuration_identity(
            settings.ocr
        ),
        "candidate_models": list(candidate_models),
        "quality_policy": (quality_policy or OCRQualityPolicy()).model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _ocr_page_metadata(
    settings: WorkflowProviderSettings,
    *,
    candidate_models: Sequence[str] = (),
    source_image_sha256: str | None = None,
    quality_policy: OCRQualityPolicy | None = None,
) -> dict[str, str]:
    """Build stable, non-secret provenance for one accepted OCR page."""
    metadata = {
        "ocr_provider": settings.ocr.provider,
        "ocr_model_name": settings.ocr.model,
        "ocr_pdf_text_mode": settings.ocr.pdf_text_mode,
        "ocr_configuration_fingerprint": _ocr_configuration_fingerprint(
            settings,
            candidate_models=candidate_models,
            quality_policy=quality_policy,
        ),
    }
    if source_image_sha256 is not None:
        metadata["source_image_sha256"] = source_image_sha256
    return metadata
@dataclass(slots=True)
class OCRPageProcessingContext:
    """Immutable per-run OCR processing context.

    This groups the values that every page in a run needs so the per-page loop
    can stay focused on the page-specific state rather than carrying a very
    long parameter list.
    """

    document_id: str
    title: str
    source_kind: str
    total_pages: int
    input_fingerprint: str
    candidate_models: list[str]
    provider_settings: WorkflowProviderSettings
    ocr_page_runner: OCRRunner
    state_store: OCRWorkflowStateStore
    probe: WorkflowProbe | None
    configuration_fingerprints: tuple[str, ...]
    retry_feedback_by_page: dict[int, tuple[str, ...]] = field(default_factory=dict)
    render_configuration_fingerprint: str = "render-v1"
    quality_policy: OCRQualityPolicy = field(default_factory=OCRQualityPolicy)


class OCRPageOutcome(BaseModel):
    """Validated result of processing one page, including reuse and failure."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    status: Literal["completed", "reused", "failed"]
    page: OCRPageJSON | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> OCRPageOutcome:
        if self.status == "failed":
            if self.page is not None or not self.error:
                raise ValueError("failed page outcomes require an error and no page")
        elif self.page is None or self.error is not None:
            raise ValueError("accepted page outcomes require a page and no error")
        return self


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "." * width
    filled = min(width, max(0, round((done / total) * width)))
    return ("#" * filled) + ("." * (width - filled))


def _emit_ocr_event(probe: WorkflowProbe | None, kind: str, /, **payload: object) -> None:
    emit_probe_event(probe, kind, **payload)


def _process_ocr_page(
    *,
    context: OCRPageProcessingContext,
    done_count: int,
    page_number: int,
    image_path: Path,
    image_sha256: str,
    page_json_path: Path,
    page_image_path: Path,
) -> OCRPageOutcome:
    """Process one page image, either by resuming or by running OCR models."""

    state_store = context.state_store
    document_id = context.document_id
    title = context.title
    source_kind = context.source_kind
    total_pages = context.total_pages
    probe = context.probe

    state_store.ensure_document(
        document_id=document_id,
        title=title,
        source_kind=source_kind,
        total_pages=total_pages,
        input_fingerprint=context.input_fingerprint,
    )
    if not state_store.should_skip_page(
        document_id=document_id,
        page_number=page_number,
        stage="render",
        content_hash=image_sha256,
        artifact_path=image_path,
        config_fingerprint=context.render_configuration_fingerprint,
    ):
        _emit_ocr_event(
            probe,
            "ocr.render_started",
            document_id=document_id,
            page_number=page_number,
            image_path=str(image_path),
        )
        render_attempt = state_store.record_attempt(
            document_id=document_id,
            page_number=page_number,
            stage="render",
            content_hash=image_sha256,
            model_name="local-materialize",
            artifact_path=image_path,
            config_fingerprint=context.render_configuration_fingerprint,
        )
        state_store.record_page_completed(
            document_id=document_id,
            page_number=page_number,
            stage="render",
            content_hash=image_sha256,
            artifact_path=image_path,
            model_name="local-materialize",
            attempt_index=render_attempt,
            config_fingerprint=context.render_configuration_fingerprint,
        )
        _emit_ocr_event(
            probe,
            "ocr.render_completed",
            document_id=document_id,
            page_number=page_number,
            image_path=str(image_path),
        )

    # Reconcile an artifact-first commit before checking the SQLite completion
    # row. A fresh recovery child starts with cursor zero, so its checkpoint
    # prefix reconciliation cannot visit page one; the artifact itself must
    # still be allowed to restore the accepted result.
    state_store.reconcile_orphan_artifact(
        document_id=document_id,
        page_number=page_number,
        content_hash=image_sha256,
        artifact_path=page_json_path,
        config_fingerprints=context.configuration_fingerprints,
    )

    # Resume boundary: if the OCR page artifact already matches the image,
    # reuse the serialized page instead of calling the model again.
    if state_store.should_skip_page(
        document_id=document_id,
        page_number=page_number,
        stage="ocr",
        content_hash=image_sha256,
        artifact_path=page_json_path,
        config_fingerprint=context.configuration_fingerprints,
    ):
        persisted_page = _validated_ocr_page_json(_read_json_object(page_json_path))
        split_page = SplitPage.model_validate(persisted_page)
        raw_page = split_page.dump_supercede_parse()
        raw_page["meaningful_ordering"] = list(split_page.meaningful_ordering)
        persisted_metadata = persisted_page.get("metadata")
        raw_page["metadata"] = (
            persisted_metadata
            if persisted_metadata is not None
            else split_page.metadata.model_dump(mode="json")
        )
        hydrated_page = _validated_ocr_page_json(
            raw_page, default_page_number=page_number
        )
        _emit_ocr_event(
            probe,
            "ocr.page_reused",
            document_id=document_id,
            page_number=page_number,
            page_json_path=str(page_json_path),
        )
        _LOGGER.info(
            "ocr resume | %s/%s | %s | reused page %s",
            done_count,
            total_pages,
            _progress_bar(done_count, total_pages),
            page_number,
        )
        return OCRPageOutcome(page_number=page_number, status="reused", page=hydrated_page)

    last_exc: Exception | None = None
    _emit_ocr_event(
        probe,
        "ocr.page_started",
        document_id=document_id,
        page_number=page_number,
        page_json_path=str(page_json_path),
    )
    # Try candidate OCR models one-by-one for this page; the first grounded
    # success wins and gets persisted as the canonical page artifact.
    for candidate_model in context.candidate_models:
        candidate_settings: WorkflowProviderSettings = context.provider_settings.model_copy(
            update={
                "ocr": context.provider_settings.ocr.model_copy(update={"model": candidate_model}),
            }
        )
        _emit_ocr_event(
            probe,
            "ocr.candidate_started",
            document_id=document_id,
            page_number=page_number,
            model_name=candidate_model,
        )
        attempt_index = state_store.record_attempt(
            document_id=document_id,
            page_number=page_number,
            stage="ocr",
            content_hash=image_sha256,
            model_name=candidate_model,
            artifact_path=page_json_path,
            config_fingerprint=_ocr_configuration_fingerprint(
                candidate_settings,
                candidate_models=context.candidate_models,
                quality_policy=context.quality_policy,
            ),
        )
        context.retry_feedback_by_page[page_number] = _retry_feedback_from_attempts(
            state_store.list_model_attempts(
                document_id=document_id,
                page_number=page_number,
                stage="ocr",
            )
        )
        try:
            response = context.ocr_page_runner(image_path, page_number, candidate_settings)
            quality_diagnostics = _evaluate_ocr_quality(
                response,
                context.quality_policy,
                source_image_path=image_path,
            )
            if any(item.severity == "error" for item in quality_diagnostics):
                raise _OCRQualityPolicyRejection(
                    page_number,
                    tuple(
                        item.code
                        for item in quality_diagnostics
                        if item.severity == "error"
                    ),
                )
            split_page: SplitPage = SplitPage(
                pdf_page_num=page_number,
                metadata=SplitPageMeta(
                    ocr_model_name=candidate_model,
                    ocr_datetime=0.0,
                    ocr_json_version="workflow_ingest_v1",
                ),
                **response.model_dump(dump_format="python"),
            )
            shutil.copy2(image_path, page_image_path)
            page_payload = split_page.dump_raw(dump_format="json")
            page_payload["metadata"] = {
                **split_page.metadata.model_dump(mode="json"),
                **_ocr_page_metadata(
                    candidate_settings,
                    candidate_models=context.candidate_models,
                    source_image_sha256=image_sha256,
                    quality_policy=context.quality_policy,
                ),
                "ocr_quality_status": "flagged" if quality_diagnostics else "accepted",
                "ocr_quality_diagnostics": [
                    item.model_dump(mode="json") for item in quality_diagnostics
                ],
            }
            # Validate before publication; the SQLite row is committed only
            # after the complete artifact is atomically visible. Publication
            # and acceptance are serialized so duplicate workers cannot
            # replace an already accepted page.
            SplitPage.model_validate(page_payload)
            published = state_store.publish_ocr_page(
                document_id=document_id,
                page_number=page_number,
                content_hash=image_sha256,
                artifact_path=page_json_path,
                payload=page_payload,
                model_name=candidate_model,
                attempt_index=attempt_index,
                config_fingerprint=_ocr_configuration_fingerprint(
                    candidate_settings,
                    candidate_models=context.candidate_models,
                    quality_policy=context.quality_policy,
                ),
            )
            if not published:
                # Another worker accepted the same page while this provider
                # call was running. Rehydrate the winner and report reuse;
                # never append this worker's unaccepted payload.
                if not state_store.should_skip_page(
                    document_id=document_id,
                    page_number=page_number,
                    stage="ocr",
                    content_hash=image_sha256,
                    artifact_path=page_json_path,
                    config_fingerprint=context.configuration_fingerprints,
                ):
                    raise RuntimeError("OCR page acceptance lost without a reusable winner")
                persisted_page = json.loads(page_json_path.read_text(encoding="utf-8"))
                hydrated_page = _validated_ocr_page_json(
                    persisted_page, default_page_number=page_number
                )
                return OCRPageOutcome(
                    page_number=page_number, status="reused", page=hydrated_page
                )
            accepted_page = _validated_ocr_page_json(
                page_payload, default_page_number=page_number
            )
            _emit_ocr_event(
                probe,
                "ocr.candidate_completed",
                document_id=document_id,
                page_number=page_number,
                model_name=candidate_model,
                page_json_path=str(page_json_path),
            )
            _LOGGER.info(
                "ocr progress | %s/%s | %s | completed page %s | model=%s",
                done_count,
                total_pages,
                _progress_bar(done_count, total_pages),
                page_number,
                candidate_model,
            )
            return OCRPageOutcome(
                page_number=page_number, status="completed", page=accepted_page
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            retry_feedback = _safe_ocr_retry_feedback(exc)
            state_store.record_page_failed(
                document_id=document_id,
                page_number=page_number,
                stage="ocr",
                content_hash=image_sha256,
                config_fingerprint=_ocr_configuration_fingerprint(
                    candidate_settings,
                    candidate_models=context.candidate_models,
                    quality_policy=context.quality_policy,
                ),
                error_message=str(exc),
                model_name=candidate_model,
                retry_feedback=retry_feedback,
                attempt_index=attempt_index,
            )
            _LOGGER.warning(
                "ocr candidate failed | page=%s model=%s attempt=%s error=%s",
                page_number,
                candidate_model,
                attempt_index,
                exc,
            )
            _emit_ocr_event(
                probe,
                "ocr.candidate_failed",
                document_id=document_id,
                page_number=page_number,
                model_name=candidate_model,
                attempt_index=attempt_index,
                error=str(exc),
            )
    _emit_ocr_event(
        probe,
        "ocr.page_failed",
        document_id=document_id,
        page_number=page_number,
        candidate_models=list(context.candidate_models),
    )
    _LOGGER.warning(
        "ocr progress | %s/%s | %s | failed page %s after %s candidate(s)",
        done_count,
        total_pages,
        _progress_bar(done_count, total_pages),
        page_number,
        len(context.candidate_models),
    )
    if last_exc is not None:
        _LOGGER.warning("last ocr error for page %s: %s", page_number, last_exc)
    return OCRPageOutcome(
        page_number=page_number,
        status="failed",
        error=str(last_exc) if last_exc is not None else "all OCR candidates failed",
    )



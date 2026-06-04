from __future__ import annotations

import json
import re
from typing import Any, Callable

from kg_doc_parser.llm_structured_output import build_structured_output_runnable

from .models import (
    CurrentLayerResult,
    CurrentLayerReview,
    LLMCurrentLayerResult,
    LLMCurrentLayerReview,
    LayerChildCandidate,
    LayerReasoningEntry,
)
from .providers import WorkflowProviderSettings, build_chat_model_for_role
from .semantics import HydratedTextPointer


def _trim_text(value: Any, *, max_chars: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(field_mode="backend", dump_format="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return {str(key): _dump_model(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dump_model(item) for item in value]
    return value


def _summarize_for_prompt(
    value: Any,
    *,
    max_depth: int = 3,
    max_items: int = 6,
    max_string: int = 400,
) -> Any:
    if hasattr(value, "model_dump"):
        value = _dump_model(value)
    if isinstance(value, dict):
        if max_depth <= 0:
            return {"summary": f"{len(value)} keys omitted"}
        items = list(value.items())
        summary: dict[str, Any] = {}
        for key, item in items[:max_items]:
            summary[str(key)] = _summarize_for_prompt(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
        if len(items) > max_items:
            summary["..."] = f"{len(items) - max_items} more keys omitted"
        return summary
    if isinstance(value, list):
        if max_depth <= 0:
            return [f"{len(value)} items omitted"]
        items = [
            _summarize_for_prompt(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"... {len(value) - max_items} more items omitted")
        return items
    if isinstance(value, str):
        return _trim_text(value, max_chars=max_string)
    return value


def _source_map_excerpt(
    parser_source_map: dict[str, dict[str, Any]],
    *,
    max_records: int = 12,
    max_text_chars: int = 1200,
) -> dict[str, dict[str, Any]]:
    excerpt: dict[str, dict[str, Any]] = {}
    for key, record in list(parser_source_map.items())[:max_records]:
        text = str(record.get("text") or "")
        excerpt[str(key)] = {
            "page_number": record.get("page_number"),
            "cluster_number": record.get("cluster_number"),
            "text": _trim_text(text, max_chars=max_text_chars),
        }
    return excerpt


def _pointer_field(pointer: Any, field_name: str) -> Any:
    if isinstance(pointer, dict):
        return pointer.get(field_name)
    return getattr(pointer, field_name, None)


def _pointer_excerpt(pointer: Any, *, parser_source_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
    start_char = _pointer_field(pointer, "start_char")
    end_char = _pointer_field(pointer, "end_char")
    verbatim_text = _pointer_field(pointer, "verbatim_text")
    excerpt = str(verbatim_text or "")
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(str(source_cluster_id))
    if record is not None and isinstance(start_char, int) and isinstance(end_char, int):
        raw_text = str(record.get("text") or "")
        if 0 <= start_char <= end_char < len(raw_text):
            excerpt = raw_text[start_char : end_char + 1]
    return {
        "source_cluster_id": source_cluster_id,
        "start_char": start_char,
        "end_char": end_char,
        "verbatim_text": _trim_text(excerpt or verbatim_text or "", max_chars=400),
    }


def _parent_context_excerpt(
    *,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_ids = list(getattr(current_layer_context, "parent_node_ids", []) or [])
    parent_titles = list(getattr(current_layer_context, "parent_titles", []) or [])
    pointers_by_id = dict(getattr(current_layer_context, "parent_content_pointers_by_id", {}) or {})
    parents: list[dict[str, Any]] = []
    for index, parent_id in enumerate(parent_ids):
        title = parent_titles[index] if index < len(parent_titles) else None
        pointers = list(pointers_by_id.get(parent_id) or [])
        parents.append(
            {
                "parent_node_id": parent_id,
                "title": title,
                "pointer_count": len(pointers),
                "pointer_excerpt": [
                    _pointer_excerpt(pointer, parser_source_map=parser_source_map)
                    for pointer in pointers[:3]
                ],
            }
        )
    return parents


def _proposal_attempt_context(*, parse_session: Any, current_layer_context: Any) -> dict[str, Any]:
    parse_session_dump = _dump_model(parse_session) if parse_session is not None else {}
    attempts = dict(parse_session_dump.get("layer_attempts") or {})
    depth_key = str(getattr(current_layer_context, "depth", 0))
    return {
        "attempt_number_for_depth": int(attempts.get(depth_key, 0)) + 1,
        "attempts_by_depth": attempts,
        "strategy_history": list(parse_session_dump.get("strategy_history") or []),
        "fallback_split_strategy": parse_session_dump.get("fallback_split_strategy"),
        "workflow_mode": parse_session_dump.get("mode"),
    }


def _proposal_validation_reason(
    *,
    parsed: Any,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> str | None:
    children = list(getattr(parsed, "children", []) or [])
    if not children:
        if getattr(parsed, "satisfied", None) is True:
            return None
        return "proposal returned no children without satisfied=true"
    current_parent_ids = set(getattr(current_layer_context, "parent_node_ids", []) or [])
    child_parent_ids = {str(getattr(child, "parent_node_id", "") or "") for child in children}
    if current_parent_ids and not child_parent_ids.issubset(current_parent_ids):
        return "proposal referenced parent ids outside the current layer"
    children_by_parent: dict[str, list[Any]] = {}
    for child in children:
        pointers = list(getattr(child, "total_content_pointers", []) or [])
        if not pointers:
            return "proposal child missing total_content_pointers"
        parent_id = str(getattr(child, "parent_node_id", "") or "")
        if not parent_id:
            return "proposal child missing parent_node_id"
        children_by_parent.setdefault(parent_id, []).append(child)
        for pointer in pointers:
            source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
            start_char = _pointer_field(pointer, "start_char")
            end_char = _pointer_field(pointer, "end_char")
            if not source_cluster_id or source_cluster_id not in parser_source_map:
                return "proposal pointer referenced source outside the supplied source map"
            if not isinstance(start_char, int) or not isinstance(end_char, int):
                return "proposal pointer missing character span integers"
            if start_char < 0 or end_char < start_char:
                return "proposal pointer used an invalid character span"
    parent_titles = dict(
        zip(
            list(getattr(current_layer_context, "parent_node_ids", []) or []),
            list(getattr(current_layer_context, "parent_titles", []) or []),
            strict=False,
        )
    )
    for parent_id, parent_children in children_by_parent.items():
        if len(parent_children) == 1:
            only_child = parent_children[0]
            child_title = str(getattr(only_child, "title", "") or "").strip().lower()
            parent_title = str(parent_titles.get(parent_id) or "").strip().lower()
            if not getattr(only_child, "expandable", False) or child_title == parent_title:
                return "proposal collapsed a parent into a single child without a real breakdown"
    return None


def _annotate_proposal_result(
    result: Any,
    *,
    proposal_source: str,
    proposal_failure_reason: str | None = None,
    provider_child_count: int | None = None,
) -> Any:
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata["proposal_source"] = proposal_source
    if proposal_failure_reason:
        metadata["proposal_failure_reason"] = proposal_failure_reason
    reasoning_history = list(getattr(result, "reasoning_history", []) or [])
    marker_payload: dict[str, Any] = {
        "source": "workflow_layered_proposal",
        "proposal_source": proposal_source,
    }
    if provider_child_count is not None:
        marker_payload["provider_child_count"] = provider_child_count
    if proposal_failure_reason:
        marker_payload["proposal_failure_reason"] = proposal_failure_reason
    reasoning_history.append(LayerReasoningEntry.model_validate(marker_payload))
    payload = result.model_dump()
    payload["metadata"] = metadata
    payload["reasoning_history"] = [entry.model_dump() if hasattr(entry, "model_dump") else entry for entry in reasoning_history]
    return result.__class__.model_validate(payload)


def _structured_invoke(model: Any, schema: Any, messages: list[tuple[str, str]]) -> Any:
    structured = build_structured_output_runnable(model, schema, include_raw=True)
    response = structured.invoke(messages)
    if isinstance(response, dict):
        parsed = response.get("parsed")
        if parsed is not None:
            return parsed
        if response.get("parsing_error") is not None:
            raise ValueError(str(response["parsing_error"]))
    return response


def _fallback_layer_result(
    *,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> Any:
    children: list[Any] = []
    if int(getattr(current_layer_context, "depth", 0)) > 0:
        return CurrentLayerResult(
            children=[],
            satisfied=True,
            reasoning_history=[{"source": "deterministic_depth_stop"}],
            metadata={"fallback": "depth_stop"},
        )
    parent_ids = list(getattr(current_layer_context, "parent_node_ids", []) or [])
    parent_id = parent_ids[0] if parent_ids else "root"
    records = list(parser_source_map.items())
    for index, (source_cluster_id, record) in enumerate(records[:8], start=1):
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        title_match = re.search(r"(?m)^#{1,3}\s+(.+)$", text)
        title = title_match.group(1).strip() if title_match else f"Section {index}"
        excerpt = text
        children.append(
            LayerChildCandidate(
                node_id=f"{parent_id}|section-{index}",
                parent_node_id=parent_id,
                title=title[:160],
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=str(source_cluster_id),
                        start_char=0,
                        end_char=max(len(excerpt) - 1, 0),
                        verbatim_text=excerpt or " ",
                    )
                ],
                expandable=False,
                metadata={"source": "workflow_layered_fallback"},
            )
        )
    return CurrentLayerResult(
        children=children,
        satisfied=True,
        reasoning_history=[{"source": "deterministic_fallback"}],
        metadata={"fallback": "llm_empty_or_unavailable"},
    )


def build_layerwise_llm_callbacks(
    provider_settings: WorkflowProviderSettings,
    *,
    event_sink: Callable[..., None] | None = None,
    fallback_layer_result_fn: Callable[..., Any] | None = None,
    max_depth: int = 2,
    allow_review: bool = True,
) -> dict[str, Any]:
    chat_model = build_chat_model_for_role("parser", provider_settings)
    fallback_builder = fallback_layer_result_fn or _fallback_layer_result

    def _emit(stage: str, **extra: Any) -> None:
        if callable(event_sink):
            event_sink(stage, **extra)

    def _propose_layer_fn(
        *,
        parser_source_map,
        current_layer_context,
        semantic_tree,
        split_strategy,
        parser_input_dict,
        parse_session,
        **kwargs,
    ):
        prompt_payload = {
            "task": "Propose the next semantic layer for the current parent nodes.",
            "output_contract": "Return a CurrentLayerResult only. Do not emit prose.",
            "split_strategy": split_strategy,
            "attempt_context": _proposal_attempt_context(
                parse_session=parse_session,
                current_layer_context=current_layer_context,
            ),
            "current_layer_context": _dump_model(current_layer_context),
            "parent_nodes": _parent_context_excerpt(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map,
            ),
            "semantic_tree_snapshot": _summarize_for_prompt(
                semantic_tree,
                max_depth=4,
                max_items=5,
                max_string=280,
            ),
            "full_document_context": _summarize_for_prompt(
                parser_input_dict,
                max_depth=4,
                max_items=5,
                max_string=280,
            ),
            "source_map_excerpt": _source_map_excerpt(parser_source_map),
            "rules": [
                "Operate on ONE layer only: propose only the immediate children for the current parent nodes.",
                "Do not edit ancestors or descendants and do not break down current children directly.",
                "Children together must preserve the parent meaning collectively.",
                "Each parent should end up with more than one child or no child at all.",
                "If a parent is already atomic, return no children for that parent and use satisfied=true only when the current layer is complete.",
                "Prefer a coarser layerwise breakdown over premature deep flattening so later iterations can refine substructure.",
                "Do not recombine separated verbatim fragments into invented longer text; keep grounding exact and local.",
                "Every pointer must use exact source_cluster_id values plus character spans from the supplied source map.",
                "Set expandable=false for leaf nodes and expandable=true only when later subdivision is still appropriate.",
            ],
        }
        messages = [
            (
                "system",
                "You are revising ONE semantic layer in an iterative document parsing workflow. "
                "Return only structured data matching CurrentLayerResult. "
                "Produce grounded immediate children for the supplied parents and preserve layerwise semantics.",
            ),
            (
                "human",
                json.dumps(prompt_payload, sort_keys=True),
            ),
        ]
        try:
            result = _structured_invoke(chat_model, LLMCurrentLayerResult, messages)
            parsed = (
                result
                if isinstance(result, LLMCurrentLayerResult)
                else LLMCurrentLayerResult.model_validate(
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
            )
            validation_reason = _proposal_validation_reason(
                parsed=parsed,
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map,
            )
            if validation_reason:
                raise ValueError(validation_reason)
            runtime_result = CurrentLayerResult.model_validate(parsed.model_dump())
            annotated = _annotate_proposal_result(
                runtime_result,
                proposal_source="llm",
                provider_child_count=len(runtime_result.children),
            )
            _emit(
                "workflow_layered_proposal_result",
                proposal_source="llm",
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                child_count=len(runtime_result.children),
                satisfied=runtime_result.satisfied,
            )
            return annotated
        except Exception as exc:
            failure_reason = _trim_text(repr(exc), max_chars=500)
            fallback = fallback_builder(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map,
            )
            annotated = _annotate_proposal_result(
                fallback,
                proposal_source="fallback",
                proposal_failure_reason=failure_reason,
                provider_child_count=0,
            )
            _emit(
                "workflow_layered_proposal_result",
                proposal_source="fallback",
                proposal_failure_reason=failure_reason,
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                child_count=len(annotated.children),
                satisfied=annotated.satisfied,
            )
            return annotated

    def _review_layer_fn(
        *,
        current_layer_context,
        current_layer_result,
        split_strategy,
        parser_source_map=None,
        parse_session=None,
        **kwargs,
    ):
        review_payload = {
            "task": "Review the current semantic layer proposal for layerwise correctness and source-grounded coverage.",
            "split_strategy": split_strategy,
            "attempt_context": _proposal_attempt_context(
                parse_session=parse_session,
                current_layer_context=current_layer_context,
            ),
            "current_layer_context": _dump_model(current_layer_context),
            "current_layer_result": _summarize_for_prompt(
                current_layer_result,
                max_depth=4,
                max_items=6,
                max_string=280,
            ),
            "parent_nodes": _parent_context_excerpt(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map or {},
            ),
            "source_map_excerpt": _source_map_excerpt(parser_source_map or {}),
            "review_rules": [
                "Review only the immediate children for this layer.",
                "Children should preserve each parent meaning collectively without inventing missing text.",
                "Coverage gaps, overlap conflicts, and duplicates should be called out explicitly.",
                "Prefer coarser layerwise structure over deep flattening when both are source-grounded.",
                "If the layer is already in a good layerwise sweet spot, satisfied=true is appropriate.",
            ],
        }
        messages = [
            (
                "system",
                "You review ONE semantic layer in an iterative document parsing workflow. "
                "Return only structured data matching CurrentLayerReview.",
            ),
            (
                "human",
                json.dumps(review_payload, sort_keys=True),
            ),
        ]
        try:
            result = _structured_invoke(chat_model, LLMCurrentLayerReview, messages)
            reviewed = (
                result
                if isinstance(result, LLMCurrentLayerReview)
                else LLMCurrentLayerReview.model_validate(
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
            )
            runtime_review = CurrentLayerReview.model_validate(reviewed.model_dump())
            _emit(
                "workflow_layered_review_result",
                review_source="llm",
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                satisfied=runtime_review.satisfied,
                coverage_ok=runtime_review.coverage_ok,
            )
            return runtime_review
        except Exception as exc:
            failure_reason = _trim_text(repr(exc), max_chars=500)
            reviewed = CurrentLayerReview(
                updated_result=current_layer_result,
                coverage_ok=True,
                satisfied=True,
                strategy_used=split_strategy,
                review_notes=["deterministic review fallback after provider failure"],
            )
            _emit(
                "workflow_layered_review_result",
                review_source="fallback",
                review_failure_reason=failure_reason,
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                satisfied=reviewed.satisfied,
                coverage_ok=reviewed.coverage_ok,
            )
            return reviewed

    return {
        "propose_layer_fn": _propose_layer_fn,
        "review_layer_fn": _review_layer_fn,
        "max_depth": max_depth,
        "allow_review": allow_review,
    }

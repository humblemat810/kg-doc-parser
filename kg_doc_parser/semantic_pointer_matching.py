"""Deterministic text matching primitives used by legacy pointer repair."""

from __future__ import annotations

import difflib
import math
import re
from typing import Literal, TypedDict

from rapidfuzz import fuzz
from rapidfuzz.distance import LCSseq


class ExactMatchDiagnostic(TypedDict):
    name: Literal["exact"]
    collapsed: bool


class SimilarityMatchDiagnostic(TypedDict):
    name: Literal["LCSseq.normalized_similarity", "difflib.SequenceMatcher"]
    threshold: float
    collapsed: bool


MatchDiagnostic = ExactMatchDiagnostic | SimilarityMatchDiagnostic


def safe_slice(text: str, start: int, end_inclusive: int) -> str:
    end_exclusive = len(text) if end_inclusive == -1 else end_inclusive + 1
    if start < 0:
        start = 0
    if end_exclusive < 0:
        end_exclusive = 0
    return text[start:end_exclusive]


def all_exact_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Return non-overlapping exact match positions with inclusive ends."""
    if not needle:
        return []
    occurrences: list[tuple[int, int]] = []
    index = 0
    needle_length = len(needle)
    while True:
        index = haystack.find(needle, index)
        if index == -1:
            break
        occurrences.append((index, index + needle_length - 1))
        index += max(1, needle_length)
    return occurrences


def best_occurrence_by_proximity(
    occurrences: list[tuple[int, int]], proposed_start: int
) -> tuple[int, int] | None:
    if not occurrences:
        return None
    return min(occurrences, key=lambda span: abs(span[0] - proposed_start))


def whitespace_collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def soft_exact_positions(
    source_text: str,
    verbatim: str,
    *,
    fuzzy_threshold: float | None = None,
    fuzzy_len_stretch: float = 0.25,
    fuzzy_stride_frac: float = 0.10,
) -> tuple[list[tuple[int, int]], MatchDiagnostic | None]:
    """Find exact, whitespace-normalized, or explicitly enabled fuzzy spans."""
    exact_occurrences = all_exact_occurrences(source_text, verbatim)
    if exact_occurrences:
        return exact_occurrences, {"name": "exact", "collapsed": False}

    normalized_query = whitespace_collapse(verbatim)
    if not normalized_query:
        return [], None

    source_spans: list[tuple[int, int]] = []
    collapsed_chars: list[str] = []
    index = 0
    while index < len(source_text):
        if source_text[index].isspace():
            end = index
            while end < len(source_text) and source_text[end].isspace():
                end += 1
            if not collapsed_chars or collapsed_chars[-1] != " ":
                collapsed_chars.append(" ")
                source_spans.append((index, end - 1))
            index = end
        else:
            collapsed_chars.append(source_text[index])
            source_spans.append((index, index))
            index += 1

    collapsed_text = "".join(collapsed_chars)
    collapsed_occurrences = all_exact_occurrences(collapsed_text, normalized_query)
    if collapsed_occurrences:
        mapped = [
            (source_spans[start][0], source_spans[end][1])
            for start, end in collapsed_occurrences
        ]
        return mapped, {"name": "exact", "collapsed": True}

    if fuzzy_threshold is None:
        return [], None

    def map_back(start: int, end: int) -> tuple[int, int]:
        return source_spans[start][0], source_spans[end][1]

    query_length = len(normalized_query)
    if not query_length or not collapsed_text:
        return [], None

    threshold = float(fuzzy_threshold)
    if threshold <= 1.0:
        threshold *= 100.0

    best_score = -1.0
    best_span: tuple[int, int] | None = None
    min_length = max(1, int(math.floor(query_length * (1.0 - fuzzy_len_stretch))))
    max_length = max(min_length, int(math.ceil(query_length * (1.0 + fuzzy_len_stretch))))
    stride = max(1, int(max(1, round(query_length * fuzzy_stride_frac))))

    for window_length in (min_length, query_length, max_length):
        if window_length > len(collapsed_text):
            continue
        for start in range(0, len(collapsed_text) - window_length + 1, stride):
            end_exclusive = start + window_length
            score = (
                LCSseq.normalized_similarity(
                    normalized_query, collapsed_text[start:end_exclusive]
                )
                * 100.0
            )
            if score > best_score:
                best_score = score
                best_span = (start, end_exclusive - 1)

    if best_span is not None and best_score >= threshold:
        start, end = map_back(*best_span)
        return [(start, end)], {
            "name": "LCSseq.normalized_similarity",
            "threshold": threshold,
            "collapsed": True,
        }

    def locate_span(query: str, text: str) -> tuple[int, int] | None:
        matcher = difflib.SequenceMatcher(None, text.lower(), query.lower())
        match = max(matcher.get_matching_blocks(), key=lambda block: block.size)
        if match.size == 0:
            return None
        return match.a, match.a + match.size - 1

    best_span = locate_span(verbatim, collapsed_text)
    if best_span is not None:
        best_score = fuzz.ratio(
            verbatim, collapsed_text[best_span[0] : best_span[1]]
        )
    if best_span is not None and best_score >= threshold:
        start, end = map_back(*best_span)
        return [(start, end)], {
            "name": "difflib.SequenceMatcher",
            "threshold": threshold,
            "collapsed": True,
        }

    best_span = locate_span(verbatim, source_text)
    if best_span is not None:
        best_score = fuzz.ratio(
            verbatim, collapsed_text[best_span[0] : best_span[1]]
        )
    if best_span is not None and best_score >= threshold:
        return [best_span], {
            "name": "difflib.SequenceMatcher",
            "threshold": threshold,
            "collapsed": False,
        }

    return [], None

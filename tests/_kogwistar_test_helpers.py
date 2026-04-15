from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import pytest


def load_kogwistar_in_memory_backend():
    from kogwistar.engine_core.in_memory_backend import build_in_memory_backend

    return build_in_memory_backend


def load_kogwistar_fake_embedding_function():
    from kg_doc_parser.workflow_ingest.providers import EmbeddingProviderConfig, build_embedding_function

    return build_embedding_function(
        EmbeddingProviderConfig(provider="fake", model="kg-doc-parser-workflow-embedding-v1", dimension=2)
    )


class _ProgressEmbeddingFunction:
    def __init__(self, inner, *, label: str, every: int = 10) -> None:
        self._inner = inner
        self._label = label
        self._every = max(1, every)
        self._calls = 0
        self._logger = logging.getLogger(__name__)
        self._bar = None
        self._total_hint = 250

    def name(self) -> str:
        if hasattr(self._inner, "name"):
            return self._inner.name()
        return self._label

    def _ensure_bar(self):
        if self._bar is not None:
            return self._bar
        try:
            from tqdm.auto import tqdm
        except Exception:
            self._bar = False
            return self._bar
        self._bar = tqdm(
            total=self._total_hint,
            desc=self._label,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.25,
        )
        return self._bar

    def __call__(self, input):
        self._calls += 1
        bar = self._ensure_bar()
        if bar and bar is not False:
            if self._calls > bar.total:
                bar.total = self._calls
                self._total_hint = self._calls
            bar.update(1)
            if self._calls == 1 or self._calls % self._every == 0:
                bar.set_postfix_str(f"batch {self._calls}")
                self._logger.info("⏳ %s | batch %s", self._label, self._calls)
        elif self._calls == 1 or self._calls % self._every == 0:
            self._logger.info("⏳ %s | batch %s", self._label, self._calls)
        return self._inner(input)


def drain_phase1_indexes_until_idle(
    *engines: Any,
    max_rounds: int = 8,
    max_jobs: int = 200,
) -> int:
    """Best-effort drain helper for tests that need derived projections to settle.

    This is only meaningful when the engine's phase-1 index jobs are enabled.
    In fast CI we use it to make read-after-write assertions deterministic after
    a workflow run without introducing a background worker thread.
    """

    total_applied = 0
    for _ in range(max_rounds):
        round_applied = 0
        round_pending = 0
        for engine in engines:
            if not getattr(engine, "_phase1_enable_index_jobs", False):
                continue
            reconcile = getattr(engine, "reconcile_indexes", None)
            meta = getattr(engine, "meta_sqlite", None)
            if reconcile is None or meta is None:
                continue
            try:
                round_applied += int(reconcile(max_jobs=max_jobs) or 0)
            except TypeError:
                round_applied += int(reconcile() or 0)

            list_jobs = getattr(meta, "list_index_jobs", None)
            if list_jobs is None:
                continue
            try:
                pending = list_jobs(status="PENDING", limit=1000)
            except TypeError:
                pending = list_jobs(limit=1000)
            round_pending += len(pending or [])
        total_applied += round_applied
        if round_applied == 0 and round_pending == 0:
            break
    return total_applied


def drain_phase1_indexes_with_workers_until_idle(
    *engines: Any,
    max_rounds: int = 8,
    batch_size: int = 50,
    namespace: str | None = None,
) -> int:
    """Drain phase-1 jobs using the real worker implementation.

    This is the test harness version of the background-worker approach:
    enqueue jobs during the run, then let a worker tick until the queue is idle.
    """

    from kogwistar.workers.index_job_worker import IndexJobWorker

    total_applied = 0
    for _ in range(max_rounds):
        round_claimed = 0
        for engine in engines:
            if not getattr(engine, "_phase1_enable_index_jobs", False):
                continue
            worker = IndexJobWorker(
                engine=engine,
                max_inflight=batch_size,
                batch_size=batch_size,
                max_jobs_per_tick=batch_size,
                namespace=namespace or getattr(engine, "namespace", "default"),
            )
            metrics = worker.tick()
            round_claimed += int(metrics.claimed or 0)
        total_applied += round_claimed
        if round_claimed == 0:
            break
    return total_applied


def build_workflow_engine_triplet(base_dir: Path, backend_kind: str):
    import os

    from kg_doc_parser.workflow_ingest.providers import EmbeddingProviderConfig, WorkflowProviderSettings, build_embedding_function
    from kg_doc_parser.workflow_ingest.service import build_default_engines

    if backend_kind == "in_memory":
        engines = build_default_engines(
            base_dir,
            backend_factory=load_kogwistar_in_memory_backend(),
            embedding_function=load_kogwistar_fake_embedding_function(),
        )
        if os.getenv("KG_DOC_DISABLE_PHASE1_INDEX_JOBS") == "1":
            for eng in engines:
                setattr(eng, "_phase1_enable_index_jobs", False)
        return engines
    if backend_kind == "chroma":
        pytest.importorskip("chromadb")
        embedding_spec = EmbeddingProviderConfig(
            provider="ollama",
            model="all-minilm:l6-v2",
            dimension=2,
            base_url="http://127.0.0.1:11434",
        )
        embedding_function = build_embedding_function(embedding_spec)
        embedding_function = _ProgressEmbeddingFunction(
            embedding_function,
            label=f"chroma:{embedding_spec.provider}:{embedding_spec.model}",
            every=5,
        )
        return build_default_engines(
            base_dir,
            backend_factory=None,
            embedding_function=embedding_function,
        )
    raise ValueError(f"unsupported workflow backend kind: {backend_kind}")


def load_kogwistar_fake_backend():
    helper_path = (
        Path(__file__).resolve().parents[1]
        / "kogwistar"
        / "tests"
        / "_helpers"
        / "fake_backend.py"
    )
    if not helper_path.exists():
        raise FileNotFoundError(f"kogwistar fake backend helper not found: {helper_path}")

    module_name = "kogwistar_tests_fake_backend"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load spec for {helper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.build_fake_backend

"""
document_ingest_logger.py

SQLite telemetry for LangChain ingestion runs:
- Before LLM call (on_llm_start)
- After LLM call with token usage + cost (on_llm_end)
- Errors (on_llm_error)

Why this design:
- No global/mutable callback state (no cb.set_document_id()).
  Document identity is passed per-invoke via `RunnableConfig.metadata`.
- Concurrency-safe + fast: callback methods enqueue events; a single writer thread
  batches inserts into SQLite (avoids "database is locked" under parallel ingestion).
- Queryable: events include run_id/parent_run_id so you can reconstruct chains.

--------------------------------------------------------------------------------
RECOMMENDED USAGE (per-document metadata; safe for concurrency)

from langchain_google_genai import ChatGoogleGenerativeAI
from document_ingest_logger import DocumentIngestSQLiteCallback

cb = DocumentIngestSQLiteCallback(db_path="logs/document_ingest.sqlite")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    # You may attach callback here OR per-call.
    # Attaching here is fine, but metadata (document_id) still comes per-call.
    callbacks=[cb],
)

# Per-document call
doc_id = "doc_123"
result = llm.invoke(
    "Summarize this document ...",
    config={
        "metadata": {
            "document_id": doc_id,
            "source_filename": "foo.pdf",    # optional; you can put anything here
            "n_try": n_try,
            "stage": "summary",
        },
        "tags": ["document_ingest"],
    },
)

# Structured output still works; callbacks still fire.
# structured_llm = llm.with_structured_output(MySchema)
# structured_llm.invoke(..., config={"metadata": {"document_id": doc_id}})

--------------------------------------------------------------------------------
ALTERNATIVE USAGE (attach callback per-call)

result = llm.invoke(
    "...",
    config={"callbacks": [cb], "metadata": {"document_id": doc_id}},
)

--------------------------------------------------------------------------------
SCHEMA

Table: ingest_events

Columns:
- id                INTEGER PK
- ts_iso            TEXT        ISO timestamp (UTC by default)
- document_id       TEXT        from config.metadata["document_id"] if present
- run_id            TEXT        LangChain run_id
- parent_run_id     TEXT        LangChain parent_run_id
- event_name        TEXT        e.g. "llm_start" / "llm_end" / "llm_error"
- model_name        TEXT        best-effort from generation_info
- filename          TEXT        best-effort from metadata or exception frames
- line_number       INTEGER     best-effort from exception frames
- token_count       INTEGER     best-effort: total tokens if known, else 0
- cost_usd          REAL        computed from usage + pricing table if possible
- try               REAL        Number of current try
- metadata_json     TEXT        JSON blob with all extra details

--------------------------------------------------------------------------------
NOTES
- You can add indices or additional columns as needed.
- Cost calculation is model-table based; if unknown model name, cost falls back
  to a conservative default (or 0.0 depending on your preference).
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Callable
from threading import Lock
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.messages import BaseMessage
from uuid import UUID

# ---------------------------
# Pricing / cost calculation
# ---------------------------

# per-1K token pricing in USD (example values; keep yours here)
_COST_TABLE: Dict[str, Dict[str, float]] = {
    "gemini-3-flash-preview": {"input": 0.0005, "output": 0.003, "cache": 0.0},
    "gemini-3.0-pro": {"input": 0.002000, "output": 0.0120, "cache": 0.00031},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004, "cache": 0.0},
    "gemini-1.5-pro": {"input": 0.001250, "output": 0.005, "cache": 0.0},
    "gemini-2.5-flash-preview-04-17": {"input": 0.000150, "output": 0.0035, "cache": 0.0000375},
    "gemini-2.5-flash": {"input": 0.000300, "output": 0.0025, "cache": 0.0000375},
    "gemini-2.5-pro-preview-03-25": {"input": 0.001250, "output": 0.0100, "cache": 0.00031},
    "gemini-2.5-pro": {"input": 0.001250, "output": 0.0100, "cache": 0.00031},
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0040, "cache": 0.00025},
}

# Some providers prefix with "models/...." (you already handled this in your file)
for _k in list(_COST_TABLE.keys()):
    if not _k.startswith("models/"):
        _COST_TABLE["models/" + _k] = _COST_TABLE[_k]


def calculate_cost_usd(
    *,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
) -> float:
    """
    Compute USD cost for a single LLM call based on token usage.

    Parameters
    ----------
    model_name:
        Provider-reported model name (e.g. "models/gemini-2.5-flash").
    input_tokens / output_tokens / cached_tokens:
        From provider usage metadata.

    Returns
    -------
    float
        Total cost in USD for this call.

    Notes
    -----
    - Cached tokens are charged at the table's "cache" rate, and are excluded from "input".
    - If the model is unknown, we default to a conservative high-price fallback
      (you can instead return 0.0 if you prefer).
    """
    price = _COST_TABLE.get(
        model_name,
        # conservative fallback: approximate "expensive"
        {"input": 1.250 / 1_000_000, "output": 5.0 / 1_000_000, "cache": 0.0},
    )
    billable_input = max(input_tokens - cached_tokens, 0)
    input_cost = (billable_input / 1000.0) * price["input"]
    cache_cost = (cached_tokens / 1000.0) * price["cache"]
    output_cost = (output_tokens / 1000.0) * price["output"]
    return float(input_cost + cache_cost + output_cost)


# ---------------------------
# SQLite writer (threaded)
# ---------------------------

@dataclass(frozen=True)
class _IngestEvent:
    """
    Internal event representation queued from callback thread(s) to writer thread.
    """
    ts_iso: str
    document_id: Optional[str]
    run_id: Optional[str]
    parent_run_id: Optional[str]
    event_name: str
    model_name: Optional[str]
    filename: Optional[str]
    line_number: Optional[int]
    token_count: int
    cost_usd: float
    n_try: float
    metadata_json: str


class SQLiteIngestEventWriter:
    """
    Background writer that persists ingestion events to SQLite.

    You generally do NOT use this class directly; it is owned by
    `DocumentIngestSQLiteCallback`.

    Design goals:
    - Avoid SQLite lock contention by using a single writer thread.
    - Keep callback fast: enqueue + return.
    - Initialize DB schema once and set WAL mode for concurrency.

    Parameters
    ----------
    db_path:
        SQLite file path, e.g. "logs/document_ingest.sqlite".
        Parent directories are created automatically.
    flush_interval_sec:
        How often the writer flushes queued events to disk.
    max_batch_size:
        Max number of events inserted per batch.

    Shutdown
    --------
    Call `.close()` to stop the writer thread and flush remaining events.
    """
    def __init__(
        self,
        db_path: str,
        *,
        flush_interval_sec: float = 0.25,
        max_batch_size: int = 200,
    ) -> None:
        self.db_path = db_path
        self.flush_interval_sec = float(flush_interval_sec)
        self.max_batch_size = int(max_batch_size)

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._q: "queue.Queue[_IngestEvent]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sqlite_ingest_writer", daemon=True)

        self._init_db()
        self._thread.start()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        # Pragmas for better concurrency and fewer "database is locked" errors.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_iso TEXT NOT NULL,
                    document_id TEXT,
                    run_id TEXT,
                    parent_run_id TEXT,
                    event_name TEXT NOT NULL,
                    model_name TEXT,
                    filename TEXT,
                    line_number INTEGER,
                    token_count INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    n_try REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_events_doc ON ingest_events(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_events_run ON ingest_events(run_id);")
            conn.commit()

    def enqueue(self, event: _IngestEvent) -> None:
        """Enqueue an event for persistence (non-blocking)."""
        self._q.put(event)

    def close(self) -> None:
        """
        Stop the writer thread and flush remaining events.

        Safe to call multiple times.
        """
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        pending: list[_IngestEvent] = []
        last_flush = time.time()

        while not self._stop.is_set():
            timeout = max(self.flush_interval_sec - (time.time() - last_flush), 0.01)
            try:
                ev = self._q.get(timeout=timeout)
                pending.append(ev)
            except queue.Empty:
                pass

            should_flush = (
                pending
                and (len(pending) >= self.max_batch_size or (time.time() - last_flush) >= self.flush_interval_sec)
            )
            if should_flush:
                self._flush(pending)
                pending.clear()
                last_flush = time.time()

        # Final drain + flush on stop
        try:
            while True:
                pending.append(self._q.get_nowait())
        except queue.Empty:
            pass
        if pending:
            self._flush(pending)

        # Optional checkpoint
        try:
            with self._connect() as conn:
                conn.execute("PRAGMA wal_checkpoint;")
                conn.commit()
        except Exception:
            # Swallow writer shutdown errors; don't crash application exit.
            pass

    def _flush(self, batch: list[_IngestEvent]) -> None:
        rows = [
            (
                e.ts_iso,
                e.document_id,
                e.run_id,
                e.parent_run_id,
                e.event_name,
                e.model_name,
                e.filename,
                e.line_number,
                e.token_count,
                e.cost_usd,
                e.n_try,
                e.metadata_json,
            )
            for e in batch
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO ingest_events (
                    ts_iso, document_id, run_id, parent_run_id, event_name,
                    model_name, filename, line_number, token_count, cost_usd, n_try, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()


# ---------------------------
# Helper: error frame info
# ---------------------------

def _best_effort_error_location(exc: BaseException) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract (filename, line_number) from the deepest traceback frame.

    Returns
    -------
    (filename, line_number)
        Both may be None if unavailable.
    """
    tb = exc.__traceback__
    if tb is None:
        return None, None
    last = None
    while tb is not None:
        last = tb
        tb = tb.tb_next
    if last is None:
        return None, None
    frame = last.tb_frame
    return frame.f_code.co_filename, int(last.tb_lineno)


def _utc_now_iso() -> str:
    """UTC ISO timestamp with timezone, suitable for sqlite TEXT."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------
# The LangChain callback
# ---------------------------
def _invoked_model_name(kwargs: Dict[str, Any]) -> Optional[str]:
    inv_params = kwargs.get("invocation_params") or {}
    m = inv_params.get("model")
    return str(m) if m else None
class DocumentIngestSQLiteCallback(BaseCallbackHandler):
    """
    LangChain callback that records ingestion telemetry into SQLite.

    What gets logged
    ---------------
    - on_llm_start: "llm_start" event with document_id + prompt stats (no token count).
    - on_llm_end:   "llm_end" event with token usage + computed cost if available.
    - on_llm_error: "llm_error" event with traceback info + best-effort filename/line.

    Passing document_id
    -------------------
    Do NOT call set_document_id() or mutate callback state.
    Instead pass it via `RunnableConfig.metadata`:

        llm.invoke(
            "...",
            config={"metadata": {"document_id": "doc_123", "source_filename": "a.pdf"}}
        )

    This is safe even under parallel ingestion because each run carries its own metadata.

    Parameters
    ----------
    db_path:
        SQLite database file path (e.g. "logs/document_ingest.sqlite").
    """
    def __init__(
        self,
        db_path: str,
        *,
        log_prompts: bool = False,
        log_chat_messages: bool = False,
        log_responses: bool = False,
        log_errors: bool = True,
        include_traceback: bool = True,
        max_text_chars: int = 40_000,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__()
        self._writer = SQLiteIngestEventWriter(db_path=db_path)

        self.log_prompts = log_prompts
        self.log_chat_messages = log_chat_messages
        self.log_responses = log_responses
        self.log_errors = log_errors
        self.include_traceback = include_traceback
        self.max_text_chars = int(max_text_chars)
        self.redact = redact or (lambda s: s)
        self._run_meta: dict[str, dict[str, Any]] = {}
        self._run_meta_lock = Lock()
    def _remember(self, run_id: UUID, metadata: dict[str, Any] | None, tags: list[str] | None) -> None:
        md = dict(metadata or {})
        md["_tags"] = list(tags or [])
        with self._run_meta_lock:
            self._run_meta[str(run_id)] = md

    def _recall(self, run_id: UUID, metadata: dict[str, Any] | None, tags: list[str] | None) -> dict[str, Any]:
        # prefer real-time metadata if present; else fallback to remembered
        if metadata:
            return dict(metadata)
        with self._run_meta_lock:
            return dict(self._run_meta.get(str(run_id), {}))

    def _forget(self, run_id: UUID) -> None:
        with self._run_meta_lock:
            self._run_meta.pop(str(run_id), None)
    def _clip(self, s: str) -> str:
        s = self.redact(s)
        if len(s) <= self.max_text_chars:
            return s
        return s[: self.max_text_chars//2] + f"\n...[clipped {len(s) - self.max_text_chars} chars]" + s[-(self.max_text_chars+1)//2:]

    def close(self) -> None:
        """Flush and stop the background writer thread."""
        self._writer.close()

    # ---- LangChain events ----

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: "UUID",
        parent_run_id: Optional["UUID"] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._remember(run_id, metadata, tags)
        md = metadata or {}
        document_id = md.get("document_id")
        filename = md.get("source_filename")
        n_try = float(md.get("n_try", 0))

        payload: Dict[str, Any] = {
            "serialized": serialized,
            "tags": tags or [],
            "metadata": md,
            "message_batches": len(messages),
            "message_counts": [len(batch) for batch in messages],
        }

        if self.log_chat_messages:
            # Convert BaseMessage to a JSON-friendly dict
            payload["chat_messages"] = [
                [
                    {
                        "type": m.type,
                        "content": self._clip(str(m.content)),
                        "additional_kwargs": getattr(m, "additional_kwargs", None),
                        "response_metadata": getattr(m, "response_metadata", None),
                        "name": getattr(m, "name", None),
                        "id": getattr(m, "id", None),
                    }
                    for m in batch
                ]
                for batch in messages
            ]

        self._writer.enqueue(
            _IngestEvent(
                ts_iso=_utc_now_iso(),
                document_id=str(document_id) if document_id is not None else None,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
                event_name="chat_model_start",
                model_name=_invoked_model_name(kwargs) or None,
                filename=str(filename) if filename is not None else None,
                line_number=md.get("line_number"),
                token_count=0,
                cost_usd=0.0,
                n_try=n_try,
                metadata_json=json.dumps(payload, ensure_ascii=False),
            )
        )
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        
        self._remember(run_id, metadata, tags)
        model_name = None
        """
        Called before an LLM request is sent.

        We intentionally do NOT try to token-count here (provider-specific).
        Instead we log prompt sizes and a few identifiers.
        """
        
        md = metadata or {}
        document_id = md.get("document_id")
        filename = md.get("source_filename")
        if model_name is None:
            if inv_params := kwargs.get('invocation_params'):
                if inv_params.get("model"):
                    model_name=inv_params.get("model")
        payload = {
            "serialized": serialized,
            "tags": tags or [],
            "metadata": md,
            "prompt_count": len(prompts),
            "prompt_chars_total": sum(len(p) for p in prompts),
        }
        if self.log_prompts:
            payload["prompts"] = [self._clip(p) for p in prompts]
        if model_name is None:
            mdn: str | None= md.get("model_name")
            if mdn is not None:
                model_name = mdn
        self._writer.enqueue(
            _IngestEvent(
                ts_iso=_utc_now_iso(),
                document_id=str(document_id) if document_id is not None else None,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
                event_name="llm_start",
                model_name=model_name,
                filename=str(filename) if filename is not None else None,
                line_number=md.get("line_number"),
                token_count=0,
                cost_usd=0.0,
                n_try = payload.get('n_try', 0),
                metadata_json=json.dumps(payload, ensure_ascii=False),
            )
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called after the LLM returns.

        Extracts provider usage if present (Gemini via usage_metadata),
        computes cost, and logs token counts + model name.
        """
        md = metadata or {}
        mdud = self._recall(run_id, metadata, tags) or {}
        md.update(mdud)
        document_id = md.get("document_id")
        filename = md.get("source_filename")

        # Defaults if we can't find usage
        input_tokens = output_tokens = cached_tokens = reasoning_tokens = 0
        model_name: Optional[str] = None

        # LangChain response.generations: List[List[Generation]]
        for gen_list in response.generations:
            for gen in gen_list:
                if isinstance(gen, ChatGeneration) and hasattr(gen, "message"):
                    msg = gen.message
                    # Gemini usage_metadata is commonly stored on the message
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        input_tokens = int(usage.get("input_tokens", 0) or 0)
                        output_tokens = int(usage.get("output_tokens", 0) or 0)
                        try:
                            cached_tokens = int(usage["input_token_details"]["cache_read"] or 0)
                        except Exception:
                            cached_tokens = 0
                        try:
                            reasoning_tokens = int(usage["output_token_details"]["reasoning"] or 0)
                        except Exception:
                            reasoning_tokens = 0

                    # Model name is often in generation_info
                    gi = getattr(gen, "generation_info", None) or {}
                    model_name = gi.get("model_name") or model_name
        if model_name is None:
            if inv_params := kwargs.get('invocation_params'):
                if inv_params.get("model"):
                    model_name=inv_params.get("model")
        token_count = int(input_tokens + output_tokens)
        cost_usd = 0.0
        if model_name and (input_tokens or output_tokens or cached_tokens):
            cost_usd = calculate_cost_usd(
                model_name=str(model_name),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )
        payload = {
            "tags": tags or [],
            "metadata": md,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "response_llm_output": getattr(response, "llm_output", None),
        }            
        if self.log_responses:
            outs: list[dict[str, Any]] = []
            for gen_list in response.generations:
                for gen in gen_list:
                    item: dict[str, Any] = {"type": type(gen).__name__}

                    # Text-y
                    if hasattr(gen, "text") and gen.text:
                        item["text"] = self._clip(str(gen.text))

                    # Chat-y
                    if isinstance(gen, ChatGeneration) and hasattr(gen, "message"):
                        msg = gen.message
                        item["message"] = {
                            "type": getattr(msg, "type", None),
                            "content": self._clip(str(getattr(msg, "content", ""))),
                            "additional_kwargs": getattr(msg, "additional_kwargs", None),
                            "response_metadata": getattr(msg, "response_metadata", None),
                        }

                    item["generation_info"] = getattr(gen, "generation_info", None)
                    outs.append(item)

            payload["outputs"] = outs

        if model_name is None:
            mdn: str | None= md.get("model_name")
            if mdn is not None:
                model_name = mdn
        self._writer.enqueue(
            _IngestEvent(
                ts_iso=_utc_now_iso(),
                document_id=str(document_id) if document_id is not None else None,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
                event_name="llm_end",
                model_name=str(model_name) if model_name is not None else None,
                filename=str(filename) if filename is not None else None,
                line_number=md.get("line_number"),
                token_count=token_count,
                cost_usd=float(cost_usd),
                n_try=float(payload.get('n_try', 0)),
                metadata_json=json.dumps(payload, ensure_ascii=False),
            )
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when the LLM call errors.

        Logs:
        - exception type/message
        - traceback string
        - best-effort filename/line_number from traceback
        """
        model_name: str|None = None
        md = metadata or {}
        mdud = self._recall(run_id, metadata, tags) or {}
        md.update(mdud)
        document_id = md.get("document_id")
        # prefer explicit metadata filename; else fallback to traceback location
        meta_filename = md.get("source_filename")

        tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        tb_file, tb_line = _best_effort_error_location(error)

        payload = {
            "tags": tags or [],
            "metadata": md,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": tb_str,
        }
        if self.log_errors:
            payload["error_message"] = self._clip(str(error))
            if self.include_traceback:
                payload["traceback"] = self._clip(tb_str)
        else:
            payload["error_type"] = type(error).__name__        
        if model_name is None:
            if inv_params := kwargs.get('invocation_params'):
                if inv_params.get("model"):
                    model_name=inv_params.get("model")
        if model_name is None:
            mdn: str | None= md.get("model_name")
            if mdn is not None:
                model_name = mdn
        self._writer.enqueue(
            _IngestEvent(
                ts_iso=_utc_now_iso(),
                document_id=str(document_id) if document_id is not None else None,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
                event_name="llm_error",
                model_name=model_name,
                filename=str(meta_filename or tb_file) if (meta_filename or tb_file) is not None else None,
                line_number=int(tb_line) if tb_line is not None else None,
                token_count=0,
                cost_usd=0.0,
                n_try=float(payload.get('n_try', 0)),
                metadata_json=json.dumps(payload, ensure_ascii=False),
            )
        )

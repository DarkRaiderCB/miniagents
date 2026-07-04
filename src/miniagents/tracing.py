"""Execution tracing: a structured, exportable record of everything a run did.

A Tracer collects TraceEvents (LLM calls, tool calls, task boundaries, errors)
tagged with a shared run_id, monotonically increasing sequence numbers, and
wall-clock timestamps. Crew.run_all() creates one automatically; standalone
components can be traced by wrapping calls in `with Tracer() as t:`.

Events are plain dicts underneath, so a trace can be saved as JSONL and
inspected with standard tools (jq, pandas, etc.).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PREVIEW_LIMIT = 200


def preview(value: Any, limit: int = PREVIEW_LIMIT) -> str:
    """Truncate a value's string form for compact trace/log payloads."""
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class TraceEvent:
    """A single recorded occurrence within a run."""

    seq: int
    timestamp: float  # epoch seconds
    run_id: str
    event: str  # e.g. "llm_call", "tool_call", "task_start"
    name: str  # subject of the event (task/agent/tool/model name)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "event": self.event,
            "name": self.name,
            **self.data,
        }


class Tracer:
    """
    Thread-safe collector of TraceEvents for one run.

    Usable as a context manager, which makes it the ambient tracer that
    module-level `record()` calls write to:

        with Tracer() as tracer:
            agent.execute("...")
        tracer.save("trace.jsonl")
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id: str = run_id or uuid.uuid4().hex[:12]
        self.started_at: float = time.time()
        self._events: List[TraceEvent] = []
        self._lock = threading.Lock()
        self._token: Optional[Token] = None

    def __enter__(self) -> "Tracer":
        self._token = _current_tracer.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            _current_tracer.reset(self._token)
            self._token = None

    def record(self, event: str, name: str = "", **data: Any) -> TraceEvent:
        """Append an event to the trace and return it."""
        with self._lock:
            evt = TraceEvent(
                seq=len(self._events),
                timestamp=time.time(),
                run_id=self.run_id,
                event=event,
                name=name,
                data=data,
            )
            self._events.append(evt)
        logger.debug("trace[%s] #%d %s %s %s", self.run_id, evt.seq, event, name, data)
        return evt

    @property
    def events(self) -> List[TraceEvent]:
        with self._lock:
            return list(self._events)

    def summary(self) -> Dict[str, Any]:
        """Aggregate view: duration, event counts, LLM/tool volume, errors."""
        events = self.events
        counts: Dict[str, int] = {}
        for evt in events:
            counts[evt.event] = counts.get(evt.event, 0) + 1
        llm_ms = sum(
            e.data.get("elapsed_ms", 0) for e in events if e.event == "llm_call"
        )
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "duration_s": round(
                (events[-1].timestamp - self.started_at), 3
            ) if events else 0.0,
            "total_events": len(events),
            "event_counts": counts,
            "llm_calls": counts.get("llm_call", 0),
            "llm_time_ms": round(llm_ms, 1),
            "tool_calls": counts.get("tool_call", 0),
            "errors": sum(n for e, n in counts.items() if e.endswith("_error")),
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Write the trace as JSON Lines (one event per line)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for evt in self.events:
                fh.write(json.dumps(evt.to_dict(), default=str) + "\n")
        logger.info("Trace %s saved to %s (%d events)", self.run_id, target, len(self._events))
        return target


_current_tracer: ContextVar[Optional[Tracer]] = ContextVar(
    "miniagents_current_tracer", default=None
)


def current_tracer() -> Optional[Tracer]:
    """Return the ambient Tracer, if one is active."""
    return _current_tracer.get()


def record(event: str, name: str = "", **data: Any) -> None:
    """Record an event on the ambient tracer; no-op when tracing is inactive."""
    tracer = _current_tracer.get()
    if tracer is not None:
        tracer.record(event, name=name, **data)


__all__ = ["Tracer", "TraceEvent", "current_tracer", "record", "preview"]

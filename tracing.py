"""
Tracing and monitoring (roadmap item 7) — OpenTelemetry spans for every agent
step, model call, Copilot tool call, pipeline run and human approval.

Spans are stored locally in the app database (table `trace_spans`), so the audit
trail stays in the clinic like the rest of the data. Set OTEL_EXPORTER_OTLP_ENDPOINT
(and install opentelemetry-exporter-otlp-proto-http) to also ship them to a
collector such as Jaeger, Grafana Tempo or a self-hosted Langfuse.

    from tracing import span
    with span("agent.run", agent="report", patient_id=4) as s:
        ...
        s.set_attribute("source", "llm")

If opentelemetry isn't installed, span() is a no-op — tracing never breaks the app.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager

import db

_lock = threading.Lock()
_tracer = None


def _schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS trace_spans (
        id INTEGER PRIMARY KEY, trace_id TEXT NOT NULL, span_id TEXT NOT NULL, parent_id TEXT,
        name TEXT NOT NULL, start_ns INTEGER NOT NULL, duration_ms REAL NOT NULL,
        status TEXT NOT NULL, error TEXT, attributes TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')))""")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_trace_spans_name ON trace_spans(name, start_ns)")


def _init():
    global _tracer
    if _tracer is not None:
        return _tracer
    with _lock:
        if _tracer is not None:
            return _tracer
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (SimpleSpanProcessor, BatchSpanProcessor,
                                                        SpanExporter, SpanExportResult)
        except ImportError:
            _tracer = False
            return _tracer

        class SQLiteSpanExporter(SpanExporter):
            def export(self, spans):
                try:
                    conn = db.connect()
                    _schema(conn)
                    conn.executemany(
                        "INSERT INTO trace_spans (trace_id,span_id,parent_id,name,start_ns,duration_ms,"
                        "status,error,attributes) VALUES (?,?,?,?,?,?,?,?,?)",
                        [(format(s.context.trace_id, "032x"), format(s.context.span_id, "016x"),
                          format(s.parent.span_id, "016x") if s.parent else None, s.name, s.start_time,
                          round((s.end_time - s.start_time) / 1e6, 2), s.status.status_code.name,
                          s.status.description, json.dumps(dict(s.attributes or {}), default=str))
                         for s in spans])
                    conn.commit(); conn.close()
                    return SpanExportResult.SUCCESS
                except Exception:
                    return SpanExportResult.FAILURE

        provider = TracerProvider(resource=Resource.create({"service.name": "dentagent"}))
        provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter()))
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            except ImportError:
                pass
        _tracer = provider.get_tracer("dentagent")
        return _tracer


class _NoSpan:
    def set_attribute(self, *a, **k):
        pass


@contextmanager
def span(name: str, **attributes):
    """Record one step. Exceptions are recorded on the span and re-raised."""
    tracer = _init()
    if not tracer:
        yield _NoSpan()
        return
    from opentelemetry.trace import Status, StatusCode
    with tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))
        try:
            yield s
        except Exception as e:
            s.set_status(Status(StatusCode.ERROR, f"{type(e).__name__}: {e}"[:300]))
            raise


# ---- read side: the Traces page and /api/health ---------------------------------

def recent(limit: int = 100, name: str | None = None) -> list[dict]:
    conn = db.connect()
    _schema(conn)
    q = "SELECT * FROM trace_spans" + (" WHERE name = ?" if name else "") + " ORDER BY id DESC LIMIT ?"
    rows = conn.execute(q, ((name, limit) if name else (limit,))).fetchall()
    conn.close()
    return [{**dict(r), "attributes": json.loads(r["attributes"] or "{}")} for r in rows]


def stats(hours: int = 24) -> list[dict]:
    """Per span name over the last `hours`: count, errors, p50/p95 latency."""
    conn = db.connect()
    _schema(conn)
    rows = conn.execute("SELECT name, status, duration_ms FROM trace_spans "
                        "WHERE created_at >= datetime('now', ?)", (f"-{int(hours)} hours",)).fetchall()
    conn.close()
    by = {}
    for r in rows:
        by.setdefault(r["name"], []).append((r["status"], r["duration_ms"]))
    out = []
    for n, vals in sorted(by.items()):
        d = sorted(v[1] for v in vals)
        pick = lambda q: d[min(len(d) - 1, int(q * len(d)))]
        out.append({"name": n, "count": len(vals), "errors": sum(v[0] == "ERROR" for v in vals),
                    "p50_ms": round(pick(0.5), 1), "p95_ms": round(pick(0.95), 1)})
    return out

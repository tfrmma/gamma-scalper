"""
infra/logging_setup.py

Structured JSON logging + optional sinks.
Replace the flat file handler from main.py with this.

Outputs:
  stdout      - JSON lines, picked up by Datadog/Grafana/Loki/any log collector
  file        - same JSON, rotated daily, 7 day retention
  metrics     - key numeric fields extracted and emitted as StatsD gauges
                (optional, only if STATSD_HOST is set)

Usage in main.py:
    from infra.logging_setup import setup_logging
    setup_logging(level="INFO")

Log format (one JSON object per line):
  {
    "ts":      "2026-01-15T10:23:45.123Z",
    "level":   "INFO",
    "logger":  "strategy",
    "msg":     "ENTER signal | ...",
    "asset":   "BTC",          # extracted from structured fields
    "premium": 0.082,          # extracted if present
    ...
  }

Datadog: set DD_API_KEY + DD_SITE, use the standard datadog-agent log pipeline.
Grafana/Loki: pipe stdout to promtail or use the Loki docker driver.
StatsD: set STATSD_HOST (e.g. "localhost") and optionally STATSD_PORT (default 8125).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---- JSON formatter ---------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    Emits one JSON object per log line.
    Extracts key=value pairs from the message into top-level fields
    so Datadog/Loki can index them without a pipeline.

    e.g. "ENTER signal | asset=BTC premium=0.08 rv=0.32"
    becomes {"msg": "ENTER signal", "asset": "BTC", "premium": 0.08, "rv": 0.32}
    """

    # pull out trailing key=value pairs from log messages
    _KV_RE = re.compile(r'(\w+)=([\w.%$,+-]+)')

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()

        obj: dict[str, Any] = {
            "ts":     datetime.fromtimestamp(record.created, tz=timezone.utc)
                      .strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(record.created % 1 * 1000):03d}Z",
            "level":  record.levelname,
            "logger": record.name,
            "msg":    msg,
        }

        # extract structured fields from message
        for key, val in self._KV_RE.findall(msg):
            try:
                # try numeric first
                obj[key] = float(val.rstrip('%')) / (100 if val.endswith('%') else 1)
            except ValueError:
                obj[key] = val

        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)

        return json.dumps(obj, separators=(',', ':'))


# ---- StatsD emitter ---------------------------------------------------------

class StatsDEmitter:
    """
    Fire-and-forget UDP StatsD gauges for key metrics.
    If STATSD_HOST isn't set, this is a no-op.
    No dependencies - raw UDP socket.

    Metrics emitted (prefix: gamma_scalper.<asset>):
      rv_ratio, vol_premium, ofi, realized_pnl, margin_util,
      live_orders, latency_p95_order_send, funding_ann
    """

    def __init__(self, host: str, port: int, prefix: str) -> None:
        self._prefix = prefix
        self._addr   = (host, port)
        self._sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def gauge(self, name: str, value: float) -> None:
        try:
            metric = f"{self._prefix}.{name}:{value:.6f}|g"
            self._sock.sendto(metric.encode(), self._addr)
        except Exception:
            pass   # metrics are best-effort, never kill the process

    def from_snapshot(self, snap: dict) -> None:
        """Extract and emit all numeric fields from a state/risk snapshot."""
        numeric_keys = {
            "rv_ratio", "vol_premium", "ofi", "realized_pnl",
            "margin_util", "live_orders", "funding_ann",
            "net_vega", "net_gamma", "accumulated_delta",
        }
        for key in numeric_keys:
            val = snap.get(key)
            if isinstance(val, (int, float)) and val == val:   # not NaN
                self.gauge(key, float(val))

        # nested latency
        latency = snap.get("latency_p95", {})
        for op, ms in latency.items():
            if isinstance(ms, (int, float)):
                self.gauge(f"latency_p95.{op}", ms)


# ---- setup ------------------------------------------------------------------

def setup_logging(
    level:    str  = "INFO",
    log_dir:  str  = ".",
    app_name: str  = "gamma-scalper",
) -> StatsDEmitter | None:
    """
    Configure structured JSON logging.
    Returns a StatsDEmitter if STATSD_HOST is set, else None.

    Call once at startup, before any loggers are used.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = JsonFormatter()

    # stdout handler - this is what Datadog/Loki agents read
    stdout_h = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    )
    stdout_h.setFormatter(formatter)

    # rotating file handler - local backup, 7 days
    log_path = Path(log_dir) / f"{app_name}.log"
    file_h = logging.handlers.TimedRotatingFileHandler(
        filename    = log_path,
        when        = "midnight",
        backupCount = 7,
        encoding    = "utf-8",
    )
    file_h.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(stdout_h)
    root.addHandler(file_h)

    # quieten noisy third-party loggers
    for noisy in ("websockets", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log = logging.getLogger("infra.logging")
    log.info(f"logging configured | level={level} file={log_path}")

    # StatsD - optional
    statsd_host = os.environ.get("STATSD_HOST", "")
    if statsd_host:
        statsd_port = int(os.environ.get("STATSD_PORT", "8125"))
        emitter = StatsDEmitter(statsd_host, statsd_port, prefix=app_name)
        log.info(f"StatsD emitter ready | {statsd_host}:{statsd_port}")
        return emitter

    return None

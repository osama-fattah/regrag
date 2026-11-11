"""Append-only JSONL audit log: one record per request, with what was retrieved, cited and
decided. Supports record-keeping and traceability expectations for AI in regulated settings."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path | None):
        # "-" writes JSON lines to stdout (e.g. Lambda -> CloudWatch Logs); None disables logging.
        self.stdout = str(path) == "-"
        self.path = Path(path) if path and not self.stdout else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if not (self.path or self.stdout):
            return
        line = json.dumps(
            {"ts": datetime.now(timezone.utc).isoformat(), "audit": True, **record}, ensure_ascii=False
        )
        with self._lock:
            if self.stdout:
                print(line, flush=True)
            else:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
